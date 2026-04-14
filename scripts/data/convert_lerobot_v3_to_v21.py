#!/usr/bin/env python3
"""Convert a LeRobot v3.0 dataset to v2.1 format for Psi0 finetuning.

LeRobot v3.0 packs multiple episodes into single parquet/video files.
LeRobot v2.1 (used by lerobot==0.3.3 / Psi0) expects one file per episode.

This script:
  1. Converts tasks.parquet -> tasks.jsonl
  2. Converts episodes parquet -> episodes.jsonl + episodes_stats.jsonl
  3. Patches info.json (codebase_version, path templates)
  4. Splits monolithic parquet into per-episode parquet files
  5. Splits packed videos into per-episode mp4 files (av1 -> h264)
  6. Creates stats_psi0.json for Psi0's action/state normalizer

Usage:
    python scripts/data/convert_lerobot_v3_to_v21.py $PSI_HOME/data/real/G1_Brainco_StackCups

Requires: pyarrow, av (pyav)
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq


def convert_tasks(meta_dir: Path):
    """Convert tasks.parquet -> tasks.jsonl"""
    tasks_pq = meta_dir / "tasks.parquet"
    if not tasks_pq.exists():
        print("  SKIP tasks.parquet (not found)")
        return
    t = pq.read_table(tasks_pq)
    d = t.to_pydict()
    out = meta_dir / "tasks.jsonl"
    with open(out, "w") as f:
        for i in range(len(d["task_index"])):
            task_text = d.get("__index_level_0__", d.get("task", ["unknown"]))[i]
            f.write(json.dumps({"task_index": d["task_index"][i], "task": task_text}) + "\n")
    print(f"  Created {out.name} ({len(d['task_index'])} tasks)")


def convert_episodes(meta_dir: Path):
    """Convert episodes parquet -> episodes.jsonl + episodes_stats.jsonl"""
    ep_pq = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    if not ep_pq.exists():
        print("  SKIP episodes parquet (not found)")
        return

    t = pq.read_table(ep_pq)
    d = t.to_pydict()
    stats_cols = [c for c in d.keys() if c.startswith("stats/")]
    non_stats_cols = [c for c in d.keys() if not c.startswith("stats/")]
    n_eps = len(d["episode_index"])

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i in range(n_eps):
            row = {}
            for c in non_stats_cols:
                val = d[c][i]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                row[c] = val
            f.write(json.dumps(row) + "\n")
    print(f"  Created episodes.jsonl ({n_eps} episodes)")

    # episodes_stats.jsonl
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for i in range(n_eps):
            stats_dict = {}
            for c in stats_cols:
                _, key, stat = c.split("/", 2)
                val = d[c][i]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                elif isinstance(val, list):
                    val = [v.tolist() if hasattr(v, "tolist") else v for v in val]
                stats_dict.setdefault(key, {})[stat] = val
            row = {"episode_index": d["episode_index"][i], "stats": stats_dict}
            f.write(json.dumps(row) + "\n")
    print(f"  Created episodes_stats.jsonl")

    return d  # return for use by video splitter


def patch_info_json(meta_dir: Path):
    """Patch info.json: codebase_version and path templates."""
    info_path = meta_dir / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4"

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Patched info.json (codebase_version=v2.1)")
    return info


def split_parquet(dataset_dir: Path):
    """Split monolithic parquet into per-episode files."""
    src = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
    if not src.exists():
        print("  SKIP parquet split (file-000.parquet not found)")
        return

    table = pq.read_table(src)
    ep_col = table.column("episode_index").to_pylist()
    episodes = sorted(set(ep_col))
    print(f"  Splitting {table.num_rows} rows into {len(episodes)} per-episode parquets...")

    # Backup original
    bak = src.with_suffix(".parquet.bak")
    if not bak.exists():
        os.rename(src, bak)

    for ep_idx in episodes:
        mask = pa.array([e == ep_idx for e in ep_col])
        ep_table = table.filter(mask)

        # Fix timestamps/frame_index to be relative (start from 0 per episode)
        ts = ep_table.column("timestamp").to_pylist()
        fi = ep_table.column("frame_index").to_pylist()
        if ts and ts[0] != 0:
            t0, f0 = ts[0], fi[0]
            idx_ts = ep_table.schema.get_field_index("timestamp")
            ep_table = ep_table.set_column(idx_ts, "timestamp",
                                           pa.array([t - t0 for t in ts], type=pa.float32()))
            idx_fi = ep_table.schema.get_field_index("frame_index")
            ep_table = ep_table.set_column(idx_fi, "frame_index",
                                           pa.array([f - f0 for f in fi], type=pa.int64()))

        out_path = src.parent / f"file-{ep_idx:03d}.parquet"
        pq.write_table(ep_table, out_path)

    # Move backup out of data dir so HF datasets won't glob it
    backups_dir = dataset_dir / ".backups"
    backups_dir.mkdir(exist_ok=True)
    bak.rename(backups_dir / bak.name)

    print(f"  Done ({len(episodes)} files)")


def split_videos(dataset_dir: Path, episodes_data: dict):
    """Split packed videos into per-episode h264 mp4 files."""
    d = episodes_data
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    # Discover video keys from info.json features
    video_keys = [
        k for k, v in info.get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    ]
    print(f"  Video keys: {video_keys}")

    for vid_key in video_keys:
        print(f"\n  === {vid_key} ===")
        vid_dir = dataset_dir / "videos" / vid_key / "chunk-000"
        if not vid_dir.exists():
            print(f"    SKIP (directory not found)")
            continue

        fi_key = f"videos/{vid_key}/file_index"
        from_key = f"videos/{vid_key}/from_timestamp"
        to_key = f"videos/{vid_key}/to_timestamp"

        if fi_key not in d:
            print(f"    SKIP (no metadata)")
            continue

        file_indices = d[fi_key]
        from_ts_list = d[from_key]
        to_ts_list = d[to_key]

        # Backup originals
        for fi in sorted(set(file_indices)):
            src = vid_dir / f"file-{fi:03d}.mp4"
            bak = src.with_suffix(".mp4.bak")
            if src.exists() and not bak.exists():
                os.rename(src, bak)

        # Process each source file sequentially
        for src_fi in sorted(set(file_indices)):
            bak = vid_dir / f"file-{src_fi:03d}.mp4.bak"
            if not bak.exists():
                print(f"    SKIP file-{src_fi:03d} (no backup)")
                continue

            container = av.open(str(bak))
            stream = container.streams.video[0]
            fps = int(float(stream.average_rate))
            width, height = stream.width, stream.height
            print(f"    Decoding file-{src_fi:03d} ({width}x{height})...")

            eps_in_file = sorted([
                (from_ts_list[i], to_ts_list[i], i)
                for i in range(len(d["episode_index"]))
                if file_indices[i] == src_fi
            ])

            # Decode all frames to numpy (avoids av1 seeking issues)
            frames_data = []
            for frame in container.decode(stream):
                ft = float(frame.pts * stream.time_base)
                frames_data.append((ft, frame.to_ndarray(format="rgb24")))
            container.close()
            print(f"    {len(frames_data)} frames -> {len(eps_in_file)} episodes")

            # Write per-episode mp4
            for ep_from, ep_to, ep_idx in eps_in_file:
                out_path = vid_dir / f"file-{ep_idx:03d}.mp4"
                ep_frames = [
                    arr for ft, arr in frames_data
                    if ft >= ep_from - 0.001 and ft < ep_to - 0.001
                ]

                out = av.open(str(out_path), "w", format="mp4")
                out_stream = out.add_stream("libx264", rate=fps)
                out_stream.width = width
                out_stream.height = height
                out_stream.pix_fmt = "yuv420p"
                out_stream.options = {"crf": "18"}

                for arr in ep_frames:
                    vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
                    out.mux(out_stream.encode(vf))

                out.mux(out_stream.encode(None))
                out.close()

            del frames_data
            gc.collect()


def patch_parquet_metadata(dataset_dir: Path):
    """Remove HuggingFace schema metadata from parquets.

    The v3.0 parquets embed HF feature metadata that uses 'List' type,
    which older versions of the `datasets` library don't understand.
    Stripping this metadata lets `load_dataset("parquet", ...)` infer
    the schema from the actual Arrow columns instead.
    """
    count = 0
    for path in dataset_dir.rglob("*.parquet"):
        if ".backups" in str(path):
            continue
        table = pq.read_table(path)
        meta = dict(table.schema.metadata or {})
        if b"huggingface" not in meta:
            continue
        meta.pop(b"huggingface")
        tmp = path.with_suffix(".tmp")
        pq.write_table(table.replace_schema_metadata(meta or None), tmp)
        tmp.replace(path)
        count += 1
    print(f"  Patched {count} parquet files")


def create_psi0_stats(dataset_dir: Path, state_key: str = "observation.state"):
    """Create stats_psi0.json from the dataset's stats.json."""
    stats_path = dataset_dir / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    psi_stats = {}
    for key in ["action", state_key]:
        if key not in stats:
            print(f"  WARNING: '{key}' not found in stats.json")
            continue
        psi_stats[key] = {
            "min": stats[key]["min"],
            "max": stats[key]["max"],
            "q01": stats[key]["q01"],
            "q99": stats[key]["q99"],
        }

    out_path = dataset_dir / "meta" / "stats_psi0.json"
    with open(out_path, "w") as f:
        json.dump(psi_stats, f, indent=2)

    for k, v in psi_stats.items():
        print(f"  {k}: {len(v['min'])}-dim")
    print(f"  Wrote {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot v3.0 dataset to v2.1 for Psi0")
    parser.add_argument("dataset_dir", type=Path, help="Path to the dataset directory")
    parser.add_argument("--state-key", default="observation.state",
                        help="Key for state observations (default: observation.state)")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip video splitting (slow step)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    meta_dir = dataset_dir / "meta"

    print(f"Converting {dataset_dir}")
    start = time.time()

    print("\n[1/6] Converting tasks...")
    convert_tasks(meta_dir)

    print("\n[2/6] Converting episodes metadata...")
    episodes_data = convert_episodes(meta_dir)

    print("\n[3/6] Patching info.json...")
    patch_info_json(meta_dir)

    print("\n[4/6] Splitting parquet data...")
    split_parquet(dataset_dir)

    if not args.skip_videos:
        print("\n[5/7] Splitting videos (this takes a while)...")
        if episodes_data:
            split_videos(dataset_dir, episodes_data)
        else:
            print("  SKIP (no episodes data)")
    else:
        print("\n[5/7] Skipping video split")

    print("\n[6/7] Patching parquet metadata...")
    patch_parquet_metadata(dataset_dir)

    print("\n[7/7] Creating stats_psi0.json...")
    create_psi0_stats(dataset_dir, state_key=args.state_key)

    # Clean up video backups
    backups_dir = dataset_dir / ".backups"
    backups_dir.mkdir(exist_ok=True)
    for bak in dataset_dir.rglob("*.mp4.bak"):
        bak.rename(backups_dir / bak.name)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
