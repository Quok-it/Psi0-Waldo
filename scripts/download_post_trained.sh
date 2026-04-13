#!/usr/bin/env bash
# Download the published Psi0 post-trained checkpoints from Hugging Face.
#
# Reads HF_TOKEN and PSI_HOME from .env in the repo root.
#
# Result:
#   $PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1/    (~4 GB, the VLM)
#   $PSI_HOME/cache/checkpoints/psi0/postpre.1by1/     (~1.9 GB, the action header)
#
# Run from repo root:
#   bash scripts/download_post_trained.sh

set -euo pipefail

# Load .env so HF_TOKEN and PSI_HOME are available to download.py
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
else
    echo "ERROR: .env not found in $(pwd). Copy .env.sample and fill in HF_TOKEN + PSI_HOME." >&2
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is empty in .env" >&2
    exit 1
fi
if [ -z "${PSI_HOME:-}" ]; then
    echo "ERROR: PSI_HOME is empty in .env" >&2
    exit 1
fi

mkdir -p "$PSI_HOME/cache/checkpoints/psi0"

REPO_ID="USC-PSI-Lab/psi-model"

# 1. VLM (Qwen3-VL-2B post-trained on EgoDex 200k + HE 30k)
echo "==> Downloading VLM checkpoint..."
uv run python scripts/data/download.py \
    --repo-id="$REPO_ID" \
    --remote-dir="psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k" \
    --local-dir="$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1" \
    --repo-type=model

# 2. Action header (~500M flow-matching diffusion expert post-trained on HE)
echo "==> Downloading action header checkpoint..."
uv run python scripts/data/download.py \
    --repo-id="$REPO_ID" \
    --remote-dir="psi0/postpre.1by1.pad36.2601131206.ckpt.he30k" \
    --local-dir="$PSI_HOME/cache/checkpoints/psi0/postpre.1by1" \
    --repo-type=model

# Clean up empty subdirs left behind by download.py's flatten step
find "$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1/psi0" -type d -empty -delete 2>/dev/null || true
find "$PSI_HOME/cache/checkpoints/psi0/postpre.1by1/psi0" -type d -empty -delete 2>/dev/null || true

echo
echo "==> Done. Disk usage:"
du -sh "$PSI_HOME/cache/checkpoints/psi0/"*
