"""Load the post-trained Psi0 checkpoints into memory and verify they're intact.

Run from repo root:
    set -a && . ./.env && set +a && uv run python scripts/smoke_test_post_trained.py

This is a "smoke test": it does NOT run inference, it just loads the weights
and asserts every tensor matches the architecture. If this script exits 0,
the downloaded checkpoints + the deps + the Psi0 package are all healthy.
"""
import os
import torch
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from psi.config.model_psi0 import Psi0ModelConfig
from psi.models.psi0 import Psi0Model

VLM_DIR = os.path.expandvars("$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1")
AH_DIR = os.path.expandvars("$PSI_HOME/cache/checkpoints/psi0/postpre.1by1")

# NOTE: these hparams MATCH the published post-trained ckpt's actual training
# settings, which differ from Psi0ModelConfig defaults AND from the README:
#   action_chunk_size=16  (config default is 6, finetune uses 30; ckpt is 16)
#   view_feature_dim=2048 (default 1920)
#   odim=36               (default 15)
# With these, action_header.safetensors loads with strict=True, 0 missing,
# 0 unexpected keys. With wrong values you get shape mismatches.
POST_TRAINED_HPARAMS = dict(
    action_dim=36,
    action_chunk_size=16,
    odim=36,
    view_feature_dim=2048,
    noise_scheduler="flow",
    train_diffusion_steps=1000,
    eval_diffusion_steps=10,
    tune_vlm=False,
)


def main():
    print(f"[1/4] Loading VLM (Qwen3-VL-2B, bf16, flash_attn_2) from {VLM_DIR}")
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_DIR, attn_implementation="flash_attention_2", dtype=torch.bfloat16,
    )
    vlm_n = sum(p.numel() for p in vlm.parameters())
    print(f"      VLM params: {vlm_n / 1e9:.2f} B")

    print(f"[2/4] Loading Qwen3-VL processor")
    _ = AutoProcessor.from_pretrained(VLM_DIR)  # exercises tokenizer + image processor

    print(f"[3/4] Building Psi0Model wrapper (action header init random)")
    cfg = Psi0ModelConfig(
        model_name_or_path=VLM_DIR,
        pretrained_action_header_path=AH_DIR,
        **POST_TRAINED_HPARAMS,
    )
    model = Psi0Model(model_cfg=cfg, vlm_model=vlm)

    print(f"[4/4] Loading action_header.safetensors with strict=True")
    state_dict = load_file(f"{AH_DIR}/action_header.safetensors")
    result = model.action_header.load_state_dict(state_dict, strict=True)
    assert len(result.missing_keys) == 0, result.missing_keys
    assert len(result.unexpected_keys) == 0, result.unexpected_keys

    total = sum(p.numel() for p in model.parameters())
    ah = sum(p.numel() for p in model.action_header.parameters())
    print()
    print("=== POST-TRAINED Psi0 LOADED ===")
    print(f"  VLM (Qwen3-VL-2B):        {vlm_n / 1e9:.2f} B params")
    print(f"  Action header (diff txf): {ah / 1e6:.1f} M params")
    print(f"  TOTAL:                    {total / 1e9:.2f} B params")
    print(f"  action_chunk_size:        {cfg.action_chunk_size}")
    print(f"  action_dim:               {cfg.action_dim}")
    print(f"  noise_scheduler:          {cfg.noise_scheduler}")
    print("OK")


if __name__ == "__main__":
    main()
