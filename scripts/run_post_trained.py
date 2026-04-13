"""Run inference with the post-trained Psi0 checkpoints using a dummy image.

Verifies the full pipeline: VLM forward pass -> action header diffusion -> action output.

Run from repo root:
    set -a && . ./.env && set +a && uv run python scripts/run_post_trained.py
"""
import os
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from psi.config.model_psi0 import Psi0ModelConfig
from psi.models.psi0 import Psi0Model

DEVICE = "cuda:0"
VLM_DIR = os.path.expandvars("$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1")
AH_DIR = os.path.expandvars("$PSI_HOME/cache/checkpoints/psi0/postpre.1by1")

# These hparams match the published post-trained ckpt's actual architecture.
# The Psi0ModelConfig defaults (action_chunk_size=6, odim=15,
# view_feature_dim=1920) are stale — every real training script overrides
# odim=36 and view_feature_dim=2048. The ckpt's action_chunk_size=16 was
# confirmed empirically from tensor shapes.
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
    # --- 1. Load VLM ---
    print(f"[1/5] Loading VLM (Qwen3-VL-2B, bf16, flash_attn_2) from {VLM_DIR}")
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        VLM_DIR, attn_implementation="flash_attention_2", dtype=torch.bfloat16,
    )
    vlm_n = sum(p.numel() for p in vlm.parameters())
    print(f"      VLM params: {vlm_n / 1e9:.2f} B")

    # --- 2. Load processor ---
    print("[2/5] Loading Qwen3-VL processor")
    processor = AutoProcessor.from_pretrained(VLM_DIR)

    # --- 3. Build Psi0Model + load action header ---
    print("[3/5] Building Psi0Model wrapper + loading action header (strict=True)")
    cfg = Psi0ModelConfig(
        model_name_or_path=VLM_DIR,
        pretrained_action_header_path=AH_DIR,
        **POST_TRAINED_HPARAMS,
    )
    model = Psi0Model(model_cfg=cfg, vlm_model=vlm)

    state_dict = load_file(f"{AH_DIR}/action_header.safetensors")
    result = model.action_header.load_state_dict(state_dict, strict=True)
    assert len(result.missing_keys) == 0, result.missing_keys
    assert len(result.unexpected_keys) == 0, result.unexpected_keys

    # --- 4. Wire up inference attributes (normally done by from_pretrained) ---
    print("[4/5] Setting up inference attributes")
    model.vlm_processor = processor
    model.noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=cfg.train_diffusion_steps,
    )
    model.action_horizon = cfg.action_chunk_size  # 16
    model.action_dim = cfg.action_dim  # 36
    model.device = DEVICE
    model.to(DEVICE)
    model.eval()

    # --- 5. Run inference with a dummy image ---
    print("[5/5] Running inference (dummy 240x320 image, dummy state)")
    dummy_img = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
    dummy_state = torch.zeros(1, 1, 36, device=DEVICE)  # (B, To, Ds)

    actions = model.predict_action(
        observations=[[dummy_img]],
        states=dummy_state,
        instructions=["pick up the object"],
        num_inference_steps=10,
        traj2ds=None,
    )

    total = sum(p.numel() for p in model.parameters())
    ah = sum(p.numel() for p in model.action_header.parameters())
    print()
    print("=== POST-TRAINED Psi0 INFERENCE OK ===")
    print(f"  VLM (Qwen3-VL-2B):        {vlm_n / 1e9:.2f} B params")
    print(f"  Action header (diff txf): {ah / 1e6:.1f} M params")
    print(f"  TOTAL:                    {total / 1e9:.2f} B params")
    print(f"  action_chunk_size:        {cfg.action_chunk_size}")
    print(f"  action_dim:               {cfg.action_dim}")
    print(f"  Output shape:             {actions.shape}")
    print(f"  Output range:             [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    print("OK")


if __name__ == "__main__":
    main()
