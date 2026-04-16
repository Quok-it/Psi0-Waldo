#!/bin/bash
# Finetune Psi0 on G1_Brainco_StackCups dataset.
#
# Adapted from finetune-real-psi0.sh for:
#   - 8x H100 PCIe GPUs (DDP)
#   - 26-dim action/state (padded to 36)
#   - observation.state key (not "states")
#   - observation.images.head_camera (not "egocentric")
#   - stats at meta/stats_psi0.json
#
# Prerequisites:
#   1. Run: python scripts/data/convert_lerobot_v3_to_v21.py $PSI_HOME/data/real/G1_Brainco_StackCups
#   2. Run: python scripts/data/patch_lerobot_meta.py $PSI_HOME/data/real/G1_Brainco_StackCups
#   3. Checkpoints downloaded via: bash scripts/download_post_trained.sh
#
# Usage:
#   bash scripts/train/psi0/finetune-stackcups.sh

set -euo pipefail

export OMP_NUM_THREADS=32
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

source .venv-psi/bin/activate
set -a && source .env && set +a

NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Training with $NPROC_PER_NODE GPUs"

export task="G1_Brainco_StackCups"
export exp="stackcups"

args="
finetune_real_psi0_config \
--seed=292285 \
--exp=$exp \
--train.name=finetune \
--train.data_parallel=ddp \
--train.mixed_precision=bf16 \
--train.train_batch_size=16 \
--train.max_checkpoints_to_keep=5 \
--train.gradient_accumulation_steps=1 \
--train.learning_rate=1e-4 \
--train.max_training_steps=40000 \
--train.warmup_ratio=None \
--train.warmup_steps=1000 \
--train.checkpointing_steps=5000 \
--train.validation_steps=1000 \
--train.val_num_batches=20 \
--train.max_grad_norm=1.0 \
--train.lr_scheduler_type=cosine \
--train.lr_scheduler_kwargs.weight_decay=1e-6 \
--train.lr_scheduler_kwargs.betas 0.95 0.999 \
--log.report_to=tensorboard \
--data.root_dir=real \
--data.train_repo_ids=$task \
--data.transform.repack.image-keys observation.images.head_camera \
--data.transform.repack.state-key=observation.state \
--data.transform.repack.pad-action-dim=36 \
--data.transform.repack.pad-state-dim=36 \
--data.transform.field.stat-path=meta/stats_psi0.json \
--data.transform.field.stat-action-key=action \
--data.transform.field.stat-state-key=observation.state \
--data.transform.field.action_norm_type=bounds \
--data.transform.field.no-use-norm-mask \
--data.transform.field.normalize-state \
--data.transform.field.pad-action-dim=36 \
--data.transform.field.pad-state-dim=36 \
--data.transform.model.img-aug \
--data.transform.model.resize.size 240 320 \
--data.transform.model.center_crop.size 240 320 \
--model.model_name_or_path=$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1 \
--model.pretrained-action-header-path=$PSI_HOME/cache/checkpoints/psi0/postpre.1by1 \
--model.noise-scheduler=flow \
--model.train-diffusion-steps=1000 \
--model.n_conditions=0 \
--model.action-chunk-size=30 \
--model.action-dim=36 \
--model.action-exec-horizon=30 \
--model.observation-horizon=1 \
--model.odim=36 \
--model.view_feature_dim=2048 \
--model.no-tune-vlm \
--model.no-use_film \
--model.no-combined_temb \
--model.rtc \
--model.max-delay=8 \
--train.resume_from_checkpoint=.runs/finetune/stackcups.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2604151753
"
# --train.resume_from_checkpoint=.runs/finetune/stackcups.real.flow1000.cosine.lr1.0e-04.b16.gpus8.resumed

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29500 scripts/train.py \
    ${args}
