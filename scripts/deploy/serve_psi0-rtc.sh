#!/bin/bash

source .venv-psi/bin/activate

export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-.runs/finetune/stackcups.real.flow1000.cosine.lr1.0e-04.b128.gpus8.2604160048}
export CHECKPOINT_STEP=${CHECKPOINT_STEP:-40000}
echo "Serving on GPU $CUDA_VISIBLE_DEVICES, checkpoint $CHECKPOINT_DIR step $CHECKPOINT_STEP"

python src/psi/deploy/psi_serve_rtc-trainingtimertc.py \
    --host 0.0.0.0 \
    --port 8014 \
    --action_exec_horizon 30 \
    --policy psi \
    --rtc \
    --run-dir=${CHECKPOINT_DIR} \
    --ckpt-step=${CHECKPOINT_STEP}
