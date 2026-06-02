#!/bin/bash
# ── Fix B: gentler optimization — lower LR + lower distillation weight ────────
# Problem: distillation gradients at full strength (weight=1.0, lr=1e-5) may
#          damage LLM's language capability during training.
# Fix: keep LLM trainable (tune_mm_llm=True) but reduce learning rate to 5e-6
#      and geometry_weight to 0.3, so the distillation is a weak auxiliary signal.
#
# NOTE: tune_mm_llm MUST be True — otherwise the model can't learn 3D language.
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_geometry_query_fix_b"
mkdir -p $OUTPUT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --tune_mm_llm True \
    --use_geometry_encoder False \
    --use_distillation True \
    --query_type geometry \
    --query_size 16 \
    --geometry_dim 2048 \
    --geometry_weight 0.3 \
    --learning_rate 5e-6 \
    --model_max_length 12800 \
    --max_pixels $((192*28*28)) \
    > ${OUTPUT_DIR}/train.log 2>&1
