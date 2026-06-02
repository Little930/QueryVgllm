#!/bin/bash
# ── Fix A: geometry_query with reduced weight and fewer tokens ────────────────
# Problem: geometry_weight=1.0 + query_size=16 causes query tokens to hurt LLM
# Fix: reduce weight to 0.1 so distillation is a weak auxiliary signal,
#      reduce query_size from 16 to 4 to minimize sequence interference
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_geometry_query_fix_a"
mkdir -p $OUTPUT_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --use_geometry_encoder False \
    --use_distillation True \
    --query_type geometry \
    --query_size 4 \
    --geometry_dim 2048 \
    --geometry_weight 0.1 \
    --model_max_length 12800 \
    --max_pixels $((192*28*28)) \
    > ${OUTPUT_DIR}/train.log 2>&1
