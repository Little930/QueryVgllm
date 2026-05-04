#!/bin/bash
# ── Ablation: geometry_query + image_token distillation (query_image=True) ───
# Tests additional image-token supervision on top of geometry query
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_geometry_query_image"
mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --use_geometry_encoder False \
    --use_distillation True \
    --query_type geometry \
    --query_size 16 \
    --geometry_dim 2048 \
    --geometry_weight 1.0 \
    --query_image True \
    > ${OUTPUT_DIR}/train.log 2>&1
