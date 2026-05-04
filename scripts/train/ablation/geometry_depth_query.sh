#!/bin/bash
# ── Ablation: geometry_query + depth_query (dual teacher: VGGT + DAv2) ───────
# 3DRS §3.2 dual-teacher distillation (geometry_weight=1.0, depth_weight=0.5)
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_geometry_depth_query"
mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --use_geometry_encoder False \
    --use_distillation True \
    --query_type geometry,depth \
    --query_size 16 \
    --geometry_dim 2048 \
    --geometry_weight 1.0 \
    --depth_dim 1024 \
    --depth_weight 0.5 \
    > ${OUTPUT_DIR}/train.log 2>&1
