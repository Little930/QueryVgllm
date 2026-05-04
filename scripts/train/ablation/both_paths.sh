#!/bin/bash
# ── Ablation: geometry_query + online VGGT encoder (both paths active) ───────
# Tests whether online encoding (Path A) + offline distillation (Path B) complement each other
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_both_paths"
mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --use_geometry_encoder True \
    --geometry_encoder_type vggt \
    --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
    --feature_fusion_method add \
    --use_distillation True \
    --query_type geometry \
    --query_size 16 \
    --geometry_dim 2048 \
    --geometry_weight 1.0 \
    > ${OUTPUT_DIR}/train.log 2>&1
