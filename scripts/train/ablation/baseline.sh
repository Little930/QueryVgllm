#!/bin/bash
# ── Baseline: no distillation, use_geometry_encoder=True (original 3DRS-1 style) ─
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_baseline"
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
    --use_distillation False \
    > ${OUTPUT_DIR}/train.log 2>&1
