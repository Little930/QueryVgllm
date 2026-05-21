#!/bin/bash
# ── Vanilla: pure Qwen2.5-VL fine-tune, no geometry encoder, no distillation ──
# This serves as the TRUE baseline to verify training convergence on 3D data.
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_vanilla"
mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --use_geometry_encoder False \
    --use_distillation False \
    > ${OUTPUT_DIR}/train.log 2>&1
