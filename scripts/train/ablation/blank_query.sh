#!/bin/bash
# ── Ablation: blank_query (control group - capacity without distillation signal) ─
# Equivalent to 3DRS query_type='blank' ablation
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_blank_query"
mkdir -p $OUTPUT_DIR

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --use_geometry_encoder False \
    --use_distillation True \
    --query_type blank \
    --query_size 16 \
    > ${OUTPUT_DIR}/train.log 2>&1
