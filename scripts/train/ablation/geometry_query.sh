#!/bin/bash
# ── Ablation: geometry_query only (single teacher: VGGT) ─────────────────────
# Core 3DRS innovation: independent query tokens distilled towards VGGT features
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_geometry_query"
mkdir -p $OUTPUT_DIR

# Query injection extends sequence length, reduce max_pixels and model_max_length to avoid OOM
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
    --query_size 16 \
    --geometry_dim 2048 \
    --geometry_weight 1.0 \
    --model_max_length 8192 \
    --max_pixels $((384*28*28)) \
    > ${OUTPUT_DIR}/train.log 2>&1
