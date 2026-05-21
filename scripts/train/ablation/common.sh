#!/bin/bash
# ── Shared configuration for all 3DRS ablation experiments ──────────────────
# Source this file at the top of each ablation script: source $(dirname $0)/common.sh

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=4

MODEL_PATH="./Qwen2.5-VL-3B-Instruct"
GEOMETRY_ENCODER_PATH="./VGGT-1B"
# Root directory for offline npz files: {FEATURE_DIR}/{scene_id}/feature_3d.npz
FEATURE_DIR="/public_datasets/VG-LLM/features/scannet"
DATASETS="scan2cap,scanrefer,scannet_det"

LR=1e-5
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))

export NCCL_NVLS_ENABLE=0

COMMON_ARGS="
    --model_name_or_path $MODEL_PATH
    --tune_mm_llm True
    --tune_mm_vision False
    --tune_mm_mlp False
    --dataset_use $DATASETS
    --bf16
    --per_device_train_batch_size 1
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS
    --learning_rate $LR
    --mm_projector_lr 1e-5
    --vision_tower_lr 1e-6
    --optim adamw_torch
    --model_max_length 12800
    --data_flatten False
    --max_pixels $((576*28*28))
    --min_pixels $((16*28*28))
    --base_interval 2
    --video_max_frames 8
    --video_min_frames 4
    --video_max_frame_pixels $((1664*28*28))
    --video_min_frame_pixels $((256*28*28))
    --num_train_epochs 1
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --weight_decay 0.01
    --logging_steps 10
    --save_steps 1000
    --save_total_limit 1
    --deepspeed scripts/zero2_opt.json
    --gradient_checkpointing
    --dataloader_num_workers 4
    --group_by_modality_length true
    --seed 0
    --report_to none
    --feature_dir $FEATURE_DIR
"
