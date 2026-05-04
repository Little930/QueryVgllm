#!/bin/bash
# VG-LLM 1000 Steps Validation Script
# Optimized for single RTX 4060 Laptop GPU (8GB)

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=1  # Single GPU

# ======================
# Path Configuration
# ======================
# 请根据你的实际路径修改以下配置
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"  # 需要修改
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="/path/to/VGGT-1B"  # 需要修改
OUTPUT_DIR="./checkpoints/vgllm_1000steps_test"
CACHE_DIR="./cache"
mkdir -p $OUTPUT_DIR

# ======================
# Model Configuration
# ======================
DATASETS="scanrefer"  # 使用单个数据集减少内存占用

# ======================
# Training Hyperparameters (Optimized for 8GB GPU)
# ======================
export NCCL_NVLS_ENABLE=0
LR=1e-5
GRADIENT_ACCUMULATION_STEPS=8  # 增加梯度累积以减少显存占用

# ======================
# Launch Training
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --tune_mm_llm True \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
            --learning_rate $LR \
            --mm_projector_lr 1e-5 \
            --vision_tower_lr 1e-6 \
            --optim adamw_torch \
            --model_max_length 8192 \
            --data_flatten False \
            --max_pixels $((256*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 4 \
            --video_min_frames 2 \
            --video_max_frame_pixels $((512*28*28)) \
            --video_min_frame_pixels $((128*28*28)) \
            --max_steps 1000 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_steps 500 \
            --save_total_limit 2 \
            --gradient_checkpointing \
            --dataloader_num_workers 2 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "tensorboard" \
            --use_geometry_encoder True \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --feature_fusion_method "add" \
            > ${OUTPUT_DIR}/train.log 2>&1

echo "Training completed. Check logs at ${OUTPUT_DIR}/train.log"
