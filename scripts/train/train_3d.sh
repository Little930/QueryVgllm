#!/bin/bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"                     # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)     # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="/public_datasets/VG-LLM/weights/Qwen2.5-VL-3B-Instruct"
GEOMETRY_ENCODER_TYPE="vggt"
GEOMETRY_ENCODER_PATH="/public_datasets/VG-LLM/weights/VGGT-1B"
OUTPUT_DIR="./checkpoints/vgllm_3d_run1"                # Directory for saving checkpoints
CACHE_DIR="./cache"                        # [TrainingArguments] Cache directory for models
mkdir -p $OUTPUT_DIR

# ======================
# Model Configuration
# ======================
DATASETS="scan2cap,scanrefer,scannet_det"                  # [DataArguments] Dataset with sampling rate

# ======================
# Training Hyperparameters
# ======================
export NCCL_NVLS_ENABLE=0
LR=1e-5
total_batch_size=64
GRADIENT_ACCUMULATION_STEPS=$(($total_batch_size / $NPROC_PER_NODE))

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
            --model_max_length 12800 \
            --data_flatten False \
            --max_pixels $((576*28*28)) \
            --min_pixels $((16*28*28)) \
            --base_interval 2 \
            --video_max_frames 8 \
            --video_min_frames 4 \
            --video_max_frame_pixels $((1664*28*28)) \
            --video_min_frame_pixels $((256*28*28)) \
            --num_train_epochs 1 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero2_opt.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 4 \
            --group_by_modality_length true \
            --seed 0 \
            --report_to "none" \
            --use_geometry_encoder True \
            --geometry_encoder_type $GEOMETRY_ENCODER_TYPE \
            --geometry_encoder_path $GEOMETRY_ENCODER_PATH \
            --feature_fusion_method "add" \
            > ${OUTPUT_DIR}/train.log 2>&1

# ===================================================
# 3DRS Optional Parameters (uncomment to enable)
# ===================================================
# Add these to the torchrun command above (before the redirect) to enable 3DRS features:
#
# --- Grounding Head ---
# --ground_head_type "infonce"            # Options: None, "infonce", "mlp", "score"
# --ground_head_temperature 0.07          # Temperature for InfoNCE loss
#
# --- Query Token Distillation ---
# --query_type "geometry"                 # Options: None, "geometry", "geometry,depth", "blank", "blank,reason"
# --query_size 4                          # Number of query tokens per frame
# --query_image False                     # Also distill on image tokens (dual distillation)
#
# --- Object Feature Aggregation ---
# --obj_feature None                      # Options: None, "sim", "center_sim", "filter"
#
# --- 3D World Position Encoding ---
# --world_position_embedding_type None    # Options: None, "avg-discrete-sin3d", "avg-sin3d", "avg-mlp"
# --voxel_size 0.1                        # Voxel size for discretization
#
# --- Distillation Configuration ---
# --distillation_mode "offline"           # Options: "offline" (precomputed features), "online" (from geometry_encoder)
# --distillation_feature_dim 2048         # VGGT feature dimension
# --distillation_depth_feature_dim 1024   # DAv2 feature dimension
# --distillation_loss_weight 1.0          # Weight for distillation loss
# --grounding_loss_weight 1.0             # Weight for grounding loss
