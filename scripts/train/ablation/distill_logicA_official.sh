#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Logic A = 官方 3DRS-1：纯 image-token 蒸馏（无 query 注入）
# ══════════════════════════════════════════════════════════════════════════════
#
# 这是 3DRS 论文真正的方法（D:\桌面\3DRS-1）：
#   feature_3d_alignment 直接取「图像 token 的 hidden state」过 proj_3d，
#   对齐 VGGT 特征（每帧池化到 14×14）。不往序列里插任何 query token。
#
# 在 QueryVgllm 里：query_type 不传 (=None) → compute_3d_distillation_loss 走
#   image-token 分支（self.proj_3d），forward 不做任何注入。
#
# 关键超参：
#   (不传 query_type)    → 选中 Logic A
#   geometry_weight=1.0  → 官方直接相加（想更保守可调 0.5）
#   distill_warmup_steps=500 → 轻量 warmup 保护 LLM（官方=0；纯复刻设 0）
# ──────────────────────────────────────────────────────────────────────────────
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_distill_logicA_official"
mkdir -p $OUTPUT_DIR

# ── 公平性：和 vanilla 同分辨率 576×28×28（见 common.sh）。Logic A 不加任何
# token，显存≈vanilla，576 必然能撑。若想省显存可下调，但 vanilla 要同步。
MAXPIX=$((576*28*28))
HALF="scan2cap%50,scanrefer%50,scannet_det%50"   # 半数据集提速

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --tune_mm_llm True \
    --use_geometry_encoder False \
    --use_distillation True \
    --geometry_dim 2048 \
    --geometry_weight 1.0 \
    --distill_warmup_steps 500 \
    --dataset_use "$HALF" \
    --model_max_length 12800 \
    --max_pixels $MAXPIX \
    > ${OUTPUT_DIR}/train.log 2>&1

# ── 训练后查：是否静默退化成 vanilla ──────────────────────────────────────────
#   grep -ciE "SHAPE MISMATCH|FAILED" ${OUTPUT_DIR}/train.log   # 期望 0
#
# ── 评测（两套逻辑通用；与 vanilla 同命令，仅换 CKPT 路径）─────────────────────
#   export LMMS_EVAL_LAUNCHER=accelerate; export NCCL_NVLS_ENABLE=0
#   CKPT=./checkpoints/ablation_distill_logicA_official     # 或 ..._logicB_fork
#   accelerate launch --num_processes=4 --main_process_port 29501 -m lmms_eval \
#     --model vgllm \
#     --model_args "pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800" \
#     --tasks scan2cap --batch_size 1 \
#     --log_samples --log_samples_suffix distill --output_path logs/eval_$(basename $CKPT)
#   对比基线 vanilla = 54.05
