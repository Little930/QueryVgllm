#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Logic B = 3DRS-FORK：一整块 query 注入（忠实复刻你 D:\桌面\3DRS 的做法）
# ══════════════════════════════════════════════════════════════════════════════
#
# 与官方 3DRS-1 的区别（你的 fork 加的）：
#   - 在所有图像之后**一次性追加一整块** geometry_query（32*query_size 行），
#     而不是官方那种纯 image-token 蒸馏。
#   - QueryVgllm 已按 fork 重写：_inject_query_tokens 现在是「一整块」注入
#     [pre-text][blank?][所有图像][geometry][depth?][post-text]，
#     query 参数 = Parameter(distill_max_frames*query_size, H)，按 n_frames*qs 切片。
#
# 关键超参（对齐 fork 默认）：
#   query_type=geometry  → 选中 Logic B（注入）；改成 None/不传 = Logic A(官方image-token)
#   query_size=4         → fork 默认（32 帧 → 32*4=128 个 query，而非旧版散插的 512）
#   geometry_weight=1.0  → fork 直接相加，不缩放
#   distill_max_frames=32→ query 参数行数上限（≥ 单样本最大帧数）
#   distill_warmup_steps=500 → 轻量 warmup 保护 LLM（fork 原版=0；想纯复刻设 0）
#
# 加 depth 教师：把 query_type 改成 "geometry,depth" 并确保有 depth.npz
# ──────────────────────────────────────────────────────────────────────────────
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_distill_logicB_fork"
mkdir -p $OUTPUT_DIR

# ── 公平性：必须和 vanilla 用同一分辨率！vanilla=576×28×28（见 common.sh）─────
# 之前蒸馏被降到 192 是为了躲旧版 512-query 的 OOM；现在 Logic B 只加 128 个 token，
# 可以撑回 576。若仍 OOM，把 MAXPIX 往下调（448/384/320/256），
# 但**记得 vanilla 也要用同一个值重训**（vanilla_half.sh 里同步改）。
MAXPIX=$((576*28*28))
# 半数据集（%50）提速；seed 固定→A/B/vanilla 抽到同一子集，可公平比
HALF="scan2cap%50,scanrefer%50,scannet_det%50"

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
    --query_type geometry \
    --query_size 4 \
    --distill_max_frames 32 \
    --geometry_dim 2048 \
    --geometry_weight 1.0 \
    --distill_warmup_steps 500 \
    --dataset_use "$HALF" \
    --model_max_length 12800 \
    --max_pixels $MAXPIX \
    > ${OUTPUT_DIR}/train.log 2>&1

# ── 训练后查：确认对齐没静默失败 ───────────────────────────────────────────────
#   grep -ciE "FAILED|MISMATCH" ${OUTPUT_DIR}/train.log   # 期望 0
# ── 评测（与 vanilla 同一套命令，仅换 checkpoint）──────────────────────────────
#   见 distill_logicA_official.sh 末尾的评测命令，把路径换成本目录即可。
