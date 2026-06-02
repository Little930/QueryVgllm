#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  输出侧图像-token 蒸馏（3DRS 官方做法）—— 目标：稳定 ≥ vanilla
# ══════════════════════════════════════════════════════════════════════════════
#
# 为什么是这个方案（论文依据）:
#   - Ross3D Q2: 把 3D 特征拼到「输入端」会掉点（未对齐的3D特征对2D LLM是噪声）；
#                改成在「输出端」加监督才涨点(+4~5)。
#   - 我们之前的 query_type=geometry 正是「输入端塞 token」(54.05→51.03 掉点)。
#   - 本脚本切回 3DRS 官方做法：query_type 不设 (=None) → 不注入任何 token，
#     直接对「图像 token 的 hidden state」过 proj_3d 去对齐 VGGT 特征。
#
# 为什么不会掉到 vanilla 以下（三重保险）:
#   1) 不往输入序列塞 token → 没有污染/位置错乱/序列膨胀三大隐患；
#   2) distill_warmup_steps: 蒸馏权重前期线性 warmup，早期几乎不动 LLM；
#   3) geometry_weight=0.5(3DRS 默认) → 辅助信号温和；
#   4) 若 teacher/student 形状对不上，loss 会自动跳过并打印 [DISTILL] SHAPE MISMATCH
#      警告（此时退化为纯 vanilla，不会更差）。
#
# 关键代码改动（已完成，见 modeling_qwen2_5_vl.py）:
#   - process_feature_for_distillation: 输出侧路径也自动对齐 teacher 帧数；
#   - compute_3d_distillation_loss(qt=None): 套上 geometry_weight + 形状不匹配大声警告；
#   - forward: distill_warmup_steps 线性 warmup。
# ──────────────────────────────────────────────────────────────────────────────
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_image_token_distill"
mkdir -p $OUTPUT_DIR

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
    --geometry_weight 0.5 \
    --distill_warmup_steps 1000 \
    --model_max_length 12800 \
    --max_pixels $((192*28*28)) \
    > ${OUTPUT_DIR}/train.log 2>&1

# ── 训练后必看（确认蒸馏真的在生效，而不是静默退化成 vanilla）────────────────
#   grep -c "SHAPE MISMATCH" ${OUTPUT_DIR}/train.log
#     → 输出 0  = 蒸馏正常生效（teacher/student 对齐 OK）
#     → 输出 >0 = 形状没对上，当前等于 vanilla，需检查 vggt.npz 分辨率 vs 图像 token 网格
#
# ── 评测（与 vanilla 用同一套命令对比）──────────────────────────────────────────
#   bash scripts/evaluation/eval_vanilla.sh ./checkpoints/ablation_image_token_distill scan2cap
#   （脚本里有 config 断言，注意它默认要求 use_distillation=False；
#     评测蒸馏模型请改用 eval_distillation.sh 或把断言去掉）
