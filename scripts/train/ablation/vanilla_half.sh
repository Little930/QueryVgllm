#!/bin/bash
# ── 公平基线：vanilla @ 576分辨率 @ 50%数据 ───────────────────────────────────
# 为什么需要它：旧的 vanilla=54.05 是 576 + 100%数据 训出来的。
# 现在 Logic A/B 用 50% 数据提速，若直接跟 54.05 比，会多一个「数据量」混淆。
# 所以三者必须同分辨率(576)+同数据量(50%)，对比才干净：
#     vanilla_half  vs  distill_logicA_official  vs  distill_logicB_fork
# 三个脚本的 MAXPIX 和 HALF 必须一致（默认都 576 + %50）。
source "$(dirname "$0")/common.sh"

OUTPUT_DIR="./checkpoints/ablation_vanilla_half"
mkdir -p $OUTPUT_DIR

MAXPIX=$((576*28*28))                              # 必须与 distill 脚本一致
HALF="scan2cap%50,scanrefer%50,scannet_det%50"     # 必须与 distill 脚本一致

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/qwen_vl/train/train_qwen.py \
    $COMMON_ARGS \
    --output_dir $OUTPUT_DIR \
    --cache_dir ./cache \
    --use_geometry_encoder False \
    --use_distillation False \
    --dataset_use "$HALF" \
    --model_max_length 12800 \
    --max_pixels $MAXPIX \
    > ${OUTPUT_DIR}/train.log 2>&1
