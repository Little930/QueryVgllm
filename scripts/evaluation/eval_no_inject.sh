#!/bin/bash
# ── Step 2 诊断：关闭 query 注入，用同一个 checkpoint eval ──────────────────
# 目的：区分掉点原因是 query 噪声还是 LLM 本体被伤
#
# 结果判读：
#   Scan2Cap 恢复 ~54 → query tokens 在推理时是噪声 → 用 fix_a（降 query_size）
#   Scan2Cap 维持 ~51 → LLM 语言能力被蒸馏梯度损伤 → 用 fix_b（降 LR + 降 weight）
#
# 原理：DISABLE_QUERY_INJECT=1 让 forward() 跳过 _inject_query_tokens()
#       模型仍然加载了 geometry_query 参数（不影响），只是不插入序列

cd ~/VG-LLM
export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0
export DISABLE_QUERY_INJECT=1   # ← 关键：跳过 query 注入

CKPT="./checkpoints/ablation_geometry_query"
OUTPUT="logs/eval_distill_no_inject"
mkdir -p $OUTPUT

echo "=== Step 2 诊断：关闭 query 注入 ==="
echo "Checkpoint: $CKPT"
echo "DISABLE_QUERY_INJECT=$DISABLE_QUERY_INJECT"

# Scan2Cap eval
accelerate launch --num_processes=4 --main_process_port 29503 -m lmms_eval \
    --model vgllm \
    --model_args "pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800" \
    --tasks scan2cap \
    --batch_size 1 \
    --log_samples --log_samples_suffix distill_no_inject \
    --output_path $OUTPUT

echo "=== 完成 ==="
echo "对比：geometry_query = 51.03, vanilla = 54.05"
echo "如果本次 ~54 → query 是噪声 → fix_a"
echo "如果本次 ~51 → LLM 被伤 → fix_b"
