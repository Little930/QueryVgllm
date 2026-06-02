#!/bin/bash
# ── Evaluation script for vanilla ablation checkpoint ─────────────────────────
# Vanilla = pure Qwen2.5-VL fine-tune, NO geometry encoder, NO distillation.
#
# Usage:
#   bash scripts/evaluation/eval_vanilla.sh [checkpoint_dir] [benchmark]
#
# Examples:
#   bash scripts/evaluation/eval_vanilla.sh ./checkpoints/ablation_vanilla scan2cap
#   bash scripts/evaluation/eval_vanilla.sh ./checkpoints/ablation_vanilla scanrefer
#   bash scripts/evaluation/eval_vanilla.sh ./checkpoints/ablation_vanilla all
set -e

# ── Arguments ─────────────────────────────────────────────────────────────────
CKPT="${1:-./checkpoints/ablation_vanilla}"
BENCHMARK="${2:-all}"   # choices: scan2cap, scanrefer, all

# ── Environment ───────────────────────────────────────────────────────────────
export LMMS_EVAL_LAUNCHER="accelerate"
export NCCL_NVLS_ENABLE=0

NPROC=4   # adjust based on available GPUs
BASE_MODEL="./Qwen2.5-VL-3B-Instruct"
CKPT_NAME=$(basename $CKPT)
OUTPUT_BASE="logs/eval_${CKPT_NAME}"

# ── Step 0: Ensure processor/tokenizer configs exist in checkpoint ────────────
echo "=== Step 0: Verifying checkpoint completeness ==="
for f in preprocessor_config.json tokenizer.json tokenizer_config.json; do
    if [ ! -f "$CKPT/$f" ] && [ -f "$BASE_MODEL/$f" ]; then
        echo "  Copying missing $f from base model..."
        cp "$BASE_MODEL/$f" "$CKPT/"
    fi
done
# Also copy chat_template.json if missing
if [ ! -f "$CKPT/chat_template.json" ] && [ -f "$BASE_MODEL/chat_template.json" ]; then
    cp "$BASE_MODEL/chat_template.json" "$CKPT/"
fi

# ── Step 1: Verify config matches vanilla (no distillation, no geometry) ──────
echo "=== Step 1: Checking config flags ==="
python -c "
import json
with open('$CKPT/config.json') as f:
    cfg = json.load(f)
flags = ['use_distillation','use_geometry_encoder']
for k in flags:
    v = cfg.get(k, 'NOT SET')
    print(f'  {k}: {v}')
# Vanilla should have both disabled
assert not cfg.get('use_distillation', False), 'ERROR: use_distillation should be False for vanilla!'
assert not cfg.get('use_geometry_encoder', False), 'ERROR: use_geometry_encoder should be False for vanilla!'
print('  ✓ Vanilla config verified OK')
"

# ── Common model args ─────────────────────────────────────────────────────────
MODEL_ARGS="pretrained=$CKPT,use_flash_attention_2=true,max_num_frames=32,max_length=12800"

# ── Step 2: Run evaluation ────────────────────────────────────────────────────
run_eval() {
    local task=$1
    local suffix=$2
    local extra_args=$3
    local port=$4
    local output_dir="${OUTPUT_BASE}_${task}"
    
    mkdir -p "$output_dir"
    echo ""
    echo "=== Running $task evaluation ==="
    echo "  Checkpoint: $CKPT"
    echo "  Output: $output_dir"
    echo ""
    
    local args="$MODEL_ARGS"
    if [ -n "$extra_args" ]; then
        args="$args,$extra_args"
    fi
    
    accelerate launch --num_processes=$NPROC --main_process_port $port -m lmms_eval \
        --model vgllm \
        --model_args "$args" \
        --tasks $task \
        --batch_size 1 \
        --log_samples --log_samples_suffix "$suffix" \
        --output_path "$output_dir"
    
    echo "  ✓ $task evaluation complete. Results in: $output_dir"
}

if [ "$BENCHMARK" = "scan2cap" ] || [ "$BENCHMARK" = "all" ]; then
    run_eval "scan2cap" "vanilla_${CKPT_NAME}" "" 29501
fi

if [ "$BENCHMARK" = "scanrefer" ] || [ "$BENCHMARK" = "all" ]; then
    run_eval "scanrefer" "vanilla_${CKPT_NAME}" "add_frame_index=true" 29502
    
    # Post-processing: Proposal Matching for ScanRefer
    SCANREFER_OUTPUT="${OUTPUT_BASE}_scanrefer"
    JSONL_FILE=$(find "$SCANREFER_OUTPUT" -name "*.jsonl" | head -1)
    if [ -n "$JSONL_FILE" ]; then
        echo ""
        echo "=== Step 3: ScanRefer Proposal Matching ==="
        python scripts/evaluation/eval_scanrefer.py \
            --input_file "$JSONL_FILE" \
            --scannet_dir data/media/scannet/ \
            --num_workers 8
    else
        echo "WARNING: No JSONL file found for ScanRefer post-processing"
    fi
fi

echo ""
echo "=== All evaluations complete ==="
