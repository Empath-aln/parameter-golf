#!/bin/bash
# 4xA100 training script - multi-GPU training
# Expected time: ~15-20 minutes on 4xA100
# Expected val_bpb: ~1.22 (baseline)

set -e  # Exit on error

# Activate conda environment to ensure correct Python is used
source /zhaokunxiang/miniconda3/bin/activate base

echo "=================================================="
echo "Running 4-GPU TRAINING (10-minute limit)"
echo "Multi-GPU configuration"
echo "Expected val_bpb: ~1.22 (baseline)"
echo "Using Python: $(which python3)"
echo "=================================================="

# Check GPU count
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "Found $GPU_COUNT GPUs available"
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "WARNING: Found only $GPU_COUNT GPUs, but script expects 4"
    echo "Continuing anyway..."
fi

RUN_ID=4gpu_$(date +%Y%m%d_%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=4 train_gpt.py

echo ""
echo "=================================================="
echo "Training completed!"
echo "Results saved to: logs/${RUN_ID}/"
echo "Check final val_bpb in the last few lines above"
echo "=================================================="
