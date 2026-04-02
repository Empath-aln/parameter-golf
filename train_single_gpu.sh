#!/bin/bash
# Single GPU training script - full 10-minute run
# Expected time: ~10 minutes on 1xH100
# Expected val_bpb: ~1.22 (baseline)

set -e  # Exit on error

# Activate conda environment to ensure correct Python is used
source /zhaokunxiang/miniconda3/bin/activate base

echo "=================================================="
echo "Running SINGLE GPU TRAINING (10-minute limit)"
echo "Expected val_bpb: ~1.22 (baseline)"
echo "Using Python: $(which python3)"
echo "=================================================="

RUN_ID=single_gpu_$(date +%Y%m%d_%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=================================================="
echo "Training completed!"
echo "Results saved to: logs/${RUN_ID}/"
echo "Check final val_bpb in the last few lines above"
echo "=================================================="
