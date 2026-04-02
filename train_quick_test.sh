#!/bin/bash
# Quick test script - runs 200 iterations to verify setup
# Expected time: 1-2 minutes on 1xH100

set -e  # Exit on error

# Activate conda environment to ensure correct Python is used
source /zhaokunxiang/miniconda3/bin/activate base

echo "=================================================="
echo "Running QUICK TEST (200 iterations)"
echo "Expected time: 1-2 minutes on 1xA100"
echo "Using Python: $(which python3)"
echo "=================================================="

RUN_ID=quick_test_$(date +%Y%m%d_%H%M%S) \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
VAL_LOSS_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=================================================="
echo "Quick test completed!"
echo "Check logs/ directory for results"
echo "=================================================="
