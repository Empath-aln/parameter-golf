#!/bin/bash
# Gauss-Newton training on single GPU with conda env 'gn'
set -e

# Activate gn conda environment
eval "$(conda shell.bash hook)"
conda activate gn

echo "=================================================="
echo "Gauss-Newton Training (single GPU)"
echo "Conda env: gn"
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "=================================================="

# Download data if not present
DATA_DIR=./data/datasets/fineweb10B_sp1024
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/fineweb_train_*.bin 2>/dev/null)" ]; then
    echo "Downloading FineWeb dataset..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

RUN_ID="gn_single_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
GN_MODE=1 \
GN_BETA=0.9 \
GN_SO_RATIO=0.1 \
GN_INNER_LR=1e-3 \
TRAIN_BATCH_TOKENS=131072 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

echo "Training completed!"
