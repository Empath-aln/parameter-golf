#!/bin/bash
# GN experiment: so_ratio=1, beta=0 (full second-order, no FW momentum)
set -e

eval "$(conda shell.bash hook)"
conda activate gn

DATA_DIR=./data/datasets/fineweb10B_sp1024
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR/fineweb_train_*.bin 2>/dev/null)" ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
fi

RUN_ID="gn_so1_beta0_$(date +%Y%m%d_%H%M%S)" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
GN_MODE=1 \
GN_BETA=0 \
GN_SO_RATIO=1 \
TRAIN_BATCH_TOKENS=131072 \
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
