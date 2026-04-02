# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parameter Golf is an OpenAI Model Craft Challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s. Models are evaluated by compression performance (bits per byte) on the FineWeb validation set using tokenizer-agnostic evaluation.

The challenge optimizes for L(N) - the lowest loss given a fixed number of parameters (N), unconstrained by data, compute, steps, or architecture.

## Training Commands

### Local Development (Mac with Apple Silicon)

Download FineWeb dataset (1024-token vocabulary, 10 training shards for quick iteration):
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

Run MLX training (smoke test, 200 iterations):
```bash
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

### Remote GPU Training (PyTorch)

Download full dataset (80 training shards = 8B tokens):
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

Single GPU training:
```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Multi-GPU training (8xH100 example):
```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Key Environment Variables

**Training Control:**
- `ITERATIONS`: Number of training steps (default: 20000)
- `TRAIN_BATCH_TOKENS`: Tokens per batch (default: 524288)
- `TRAIN_SEQ_LEN`: Sequence length (default: 1024)
- `MAX_WALLCLOCK_SECONDS`: Time limit in seconds (default: 600 = 10 minutes)
- `VAL_LOSS_EVERY`: Validation frequency (default: 1000, set to 0 to skip periodic validation)
- `SEED`: Random seed (default: 1337)

**Model Architecture:**
- `NUM_LAYERS`: Transformer layers (default: 9)
- `MODEL_DIM`: Model dimension (default: 512)
- `NUM_HEADS`: Attention heads (default: 8)
- `NUM_KV_HEADS`: KV heads for GQA (default: 4)
- `VOCAB_SIZE`: Vocabulary size (default: 1024)
- `TIE_EMBEDDINGS`: Tie input/output embeddings (default: 1)

**Optimizer:**
- `MATRIX_LR`, `SCALAR_LR`, `TIED_EMBED_LR`: Learning rates for different parameter groups
- `MUON_MOMENTUM`, `MUON_BACKEND_STEPS`: Muon optimizer settings
- `BETA1`, `BETA2`, `ADAM_EPS`: Adam optimizer settings

## Architecture

### Training Scripts

**train_gpt.py** (PyTorch): Main training script for GPU clusters. Supports distributed training via torchrun. Hard limit of 1500 lines to keep it readable for newcomers.

**train_gpt_mlx.py** (MLX): Adapted version for Apple Silicon Macs. Uses MLX framework instead of PyTorch. Same model architecture and hyperparameters.

### Model Components (train_gpt.py)

- **GPT**: Main transformer model with configurable depth, width, and tied embeddings
- **Block**: Transformer block with RMSNorm, CausalSelfAttention, and MLP
- **CausalSelfAttention**: Grouped-query attention (GQA) with rotary positional embeddings (RoPE)
- **MLP**: SwiGLU activation with configurable expansion ratio
- **RMSNorm**: Root mean square layer normalization
- **Rotary**: RoPE implementation for positional encoding

### Test-Time Training (TTT)

The codebase supports test-time training with LoRA for evaluation:
- **BatchedTTTLoRA**: Applies low-rank adaptation during inference
- **BatchedLinearLoRA**: LoRA-augmented linear layers
- Controlled by `TTT_LORA_RANK`, `TTT_LORA_LR`, `TTT_CHUNK_SIZE` environment variables

### Muon Optimizer

Custom optimizer from modded-nanogpt that uses Newton-Schulz iteration to orthogonalize matrix gradients. Key function: `zeropower_via_newtonschulz5()` normalizes 2D update matrices before applying them.

### Evaluation

**Tokenizer-Agnostic Metrics:**
- `val_loss`: Token cross-entropy (natural log)
- `val_bpb`: Bits per byte - the official challenge metric
- BPB calculation accounts for SentencePiece tokenizer specifics (boundary tokens, leading spaces, byte fallbacks)

**Artifact Size Calculation:**
- Code bytes (from train_gpt.py) + compressed model bytes (int8 quantized + zlib)
- Hard limit: 16,000,000 bytes (decimal 16MB, not 16 MiB)
- No external downloads or network calls allowed during evaluation

## Submission Structure

Leaderboard submissions go in `/records/track_10min_16mb/<date>_<name>/`:
- `README.md`: Detailed explanation of approach
- `submission.json`: Metadata (name, GitHub ID, val_bpb, etc.)
- `train.log` or multiple seed logs: Training output demonstrating statistical significance
- `train_gpt.py`: Standalone training script (must compile and run within the records folder)

Non-record submissions (unlimited compute, interesting approaches) go in `/records/track_non_record_16mb/`.

### Statistical Significance Requirements

New SOTA records must:
1. Beat existing SOTA by at least 0.005 nats
2. Demonstrate improvement at p < 0.01 (typically 3+ training runs with different seeds)
3. Run reproducibly in under 10 minutes on 8xH100 SXM

## Data Pipeline

**FineWeb Dataset**: Cached and pre-tokenized versions available via HuggingFace
- Validation: Fixed first-50k-document set (`fineweb_val_*.bin`)
- Training: Shuffled export in 100M token shards (`fineweb_train_*.bin`)
- Default download: 80 training shards (8B tokens)

**Tokenizers**: Located in `data/tokenizers/`
- Default: `fineweb_1024_bpe.model` (SentencePiece BPE with 1024 vocab)
- Custom tokenizers allowed but scrutinized carefully for correct BPB calculation

**Custom Tokenizer Workflow**: See `data/README.md` for rebuilding tokenizers from published docs using `download_hf_docs_and_tokenize.py`.

## Key Constraints

- **Time Limit**: 10 minutes training + up to 10 minutes evaluation on 8xH100s
- **Artifact Size**: 16MB total (code + compressed model)
- **No External Access**: No downloads, training data access, or network calls during evaluation
- **Evaluation Freedom**: Any sequence length, any evaluation method (e.g., sliding window, test-time training)

## Development Notes

- Baseline config: 9 layers × 512 dim, 1024 vocab, tied embeddings, GQA with 8 heads / 4 KV heads, ~1.22 val_bpb
- Train on cheap GPUs first (1xH100 or smaller) before scaling to 8xH100
- Use `--train-shards 1` for fast local iteration (100M tokens)
- The `/records` folder contains SOTA submissions with detailed approaches - study these for inspiration
- `train_gpt.py` and `train_gpt_mlx.py` are starter scripts, not SOTA configs - competitive submissions go in `/records`