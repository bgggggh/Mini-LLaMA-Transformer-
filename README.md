# Mini-LLaMA-Transformer

A complete Transformer language model built **entirely from scratch** using only PyTorch primitives — no `nn.Linear`, `nn.Transformer`, or high-level modules. Follows the LLaMA architecture design (RoPE + SwiGLU + RMSNorm).

Built as part of [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/).

## Highlights

- **BPE Tokenizer** — Full byte-pair encoding implementation with training, encoding, and decoding. Output matches GPT-2's tokenizer (tiktoken) exactly.
- **LLaMA-style Transformer** — Multi-head self-attention with Rotary Position Embeddings (RoPE), SwiGLU feed-forward networks, RMSNorm, and pre-norm residual connections.
- **Training from Scratch** — AdamW optimizer, cosine LR schedule with warmup, gradient clipping, and checkpointing — all implemented from scratch.
- **Efficient Data Pipeline** — Chunked BPE training and memory-mapped dataset loading for handling large corpora on consumer hardware.

## Architecture

```
Input Token IDs
       ↓
  Token Embedding
       ↓
┌─────────────────────────┐
│   Transformer Block × N │
│                         │
│   RMSNorm → MHA (RoPE) │
│        + residual       │
│   RMSNorm → SwiGLU FFN │
│        + residual       │
└─────────────────────────┘
       ↓
   Final RMSNorm
       ↓
     LM Head
       ↓
  Logits (vocab_size)
```

## Results

Trained a 5M-parameter model on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (540M tokens):

| Metric | Value |
|--------|-------|
| Parameters | ~5M |
| Training Tokens | 541M |
| Final Train Loss | 1.99 |
| Perplexity | ~6.7 |
| Training Steps | 5,000 |
| Device | Apple Silicon (MPS) |

### Sample Generation

**Prompt:** `Once upon a time`

> Once upon a time, there was a little boy named Jack. He had a big, orange box of paper. Everyday, he was an amazing boy who loved to draw.
> One day, Jack was playing with his orange blocks when he accidentally dropped the red pencil. He was very sad because his fault didn't like the blue balloon. But then Tim heard the noise and came to see what it was...

## Project Structure

```
Mini-LLaMA-Transformer/
├── cs336_basics/
│   ├── model.py              # Transformer model + training utilities
│   └── tokenizer.py          # BPE tokenizer
├── tests/
│   └── adapters.py           # Test interface
├── tokenize_data.py          # Data preprocessing script
├── train.py                  # Training loop
├── generate.py               # Text generation
├── data/                     # Datasets and tokenizer
└── checkpoints/              # Saved model weights
```

## What's Implemented from Scratch

### Tokenizer (`tokenizer.py`)

| Component | Description |
|-----------|-------------|
| Pre-tokenization | GPT-2-style regex splitting with Unicode support |
| BPE Training | Greedy pair merging with frequency-weighted counting |
| Encode / Decode | Merge-priority-based encoding, byte-level decoding |
| Special Tokens | Proper handling of `<\|endoftext\|>` and custom tokens |
| Memory Efficiency | Chunked file reading for large corpora |

### Model (`model.py`)

| Component | Description |
|-----------|-------------|
| `linear` | Matrix multiplication (no `nn.Linear`) |
| `embedding` | Lookup table (no `nn.Embedding` for forward pass) |
| `silu` | SiLU activation: `x * σ(x)` |
| `softmax` | Numerically stable softmax |
| `rmsnorm` | Root Mean Square normalization |
| `swiglu` | Gated FFN: `W₂(SiLU(W₁x) ⊙ W₃x)` |
| `rope` | Rotary Position Embeddings |
| `scaled_dot_product_attention` | `softmax(QKᵀ/√dₖ + mask) · V` |
| `multihead_self_attention` | Multi-head attention with RoPE and causal mask |
| `transformer_block` | Pre-norm block with residual connections |
| `transformer_lm` | Full language model |

### Training Utilities (`model.py`)

| Component | Description |
|-----------|-------------|
| `cross_entropy` | Numerically stable loss using logsumexp |
| `AdamW` | Optimizer with bias correction and decoupled weight decay |
| `get_lr_cosine_schedule` | Linear warmup + cosine decay |
| `gradient_clipping` | L2 norm-based gradient clipping |
| `get_batch` | Random batch sampling from token arrays |
| `save/load_checkpoint` | Full state serialization (model + optimizer + step) |

## Quick Start

### Setup

```bash
git clone https://github.com/yourusername/Mini-LLaMA-Transformer.git
cd Mini-LLaMA-Transformer
pip install uv
uv sync
```

### Download Data

```bash
mkdir -p data && cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..
```

### Tokenize

```bash
uv run python tokenize_data.py \
  --input_path data/TinyStoriesV2-GPT4-train.txt \
  --val_input_path data/TinyStoriesV2-GPT4-valid.txt \
  --vocab_size 10000
```

### Train

```bash
uv run python train.py \
  --vocab_size 10000 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 8 \
  --d_ff 512 \
  --context_length 256 \
  --batch_size 16 \
  --max_steps 5000
```

### Generate

```bash
uv run python generate.py --prompt "Once upon a time"
```

## Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Vocabulary Size | 10,000 |
| Model Dimension | 256 |
| Layers | 4 |
| Attention Heads | 8 |
| FFN Dimension | 512 |
| Context Length | 256 |
| Batch Size | 16 |
| Max Learning Rate | 1e-3 |
| Min Learning Rate | 1e-4 |
| Warmup Steps | 200 |
| Weight Decay | 0.01 |
| RoPE θ | 10,000 |

## Tech Stack

- **Python** — Core implementation
- **PyTorch** — Tensor operations and autograd only (no high-level modules for forward pass)
- **einops** — Readable tensor reshaping for multi-head attention
- **NumPy** — Data loading with memory mapping
- **regex** — Unicode-aware pre-tokenization

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020)
- [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)

## License

MIT
