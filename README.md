# Braille Frontier Model

A proof-of-concept language model that uses Braille encoding (256 tokens) instead of standard BPE tokenization (32K-100K tokens). This explores whether a smaller, geometrically-structured vocabulary can reduce compute costs while maintaining language modeling capability.

## What This Is

This is an experimental architecture, not a production model. It demonstrates:

1. **A 3-layer token system** that maps text through geometric primitives, learned contractions, and semantic tokens
2. **Hybrid embeddings** that combine learned representations with structural information from Braille bit patterns
3. **Dynamic vocabulary expansion** via shadow token mining during training

## Architecture

### Token Layers

| Layer | Description | Size |
|-------|-------------|------|
| Layer 1 (Geometric) | 8-dot Braille cells with bit-pattern structure | 256 base + 3 special |
| Layer 2 (Contraction) | Learned n-gram compressions discovered during training | Up to 512 |
| Layer 3 (Semantic) | Reserved for concept-level tokens | Up to 256 |

### Hybrid Embedding

```
E(token) = E_learned(token) + gate(token) * W · bits(token)
```

- `E_learned`: Standard learned embedding (d_model dimensions)
- `gate`: Per-token scalar controlling geometric influence
- `W`: 8 × d_model projection from bit patterns
- `bits`: 8-dimensional binary vector representing dot states

### Model Specs

- **Parameters**: 58M
- **Layers**: 6 transformer blocks
- **Attention**: 8 heads, 512 dimensions
- **FFN**: SwiGLU with 1376 intermediate dimension
- **Normalization**: RMSNorm

## Training Results

Trained on 10K TinyStories (GPT-4 generated short stories) using Modal A100:

| Metric | Value |
|--------|-------|
| Initial Loss | 2.03 |
| Final Loss | 0.016 |
| Epochs | 50 |
| Training Time | ~45 minutes |
| Vocabulary Growth | 259 → 309 tokens |
| Contractions Discovered | 50 |

### Gate Values

- **Layer 1 Gate**: 0.72 (geometric structure remains influential)
- **Layer 2 Gate**: 0.62 (contractions partially use geometric prior)

## Theoretical Compute Comparison

Compared to a standard 50K-vocabulary model of the same size:

| Component | Standard | Braille | Reduction |
|-----------|----------|---------|-----------|
| Embedding parameters | 25.6M | 158K | 99.4% |
| Output projection | 25.6M | 158K | 99.4% |
| KV cache (per token) | 50K logits | 309 logits | 99.4% |

These savings are offset by longer sequences (Braille encodes ~1 character per token vs ~4 for BPE).

## Project Structure

```
braille-frontier-poc/
├── tokenizer.py          # Braille ↔ text conversion
├── embedding.py          # Hybrid embedding layer
├── model.py              # Base transformer
├── train.py              # Local training
├── inference.py          # Generation demo
├── modal_train.py        # A100 training via Modal
├── download_model.py     # Fetch checkpoints from Modal
├── infinity/
│   ├── layers.py         # 3-layer token system
│   ├── model.py          # InfinityModel with dynamic vocab
│   └── train_infinity.py # Training with shadow mining
└── distill/
    ├── download_corpus.py    # TinyStories download
    ├── convert_to_braille.py # Data preprocessing
    └── upload_data.py        # Modal volume upload
```

## Usage

### Requirements

```bash
pip install torch modal datasets
```

### Local Training (CPU/MPS)

```bash
python train.py
```

### Modal Training (A100)

```bash
# Download and prepare data
python distill/download_corpus.py
python distill/convert_to_braille.py
modal run distill/upload_data.py

# Train
modal run infinity/train_infinity.py --epochs 50

# Download checkpoint
python download_model.py
```

### Inference

```bash
python inference.py
```

## Limitations

- **Sequence length**: Braille encoding produces longer sequences than BPE (~4x for English)
- **Training data**: Only tested on TinyStories; unclear how it scales
- **Contractions not used in training**: Current implementation discovers contractions but doesn't re-encode training data with them
- **No benchmark comparisons**: Haven't compared perplexity against equivalent BPE model

## What's Next

Potential directions if this approach proves useful:

1. Re-tokenize training data with discovered contractions to measure actual compression benefit
2. Implement adaptive context windows that shrink as vocabulary density increases
3. Benchmark against BPE tokenization at equivalent compute budget
4. Explore Grade 2/3 Braille rules as initialization for contraction layer

## License

MIT
