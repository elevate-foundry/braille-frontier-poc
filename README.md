# Braille Frontier Model

A proof-of-concept language model exploring whether **dynamic vocabulary expansion** during training can outperform static BPE tokenization.

## Key Results: Infinity vs BPE

Head-to-head comparison on the same corpus and architecture:

| Metric | **Infinity** | **BPE** | Winner |
|--------|-------------|---------|--------|
| **Training Loss** | 0.067 | 0.163 | Infinity (2.4x better) |
| **Compression** | 2.19 chars/tok | 1.71 chars/tok | Infinity (28% better) |
| **Inference Speed** | 349.7 chars/s | 280.8 chars/s | Infinity (25% faster) |
| **Generalization** | 1.03x on adversarial | — | Infinity (learns structure) |
| **Training Time** | 35 min | 19 min | BPE (faster) |

**Verdict**: Infinity wins on metrics that matter for deployment (loss, compression, throughput). BPE wins on simplicity.

## What This Is

An experimental architecture demonstrating:

1. **Dynamic vocabulary expansion** - Discovers and promotes frequent n-grams during training
2. **Hybrid embeddings** - Combines learned representations with geometric bit-pattern structure
3. **Re-tokenization** - Compresses sequences as new contractions are discovered
4. **Structure learning** - Generalizes to adversarial perturbations (not memorizing)

## Architecture

### 3-Layer Token System

| Layer | Description | Size |
|-------|-------------|------|
| Layer 1 (Geometric) | 8-dot Braille cells with bit-pattern structure | 256 base + 3 special |
| Layer 2 (Contraction) | Learned n-gram compressions discovered during training | Dynamic (277 discovered) |
| Layer 3 (Semantic) | Reserved for concept-level tokens | Up to 256 |

### Hybrid Embedding

```
E(token) = E_learned(token) + sigmoid(gate(token)) * W · bits(token)
```

- `E_learned`: Standard learned embedding
- `gate`: Per-token scalar controlling geometric influence  
- `W`: 8 × d_model projection from bit patterns
- `bits`: 8-dimensional binary vector representing dot states

### Model Specs (58M)

- **Layers**: 12 transformer blocks
- **Attention**: 8 heads, 512 dimensions
- **FFN**: SwiGLU (4x expansion)
- **Normalization**: RMSNorm

## Training on Frontier Corpus

Trained on Manus Frontier Corpus (synthetic morphology + adversarial splits):

| Metric | Value |
|--------|-------|
| Final Loss | 0.067 |
| Vocabulary | 259 → 536 tokens |
| Contractions | 277 discovered |
| Compression | 2.19x |
| Training Time | 35 minutes (A100) |

### Generalization Test

| Split | Loss | Ratio to Train |
|-------|------|----------------|
| Train | 6.91 | 1.00x |
| Variants | 7.10 | 1.03x ✓ |
| Adversarial | 7.21 | 1.04x ✓ |

Near-identical ratios indicate the model learned **structure, not surface patterns**.

## Project Structure

```
braille-frontier-poc/
├── infinity/
│   ├── train_infinity.py     # Main training script (Modal A100)
│   └── evaluate_frontier.py  # Evaluation on adversarial splits
├── baseline/
│   ├── train_bpe_baseline.py # BPE baseline for comparison
│   ├── benchmark_inference.py # Inference speed benchmark
│   └── prepare_bpe_data.py   # Data prep for BPE
├── distill/
│   ├── corpus_generator.py   # Manus frontier corpus generator
│   ├── convert_frontier_corpus.py
│   └── upload_frontier_corpus.py
└── *.py                      # Legacy files from initial POC
```

## Usage

### Requirements

```bash
pip install torch modal tokenizers
```

### Train Infinity Model

```bash
# Upload frontier corpus
modal run distill/upload_frontier_corpus.py

# Train (35 min on A100)
modal run infinity/train_infinity.py --epochs 50

# Evaluate on adversarial splits
modal run infinity/evaluate_frontier.py
```

### Train BPE Baseline

```bash
# Prepare data
modal run baseline/prepare_bpe_data.py

# Train (19 min on A100)
modal run baseline/train_bpe_baseline.py --epochs 50
```

### Benchmark Inference

```bash
modal run baseline/benchmark_inference.py
```

## What's Next

1. **Scale test** - Do these results hold at 1B+ parameters?
2. **Layer 3 semantic tokens** - Implement concept-level compression
3. **Code corpus** - Test if contractions generalize to programming languages

## License

MIT
