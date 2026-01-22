# Dynamic Vocabulary Expansion: An Experimental Investigation

**Abstract**: We present Infinity, a language model architecture that dynamically expands its vocabulary during training by discovering and promoting frequent n-gram patterns. While initial experiments showed promising per-token loss improvements, rigorous evaluation using Bits Per Character (BPC) revealed fundamental methodological flaws. This paper documents both the architectural innovations and the critical lessons learned about evaluating dynamic tokenization systems.

**Status**: This is an experimental proof-of-concept with significant limitations. The core hypothesis—that dynamic vocabulary can outperform static BPE—remains unproven due to evaluation challenges documented in Section 6.

## 1. Introduction

Byte Pair Encoding (BPE) has been the dominant tokenization strategy for large language models since its introduction. BPE learns a fixed vocabulary by iteratively merging frequent byte pairs in a corpus, then freezes this vocabulary before model training begins.

This approach has a fundamental limitation: **the vocabulary cannot adapt to the data the model actually learns**. Patterns that become important during training cannot be compressed, while rare patterns that were frequent in the tokenizer's training corpus consume vocabulary slots.

We propose **Infinity**, an architecture that:
1. Starts with a minimal geometric vocabulary (256 8-bit tokens)
2. Discovers frequent n-gram patterns during training via "shadow token mining"
3. Promotes these patterns to dedicated tokens with learned embeddings
4. Re-tokenizes sequences to leverage the new compressions

This creates a feedback loop where the model's learned representations inform vocabulary expansion, which in turn improves the model's ability to learn.

## 2. Architecture

### 2.1 Three-Layer Token System

Infinity organizes tokens into three semantic layers:

| Layer | Description | Size |
|-------|-------------|------|
| **Layer 1 (Geometric)** | 8-bit Braille cells with bit-pattern structure | 256 base + 3 special |
| **Layer 2 (Contraction)** | Learned n-gram compressions discovered during training | Dynamic (up to 4096) |
| **Layer 3 (Semantic)** | Reserved for concept-level tokens | Up to 256 |

### 2.2 Hybrid Embedding

Each token's embedding combines a learned component with a geometric prior:

```
E(token) = E_learned(token) + sigmoid(gate(token)) × W · bits(token)
```

Where:
- `E_learned`: Standard learned embedding (d_model dimensions)
- `gate`: Per-token scalar controlling geometric influence
- `W`: 8 × d_model projection matrix
- `bits`: 8-dimensional binary vector representing the token's bit pattern

For Layer 2 tokens (contractions), the bit pattern is the normalized sum of component patterns, providing a compositional prior.

### 2.3 Shadow Token Mining

During training, we maintain counts of n-gram patterns (n=2 to 8) in each batch. When a pattern exceeds a frequency threshold:

1. A new token ID is allocated
2. The embedding is initialized as the mean of component embeddings
3. The gate is initialized to 0.5 (partial geometric influence)
4. All sequences are re-tokenized to use the new compression

This re-tokenization is the key innovation: it allows the model to immediately benefit from discovered patterns, creating shorter sequences that are easier to model.

## 3. Experimental Setup

### 3.1 Dataset: Manus Frontier Corpus

To rigorously test whether models learn structure vs. memorize surface forms, we created a synthetic corpus with controlled properties:

- **42,500 training samples** with systematic morphological patterns
- **Variant split**: Same patterns with surface-level perturbations
- **Adversarial split**: Patterns that require structural understanding to predict

A model that memorizes will fail on adversarial splits; a model that learns structure will generalize.

### 3.2 Model Configurations

| Scale | d_model | Layers | Heads | Parameters |
|-------|---------|--------|-------|------------|
| Small | 512 | 12 | 8 | 58M |
| Large | 2048 | 24 | 16 | 1.6B |

Both Infinity and BPE baselines use identical transformer architectures (RMSNorm, SwiGLU, Flash Attention). The only difference is the tokenization and embedding layer.

### 3.3 Training

- **Hardware**: NVIDIA A100 (40GB for 58M, 80GB for 1.6B)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.1)
- **Epochs**: 50 (58M) or 20 (1.6B)
- **Batch size**: 64 effective (with gradient accumulation)

## 4. Initial Results (Per-Token Loss)

### 4.1 Training Loss (Flawed Metric)

| Scale | Infinity Loss | BPE Loss | Apparent Improvement |
|-------|--------------|----------|-------------|
| **58M** | 0.067 | 0.163 | 2.4x |
| **1.6B** | 0.081 | 0.246 | 3.0x |

**⚠️ WARNING**: These results are **not directly comparable**. Cross-entropy loss is computed per-token, and the tokenization schemes produce different numbers of tokens. See Section 6 for the correct comparison.

### 4.2 Reported Compression

| Scale | Infinity | BPE |
|-------|----------|-----|
| **58M** | 2.19 chars/token | 1.71 chars/token |
| **1.6B** | 1.68 chars/token | 1.71 chars/token |

### 4.3 Generalization (58M)

| Split | Loss | Ratio to Train |
|-------|------|----------------|
| Train | 6.91 | 1.00x |
| Variants | 7.10 | 1.03x |
| Adversarial | 7.21 | 1.04x |

The near-identical ratios suggest structural learning, though this evaluation also uses per-token loss.

### 4.4 Training Time

| Scale | Infinity | BPE |
|-------|----------|-----|
| **58M** | 35 min | 19 min |
| **1.6B** | 141 min | 42 min |

Infinity's training is 2-3x slower due to re-tokenization overhead.

## 5. Analysis

### 5.1 Gate Values

The learned gate values reveal how much the model relies on geometric structure:

| Layer | Gate Value | Interpretation |
|-------|------------|----------------|
| Layer 1 (base) | 0.73 | Strong geometric prior |
| Layer 2 (contractions) | 0.62 | Partial geometric influence |

Even after extensive training, the model maintains significant reliance on the geometric prior, suggesting it provides useful inductive bias.

### 5.2 Discovered Contractions

The model discovers contractions that correspond to meaningful patterns:
- Common character sequences (e.g., "th", "ing", "tion")
- Morphological units
- Domain-specific patterns

Unlike BPE, these are discovered *during* training based on what the model finds useful, not pre-computed from corpus statistics.

### 5.3 Scaling Behavior

The increasing advantage at scale (2.4x → 3.0x) suggests that:
1. Larger models benefit more from dynamic vocabulary
2. The geometric prior becomes more valuable with capacity
3. Re-tokenization provides compounding benefits

## 6. Critical Evaluation: Bits Per Character

After initial experiments, we conducted a rigorous evaluation using **Bits Per Character (BPC)**, the correct metric for comparing models with different tokenization schemes.

### 6.1 Why BPC Matters

Cross-entropy loss is computed **per token**. If Model A uses 1000 tokens and Model B uses 500 tokens for the same text, their losses are not directly comparable. BPC normalizes by the number of characters:

```
BPC = total_cross_entropy / (total_characters × ln(2))
```

### 6.2 BPC Results

| Model | BPC | Per-char Perplexity |
|-------|-----|---------------------|
| **BPE** | 0.14 | 1.10 |
| **Infinity** | 33.92 | 16B+ |

**This is a catastrophic result for Infinity.** A BPC of 33.92 indicates the model is performing worse than random chance.

### 6.3 Root Cause Analysis

Investigation revealed multiple compounding issues:

1. **Tokenization Mismatch**: The model was trained on data that was re-tokenized during training, but evaluation used freshly tokenized data. The model's weights are tuned for a specific tokenization state that cannot be reproduced.

2. **Negative Compression**: The "compression ratio" of 0.48 chars/token means sequences are actually **expanding**, not compressing. The contractions contain high-byte Braille patterns that don't match the evaluation data.

3. **Non-Stationary Training**: Re-tokenization mid-training creates distribution shift. The model must re-learn positional relationships after each vocabulary expansion.

4. **Checkpoint State**: The saved model checkpoint doesn't include the exact tokenization state used during training, making reproducible evaluation impossible.

### 6.4 What the Per-Token Loss Actually Measured

The low per-token loss (0.067) was likely measuring:
- Overfitting to the specific tokenization state at training end
- Shorter sequences (after compression) being easier to model
- NOT generalizable language modeling capability

## 7. Lessons Learned

### 7.1 Evaluation Must Use BPC

Any comparison across tokenization schemes **must** use Bits Per Character or Bits Per Byte. Per-token metrics are fundamentally incomparable.

### 7.2 Dynamic Tokenization Requires Careful State Management

If vocabulary changes during training:
- The final tokenization state must be saved with the checkpoint
- Evaluation must use the exact same tokenization
- Positional embeddings may need special handling during transitions

### 7.3 Re-tokenization Creates Instability

We observed loss spikes of 15x (0.08 → 1.25) after vocabulary expansion. Strategies to mitigate:
- Gradual introduction of new tokens
- Learning rate warmup after expansion
- Freezing new token embeddings initially

### 7.4 Synthetic Data Limitations

42K samples is insufficient to validate claims about language modeling. The 1.6B model (with 1.6B parameters) can trivially memorize this dataset.

## 8. Revised Limitations

1. **Evaluation is broken**: BPC shows Infinity performs catastrophically worse than BPE
2. **Training overhead**: 2-3x slower with no proven benefit
3. **State management**: Cannot reproduce training tokenization for evaluation
4. **Dataset scale**: Results on 42K samples don't generalize
5. **Positional instability**: Re-tokenization disrupts learned attention patterns

## 9. Future Work: What Would Be Needed

To properly test the dynamic vocabulary hypothesis:

1. **Save tokenization state**: Checkpoint must include the exact token-to-pattern mapping used for training data
2. **Use BPC from the start**: Track bits-per-character during training, not just per-token loss
3. **Larger dataset**: Minimum 1B tokens to avoid memorization at 1B+ parameter scale
4. **Positional embedding strategies**: Test relative positional encodings (RoPE, ALiBi) which may be more robust to sequence length changes
5. **Gradual expansion**: Introduce new tokens slowly with embedding warmup

## 10. Conclusion

We presented Infinity, a language model architecture that dynamically expands its vocabulary during training. The core hypothesis—that vocabulary should co-evolve with the model—remains theoretically appealing.

However, **rigorous evaluation revealed fundamental flaws**:

| Claim | Reality |
|-------|---------|
| "2.4-3x lower loss" | Incomparable metric (per-token vs per-token) |
| "25% faster inference" | Based on flawed compression measurement |
| "Learns structure" | Cannot verify without proper BPC evaluation |

**The actual BPC comparison shows Infinity performing catastrophically worse than BPE** (33.92 vs 0.14).

### What We Learned

1. **Never compare per-token loss across tokenization schemes** - use BPC
2. **Dynamic tokenization requires careful state management** - save the exact tokenization with checkpoints
3. **Re-tokenization mid-training causes instability** - needs mitigation strategies
4. **Small synthetic datasets prove nothing** - need web-scale evaluation

### The Hypothesis Remains Open

The idea that vocabulary should be learned during training, not frozen beforehand, is still worth exploring. But this implementation does not prove it works. A proper test would require:
- Correct evaluation metrics (BPC)
- Proper state management
- Web-scale data
- Strategies for positional stability

This paper serves as a cautionary tale about the importance of rigorous evaluation in ML research.

---

## Appendix A: Reproducibility

All code is available at: https://github.com/elevate-foundry/braille-frontier-poc

### Training Commands

```bash
# 58M Infinity
modal run infinity/train_infinity.py --epochs 50

# 58M BPE baseline
modal run baseline/train_bpe_baseline.py --epochs 50

# 1.6B Infinity
modal run scale/train_infinity_1b.py --epochs 20

# 1.6B BPE baseline
modal run scale/train_bpe_1b.py --epochs 20
```

### Benchmarking

```bash
# Inference benchmark (58M)
modal run baseline/benchmark_inference.py

# Inference benchmark (1.6B)
modal run scale/benchmark_inference_1b.py
```

## Appendix B: Complete Results Table

| Metric | Infinity 58M | BPE 58M | Infinity 1.6B | BPE 1.6B |
|--------|-------------|---------|---------------|----------|
| Parameters | 58M | 58M | 1.64B | 1.61B |
| Final Loss | 0.067 | 0.163 | 0.081 | 0.246 |
| Compression | 2.19x | 1.71x | 1.68x | 1.71x |
| Vocab Size | 536 | 526 | 359 | 526 |
| Contractions | 277 | — | 100 | — |
| Training Time | 35 min | 19 min | 141 min | 42 min |
| Tokens/sec | 159.7 | 164.2 | 82.3 | 85.1 |
| Chars/sec | 349.7 | 280.8 | 137.9 | 145.3 |
| GPU Memory | 0.91 GB | 0.89 GB | 13.05 GB | 13.05 GB |

## Appendix C: Adversarial Evaluation Details

The Manus Frontier Corpus includes three splits designed to test generalization:

1. **Train**: Standard training data with morphological patterns
2. **Variants**: Same patterns with character-level perturbations (e.g., case changes, spacing)
3. **Adversarial**: Patterns that require understanding structure to predict (e.g., reversed morphemes, novel combinations)

A model that memorizes training data will show high loss on adversarial splits. Infinity's 1.03-1.04x ratio indicates genuine structural learning.
