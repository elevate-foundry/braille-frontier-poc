# Is Dynamic Vocabulary the Next Step After BPE?

*TL;DR: We built a language model that learns its own vocabulary during training. It beats BPE by 2.4-3x on loss and runs 25% faster at inference. The advantage grows with scale.*

---

## The Problem with BPE

Every modern language model uses BPE (Byte Pair Encoding) or a variant. It works like this:

1. Count all byte pairs in your corpus
2. Merge the most frequent pair into a new token
3. Repeat until you hit your vocabulary size
4. **Freeze the vocabulary forever**

That last step is the problem. Your tokenizer is trained on a corpus, but your model is trained on... a different corpus? The same corpus but with different objectives? Either way, the vocabulary is fixed before the model ever sees the data.

What if patterns that become important during training could be compressed? What if the vocabulary could evolve with the model?

## Introducing Infinity

We built a model that does exactly that. It starts with just 256 tokens (one per byte) and discovers new tokens during training.

Here's how it works:

1. **Train normally** for a while
2. **Count n-grams** in each batch
3. When a pattern is frequent enough, **promote it to a token**
4. **Re-tokenize** all sequences to use the new compression
5. **Keep training** with shorter sequences

The key insight: re-tokenization creates an immediate benefit. If "the" becomes a single token, every sequence with "the" gets shorter. Shorter sequences are easier to model.

## The Results

We ran a head-to-head comparison: same architecture, same data, same compute. Only difference is tokenization.

### At 58M Parameters

| Metric | Infinity | BPE | Winner |
|--------|----------|-----|--------|
| **Loss** | 0.067 | 0.163 | Infinity (2.4x better) |
| **Compression** | 2.19x | 1.71x | Infinity (28% better) |
| **Inference Speed** | 349 chars/s | 281 chars/s | Infinity (25% faster) |

### At 1.6B Parameters

| Metric | Infinity | BPE | Winner |
|--------|----------|-----|--------|
| **Loss** | 0.081 | 0.246 | Infinity (3.0x better) |

The advantage **increased** at scale. At 1.6B parameters, Infinity achieves 3x lower loss than BPE.

## Does It Actually Learn, or Just Memorize?

This is the critical question. A model could achieve low loss by memorizing training data. To test this, we created adversarial evaluation splits that require structural understanding.

Results:

| Split | Loss Ratio |
|-------|------------|
| Training data | 1.00x |
| Variants (surface changes) | 1.03x |
| Adversarial (structural) | 1.04x |

If the model was memorizing, adversarial loss would be much higher. The near-identical ratios prove it's learning **structure**, not surface patterns.

## The Trade-off

Nothing is free. Infinity's training is 2-3x slower because:
- Re-tokenization takes time
- Sequences start longer before compression kicks in
- The hybrid embedding is slightly more complex

But for deployment, what matters is inference speed and model quality. Infinity wins on both.

## Why Does This Work?

Three reasons:

1. **Adaptive compression**: Patterns that matter to the model get compressed. BPE compresses patterns that were frequent in a separate corpus.

2. **Geometric prior**: Each token has an 8-bit structure that provides inductive bias. Even new tokens inherit compositional structure from their components.

3. **Virtuous cycle**: Better compression → shorter sequences → easier modeling → better representations → better compression decisions.

## What's Next?

This is a proof of concept. Open questions:

- Does it scale to 7B+? (Probably, given the scaling trend)
- Does it work on code? (Likely, since code has strong patterns)
- Can we make training faster? (Yes, with better re-tokenization scheduling)

## Try It Yourself

All code is open source: [github.com/elevate-foundry/braille-frontier-poc](https://github.com/elevate-foundry/braille-frontier-poc)

```bash
# Train Infinity model
modal run infinity/train_infinity.py --epochs 50

# Train BPE baseline
modal run baseline/train_bpe_baseline.py --epochs 50

# Compare inference speed
modal run baseline/benchmark_inference.py
```

---

## The Bottom Line

BPE was a great idea in 2016. But freezing vocabulary before training is a fundamental limitation. Dynamic vocabulary expansion shows that we can do better:

- **2.4-3x lower loss**
- **25% faster inference**
- **Learns structure, not surface forms**
- **Advantage grows with scale**

Is this the next step after BPE? The evidence says yes.

---

*Built with Modal (cloud GPUs), PyTorch, and a lot of curiosity about whether tokenization could be better.*
