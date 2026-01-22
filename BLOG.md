# What I Learned By Getting Dynamic Vocabulary Wrong

*TL;DR: We built a language model that learns its own vocabulary during training. Initial results looked amazing—2.4x better loss! Then we measured it correctly and discovered we'd made a fundamental error. Here's what went wrong and what it teaches us about ML evaluation.*

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

## The Initial Results (What We Thought We Found)

We ran a head-to-head comparison: same architecture, same data, same compute.

| Metric | Infinity | BPE | Apparent Winner |
|--------|----------|-----|--------|
| **Loss** | 0.067 | 0.163 | Infinity (2.4x better!) |
| **Compression** | 2.19x | 1.71x | Infinity |

At 1.6B parameters, the advantage grew to 3x. We were excited.

## Then Someone Asked the Right Question

A reviewer pointed out: **"You're comparing apples to oranges."**

Cross-entropy loss is computed *per token*. If Infinity uses fewer tokens (due to compression), of course the loss looks better—you're making fewer predictions!

The correct metric is **Bits Per Character (BPC)**, which normalizes by the actual text length:

```
BPC = total_loss / (total_characters × ln(2))
```

## The Real Results

We computed BPC. The results were... not what we expected.

| Model | BPC | Per-char Perplexity |
|-------|-----|---------------------|
| **BPE** | 0.14 | 1.10 |
| **Infinity** | 33.92 | 16,000,000,000+ |

**Infinity was performing worse than random chance.**

## What Went Wrong?

Investigation revealed multiple compounding failures:

### 1. Tokenization State Mismatch

During training, we re-tokenized data whenever new contractions were discovered. But we didn't save the exact tokenization state with the checkpoint.

When evaluating, we re-tokenized from scratch—but the model's weights were tuned for a *different* tokenization that no longer existed.

### 2. "Compression" Was Actually Expansion

The reported "2.19 chars/token" compression was measured during training. When we re-tokenized for evaluation, we got **0.48 chars/token**—the sequences were actually *longer* than the original.

### 3. Positional Chaos

Every time we re-tokenized, the positional relationships changed. Token 5 might become token 3. The attention patterns the model learned became meaningless.

### 4. Tiny Dataset

42K samples for a 1.6B parameter model? The model could memorize the entire dataset. We weren't testing generalization at all.

## The Lessons

### 1. Always Use BPC for Tokenization Comparisons

Per-token loss is meaningless when comparing different tokenization schemes. This should have been obvious, but we got excited by good-looking numbers.

### 2. Save Your Tokenization State

If vocabulary changes during training, you MUST save the exact token-to-pattern mapping. Otherwise evaluation is impossible.

### 3. Re-tokenization Is Dangerous

Changing the input distribution mid-training causes instability. We saw 15x loss spikes after vocabulary expansion. The model never fully recovered.

### 4. Small Datasets Prove Nothing

You can't validate language modeling claims on 42K samples. Period.

## Is Dynamic Vocabulary Still Worth Exploring?

**Yes, but not like this.**

The core idea—that vocabulary should evolve with the model—is still theoretically appealing. But a proper implementation would need:

- Correct evaluation metrics from day one
- Careful state management
- Web-scale data
- Strategies for positional stability (maybe RoPE or ALiBi)
- Gradual token introduction with embedding warmup

## The Real Bottom Line

We thought we'd found the next step after BPE. We hadn't.

But we learned something more valuable: **how easy it is to fool yourself with the wrong metrics.**

The exciting-looking results (2.4x better!) were an artifact of comparing incomparable numbers. The moment we used the right metric, the illusion collapsed.

This is why rigorous evaluation matters. This is why peer review matters. This is why you should always ask: "Am I measuring what I think I'm measuring?"

---

*Built with Modal, PyTorch, and a healthy dose of humility.*

**Code**: [github.com/elevate-foundry/braille-frontier-poc](https://github.com/elevate-foundry/braille-frontier-poc)
