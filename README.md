# Braille Frontier Model - Proof of Concept

A minimal implementation demonstrating a language model that "thinks" in Braille for lower compute requirements.

## Core Thesis

Train a model using Braille's 256-token vocabulary (8-dot cells) as the internal representation:
- **Smaller embedding tables**: 256 vs 50k+ tokens
- **Shorter sequences**: Grade 2 contractions compress text
- **Geometric inductive bias**: Dot patterns provide mathematical structure
- **Faster inference**: Smaller KV cache, faster attention

The model reasons in Braille-space and translates to English only at output.

## Architecture

```
Text → Braille Tokenizer → Hybrid Embedding → Transformer → Braille Output → English Decoder
                              ↓
                    E(t) = E_learned(t) + gate(t) * W·bits(t)
```

## Files

- `tokenizer.py` - Text ↔ Braille conversion (Grade 1 for simplicity)
- `embedding.py` - Hybrid embedding with geometric prior + learned residual
- `model.py` - Minimal transformer backbone
- `train.py` - Toy training loop
- `inference.py` - Demo script

## Usage

```bash
pip install torch
python train.py      # Train on toy data
python inference.py  # Run inference demo
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
