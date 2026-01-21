# Distillation Pipeline

Train the Braille Frontier Model using knowledge distilled from large open-source models.

## Overview

```
Large Model (DeepSeek/Qwen/Llama) → Generate Text → Convert to Braille → Train Braille Model
```

## Setup

1. Get an OpenRouter API key from https://openrouter.ai/
2. Set the environment variable:
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

## Pipeline Steps

### Step 1: Generate Training Data

Generate diverse text samples from a large model:

```bash
cd /Users/ryanbarrett/Desktop/braille-frontier-poc

# Generate 10,000 samples using DeepSeek (cheapest, high quality)
python distill/generate_data.py --samples 10000 --model deepseek --output distill/data/generated.jsonl

# Or use other models:
# python distill/generate_data.py --samples 10000 --model qwen --output distill/data/generated.jsonl
# python distill/generate_data.py --samples 10000 --model llama --output distill/data/generated.jsonl
```

### Step 2: Convert to Braille

Convert the generated text to Braille training format:

```bash
python distill/convert_to_braille.py \
    --input distill/data/generated.jsonl \
    --output distill/data/braille_train.pt \
    --ctx-len 512
```

### Step 3: Upload Data to Modal

```bash
modal run distill/upload_data.py distill/data/braille_train.pt
```

### Step 4: Train on A100

```bash
modal run distill/train_distilled.py --epochs 50
```

### Step 5: Download Trained Model

```bash
modal run download_model.py
```

## Cost Estimates

| Model | Cost per 1M tokens | 10k samples (~2M tokens) |
|-------|-------------------|--------------------------|
| DeepSeek | $0.14 | ~$0.28 |
| Qwen 72B | $0.90 | ~$1.80 |
| Llama 70B | $0.90 | ~$1.80 |

Modal A100 training: ~$2-3/hour

## Expected Results

With 10,000 samples from a high-quality teacher model:
- Training loss should drop to ~1.0-1.5
- Model should generate coherent Braille sequences
- Inference remains 99%+ cheaper than standard LLMs
