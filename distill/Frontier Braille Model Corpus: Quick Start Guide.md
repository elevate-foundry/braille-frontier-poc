# Frontier Braille Model Corpus: Quick Start Guide

## Installation

```bash
# Clone or download the repository
cd frontier_braille_corpus

# Install dependencies (none required for basic usage)
# Optional: install for development
pip install -e .
```

## Generate a Corpus

### Basic Usage

```bash
python3 generate_corpus.py --total-samples 10000 --output-dir ./my_corpus
```

This generates:
- `train.jsonl`: Training split (85% of samples)
- `eval_held_out.jsonl`: Held-out evaluation split
- `eval_variants.jsonl`: Surface re-encoding test split
- `eval_adversarial.jsonl`: Adversarial perturbation split
- `corpus_combined.jsonl`: All samples combined
- `corpus_stats.json`: Corpus statistics

### Advanced Options

```bash
# Custom seed for reproducibility
python3 generate_corpus.py --total-samples 50000 --seed 123 --output-dir ./corpus_v2

# Quiet mode (suppress verbose output)
python3 generate_corpus.py --total-samples 5000 --quiet
```

## Explore the Corpus

### View a Sample

```bash
# View first sample from training split
head -1 my_corpus/train.jsonl | python3 -m json.tool
```

### Check Statistics

```bash
cat my_corpus/corpus_stats.json
```

## Evaluate a Corpus

### Generate Evaluation Report

```bash
python3 evaluation.py my_corpus/corpus_combined.jsonl \
  --output-report my_corpus/report.md \
  --output-json my_corpus/report.json
```

This generates:
- `report.md`: Human-readable evaluation report
- `report.json`: Machine-readable evaluation data

## Use in Your Model

### Load JSONL Corpus

```python
import json

def load_corpus(jsonl_path):
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

# Load training data
train_samples = load_corpus('my_corpus/train.jsonl')

# Extract Braille encodings for model input
train_inputs = [sample['braille_encoding'] for sample in train_samples]
```

### Understand Sample Structure

```python
sample = train_samples[0]

# Sample fields:
# - id: unique identifier
# - source_category: 'synthetic_morphology', 'english_stressor', 'multilingual', 'adversarial'
# - structural_type: specific type for evaluation
# - text: raw text (not pre-tokenized)
# - braille_encoding: list of 8-dot Braille cell values (0-255)
# - held_out: whether sample is for evaluation
# - has_variants: whether surface variants exist
# - latent_structure: structural annotation (for evaluation only)
# - metadata: additional information specific to sample type

print(f"Text: {sample['text']}")
print(f"Braille encoding: {sample['braille_encoding']}")
print(f"Structural type: {sample['structural_type']}")
print(f"Metadata: {sample['metadata']}")
```

## Evaluation Protocol

### 1. Macro Masking Test

```python
# After training, identify learned contractions
# Mask them in the evaluation set
# Measure loss increase

# Expected result: Loss increase proportional to contraction frequency
```

### 2. Held-Out Composition Test

```python
# Train on training split (excluding held-out samples)
# Evaluate on held-out samples
# Compare to random baseline

# Expected result: Held-out loss significantly lower than random
```

### 3. Surface Re-Encoding Test

```python
# Evaluate on eval_variants.jsonl
# Compare to original samples

# Expected result: Loss difference is minimal
```

### 4. Ablation Test

```python
# Remove specific morpheme types
# Measure loss impact

# Expected result: Impact correlates with morpheme statistics
```

## Understanding the Corpus

### Synthetic Morphology

- **Concatenative**: prefix + root + suffix (e.g., "un-happy-ness")
- **Agglutinative**: linear stacks (e.g., "talk-ler")
- **Fusional**: merged features (e.g., "habl-o")
- **Templatic**: consonantal root + template (e.g., "ktb" + "CaCaC" → "katab")

### English Stressors

- **Dialogue**: turn-taking with pronoun reference
- **Instruction**: procedural text with conditionals
- **Logical**: if/unless with negation scope
- **Temporal**: non-linear narratives with flashbacks
- **List**: parallel structure and implicit grouping

### Adversarial Perturbations

- **Tokenization**: merge/split words
- **Punctuation**: alter punctuation
- **Entities**: rename entities consistently
- **Phrase order**: reorder phrases

## Troubleshooting

### Issue: No held-out samples generated

This is expected for small corpora. Increase `--total-samples` to ensure held-out samples are generated.

### Issue: Corpus too large

Reduce `--total-samples` or filter the JSONL files:

```bash
# Keep only first 1000 samples
head -1000 my_corpus/train.jsonl > my_corpus/train_small.jsonl
```

### Issue: Reproducibility

Use the same `--seed` value:

```bash
python3 generate_corpus.py --total-samples 10000 --seed 42 --output-dir ./corpus_v1
python3 generate_corpus.py --total-samples 10000 --seed 42 --output-dir ./corpus_v2
# Both should be identical
```

## Next Steps

1. **Generate a corpus**: `python3 generate_corpus.py --total-samples 10000 --output-dir ./corpus`
2. **Explore the data**: `head -1 corpus/train.jsonl | python3 -m json.tool`
3. **Generate evaluation report**: `python3 evaluation.py corpus/corpus_combined.jsonl --output-report corpus/report.md`
4. **Train your model**: Use `corpus/train.jsonl` as input
5. **Evaluate your model**: Use `corpus/eval_*.jsonl` for testing

## Questions?

Refer to:
- `README.md`: Overview and architecture
- `DESIGN.md`: Design philosophy and principles
- `corpus_schema.json`: Detailed metadata schema
