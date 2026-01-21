# Frontier Braille Model Corpus: Evaluation Report

## Executive Summary

- **Total Samples**: 500
- **Source Categories**: 4
- **Structural Types**: 76

## Held-Out Analysis

- **Total Held-Out Samples**: 0
- **Training Samples**: 500
- **Held-Out Ratio**: 0.00%

### Held-Out by Structural Type


## Morphological Coverage

- concatenative: 27 (5.4%)
- agglutinative: 149 (29.8%)
- fusional: 28 (5.6%)
- templatic: 41 (8.2%)

## English Stressor Coverage

- dialogue: 35
- instruction: 16
- logical: 49
- temporal: 70
- list: 42

## Adversarial Perturbation Coverage

- tokenization: 21
- punctuation: 15
- entities: 15
- phrase_order: 24

## Variant Pair Analysis

- **Samples with Variants**: 139
- **Variant Pairs**: 68

## Test Case Identification

- **Macro Masking Candidates**: 125
- **Held-Out Composition Tests**: 0
- **Surface Re-Encoding Tests**: 75
- **Ablation Test Candidates**: 125

## Evaluation Protocol

### 1. Macro Masking Test

**Hypothesis**: If the model learned operators, loss should increase predictably when contractions are masked.

**Procedure**:
1. Train model on corpus
2. Identify learned contractions
3. Mask contractions in test set
4. Measure loss increase

**Success Criteria**: Loss increase is proportional to contraction frequency and generality.

### 2. Held-Out Composition Test

**Hypothesis**: If the model learned structure, loss should be lower than random baseline on unseen feature combinations.

**Procedure**:
1. Train model on corpus (excluding held-out samples)
2. Evaluate on held-out samples
3. Compare to random baseline

**Success Criteria**: Held-out loss is significantly lower than random baseline.

### 3. Surface Re-Encoding Test

**Hypothesis**: If the model learned structure, loss should be stable across surface perturbations.

**Procedure**:
1. Train model on corpus
2. Evaluate on perturbed variants
3. Compare to original samples

**Success Criteria**: Loss difference between original and perturbed is minimal.

### 4. Ablation Test

**Hypothesis**: Impact of removing morpheme types should correlate with morpheme frequency and generality.

**Procedure**:
1. Train model on corpus
2. Remove specific morpheme types
3. Measure loss impact

**Success Criteria**: Impact correlates with morpheme statistics.
