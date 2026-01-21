"""
Convert Manus frontier corpus to Braille training format.

The frontier corpus already has braille_encoding, so we just need to:
1. Load the JSONL files
2. Pack sequences into fixed-length chunks
3. Create input_ids/target_ids tensors
4. Save as .pt files for Modal training
"""

import json
import torch
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def pack_sequences(samples: List[Dict], seq_len: int = 512, pad_id: int = 0) -> torch.Tensor:
    """
    Pack braille encodings into fixed-length sequences.
    
    Uses simple concatenation with separator, then chunks into seq_len pieces.
    """
    # Concatenate all braille encodings with separator (EOS = 257)
    all_tokens = []
    for sample in samples:
        tokens = sample['braille_encoding']
        all_tokens.extend(tokens)
        all_tokens.append(257)  # EOS separator
    
    # Chunk into sequences
    sequences = []
    for i in range(0, len(all_tokens) - seq_len, seq_len // 2):  # 50% overlap
        chunk = all_tokens[i:i + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [pad_id] * (seq_len - len(chunk))
        sequences.append(chunk)
    
    return torch.tensor(sequences, dtype=torch.long)


def create_training_data(input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Create input/target pairs for language modeling."""
    # For LM: input is tokens[:-1], target is tokens[1:]
    return {
        'input_ids': input_ids[:, :-1],
        'target_ids': input_ids[:, 1:],
    }


def main():
    corpus_dir = Path('distill/frontier_corpus')
    output_dir = Path('distill/data')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading frontier corpus...")
    
    # Load training data
    train_samples = load_jsonl(corpus_dir / 'train.jsonl')
    print(f"  Train samples: {len(train_samples)}")
    
    # Load evaluation splits
    eval_variants = load_jsonl(corpus_dir / 'eval_variants.jsonl')
    eval_adversarial = load_jsonl(corpus_dir / 'eval_adversarial.jsonl')
    print(f"  Eval variants: {len(eval_variants)}")
    print(f"  Eval adversarial: {len(eval_adversarial)}")
    
    # Pack into sequences
    print("\nPacking sequences (seq_len=512)...")
    train_seqs = pack_sequences(train_samples, seq_len=512)
    print(f"  Train sequences: {train_seqs.shape}")
    
    # Create training data
    train_data = create_training_data(train_seqs)
    print(f"  Input shape: {train_data['input_ids'].shape}")
    print(f"  Target shape: {train_data['target_ids'].shape}")
    
    # Save training data
    output_path = output_dir / 'frontier_train.pt'
    torch.save(train_data, output_path)
    print(f"\nSaved training data to {output_path}")
    
    # Also create eval data
    if eval_variants:
        eval_var_seqs = pack_sequences(eval_variants, seq_len=512)
        eval_var_data = create_training_data(eval_var_seqs)
        torch.save(eval_var_data, output_dir / 'frontier_eval_variants.pt')
        print(f"Saved eval variants: {eval_var_seqs.shape}")
    
    if eval_adversarial:
        eval_adv_seqs = pack_sequences(eval_adversarial, seq_len=512)
        eval_adv_data = create_training_data(eval_adv_seqs)
        torch.save(eval_adv_data, output_dir / 'frontier_eval_adversarial.pt')
        print(f"Saved eval adversarial: {eval_adv_seqs.shape}")
    
    # Print corpus statistics
    print("\n" + "="*50)
    print("CORPUS STATISTICS")
    print("="*50)
    
    # Source distribution
    source_counts = {}
    for s in train_samples:
        cat = s['source_category']
        source_counts[cat] = source_counts.get(cat, 0) + 1
    
    print("\nSource distribution (train):")
    for cat, count in sorted(source_counts.items()):
        print(f"  {cat}: {count} ({100*count/len(train_samples):.1f}%)")
    
    # Structural types
    struct_types = set(s['structural_type'] for s in train_samples)
    print(f"\nUnique structural types: {len(struct_types)}")
    
    # Token statistics
    total_tokens = sum(len(s['braille_encoding']) for s in train_samples)
    avg_len = total_tokens / len(train_samples)
    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Avg sample length: {avg_len:.1f} tokens")


if __name__ == '__main__':
    main()
