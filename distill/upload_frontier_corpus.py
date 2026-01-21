"""
Upload frontier corpus to Modal volume and convert to training format.
"""

import modal
import json
from pathlib import Path

app = modal.App("braille-upload-frontier")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch>=2.0.0")

data_volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": data_volume}, timeout=600)
def convert_and_save(train_samples: list, eval_variants: list, eval_adversarial: list):
    """Convert corpus to training format and save to Modal volume."""
    import torch
    
    def pack_sequences(samples, seq_len=512, pad_id=0):
        all_tokens = []
        for sample in samples:
            tokens = sample['braille_encoding']
            all_tokens.extend(tokens)
            all_tokens.append(257)  # EOS
        
        sequences = []
        for i in range(0, len(all_tokens) - seq_len, seq_len // 2):
            chunk = all_tokens[i:i + seq_len]
            if len(chunk) < seq_len:
                chunk = chunk + [pad_id] * (seq_len - len(chunk))
            sequences.append(chunk)
        
        return torch.tensor(sequences, dtype=torch.long)
    
    def create_training_data(input_ids):
        return {
            'input_ids': input_ids[:, :-1],
            'target_ids': input_ids[:, 1:],
        }
    
    print("Converting training data...")
    train_seqs = pack_sequences(train_samples, seq_len=512)
    train_data = create_training_data(train_seqs)
    print(f"  Train: {train_data['input_ids'].shape}")
    
    torch.save(train_data, "/data/frontier_train.pt")
    print("Saved /data/frontier_train.pt")
    
    if eval_variants:
        eval_var_seqs = pack_sequences(eval_variants, seq_len=512)
        eval_var_data = create_training_data(eval_var_seqs)
        torch.save(eval_var_data, "/data/frontier_eval_variants.pt")
        print(f"  Eval variants: {eval_var_seqs.shape}")
    
    if eval_adversarial:
        eval_adv_seqs = pack_sequences(eval_adversarial, seq_len=512)
        eval_adv_data = create_training_data(eval_adv_seqs)
        torch.save(eval_adv_data, "/data/frontier_eval_adversarial.pt")
        print(f"  Eval adversarial: {eval_adv_seqs.shape}")
    
    data_volume.commit()
    
    return {
        "train_shape": list(train_data['input_ids'].shape),
        "eval_variants_count": len(eval_variants),
        "eval_adversarial_count": len(eval_adversarial),
    }


def load_jsonl(path: str) -> list:
    samples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


@app.local_entrypoint()
def main():
    corpus_dir = Path('distill/frontier_corpus')
    
    print("Loading frontier corpus locally...")
    train_samples = load_jsonl(corpus_dir / 'train.jsonl')
    eval_variants = load_jsonl(corpus_dir / 'eval_variants.jsonl')
    eval_adversarial = load_jsonl(corpus_dir / 'eval_adversarial.jsonl')
    
    print(f"  Train: {len(train_samples)}")
    print(f"  Eval variants: {len(eval_variants)}")
    print(f"  Eval adversarial: {len(eval_adversarial)}")
    
    # Source distribution
    source_counts = {}
    for s in train_samples:
        cat = s['source_category']
        source_counts[cat] = source_counts.get(cat, 0) + 1
    
    print("\nSource distribution:")
    for cat, count in sorted(source_counts.items()):
        print(f"  {cat}: {count} ({100*count/len(train_samples):.1f}%)")
    
    print("\nUploading to Modal and converting...")
    result = convert_and_save.remote(train_samples, eval_variants, eval_adversarial)
    
    print("\nDone!")
    print(f"  Train shape: {result['train_shape']}")
    print(f"  Eval variants: {result['eval_variants_count']}")
    print(f"  Eval adversarial: {result['eval_adversarial_count']}")
