"""
Prepare raw text data for BPE baseline training.
Extracts text from frontier corpus JSONL and uploads to Modal volume.
"""

import modal
import json
from pathlib import Path

app = modal.App("braille-prepare-bpe")

data_volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)


@app.function(volumes={"/data": data_volume}, timeout=300)
def save_texts(train_texts: list, variant_texts: list, adversarial_texts: list):
    """Save text lists to Modal volume."""
    import json
    
    with open("/data/frontier_train_texts.json", "w") as f:
        json.dump(train_texts, f)
    print(f"Saved {len(train_texts)} train texts")
    
    with open("/data/frontier_variant_texts.json", "w") as f:
        json.dump(variant_texts, f)
    print(f"Saved {len(variant_texts)} variant texts")
    
    with open("/data/frontier_adversarial_texts.json", "w") as f:
        json.dump(adversarial_texts, f)
    print(f"Saved {len(adversarial_texts)} adversarial texts")
    
    data_volume.commit()
    return {"train": len(train_texts), "variant": len(variant_texts), "adversarial": len(adversarial_texts)}


def load_jsonl(path: str) -> list:
    texts = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                texts.append(sample['text'])
    return texts


@app.local_entrypoint()
def main():
    corpus_dir = Path('distill/frontier_corpus')
    
    print("Loading frontier corpus texts...")
    train_texts = load_jsonl(corpus_dir / 'train.jsonl')
    variant_texts = load_jsonl(corpus_dir / 'eval_variants.jsonl')
    adversarial_texts = load_jsonl(corpus_dir / 'eval_adversarial.jsonl')
    
    print(f"  Train: {len(train_texts)} texts, {sum(len(t) for t in train_texts):,} chars")
    print(f"  Variants: {len(variant_texts)} texts")
    print(f"  Adversarial: {len(adversarial_texts)} texts")
    
    print("\nUploading to Modal...")
    result = save_texts.remote(train_texts, variant_texts, adversarial_texts)
    print(f"Done: {result}")
