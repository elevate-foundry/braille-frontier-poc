"""
Convert generated text data to Braille training format.

Reads JSONL files with text and converts them to Braille sequences
suitable for training the Braille Frontier Model.

Usage:
    python convert_to_braille.py --input data/generated.jsonl --output data/braille_train.pt
"""

import json
import argparse
import torch
from pathlib import Path
from typing import List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from tokenizer import text_to_braille_ids, VOCAB_SIZE, PAD_ID, EOS_ID


def load_jsonl(path: Path) -> List[str]:
    """Load text samples from JSONL file."""
    texts = []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            texts.append(record["text"])
    return texts


def create_packed_sequences(
    texts: List[str],
    ctx_len: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack text samples into fixed-length sequences for efficient training.
    
    Returns:
        input_ids: [n_sequences, ctx_len-1]
        target_ids: [n_sequences, ctx_len-1]
    """
    # Convert all texts to Braille IDs
    all_ids = []
    for text in texts:
        ids = text_to_braille_ids(text, add_special=False)
        all_ids.extend(ids)
        all_ids.append(EOS_ID)
    
    print(f"Total Braille tokens: {len(all_ids):,}")
    
    # Pack into fixed-length chunks
    chunks = []
    for i in range(0, len(all_ids) - ctx_len, ctx_len // 2):  # 50% overlap
        chunk = all_ids[i:i + ctx_len]
        if len(chunk) == ctx_len:
            chunks.append(chunk)
    
    print(f"Created {len(chunks):,} packed sequences of length {ctx_len}")
    
    # Convert to tensors
    data = torch.tensor(chunks, dtype=torch.long)
    input_ids = data[:, :-1]
    target_ids = data[:, 1:]
    
    return input_ids, target_ids


def compute_stats(texts: List[str]):
    """Compute and print dataset statistics."""
    total_chars = sum(len(t) for t in texts)
    total_braille = sum(len(text_to_braille_ids(t, add_special=False)) for t in texts)
    
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(texts):,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total Braille tokens: {total_braille:,}")
    print(f"  Avg chars per sample: {total_chars / len(texts):.1f}")
    print(f"  Avg Braille tokens per sample: {total_braille / len(texts):.1f}")
    print(f"  Compression ratio: {total_chars / total_braille:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Convert text to Braille training data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file")
    parser.add_argument("--ctx-len", type=int, default=512, help="Context length")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    print(f"Loading data from {input_path}...")
    texts = load_jsonl(input_path)
    
    compute_stats(texts)
    
    print(f"\nConverting to Braille sequences...")
    input_ids, target_ids = create_packed_sequences(texts, args.ctx_len)
    
    # Save as PyTorch tensors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "input_ids": input_ids,
        "target_ids": target_ids,
        "ctx_len": args.ctx_len,
        "vocab_size": VOCAB_SIZE,
        "n_samples": len(texts),
    }, output_path)
    
    print(f"\nSaved to {output_path}")
    print(f"  Shape: {input_ids.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
