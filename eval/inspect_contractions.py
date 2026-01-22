"""
Inspect the learned contractions to understand what's happening.
"""

import modal

app = modal.App("braille-inspect")

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch>=2.0.0")
volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    timeout=300,
    volumes={"/checkpoints": volume},
)
def inspect_contractions():
    import torch
    
    print("Loading Infinity checkpoint...")
    ckpt = torch.load("/checkpoints/infinity_final.pt", map_location="cpu")
    
    print(f"\nCheckpoint keys: {list(ckpt.keys())}")
    print(f"Vocab size: {ckpt.get('vocab_size', 'N/A')}")
    print(f"Compression: {ckpt.get('compression', 'N/A')}")
    
    contractions = ckpt.get("contractions", {})
    print(f"\nNumber of contractions: {len(contractions)}")
    
    if contractions:
        print("\nFirst 20 contractions:")
        for i, (pattern, token_id) in enumerate(list(contractions.items())[:20]):
            # Parse pattern if it's a string
            if isinstance(pattern, str):
                pattern_tuple = tuple(int(x) for x in pattern.strip("()").split(", "))
            else:
                pattern_tuple = pattern
            
            # Convert to characters
            chars = "".join(chr(t) if 32 <= t < 127 else f"[{t}]" for t in pattern_tuple)
            print(f"  {token_id}: {pattern_tuple} -> '{chars}' (len={len(pattern_tuple)})")
        
        # Analyze pattern lengths
        lengths = []
        for pattern in contractions.keys():
            if isinstance(pattern, str):
                pattern_tuple = tuple(int(x) for x in pattern.strip("()").split(", "))
            else:
                pattern_tuple = pattern
            lengths.append(len(pattern_tuple))
        
        print(f"\nPattern length distribution:")
        for length in sorted(set(lengths)):
            count = lengths.count(length)
            print(f"  Length {length}: {count} patterns")
        
        print(f"\nAverage pattern length: {sum(lengths)/len(lengths):.2f}")
        print(f"Total compression potential: {sum(lengths) - len(lengths)} chars saved per occurrence")


@app.local_entrypoint()
def main():
    inspect_contractions.remote()
