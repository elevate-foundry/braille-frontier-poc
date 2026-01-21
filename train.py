"""
Training script for Braille Frontier Model

Toy training loop to demonstrate the concept.
Uses synthetic data (simple text patterns) for quick iteration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from model import BrailleFrontierModel
from tokenizer import text_to_braille_ids, braille_ids_to_text, VOCAB_SIZE, PAD_ID, EOS_ID

# Training sentences (simple patterns for POC)
TRAINING_DATA = [
    "hello world",
    "the quick brown fox",
    "jumps over the lazy dog",
    "braille is efficient",
    "small vocab big ideas",
    "thinking in dots",
    "compress the language",
    "fewer tokens faster inference",
    "geometry meets semantics",
    "the future is braille",
    "learning to read",
    "patterns in the dots",
    "eight bits of meaning",
    "from text to touch",
    "universal language",
    "simple is beautiful",
    "less is more",
    "dense representations",
    "efficient computing",
    "the model thinks",
]


class BrailleDataset(Dataset):
    """Simple dataset that converts text to Braille sequences."""
    
    def __init__(self, texts: list, max_len: int = 64):
        self.texts = texts
        self.max_len = max_len
        self.data = [text_to_braille_ids(t) for t in texts]
    
    def __len__(self):
        return len(self.data) * 100  # Repeat for more training steps
    
    def __getitem__(self, idx):
        seq = self.data[idx % len(self.data)]
        
        # Pad or truncate
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        else:
            seq = seq + [PAD_ID] * (self.max_len - len(seq))
        
        # Input is all but last, target is all but first (next-token prediction)
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        
        return input_ids, target_ids


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss (ignore padding)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            target_ids.view(-1),
            ignore_index=PAD_ID,
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                target_ids.view(-1),
                ignore_index=PAD_ID,
            )
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_sample(model, prompt: str, device, max_tokens: int = 30):
    """Generate text from a prompt."""
    model.eval()
    
    # Tokenize prompt
    input_ids = torch.tensor([text_to_braille_ids(prompt, add_special=True)], device=device)
    
    # Generate
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.8)
    
    # Decode
    generated = braille_ids_to_text(output_ids[0].tolist())
    return generated


def main():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model config (small for POC)
    config = {
        "d_model": 256,
        "n_layers": 6,
        "n_heads": 8,
        "dropout": 0.1,
    }
    
    # Create model
    model = BrailleFrontierModel(**config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Compare embedding sizes
    standard_embed = 50000 * config["d_model"]
    our_embed = sum(p.numel() for p in model.embed.parameters())
    print(f"\nEmbedding comparison:")
    print(f"  Standard 50k vocab: {standard_embed:,} params")
    print(f"  Braille 259 vocab:  {our_embed:,} params")
    print(f"  Embedding reduction: {(1 - our_embed/standard_embed)*100:.1f}%")
    
    # Dataset
    dataset = BrailleDataset(TRAINING_DATA, max_len=64)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training loop
    n_epochs = 10
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        train_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Generate sample
        if (epoch + 1) % 2 == 0:
            sample = generate_sample(model, "the", device)
            print(f"  Sample: 'the' → '{sample}'")
        
        # Monitor geometry gate
        geom_weight = model.embed.get_geometry_weight()
        print(f"  Geometry weight: {geom_weight:.3f}")
    
    # Save model
    torch.save(model.state_dict(), "braille_frontier_model.pt")
    print("\nModel saved to braille_frontier_model.pt")
    
    # Final generation samples
    print("\n" + "=" * 50)
    print("Generation samples:")
    print("=" * 50)
    
    prompts = ["hello", "the", "braille", "small"]
    for prompt in prompts:
        generated = generate_sample(model, prompt, device)
        print(f"  '{prompt}' → '{generated}'")


if __name__ == "__main__":
    main()
