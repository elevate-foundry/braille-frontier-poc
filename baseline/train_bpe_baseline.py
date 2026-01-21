"""
BPE Baseline Training Script

Train the same transformer architecture with standard BPE tokenization
for head-to-head comparison with Infinity model.

Comparison metrics:
- Final loss on train/variant/adversarial splits
- Tokens per character (compression)
- Training time
"""

import modal

app = modal.App("braille-bpe-baseline")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
    "tokenizers>=0.15.0",
)

volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=14400,
    volumes={
        "/checkpoints": volume,
        "/data": data_volume,
    },
)
def train_bpe_baseline(epochs: int = 50, vocab_size: int = 1000):
    """Train BPE baseline model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    import time
    import json
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    
    device = torch.device("cuda")
    print("=" * 60)
    print("BPE BASELINE TRAINING")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Target vocab size: {vocab_size}")
    
    # =========================================================================
    # LOAD RAW TEXT DATA
    # =========================================================================
    
    print("\nLoading raw text from frontier corpus...")
    
    # Load the JSONL files to get raw text
    train_texts = []
    with open("/data/frontier_train_texts.json", "r") as f:
        train_texts = json.load(f)
    
    print(f"  Loaded {len(train_texts)} training texts")
    total_chars = sum(len(t) for t in train_texts)
    print(f"  Total characters: {total_chars:,}")
    
    # =========================================================================
    # TRAIN BPE TOKENIZER
    # =========================================================================
    
    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
    )
    
    tokenizer.train_from_iterator(train_texts, trainer=trainer)
    
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"  Actual vocab size: {actual_vocab_size}")
    
    # Save tokenizer
    tokenizer.save("/checkpoints/bpe_tokenizer.json")
    
    # =========================================================================
    # TOKENIZE DATA
    # =========================================================================
    
    print("\nTokenizing data...")
    
    def tokenize_texts(texts, max_len=512):
        all_ids = []
        for text in texts:
            encoded = tokenizer.encode(text)
            ids = encoded.ids
            # Add EOS
            ids = ids + [2]  # <eos> = 2
            all_ids.extend(ids)
        
        # Chunk into sequences
        sequences = []
        for i in range(0, len(all_ids) - max_len, max_len // 2):
            chunk = all_ids[i:i + max_len]
            if len(chunk) < max_len:
                chunk = chunk + [0] * (max_len - len(chunk))
            sequences.append(chunk)
        
        return torch.tensor(sequences, dtype=torch.long)
    
    train_ids = tokenize_texts(train_texts)
    print(f"  Train sequences: {train_ids.shape}")
    
    # Compression ratio
    total_tokens = (train_ids != 0).sum().item()
    bpe_compression = total_chars / total_tokens
    print(f"  BPE compression: {bpe_compression:.2f} chars/token")
    
    # Create input/target pairs (keep on CPU for DataLoader, move to GPU in training loop)
    input_ids = train_ids[:, :-1]
    target_ids = train_ids[:, 1:]
    
    # =========================================================================
    # MODEL DEFINITION (same architecture as Infinity, but standard embedding)
    # =========================================================================
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    
    class Attention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
        def forward(self, x):
            B, T, D = x.shape
            q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))
    
    class FeedForward(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            d_ff = d_model * 4
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.attn_norm = RMSNorm(d_model)
            self.attn = Attention(d_model, n_heads)
            self.ff_norm = RMSNorm(d_model)
            self.ff = FeedForward(d_model)
        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ff(self.ff_norm(x))
            return x
    
    class BPEModel(nn.Module):
        def __init__(self, vocab_size, d_model=512, n_layers=12, n_heads=8, dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(self.embedding.weight, std=0.02)
            self.embed_dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embedding.weight  # Weight tying
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.embed_dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.head(x)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    model = BPEModel(actual_vocab_size).to(device)
    model = torch.compile(model)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            logits = model(input_batch)
            loss = F.cross_entropy(
                logits.view(-1, actual_vocab_size),
                target_batch.view(-1),
                ignore_index=0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "vocab_size": actual_vocab_size,
                "compression": bpe_compression,
            }, "/checkpoints/bpe_best.pt")
        
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - total_start
    
    # Save final
    torch.save({
        "model": model.state_dict(),
        "vocab_size": actual_vocab_size,
        "compression": bpe_compression,
    }, "/checkpoints/bpe_final.pt")
    volume.commit()
    
    print("-" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Vocab size: {actual_vocab_size}")
    print(f"BPE compression: {bpe_compression:.2f} chars/token")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    return {
        "best_loss": best_loss,
        "vocab_size": actual_vocab_size,
        "compression": bpe_compression,
        "time": total_time,
    }


@app.local_entrypoint()
def main(epochs: int = 50, vocab_size: int = 1000):
    print("Starting BPE Baseline training on A100...")
    result = train_bpe_baseline.remote(epochs=epochs, vocab_size=vocab_size)
    print(f"\nTraining complete!")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Vocab size: {result['vocab_size']}")
    print(f"BPE compression: {result['compression']:.2f} chars/token")
    print(f"Total time: {result['time']/60:.1f} minutes")
