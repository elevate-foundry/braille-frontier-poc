"""
1B+ Parameter BPE Baseline Training

Scale test to compare against Infinity at larger scale.

Model config (~1.3B params):
- d_model: 2048
- n_layers: 24
- n_heads: 16
- FFN: 8192 (4x)

Requires A100 80GB.
"""

import modal

app = modal.App("braille-bpe-1b")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
    "tokenizers>=0.15.0",
)

volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=28800,  # 8 hours
    volumes={
        "/checkpoints": volume,
        "/data": data_volume,
    },
)
def train_bpe_1b(
    epochs: int = 20,
    d_model: int = 2048,
    n_layers: int = 24,
    n_heads: int = 16,
    batch_size: int = 16,
    grad_accum: int = 4,
    vocab_size: int = 1000,
):
    """Train 1B+ BPE baseline model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    import time
    import json
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')
    
    print("=" * 60)
    print("BPE BASELINE - 1B+ SCALE TEST")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Config: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    
    # =========================================================================
    # LOAD AND TOKENIZE DATA
    # =========================================================================
    
    print("\nLoading raw text...")
    with open("/data/frontier_train_texts.json", "r") as f:
        train_texts = json.load(f)
    
    print(f"  Loaded {len(train_texts)} texts")
    total_chars = sum(len(t) for t in train_texts)
    print(f"  Total characters: {total_chars:,}")
    
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
    
    tokenizer.save("/checkpoints/bpe_1b_tokenizer.json")
    
    print("\nTokenizing data...")
    def tokenize_texts(texts, max_len=512):
        all_ids = []
        for text in texts:
            encoded = tokenizer.encode(text)
            ids = encoded.ids + [2]  # Add EOS
            all_ids.extend(ids)
        
        sequences = []
        for i in range(0, len(all_ids) - max_len, max_len // 2):
            chunk = all_ids[i:i + max_len]
            if len(chunk) < max_len:
                chunk = chunk + [0] * (max_len - len(chunk))
            sequences.append(chunk)
        return torch.tensor(sequences, dtype=torch.long)
    
    train_ids = tokenize_texts(train_texts)
    print(f"  Train sequences: {train_ids.shape}")
    
    total_tokens = (train_ids != 0).sum().item()
    bpe_compression = total_chars / total_tokens
    print(f"  BPE compression: {bpe_compression:.2f} chars/token")
    
    input_ids = train_ids[:, :-1]
    target_ids = train_ids[:, 1:]
    
    # =========================================================================
    # MODEL DEFINITION
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
        def __init__(self, vocab_size, d_model=2048, n_layers=24, n_heads=16, dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(self.embedding.weight, std=0.02)
            self.embed_dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embedding.weight
        
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
    
    model = BPEModel(actual_vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    model = torch.compile(model)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,} ({n_params/1e9:.2f}B)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print("-" * 60)
    
    best_loss = float('inf')
    total_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            logits = model(input_batch)
            loss = F.cross_entropy(
                logits.view(-1, actual_vocab_size),
                target_batch.view(-1),
                ignore_index=0
            )
            loss = loss / grad_accum
            loss.backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum
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
                "config": {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads},
            }, "/checkpoints/bpe_1b_best.pt")
        
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Mem: {mem_used:.1f}GB | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - total_start
    
    # Save final
    torch.save({
        "model": model.state_dict(),
        "vocab_size": actual_vocab_size,
        "compression": bpe_compression,
        "config": {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads},
    }, "/checkpoints/bpe_1b_final.pt")
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
        "params": n_params,
    }


@app.local_entrypoint()
def main(epochs: int = 20):
    print("Starting 1B+ BPE baseline training...")
    result = train_bpe_1b.remote(epochs=epochs)
    print(f"\nTraining complete!")
    print(f"Parameters: {result['params']/1e9:.2f}B")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"BPE compression: {result['compression']:.2f} chars/token")
    print(f"Total time: {result['time']/60:.1f} minutes")
