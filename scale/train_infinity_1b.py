"""
1B+ Parameter Infinity Model Training

Scale test to verify if Infinity advantages hold at larger scale.

Model config (~1.3B params):
- d_model: 2048
- n_layers: 24
- n_heads: 16
- FFN: 8192 (4x)

Requires A100 80GB.
"""

import modal

app = modal.App("braille-infinity-1b")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
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
def train_infinity_1b(
    epochs: int = 20,
    d_model: int = 2048,
    n_layers: int = 24,
    n_heads: int = 16,
    batch_size: int = 16,
    grad_accum: int = 4,
    data_path: str = "/data/frontier_train.pt",
):
    """Train 1B+ Infinity model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from typing import Dict, List, Tuple
    from dataclasses import dataclass
    import time
    
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')
    
    print("=" * 60)
    print("INFINITY MODEL - 1B+ SCALE TEST")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Config: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    
    # =========================================================================
    # VOCABULARY AND MODEL CLASSES
    # =========================================================================
    
    @dataclass
    class TokenLayer:
        layer: int
        components: List[int]
        frequency: int
        compression_ratio: float
    
    class InfinityVocabulary:
        LAYER1_END = 256
        LAYER2_END = 4096
        PAD_ID = 0
        BOS_ID = 256
        EOS_ID = 257
        UNK_ID = 258
        
        def __init__(self, max_vocab_size: int = 16384):
            self.max_vocab_size = max_vocab_size
            self.current_size = 259
            self.token_info: Dict[int, TokenLayer] = {}
            for i in range(256):
                self.token_info[i] = TokenLayer(layer=1, components=[i], frequency=0, compression_ratio=1.0)
            self.shadow_tokens: Dict[Tuple[int, ...], int] = {}
            self.contractions: Dict[Tuple[int, ...], int] = {}
        
        def promote_to_layer2(self, pattern: Tuple[int, ...]) -> int:
            if pattern in self.contractions:
                return self.contractions[pattern]
            new_id = self.current_size
            self.current_size += 1
            self.contractions[pattern] = new_id
            self.token_info[new_id] = TokenLayer(
                layer=2, components=list(pattern),
                frequency=0, compression_ratio=len(pattern)
            )
            return new_id
    
    class ShadowTokenMiner:
        def __init__(self, vocab, min_freq: int = 100, max_len: int = 8):
            self.vocab = vocab
            self.min_freq = min_freq
            self.max_len = max_len
            self.ngram_counts: Dict[Tuple[int, ...], int] = {}
        
        def mine(self, batch: torch.Tensor):
            for seq in batch:
                tokens = seq.tolist()
                for n in range(2, self.max_len + 1):
                    for i in range(len(tokens) - n + 1):
                        ngram = tuple(tokens[i:i+n])
                        if 0 not in ngram and all(t < 256 for t in ngram):
                            self.ngram_counts[ngram] = self.ngram_counts.get(ngram, 0) + 1
        
        def should_promote(self) -> bool:
            return any(c >= self.min_freq for c in self.ngram_counts.values())
        
        def promote_top_k(self, k: int = 50) -> List[int]:
            candidates = [(p, c) for p, c in self.ngram_counts.items() 
                         if c >= self.min_freq and p not in self.vocab.contractions]
            candidates.sort(key=lambda x: -x[1] * len(x[0]))
            new_ids = []
            for pattern, _ in candidates[:k]:
                if self.vocab.current_size < self.vocab.max_vocab_size:
                    new_id = self.vocab.promote_to_layer2(pattern)
                    new_ids.append(new_id)
            return new_ids
    
    def build_mask_table(vocab, max_size):
        table = torch.zeros(max_size, 8)
        for i in range(256):
            for bit in range(8):
                table[i, bit] = (i >> bit) & 1
        for pattern, token_id in vocab.contractions.items():
            masks = torch.zeros(8)
            for comp in pattern:
                if comp < 256:
                    for bit in range(8):
                        masks[bit] += (comp >> bit) & 1
            table[token_id] = masks / len(pattern)
        return table
    
    def retokenize_with_contractions(sequences: torch.Tensor, vocab) -> torch.Tensor:
        if not vocab.contractions:
            return sequences
        sorted_contractions = sorted(
            vocab.contractions.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        new_sequences = []
        original_len = sequences.shape[1]
        for seq in sequences:
            tokens = seq.tolist()
            non_pad = [t for t in tokens if t != 0]
            i = 0
            new_tokens = []
            while i < len(non_pad):
                matched = False
                for pattern, token_id in sorted_contractions:
                    plen = len(pattern)
                    if i + plen <= len(non_pad):
                        if tuple(non_pad[i:i+plen]) == pattern:
                            new_tokens.append(token_id)
                            i += plen
                            matched = True
                            break
                if not matched:
                    new_tokens.append(non_pad[i])
                    i += 1
            if len(new_tokens) < original_len:
                new_tokens = new_tokens + [0] * (original_len - len(new_tokens))
            else:
                new_tokens = new_tokens[:original_len]
            new_sequences.append(new_tokens)
        return torch.tensor(new_sequences, dtype=sequences.dtype)
    
    def measure_compression(original: torch.Tensor, compressed: torch.Tensor) -> float:
        orig_len = (original != 0).sum().item()
        comp_len = (compressed != 0).sum().item()
        return orig_len / comp_len if comp_len > 0 else 1.0
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    
    class Attention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.0):
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
        def __init__(self, d_model, d_ff=None, dropout=0.0):
            super().__init__()
            d_ff = d_ff or d_model * 4
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.0):
            super().__init__()
            self.attn_norm = RMSNorm(d_model)
            self.attn = Attention(d_model, n_heads, dropout)
            self.ff_norm = RMSNorm(d_model)
            self.ff = FeedForward(d_model, None, dropout)
        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ff(self.ff_norm(x))
            return x
    
    class InfinityModel(nn.Module):
        def __init__(self, vocab, d_model=2048, n_layers=24, n_heads=16, dropout=0.1):
            super().__init__()
            self.vocab = vocab
            self.d_model = d_model
            
            self.geom_proj = nn.Linear(8, d_model, bias=False)
            self.residual = nn.Embedding(vocab.max_vocab_size, d_model)
            nn.init.normal_(self.residual.weight, std=0.02)
            self.gate = nn.Embedding(vocab.max_vocab_size, 1)
            with torch.no_grad():
                self.gate.weight[:256] = 1.0
                self.gate.weight[256:4096] = 0.5
                self.gate.weight[4096:] = 0.0
            
            self.register_buffer("mask_table", build_mask_table(vocab, vocab.max_vocab_size))
            
            self.embed_dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab.max_vocab_size, bias=False)
            self.head.weight = self.residual.weight
            
            self.miner = ShadowTokenMiner(vocab)
        
        def forward(self, input_ids):
            masks = self.mask_table[input_ids]
            geom = self.geom_proj(masks)
            res = self.residual(input_ids)
            g = torch.sigmoid(self.gate(input_ids))
            x = res + g * geom
            x = self.embed_dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.head(x)
        
        def expand_vocabulary(self, new_token_ids):
            with torch.no_grad():
                for new_id in new_token_ids:
                    if new_id in self.vocab.token_info:
                        components = self.vocab.token_info[new_id].components
                        if len(components) > 0:
                            mean_emb = self.residual.weight[components].mean(dim=0)
                            self.residual.weight[new_id] = mean_emb
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print(f"\nLoading data from {data_path}...")
    data = torch.load(data_path)
    input_ids_original = data["input_ids"]
    target_ids_original = data["target_ids"]
    print(f"  Sequences: {input_ids_original.shape[0]}")
    print(f"  Seq length: {input_ids_original.shape[1]}")
    
    # =========================================================================
    # INITIALIZE MODEL
    # =========================================================================
    
    vocab = InfinityVocabulary()
    model = InfinityModel(vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    model = torch.compile(model)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,} ({n_params/1e9:.2f}B)")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    input_ids = input_ids_original.clone()
    target_ids = target_ids_original.clone()
    current_compression = 1.0
    
    def rebuild_dataloader(inp, tgt, bs):
        ds = TensorDataset(inp, tgt)
        return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4)
    
    dataloader = rebuild_dataloader(input_ids, target_ids, batch_size)
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Batch size: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print("-" * 60)
    
    best_loss = float('inf')
    total_start = time.time()
    global_step = 0
    expand_every = 1000
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Mine n-grams
            model.module.miner.mine(input_batch) if hasattr(model, 'module') else model._orig_mod.miner.mine(input_batch)
            
            logits = model(input_batch)
            loss = F.cross_entropy(
                logits.view(-1, vocab.max_vocab_size),
                target_batch.view(-1),
                ignore_index=0
            )
            loss = loss / grad_accum
            loss.backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Vocabulary expansion
                if global_step >= 500 and global_step % expand_every == 0:
                    miner = model._orig_mod.miner if hasattr(model, '_orig_mod') else model.miner
                    if miner.should_promote():
                        old_size = vocab.current_size
                        new_ids = miner.promote_top_k(k=50)
                        if new_ids and vocab.current_size > old_size:
                            base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                            base_model.expand_vocabulary(new_ids)
                            base_model.mask_table = build_mask_table(vocab, vocab.max_vocab_size).to(device)
                            
                            input_ids = retokenize_with_contractions(input_ids_original, vocab)
                            target_ids = retokenize_with_contractions(target_ids_original, vocab)
                            current_compression = measure_compression(input_ids_original, input_ids)
                            dataloader = rebuild_dataloader(input_ids, target_ids, batch_size)
                            
                            print(f"  [Step {global_step}] Vocab: +{len(new_ids)} -> {vocab.current_size} | Compress: {current_compression:.2f}x")
            
            total_loss += loss.item() * grad_accum
            n_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "vocab_size": vocab.current_size,
                "contractions": {str(k): v for k, v in vocab.contractions.items()},
                "compression": current_compression,
            }, "/checkpoints/infinity_1b_best.pt")
        
        # Get gate values
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        l1_gate = torch.sigmoid(base_model.gate.weight[:256]).mean().item()
        l2_gate = torch.sigmoid(base_model.gate.weight[259:vocab.current_size]).mean().item() if vocab.current_size > 259 else 0
        
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Vocab: {vocab.current_size} | Compress: {current_compression:.2f}x | L1: {l1_gate:.2f} | L2: {l2_gate:.2f} | Mem: {mem_used:.1f}GB | Time: {epoch_time:.1f}s")
    
    total_time = time.time() - total_start
    
    # Save final
    torch.save({
        "model": model.state_dict(),
        "vocab_size": vocab.current_size,
        "contractions": {str(k): v for k, v in vocab.contractions.items()},
        "compression": current_compression,
        "config": {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads},
    }, "/checkpoints/infinity_1b_final.pt")
    volume.commit()
    
    print("-" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final vocab: {vocab.current_size}")
    print(f"Contractions: {len(vocab.contractions)}")
    print(f"Compression: {current_compression:.2f}x")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    return {
        "best_loss": best_loss,
        "final_vocab": vocab.current_size,
        "contractions": len(vocab.contractions),
        "compression": current_compression,
        "time": total_time,
        "params": n_params,
    }


@app.local_entrypoint()
def main(epochs: int = 20):
    print("Starting 1B+ Infinity model training...")
    result = train_infinity_1b.remote(epochs=epochs)
    print(f"\nTraining complete!")
    print(f"Parameters: {result['params']/1e9:.2f}B")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Compression: {result['compression']:.2f}x")
    print(f"Total time: {result['time']/60:.1f} minutes")
