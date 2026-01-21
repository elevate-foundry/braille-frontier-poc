"""
Modal Training Script for Braille Frontier Model

Runs on A100 GPU via Modal (modal.com).
Designed to maximize GPU utilization with sequence packing.

Usage:
    modal run modal_train.py
    
First time setup:
    pip install modal
    modal token new
"""

import modal

# Define the Modal app
app = modal.App("braille-frontier")

# Container image with PyTorch
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
)

# Volume for saving checkpoints
volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",  # Request A100 GPU
    timeout=3600,  # 1 hour max
    volumes={"/checkpoints": volume},
)
def train_on_a100():
    """
    Train the Braille Frontier Model on an A100.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import time
    
    # =========================================================================
    # INLINE DEFINITIONS (Modal needs everything in one file or mounted)
    # =========================================================================
    
    BRAILLE_BASE = 0x2800
    PAD_ID = 0
    BOS_ID = 256
    EOS_ID = 257
    UNK_ID = 258
    VOCAB_SIZE = 259
    
    LETTER_TO_BRAILLE = {
        'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
        'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
        'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
        'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
        'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽',
        'z': '⠵', ' ': '⠀', '.': '⠲', ',': '⠂', '!': '⠖',
        '?': '⠦', "'": '⠄', '-': '⠤', ':': '⠒', ';': '⠆',
    }
    
    def char_to_braille_id(char):
        char = char.lower()
        if char in LETTER_TO_BRAILLE:
            return ord(LETTER_TO_BRAILLE[char]) - BRAILLE_BASE
        return UNK_ID - 256
    
    def text_to_braille_ids(text, add_special=True):
        ids = []
        if add_special:
            ids.append(BOS_ID)
        for char in text:
            ids.append(char_to_braille_id(char))
        if add_special:
            ids.append(EOS_ID)
        return ids
    
    def build_mask_table(vocab_size=VOCAB_SIZE):
        table = torch.zeros(vocab_size, 8, dtype=torch.float32)
        for i in range(256):
            mask = torch.tensor([(i >> bit) & 1 for bit in range(8)], dtype=torch.float32)
            table[i] = mask
        return table
    
    # =========================================================================
    # MODEL DEFINITION
    # =========================================================================
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return x * rms * self.weight
    
    class BrailleInfinityEmbedding(nn.Module):
        def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, gate_init=1.0):
            super().__init__()
            self.d_model = d_model
            self.residual = nn.Embedding(vocab_size, d_model)
            nn.init.normal_(self.residual.weight, std=0.02)
            self.geom_proj = nn.Linear(8, d_model, bias=False)
            self.gate = nn.Embedding(vocab_size, 1)
            nn.init.constant_(self.gate.weight, gate_init)
            self.register_buffer("mask_table", build_mask_table(vocab_size))
        
        def forward(self, token_ids):
            masks = self.mask_table[token_ids]
            geom = self.geom_proj(masks)
            res = self.residual(token_ids)
            g = torch.sigmoid(self.gate(token_ids))
            return res + g * geom
    
    class Attention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.0):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, mask=None):
            B, T, D = x.shape
            q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
            return self.wo(attn_out)
    
    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff=None, dropout=0.0):
            super().__init__()
            d_ff = d_ff or d_model * 4
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
            self.dropout = nn.Dropout(dropout)
        
        def forward(self, x):
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
            super().__init__()
            self.attn_norm = RMSNorm(d_model)
            self.attn = Attention(d_model, n_heads, dropout)
            self.ff_norm = RMSNorm(d_model)
            self.ff = FeedForward(d_model, d_ff, dropout)
        
        def forward(self, x, mask=None):
            x = x + self.attn(self.attn_norm(x), mask)
            x = x + self.ff(self.ff_norm(x))
            return x
    
    class BrailleFrontierModel(nn.Module):
        def __init__(self, vocab_size=VOCAB_SIZE, d_model=512, n_layers=12, n_heads=8, dropout=0.1):
            super().__init__()
            self.embed = BrailleInfinityEmbedding(vocab_size, d_model)
            self.embed_dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, None, dropout)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embed.residual.weight  # Weight tying
        
        def forward(self, input_ids, mask=None):
            x = self.embed(input_ids)
            x = self.embed_dropout(x)
            for layer in self.layers:
                x = layer(x, mask)
            x = self.norm(x)
            return self.head(x)
        
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    # =========================================================================
    # DATASET WITH SEQUENCE PACKING
    # =========================================================================
    
    TRAINING_DATA = [
        "hello world", "the quick brown fox", "jumps over the lazy dog",
        "braille is efficient", "small vocab big ideas", "thinking in dots",
        "compress the language", "fewer tokens faster inference",
        "geometry meets semantics", "the future is braille",
        "learning to read", "patterns in the dots", "eight bits of meaning",
        "from text to touch", "universal language", "simple is beautiful",
        "less is more", "dense representations", "efficient computing",
        "the model thinks", "artificial intelligence", "machine learning",
        "deep neural networks", "transformer architecture",
        "attention is all you need", "language models", "natural language",
        "text generation", "sequence to sequence", "encoder decoder",
    ] * 100  # Repeat for more data
    
    class PackedBrailleDataset(Dataset):
        """Dataset with sequence packing to maximize GPU utilization."""
        
        def __init__(self, texts, ctx_len=512):
            self.ctx_len = ctx_len
            self.eos_id = EOS_ID
            
            # Convert all texts to Braille IDs
            all_ids = []
            for text in texts:
                ids = text_to_braille_ids(text, add_special=False)
                all_ids.extend(ids)
                all_ids.append(self.eos_id)
            
            # Pack into fixed-length chunks
            self.chunks = []
            for i in range(0, len(all_ids) - ctx_len, ctx_len // 2):
                chunk = all_ids[i:i + ctx_len]
                if len(chunk) == ctx_len:
                    self.chunks.append(chunk)
        
        def __len__(self):
            return len(self.chunks)
        
        def __getitem__(self, idx):
            chunk = self.chunks[idx]
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            target_ids = torch.tensor(chunk[1:], dtype=torch.long)
            return input_ids, target_ids
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print("=" * 60)
    print("BRAILLE FRONTIER MODEL - A100 TRAINING")
    print("=" * 60)
    
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Model config (larger for A100)
    config = {
        "d_model": 512,
        "n_layers": 12,
        "n_heads": 8,
        "dropout": 0.1,
    }
    
    model = BrailleFrontierModel(**config).to(device)
    model = torch.compile(model)  # PyTorch 2.0 compilation
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # Compare to standard model
    standard_embed = 50000 * config["d_model"]
    our_embed = VOCAB_SIZE * config["d_model"] + 8 * config["d_model"]
    print(f"Embedding reduction: {(1 - our_embed/standard_embed)*100:.1f}%")
    
    # Dataset
    dataset = PackedBrailleDataset(TRAINING_DATA, ctx_len=512)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    print(f"Dataset size: {len(dataset)} packed sequences")
    print(f"Batch size: 64")
    print(f"Context length: 512")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.01,
        fused=True,  # Fused optimizer for A100
    )
    
    # Mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Training
    n_epochs = 20
    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 60)
    
    total_start = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Mixed precision forward pass
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE),
                    target_ids.view(-1),
                    ignore_index=PAD_ID,
                )
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(dataloader)
        
        print(f"Epoch {epoch+1:2d}/{n_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"/checkpoints/model_epoch_{epoch+1}.pt")
            volume.commit()
    
    total_time = time.time() - total_start
    print("-" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    
    # Save final model
    torch.save(model.state_dict(), "/checkpoints/model_final.pt")
    volume.commit()
    print("Model saved to /checkpoints/model_final.pt")
    
    # GPU utilization stats
    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    return {"loss": avg_loss, "time": total_time}


@app.local_entrypoint()
def main():
    """Entry point for `modal run modal_train.py`"""
    print("Starting Braille Frontier training on A100...")
    result = train_on_a100.remote()
    print(f"\nTraining complete!")
    print(f"Final loss: {result['loss']:.4f}")
    print(f"Total time: {result['time']/60:.1f} minutes")
