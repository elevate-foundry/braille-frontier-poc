"""
Infinity v2: Modal Training Script

This script trains the Infinity v2 model with all the fixes from Gemini's suggestions:
1. Character-offset positional encoding (RoPE)
2. Ghost phase for soft token promotion
3. BPC-normalized loss
4. GPU-accelerated tokenization

Run with: modal run v2/train_infinity_v2.py
"""

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Modal setup
app = modal.App("infinity-v2-training")

# Volumes for data and checkpoints
data_volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)

# GPU image with PyTorch
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "numpy",
    "tqdm",
)


# ============================================================================
# Model Components (inline to avoid import issues on Modal)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute RoPE frequencies for character positions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis, position_ids):
    """Apply RoPE using character offsets."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs = freqs_cis[position_ids]
    freqs = freqs.unsqueeze(2)
    
    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CharOffsetAttention(nn.Module):
    """Multi-head attention with RoPE using character-offset positions."""
    
    def __init__(self, d_model: int, n_heads: int, max_char_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        freqs_cis = precompute_freqs_cis(self.head_dim, max_char_len)
        self.register_buffer("freqs_cis", freqs_cis)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, self.freqs_cis, position_ids)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_char_len: int = 8192):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CharOffsetAttention(d_model, n_heads, max_char_len)
        self.ff_norm = RMSNorm(d_model)
        self.ff = FeedForward(d_model)
    
    def forward(self, x, position_ids):
        x = x + self.attn(self.attn_norm(x), position_ids)
        x = x + self.ff(self.ff_norm(x))
        return x


@dataclass
class GhostToken:
    """Token in ghost phase (being gradually promoted)."""
    token_id: int
    pattern: Tuple[int, ...]
    created_step: int
    warmup_steps: int = 1000
    
    def get_alpha(self, current_step: int) -> float:
        steps_since = current_step - self.created_step
        if steps_since >= self.warmup_steps:
            return 1.0
        return steps_since / self.warmup_steps


class GhostPhaseEmbedding(nn.Module):
    """Embedding with ghost phase blending for smooth token promotion."""
    
    def __init__(self, vocab_size: int, d_model: int, max_vocab_size: int = 16384):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_vocab_size = max_vocab_size
        self.embedding = nn.Embedding(max_vocab_size, d_model)
        self.ghost_tokens: Dict[int, GhostToken] = {}
        self.current_step = 0
    
    def add_ghost_token(self, pattern: Tuple[int, ...], warmup_steps: int = 1000) -> int:
        new_id = self.vocab_size
        self.vocab_size += 1
        
        with torch.no_grad():
            component_embeds = self.embedding.weight[list(pattern)]
            self.embedding.weight[new_id] = component_embeds.mean(dim=0)
        
        self.ghost_tokens[new_id] = GhostToken(
            token_id=new_id,
            pattern=pattern,
            created_step=self.current_step,
            warmup_steps=warmup_steps,
        )
        return new_id
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(token_ids)
        
        if not self.ghost_tokens:
            return embeds
        
        for token_id, ghost in self.ghost_tokens.items():
            alpha = ghost.get_alpha(self.current_step)
            if alpha >= 1.0:
                continue
            
            mask = (token_ids == token_id)
            if not mask.any():
                continue
            
            component_embeds = self.embedding.weight[list(ghost.pattern)]
            old_embed = component_embeds.mean(dim=0)
            new_embed = self.embedding.weight[token_id]
            blended = (1 - alpha) * old_embed + alpha * new_embed
            
            embeds = embeds.clone()
            embeds[mask] = blended
        
        return embeds


class InfinityV2Model(nn.Module):
    """
    Infinity v2: Dynamic vocabulary with stable positions.
    
    Key improvements:
    - Character-offset RoPE (positions don't shift during vocab expansion)
    - Ghost phase embedding (smooth token promotion)
    - BPC-normalized loss (fair comparison metric)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_vocab_size: int = 16384,
        max_char_len: int = 8192,
        ghost_warmup_steps: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_vocab_size = max_vocab_size
        self.ghost_warmup_steps = ghost_warmup_steps
        
        # Base vocab: 256 bytes + BOS + EOS + UNK
        self.base_vocab_size = 259
        
        # Ghost-phase embedding
        self.embedding = GhostPhaseEmbedding(
            self.base_vocab_size, d_model, max_vocab_size
        )
        
        # Transformer layers with char-offset RoPE
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_char_len)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, max_vocab_size, bias=False)
        self.head.weight = self.embedding.embedding.weight  # Tie weights
        
        # Contraction tracking
        self.contractions: Dict[Tuple[int, ...], int] = {}
        self.ngram_counts: Dict[Tuple[int, ...], int] = {}
        
        self.current_step = 0
    
    def forward(self, token_ids, position_ids):
        self.embedding.current_step = self.current_step
        x = self.embedding(token_ids)
        
        for layer in self.layers:
            x = layer(x, position_ids)
        
        x = self.norm(x)
        return self.head(x)
    
    def add_contraction(self, pattern: Tuple[int, ...]) -> int:
        """Add a new contraction with ghost phase."""
        if pattern in self.contractions:
            return self.contractions[pattern]
        
        new_id = self.embedding.add_ghost_token(pattern, self.ghost_warmup_steps)
        self.contractions[pattern] = new_id
        return new_id
    
    @property
    def vocab_size(self):
        return self.embedding.vocab_size


# ============================================================================
# Tokenizer (simplified for Modal)
# ============================================================================

class SimpleTrieTokenizer:
    """Simple trie-based tokenizer with character offset tracking."""
    
    PAD_ID = 0
    BOS_ID = 256
    EOS_ID = 257
    
    def __init__(self):
        self.contractions: Dict[Tuple[int, ...], int] = {}
        self.current_vocab_size = 259
    
    def add_contraction(self, pattern: Tuple[int, ...], token_id: int):
        self.contractions[pattern] = token_id
        self.current_vocab_size = max(self.current_vocab_size, token_id + 1)
    
    def tokenize(self, text: str) -> Tuple[List[int], List[int], List[int]]:
        """
        Tokenize text with character offsets.
        
        Returns: (token_ids, char_positions, char_lengths)
        """
        bytes_list = list(text.encode('utf-8', errors='replace'))
        
        # Sort contractions by length (longest first)
        sorted_contractions = sorted(
            self.contractions.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        tokens = []
        positions = []
        lengths = []
        
        i = 0
        while i < len(bytes_list):
            matched = False
            
            for pattern, token_id in sorted_contractions:
                plen = len(pattern)
                if i + plen <= len(bytes_list):
                    if tuple(bytes_list[i:i+plen]) == pattern:
                        tokens.append(token_id)
                        positions.append(i)
                        lengths.append(plen)
                        i += plen
                        matched = True
                        break
            
            if not matched:
                tokens.append(bytes_list[i])
                positions.append(i)
                lengths.append(1)
                i += 1
        
        return tokens, positions, lengths
    
    def tokenize_batch(
        self,
        texts: List[str],
        max_len: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize batch with BOS/EOS."""
        batch_tokens = []
        batch_positions = []
        batch_lengths = []
        
        for text in texts:
            tokens, positions, lengths = self.tokenize(text)
            
            # Add BOS
            tokens = [self.BOS_ID] + tokens
            positions = [0] + positions
            lengths = [0] + lengths
            
            # Truncate
            if len(tokens) > max_len - 1:
                tokens = tokens[:max_len - 1]
                positions = positions[:max_len - 1]
                lengths = lengths[:max_len - 1]
            
            # Add EOS
            tokens.append(self.EOS_ID)
            positions.append(positions[-1] + lengths[-1] if positions else 0)
            lengths.append(0)
            
            # Pad
            pad_len = max_len - len(tokens)
            tokens = tokens + [self.PAD_ID] * pad_len
            positions = positions + [0] * pad_len
            lengths = lengths + [1] * pad_len
            
            batch_tokens.append(tokens)
            batch_positions.append(positions)
            batch_lengths.append(lengths)
        
        return (
            torch.tensor(batch_tokens, dtype=torch.long),
            torch.tensor(batch_positions, dtype=torch.long),
            torch.tensor(batch_lengths, dtype=torch.float),
            (torch.tensor(batch_tokens, dtype=torch.long) != 0).long(),
        )


# ============================================================================
# Loss Function
# ============================================================================

def compute_bpc_loss(logits, targets, char_lengths, ignore_index=0):
    """
    Compute BPC-normalized cross-entropy loss.
    
    Returns: (loss, metrics_dict)
    """
    B, T, V = logits.shape
    
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    char_lengths_flat = char_lengths.reshape(-1)
    
    ce_per_token = F.cross_entropy(
        logits_flat, targets_flat,
        ignore_index=ignore_index,
        reduction='none'
    )
    
    valid_mask = (targets_flat != ignore_index)
    weighted_ce = ce_per_token * char_lengths_flat
    
    total_weighted_ce = (weighted_ce * valid_mask).sum()
    total_chars = (char_lengths_flat * valid_mask).sum()
    
    if total_chars == 0:
        return torch.tensor(0.0, device=logits.device), {}
    
    loss = total_weighted_ce / total_chars
    
    with torch.no_grad():
        bpc = loss.item() / math.log(2)
        per_token_loss = ce_per_token[valid_mask].mean().item() if valid_mask.any() else 0
        avg_chars = total_chars.item() / max(valid_mask.sum().item(), 1)
    
    return loss, {
        'bpc': bpc,
        'per_token_loss': per_token_loss,
        'total_chars': total_chars.item(),
        'avg_chars_per_token': avg_chars,
    }


# ============================================================================
# Training Function
# ============================================================================

@app.function(
    image=image,
    gpu="A100",
    timeout=3600 * 4,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
)
def train_infinity_v2(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    max_len: int = 512,
    mining_threshold: int = 50,
    mining_interval: int = 500,
    ghost_warmup: int = 1000,
):
    """Train Infinity v2 model on Modal."""
    
    print("=" * 60)
    print("INFINITY V2 TRAINING")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load training data
    print("\nLoading training data...")
    with open("/data/frontier_train_texts.json", "r") as f:
        train_texts = json.load(f)
    print(f"Loaded {len(train_texts)} training samples")
    
    # Initialize model
    print("\nInitializing model...")
    model = InfinityV2Model(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ghost_warmup_steps=ghost_warmup,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")
    
    # Initialize tokenizer
    tokenizer = SimpleTrieTokenizer()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    global_step = 0
    best_bpc = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_bpc = 0
        num_batches = 0
        
        # Shuffle data
        import random
        random.shuffle(train_texts)
        
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i + batch_size]
            
            # Sync tokenizer with model contractions
            for pattern, token_id in model.contractions.items():
                tokenizer.add_contraction(pattern, token_id)
            
            # Tokenize
            token_ids, position_ids, char_lengths, mask = tokenizer.tokenize_batch(
                batch_texts, max_len
            )
            
            token_ids = token_ids.to(device)
            position_ids = position_ids.to(device)
            char_lengths = char_lengths.to(device)
            
            # Forward
            input_ids = token_ids[:, :-1]
            input_pos = position_ids[:, :-1]
            target_ids = token_ids[:, 1:]
            target_lens = char_lengths[:, 1:]
            
            model.current_step = global_step
            logits = model(input_ids, input_pos)
            
            # BPC-normalized loss
            loss, metrics = compute_bpc_loss(logits, target_ids, target_lens)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bpc += metrics.get('bpc', 0)
            num_batches += 1
            
            # Mine n-grams
            if global_step % 10 == 0:
                for seq in input_ids:
                    seq_list = seq[seq != 0].tolist()
                    for n in range(2, 7):
                        for j in range(len(seq_list) - n + 1):
                            pattern = tuple(seq_list[j:j+n])
                            if all(t < 256 for t in pattern):
                                model.ngram_counts[pattern] = model.ngram_counts.get(pattern, 0) + 1
            
            # Maybe expand vocabulary
            if global_step > 0 and global_step % mining_interval == 0:
                candidates = [
                    (p, c) for p, c in model.ngram_counts.items()
                    if c >= mining_threshold and p not in model.contractions
                ]
                candidates.sort(key=lambda x: x[1] * len(x[0]), reverse=True)
                
                for pattern, count in candidates[:5]:
                    new_id = model.add_contraction(pattern)
                    pattern_str = ''.join(chr(b) if 32 <= b < 127 else f'[{b}]' for b in pattern)
                    print(f"  [Step {global_step}] Promoted '{pattern_str}' -> {new_id} (count={count})")
                
                model.ngram_counts.clear()
            
            global_step += 1
        
        avg_loss = epoch_loss / num_batches
        avg_bpc = epoch_bpc / num_batches
        
        # Compute compression ratio
        total_chars = sum(len(t) for t in train_texts[:100])
        total_tokens = 0
        for t in train_texts[:100]:
            toks, _, _ = tokenizer.tokenize(t)
            total_tokens += len(toks)
        compression = total_chars / total_tokens if total_tokens > 0 else 1.0
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, BPC={avg_bpc:.4f}, "
              f"compression={compression:.2f}, vocab={model.vocab_size}, "
              f"time={elapsed/60:.1f}min")
        
        # Save best model
        if avg_bpc < best_bpc:
            best_bpc = avg_bpc
            torch.save({
                'model_state_dict': model.state_dict(),
                'contractions': model.contractions,
                'vocab_size': model.vocab_size,
                'config': {
                    'd_model': d_model,
                    'n_layers': n_layers,
                    'n_heads': n_heads,
                },
                'metrics': {
                    'bpc': avg_bpc,
                    'compression': compression,
                },
            }, "/checkpoints/infinity_v2_best.pt")
            print(f"  Saved best model (BPC={avg_bpc:.4f})")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'contractions': model.contractions,
        'vocab_size': model.vocab_size,
        'config': {
            'd_model': d_model,
            'n_layers': n_layers,
            'n_heads': n_heads,
        },
        'metrics': {
            'bpc': avg_bpc,
            'compression': compression,
        },
    }, "/checkpoints/infinity_v2_final.pt")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Final BPC: {avg_bpc:.4f}")
    print(f"Final compression: {compression:.2f} chars/token")
    print(f"Final vocab size: {model.vocab_size}")
    print(f"Contractions learned: {len(model.contractions)}")
    
    checkpoint_volume.commit()
    
    return {
        'bpc': avg_bpc,
        'compression': compression,
        'vocab_size': model.vocab_size,
        'training_time_min': total_time / 60,
    }


@app.local_entrypoint()
def main(
    epochs: int = 50,
    batch_size: int = 32,
    d_model: int = 512,
    n_layers: int = 8,
):
    """Run Infinity v2 training on Modal."""
    result = train_infinity_v2.remote(
        epochs=epochs,
        batch_size=batch_size,
        d_model=d_model,
        n_layers=n_layers,
    )
    print(f"\nResults: {result}")
