"""
Infinity Model - Transformer with 3-Layer Token System

Features:
- Dynamic vocabulary expansion during training
- Shadow token mining for automatic contraction discovery
- Adaptive context window based on token density
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from .layers import InfinityVocabulary, InfinityEmbedding, ShadowTokenMiner


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ff_norm = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class InfinityModel(nn.Module):
    """
    Transformer with 3-Layer Infinity Token System.
    
    Supports:
    - Dynamic vocabulary expansion
    - Shadow token mining
    - Adaptive compression
    """
    
    def __init__(
        self,
        vocab: InfinityVocabulary,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        
        # Infinity embedding
        self.embed = InfinityEmbedding(vocab, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, None, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab.max_vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embed.residual.weight
        
        # Shadow token miner
        self.miner = ShadowTokenMiner(vocab)
        
        # Training stats
        self.steps = 0
        self.promotions = []
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.head(x)
    
    def training_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        mine_tokens: bool = True,
    ) -> Dict[str, float]:
        """
        Single training step with optional token mining.
        
        Returns dict with loss and stats.
        """
        # Forward pass
        logits = self(input_ids)
        
        # Loss (only on valid vocab range)
        loss = F.cross_entropy(
            logits[:, :, :self.vocab.current_size].reshape(-1, self.vocab.current_size),
            target_ids.reshape(-1),
            ignore_index=0,  # PAD
        )
        
        # Mine tokens
        if mine_tokens and self.training:
            self.miner.mine(input_ids)
        
        self.steps += 1
        
        return {
            "loss": loss,
            "vocab_size": self.vocab.current_size,
            "layer1_gate": self.embed.get_layer_stats()["layer1_gate"],
        }
    
    def maybe_expand_vocabulary(
        self,
        k: int = 50,
        min_steps: int = 5000,
    ) -> List[int]:
        """
        Check if we should expand vocabulary and do so.
        
        Returns list of newly promoted token IDs.
        """
        if not self.miner.should_promote():
            return []
        
        if self.steps < min_steps:
            return []
        
        # Promote top-k candidates
        new_ids = self.miner.promote_top_k(k)
        
        # Expand embeddings
        for new_id in new_ids:
            info = self.vocab.token_info[new_id]
            self.embed.expand_vocabulary(new_id, info.components)
        
        self.promotions.append({
            "step": self.steps,
            "new_tokens": len(new_ids),
            "vocab_size": self.vocab.current_size,
        })
        
        # Rebuild mask table
        self.embed._build_mask_table()
        
        return new_ids
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get statistics about compression efficiency."""
        layer2_count = sum(1 for t in self.vocab.token_info.values() if t.layer == 2)
        layer3_count = sum(1 for t in self.vocab.token_info.values() if t.layer == 3)
        
        avg_compression = 1.0
        if layer2_count > 0:
            compressions = [t.compression_ratio for t in self.vocab.token_info.values() if t.layer == 2]
            avg_compression = sum(compressions) / len(compressions)
        
        return {
            "layer1_tokens": 256,
            "layer2_tokens": layer2_count,
            "layer3_tokens": layer3_count,
            "total_vocab": self.vocab.current_size,
            "avg_compression": avg_compression,
        }
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids)
                logits = logits[:, -1, :self.vocab.current_size] / temperature
                
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if next_token.item() == self.vocab.EOS_ID:
                    break
        
        return input_ids
