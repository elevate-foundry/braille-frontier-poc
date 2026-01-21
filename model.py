"""
Minimal Transformer for Braille Frontier Model

A small transformer that operates in Braille-space.
Designed for low compute: small vocab, short sequences, efficient attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from embedding import BrailleInfinityEmbedding, BrailleOutputHead
from tokenizer import VOCAB_SIZE


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, seq_len: int, device: torch.device) -> tuple:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    # x: [B, H, T, D]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # Rotate
    rotated = torch.cat([-x2, x1], dim=-1)
    # Apply rotation
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.repeat(1, 1, 1, 2)  # [1, 1, T, D]
    sin = sin.repeat(1, 1, 1, 2)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Multi-head attention with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, D = x.shape
        
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary(T, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Scaled dot-product attention (uses FlashAttention if available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True if mask is None else False,
        )
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(attn_out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""
    
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
    """Single transformer block with pre-norm."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ff_norm = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class BrailleFrontierModel(nn.Module):
    """
    Minimal transformer that thinks in Braille.
    
    Architecture:
    - Hybrid Braille embedding (geometry + learned)
    - N transformer blocks with RoPE
    - Output head projecting to Braille vocab
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding
        self.embed = BrailleInfinityEmbedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = RMSNorm(d_model)
        self.head = BrailleOutputHead(self.embed)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] token IDs
            mask: Optional attention mask
        
        Returns:
            [batch, seq_len, vocab_size] logits
        """
        x = self.embed(input_ids)
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self(input_ids)
                
                # Get last token logits
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop at EOS
                if next_token.item() == 257:  # EOS_ID
                    break
        
        return input_ids
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    model = BrailleFrontierModel(
        d_model=256,
        n_layers=6,
        n_heads=8,
    )
    
    # Test forward pass
    batch = torch.randint(0, 256, (2, 32))
    logits = model(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Compare to a standard model with 50k vocab
    standard_embed_params = 50000 * 256
    our_embed_params = sum(p.numel() for p in model.embed.parameters())
    print(f"\nEmbedding comparison:")
    print(f"  Standard (50k vocab): {standard_embed_params:,} params")
    print(f"  Braille (259 vocab):  {our_embed_params:,} params")
    print(f"  Reduction: {(1 - our_embed_params/standard_embed_params)*100:.1f}%")
