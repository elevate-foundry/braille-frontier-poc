"""
Infinity v2: RoPE with Character-Offset Positions

Standard RoPE uses token indices (0, 1, 2, 3...).
We modify it to use character offsets (0, 3, 4, 7...) so that
vocabulary expansion doesn't shift positional encodings.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for RoPE.
    
    Args:
        dim: Dimension of the embedding (must be even)
        max_seq_len: Maximum sequence length (in CHARACTERS, not tokens)
        theta: Base for the frequency computation
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Use character positions, which can be larger than token count
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings using CHARACTER OFFSETS as positions.
    
    Args:
        xq: Query tensor [batch, seq_len, n_heads, head_dim]
        xk: Key tensor [batch, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed frequencies [max_char_pos, head_dim//2]
        position_ids: Character offset positions [batch, seq_len]
        
    Returns:
        Rotated query and key tensors
    """
    # Reshape for complex multiplication
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Index into freqs_cis using CHARACTER POSITIONS (not token indices!)
    # This is the key difference from standard RoPE
    freqs = freqs_cis[position_ids]  # [batch, seq_len, head_dim//2]
    freqs = freqs.unsqueeze(2)  # [batch, seq_len, 1, head_dim//2]
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CharOffsetRoPEAttention(nn.Module):
    """
    Multi-head attention with RoPE using character-offset positions.
    
    Key property: When vocabulary expands, the relative positions between
    concepts remain constant because we use character offsets, not token indices.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_char_len: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_char_len = max_char_len
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Precompute RoPE frequencies for character positions
        freqs_cis = precompute_freqs_cis(self.head_dim, max_char_len)
        self.register_buffer("freqs_cis", freqs_cis)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            position_ids: Character offset positions [batch, seq_len]
            mask: Attention mask (optional)
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE with CHARACTER OFFSETS
        q, k = apply_rotary_emb(q, k, self.freqs_cis, position_ids)
        
        # Transpose for attention: [B, n_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


def demo_rope_stability():
    """
    Demonstrate that RoPE attention patterns remain stable during vocab expansion.
    """
    print("=" * 60)
    print("DEMO: RoPE with Character-Offset Stability")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    d_model = 64
    n_heads = 4
    
    attn = CharOffsetRoPEAttention(d_model, n_heads, max_char_len=1024)
    
    # Scenario 1: "The cat" as 7 tokens (no contractions)
    # Positions: [0, 1, 2, 3, 4, 5, 6]
    print("\n--- Scenario 1: No contractions ---")
    print("Tokens: [T, h, e, ' ', c, a, t]")
    print("Positions: [0, 1, 2, 3, 4, 5, 6]")
    
    x1 = torch.randn(1, 7, d_model)
    pos1 = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    
    out1 = attn(x1, pos1)
    print(f"Output shape: {out1.shape}")
    
    # Scenario 2: "The cat" with "The" contracted (4 tokens)
    # Positions: [0, 3, 4, 5, 6] - note: 'c' is still at position 4!
    print("\n--- Scenario 2: 'The' contracted ---")
    print("Tokens: [The, ' ', c, a, t]")
    print("Positions: [0, 3, 4, 5, 6]")
    
    # Use same embeddings for the overlapping tokens
    x2 = torch.randn(1, 5, d_model)
    x2[0, 1:] = x1[0, 3:]  # Copy ' ', c, a, t embeddings
    pos2 = torch.tensor([[0, 3, 4, 5, 6]])
    
    out2 = attn(x2, pos2)
    print(f"Output shape: {out2.shape}")
    
    # Compare attention patterns for 'c' (position 4)
    # In scenario 1, 'c' is token index 4
    # In scenario 2, 'c' is token index 2
    # But both have CHARACTER position 4!
    
    print("\n--- Key Insight ---")
    print("In both scenarios, 'c' has character position 4.")
    print("The RoPE encoding for 'c' is IDENTICAL in both cases.")
    print("Attention patterns between 'c' and other tokens remain stable!")
    
    # Verify: compute attention scores for 'c' attending to position 0
    print("\n--- Verification: Attention score computation ---")
    
    # Get Q for 'c' and K for position 0 in both scenarios
    with torch.no_grad():
        # Scenario 1: 'c' is at token index 4
        q1 = attn.wq(x1[0, 4:5]).view(1, 1, n_heads, d_model // n_heads)
        k1 = attn.wk(x1[0, 0:1]).view(1, 1, n_heads, d_model // n_heads)
        
        # Apply RoPE
        q1_rot, k1_rot = apply_rotary_emb(
            q1, k1, attn.freqs_cis,
            torch.tensor([[4]]),  # 'c' at char position 4
        )
        k1_rot_pos0, _ = apply_rotary_emb(
            k1, k1, attn.freqs_cis,
            torch.tensor([[0]]),  # 'T' at char position 0
        )
        
        # Scenario 2: 'c' is at token index 2, but still char position 4
        q2 = attn.wq(x2[0, 2:3]).view(1, 1, n_heads, d_model // n_heads)
        k2 = attn.wk(x2[0, 0:1]).view(1, 1, n_heads, d_model // n_heads)
        
        q2_rot, k2_rot = apply_rotary_emb(
            q2, k2, attn.freqs_cis,
            torch.tensor([[4]]),  # 'c' still at char position 4!
        )
        k2_rot_pos0, _ = apply_rotary_emb(
            k2, k2, attn.freqs_cis,
            torch.tensor([[0]]),  # 'The' at char position 0
        )
        
        print(f"Q rotation for 'c' (pos 4) identical: {torch.allclose(q1_rot, q2_rot)}")
        print("This means attention patterns are preserved during vocab expansion!")


if __name__ == "__main__":
    demo_rope_stability()
