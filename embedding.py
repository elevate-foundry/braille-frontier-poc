"""
Braille Infinity Embedding

Hybrid embedding that combines:
1. Geometric prior: Linear projection of 8-bit dot patterns
2. Learned residual: Standard embedding table
3. Gating: Learned scalar to balance geometry vs semantics

E(t) = E_learned(t) + gate(t) * W·bits(t)

This gives the model a "head start" from dot geometry while allowing
semantics to drift away from physical structure when needed.
"""

import torch
import torch.nn as nn
from tokenizer import build_mask_table, VOCAB_SIZE


class BrailleInfinityEmbedding(nn.Module):
    """
    Hybrid embedding with geometric prior + learned residual + gating.
    """
    
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        gate_init: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Learned residual embedding (standard)
        self.residual = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.residual.weight, std=0.02)
        
        # Geometric projection: 8-bit pattern → d_model
        self.geom_proj = nn.Linear(8, d_model, bias=False)
        
        # Per-token gate (scalar) - controls geometry influence
        # Initialize high so geometry dominates early training
        self.gate = nn.Embedding(vocab_size, 1)
        nn.init.constant_(self.gate.weight, gate_init)
        
        # Register mask table as buffer (moves with model, saved in state_dict)
        mask_table = build_mask_table(vocab_size)
        self.register_buffer("mask_table", mask_table, persistent=True)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] tensor of token IDs
        
        Returns:
            [batch, seq_len, d_model] embeddings
        """
        # Get 8-bit masks from lookup table
        masks = self.mask_table[token_ids]  # [B, T, 8]
        
        # Geometric embedding
        geom = self.geom_proj(masks)  # [B, T, d_model]
        
        # Learned residual
        res = self.residual(token_ids)  # [B, T, d_model]
        
        # Gating (sigmoid so it's bounded 0-1)
        g = torch.sigmoid(self.gate(token_ids))  # [B, T, 1]
        
        # Combine: residual + gated geometry
        return res + g * geom
    
    def get_geometry_weight(self) -> float:
        """Return average gate value (for monitoring)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate.weight).mean().item()


class BrailleOutputHead(nn.Module):
    """
    Output head that projects back to Braille vocab.
    
    For the POC, we use weight tying with the embedding.
    """
    
    def __init__(self, embedding: BrailleInfinityEmbedding):
        super().__init__()
        self.embedding = embedding
        # Project from d_model to vocab_size
        # We tie weights with the residual embedding
        self.proj = nn.Linear(embedding.d_model, embedding.vocab_size, bias=False)
        self.proj.weight = embedding.residual.weight  # Weight tying
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [batch, seq_len, d_model]
        
        Returns:
            [batch, seq_len, vocab_size] logits
        """
        return self.proj(hidden)


# Quick test
if __name__ == "__main__":
    embed = BrailleInfinityEmbedding(d_model=256)
    head = BrailleOutputHead(embed)
    
    # Test forward pass
    batch = torch.randint(0, 256, (2, 10))
    emb = embed(batch)
    logits = head(emb)
    
    print(f"Input shape: {batch.shape}")
    print(f"Embedding shape: {emb.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Geometry weight: {embed.get_geometry_weight():.3f}")
    
    # Count parameters
    total = sum(p.numel() for p in embed.parameters())
    print(f"Embedding params: {total:,}")
