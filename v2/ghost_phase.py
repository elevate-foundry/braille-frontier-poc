"""
Infinity v2: Ghost Phase (Soft Promotion)

Instead of hard-switching tokens (which causes loss spikes), we gradually
blend from the old representation to the new one over ~1000 steps.

When "ing" is promoted to a single token:
1. For warmup_steps, compute BOTH embeddings
2. Blend: embedding = (1-alpha) * old_3token + alpha * new_1token
3. Alpha goes from 0 → 1 over the warmup period
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
import math


@dataclass
class GhostToken:
    """A token in the ghost phase (being gradually promoted)."""
    token_id: int                    # New token ID
    pattern: Tuple[int, ...]         # Original pattern (e.g., (105, 110, 103) for "ing")
    created_step: int                # Training step when created
    warmup_steps: int = 1000         # Steps to fully transition
    
    def get_alpha(self, current_step: int) -> float:
        """Get blending alpha (0 = all old, 1 = all new)."""
        steps_since_creation = current_step - self.created_step
        if steps_since_creation >= self.warmup_steps:
            return 1.0
        return steps_since_creation / self.warmup_steps


class GhostPhaseEmbedding(nn.Module):
    """
    Embedding layer that supports ghost phase blending.
    
    When a new token is in ghost phase:
    - We compute the mean of component embeddings (old representation)
    - We compute the new token's embedding (new representation)
    - We blend them based on alpha
    
    This prevents the sudden distribution shift that caused 15x loss spikes in v1.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_vocab_size: int = 16384,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_vocab_size = max_vocab_size
        
        # Main embedding table
        self.embedding = nn.Embedding(max_vocab_size, d_model)
        
        # Ghost tokens currently in warmup phase
        self.ghost_tokens: Dict[int, GhostToken] = {}
        
        # Current training step (updated externally)
        self.current_step = 0
    
    def add_ghost_token(
        self,
        pattern: Tuple[int, ...],
        warmup_steps: int = 1000,
    ) -> int:
        """
        Add a new token in ghost phase.
        
        Args:
            pattern: The token IDs this new token replaces
            warmup_steps: Steps to fully transition
            
        Returns:
            The new token ID
        """
        new_id = self.vocab_size
        self.vocab_size += 1
        
        if new_id >= self.max_vocab_size:
            raise ValueError("Vocabulary full")
        
        # Initialize new embedding as mean of components
        with torch.no_grad():
            component_embeds = self.embedding.weight[list(pattern)]
            self.embedding.weight[new_id] = component_embeds.mean(dim=0)
        
        # Register as ghost token
        ghost = GhostToken(
            token_id=new_id,
            pattern=pattern,
            created_step=self.current_step,
            warmup_steps=warmup_steps,
        )
        self.ghost_tokens[new_id] = ghost
        
        return new_id
    
    def forward(
        self,
        token_ids: torch.Tensor,
        component_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get embeddings with ghost phase blending.
        
        Args:
            token_ids: Token IDs [batch, seq_len]
            component_ids: For ghost tokens, the original component IDs
                          [batch, seq_len, max_pattern_len] or None
                          
        Returns:
            Embeddings [batch, seq_len, d_model]
        """
        # Get base embeddings
        embeds = self.embedding(token_ids)
        
        if not self.ghost_tokens:
            return embeds
        
        # Apply ghost phase blending for tokens in warmup
        for token_id, ghost in self.ghost_tokens.items():
            alpha = ghost.get_alpha(self.current_step)
            
            if alpha >= 1.0:
                continue  # Fully transitioned, no blending needed
            
            # Find positions where this ghost token appears
            mask = (token_ids == token_id)
            if not mask.any():
                continue
            
            # Compute old representation (mean of components)
            component_embeds = self.embedding.weight[list(ghost.pattern)]
            old_embed = component_embeds.mean(dim=0)
            
            # Blend: (1-alpha) * old + alpha * new
            new_embed = self.embedding.weight[token_id]
            blended = (1 - alpha) * old_embed + alpha * new_embed
            
            # Apply to masked positions
            embeds = embeds.clone()
            embeds[mask] = blended
        
        return embeds
    
    def cleanup_graduated_ghosts(self):
        """Remove ghost tokens that have fully graduated."""
        graduated = [
            tid for tid, ghost in self.ghost_tokens.items()
            if ghost.get_alpha(self.current_step) >= 1.0
        ]
        for tid in graduated:
            del self.ghost_tokens[tid]


class BPCNormalizedLoss(nn.Module):
    """
    Character-weighted cross-entropy loss for fair BPC computation.
    
    Instead of treating all tokens equally, we weight by the number of
    characters each token represents. This prevents the optimizer from
    "cheating" by preferring longer tokens.
    
    Loss = sum(CE_i * char_len_i) / sum(char_len_i)
    
    This is equivalent to computing loss per character, not per token.
    """
    
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        char_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute BPC-normalized cross-entropy loss.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            targets: Target token IDs [batch, seq_len]
            char_lengths: Character length of each target token [batch, seq_len]
            
        Returns:
            loss: Scalar loss value
            metrics: Dict with 'bpc', 'per_token_loss', 'total_chars'
        """
        B, T, V = logits.shape
        
        # Flatten for cross-entropy
        logits_flat = logits.reshape(-1, V)
        targets_flat = targets.reshape(-1)
        char_lengths_flat = char_lengths.reshape(-1)
        
        # Compute per-token cross-entropy (no reduction)
        ce_per_token = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        # Mask for valid tokens
        valid_mask = (targets_flat != self.ignore_index)
        
        # Weight by character length
        weighted_ce = ce_per_token * char_lengths_flat
        
        # Sum of weighted CE / sum of char lengths = loss per character
        total_weighted_ce = (weighted_ce * valid_mask).sum()
        total_chars = (char_lengths_flat * valid_mask).sum()
        
        # Avoid division by zero
        if total_chars == 0:
            return torch.tensor(0.0, device=logits.device), {}
        
        # Loss normalized by characters
        loss = total_weighted_ce / total_chars
        
        # Compute metrics
        with torch.no_grad():
            # BPC = loss / ln(2)
            bpc = loss.item() / math.log(2)
            
            # Standard per-token loss for comparison
            per_token_loss = ce_per_token[valid_mask].mean().item()
            
            metrics = {
                'bpc': bpc,
                'per_token_loss': per_token_loss,
                'total_chars': total_chars.item(),
                'total_tokens': valid_mask.sum().item(),
                'avg_chars_per_token': total_chars.item() / max(valid_mask.sum().item(), 1),
            }
        
        return loss, metrics


def demo_ghost_phase():
    """Demonstrate ghost phase blending."""
    print("=" * 60)
    print("DEMO: Ghost Phase (Soft Promotion)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    vocab_size = 259  # Base vocab
    d_model = 64
    
    embed = GhostPhaseEmbedding(vocab_size, d_model)
    
    # Simulate training steps
    print("\n--- Adding 'ing' as ghost token at step 0 ---")
    embed.current_step = 0
    
    # "ing" = (105, 110, 103)
    ing_pattern = (105, 110, 103)
    ing_id = embed.add_ghost_token(ing_pattern, warmup_steps=1000)
    print(f"New token ID: {ing_id}")
    
    # Get embeddings at different steps
    test_ids = torch.tensor([[ing_id]])
    
    print("\n--- Blending over time ---")
    for step in [0, 250, 500, 750, 1000, 1500]:
        embed.current_step = step
        alpha = embed.ghost_tokens[ing_id].get_alpha(step)
        
        emb = embed(test_ids)
        
        # Compare to pure old and pure new
        with torch.no_grad():
            old_emb = embed.embedding.weight[list(ing_pattern)].mean(dim=0)
            new_emb = embed.embedding.weight[ing_id]
            
            # Check if blending is correct
            expected = (1 - alpha) * old_emb + alpha * new_emb
            is_correct = torch.allclose(emb[0, 0], expected, atol=1e-5)
        
        print(f"  Step {step:4d}: alpha={alpha:.2f}, blend_correct={is_correct}")
    
    print("\n✓ Ghost phase allows gradual transition without sudden shifts!")


def demo_bpc_loss():
    """Demonstrate BPC-normalized loss."""
    print("\n" + "=" * 60)
    print("DEMO: BPC-Normalized Loss")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    loss_fn = BPCNormalizedLoss()
    
    # Scenario: Same prediction quality, different tokenization
    vocab_size = 300
    
    # Model A: 10 tokens, each 1 character
    logits_a = torch.randn(1, 10, vocab_size)
    targets_a = torch.randint(1, vocab_size, (1, 10))
    char_lens_a = torch.ones(1, 10)  # 1 char per token
    
    # Model B: 5 tokens, each 2 characters (same total chars)
    logits_b = torch.randn(1, 5, vocab_size)
    targets_b = torch.randint(1, vocab_size, (1, 5))
    char_lens_b = torch.ones(1, 5) * 2  # 2 chars per token
    
    loss_a, metrics_a = loss_fn(logits_a, targets_a, char_lens_a)
    loss_b, metrics_b = loss_fn(logits_b, targets_b, char_lens_b)
    
    print("\n--- Scenario: Same text, different tokenization ---")
    print(f"\nModel A (10 tokens × 1 char):")
    print(f"  Per-token loss: {metrics_a['per_token_loss']:.4f}")
    print(f"  BPC:            {metrics_a['bpc']:.4f}")
    print(f"  Total chars:    {metrics_a['total_chars']:.0f}")
    
    print(f"\nModel B (5 tokens × 2 chars):")
    print(f"  Per-token loss: {metrics_b['per_token_loss']:.4f}")
    print(f"  BPC:            {metrics_b['bpc']:.4f}")
    print(f"  Total chars:    {metrics_b['total_chars']:.0f}")
    
    print("\n--- Key Insight ---")
    print("BPC normalizes by characters, making comparison fair.")
    print("Per-token loss would favor Model B (fewer predictions),")
    print("but BPC accounts for the information content per character.")


if __name__ == "__main__":
    demo_ghost_phase()
    demo_bpc_loss()
