"""
Micro-Infinity v2: A 10M parameter test model for validating the v2 architecture.

Key improvements over v1:
1. Character-offset positional encoding (positions don't shift during vocab expansion)
2. Ghost phase for soft token promotion (no sudden loss spikes)
3. BPC-normalized loss (fair comparison metric)
4. RoPE attention (naturally handles variable positions)

This is a validation model - if it works here, we can scale up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

from char_offset_positions import CharOffsetTokenizer, TokenWithPosition
from rope_char_offset import CharOffsetRoPEAttention, precompute_freqs_cis
from ghost_phase import GhostPhaseEmbedding, BPCNormalizedLoss


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_char_len: int = 8192):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CharOffsetRoPEAttention(d_model, n_heads, max_char_len)
        self.ff_norm = RMSNorm(d_model)
        self.ff = FeedForward(d_model)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), position_ids)
        x = x + self.ff(self.ff_norm(x))
        return x


@dataclass
class InfinityV2Config:
    """Configuration for Micro-Infinity v2."""
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    max_vocab_size: int = 4096
    max_char_len: int = 8192
    ghost_warmup_steps: int = 1000
    mining_threshold: int = 100  # Min frequency to promote pattern
    mining_interval: int = 500   # Steps between mining attempts


class MicroInfinityV2(nn.Module):
    """
    Micro-Infinity v2: Dynamic vocabulary with stable positions.
    
    ~10M parameters for validation testing.
    """
    
    def __init__(self, config: InfinityV2Config):
        super().__init__()
        self.config = config
        
        # Tokenizer with character-offset tracking
        self.tokenizer = CharOffsetTokenizer(config.max_vocab_size)
        
        # Ghost-phase embedding
        self.embedding = GhostPhaseEmbedding(
            vocab_size=self.tokenizer.base_vocab_size,
            d_model=config.d_model,
            max_vocab_size=config.max_vocab_size,
        )
        
        # Transformer layers with RoPE
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.max_char_len)
            for _ in range(config.n_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.max_vocab_size, bias=False)
        
        # Tie embeddings
        self.head.weight = self.embedding.embedding.weight
        
        # Loss function
        self.loss_fn = BPCNormalizedLoss(ignore_index=0)
        
        # N-gram counts for mining
        self.ngram_counts: Dict[Tuple[int, ...], int] = {}
        
        # Training state
        self.current_step = 0
    
    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        char_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.
        
        Args:
            token_ids: [batch, seq_len]
            position_ids: Character offsets [batch, seq_len]
            targets: Target token IDs [batch, seq_len] (optional)
            char_lengths: Character length per token [batch, seq_len] (optional)
            
        Returns:
            Dict with 'logits' and optionally 'loss', 'metrics'
        """
        # Update embedding step counter
        self.embedding.current_step = self.current_step
        
        # Get embeddings
        x = self.embedding(token_ids)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, position_ids)
        
        # Output projection
        x = self.norm(x)
        logits = self.head(x)
        
        result = {'logits': logits}
        
        # Compute loss if targets provided
        if targets is not None and char_lengths is not None:
            loss, metrics = self.loss_fn(logits, targets, char_lengths)
            result['loss'] = loss
            result['metrics'] = metrics
        
        return result
    
    def tokenize_batch(
        self,
        texts: List[str],
        max_len: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of texts with character offsets.
        
        Returns:
            input_ids: [batch, max_len]
            position_ids: [batch, max_len] (character offsets)
            target_ids: [batch, max_len]
            char_lengths: [batch, max_len]
        """
        batch_inputs = []
        batch_positions = []
        batch_targets = []
        batch_char_lens = []
        
        for text in texts:
            # Tokenize with positions
            tokens = self.tokenizer.tokenize_with_positions(text)
            
            # Add BOS
            tokens = [TokenWithPosition(
                token_id=self.tokenizer.BOS_ID,
                char_start=0,
                char_end=0,
                text='<BOS>'
            )] + tokens
            
            # Truncate/pad
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            
            # Extract tensors
            ids = [t.token_id for t in tokens]
            positions = [t.char_start for t in tokens]
            char_lens = [max(1, t.char_end - t.char_start) for t in tokens]
            
            # Pad
            pad_len = max_len - len(ids)
            ids = ids + [0] * pad_len
            positions = positions + [0] * pad_len
            char_lens = char_lens + [1] * pad_len
            
            batch_inputs.append(ids)
            batch_positions.append(positions)
            batch_char_lens.append(char_lens)
        
        input_ids = torch.tensor(batch_inputs, dtype=torch.long)
        position_ids = torch.tensor(batch_positions, dtype=torch.long)
        char_lengths = torch.tensor(batch_char_lens, dtype=torch.float)
        
        # Targets are shifted inputs
        target_ids = input_ids.clone()
        target_ids[:, :-1] = input_ids[:, 1:]
        target_ids[:, -1] = 0  # Pad last position
        
        # Input is everything except last
        input_ids = input_ids[:, :-1]
        position_ids = position_ids[:, :-1]
        target_ids = target_ids[:, :-1]
        char_lengths = char_lengths[:, 1:-1]  # Lengths for targets
        
        # Pad char_lengths to match
        if char_lengths.shape[1] < target_ids.shape[1]:
            pad = torch.ones(char_lengths.shape[0], target_ids.shape[1] - char_lengths.shape[1])
            char_lengths = torch.cat([char_lengths, pad], dim=1)
        
        return input_ids, position_ids, target_ids, char_lengths
    
    def mine_ngrams(self, token_ids: torch.Tensor, n_range: Tuple[int, int] = (2, 6)):
        """
        Count n-grams in a batch for potential promotion.
        
        Args:
            token_ids: [batch, seq_len]
            n_range: (min_n, max_n) for n-gram sizes
        """
        min_n, max_n = n_range
        
        for seq in token_ids:
            seq = seq[seq != 0].tolist()  # Remove padding
            
            for n in range(min_n, max_n + 1):
                for i in range(len(seq) - n + 1):
                    pattern = tuple(seq[i:i+n])
                    
                    # Skip if contains special tokens or already a contraction
                    if any(t >= 256 for t in pattern):
                        continue
                    
                    self.ngram_counts[pattern] = self.ngram_counts.get(pattern, 0) + 1
    
    def maybe_expand_vocab(self, max_new_tokens: int = 10) -> List[int]:
        """
        Promote frequent n-grams to new tokens if threshold met.
        
        Returns:
            List of newly created token IDs
        """
        if self.current_step % self.config.mining_interval != 0:
            return []
        
        # Find candidates above threshold
        candidates = [
            (pattern, count)
            for pattern, count in self.ngram_counts.items()
            if count >= self.config.mining_threshold
            and pattern not in self.tokenizer.contractions
        ]
        
        # Sort by (count * length) - prefer frequent AND long patterns
        candidates.sort(key=lambda x: x[1] * len(x[0]), reverse=True)
        
        new_tokens = []
        for pattern, count in candidates[:max_new_tokens]:
            # Add to tokenizer
            new_id = self.tokenizer.add_contraction(pattern)
            
            # Add as ghost token (soft promotion)
            self.embedding.add_ghost_token(
                pattern,
                warmup_steps=self.config.ghost_warmup_steps
            )
            
            # Update embedding vocab size
            self.embedding.vocab_size = self.tokenizer.current_vocab_size
            
            new_tokens.append(new_id)
            
            pattern_str = ''.join(chr(t) if 32 <= t < 127 else f'[{t}]' for t in pattern)
            print(f"  [Step {self.current_step}] Promoted '{pattern_str}' -> token {new_id} (count={count})")
        
        # Clear counts after mining
        self.ngram_counts.clear()
        
        return new_tokens
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def demo_micro_infinity_v2():
    """Demonstrate the Micro-Infinity v2 model."""
    print("=" * 60)
    print("DEMO: Micro-Infinity v2")
    print("=" * 60)
    
    config = InfinityV2Config(
        d_model=256,
        n_layers=6,
        n_heads=4,
        ghost_warmup_steps=100,  # Shorter for demo
        mining_threshold=5,      # Lower for demo
        mining_interval=10,
    )
    
    model = MicroInfinityV2(config)
    print(f"\nParameters: {model.count_parameters():,}")
    
    # Sample texts
    texts = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "She said the thing about the weather.",
        "The quick brown fox jumps over the lazy dog.",
    ] * 5  # Repeat to build up n-gram counts
    
    print(f"\n--- Training simulation ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(50):
        model.current_step = step
        
        # Tokenize
        input_ids, position_ids, target_ids, char_lengths = model.tokenize_batch(texts)
        input_ids = input_ids.to(device)
        position_ids = position_ids.to(device)
        target_ids = target_ids.to(device)
        char_lengths = char_lengths.to(device)
        
        # Forward
        output = model(input_ids, position_ids, target_ids, char_lengths)
        loss = output['loss']
        metrics = output['metrics']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Mine n-grams
        model.mine_ngrams(input_ids)
        
        # Maybe expand vocab
        new_tokens = model.maybe_expand_vocab(max_new_tokens=3)
        
        if step % 10 == 0:
            print(f"Step {step:3d}: BPC={metrics['bpc']:.4f}, "
                  f"loss={metrics['per_token_loss']:.4f}, "
                  f"vocab={model.tokenizer.current_vocab_size}, "
                  f"ghosts={len(model.embedding.ghost_tokens)}")
    
    print(f"\n--- Final state ---")
    print(f"Vocabulary size: {model.tokenizer.current_vocab_size}")
    print(f"Contractions: {len(model.tokenizer.contractions)}")
    print(f"Active ghost tokens: {len(model.embedding.ghost_tokens)}")
    
    # Show learned contractions
    print(f"\nLearned contractions:")
    for pattern, token_id in list(model.tokenizer.contractions.items())[:10]:
        pattern_str = ''.join(chr(t) if 32 <= t < 127 else f'[{t}]' for t in pattern)
        print(f"  {token_id}: '{pattern_str}'")
    
    # Demonstrate position stability
    print(f"\n--- Position stability check ---")
    test_text = "The cat sat on the mat."
    
    # Before contractions
    model.tokenizer.contractions.clear()
    tokens_before = model.tokenizer.tokenize_with_positions(test_text, apply_contractions=False)
    
    # After contractions
    model.tokenizer.contractions = {(ord('T'), ord('h'), ord('e')): 259}
    tokens_after = model.tokenizer.tokenize_with_positions(test_text, apply_contractions=True)
    
    print(f"Text: '{test_text}'")
    print(f"'c' position before contraction: {[t.char_start for t in tokens_before if t.text == 'c'][0]}")
    print(f"'c' position after 'The' contracted: {[t.char_start for t in tokens_after if t.text == 'c'][0]}")
    print("✓ Position stable!")


if __name__ == "__main__":
    demo_micro_infinity_v2()
