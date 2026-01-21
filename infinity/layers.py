"""
3-Layer Infinity Token System

Layer 1: Geometric Ground Truth
    - 8-bit dot patterns for fundamental primitives
    - Math operators, logic gates, basic elements
    - Uses structured prior W·b(t)

Layer 2: Contraction-Based Compression
    - Multi-cell patterns promoted to single tokens
    - "Braille-grams" for high-frequency sequences
    - 40-60% sequence length reduction

Layer 3: Semantic Hyper-Tokens
    - Entire concepts as single tokens
    - Learned residual stores semantic deltas
    - Maximum information density
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class TokenLayer:
    """Metadata for a token's layer assignment."""
    layer: int  # 1=geometric, 2=contraction, 3=semantic
    components: List[int]  # Original token IDs this was built from
    frequency: int  # How often this appears
    compression_ratio: float  # How much sequence length it saves


class InfinityVocabulary:
    """
    Dynamic vocabulary that grows across 3 layers.
    
    Layer 1 (0-255): Base Braille cells (geometric)
    Layer 2 (256-4095): Contractions (multi-cell patterns)
    Layer 3 (4096+): Semantic hyper-tokens (concepts)
    """
    
    LAYER1_END = 256
    LAYER2_END = 4096
    
    # Special tokens
    PAD_ID = 0
    BOS_ID = 256
    EOS_ID = 257
    UNK_ID = 258
    
    def __init__(self, max_vocab_size: int = 16384):
        self.max_vocab_size = max_vocab_size
        self.current_size = 259  # Base + special tokens
        
        # Token metadata
        self.token_info: Dict[int, TokenLayer] = {}
        
        # Initialize Layer 1 (geometric)
        for i in range(256):
            self.token_info[i] = TokenLayer(
                layer=1,
                components=[i],
                frequency=0,
                compression_ratio=1.0,
            )
        
        # Shadow tokens (candidates for promotion)
        self.shadow_tokens: Dict[Tuple[int, ...], int] = {}  # pattern -> count
        
        # Promoted contractions (Layer 2)
        self.contractions: Dict[Tuple[int, ...], int] = {}  # pattern -> token_id
        
        # Semantic tokens (Layer 3)
        self.semantic_tokens: Dict[str, int] = {}  # concept_name -> token_id
    
    def get_layer(self, token_id: int) -> int:
        """Get the layer a token belongs to."""
        if token_id < self.LAYER1_END:
            return 1
        elif token_id < self.LAYER2_END:
            return 2
        else:
            return 3
    
    def record_ngram(self, pattern: Tuple[int, ...]):
        """Record an n-gram for potential promotion."""
        if pattern in self.shadow_tokens:
            self.shadow_tokens[pattern] += 1
        else:
            self.shadow_tokens[pattern] = 1
    
    def get_promotion_candidates(
        self,
        min_frequency: int = 100,
        min_compression: float = 1.5,
    ) -> List[Tuple[Tuple[int, ...], int]]:
        """Get n-grams ready for promotion to Layer 2."""
        candidates = []
        for pattern, count in self.shadow_tokens.items():
            if count >= min_frequency:
                compression = len(pattern)  # Saves (n-1) tokens
                if compression >= min_compression:
                    candidates.append((pattern, count))
        
        # Sort by compression benefit
        candidates.sort(key=lambda x: x[1] * len(x[0]), reverse=True)
        return candidates
    
    def promote_to_layer2(self, pattern: Tuple[int, ...]) -> int:
        """Promote an n-gram to a Layer 2 contraction token."""
        if pattern in self.contractions:
            return self.contractions[pattern]
        
        if self.current_size >= self.LAYER2_END:
            raise ValueError("Layer 2 vocabulary full")
        
        new_id = self.current_size
        self.current_size += 1
        
        self.contractions[pattern] = new_id
        self.token_info[new_id] = TokenLayer(
            layer=2,
            components=list(pattern),
            frequency=self.shadow_tokens.get(pattern, 0),
            compression_ratio=len(pattern),
        )
        
        return new_id
    
    def add_semantic_token(self, concept: str, component_ids: List[int]) -> int:
        """Add a Layer 3 semantic hyper-token."""
        if concept in self.semantic_tokens:
            return self.semantic_tokens[concept]
        
        if self.current_size >= self.max_vocab_size:
            raise ValueError("Vocabulary full")
        
        new_id = max(self.LAYER2_END, self.current_size)
        self.current_size = new_id + 1
        
        self.semantic_tokens[concept] = new_id
        self.token_info[new_id] = TokenLayer(
            layer=3,
            components=component_ids,
            frequency=0,
            compression_ratio=len(component_ids),
        )
        
        return new_id
    
    def save(self, path: str):
        """Save vocabulary to disk."""
        data = {
            "current_size": self.current_size,
            "contractions": {str(k): v for k, v in self.contractions.items()},
            "semantic_tokens": self.semantic_tokens,
            "shadow_tokens": {str(k): v for k, v in self.shadow_tokens.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: str):
        """Load vocabulary from disk."""
        with open(path) as f:
            data = json.load(f)
        self.current_size = data["current_size"]
        self.contractions = {eval(k): v for k, v in data["contractions"].items()}
        self.semantic_tokens = data["semantic_tokens"]
        self.shadow_tokens = {eval(k): v for k, v in data["shadow_tokens"].items()}


class InfinityEmbedding(nn.Module):
    """
    3-Layer Infinity Embedding with dynamic vocabulary.
    
    E(t) = E_learned(t) + gate(t) * W·bits(t)
    
    Layer 1: Full geometric prior (gate high)
    Layer 2: Partial geometric (gate medium, uses component geometry)
    Layer 3: Pure semantic (gate low, learned only)
    """
    
    def __init__(
        self,
        vocab: InfinityVocabulary,
        d_model: int = 512,
        initial_gate: float = 1.0,
    ):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        
        # Geometric projection (Layer 1)
        self.geom_proj = nn.Linear(8, d_model, bias=False)
        
        # Learned embeddings (all layers)
        self.residual = nn.Embedding(vocab.max_vocab_size, d_model)
        nn.init.normal_(self.residual.weight, std=0.02)
        
        # Per-token gate
        self.gate = nn.Embedding(vocab.max_vocab_size, 1)
        
        # Initialize gates by layer
        with torch.no_grad():
            # Layer 1: High gate (geometry dominates)
            self.gate.weight[:256] = initial_gate
            # Layer 2: Medium gate
            self.gate.weight[256:4096] = initial_gate * 0.5
            # Layer 3: Low gate (semantics dominate)
            self.gate.weight[4096:] = 0.0
        
        # Build geometric mask table
        self._build_mask_table()
    
    def _build_mask_table(self):
        """Build lookup table for geometric masks."""
        table = torch.zeros(self.vocab.max_vocab_size, 8)
        
        # Layer 1: Direct bit patterns
        for i in range(256):
            for bit in range(8):
                table[i, bit] = (i >> bit) & 1
        
        # Layer 2: Average of component patterns
        for pattern, token_id in self.vocab.contractions.items():
            component_masks = torch.zeros(8)
            for comp in pattern:
                if comp < 256:
                    for bit in range(8):
                        component_masks[bit] += (comp >> bit) & 1
            table[token_id] = component_masks / len(pattern)
        
        # Layer 3: Zero (pure semantic)
        # Already initialized to zero
        
        self.register_buffer("mask_table", table)
    
    def expand_vocabulary(self, new_token_id: int, component_ids: List[int]):
        """Expand embedding for a newly promoted token."""
        with torch.no_grad():
            # Initialize residual as mean of components
            component_embeds = self.residual.weight[component_ids]
            self.residual.weight[new_token_id] = component_embeds.mean(dim=0)
            
            # Set gate based on layer
            layer = self.vocab.get_layer(new_token_id)
            if layer == 2:
                self.gate.weight[new_token_id] = 0.5
            else:
                self.gate.weight[new_token_id] = 0.0
            
            # Update mask table
            if layer == 2:
                component_masks = self.mask_table[component_ids].mean(dim=0)
                self.mask_table[new_token_id] = component_masks
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]
        Returns:
            [batch, seq_len, d_model]
        """
        # Geometric embedding
        masks = self.mask_table[token_ids]  # [B, T, 8]
        geom = self.geom_proj(masks)  # [B, T, d_model]
        
        # Learned residual
        res = self.residual(token_ids)  # [B, T, d_model]
        
        # Gating
        g = torch.sigmoid(self.gate(token_ids))  # [B, T, 1]
        
        return res + g * geom
    
    def get_layer_stats(self) -> Dict[str, float]:
        """Get statistics about gate values per layer."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate.weight).squeeze()
            return {
                "layer1_gate": gates[:256].mean().item(),
                "layer2_gate": gates[256:4096].mean().item(),
                "layer3_gate": gates[4096:].mean().item(),
            }


class ShadowTokenMiner:
    """
    Mines n-grams from training data for potential promotion.
    
    Runs during training to identify high-frequency patterns
    that should become Layer 2 contractions.
    """
    
    def __init__(
        self,
        vocab: InfinityVocabulary,
        min_n: int = 2,
        max_n: int = 6,
        window_size: int = 10000,
    ):
        self.vocab = vocab
        self.min_n = min_n
        self.max_n = max_n
        self.window_size = window_size
        self.samples_seen = 0
    
    def mine(self, token_ids: torch.Tensor):
        """Extract n-grams from a batch of sequences."""
        batch_size, seq_len = token_ids.shape
        
        for b in range(batch_size):
            seq = token_ids[b].tolist()
            
            for n in range(self.min_n, self.max_n + 1):
                for i in range(seq_len - n + 1):
                    pattern = tuple(seq[i:i+n])
                    # Only mine Layer 1 tokens
                    if all(t < 256 for t in pattern):
                        self.vocab.record_ngram(pattern)
        
        self.samples_seen += batch_size
    
    def should_promote(self) -> bool:
        """Check if we should run promotion."""
        return self.samples_seen >= self.window_size
    
    def promote_top_k(self, k: int = 100) -> List[int]:
        """Promote top-k candidates to Layer 2."""
        candidates = self.vocab.get_promotion_candidates()[:k]
        new_ids = []
        
        for pattern, count in candidates:
            try:
                new_id = self.vocab.promote_to_layer2(pattern)
                new_ids.append(new_id)
                print(f"Promoted {pattern} -> token {new_id} (count={count})")
            except ValueError:
                break  # Vocabulary full
        
        # Reset mining window
        self.samples_seen = 0
        self.vocab.shadow_tokens.clear()
        
        return new_ids
