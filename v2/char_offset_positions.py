"""
Infinity v2: Character-Offset Positional Encoding

The key insight: decouple Token Index from Positional Index.
Each token's position = the character offset of its first character in the original string.

This makes vocabulary expansion INVISIBLE to attention - "cat" stays at position 4
regardless of whether "the" is 3 tokens or 1 token.
"""

import torch
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TokenWithPosition:
    """A token with its character-offset position."""
    token_id: int
    char_start: int  # Position in original string
    char_end: int    # End position (exclusive)
    text: str        # Original text (for debugging)


class CharOffsetTokenizer:
    """
    Tokenizer that tracks character offsets for each token.
    
    Key property: When contractions are applied, positions don't shift.
    
    Example:
        "The cat" with no contractions:
        [T](0), [h](1), [e](2), [ ](3), [c](4), [a](5), [t](6)
        
        "The cat" after "The" becomes token 259:
        [The](0), [ ](3), [c](4), [a](5), [t](6)
        
        Notice: "c" is still at position 4!
    """
    
    # Special tokens
    PAD_ID = 0
    BOS_ID = 256
    EOS_ID = 257
    UNK_ID = 258
    
    def __init__(self, max_vocab_size: int = 16384):
        self.max_vocab_size = max_vocab_size
        self.base_vocab_size = 259  # 256 bytes + BOS + EOS + UNK
        self.current_vocab_size = self.base_vocab_size
        
        # Contractions: pattern (tuple of token IDs) -> new token ID
        self.contractions: Dict[Tuple[int, ...], int] = {}
        
        # Reverse mapping for decoding
        self.id_to_pattern: Dict[int, Tuple[int, ...]] = {}
    
    def add_contraction(self, pattern: Tuple[int, ...]) -> int:
        """
        Add a new contraction (e.g., (84, 104, 101) for "The" -> 259).
        Returns the new token ID.
        """
        if pattern in self.contractions:
            return self.contractions[pattern]
        
        if self.current_vocab_size >= self.max_vocab_size:
            raise ValueError("Vocabulary full")
        
        new_id = self.current_vocab_size
        self.current_vocab_size += 1
        self.contractions[pattern] = new_id
        self.id_to_pattern[new_id] = pattern
        
        return new_id
    
    def tokenize_with_positions(
        self, 
        text: str,
        apply_contractions: bool = True
    ) -> List[TokenWithPosition]:
        """
        Tokenize text and return tokens with character-offset positions.
        
        Args:
            text: Input string
            apply_contractions: Whether to apply learned contractions
            
        Returns:
            List of TokenWithPosition, each with char_start = character offset
        """
        # Step 1: Convert to base tokens (one per character)
        base_tokens = []
        for i, char in enumerate(text):
            token_id = ord(char) % 256  # Map to byte
            base_tokens.append(TokenWithPosition(
                token_id=token_id,
                char_start=i,
                char_end=i + 1,
                text=char
            ))
        
        if not apply_contractions or not self.contractions:
            return base_tokens
        
        # Step 2: Apply contractions (greedy, longest-first)
        # Sort contractions by length (longest first)
        sorted_contractions = sorted(
            self.contractions.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        result = []
        i = 0
        while i < len(base_tokens):
            matched = False
            
            for pattern, new_id in sorted_contractions:
                plen = len(pattern)
                if i + plen <= len(base_tokens):
                    # Check if pattern matches
                    window = tuple(t.token_id for t in base_tokens[i:i+plen])
                    if window == pattern:
                        # Create merged token with FIRST character's position
                        merged = TokenWithPosition(
                            token_id=new_id,
                            char_start=base_tokens[i].char_start,  # Key: use first char's position
                            char_end=base_tokens[i+plen-1].char_end,
                            text=''.join(t.text for t in base_tokens[i:i+plen])
                        )
                        result.append(merged)
                        i += plen
                        matched = True
                        break
            
            if not matched:
                result.append(base_tokens[i])
                i += 1
        
        return result
    
    def get_position_ids(self, tokens: List[TokenWithPosition]) -> torch.Tensor:
        """Extract position IDs (character offsets) from tokens."""
        return torch.tensor([t.char_start for t in tokens], dtype=torch.long)
    
    def get_token_ids(self, tokens: List[TokenWithPosition]) -> torch.Tensor:
        """Extract token IDs from tokens."""
        return torch.tensor([t.token_id for t in tokens], dtype=torch.long)
    
    def get_char_lengths(self, tokens: List[TokenWithPosition]) -> torch.Tensor:
        """Get character length of each token (for BPC-normalized loss)."""
        return torch.tensor([t.char_end - t.char_start for t in tokens], dtype=torch.float)


def demo_char_offset_stability():
    """
    Demonstrate that character offsets remain stable during vocabulary expansion.
    """
    print("=" * 60)
    print("DEMO: Character-Offset Position Stability")
    print("=" * 60)
    
    tokenizer = CharOffsetTokenizer()
    text = "The cat sat on the mat."
    
    # Before any contractions
    print(f"\nText: '{text}'")
    print("\n--- Before contractions ---")
    tokens_v1 = tokenizer.tokenize_with_positions(text, apply_contractions=False)
    for t in tokens_v1[:10]:  # First 10 tokens
        print(f"  '{t.text}' -> token {t.token_id:3d} @ position {t.char_start}")
    print(f"  ... ({len(tokens_v1)} total tokens)")
    
    # Add contraction for "The" (T=84, h=104, e=101)
    the_pattern = (ord('T'), ord('h'), ord('e'))
    the_id = tokenizer.add_contraction(the_pattern)
    print(f"\n--- After adding 'The' as token {the_id} ---")
    
    tokens_v2 = tokenizer.tokenize_with_positions(text, apply_contractions=True)
    for t in tokens_v2[:10]:
        print(f"  '{t.text}' -> token {t.token_id:3d} @ position {t.char_start}")
    print(f"  ... ({len(tokens_v2)} total tokens)")
    
    # Add more contractions
    cat_pattern = (ord('c'), ord('a'), ord('t'))
    cat_id = tokenizer.add_contraction(cat_pattern)
    
    the_lower_pattern = (ord('t'), ord('h'), ord('e'))
    the_lower_id = tokenizer.add_contraction(the_lower_pattern)
    
    print(f"\n--- After adding 'cat' ({cat_id}) and 'the' ({the_lower_id}) ---")
    
    tokens_v3 = tokenizer.tokenize_with_positions(text, apply_contractions=True)
    for t in tokens_v3:
        print(f"  '{t.text}' -> token {t.token_id:3d} @ position {t.char_start}")
    
    # Show position stability
    print("\n" + "=" * 60)
    print("KEY OBSERVATION: Position Stability")
    print("=" * 60)
    
    # Find 'cat' in each version
    def find_token_position(tokens, text_match):
        for t in tokens:
            if t.text == text_match or t.text.startswith(text_match):
                return t.char_start
        return None
    
    print(f"\nPosition of 'c' (start of 'cat'):")
    print(f"  v1 (no contractions):  {find_token_position(tokens_v1, 'c')}")
    print(f"  v2 (The contracted):   {find_token_position(tokens_v2, 'c')}")
    print(f"  v3 (cat contracted):   {find_token_position(tokens_v3, 'cat')}")
    print("\n✓ Position remains 4 regardless of tokenization!")
    
    # Show compression stats
    print("\n" + "=" * 60)
    print("COMPRESSION STATS")
    print("=" * 60)
    print(f"Original characters: {len(text)}")
    print(f"Tokens (no contractions): {len(tokens_v1)}")
    print(f"Tokens (with contractions): {len(tokens_v3)}")
    print(f"Compression ratio: {len(text) / len(tokens_v3):.2f} chars/token")
    
    return tokenizer


if __name__ == "__main__":
    demo_char_offset_stability()
