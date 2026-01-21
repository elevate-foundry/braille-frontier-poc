"""
Braille Tokenizer - Text ↔ Braille conversion

Uses Grade 1 Braille (letter-for-letter) for simplicity.
Unicode Braille Patterns: U+2800 to U+28FF (256 cells)

The offset (codepoint - 0x2800) maps directly to dot patterns:
- Bit 0 = Dot 1, Bit 1 = Dot 2, ... Bit 7 = Dot 8
"""

import torch
from typing import List, Tuple

BRAILLE_BASE = 0x2800
BRAILLE_MAX = 0x28FF

# Special tokens (using high end of Braille range that aren't standard letters)
PAD_ID = 0      # ⠀ (empty cell)
BOS_ID = 256    # Beyond Braille range
EOS_ID = 257
UNK_ID = 258

VOCAB_SIZE = 259  # 256 Braille cells + PAD/BOS/EOS/UNK

# Grade 1 Braille: letter-for-letter mapping
# Lowercase letters a-z map to specific dot patterns
LETTER_TO_BRAILLE = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑',
    'f': '⠋', 'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚',
    'k': '⠅', 'l': '⠇', 'm': '⠍', 'n': '⠝', 'o': '⠕',
    'p': '⠏', 'q': '⠟', 'r': '⠗', 's': '⠎', 't': '⠞',
    'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭', 'y': '⠽',
    'z': '⠵',
    ' ': '⠀',  # Space = empty cell
    '.': '⠲', ',': '⠂', '!': '⠖', '?': '⠦',
    "'": '⠄', '-': '⠤', ':': '⠒', ';': '⠆',
    '0': '⠴', '1': '⠂', '2': '⠆', '3': '⠒', '4': '⠲',
    '5': '⠢', '6': '⠖', '7': '⠶', '8': '⠦', '9': '⠔',
}

BRAILLE_TO_LETTER = {v: k for k, v in LETTER_TO_BRAILLE.items()}


def char_to_braille_id(char: str) -> int:
    """Convert a single character to a Braille token ID."""
    char = char.lower()
    if char in LETTER_TO_BRAILLE:
        braille_char = LETTER_TO_BRAILLE[char]
        return ord(braille_char) - BRAILLE_BASE
    return UNK_ID - 256  # Map UNK to a valid range


def braille_id_to_char(token_id: int) -> str:
    """Convert a Braille token ID back to a character."""
    if token_id == PAD_ID:
        return ''
    if token_id >= 256:  # Special tokens
        return ''
    braille_char = chr(token_id + BRAILLE_BASE)
    return BRAILLE_TO_LETTER.get(braille_char, '?')


def text_to_braille_ids(text: str, add_special: bool = True) -> List[int]:
    """Convert text to a list of Braille token IDs."""
    ids = []
    if add_special:
        ids.append(BOS_ID)
    for char in text:
        ids.append(char_to_braille_id(char))
    if add_special:
        ids.append(EOS_ID)
    return ids


def braille_ids_to_text(ids: List[int]) -> str:
    """Convert Braille token IDs back to text."""
    chars = []
    for token_id in ids:
        if token_id in (BOS_ID, EOS_ID, PAD_ID):
            continue
        chars.append(braille_id_to_char(token_id))
    return ''.join(chars)


def get_braille_bitmask(token_ids: torch.Tensor) -> torch.Tensor:
    """
    Convert Braille token IDs to 8-bit dot patterns.
    
    Args:
        token_ids: Tensor of shape [...] containing token IDs (0-255 for Braille cells)
    
    Returns:
        Tensor of shape [..., 8] with float values 0.0 or 1.0 for each dot
    """
    # Clamp to valid Braille range (special tokens get zero mask)
    clamped = torch.clamp(token_ids, 0, 255)
    
    # Extract bits: bit i corresponds to dot (i+1)
    bits = ((clamped.unsqueeze(-1) >> torch.arange(8, device=token_ids.device)) & 1)
    
    # Zero out masks for special tokens (>= 256)
    mask = (token_ids < 256).unsqueeze(-1).float()
    
    return bits.float() * mask


def build_mask_table(vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """
    Build a lookup table mapping vocab IDs to 8-bit masks.
    
    Returns:
        Tensor of shape [vocab_size, 8]
    """
    table = torch.zeros(vocab_size, 8, dtype=torch.float32)
    for i in range(256):  # Only Braille cells get masks
        mask = torch.tensor([(i >> bit) & 1 for bit in range(8)], dtype=torch.float32)
        table[i] = mask
    return table


# Quick test
if __name__ == "__main__":
    test_text = "hello world"
    ids = text_to_braille_ids(test_text)
    print(f"Text: {test_text}")
    print(f"IDs: {ids}")
    print(f"Braille: {''.join(chr(i + BRAILLE_BASE) if i < 256 else '?' for i in ids)}")
    print(f"Decoded: {braille_ids_to_text(ids)}")
    
    # Test bitmask
    tensor_ids = torch.tensor(ids)
    masks = get_braille_bitmask(tensor_ids)
    print(f"Bitmask shape: {masks.shape}")
