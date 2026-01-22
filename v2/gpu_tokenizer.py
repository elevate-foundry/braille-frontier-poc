"""
Infinity v2: GPU-Accelerated Tokenization

The v1 bottleneck: Re-tokenizing the entire dataset in Python after each
vocabulary expansion caused a 3.3x training slowdown.

Solution: Keep data as raw bytes, perform contraction matching ON THE GPU
using vectorized operations.

Architecture:
1. TrieMatcher: Simple trie for pattern matching (CPU, used for building)
2. GPUTrieTokenizer: Tensor-based trie for GPU-accelerated batch tokenization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time


class TrieNode:
    """Node in a trie for pattern matching."""
    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}
        self.token_id: Optional[int] = None  # If this node is end of a pattern
        self.pattern_len: int = 0


class TrieMatcher:
    """
    Simple trie-based pattern matcher.
    
    Used for:
    1. Building the pattern database
    2. Converting to GPU-friendly tensor format
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.patterns: Dict[Tuple[int, ...], int] = {}
    
    def add_pattern(self, pattern: Tuple[int, ...], token_id: int):
        """Add a pattern to the trie."""
        if pattern in self.patterns:
            return
        
        self.patterns[pattern] = token_id
        
        node = self.root
        for byte_val in pattern:
            if byte_val not in node.children:
                node.children[byte_val] = TrieNode()
            node = node.children[byte_val]
        
        node.token_id = token_id
        node.pattern_len = len(pattern)
    
    def match_longest(self, seq: List[int], start: int) -> Tuple[Optional[int], int]:
        """
        Find longest matching pattern starting at position `start`.
        
        Returns (token_id, length) or (None, 0) if no match.
        """
        node = self.root
        best_match = None
        best_len = 0
        
        for i in range(start, len(seq)):
            byte_val = seq[i]
            if byte_val not in node.children:
                break
            
            node = node.children[byte_val]
            if node.token_id is not None:
                best_match = node.token_id
                best_len = node.pattern_len
        
        return best_match, best_len
    
    def tokenize(self, seq: List[int]) -> Tuple[List[int], List[int], List[int]]:
        """
        Tokenize a sequence using greedy longest-match.
        
        Returns (token_ids, char_positions, char_lengths).
        """
        tokens = []
        positions = []
        lengths = []
        
        i = 0
        while i < len(seq):
            match_id, match_len = self.match_longest(seq, i)
            
            if match_id is not None and match_len > 1:
                tokens.append(match_id)
                positions.append(i)
                lengths.append(match_len)
                i += match_len
            else:
                # No contraction, emit raw byte
                tokens.append(seq[i])
                positions.append(i)
                lengths.append(1)
                i += 1
        
        return tokens, positions, lengths


class GPUTrieTokenizer(nn.Module):
    """
    GPU-accelerated tokenizer using tensor-based trie.
    
    The trie is flattened into tensors:
    - children_table: [num_nodes, 256] -> child node index (-1 if none)
    - output_table: [num_nodes] -> token_id (-1 if not end of pattern)
    - length_table: [num_nodes] -> pattern length (0 if not end)
    
    This allows batch processing on GPU.
    """
    
    def __init__(self, max_nodes: int = 65536):
        super().__init__()
        self.max_nodes = max_nodes
        
        # Trie tables as buffers (move to GPU with model)
        self.register_buffer(
            'children_table',
            torch.full((max_nodes, 256), -1, dtype=torch.long)
        )
        self.register_buffer(
            'output_table',
            torch.full((max_nodes,), -1, dtype=torch.long)
        )
        self.register_buffer(
            'length_table',
            torch.zeros(max_nodes, dtype=torch.long)
        )
        
        self.num_nodes = 1  # Node 0 is root
        self.cpu_trie = TrieMatcher()  # For building
    
    def add_pattern(self, pattern: Tuple[int, ...], token_id: int):
        """Add a pattern and update tensor representation."""
        if pattern in self.cpu_trie.patterns:
            return
        
        self.cpu_trie.add_pattern(pattern, token_id)
        
        # Also update tensor trie
        node_idx = 0  # Start at root
        for byte_val in pattern:
            child_idx = self.children_table[node_idx, byte_val].item()
            
            if child_idx == -1:
                # Create new node
                child_idx = self.num_nodes
                self.num_nodes += 1
                
                if child_idx >= self.max_nodes:
                    raise ValueError(f"Exceeded max nodes ({self.max_nodes})")
                
                self.children_table[node_idx, byte_val] = child_idx
            
            node_idx = child_idx
        
        # Mark end of pattern
        self.output_table[node_idx] = token_id
        self.length_table[node_idx] = len(pattern)
    
    @torch.no_grad()
    def tokenize_batch_gpu(
        self,
        byte_sequences: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of byte sequences on GPU.
        
        This uses a vectorized approach where we process all sequences
        in parallel, though the sequential nature of tokenization limits
        full parallelism.
        
        Args:
            byte_sequences: [batch, max_len] byte values (0-255)
            lengths: [batch] actual length of each sequence
            
        Returns:
            token_ids: [batch, max_tokens]
            char_positions: [batch, max_tokens]
            char_lengths: [batch, max_tokens]
            num_tokens: [batch]
        """
        batch_size, max_len = byte_sequences.shape
        device = byte_sequences.device
        
        # Output buffers
        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        char_positions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        char_lengths = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        num_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Process each sequence
        # Note: This loop can be replaced with a custom CUDA kernel for full parallelism
        for b in range(batch_size):
            seq_len = lengths[b].item()
            seq = byte_sequences[b, :seq_len]
            
            toks, pos, lens = self._tokenize_single_gpu(seq)
            
            n = len(toks)
            token_ids[b, :n] = torch.tensor(toks, device=device)
            char_positions[b, :n] = torch.tensor(pos, device=device)
            char_lengths[b, :n] = torch.tensor(lens, device=device)
            num_tokens[b] = n
        
        return token_ids, char_positions, char_lengths, num_tokens
    
    def _tokenize_single_gpu(self, seq: torch.Tensor) -> Tuple[List[int], List[int], List[int]]:
        """Tokenize a single sequence using tensor trie."""
        tokens = []
        positions = []
        lengths = []
        
        seq_list = seq.tolist()
        n = len(seq_list)
        i = 0
        
        while i < n:
            # Find longest match starting at i
            node_idx = 0
            best_match = -1
            best_len = 0
            
            for j in range(i, n):
                byte_val = seq_list[j]
                child_idx = self.children_table[node_idx, byte_val].item()
                
                if child_idx == -1:
                    break
                
                node_idx = child_idx
                
                # Check if this is end of a pattern
                if self.output_table[node_idx].item() != -1:
                    best_match = self.output_table[node_idx].item()
                    best_len = self.length_table[node_idx].item()
            
            if best_match != -1 and best_len > 1:
                tokens.append(best_match)
                positions.append(i)
                lengths.append(best_len)
                i += best_len
            else:
                tokens.append(seq_list[i])
                positions.append(i)
                lengths.append(1)
                i += 1
        
        return tokens, positions, lengths
    
    def tokenize_batch_parallel(
        self,
        byte_sequences: torch.Tensor,
        lengths: torch.Tensor,
        max_pattern_len: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fully parallel tokenization using tensor operations.
        
        Strategy: For each position, compute the longest match in parallel
        using matrix operations. Then greedily select non-overlapping matches.
        
        This is O(n * max_pattern_len) per sequence but fully parallelizable.
        """
        batch_size, max_len = byte_sequences.shape
        device = byte_sequences.device
        
        # Step 1: For each position, find longest match ending there
        # match_lengths[b, i] = length of longest pattern ending at position i
        # match_tokens[b, i] = token_id of that pattern (-1 if none)
        
        match_lengths = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        match_tokens = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
        
        # For each possible pattern length, check all positions in parallel
        for pattern_len in range(2, max_pattern_len + 1):
            if pattern_len > max_len:
                break
            
            # Check all windows of this length
            for start_offset in range(max_len - pattern_len + 1):
                # Get the window for all batches
                windows = byte_sequences[:, start_offset:start_offset + pattern_len]
                
                # Walk the trie for this window (vectorized across batch)
                node_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                valid = torch.ones(batch_size, dtype=torch.bool, device=device)
                
                for pos_in_window in range(pattern_len):
                    bytes_at_pos = windows[:, pos_in_window]
                    
                    # Look up children for all batches
                    # children_table is [num_nodes, 256]
                    # node_indices is [batch]
                    # bytes_at_pos is [batch]
                    
                    # Gather: for each batch, get children_table[node_indices[b], bytes_at_pos[b]]
                    next_nodes = self.children_table[node_indices, bytes_at_pos]
                    
                    # Mark invalid if no child
                    valid = valid & (next_nodes != -1)
                    node_indices = torch.where(valid, next_nodes, node_indices)
                
                # Check which batches found a valid pattern
                end_pos = start_offset + pattern_len - 1
                found_pattern = valid & (self.output_table[node_indices] != -1)
                
                # Update match info where we found a longer match
                longer_match = found_pattern & (pattern_len > match_lengths[:, end_pos])
                match_lengths[:, end_pos] = torch.where(
                    longer_match, 
                    torch.tensor(pattern_len, device=device),
                    match_lengths[:, end_pos]
                )
                match_tokens[:, end_pos] = torch.where(
                    longer_match,
                    self.output_table[node_indices],
                    match_tokens[:, end_pos]
                )
        
        # Step 2: Greedy selection of non-overlapping matches
        # This part is sequential per sequence but fast
        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        char_positions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        char_lengths = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        num_tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            seq_len = lengths[b].item()
            toks, pos, lens = self._greedy_select(
                byte_sequences[b, :seq_len].tolist(),
                match_lengths[b, :seq_len].tolist(),
                match_tokens[b, :seq_len].tolist(),
            )
            n = len(toks)
            token_ids[b, :n] = torch.tensor(toks, device=device)
            char_positions[b, :n] = torch.tensor(pos, device=device)
            char_lengths[b, :n] = torch.tensor(lens, device=device)
            num_tokens[b] = n
        
        return token_ids, char_positions, char_lengths, num_tokens
    
    def _greedy_select(
        self,
        seq: List[int],
        match_lengths: List[int],
        match_tokens: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        """Greedy selection of non-overlapping matches."""
        tokens = []
        positions = []
        lengths = []
        
        i = 0
        n = len(seq)
        
        while i < n:
            # Look for longest match starting at i
            best_len = 0
            best_token = -1
            
            # Check all positions that could be the end of a match starting at i
            for end in range(i + 1, min(i + 17, n)):  # max pattern len + 1
                if match_lengths[end] > 0:
                    start = end - match_lengths[end] + 1
                    if start == i and match_lengths[end] > best_len:
                        best_len = match_lengths[end]
                        best_token = match_tokens[end]
            
            if best_token != -1 and best_len > 1:
                tokens.append(best_token)
                positions.append(i)
                lengths.append(best_len)
                i += best_len
            else:
                tokens.append(seq[i])
                positions.append(i)
                lengths.append(1)
                i += 1
        
        return tokens, positions, lengths


class GPUTokenizerV2(nn.Module):
    """
    Complete GPU tokenizer for Infinity v2.
    
    Combines:
    - Trie-based pattern matching (GPU-accelerated)
    - Character-offset position tracking
    - Dynamic vocabulary expansion
    """
    
    # Special tokens
    PAD_ID = 0
    BOS_ID = 256
    EOS_ID = 257
    UNK_ID = 258
    BASE_VOCAB_SIZE = 259
    
    def __init__(self, max_vocab_size: int = 16384):
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.current_vocab_size = self.BASE_VOCAB_SIZE
        
        # Trie-based tokenizer for contractions
        self.trie = GPUTrieTokenizer(max_nodes=max_vocab_size * 8)
        
        # Pattern storage for reference
        self.contractions: Dict[Tuple[int, ...], int] = {}
    
    def add_contraction(self, pattern: Tuple[int, ...]) -> int:
        """Add a new contraction pattern."""
        if pattern in self.contractions:
            return self.contractions[pattern]
        
        new_id = self.current_vocab_size
        self.current_vocab_size += 1
        
        self.contractions[pattern] = new_id
        self.trie.add_pattern(pattern, new_id)
        
        return new_id
    
    def tokenize_texts(
        self,
        texts: List[str],
        max_len: int = 512,
        add_special_tokens: bool = True,
        use_parallel: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of input strings
            max_len: Maximum sequence length
            add_special_tokens: Whether to add BOS/EOS
            use_parallel: Use fully parallel tokenization (faster on GPU)
            
        Returns:
            token_ids: [batch, max_len]
            position_ids: [batch, max_len] (character offsets)
            char_lengths: [batch, max_len]
            attention_mask: [batch, max_len]
        """
        batch_size = len(texts)
        device = self.trie.children_table.device
        
        # Convert texts to byte tensors
        byte_seqs = []
        lengths = []
        
        for text in texts:
            bytes_list = list(text.encode('utf-8', errors='replace'))
            byte_seqs.append(bytes_list)
            lengths.append(len(bytes_list))
        
        # Pad to max length
        max_text_len = max(lengths) if lengths else 1
        byte_tensor = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
        for i, seq in enumerate(byte_seqs):
            byte_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
        length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        
        # Run trie matching
        if use_parallel:
            token_ids, char_positions, char_lengths, num_tokens = self.trie.tokenize_batch_parallel(
                byte_tensor, length_tensor
            )
        else:
            token_ids, char_positions, char_lengths, num_tokens = self.trie.tokenize_batch_gpu(
                byte_tensor, length_tensor
            )
        
        # Add special tokens if requested
        if add_special_tokens:
            # Shift everything right by 1 for BOS
            new_token_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            new_positions = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            new_char_lens = torch.ones(batch_size, max_len, dtype=torch.long, device=device)
            
            new_token_ids[:, 0] = self.BOS_ID
            new_positions[:, 0] = 0
            new_char_lens[:, 0] = 0  # BOS has no character length
            
            for b in range(batch_size):
                n = min(num_tokens[b].item(), max_len - 2)  # Leave room for BOS/EOS
                new_token_ids[b, 1:n+1] = token_ids[b, :n]
                new_positions[b, 1:n+1] = char_positions[b, :n]
                new_char_lens[b, 1:n+1] = char_lengths[b, :n]
                
                # Add EOS
                if n + 1 < max_len:
                    new_token_ids[b, n+1] = self.EOS_ID
                    new_positions[b, n+1] = lengths[b]
                    new_char_lens[b, n+1] = 0
                    num_tokens[b] = n + 2
                else:
                    num_tokens[b] = n + 1
            
            token_ids = new_token_ids
            char_positions = new_positions
            char_lengths = new_char_lens
        
        # Create attention mask
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for b in range(batch_size):
            attention_mask[b, :num_tokens[b]] = 1
        
        return token_ids, char_positions, char_lengths, attention_mask


def benchmark_gpu_tokenizer():
    """Benchmark GPU tokenizer vs Python tokenizer."""
    print("=" * 60)
    print("BENCHMARK: GPU Trie Tokenizer")
    print("=" * 60)
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers.",
        "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    ] * 100  # 400 texts
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create GPU tokenizer
    gpu_tok = GPUTokenizerV2().to(device)
    
    # Add some contractions
    contractions = [
        (ord('t'), ord('h'), ord('e')),  # "the"
        (ord('i'), ord('n'), ord('g')),  # "ing"
        (ord('o'), ord('u'), ord('l'), ord('d')),  # "ould"
        (ord('c'), ord('h'), ord('u'), ord('c'), ord('k')),  # "chuck"
        (ord(' '), ord('t'), ord('h'), ord('e'), ord(' ')),  # " the "
    ]
    
    for pattern in contractions:
        gpu_tok.add_contraction(pattern)
    
    print(f"\nContractions: {len(contractions)}")
    print(f"Texts: {len(texts)}")
    
    # Warmup
    _ = gpu_tok.tokenize_texts(texts[:10])
    
    # Benchmark sequential tokenizer
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(10):
        token_ids, positions, char_lens, mask = gpu_tok.tokenize_texts(texts)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    seq_time = (time.perf_counter() - start) / 10
    
    print(f"\n--- Sequential Results ---")
    print(f"Time: {seq_time*1000:.2f} ms for {len(texts)} texts")
    print(f"Throughput: {len(texts) / seq_time:.0f} texts/sec")
    
    # Benchmark parallel tokenizer
    start = time.perf_counter()
    for _ in range(10):
        token_ids, positions, char_lens, mask = gpu_tok.tokenize_texts(texts, use_parallel=True)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    par_time = (time.perf_counter() - start) / 10
    
    print(f"\n--- Parallel Results ---")
    print(f"Time: {par_time*1000:.2f} ms for {len(texts)} texts")
    print(f"Throughput: {len(texts) / par_time:.0f} texts/sec")
    print(f"Speedup: {seq_time / par_time:.2f}x")
    
    # Show sample output
    print(f"\n--- Sample output ---")
    sample_text = texts[0]
    tok_ids, pos_ids, char_lens, mask = gpu_tok.tokenize_texts([sample_text], max_len=64)
    
    print(f"Text: '{sample_text}'")
    print(f"Tokens: {tok_ids[0, :20].tolist()}")
    print(f"Positions: {pos_ids[0, :20].tolist()}")
    print(f"Char lengths: {char_lens[0, :20].tolist()}")
    
    # Verify position stability
    print(f"\n--- Position stability test ---")
    
    # Without contractions
    gpu_tok_no_contract = GPUTokenizerV2().to(device)
    tok_ids_v1, pos_ids_v1, _, _ = gpu_tok_no_contract.tokenize_texts([sample_text], max_len=64)
    
    # Find 'q' in "quick" (should be at position 4)
    # In byte encoding, 'q' = 113
    q_pos_v1 = None
    q_pos_v2 = None
    
    for i in range(tok_ids_v1.shape[1]):
        if tok_ids_v1[0, i].item() == ord('q'):
            q_pos_v1 = pos_ids_v1[0, i].item()
            break
    
    for i in range(tok_ids.shape[1]):
        if tok_ids[0, i].item() == ord('q'):
            q_pos_v2 = pos_ids[0, i].item()
            break
    
    print(f"'q' position without contractions: {q_pos_v1}")
    print(f"'q' position with 'the' contracted: {q_pos_v2}")
    print(f"Position stable: {q_pos_v1 == q_pos_v2}")


def demo_trie_tokenizer():
    """Demonstrate trie-based pattern matching."""
    print("=" * 60)
    print("DEMO: Trie-Based Pattern Matching")
    print("=" * 60)
    
    trie = GPUTrieTokenizer()
    
    # Add patterns
    patterns = [
        ((ord('h'), ord('e')), 259),      # "he"
        ((ord('s'), ord('h'), ord('e')), 260),  # "she"
        ((ord('h'), ord('i'), ord('s')), 261),  # "his"
        ((ord('h'), ord('e'), ord('r'), ord('s')), 262),  # "hers"
    ]
    
    print("\nPatterns:")
    for pattern, token_id in patterns:
        pattern_str = ''.join(chr(b) for b in pattern)
        print(f"  '{pattern_str}' -> token {token_id}")
        trie.add_pattern(pattern, token_id)
    
    # Test matching
    test_text = "ushers"
    print(f"\nTest text: '{test_text}'")
    
    seq = torch.tensor([ord(c) for c in test_text], dtype=torch.long)
    tokens, positions, lengths = trie._tokenize_single_gpu(seq)
    
    print(f"\nMatches:")
    for tok, pos, length in zip(tokens, positions, lengths):
        if tok >= 259:
            pattern = [p for p, t in patterns if t == tok][0]
            pattern_str = ''.join(chr(b) for b in pattern)
            print(f"  Position {pos}: '{pattern_str}' (token {tok}, len={length})")
        else:
            print(f"  Position {pos}: '{chr(tok)}' (byte {tok})")


if __name__ == "__main__":
    demo_trie_tokenizer()
    print("\n")
    benchmark_gpu_tokenizer()
