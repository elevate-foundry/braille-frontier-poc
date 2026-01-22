"""
Infinity v2 vs BPE: Fair BPC Comparison

This script evaluates both models on the same test set using Bits Per Character (BPC),
which normalizes for different compression ratios and provides a fair comparison.

BPC = total_cross_entropy / (total_characters * log(2))

Run with: modal run v2/compare_bpc.py
"""

import modal

app = modal.App("infinity-v2-bpc-comparison")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
    "tokenizers>=0.15.0",
    "tqdm",
)

checkpoint_volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={
        "/checkpoints": checkpoint_volume,
        "/data": data_volume,
    },
)
def compare_bpc():
    """Compare Infinity v2 and BPE models using BPC."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json
    import math
    import os
    from typing import Dict, List, Tuple, Optional
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("INFINITY V2 vs BPE: FAIR BPC COMPARISON")
    print("=" * 70)
    print(f"Device: {device}\n")
    
    # =========================================================================
    # MODEL DEFINITIONS
    # =========================================================================
    
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    
    class Attention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
        def forward(self, x):
            B, T, D = x.shape
            q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))
    
    class FeedForward(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            d_ff = d_model * 4
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.attn_norm = RMSNorm(d_model)
            self.attn = Attention(d_model, n_heads)
            self.ff_norm = RMSNorm(d_model)
            self.ff = FeedForward(d_model)
        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ff(self.ff_norm(x))
            return x
    
    # =========================================================================
    # INFINITY V2 MODEL (with char-offset RoPE)
    # =========================================================================
    
    def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def apply_rotary_emb(xq, xk, freqs_cis, position_ids):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs = freqs_cis[position_ids].unsqueeze(2)
        xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    class CharOffsetAttention(nn.Module):
        def __init__(self, d_model, n_heads, max_char_len=8192):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.wo = nn.Linear(d_model, d_model, bias=False)
            self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_dim, max_char_len))
        
        def forward(self, x, position_ids):
            B, T, D = x.shape
            q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
            k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
            v = self.wv(x).view(B, T, self.n_heads, self.head_dim)
            q, k = apply_rotary_emb(q, k, self.freqs_cis, position_ids)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))
    
    class TransformerBlockV2(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.attn_norm = RMSNorm(d_model)
            self.attn = CharOffsetAttention(d_model, n_heads)
            self.ff_norm = RMSNorm(d_model)
            self.ff = FeedForward(d_model)
        def forward(self, x, position_ids):
            x = x + self.attn(self.attn_norm(x), position_ids)
            x = x + self.ff(self.ff_norm(x))
            return x
    
    class InfinityV2Model(nn.Module):
        def __init__(self, d_model=512, n_layers=8, n_heads=8, max_vocab=16384):
            super().__init__()
            self.embedding = nn.Embedding(max_vocab, d_model)
            self.layers = nn.ModuleList([TransformerBlockV2(d_model, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, max_vocab, bias=False)
            self.head.weight = self.embedding.weight
        
        def forward(self, x, position_ids):
            h = self.embedding(x)
            for layer in self.layers:
                h = layer(h, position_ids)
            return self.head(self.norm(h))
    
    # =========================================================================
    # BPE MODEL (standard transformer)
    # =========================================================================
    
    class BPEModel(nn.Module):
        def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8, max_len=512):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(max_len, d_model)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        def forward(self, x):
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            h = self.embedding(x) + self.pos_embedding(pos)
            for layer in self.layers:
                h = layer(h)
            return self.head(self.norm(h))
    
    # =========================================================================
    # TOKENIZERS
    # =========================================================================
    
    class InfinityTokenizer:
        """Tokenizer for Infinity v2 with contraction support."""
        PAD_ID = 0
        BOS_ID = 256
        EOS_ID = 257
        
        def __init__(self, contractions: Dict[Tuple[int, ...], int] = None):
            self.contractions = contractions or {}
            self.sorted_contractions = sorted(
                self.contractions.items(),
                key=lambda x: len(x[0]),
                reverse=True
            )
        
        def tokenize(self, text: str) -> Tuple[List[int], List[int], List[int]]:
            """Returns (tokens, positions, char_lengths)."""
            bytes_list = list(text.encode('utf-8', errors='replace'))
            tokens, positions, lengths = [], [], []
            i = 0
            
            while i < len(bytes_list):
                matched = False
                for pattern, token_id in self.sorted_contractions:
                    plen = len(pattern)
                    if i + plen <= len(bytes_list) and tuple(bytes_list[i:i+plen]) == pattern:
                        tokens.append(token_id)
                        positions.append(i)
                        lengths.append(plen)
                        i += plen
                        matched = True
                        break
                if not matched:
                    tokens.append(bytes_list[i])
                    positions.append(i)
                    lengths.append(1)
                    i += 1
            
            return tokens, positions, lengths
        
        def encode_batch(self, texts: List[str], max_len: int = 512):
            """Encode batch with BOS/EOS and padding."""
            batch_tokens, batch_positions, batch_lengths = [], [], []
            
            for text in texts:
                tokens, positions, lengths = self.tokenize(text)
                tokens = [self.BOS_ID] + tokens[:max_len-2] + [self.EOS_ID]
                positions = [0] + positions[:max_len-2] + [positions[-1] + lengths[-1] if positions else 0]
                lengths = [0] + lengths[:max_len-2] + [0]
                
                pad_len = max_len - len(tokens)
                tokens += [self.PAD_ID] * pad_len
                positions += [0] * pad_len
                lengths += [1] * pad_len
                
                batch_tokens.append(tokens)
                batch_positions.append(positions)
                batch_lengths.append(lengths)
            
            return (
                torch.tensor(batch_tokens, dtype=torch.long),
                torch.tensor(batch_positions, dtype=torch.long),
                torch.tensor(batch_lengths, dtype=torch.float),
            )
    
    # =========================================================================
    # BPC COMPUTATION
    # =========================================================================
    
    def compute_bpc_infinity(model, tokenizer, texts, batch_size=32, max_len=512):
        """Compute BPC for Infinity v2 model."""
        model.eval()
        total_ce = 0.0
        total_chars = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Infinity v2"):
                batch_texts = texts[i:i+batch_size]
                tokens, positions, char_lens = tokenizer.encode_batch(batch_texts, max_len)
                tokens = tokens.to(device)
                positions = positions.to(device)
                char_lens = char_lens.to(device)
                
                input_ids = tokens[:, :-1]
                input_pos = positions[:, :-1]
                target_ids = tokens[:, 1:]
                target_lens = char_lens[:, 1:]
                
                logits = model(input_ids, input_pos)
                
                # Per-token CE
                B, T, V = logits.shape
                ce = F.cross_entropy(
                    logits.reshape(-1, V),
                    target_ids.reshape(-1),
                    reduction='none'
                ).reshape(B, T)
                
                # Mask padding
                mask = (target_ids != 0).float()
                
                # Weight by character length
                weighted_ce = (ce * target_lens * mask).sum()
                chars = (target_lens * mask).sum()
                
                total_ce += weighted_ce.item()
                total_chars += chars.item()
                total_tokens += mask.sum().item()
        
        bpc = total_ce / (total_chars * math.log(2)) if total_chars > 0 else float('inf')
        compression = total_chars / total_tokens if total_tokens > 0 else 1.0
        
        return {
            'bpc': bpc,
            'total_ce': total_ce,
            'total_chars': int(total_chars),
            'total_tokens': int(total_tokens),
            'compression': compression,
        }
    
    def compute_bpc_bpe(model, tokenizer, texts, batch_size=32, max_len=512):
        """Compute BPC for BPE model."""
        model.eval()
        total_ce = 0.0
        total_chars = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="BPE"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize with BPE
                encodings = [tokenizer.encode(t) for t in batch_texts]
                
                # Pad
                max_batch_len = min(max(len(e.ids) for e in encodings), max_len)
                tokens = torch.zeros(len(batch_texts), max_batch_len, dtype=torch.long)
                char_lens = torch.zeros(len(batch_texts), max_batch_len, dtype=torch.float)
                
                for j, enc in enumerate(encodings):
                    seq_len = min(len(enc.ids), max_batch_len)
                    tokens[j, :seq_len] = torch.tensor(enc.ids[:seq_len])
                    
                    # Get character lengths from offsets
                    for k in range(seq_len):
                        if k < len(enc.offsets):
                            start, end = enc.offsets[k]
                            char_lens[j, k] = end - start
                        else:
                            char_lens[j, k] = 1
                
                tokens = tokens.to(device)
                char_lens = char_lens.to(device)
                
                input_ids = tokens[:, :-1]
                target_ids = tokens[:, 1:]
                target_lens = char_lens[:, 1:]
                
                logits = model(input_ids)
                
                B, T, V = logits.shape
                ce = F.cross_entropy(
                    logits.reshape(-1, V),
                    target_ids.reshape(-1),
                    reduction='none'
                ).reshape(B, T)
                
                mask = (target_ids != 0).float()
                weighted_ce = (ce * target_lens * mask).sum()
                chars = (target_lens * mask).sum()
                
                total_ce += weighted_ce.item()
                total_chars += chars.item()
                total_tokens += mask.sum().item()
        
        bpc = total_ce / (total_chars * math.log(2)) if total_chars > 0 else float('inf')
        compression = total_chars / total_tokens if total_tokens > 0 else 1.0
        
        return {
            'bpc': bpc,
            'total_ce': total_ce,
            'total_chars': int(total_chars),
            'total_tokens': int(total_tokens),
            'compression': compression,
        }
    
    # =========================================================================
    # MAIN EVALUATION
    # =========================================================================
    
    # Load test data
    print("Loading test data...")
    test_file = "/data/frontier_test_texts.json"
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            test_texts = json.load(f)
    else:
        # Fall back to train data subset
        with open("/data/frontier_train_texts.json", "r") as f:
            all_texts = json.load(f)
        test_texts = all_texts[-500:]  # Last 500 for testing
    
    print(f"Test samples: {len(test_texts)}")
    total_test_chars = sum(len(t) for t in test_texts)
    print(f"Total test characters: {total_test_chars:,}\n")
    
    results = {}
    
    # =========================================================================
    # EVALUATE INFINITY V2
    # =========================================================================
    
    infinity_checkpoint = "/checkpoints/infinity_v2_best.pt"
    if os.path.exists(infinity_checkpoint):
        print("-" * 70)
        print("EVALUATING INFINITY V2")
        print("-" * 70)
        
        checkpoint = torch.load(infinity_checkpoint, map_location=device)
        config = checkpoint.get('config', {'d_model': 512, 'n_layers': 8, 'n_heads': 8})
        
        model = InfinityV2Model(
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
        ).to(device)
        
        # Load weights (handle potential key mismatches)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        
        # Load contractions
        contractions = checkpoint.get('contractions', {})
        # Convert string keys back to tuples if needed
        if contractions and isinstance(list(contractions.keys())[0], str):
            contractions = {eval(k): v for k, v in contractions.items()}
        
        tokenizer = InfinityTokenizer(contractions)
        
        print(f"Model loaded: {config}")
        print(f"Contractions: {len(contractions)}")
        print(f"Vocab size: {checkpoint.get('vocab_size', 259)}")
        
        infinity_results = compute_bpc_infinity(model, tokenizer, test_texts)
        results['infinity_v2'] = infinity_results
        
        print(f"\nInfinity v2 Results:")
        print(f"  BPC: {infinity_results['bpc']:.4f}")
        print(f"  Compression: {infinity_results['compression']:.2f} chars/token")
        print(f"  Total tokens: {infinity_results['total_tokens']:,}")
    else:
        print(f"Infinity v2 checkpoint not found: {infinity_checkpoint}")
        print("Run training first: modal run v2/train_infinity_v2.py\n")
    
    # =========================================================================
    # EVALUATE BPE BASELINE
    # =========================================================================
    
    bpe_checkpoint = "/checkpoints/bpe_baseline_final.pt"
    bpe_tokenizer_path = "/checkpoints/bpe_tokenizer.json"
    
    if os.path.exists(bpe_checkpoint) and os.path.exists(bpe_tokenizer_path):
        print("\n" + "-" * 70)
        print("EVALUATING BPE BASELINE")
        print("-" * 70)
        
        from tokenizers import Tokenizer
        
        bpe_tokenizer = Tokenizer.from_file(bpe_tokenizer_path)
        vocab_size = bpe_tokenizer.get_vocab_size()
        
        checkpoint = torch.load(bpe_checkpoint, map_location=device)
        config = checkpoint.get('config', {'d_model': 512, 'n_layers': 8, 'n_heads': 8})
        
        model = BPEModel(
            vocab_size=vocab_size,
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print(f"Model loaded: {config}")
        print(f"Vocab size: {vocab_size}")
        
        bpe_results = compute_bpc_bpe(model, bpe_tokenizer, test_texts)
        results['bpe'] = bpe_results
        
        print(f"\nBPE Results:")
        print(f"  BPC: {bpe_results['bpc']:.4f}")
        print(f"  Compression: {bpe_results['compression']:.2f} chars/token")
        print(f"  Total tokens: {bpe_results['total_tokens']:,}")
    else:
        print(f"\nBPE checkpoint not found: {bpe_checkpoint}")
        print("Run training first: modal run baseline/train_bpe_baseline.py\n")
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    if 'infinity_v2' in results and 'bpe' in results:
        inf = results['infinity_v2']
        bpe = results['bpe']
        
        print(f"\n{'Metric':<25} {'Infinity v2':>15} {'BPE':>15} {'Diff':>15}")
        print("-" * 70)
        print(f"{'BPC (lower is better)':<25} {inf['bpc']:>15.4f} {bpe['bpc']:>15.4f} {inf['bpc'] - bpe['bpc']:>+15.4f}")
        print(f"{'Compression (chars/tok)':<25} {inf['compression']:>15.2f} {bpe['compression']:>15.2f} {inf['compression'] - bpe['compression']:>+15.2f}")
        print(f"{'Total tokens':<25} {inf['total_tokens']:>15,} {bpe['total_tokens']:>15,} {inf['total_tokens'] - bpe['total_tokens']:>+15,}")
        
        # Winner determination
        print("\n" + "-" * 70)
        if inf['bpc'] < bpe['bpc']:
            improvement = (bpe['bpc'] - inf['bpc']) / bpe['bpc'] * 100
            print(f"WINNER: Infinity v2 ({improvement:.1f}% better BPC)")
        elif bpe['bpc'] < inf['bpc']:
            improvement = (inf['bpc'] - bpe['bpc']) / inf['bpc'] * 100
            print(f"WINNER: BPE ({improvement:.1f}% better BPC)")
        else:
            print("TIE: Both models have equal BPC")
        
        # Compression analysis
        if inf['compression'] > bpe['compression']:
            print(f"Infinity v2 achieves {inf['compression']/bpe['compression']:.2f}x better compression")
        
    elif 'infinity_v2' in results:
        print("\nOnly Infinity v2 results available.")
        print(f"BPC: {results['infinity_v2']['bpc']:.4f}")
    elif 'bpe' in results:
        print("\nOnly BPE results available.")
        print(f"BPC: {results['bpe']['bpc']:.4f}")
    else:
        print("\nNo models evaluated. Train models first:")
        print("  modal run v2/train_infinity_v2.py")
        print("  modal run baseline/train_bpe_baseline.py")
    
    return results


@app.local_entrypoint()
def main():
    """Run BPC comparison."""
    results = compare_bpc.remote()
    print(f"\nFinal results: {results}")
