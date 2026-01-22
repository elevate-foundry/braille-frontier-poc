"""
Compute Bits Per Character (BPC) for fair comparison across tokenization schemes.

BPC = (cross_entropy_loss * num_tokens) / (num_characters * log(2))

This normalizes for different compression ratios, giving a fair comparison
of model quality regardless of tokenization.
"""

import modal

app = modal.App("braille-compute-bpc")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
    "tokenizers>=0.15.0",
)

volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("braille-training-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={
        "/checkpoints": volume,
        "/data": data_volume,
    },
)
def compute_bpc():
    """Compute BPC for both Infinity and BPE models."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import json
    import math
    from typing import Dict, List, Tuple
    from dataclasses import dataclass
    
    device = torch.device("cuda")
    
    print("=" * 60)
    print("BITS PER CHARACTER (BPC) EVALUATION")
    print("=" * 60)
    print("This provides a fair comparison across tokenization schemes.\n")
    
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
    
    # Infinity Model
    @dataclass
    class TokenLayer:
        layer: int
        components: List[int]
        frequency: int
        compression_ratio: float
    
    class InfinityVocabulary:
        def __init__(self, max_vocab_size: int = 16384):
            self.max_vocab_size = max_vocab_size
            self.current_size = 259
            self.token_info: Dict[int, TokenLayer] = {}
            for i in range(256):
                self.token_info[i] = TokenLayer(layer=1, components=[i], frequency=0, compression_ratio=1.0)
            self.contractions: Dict[Tuple[int, ...], int] = {}
    
    def build_mask_table(vocab, max_size):
        table = torch.zeros(max_size, 8)
        for i in range(256):
            for bit in range(8):
                table[i, bit] = (i >> bit) & 1
        for pattern, token_id in vocab.contractions.items():
            masks = torch.zeros(8)
            for comp in pattern:
                if comp < 256:
                    for bit in range(8):
                        masks[bit] += (comp >> bit) & 1
            table[token_id] = masks / len(pattern)
        return table
    
    class InfinityModel(nn.Module):
        def __init__(self, vocab, d_model=512, n_layers=12, n_heads=8):
            super().__init__()
            self.vocab = vocab
            self.d_model = d_model
            self.geom_proj = nn.Linear(8, d_model, bias=False)
            self.residual = nn.Embedding(vocab.max_vocab_size, d_model)
            self.gate = nn.Embedding(vocab.max_vocab_size, 1)
            self.register_buffer("mask_table", build_mask_table(vocab, vocab.max_vocab_size))
            self.embed_dropout = nn.Dropout(0.0)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab.max_vocab_size, bias=False)
            self.head.weight = self.residual.weight
        
        def forward(self, input_ids):
            masks = self.mask_table[input_ids]
            geom = self.geom_proj(masks)
            res = self.residual(input_ids)
            g = torch.sigmoid(self.gate(input_ids))
            x = res + g * geom
            x = self.embed_dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.head(x)
    
    class BPEModel(nn.Module):
        def __init__(self, vocab_size, d_model=512, n_layers=12, n_heads=8):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.embed_dropout = nn.Dropout(0.0)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
            self.norm = RMSNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.head.weight = self.embedding.weight
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.embed_dropout(x)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.head(x)
    
    # =========================================================================
    # LOAD RAW TEXT FOR CHARACTER COUNTING
    # =========================================================================
    
    print("Loading raw text for character counting...")
    with open("/data/frontier_train_texts.json", "r") as f:
        train_texts = json.load(f)
    
    total_chars = sum(len(t) for t in train_texts)
    print(f"  Total characters: {total_chars:,}")
    
    # =========================================================================
    # LOAD INFINITY MODEL
    # =========================================================================
    
    print("\nLoading Infinity model...")
    infinity_ckpt = torch.load("/checkpoints/infinity_final.pt", map_location="cpu")
    
    vocab = InfinityVocabulary()
    for pattern, token_id in infinity_ckpt["contractions"].items():
        if isinstance(pattern, str):
            pattern = tuple(int(x) for x in pattern.strip("()").split(", "))
        vocab.contractions[pattern] = token_id
        vocab.current_size = max(vocab.current_size, token_id + 1)
    
    infinity_model = InfinityModel(vocab).to(device)
    state_dict = infinity_ckpt["model"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    infinity_model.load_state_dict(new_state_dict, strict=False)
    infinity_model.eval()
    print(f"  Vocab size: {vocab.current_size}")
    
    # Re-tokenize data with learned contractions
    def text_to_braille_tokens(text):
        """Convert text to Braille tokens (Layer 1)."""
        tokens = [256]  # BOS
        for char in text:
            tokens.append(ord(char) % 256)
        tokens.append(257)  # EOS
        return tokens
    
    def retokenize_with_contractions(tokens, contractions):
        """Apply learned contractions to compress token sequence."""
        if not contractions:
            return tokens
        
        # Sort by length (longest first) for greedy matching
        sorted_contractions = sorted(
            contractions.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        result = []
        i = 0
        while i < len(tokens):
            matched = False
            for pattern, token_id in sorted_contractions:
                plen = len(pattern)
                if i + plen <= len(tokens):
                    if tuple(tokens[i:i+plen]) == pattern:
                        result.append(token_id)
                        i += plen
                        matched = True
                        break
            if not matched:
                result.append(tokens[i])
                i += 1
        return result
    
    print("  Re-tokenizing with learned contractions...")
    all_infinity_tokens = []
    for text in train_texts:
        base_tokens = text_to_braille_tokens(text)
        compressed = retokenize_with_contractions(base_tokens, vocab.contractions)
        all_infinity_tokens.extend(compressed)
    
    # Create sequences
    max_len = 512
    infinity_sequences = []
    for i in range(0, len(all_infinity_tokens) - max_len, max_len // 2):
        chunk = all_infinity_tokens[i:i + max_len]
        if len(chunk) < max_len:
            chunk = chunk + [0] * (max_len - len(chunk))
        infinity_sequences.append(chunk)
    
    infinity_ids = torch.tensor(infinity_sequences, dtype=torch.long)
    infinity_input = infinity_ids[:, :-1].to(device)
    infinity_target = infinity_ids[:, 1:].to(device)
    infinity_tokens = (infinity_target != 0).sum().item()
    print(f"  Total tokens (after compression): {infinity_tokens:,}")
    print(f"  Compression: {total_chars / infinity_tokens:.2f} chars/token")
    
    # =========================================================================
    # LOAD BPE MODEL
    # =========================================================================
    
    print("\nLoading BPE model...")
    bpe_ckpt = torch.load("/checkpoints/bpe_final.pt", map_location="cpu")
    bpe_vocab_size = bpe_ckpt["vocab_size"]
    
    bpe_model = BPEModel(bpe_vocab_size).to(device)
    state_dict = bpe_ckpt["model"]
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    bpe_model.load_state_dict(new_state_dict, strict=False)
    bpe_model.eval()
    print(f"  Vocab size: {bpe_vocab_size}")
    
    # Tokenize data with BPE tokenizer
    from tokenizers import Tokenizer
    bpe_tokenizer = Tokenizer.from_file("/checkpoints/bpe_tokenizer.json")
    
    def tokenize_texts_bpe(texts, max_len=512):
        all_ids = []
        for text in texts:
            encoded = bpe_tokenizer.encode(text)
            ids = encoded.ids + [2]  # Add EOS
            all_ids.extend(ids)
        
        sequences = []
        for i in range(0, len(all_ids) - max_len, max_len // 2):
            chunk = all_ids[i:i + max_len]
            if len(chunk) < max_len:
                chunk = chunk + [0] * (max_len - len(chunk))
            sequences.append(chunk)
        return torch.tensor(sequences, dtype=torch.long)
    
    bpe_ids = tokenize_texts_bpe(train_texts)
    bpe_input = bpe_ids[:, :-1].to(device)
    bpe_target = bpe_ids[:, 1:].to(device)
    bpe_tokens = (bpe_target != 0).sum().item()
    print(f"  Total tokens: {bpe_tokens:,}")
    
    # =========================================================================
    # COMPUTE LOSSES
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("COMPUTING CROSS-ENTROPY LOSSES")
    print("=" * 60)
    
    @torch.no_grad()
    def compute_total_loss(model, input_ids, target_ids, vocab_size, batch_size=32):
        """Compute total cross-entropy loss (sum, not mean)."""
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(0, len(input_ids), batch_size):
            batch_input = input_ids[i:i+batch_size]
            batch_target = target_ids[i:i+batch_size]
            
            logits = model(batch_input)
            
            # Compute per-token loss
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                batch_target.view(-1),
                ignore_index=0,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += (batch_target != 0).sum().item()
        
        return total_loss, total_tokens
    
    print("\nInfinity model...")
    inf_total_loss, inf_total_tokens = compute_total_loss(
        infinity_model, infinity_input, infinity_target, vocab.max_vocab_size
    )
    inf_mean_loss = inf_total_loss / inf_total_tokens
    print(f"  Total loss: {inf_total_loss:.2f}")
    print(f"  Total tokens: {inf_total_tokens:,}")
    print(f"  Mean loss (per token): {inf_mean_loss:.4f}")
    
    print("\nBPE model...")
    bpe_total_loss, bpe_total_tokens = compute_total_loss(
        bpe_model, bpe_input, bpe_target, bpe_vocab_size
    )
    bpe_mean_loss = bpe_total_loss / bpe_total_tokens
    print(f"  Total loss: {bpe_total_loss:.2f}")
    print(f"  Total tokens: {bpe_total_tokens:,}")
    print(f"  Mean loss (per token): {bpe_mean_loss:.4f}")
    
    # =========================================================================
    # COMPUTE BITS PER CHARACTER
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("BITS PER CHARACTER (BPC)")
    print("=" * 60)
    
    # BPC = total_nats / (total_chars * ln(2))
    # where total_nats = sum of cross-entropy losses
    ln2 = math.log(2)
    
    inf_bpc = inf_total_loss / (total_chars * ln2)
    bpe_bpc = bpe_total_loss / (total_chars * ln2)
    
    print(f"\nFormula: BPC = total_cross_entropy / (total_chars × ln(2))")
    print(f"Total characters: {total_chars:,}")
    print()
    print(f"Infinity:")
    print(f"  Total CE loss (nats): {inf_total_loss:.2f}")
    print(f"  BPC: {inf_bpc:.4f}")
    print()
    print(f"BPE:")
    print(f"  Total CE loss (nats): {bpe_total_loss:.2f}")
    print(f"  BPC: {bpe_bpc:.4f}")
    
    # =========================================================================
    # PERPLEXITY PER BYTE
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PERPLEXITY METRICS")
    print("=" * 60)
    
    # Per-token perplexity
    inf_ppl_token = math.exp(inf_mean_loss)
    bpe_ppl_token = math.exp(bpe_mean_loss)
    
    # Per-character perplexity (2^BPC)
    inf_ppl_char = 2 ** inf_bpc
    bpe_ppl_char = 2 ** bpe_bpc
    
    print(f"\nPer-token perplexity:")
    print(f"  Infinity: {inf_ppl_token:.2f}")
    print(f"  BPE:      {bpe_ppl_token:.2f}")
    print()
    print(f"Per-character perplexity (2^BPC):")
    print(f"  Infinity: {inf_ppl_char:.4f}")
    print(f"  BPE:      {bpe_ppl_char:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("FAIR COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│ METRIC                    │ INFINITY    │ BPE         │ RATIO │
├─────────────────────────────────────────────────────────────┤
│ Mean Loss (per token)     │ {inf_mean_loss:>10.4f}  │ {bpe_mean_loss:>10.4f}  │ {bpe_mean_loss/inf_mean_loss:>5.2f}x │
│ Tokens                    │ {inf_total_tokens:>10,}  │ {bpe_total_tokens:>10,}  │ {bpe_total_tokens/inf_total_tokens:>5.2f}x │
│ Compression (chars/tok)   │ {total_chars/inf_total_tokens:>10.2f}  │ {total_chars/bpe_total_tokens:>10.2f}  │ {(total_chars/inf_total_tokens)/(total_chars/bpe_total_tokens):>5.2f}x │
├─────────────────────────────────────────────────────────────┤
│ BITS PER CHARACTER (BPC)  │ {inf_bpc:>10.4f}  │ {bpe_bpc:>10.4f}  │ {bpe_bpc/inf_bpc:>5.2f}x │
│ Per-char Perplexity       │ {inf_ppl_char:>10.4f}  │ {bpe_ppl_char:>10.4f}  │ {bpe_ppl_char/inf_ppl_char:>5.2f}x │
└─────────────────────────────────────────────────────────────┘

NOTE: BPC is the correct metric for comparing across tokenization schemes.
Lower BPC = better model (fewer bits needed to encode each character).
""")
    
    return {
        "infinity_bpc": inf_bpc,
        "bpe_bpc": bpe_bpc,
        "infinity_mean_loss": inf_mean_loss,
        "bpe_mean_loss": bpe_mean_loss,
        "infinity_tokens": inf_total_tokens,
        "bpe_tokens": bpe_total_tokens,
        "total_chars": total_chars,
    }


@app.local_entrypoint()
def main():
    print("Computing Bits Per Character for fair comparison...")
    result = compute_bpc.remote()
    print(f"\n=== FINAL RESULT ===")
    print(f"Infinity BPC: {result['infinity_bpc']:.4f}")
    print(f"BPE BPC:      {result['bpe_bpc']:.4f}")
    print(f"Ratio:        {result['bpe_bpc']/result['infinity_bpc']:.2f}x")
