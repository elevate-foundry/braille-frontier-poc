"""
Inference Benchmark: Infinity vs BPE

Measures:
1. Tokens per second (generation speed)
2. Time to generate N characters of text
3. Memory usage during generation
"""

import modal

app = modal.App("braille-inference-benchmark")

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
    timeout=1800,
    volumes={
        "/checkpoints": volume,
        "/data": data_volume,
    },
)
def benchmark_inference():
    """Benchmark inference speed for both models."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    import json
    from typing import Dict, List, Tuple
    from dataclasses import dataclass
    from tokenizers import Tokenizer
    
    device = torch.device("cuda")
    print("=" * 60)
    print("INFERENCE BENCHMARK: INFINITY vs BPE")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # =========================================================================
    # MODEL DEFINITIONS
    # =========================================================================
    
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
            self.shadow_tokens: Dict[Tuple[int, ...], int] = {}
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
    
    class InfinityModel(nn.Module):
        def __init__(self, vocab, d_model=512, n_layers=12, n_heads=8, dropout=0.1):
            super().__init__()
            self.vocab = vocab
            self.d_model = d_model
            self.geom_proj = nn.Linear(8, d_model, bias=False)
            self.residual = nn.Embedding(vocab.max_vocab_size, d_model)
            self.gate = nn.Embedding(vocab.max_vocab_size, 1)
            self.register_buffer("mask_table", build_mask_table(vocab, vocab.max_vocab_size))
            self.embed_dropout = nn.Dropout(dropout)
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
        def __init__(self, vocab_size, d_model=512, n_layers=12, n_heads=8, dropout=0.1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.embed_dropout = nn.Dropout(dropout)
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
    # LOAD MODELS
    # =========================================================================
    
    print("\nLoading Infinity model...")
    infinity_ckpt = torch.load("/checkpoints/infinity_final.pt", map_location=device)
    
    vocab = InfinityVocabulary()
    for pattern, token_id in infinity_ckpt["contractions"].items():
        # Convert string pattern back to tuple if needed
        if isinstance(pattern, str):
            pattern = tuple(int(x) for x in pattern.strip("()").split(", "))
        vocab.contractions[pattern] = token_id
        vocab.current_size = max(vocab.current_size, token_id + 1)
    
    infinity_model = InfinityModel(vocab).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in infinity_ckpt["model"].items()}
    infinity_model.load_state_dict(state_dict)
    infinity_model.eval()
    print(f"  Vocab size: {vocab.current_size}")
    
    print("\nLoading BPE model...")
    bpe_ckpt = torch.load("/checkpoints/bpe_final.pt", map_location=device)
    bpe_vocab_size = bpe_ckpt["vocab_size"]
    
    bpe_model = BPEModel(bpe_vocab_size).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in bpe_ckpt["model"].items()}
    bpe_model.load_state_dict(state_dict)
    bpe_model.eval()
    print(f"  Vocab size: {bpe_vocab_size}")
    
    # Load BPE tokenizer
    tokenizer = Tokenizer.from_file("/checkpoints/bpe_tokenizer.json")
    
    # =========================================================================
    # GENERATION FUNCTIONS
    # =========================================================================
    
    @torch.no_grad()
    def generate_infinity(model, prompt_tokens, max_new_tokens, temperature=0.8):
        """Generate tokens with Infinity model."""
        tokens = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -512:])  # Context window
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
    
    @torch.no_grad()
    def generate_bpe(model, prompt_tokens, max_new_tokens, temperature=0.8):
        """Generate tokens with BPE model."""
        tokens = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -512:])  # Context window
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
    
    # =========================================================================
    # BENCHMARK
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("BENCHMARK: Tokens per Second")
    print("=" * 60)
    
    # Test configurations
    gen_lengths = [50, 100, 200, 500]
    n_runs = 5
    
    results = {"infinity": {}, "bpe": {}}
    
    # Warmup
    print("\nWarming up...")
    warmup_prompt = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    _ = generate_infinity(infinity_model, warmup_prompt, 10)
    _ = generate_bpe(bpe_model, warmup_prompt, 10)
    torch.cuda.synchronize()
    
    for gen_len in gen_lengths:
        print(f"\nGenerating {gen_len} tokens...")
        
        # Infinity benchmark
        infinity_times = []
        for _ in range(n_runs):
            prompt = torch.tensor([[ord(c) for c in "Once upon"]], device=device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = generate_infinity(infinity_model, prompt, gen_len)
            torch.cuda.synchronize()
            infinity_times.append(time.perf_counter() - start)
        
        avg_infinity = sum(infinity_times) / len(infinity_times)
        infinity_tps = gen_len / avg_infinity
        results["infinity"][gen_len] = {"time": avg_infinity, "tps": infinity_tps}
        
        # BPE benchmark
        bpe_times = []
        for _ in range(n_runs):
            prompt_text = "Once upon"
            prompt_ids = tokenizer.encode(prompt_text).ids
            prompt = torch.tensor([prompt_ids], device=device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = generate_bpe(bpe_model, prompt, gen_len)
            torch.cuda.synchronize()
            bpe_times.append(time.perf_counter() - start)
        
        avg_bpe = sum(bpe_times) / len(bpe_times)
        bpe_tps = gen_len / avg_bpe
        results["bpe"][gen_len] = {"time": avg_bpe, "tps": bpe_tps}
        
        print(f"  Infinity: {infinity_tps:.1f} tok/s ({avg_infinity*1000:.1f}ms)")
        print(f"  BPE:      {bpe_tps:.1f} tok/s ({avg_bpe*1000:.1f}ms)")
        print(f"  Ratio:    {infinity_tps/bpe_tps:.2f}x")
    
    # =========================================================================
    # EFFECTIVE THROUGHPUT (chars per second)
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("EFFECTIVE THROUGHPUT: Characters per Second")
    print("=" * 60)
    
    infinity_compression = 2.19  # chars per token
    bpe_compression = 1.71  # chars per token
    
    print(f"\nCompression ratios:")
    print(f"  Infinity: {infinity_compression:.2f} chars/token")
    print(f"  BPE:      {bpe_compression:.2f} chars/token")
    
    print(f"\nEffective character throughput (for 200 tokens):")
    infinity_tps = results["infinity"][200]["tps"]
    bpe_tps = results["bpe"][200]["tps"]
    
    infinity_cps = infinity_tps * infinity_compression
    bpe_cps = bpe_tps * bpe_compression
    
    print(f"  Infinity: {infinity_cps:.1f} chars/s")
    print(f"  BPE:      {bpe_cps:.1f} chars/s")
    print(f"  Ratio:    {infinity_cps/bpe_cps:.2f}x")
    
    # =========================================================================
    # MEMORY USAGE
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("MEMORY USAGE")
    print("=" * 60)
    
    torch.cuda.reset_peak_memory_stats()
    
    # Infinity memory
    prompt = torch.tensor([[ord(c) for c in "Once upon"]], device=device)
    _ = generate_infinity(infinity_model, prompt, 200)
    infinity_mem = torch.cuda.max_memory_allocated() / 1e9
    
    torch.cuda.reset_peak_memory_stats()
    
    # BPE memory
    prompt_ids = tokenizer.encode("Once upon").ids
    prompt = torch.tensor([prompt_ids], device=device)
    _ = generate_bpe(bpe_model, prompt, 200)
    bpe_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"  Infinity: {infinity_mem:.2f} GB")
    print(f"  BPE:      {bpe_mem:.2f} GB")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_infinity_tps = sum(r["tps"] for r in results["infinity"].values()) / len(results["infinity"])
    avg_bpe_tps = sum(r["tps"] for r in results["bpe"].values()) / len(results["bpe"])
    
    print(f"\nAverage tokens/second:")
    print(f"  Infinity: {avg_infinity_tps:.1f}")
    print(f"  BPE:      {avg_bpe_tps:.1f}")
    print(f"  Ratio:    {avg_infinity_tps/avg_bpe_tps:.2f}x")
    
    print(f"\nEffective chars/second (accounting for compression):")
    print(f"  Infinity: {avg_infinity_tps * infinity_compression:.1f}")
    print(f"  BPE:      {avg_bpe_tps * bpe_compression:.1f}")
    print(f"  Ratio:    {(avg_infinity_tps * infinity_compression)/(avg_bpe_tps * bpe_compression):.2f}x")
    
    return {
        "infinity_tps": avg_infinity_tps,
        "bpe_tps": avg_bpe_tps,
        "infinity_cps": avg_infinity_tps * infinity_compression,
        "bpe_cps": avg_bpe_tps * bpe_compression,
        "infinity_mem": infinity_mem,
        "bpe_mem": bpe_mem,
    }


@app.local_entrypoint()
def main():
    print("Starting inference benchmark...")
    result = benchmark_inference.remote()
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Infinity: {result['infinity_tps']:.1f} tok/s, {result['infinity_cps']:.1f} chars/s")
    print(f"BPE:      {result['bpe_tps']:.1f} tok/s, {result['bpe_cps']:.1f} chars/s")
    print(f"Speedup:  {result['infinity_cps']/result['bpe_cps']:.2f}x effective throughput")
