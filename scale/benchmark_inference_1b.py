"""
1B+ Inference Benchmark: Infinity vs BPE

Measures tokens/second, characters/second, and memory usage at 1.6B scale.
"""

import modal

app = modal.App("braille-benchmark-1b")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
    "tokenizers>=0.15.0",
)

volume = modal.Volume.from_name("braille-checkpoints", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/checkpoints": volume},
)
def benchmark_inference_1b():
    """Benchmark inference speed for 1B+ models."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import time
    from typing import Dict, List, Tuple
    from dataclasses import dataclass
    
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')
    
    print("=" * 60)
    print("INFERENCE BENCHMARK: INFINITY vs BPE (1B+ Scale)")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # =========================================================================
    # MODEL DEFINITIONS (must match training scripts)
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
        def __init__(self, vocab, d_model=2048, n_layers=24, n_heads=16):
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
    
    # BPE Model
    class BPEModel(nn.Module):
        def __init__(self, vocab_size, d_model=2048, n_layers=24, n_heads=16):
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
    # LOAD MODELS
    # =========================================================================
    
    print("\nLoading Infinity 1B model...")
    # Use final checkpoint which has vocabulary expansion
    infinity_ckpt = torch.load("/checkpoints/infinity_1b_final.pt", map_location="cpu")
    
    vocab = InfinityVocabulary()
    for pattern, token_id in infinity_ckpt["contractions"].items():
        if isinstance(pattern, str):
            pattern = tuple(int(x) for x in pattern.strip("()").split(", "))
        vocab.contractions[pattern] = token_id
        vocab.current_size = max(vocab.current_size, token_id + 1)
    
    infinity_model = InfinityModel(vocab).to(device)
    
    # Load state dict, handling _orig_mod prefix
    state_dict = infinity_ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    infinity_model.load_state_dict(new_state_dict, strict=False)
    infinity_model.eval()
    infinity_compression = infinity_ckpt.get("compression", 1.68)
    print(f"  Vocab size: {vocab.current_size}")
    print(f"  Compression: {infinity_compression:.2f}x")
    
    print("\nLoading BPE 1B model...")
    bpe_ckpt = torch.load("/checkpoints/bpe_1b_best.pt", map_location="cpu")
    bpe_vocab_size = bpe_ckpt["vocab_size"]
    
    bpe_model = BPEModel(bpe_vocab_size).to(device)
    state_dict = bpe_ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    bpe_model.load_state_dict(new_state_dict, strict=False)
    bpe_model.eval()
    bpe_compression = bpe_ckpt.get("compression", 1.71)
    print(f"  Vocab size: {bpe_vocab_size}")
    print(f"  Compression: {bpe_compression:.2f}x")
    
    # =========================================================================
    # GENERATION FUNCTIONS
    # =========================================================================
    
    @torch.no_grad()
    def generate_infinity(model, start_tokens, max_new_tokens, temperature=1.0):
        tokens = start_tokens.clone()
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -512:])
            next_logits = logits[:, -1, :vocab.current_size] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
    
    @torch.no_grad()
    def generate_bpe(model, start_tokens, max_new_tokens, temperature=1.0):
        tokens = start_tokens.clone()
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -512:])
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
    
    # Warmup
    print("\nWarming up...")
    start_inf = torch.tensor([[256]], device=device)  # BOS
    start_bpe = torch.tensor([[1]], device=device)    # BOS
    
    for _ in range(3):
        _ = generate_infinity(infinity_model, start_inf, 10)
        _ = generate_bpe(bpe_model, start_bpe, 10)
    torch.cuda.synchronize()
    
    results = {"infinity": [], "bpe": []}
    
    for n_tokens in [50, 100, 200]:
        print(f"\nGenerating {n_tokens} tokens...")
        
        # Infinity
        torch.cuda.synchronize()
        start = time.time()
        _ = generate_infinity(infinity_model, start_inf, n_tokens)
        torch.cuda.synchronize()
        inf_time = time.time() - start
        inf_tps = n_tokens / inf_time
        results["infinity"].append(inf_tps)
        
        # BPE
        torch.cuda.synchronize()
        start = time.time()
        _ = generate_bpe(bpe_model, start_bpe, n_tokens)
        torch.cuda.synchronize()
        bpe_time = time.time() - start
        bpe_tps = n_tokens / bpe_time
        results["bpe"].append(bpe_tps)
        
        print(f"  Infinity: {inf_tps:.1f} tok/s ({inf_time*1000:.1f}ms)")
        print(f"  BPE:      {bpe_tps:.1f} tok/s ({bpe_time*1000:.1f}ms)")
        print(f"  Ratio:    {inf_tps/bpe_tps:.2f}x")
    
    # =========================================================================
    # EFFECTIVE THROUGHPUT
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("EFFECTIVE THROUGHPUT: Characters per Second")
    print("=" * 60)
    
    print(f"\nCompression ratios:")
    print(f"  Infinity: {infinity_compression:.2f} chars/token")
    print(f"  BPE:      {bpe_compression:.2f} chars/token")
    
    avg_inf_tps = sum(results["infinity"]) / len(results["infinity"])
    avg_bpe_tps = sum(results["bpe"]) / len(results["bpe"])
    
    inf_cps = avg_inf_tps * infinity_compression
    bpe_cps = avg_bpe_tps * bpe_compression
    
    print(f"\nEffective character throughput:")
    print(f"  Infinity: {inf_cps:.1f} chars/s")
    print(f"  BPE:      {bpe_cps:.1f} chars/s")
    print(f"  Ratio:    {inf_cps/bpe_cps:.2f}x")
    
    # =========================================================================
    # MEMORY USAGE
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("MEMORY USAGE")
    print("=" * 60)
    
    torch.cuda.reset_peak_memory_stats()
    _ = generate_infinity(infinity_model, start_inf, 100)
    torch.cuda.synchronize()
    inf_mem = torch.cuda.max_memory_allocated() / 1e9
    
    torch.cuda.reset_peak_memory_stats()
    _ = generate_bpe(bpe_model, start_bpe, 100)
    torch.cuda.synchronize()
    bpe_mem = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"  Infinity: {inf_mem:.2f} GB")
    print(f"  BPE:      {bpe_mem:.2f} GB")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SUMMARY (1B+ Scale)")
    print("=" * 60)
    
    print(f"\nAverage tokens/second:")
    print(f"  Infinity: {avg_inf_tps:.1f}")
    print(f"  BPE:      {avg_bpe_tps:.1f}")
    print(f"  Ratio:    {avg_inf_tps/avg_bpe_tps:.2f}x")
    
    print(f"\nEffective chars/second (accounting for compression):")
    print(f"  Infinity: {inf_cps:.1f}")
    print(f"  BPE:      {bpe_cps:.1f}")
    print(f"  Ratio:    {inf_cps/bpe_cps:.2f}x")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (1B+ Scale)")
    print("=" * 60)
    print(f"Infinity: {avg_inf_tps:.1f} tok/s, {inf_cps:.1f} chars/s")
    print(f"BPE:      {avg_bpe_tps:.1f} tok/s, {bpe_cps:.1f} chars/s")
    print(f"Speedup:  {inf_cps/bpe_cps:.2f}x effective throughput")
    
    return {
        "infinity_tps": avg_inf_tps,
        "bpe_tps": avg_bpe_tps,
        "infinity_cps": inf_cps,
        "bpe_cps": bpe_cps,
        "speedup": inf_cps / bpe_cps,
    }


@app.local_entrypoint()
def main():
    print("Starting 1B+ inference benchmark...")
    result = benchmark_inference_1b.remote()
    print(f"\nBenchmark complete!")
    print(f"Effective throughput speedup: {result['speedup']:.2f}x")
