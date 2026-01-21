"""
Inference demo for Braille Frontier Model

Demonstrates:
1. Loading a trained model
2. Generating text in Braille-space
3. Translating back to English
4. Comparing inference costs to standard models
"""

import torch
import time
from model import BrailleFrontierModel
from tokenizer import text_to_braille_ids, braille_ids_to_text, VOCAB_SIZE


def load_model(checkpoint_path: str = None, device: str = "cpu", from_modal: bool = False):
    """Load model from checkpoint or create fresh."""
    # Use larger config if loading from Modal-trained model
    if from_modal:
        model = BrailleFrontierModel(
            d_model=512,
            n_layers=12,
            n_heads=8,
            dropout=0.0,
        ).to(device)
    else:
        model = BrailleFrontierModel(
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.0,
        ).to(device)
    
    if checkpoint_path:
        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            # Handle torch.compile() prefix
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            # Load with strict=False to handle minor architecture differences
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            print("No checkpoint found, using random weights")
    
    model.eval()
    return model


def benchmark_inference(model, input_ids, n_runs: int = 100):
    """Benchmark inference speed."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids)
    
    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(input_ids)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


def compute_cost_comparison():
    """
    Compare theoretical compute costs between Braille and standard tokenization.
    """
    print("\n" + "=" * 60)
    print("COMPUTE COST COMPARISON")
    print("=" * 60)
    
    # Standard model assumptions
    std_vocab = 50000
    std_d_model = 4096
    std_seq_len = 512  # Tokens for a paragraph
    
    # Braille model assumptions
    braille_vocab = 259
    braille_d_model = 256  # Can be smaller due to simpler vocab
    braille_seq_len = 800  # Longer due to character-level, but Grade 2 helps
    
    # Embedding layer comparison
    std_embed_params = std_vocab * std_d_model
    braille_embed_params = braille_vocab * braille_d_model + 8 * braille_d_model  # + geometry proj
    
    print("\n1. EMBEDDING LAYER")
    print(f"   Standard: {std_vocab:,} × {std_d_model} = {std_embed_params:,} params")
    print(f"   Braille:  {braille_vocab:,} × {braille_d_model} + geom = {braille_embed_params:,} params")
    print(f"   Reduction: {(1 - braille_embed_params/std_embed_params)*100:.1f}%")
    
    # Attention cost (O(n²d) for standard attention)
    std_attn_flops = std_seq_len ** 2 * std_d_model
    braille_attn_flops = braille_seq_len ** 2 * braille_d_model
    
    print("\n2. ATTENTION (per layer)")
    print(f"   Standard: {std_seq_len}² × {std_d_model} = {std_attn_flops:,} FLOPs")
    print(f"   Braille:  {braille_seq_len}² × {braille_d_model} = {braille_attn_flops:,} FLOPs")
    print(f"   Reduction: {(1 - braille_attn_flops/std_attn_flops)*100:.1f}%")
    
    # KV Cache (memory bound at inference)
    std_kv_cache = std_seq_len * std_d_model * 2  # K and V
    braille_kv_cache = braille_seq_len * braille_d_model * 2
    
    print("\n3. KV CACHE (per layer)")
    print(f"   Standard: {std_seq_len} × {std_d_model} × 2 = {std_kv_cache:,} elements")
    print(f"   Braille:  {braille_seq_len} × {braille_d_model} × 2 = {braille_kv_cache:,} elements")
    print(f"   Reduction: {(1 - braille_kv_cache/std_kv_cache)*100:.1f}%")
    
    # Output projection (vocab size matters here)
    std_output_flops = std_seq_len * std_d_model * std_vocab
    braille_output_flops = braille_seq_len * braille_d_model * braille_vocab
    
    print("\n4. OUTPUT PROJECTION")
    print(f"   Standard: {std_seq_len} × {std_d_model} × {std_vocab:,} = {std_output_flops:,} FLOPs")
    print(f"   Braille:  {braille_seq_len} × {braille_d_model} × {braille_vocab} = {braille_output_flops:,} FLOPs")
    print(f"   Reduction: {(1 - braille_output_flops/std_output_flops)*100:.1f}%")
    
    print("\n" + "-" * 60)
    print("KEY INSIGHT:")
    print("  The Braille model trades sequence length for vocab/dimension size.")
    print("  Net win comes from:")
    print("    - Massively smaller embedding tables")
    print("    - Smaller KV cache (memory bandwidth is often the bottleneck)")
    print("    - Cheaper output projection")
    print("  The model 'thinks' in compressed Braille, then translates to English.")
    print("-" * 60)


def demo_generation(model, device):
    """Demo text generation."""
    print("\n" + "=" * 60)
    print("GENERATION DEMO")
    print("=" * 60)
    
    prompts = [
        "hello",
        "the quick",
        "braille is",
        "thinking",
    ]
    
    for prompt in prompts:
        input_ids = torch.tensor([text_to_braille_ids(prompt)], device=device)
        
        # Time generation
        start = time.perf_counter()
        output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
        elapsed = time.perf_counter() - start
        
        # Decode
        generated = braille_ids_to_text(output_ids[0].tolist())
        
        # Show Braille representation
        braille_str = ''.join(chr(0x2800 + i) if i < 256 else '?' for i in output_ids[0].tolist())
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Braille: {braille_str[:40]}...")
        print(f"English: '{generated}'")
        print(f"Time: {elapsed*1000:.1f}ms")


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model (try Modal-trained first, then local)
    import os
    if os.path.exists("checkpoints/model_final.pt"):
        model = load_model("checkpoints/model_final.pt", device, from_modal=True)
    else:
        model = load_model("braille_frontier_model.pt", device, from_modal=False)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Cost comparison
    compute_cost_comparison()
    
    # Generation demo
    demo_generation(model, device)
    
    # Benchmark
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK")
    print("=" * 60)
    
    test_input = torch.randint(0, 256, (1, 64))
    avg_time = benchmark_inference(model, test_input, n_runs=100)
    print(f"Average forward pass: {avg_time*1000:.2f}ms")
    print(f"Throughput: {1/avg_time:.0f} forward passes/sec")


if __name__ == "__main__":
    main()
