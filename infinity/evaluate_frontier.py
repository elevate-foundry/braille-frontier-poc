"""
Evaluate trained model on frontier corpus adversarial and variant splits.

Tests structure learning vs memorization:
- eval_variants: Surface re-encodings (same meaning, different tokens)
- eval_adversarial: Perturbed samples (tokenization, punctuation, entities, phrase order)

If the model learned structure, loss should be similar on these splits.
If the model memorized, loss will be much higher.
"""

import modal

app = modal.App("braille-frontier-eval")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "numpy",
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
def evaluate_splits():
    """Evaluate model on train, variant, and adversarial splits."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from typing import Dict, List, Tuple
    from dataclasses import dataclass
    
    # =========================================================================
    # INLINE MODEL DEFINITION (matching train_infinity.py exactly)
    # =========================================================================
    
    @dataclass
    class TokenLayer:
        layer: int
        components: List[int]
        frequency: int
        compression_ratio: float
    
    class InfinityVocabulary:
        LAYER1_END = 256
        LAYER2_END = 4096
        PAD_ID = 0
        BOS_ID = 256
        EOS_ID = 257
        UNK_ID = 258
        
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
        def __init__(self, d_model, n_heads, dropout=0.0):
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
        def __init__(self, d_model, d_ff=None, dropout=0.0):
            super().__init__()
            d_ff = d_ff or d_model * 4
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.0):
            super().__init__()
            self.attn_norm = RMSNorm(d_model)
            self.attn = Attention(d_model, n_heads, dropout)
            self.ff_norm = RMSNorm(d_model)
            self.ff = FeedForward(d_model, None, dropout)
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
            nn.init.normal_(self.residual.weight, std=0.02)
            self.gate = nn.Embedding(vocab.max_vocab_size, 1)
            with torch.no_grad():
                self.gate.weight[:256] = 1.0
                self.gate.weight[256:4096] = 0.5
                self.gate.weight[4096:] = 0.0
            
            self.register_buffer("mask_table", build_mask_table(vocab, vocab.max_vocab_size))
            
            self.embed_dropout = nn.Dropout(dropout)
            self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
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
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    
    device = torch.device("cuda")
    print("=" * 60)
    print("FRONTIER CORPUS EVALUATION")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Load checkpoint
    print("\nLoading trained model...")
    checkpoint = torch.load("/checkpoints/infinity_final.pt", map_location=device)
    
    # Reconstruct vocabulary
    vocab = InfinityVocabulary()
    contractions_dict = checkpoint.get("contractions", {})
    for pattern_str, token_id in contractions_dict.items():
        pattern = tuple(eval(pattern_str))
        vocab.contractions[pattern] = token_id
        vocab.token_info[token_id] = TokenLayer(
            layer=2, components=list(pattern), frequency=0, compression_ratio=len(pattern)
        )
    vocab.current_size = checkpoint["vocab_size"]
    print(f"  Vocabulary size: {vocab.current_size}")
    print(f"  Contractions: {len(vocab.contractions)}")
    
    # Build model - need to strip _orig_mod. prefix from compiled model checkpoint
    state_dict = checkpoint["model"]
    # Remove _orig_mod. prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v
    
    # Build model matching the training architecture
    model = InfinityModel(vocab).to(device)
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Load datasets
    print("\nLoading evaluation splits...")
    
    def load_split(path):
        try:
            data = torch.load(path)
            return data["input_ids"].to(device), data["target_ids"].to(device)
        except Exception as e:
            print(f"  Could not load {path}: {e}")
            return None, None
    
    train_inp, train_tgt = load_split("/data/frontier_train.pt")
    var_inp, var_tgt = load_split("/data/frontier_eval_variants.pt")
    adv_inp, adv_tgt = load_split("/data/frontier_eval_adversarial.pt")
    
    print(f"  Train: {train_inp.shape if train_inp is not None else 'N/A'}")
    print(f"  Variants: {var_inp.shape if var_inp is not None else 'N/A'}")
    print(f"  Adversarial: {adv_inp.shape if adv_inp is not None else 'N/A'}")
    
    # Evaluate function
    def evaluate(input_ids, target_ids, name, batch_size=64):
        if input_ids is None:
            return None
        
        dataset = TensorDataset(input_ids, target_ids)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for inp, tgt in loader:
                logits = model(inp)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt.view(-1),
                    ignore_index=0,
                    reduction='sum'
                )
                non_pad = (tgt != 0).sum().item()
                total_loss += loss.item()
                total_tokens += non_pad
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"  {name}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}, Tokens={total_tokens:,}")
        return {"loss": avg_loss, "perplexity": perplexity, "tokens": total_tokens}
    
    # Run evaluations
    print("\n" + "-" * 60)
    print("EVALUATION RESULTS")
    print("-" * 60)
    
    results = {}
    results["train"] = evaluate(train_inp, train_tgt, "Train")
    results["variants"] = evaluate(var_inp, var_tgt, "Variants")
    results["adversarial"] = evaluate(adv_inp, adv_tgt, "Adversarial")
    
    # Analysis
    print("\n" + "-" * 60)
    print("STRUCTURE VS MEMORIZATION ANALYSIS")
    print("-" * 60)
    
    if results["train"] and results["variants"]:
        var_ratio = results["variants"]["loss"] / results["train"]["loss"]
        print(f"\nVariant/Train loss ratio: {var_ratio:.2f}x")
        if var_ratio < 1.5:
            print("  ✓ Model generalizes well to surface re-encodings")
        else:
            print("  ✗ Model may be memorizing surface patterns")
    
    if results["train"] and results["adversarial"]:
        adv_ratio = results["adversarial"]["loss"] / results["train"]["loss"]
        print(f"\nAdversarial/Train loss ratio: {adv_ratio:.2f}x")
        if adv_ratio < 2.0:
            print("  ✓ Model is robust to adversarial perturbations")
        else:
            print("  ✗ Model is sensitive to perturbations (possible memorization)")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return results


@app.local_entrypoint()
def main():
    print("Evaluating Infinity Model on Frontier Corpus splits...")
    results = evaluate_splits.remote()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split, data in results.items():
        if data:
            print(f"  {split}: Loss={data['loss']:.4f}, PPL={data['perplexity']:.2f}")
