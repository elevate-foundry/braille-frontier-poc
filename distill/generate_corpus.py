#!/usr/bin/env python3
"""
Generate Frontier Braille Model Corpus in JSONL format with splits.

This script:
1. Generates a complete corpus using the FrontierBrailleCorpusGenerator
2. Splits into train/eval/adversarial subsets
3. Saves as JSONL files
4. Generates metadata and evaluation schema
"""

import sys
import json
from pathlib import Path
from corpus_generator import FrontierBrailleCorpusGenerator, CorpusSample


def generate_and_save_corpus(
    output_dir: str = "./corpus_output",
    total_samples: int = 5000,
    seed: int = 42,
    verbose: bool = True
):
    """
    Generate corpus and save to JSONL files with splits.
    
    Args:
        output_dir: Directory to save corpus files
        total_samples: Total number of samples to generate
        seed: Random seed for deterministic generation
        verbose: Print progress information
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"[*] Initializing corpus generator (seed={seed})...")
    
    generator = FrontierBrailleCorpusGenerator(seed=seed)
    
    if verbose:
        print(f"[*] Generating {total_samples} samples...")
    
    corpus = generator.generate_corpus(
        synthetic_ratio=0.25,
        english_ratio=0.45,
        multilingual_ratio=0.15,
        adversarial_ratio=0.15,
        total_samples=total_samples
    )
    
    if verbose:
        print(f"[+] Generated {len(corpus)} samples")
    
    # Organize into splits
    train_samples = [s for s in corpus if not s.held_out and s.source_category != "adversarial"]
    eval_held_out = [s for s in corpus if s.held_out]
    eval_variants = [s for s in corpus if s.has_variants and not s.held_out]
    eval_adversarial = [s for s in corpus if s.source_category == "adversarial"]
    
    if verbose:
        print(f"[*] Split breakdown:")
        print(f"    Train: {len(train_samples)} samples")
        print(f"    Eval (held-out): {len(eval_held_out)} samples")
        print(f"    Eval (variants): {len(eval_variants)} samples")
        print(f"    Eval (adversarial): {len(eval_adversarial)} samples")
    
    # Save splits to JSONL
    splits = {
        "train": train_samples,
        "eval_held_out": eval_held_out,
        "eval_variants": eval_variants,
        "eval_adversarial": eval_adversarial,
    }
    
    for split_name, samples in splits.items():
        filepath = output_path / f"{split_name}.jsonl"
        if verbose:
            print(f"[*] Saving {split_name} to {filepath}...")
        
        with open(filepath, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + '\n')
        
        if verbose:
            print(f"[+] Saved {len(samples)} samples to {filepath}")
    
    # Save combined corpus
    combined_path = output_path / "corpus_combined.jsonl"
    if verbose:
        print(f"[*] Saving combined corpus to {combined_path}...")
    
    with open(combined_path, 'w') as f:
        for sample in corpus:
            f.write(json.dumps(sample.to_dict()) + '\n')
    
    if verbose:
        print(f"[+] Saved combined corpus with {len(corpus)} samples")
    
    # Generate corpus statistics
    stats = {
        "total_samples": len(corpus),
        "seed": seed,
        "splits": {
            "train": len(train_samples),
            "eval_held_out": len(eval_held_out),
            "eval_variants": len(eval_variants),
            "eval_adversarial": len(eval_adversarial),
        },
        "source_distribution": {},
        "structural_types": {},
        "samples_with_variants": sum(1 for s in corpus if s.has_variants),
        "held_out_samples": sum(1 for s in corpus if s.held_out),
    }
    
    # Count source categories
    for sample in corpus:
        cat = sample.source_category
        stats["source_distribution"][cat] = stats["source_distribution"].get(cat, 0) + 1
        
        stype = sample.structural_type
        stats["structural_types"][stype] = stats["structural_types"].get(stype, 0) + 1
    
    stats_path = output_path / "corpus_stats.json"
    if verbose:
        print(f"[*] Saving statistics to {stats_path}...")
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if verbose:
        print(f"[+] Corpus generation complete!")
        print(f"\nStatistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Source distribution:")
        for cat, count in stats['source_distribution'].items():
            pct = 100 * count / stats['total_samples']
            print(f"    {cat}: {count} ({pct:.1f}%)")
        print(f"  Structural types: {len(stats['structural_types'])}")
        print(f"  Samples with variants: {stats['samples_with_variants']}")
        print(f"  Held-out samples: {stats['held_out_samples']}")
    
    return output_path, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Frontier Braille Model Corpus"
    )
    parser.add_argument(
        "--output-dir",
        default="./corpus_output",
        help="Output directory for corpus files"
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=5000,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    output_path, stats = generate_and_save_corpus(
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    print(f"\nCorpus saved to: {output_path}")
