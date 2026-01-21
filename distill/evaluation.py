"""
Evaluation utilities for Frontier Braille Model Corpus.

Supports:
1. Macro masking: Mask learned contractions and measure loss increase
2. Held-out composition: Measure loss on unseen feature combinations
3. Surface re-encoding: Measure loss on perturbed variants
4. Ablation tests: Remove specific morpheme types and measure impact
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EvaluationResult:
    """Result of an evaluation test."""
    test_name: str
    metric: str
    value: float
    description: str
    hypothesis: str


class CorpusEvaluator:
    """
    Evaluation framework for Frontier Braille Model Corpus.
    """
    
    def __init__(self, corpus_path: str):
        """
        Initialize evaluator with corpus JSONL file.
        
        Args:
            corpus_path: Path to JSONL corpus file
        """
        self.corpus_path = Path(corpus_path)
        self.samples = self._load_corpus()
    
    def _load_corpus(self) -> List[Dict]:
        """Load corpus from JSONL file."""
        samples = []
        with open(self.corpus_path, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def analyze_held_out_distribution(self) -> Dict:
        """
        Analyze the distribution of held-out samples.
        
        Returns:
            Dictionary with held-out statistics
        """
        held_out = [s for s in self.samples if s.get('held_out', False)]
        non_held_out = [s for s in self.samples if not s.get('held_out', False)]
        
        # Group by structural type
        held_out_by_type = defaultdict(list)
        for sample in held_out:
            held_out_by_type[sample['structural_type']].append(sample)
        
        return {
            "total_held_out": len(held_out),
            "total_training": len(non_held_out),
            "held_out_ratio": len(held_out) / len(self.samples) if self.samples else 0,
            "held_out_by_type": {k: len(v) for k, v in held_out_by_type.items()},
            "held_out_samples": held_out,
        }
    
    def analyze_morphological_coverage(self) -> Dict:
        """
        Analyze morphological coverage across types.
        
        Returns:
            Dictionary with morphology statistics
        """
        morphology_types = {
            "concatenative": [],
            "agglutinative": [],
            "fusional": [],
            "templatic": [],
        }
        
        for sample in self.samples:
            stype = sample.get('structural_type', '')
            for morph_type in morphology_types:
                if morph_type in stype:
                    morphology_types[morph_type].append(sample)
        
        return {
            "morphology_distribution": {k: len(v) for k, v in morphology_types.items()},
            "morphology_samples": morphology_types,
        }
    
    def analyze_english_stressor_coverage(self) -> Dict:
        """
        Analyze coverage of English structural stressors.
        
        Returns:
            Dictionary with stressor statistics
        """
        stressor_types = {
            "dialogue": [],
            "instruction": [],
            "logical": [],
            "temporal": [],
            "list": [],
        }
        
        for sample in self.samples:
            if sample.get('source_category') == 'english_stressor':
                stype = sample.get('structural_type', '')
                for stressor in stressor_types:
                    if stressor in stype:
                        stressor_types[stressor].append(sample)
        
        return {
            "stressor_distribution": {k: len(v) for k, v in stressor_types.items()},
            "stressor_samples": stressor_types,
        }
    
    def analyze_perturbation_coverage(self) -> Dict:
        """
        Analyze coverage of adversarial perturbations.
        
        Returns:
            Dictionary with perturbation statistics
        """
        perturbation_types = {
            "tokenization": [],
            "punctuation": [],
            "entities": [],
            "phrase_order": [],
        }
        
        for sample in self.samples:
            if sample.get('source_category') == 'adversarial':
                metadata = sample.get('metadata', {})
                ptype = metadata.get('perturbation_type', '')
                if ptype in perturbation_types:
                    perturbation_types[ptype].append(sample)
        
        return {
            "perturbation_distribution": {k: len(v) for k, v in perturbation_types.items()},
            "perturbation_samples": perturbation_types,
        }
    
    def analyze_variant_pairs(self) -> Dict:
        """
        Analyze samples with variants (for surface re-encoding tests).
        
        Returns:
            Dictionary with variant pair statistics
        """
        variants = [s for s in self.samples if s.get('has_variants', False)]
        
        # Group by base sample
        variant_groups = defaultdict(list)
        for sample in self.samples:
            if sample.get('source_category') == 'adversarial':
                base_id = sample.get('metadata', {}).get('base_sample_id')
                if base_id:
                    variant_groups[base_id].append(sample)
        
        return {
            "total_samples_with_variants": len(variants),
            "total_variant_pairs": len(variant_groups),
            "variant_pair_sizes": {
                k: len(v) for k, v in variant_groups.items()
            },
        }
    
    def generate_evaluation_report(self) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        report = {
            "corpus_summary": {
                "total_samples": len(self.samples),
                "source_categories": list(set(s.get('source_category') for s in self.samples)),
                "structural_types": list(set(s.get('structural_type') for s in self.samples)),
            },
            "held_out_analysis": self.analyze_held_out_distribution(),
            "morphology_analysis": self.analyze_morphological_coverage(),
            "english_stressor_analysis": self.analyze_english_stressor_coverage(),
            "perturbation_analysis": self.analyze_perturbation_coverage(),
            "variant_analysis": self.analyze_variant_pairs(),
        }
        
        return report
    
    def identify_test_cases(self) -> Dict:
        """
        Identify specific test cases for model evaluation.
        
        Returns:
            Dictionary with organized test cases
        """
        test_cases = {
            "macro_masking_candidates": [],
            "held_out_composition_tests": [],
            "surface_reencoding_tests": [],
            "ablation_test_candidates": [],
        }
        
        # Macro masking candidates: synthetic morphology with clear structure
        for sample in self.samples:
            if sample.get('source_category') == 'synthetic_morphology':
                test_cases["macro_masking_candidates"].append(sample)
        
        # Held-out composition tests: held-out samples
        for sample in self.samples:
            if sample.get('held_out'):
                test_cases["held_out_composition_tests"].append(sample)
        
        # Surface re-encoding tests: adversarial samples
        for sample in self.samples:
            if sample.get('source_category') == 'adversarial':
                test_cases["surface_reencoding_tests"].append(sample)
        
        # Ablation test candidates: samples with clear latent structure
        for sample in self.samples:
            if sample.get('latent_structure'):
                test_cases["ablation_test_candidates"].append(sample)
        
        return test_cases


class EvaluationReportGenerator:
    """
    Generate human-readable evaluation reports.
    """
    
    def __init__(self, evaluator: CorpusEvaluator):
        self.evaluator = evaluator
    
    def generate_markdown_report(self, output_path: str):
        """
        Generate Markdown evaluation report.
        
        Args:
            output_path: Path to save report
        """
        report = self.evaluator.generate_evaluation_report()
        test_cases = self.evaluator.identify_test_cases()
        
        lines = [
            "# Frontier Braille Model Corpus: Evaluation Report",
            "",
            "## Executive Summary",
            "",
            f"- **Total Samples**: {report['corpus_summary']['total_samples']}",
            f"- **Source Categories**: {len(report['corpus_summary']['source_categories'])}",
            f"- **Structural Types**: {len(report['corpus_summary']['structural_types'])}",
            "",
            "## Held-Out Analysis",
            "",
            f"- **Total Held-Out Samples**: {report['held_out_analysis']['total_held_out']}",
            f"- **Training Samples**: {report['held_out_analysis']['total_training']}",
            f"- **Held-Out Ratio**: {report['held_out_analysis']['held_out_ratio']:.2%}",
            "",
            "### Held-Out by Structural Type",
            "",
        ]
        
        for stype, count in report['held_out_analysis']['held_out_by_type'].items():
            lines.append(f"- {stype}: {count}")
        
        lines.extend([
            "",
            "## Morphological Coverage",
            "",
        ])
        
        for morph_type, count in report['morphology_analysis']['morphology_distribution'].items():
            pct = 100 * count / report['corpus_summary']['total_samples']
            lines.append(f"- {morph_type}: {count} ({pct:.1f}%)")
        
        lines.extend([
            "",
            "## English Stressor Coverage",
            "",
        ])
        
        for stressor, count in report['english_stressor_analysis']['stressor_distribution'].items():
            if count > 0:
                lines.append(f"- {stressor}: {count}")
        
        lines.extend([
            "",
            "## Adversarial Perturbation Coverage",
            "",
        ])
        
        for ptype, count in report['perturbation_analysis']['perturbation_distribution'].items():
            if count > 0:
                lines.append(f"- {ptype}: {count}")
        
        lines.extend([
            "",
            "## Variant Pair Analysis",
            "",
            f"- **Samples with Variants**: {report['variant_analysis']['total_samples_with_variants']}",
            f"- **Variant Pairs**: {report['variant_analysis']['total_variant_pairs']}",
            "",
            "## Test Case Identification",
            "",
            f"- **Macro Masking Candidates**: {len(test_cases['macro_masking_candidates'])}",
            f"- **Held-Out Composition Tests**: {len(test_cases['held_out_composition_tests'])}",
            f"- **Surface Re-Encoding Tests**: {len(test_cases['surface_reencoding_tests'])}",
            f"- **Ablation Test Candidates**: {len(test_cases['ablation_test_candidates'])}",
            "",
            "## Evaluation Protocol",
            "",
            "### 1. Macro Masking Test",
            "",
            "**Hypothesis**: If the model learned operators, loss should increase predictably when contractions are masked.",
            "",
            "**Procedure**:",
            "1. Train model on corpus",
            "2. Identify learned contractions",
            "3. Mask contractions in test set",
            "4. Measure loss increase",
            "",
            "**Success Criteria**: Loss increase is proportional to contraction frequency and generality.",
            "",
            "### 2. Held-Out Composition Test",
            "",
            "**Hypothesis**: If the model learned structure, loss should be lower than random baseline on unseen feature combinations.",
            "",
            "**Procedure**:",
            "1. Train model on corpus (excluding held-out samples)",
            "2. Evaluate on held-out samples",
            "3. Compare to random baseline",
            "",
            "**Success Criteria**: Held-out loss is significantly lower than random baseline.",
            "",
            "### 3. Surface Re-Encoding Test",
            "",
            "**Hypothesis**: If the model learned structure, loss should be stable across surface perturbations.",
            "",
            "**Procedure**:",
            "1. Train model on corpus",
            "2. Evaluate on perturbed variants",
            "3. Compare to original samples",
            "",
            "**Success Criteria**: Loss difference between original and perturbed is minimal.",
            "",
            "### 4. Ablation Test",
            "",
            "**Hypothesis**: Impact of removing morpheme types should correlate with morpheme frequency and generality.",
            "",
            "**Procedure**:",
            "1. Train model on corpus",
            "2. Remove specific morpheme types",
            "3. Measure loss impact",
            "",
            "**Success Criteria**: Impact correlates with morpheme statistics.",
            "",
        ])
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Frontier Braille Model Corpus"
    )
    parser.add_argument(
        "corpus_path",
        help="Path to corpus JSONL file"
    )
    parser.add_argument(
        "--output-report",
        default="evaluation_report.md",
        help="Output path for evaluation report"
    )
    parser.add_argument(
        "--output-json",
        help="Output path for JSON evaluation data"
    )
    
    args = parser.parse_args()
    
    # Load and evaluate corpus
    evaluator = CorpusEvaluator(args.corpus_path)
    report = evaluator.generate_evaluation_report()
    
    # Generate Markdown report
    report_gen = EvaluationReportGenerator(evaluator)
    report_gen.generate_markdown_report(args.output_report)
    
    print(f"[+] Evaluation report saved to {args.output_report}")
    
    # Save JSON report if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[+] JSON report saved to {args.output_json}")
