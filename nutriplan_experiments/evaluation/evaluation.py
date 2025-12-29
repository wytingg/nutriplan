"""
Main Evaluation Script for NutriPlan
Evaluates model outputs on all 8 metrics and generates comprehensive reports
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import NutriPlanMetrics, evaluate_batch
import matplotlib.pyplot as plt
import seaborn as sns


class NutriPlanEvaluator:
    """Main evaluator for NutriPlan experiments"""

    def __init__(
        self,
        predictions_file: str,
        references_file: str,
        constraints_file: str,
        kg_facts_file: str = None,
        output_dir: str = "results"
    ):
        """
        Args:
            predictions_file: JSONL file with model predictions
            references_file: JSONL file with reference recipes
            constraints_file: JSONL file with user constraints
            kg_facts_file: Optional JSONL file with KG facts per sample
            output_dir: Directory to save results
        """
        self.predictions_file = Path(predictions_file)
        self.references_file = Path(references_file)
        self.constraints_file = Path(constraints_file)
        self.kg_facts_file = Path(kg_facts_file) if kg_facts_file else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = NutriPlanMetrics()

        # Load data
        self.predictions = self._load_jsonl(self.predictions_file)
        self.references = self._load_jsonl(self.references_file)
        self.constraints = self._load_jsonl(self.constraints_file)
        self.kg_facts = self._load_jsonl(self.kg_facts_file) if self.kg_facts_file else [[] for _ in self.predictions]

        print(f"Loaded {len(self.predictions)} predictions")
        print(f"Loaded {len(self.references)} references")
        print(f"Loaded {len(self.constraints)} constraints")

        # Validate lengths
        assert len(self.predictions) == len(self.references) == len(self.constraints), \
            "Predictions, references, and constraints must have the same length"

    def _load_jsonl(self, filepath: Path) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _save_jsonl(self, data: List[Dict], filepath: Path):
        """Save JSONL file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Evaluate all samples and return comprehensive results

        Returns:
            Dictionary with aggregate metrics and per-sample results
        """
        print("\n" + "="*80)
        print("Starting NutriPlan Evaluation")
        print("="*80)

        all_metrics = []
        all_generated_texts = [json.dumps(pred, ensure_ascii=False) for pred in self.predictions]

        # Evaluate each sample
        for i in tqdm(range(len(self.predictions)), desc="Evaluating samples"):
            pred = self.predictions[i]
            ref = self.references[i]
            constraints = self.constraints[i]
            kg_facts = self.kg_facts[i] if i < len(self.kg_facts) else []

            # Compute metrics for this sample
            metrics = self.metrics_calculator.compute_all_metrics(
                generated_recipe=pred,
                reference_recipe=ref,
                constraints=constraints,
                kg_facts=kg_facts,
                all_generated_texts=all_generated_texts
            )

            metrics['sample_id'] = i
            all_metrics.append(metrics)

        # Compute aggregate statistics
        results = self._compute_aggregate_stats(all_metrics)

        # Save results
        self._save_results(results, all_metrics)

        # Generate visualizations
        self._generate_visualizations(all_metrics)

        return results

    def _compute_aggregate_stats(self, all_metrics: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics across all samples"""
        metric_names = ['sncr', 'upm', 'k_faith', 'avc', 'dist_2', 'bleu', 'rouge_l', 'nutrition_accuracy']

        aggregate_stats = {}

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]

            aggregate_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }

        # Special handling for AVC (lower is better)
        aggregate_stats['avc']['mean_violations'] = aggregate_stats['avc']['mean']

        return aggregate_stats

    def _save_results(self, aggregate_stats: Dict, per_sample_metrics: List[Dict]):
        """Save results to files"""

        # 1. Save aggregate statistics (JSON)
        aggregate_file = self.output_dir / "aggregate_metrics.json"
        with open(aggregate_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate_stats, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Aggregate metrics saved to: {aggregate_file}")

        # 2. Save per-sample metrics (CSV)
        per_sample_file = self.output_dir / "per_sample_metrics.csv"
        df = pd.DataFrame(per_sample_metrics)
        df.to_csv(per_sample_file, index=False)
        print(f"✅ Per-sample metrics saved to: {per_sample_file}")

        # 3. Save per-sample metrics (JSONL for detailed analysis)
        per_sample_jsonl = self.output_dir / "per_sample_metrics.jsonl"
        self._save_jsonl(per_sample_metrics, per_sample_jsonl)
        print(f"✅ Per-sample metrics (JSONL) saved to: {per_sample_jsonl}")

        # 4. Generate summary table (for paper)
        self._generate_summary_table(aggregate_stats)

    def _generate_summary_table(self, aggregate_stats: Dict):
        """Generate LaTeX-style summary table for paper"""

        table_file = self.output_dir / "summary_table.txt"

        with open(table_file, 'w', encoding='utf-8') as f:
            f.write("NutriPlan Evaluation Results Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'Median':<12}\n")
            f.write("-" * 80 + "\n")

            # Core metrics (higher is better except AVC)
            for metric in ['sncr', 'upm', 'k_faith']:
                stats = aggregate_stats[metric]
                f.write(f"{metric.upper():<25} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            # AVC (lower is better)
            avc_stats = aggregate_stats['avc']
            f.write(f"{'AVC (↓)':<25} {avc_stats['mean']:.4f}      {avc_stats['std']:.4f}      {avc_stats['median']:.4f}\n")

            # Text quality metrics
            for metric in ['dist_2', 'bleu', 'rouge_l', 'nutrition_accuracy']:
                stats = aggregate_stats[metric]
                f.write(f"{metric.upper():<25} {stats['mean']:.4f}      {stats['std']:.4f}      {stats['median']:.4f}\n")

            f.write("-" * 80 + "\n")

        print(f"✅ Summary table saved to: {table_file}")

        # Also print to console
        print("\n" + "="*80)
        print("Evaluation Results Summary")
        print("="*80)
        with open(table_file, 'r') as f:
            print(f.read())

    def _generate_visualizations(self, per_sample_metrics: List[Dict]):
        """Generate visualization plots"""

        df = pd.DataFrame(per_sample_metrics)

        # 1. Distribution plots for all metrics
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        metrics = ['sncr', 'upm', 'k_faith', 'avc', 'dist_2', 'bleu', 'rouge_l', 'nutrition_accuracy']

        for i, metric in enumerate(metrics):
            axes[i].hist(df[metric], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Distribution', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(metric.upper())
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(df[metric].mean(), color='red', linestyle='--', label=f'Mean: {df[metric].mean():.3f}')
            axes[i].legend()

        plt.tight_layout()
        dist_plot_file = self.output_dir / "metrics_distribution.png"
        plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Distribution plot saved to: {dist_plot_file}")
        plt.close()

        # 2. Box plot comparison
        fig, ax = plt.subplots(figsize=(14, 6))
        df[metrics].boxplot(ax=ax)
        ax.set_title('Metrics Comparison (Box Plot)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticklabels([m.upper() for m in metrics], rotation=45)
        ax.grid(axis='y', alpha=0.3)

        boxplot_file = self.output_dir / "metrics_boxplot.png"
        plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
        print(f"✅ Box plot saved to: {boxplot_file}")
        plt.close()

        # 3. Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Metrics Correlation Heatmap', fontsize=14, fontweight='bold')

        heatmap_file = self.output_dir / "metrics_correlation.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"✅ Correlation heatmap saved to: {heatmap_file}")
        plt.close()

    def identify_failure_cases(self, threshold: float = 0.5) -> List[Dict]:
        """
        Identify failure cases where SNCR < threshold

        Args:
            threshold: SNCR threshold for failure (default 0.5)

        Returns:
            List of failure cases with sample_id and metrics
        """
        per_sample_file = self.output_dir / "per_sample_metrics.csv"
        df = pd.read_csv(per_sample_file)

        failure_cases = df[df['sncr'] < threshold].to_dict('records')

        failure_file = self.output_dir / "failure_cases.jsonl"
        self._save_jsonl(failure_cases, failure_file)

        print(f"\n⚠️  Found {len(failure_cases)} failure cases (SNCR < {threshold})")
        print(f"✅ Failure cases saved to: {failure_file}")

        return failure_cases


def main():
    parser = argparse.ArgumentParser(description="Evaluate NutriPlan model predictions")

    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions JSONL file')
    parser.add_argument('--references', type=str, required=True,
                        help='Path to references JSONL file')
    parser.add_argument('--constraints', type=str, required=True,
                        help='Path to constraints JSONL file')
    parser.add_argument('--kg_facts', type=str, default=None,
                        help='Path to KG facts JSONL file (optional)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--failure_threshold', type=float, default=0.5,
                        help='SNCR threshold for identifying failure cases')

    args = parser.parse_args()

    # Create evaluator
    evaluator = NutriPlanEvaluator(
        predictions_file=args.predictions,
        references_file=args.references,
        constraints_file=args.constraints,
        kg_facts_file=args.kg_facts,
        output_dir=args.output_dir
    )

    # Run evaluation
    results = evaluator.evaluate_all()

    # Identify failure cases
    failure_cases = evaluator.identify_failure_cases(threshold=args.failure_threshold)

    print("\n" + "="*80)
    print("✅ Evaluation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
