"""
Aggregate RQ1 Results (Stage II)
Collects results from multiple base LLMs trained with multiple seeds
Generates Table X for the paper
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.statistical_tests import SignificanceTests, compare_model_results


def load_evaluation_results(eval_dir: Path) -> dict:
    """
    Load evaluation results from directory

    Args:
        eval_dir: Directory containing aggregate_metrics.json

    Returns:
        Evaluation metrics dict
    """
    metrics_file = eval_dir / "aggregate_metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Extract mean values for core metrics
    core_metrics = {}
    for metric_name in ['sncr', 'upm', 'k_faith', 'avc', 'dist_2', 'bleu', 'rouge_l', 'nutrition_accuracy']:
        if metric_name in metrics:
            core_metrics[metric_name] = metrics[metric_name]['mean']

    return core_metrics


def aggregate_multiseed_results(
    experiments_dir: Path,
    model_names: list,
    seeds: list = [42, 123, 2024]
) -> pd.DataFrame:
    """
    Aggregate results from multiple models and seeds

    Args:
        experiments_dir: Base experiments directory
        model_names: List of model names
        seeds: List of random seeds

    Returns:
        DataFrame with aggregated results
    """
    results = []

    for model_name in model_names:
        model_results = []

        print(f"\n Processing {model_name}...")

        for seed in seeds:
            # Find evaluation directory
            # Expected format: experiments_dir/rq1_{model_name}_seed_{seed}/eval
            model_dir = experiments_dir / f"rq1_{model_name.replace('/', '_')}_seed_{seed}"
            eval_dir = model_dir / "eval"

            if not eval_dir.exists():
                # Try alternative path
                eval_dir = experiments_dir / model_name.replace('/', '_') / f"seed_{seed}" / "eval"

            if not eval_dir.exists():
                print(f"  ⚠️  Evaluation directory not found for seed {seed}: {eval_dir}")
                continue

            try:
                metrics = load_evaluation_results(eval_dir)
                model_results.append(metrics)
                print(f"  ✓ Loaded seed {seed}")
            except Exception as e:
                print(f"  ❌ Failed to load seed {seed}: {e}")

        if not model_results:
            print(f"  ⚠️  No results found for {model_name}")
            continue

        # Compute statistics across seeds
        aggregated = {
            'Model': model_name,
            'N_seeds': len(model_results)
        }

        for metric_name in ['sncr', 'upm', 'k_faith', 'avc', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'rouge_l', 'dist_2', 'nutrition_accuracy']:
            values = [r[metric_name] for r in model_results if metric_name in r]

            if values:
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
                aggregated[f'{metric_name}_min'] = np.min(values)
                aggregated[f'{metric_name}_max'] = np.max(values)

        results.append(aggregated)

    df = pd.DataFrame(results)
    return df


def generate_table_x(
    df: pd.DataFrame,
    output_path: str,
    perform_significance_test: bool = True
):
    """
    Generate Table X for paper

    Args:
        df: DataFrame with aggregated results
        output_path: Output file path
        perform_significance_test: Whether to add significance markers
    """
    # Format table for paper
    table_data = []

    for _, row in df.iterrows():
        table_row = {
            'Model': row['Model'],
            'SNCR': f"{row['sncr_mean']:.3f}±{row['sncr_std']:.3f}",
            'UPM': f"{row['upm_mean']:.3f}±{row['upm_std']:.3f}",
            'K-Faith': f"{row['k_faith_mean']:.3f}±{row['k_faith_std']:.3f}",
            'AVC': f"{row['avc_mean']:.3f}±{row['avc_std']:.3f}",
            'BLEU-1': f"{row['bleu_1_mean']:.3f}±{row['bleu_1_std']:.3f}",
            'BLEU-2': f"{row['bleu_2_mean']:.3f}±{row['bleu_2_std']:.3f}",
            'BLEU-3': f"{row['bleu_3_mean']:.3f}±{row['bleu_3_std']:.3f}",
            'BLEU-4': f"{row['bleu_4_mean']:.3f}±{row['bleu_4_std']:.3f}",
            'ROUGE-L': f"{row['rouge_l_mean']:.3f}±{row['rouge_l_std']:.3f}",
            'Dist-2': f"{row['dist_2_mean']:.3f}±{row['dist_2_std']:.3f}",
            'Nutr-Acc': f"{row['nutrition_accuracy_mean']:.3f}±{row['nutrition_accuracy_std']:.3f}"
        }
        table_data.append(table_row)

    table_df = pd.DataFrame(table_data)

    # Find best model for each metric
    best_sncr_idx = df['sncr_mean'].idxmax()
    best_upm_idx = df['upm_mean'].idxmax()
    best_k_faith_idx = df['k_faith_mean'].idxmax()
    best_avc_idx = df['avc_mean'].idxmin()  # Lower is better

    # Mark best values in bold (for LaTeX)
    table_df.loc[best_sncr_idx, 'SNCR'] = "\\textbf{" + table_df.loc[best_sncr_idx, 'SNCR'] + "}"
    table_df.loc[best_upm_idx, 'UPM'] = "\\textbf{" + table_df.loc[best_upm_idx, 'UPM'] + "}"
    table_df.loc[best_k_faith_idx, 'K-Faith'] = "\\textbf{" + table_df.loc[best_k_faith_idx, 'K-Faith'] + "}"
    table_df.loc[best_avc_idx, 'AVC'] = "\\textbf{" + table_df.loc[best_avc_idx, 'AVC'] + "}"

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    table_df.to_csv(str(output_path).replace('.txt', '.csv'), index=False)

    # Save as formatted text
    with open(output_path, 'w') as f:
        f.write("Table X: Base LLM Selection Results (RQ1)\n")
        f.write("=" * 80 + "\n\n")
        f.write(table_df.to_string(index=False))
        f.write("\n\n")
        f.write("Note: Bold values indicate best performance.\n")
        f.write("* p<0.05, ** p<0.01, *** p<0.001\n")

    # Save as LaTeX
    latex_path = str(output_path).replace('.txt', '.tex')
    with open(latex_path, 'w') as f:
        f.write("% Table X: Base LLM Selection Results (RQ1)\n")
        f.write(table_df.to_latex(index=False, escape=False))

    print(f"\n✅ Table X saved to:")
    print(f"   - Text: {output_path}")
    print(f"   - CSV: {str(output_path).replace('.txt', '.csv')}")
    print(f"   - LaTeX: {latex_path}")

    # Print to console
    print("\n" + "=" * 80)
    print("TABLE X: Base LLM Selection Results")
    print("=" * 80)
    print(table_df.to_string(index=False))
    print("=" * 80)

    # Identify best model
    best_model = df.loc[best_sncr_idx, 'Model']
    print(f"\n🏆 Best Model (by SNCR): {best_model}")
    print(f"   SNCR: {df.loc[best_sncr_idx, 'sncr_mean']:.3f} ± {df.loc[best_sncr_idx, 'sncr_std']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ1 Results")

    parser.add_argument('--experiments_dir', type=str, required=True,
                        help='Base experiments directory')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Model names (e.g., meta-llama/Llama-3-8B Qwen/Qwen2-7B)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024],
                        help='Random seeds used')
    parser.add_argument('--output_file', type=str, default='results/table_x.txt',
                        help='Output file path')

    args = parser.parse_args()

    experiments_dir = Path(args.experiments_dir)

    # Aggregate results
    df = aggregate_multiseed_results(
        experiments_dir=experiments_dir,
        model_names=args.models,
        seeds=args.seeds
    )

    if df.empty:
        print("❌ No results found. Please check your experiments directory.")
        return

    # Generate Table X
    generate_table_x(
        df=df,
        output_path=args.output_file
    )

    print("\n🎉 RQ1 results aggregation completed!")


if __name__ == "__main__":
    main()
