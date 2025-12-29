"""
Statistical Testing Utilities for NutriPlan
Provides significance testing for model comparisons
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import pandas as pd


class SignificanceTests:
    """Statistical significance testing for model comparisons"""

    @staticmethod
    def paired_t_test(
        values_a: List[float],
        values_b: List[float],
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Paired t-test

        Args:
            values_a: Values from model A (multiple seeds)
            values_b: Values from model B (multiple seeds)
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            (t_statistic, p_value)
        """
        assert len(values_a) == len(values_b), "Samples must have same length"

        t_stat, p_value = stats.ttest_rel(values_a, values_b, alternative=alternative)
        return t_stat, p_value

    @staticmethod
    def wilcoxon_test(
        values_a: List[float],
        values_b: List[float],
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test)

        Args:
            values_a: Values from model A
            values_b: Values from model B
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            (statistic, p_value)
        """
        assert len(values_a) == len(values_b), "Samples must have same length"

        stat, p_value = stats.wilcoxon(values_a, values_b, alternative=alternative)
        return stat, p_value

    @staticmethod
    def mann_whitney_u_test(
        values_a: List[float],
        values_b: List[float],
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Mann-Whitney U test (independent samples, non-parametric)

        Args:
            values_a: Values from model A
            values_b: Values from model B
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            (u_statistic, p_value)
        """
        stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative=alternative)
        return stat, p_value

    @staticmethod
    def effect_size_cohens_d(
        values_a: List[float],
        values_b: List[float]
    ) -> float:
        """
        Compute Cohen's d effect size

        Args:
            values_a: Values from model A
            values_b: Values from model B

        Returns:
            Cohen's d (standardized mean difference)
        """
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)

        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)

        # Pooled standard deviation
        n_a = len(values_a)
        n_b = len(values_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))

        if pooled_std == 0:
            return 0.0

        cohens_d = (mean_a - mean_b) / pooled_std
        return cohens_d

    @staticmethod
    def bootstrap_confidence_interval(
        values: List[float],
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval

        Args:
            values: Sample values
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default 95%)

        Returns:
            (mean, lower_bound, upper_bound)
        """
        bootstrap_means = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))

        mean = np.mean(values)
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return mean, lower_bound, upper_bound

    @staticmethod
    def format_p_value(p_value: float) -> str:
        """
        Format p-value with significance stars

        Args:
            p_value: P-value

        Returns:
            Formatted string with stars
        """
        if p_value < 0.001:
            return f"{p_value:.4f}***"
        elif p_value < 0.01:
            return f"{p_value:.4f}**"
        elif p_value < 0.05:
            return f"{p_value:.4f}*"
        else:
            return f"{p_value:.4f}"

    @staticmethod
    def interpret_effect_size(cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size

        Args:
            cohens_d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(cohens_d)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"


def compare_model_results(
    model_a_results: List[Dict[str, float]],
    model_b_results: List[Dict[str, float]],
    metrics: List[str],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B"
) -> pd.DataFrame:
    """
    Compare two models across multiple metrics with statistical tests

    Args:
        model_a_results: List of result dicts from model A (different seeds)
        model_b_results: List of result dicts from model B (different seeds)
        metrics: List of metric names to compare
        model_a_name: Name of model A
        model_b_name: Name of model B

    Returns:
        DataFrame with comparison results
    """
    tester = SignificanceTests()
    comparison_results = []

    for metric in metrics:
        # Extract values for this metric
        values_a = [result[metric] for result in model_a_results if metric in result]
        values_b = [result[metric] for result in model_b_results if metric in result]

        if not values_a or not values_b:
            continue

        # Compute statistics
        mean_a = np.mean(values_a)
        std_a = np.std(values_a)
        mean_b = np.mean(values_b)
        std_b = np.std(values_b)

        # Paired t-test
        t_stat, p_value = tester.paired_t_test(values_a, values_b)

        # Effect size
        cohens_d = tester.effect_size_cohens_d(values_a, values_b)
        effect_interpretation = tester.interpret_effect_size(cohens_d)

        # Confidence intervals
        _, ci_lower_a, ci_upper_a = tester.bootstrap_confidence_interval(values_a)
        _, ci_lower_b, ci_upper_b = tester.bootstrap_confidence_interval(values_b)

        comparison_results.append({
            'Metric': metric,
            f'{model_a_name} Mean': mean_a,
            f'{model_a_name} Std': std_a,
            f'{model_a_name} 95% CI': f"[{ci_lower_a:.4f}, {ci_upper_a:.4f}]",
            f'{model_b_name} Mean': mean_b,
            f'{model_b_name} Std': std_b,
            f'{model_b_name} 95% CI': f"[{ci_lower_b:.4f}, {ci_upper_b:.4f}]",
            'Difference': mean_a - mean_b,
            't-statistic': t_stat,
            'p-value': p_value,
            'Significance': tester.format_p_value(p_value),
            "Cohen's d": cohens_d,
            'Effect Size': effect_interpretation
        })

    df = pd.DataFrame(comparison_results)
    return df


if __name__ == "__main__":
    # Test statistical functions
    tester = SignificanceTests()

    # Example: Compare two models
    model_a_scores = [0.85, 0.87, 0.86]  # 3 seeds
    model_b_scores = [0.80, 0.82, 0.81]  # 3 seeds

    t_stat, p_value = tester.paired_t_test(model_a_scores, model_b_scores)
    cohens_d = tester.effect_size_cohens_d(model_a_scores, model_b_scores)

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {tester.format_p_value(p_value)}")
    print(f"Cohen's d: {cohens_d:.4f} ({tester.interpret_effect_size(cohens_d)})")

    # Example: Full comparison
    model_a_results = [
        {'sncr': 0.85, 'upm': 0.78, 'k_faith': 0.82},
        {'sncr': 0.87, 'upm': 0.79, 'k_faith': 0.83},
        {'sncr': 0.86, 'upm': 0.77, 'k_faith': 0.81}
    ]

    model_b_results = [
        {'sncr': 0.80, 'upm': 0.75, 'k_faith': 0.79},
        {'sncr': 0.82, 'upm': 0.76, 'k_faith': 0.80},
        {'sncr': 0.81, 'upm': 0.74, 'k_faith': 0.78}
    ]

    df = compare_model_results(
        model_a_results,
        model_b_results,
        metrics=['sncr', 'upm', 'k_faith'],
        model_a_name="NutriPlan",
        model_b_name="SFT"
    )

    print("\nComparison Table:")
    print(df.to_string(index=False))
