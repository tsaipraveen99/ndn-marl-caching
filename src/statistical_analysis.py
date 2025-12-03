"""
Statistical Analysis Module for Algorithm Comparison
Provides functions for calculating statistics, confidence intervals, and significance tests
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Fallback implementation without scipy
    import statistics

logger = logging.getLogger(__name__)


def calculate_mean_std_ci(values: List[float], confidence: float = 0.95) -> Dict:
    """
    Calculate mean, standard deviation, and confidence interval for a list of values
    
    Args:
        values: List of values from multiple runs
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Dictionary with 'mean', 'std', 'ci_lower', 'ci_upper'
    """
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
    
    if len(values) < 2:
        return {'mean': float(values[0]), 'std': 0.0, 'ci_lower': float(values[0]), 'ci_upper': float(values[0])}
    
    mean = np.mean(values)
    std = calculate_std(values)
    ci_lower, ci_upper = calculate_confidence_interval(values, confidence)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }


def calculate_mean(values: List[float]) -> float:
    """Calculate mean across runs"""
    if not values:
        return 0.0
    return float(np.mean(values))


def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation across runs"""
    if not values or len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))  # Sample standard deviation


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval using t-distribution
    
    Args:
        values: List of values from multiple runs
        confidence: Confidence level (default: 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not values:
        return (0.0, 0.0)
    
    if len(values) < 2:
        return (values[0], values[0])
    
    try:
        mean = np.mean(values)
        if not SCIPY_AVAILABLE:
            # Fallback: use simple approximation
            std = np.std(values, ddof=1)
            n = len(values)
            # Approximate t-critical for 95% CI (t ~ 2 for large n)
            t_approx = 2.0 if n > 30 else 2.5
            margin = t_approx * (std / np.sqrt(n))
            return (float(mean - margin), float(mean + margin))
        
        sem = stats.sem(values)  # Standard error of the mean
        df = len(values) - 1  # Degrees of freedom
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df)
        margin = t_critical * sem
        
        return (float(mean - margin), float(mean + margin))
    except Exception as e:
        logger.warning(f"Error calculating confidence interval: {e}")
        return (float(np.mean(values)), float(np.mean(values)))


def t_test(group1: List[float], group2: List[float], alternative: str = 'two-sided') -> Dict:
    """
    Perform t-test to compare two groups
    
    Args:
        group1: First group of values
        group2: Second group of values
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    if not group1 or not group2:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': 'Insufficient data'
        }
    
    try:
        if not SCIPY_AVAILABLE:
            # Fallback: simple comparison without statistical test
            mean1 = np.mean(group1)
            mean2 = np.mean(group2)
            std1 = np.std(group1, ddof=1)
            std2 = np.std(group2, ddof=1)
            # Simple heuristic: significant if means differ by more than combined std
            combined_std = np.sqrt(std1**2 + std2**2)
            diff = abs(mean1 - mean2)
            significant = diff > 2 * combined_std
            
            return {
                't_statistic': diff / max(combined_std, 1e-10),
                'p_value': 0.05 if significant else 0.5,  # Approximation
                'significant': significant,
                'interpretation': "Significant difference" if significant else "No significant difference",
                'mean1': float(mean1),
                'mean2': float(mean2)
            }
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
        
        # Determine significance (p < 0.05)
        significant = p_value < 0.05
        
        interpretation = "Significant difference" if significant else "No significant difference"
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': significant,
            'interpretation': interpretation,
            'mean1': float(np.mean(group1)),
            'mean2': float(np.mean(group2))
        }
    except Exception as e:
        logger.warning(f"Error performing t-test: {e}")
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': f'Error: {e}'
        }


def effect_size(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size
    
    Args:
        group1: First group of values
        group2: Second group of values
    
    Returns:
        Cohen's d effect size
    """
    if not group1 or not group2:
        return 0.0
    
    try:
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1)
        std2 = np.std(group2, ddof=1)
        
        # Pooled standard deviation
        n1 = len(group1)
        n2 = len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        return float(cohens_d)
    except Exception as e:
        logger.warning(f"Error calculating effect size: {e}")
        return 0.0


def mann_whitney_u_test(group1: List[float], group2: List[float], alternative: str = 'two-sided') -> Dict:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test)
    Useful when data is not normally distributed
    
    Args:
        group1: First group of values
        group2: Second group of values
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with U-statistic, p-value, and interpretation
    """
    if not group1 or not group2:
        return {
            'u_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': 'Insufficient data'
        }
    
    try:
        if not SCIPY_AVAILABLE:
            # Fallback: use t-test approximation
            return t_test(group1, group2, alternative)
        
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)
        
        # Determine significance (p < 0.05)
        significant = p_value < 0.05
        
        interpretation = "Significant difference" if significant else "No significant difference"
        
        return {
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': significant,
            'interpretation': interpretation,
            'median1': float(np.median(group1)),
            'median2': float(np.median(group2))
        }
    except Exception as e:
        logger.warning(f"Error performing Mann-Whitney U test: {e}")
        return {
            'u_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': f'Error: {e}'
        }


def generate_statistics_report(results: Dict[str, List[float]], metric_name: str = "Metric") -> Dict:
    """
    Generate comprehensive statistics report for comparison
    
    Args:
        results: Dictionary mapping algorithm names to lists of values from multiple runs
        metric_name: Name of the metric being analyzed
    
    Returns:
        Dictionary with statistics for each algorithm
    """
    report = {
        'metric_name': metric_name,
        'algorithms': {}
    }
    
    for alg_name, values in results.items():
        if not values:
            continue
        
        mean_val = calculate_mean(values)
        std_val = calculate_std(values)
        ci_lower, ci_upper = calculate_confidence_interval(values)
        
        report['algorithms'][alg_name] = {
            'mean': mean_val,
            'std': std_val,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'count': len(values)
        }
    
    # Calculate pairwise comparisons
    alg_names = list(results.keys())
    comparisons = {}
    
    for i, alg1 in enumerate(alg_names):
        for alg2 in alg_names[i+1:]:
            if alg1 in results and alg2 in results:
                comparison_key = f"{alg1}_vs_{alg2}"
                t_test_result = t_test(results[alg1], results[alg2])
                effect = effect_size(results[alg1], results[alg2])
                
                comparisons[comparison_key] = {
                    **t_test_result,
                    'effect_size': effect,
                    'improvement_ratio': float(np.mean(results[alg1]) / max(np.mean(results[alg2]), 1e-10))
                }
    
    report['comparisons'] = comparisons
    
    return report


def format_statistics_table(report: Dict) -> str:
    """
    Format statistics report as a markdown table
    
    Args:
        report: Statistics report dictionary
    
    Returns:
        Formatted markdown table string
    """
    lines = []
    lines.append(f"## {report['metric_name']} Statistics")
    lines.append("")
    lines.append("| Algorithm | Mean | Std Dev | 95% CI Lower | 95% CI Upper | Min | Max | Count |")
    lines.append("|-----------|------|---------|--------------|--------------|-----|-----|-------|")
    
    for alg_name, stats_dict in report['algorithms'].items():
        lines.append(
            f"| {alg_name} | {stats_dict['mean']:.4f} | {stats_dict['std']:.4f} | "
            f"{stats_dict['ci_95_lower']:.4f} | {stats_dict['ci_95_upper']:.4f} | "
            f"{stats_dict['min']:.4f} | {stats_dict['max']:.4f} | {stats_dict['count']} |"
        )
    
    if report.get('comparisons'):
        lines.append("")
        lines.append("### Pairwise Comparisons")
        lines.append("")
        lines.append("| Comparison | Mean Difference | p-value | Significant | Effect Size | Improvement Ratio |")
        lines.append("|------------|-----------------|---------|-------------|-------------|-------------------|")
        
        for comp_key, comp_stats in report['comparisons'].items():
            lines.append(
                f"| {comp_key} | {comp_stats['mean1'] - comp_stats['mean2']:.4f} | "
                f"{comp_stats['p_value']:.4f} | {comp_stats['significant']} | "
                f"{comp_stats['effect_size']:.4f} | {comp_stats['improvement_ratio']:.2f}x |"
            )
    
    return "\n".join(lines)

