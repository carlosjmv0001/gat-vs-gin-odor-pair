import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import json
import torch
from collections import defaultdict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_all_results():
    """Load all results from previous analyses"""
    results = {}

    # Get project base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Adjust paths for the results directory using absolute path
    results_dir = os.path.join(base_dir, "results")

    # Load cross-validation results
    try:
        with open(os.path.join(results_dir, 'cross_validation_results.json'), 'r') as f:
            cv_results = json.load(f)
            results['cross_validation'] = cv_results
    except FileNotFoundError:
        print("File cross_validation_results.json not found")

    # Load KL analysis results
    try:
        with open(os.path.join(results_dir, 'kl_analysis_results.json'), 'r') as f:
            kl_results = json.load(f)
            results['kl_analysis'] = kl_results
    except FileNotFoundError:
        print("File kl_analysis_results.json not found")

    # Load standardized evaluation results
    try:
        with open(os.path.join(results_dir, 'standardized_evaluation_results.json'), 'r') as f:
            std_results = json.load(f)
            results['standardized'] = std_results
    except FileNotFoundError:
        print("File standardized_evaluation_results.json not found")

    return results

def calculate_confidence_intervals(scores, confidence=0.95):
    """Calculate confidence intervals using the stddev.ipynb framework"""
    # Step 1: Calculate the mean
    mean = np.mean(scores)

    # Step 2: Calculate the standard error of the mean (SEM)
    sem = stats.sem(scores)

    # Step 3: Get the t-critical value for 95% confidence with df = len(scores) - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df=len(scores) - 1)

    # Step 4: Calculate the margin of error
    margin_of_error = t_critical * sem

    # Step 5: Calculate the confidence interval
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    return mean, margin_of_error, confidence_interval

def comprehensive_statistical_analysis():
    """Comprehensive statistical analysis of all results"""
    results = load_all_results()

    print("=== COMPREHENSIVE COMPARATIVE STATISTICAL ANALYSIS ===\n")

    # Collect all AUROC metrics
    gin_metrics = []
    gat_metrics = []
    analysis_names = []

    # Cross-validation
    if 'cross_validation' in results:
        cv = results['cross_validation']
        gin_metrics.append(cv['gin_mean'])
        gat_metrics.append(cv['gat_mean'])
        analysis_names.append('Cross-Validation')

        print("1. CROSS-VALIDATION:")
        gin_mean, gin_margin, gin_ci = calculate_confidence_intervals(cv['gin_aurocs'])
        gat_mean, gat_margin, gat_ci = calculate_confidence_intervals(cv['gat_aurocs'])

        print(f"   GIN: {gin_mean:.4f} ± {gin_margin:.4f} [{gin_ci[0]:.4f}, {gin_ci[1]:.4f}]")
        print(f"   GAT: {gat_mean:.4f} ± {gat_margin:.4f} [{gat_ci[0]:.4f}, {gat_ci[1]:.4f}]")
        print()

    # Standardized evaluation
    if 'standardized' in results:
        std = results['standardized']
        gin_metrics.append(std['comparison_summary']['gin_auroc_micro'])
        gat_metrics.append(std['comparison_summary']['gat_auroc_micro'])
        analysis_names.append('Standardized (Micro)')

        gin_metrics.append(std['comparison_summary']['gin_auroc_macro'])
        gat_metrics.append(std['comparison_summary']['gat_auroc_macro'])
        analysis_names.append('Standardized (Macro)')

        print("2. STANDARDIZED EVALUATION:")
        print(f"   GIN (Micro): {std['comparison_summary']['gin_auroc_micro']:.4f}")
        print(f"   GAT (Micro): {std['comparison_summary']['gat_auroc_micro']:.4f}")
        print(f"   GIN (Macro): {std['comparison_summary']['gin_auroc_macro']:.4f}")
        print(f"   GAT (Macro): {std['comparison_summary']['gat_auroc_macro']:.4f}")
        print()

    # KL analysis (similarities)
    if 'kl_analysis' in results:
        kl = results['kl_analysis']
        print("3. KL SIMILARITY ANALYSIS:")
        print(f"   GIN vs Real: {kl['gin_similarity']:.4f}")
        print(f"   GAT vs Real: {kl['gat_similarity']:.4f}")
        print(f"   GIN vs GAT: {kl['gin_gat_similarity']:.4f}")
        print()

    # Consolidated statistical analysis
    if len(gin_metrics) > 1:
        print("4. CONSOLIDATED STATISTICAL ANALYSIS:")

        # Paired t-test for all metrics
        t_stat, p_value = stats.ttest_rel(gat_metrics, gin_metrics)

        print(f"   Paired t-test (all metrics):")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value:.4f}")

        if p_value < 0.05:
            winner = "GAT" if t_stat > 0 else "GIN"
            print(f"   Result: {winner} is statistically superior (p < 0.05)")
        else:
            print("   Result: No statistically significant difference")
        print()

    return gin_metrics, gat_metrics, analysis_names, results

def create_comprehensive_visualization(gin_metrics, gat_metrics, analysis_names, results, results_dir):
    """Create comprehensive visualization based on the kldiv.ipynb framework"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Comparison by analysis
    x = np.arange(len(analysis_names))
    width = 0.35

    ax1.bar(x - width/2, gin_metrics, width, label='GIN', alpha=0.8, color='blue')
    ax1.bar(x + width/2, gat_metrics, width, label='GAT', alpha=0.8, color='red')

    ax1.set_ylabel('AUROC Score')
    ax1.set_title('AUROC Comparison by Analysis Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(analysis_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add values to bars
    for i, (gin, gat) in enumerate(zip(gin_metrics, gat_metrics)):
        ax1.text(i - width/2, gin + 0.01, f'{gin:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, gat + 0.01, f'{gat:.3f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: KL distributions (if available)
    if 'kl_analysis' in results:
        kl = results['kl_analysis']
        similarities = [kl['gin_similarity'], kl['gat_similarity'], kl['gin_gat_similarity']]
        labels = ['GIN vs Real', 'GAT vs Real', 'GIN vs GAT']
        colors = ['blue', 'red', 'green']

        bars = ax2.bar(labels, similarities, color=colors, alpha=0.7)
        ax2.set_ylabel('KL Similarity')
        ax2.set_title('KL Similarities')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        for bar, sim in zip(bars, similarities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{sim:.4f}', ha='center', va='bottom')

    # Plot 3: Cross-validation (if available)
    if 'cross_validation' in results:
        cv = results['cross_validation']
        gin_aurocs = cv['gin_aurocs']
        gat_aurocs = cv['gat_aurocs']

        folds = range(1, len(gin_aurocs) + 1)
        ax3.plot(folds, gin_aurocs, 'o-', label='GIN', color='blue', linewidth=2)
        ax3.plot(folds, gat_aurocs, 'o-', label='GAT', color='red', linewidth=2)

        ax3.set_xlabel('Fold')
        ax3.set_ylabel('AUROC')
        ax3.set_title('Performance per Fold (Cross-Validation)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Statistical summary
    models = ['GIN', 'GAT']
    means = [np.mean(gin_metrics), np.mean(gat_metrics)]
    stds = [np.std(gin_metrics), np.std(gat_metrics)]

    bars = ax4.bar(models, means, yerr=stds, capsize=5, alpha=0.7, color=['blue', 'red'])
    ax4.set_ylabel('Mean AUROC')
    ax4.set_title('Overall Statistical Summary')
    ax4.grid(True, alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.4f}±{std:.4f}', ha='center', va='bottom')

    plt.tight_layout()

    analysis_path = os.path.join(results_dir, 'comprehensive_statistical_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as '{analysis_path}'")

if __name__ == "__main__":
    print("Starting comparative statistical analysis...")

    # Get project base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create results directory using absolute path
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    gin_metrics, gat_metrics, analysis_names, results = comprehensive_statistical_analysis()

    if gin_metrics and gat_metrics:
        create_comprehensive_visualization(gin_metrics, gat_metrics, analysis_names, results, results_dir)

        # Save final summary using absolute path
        summary = {
            'gin_metrics': gin_metrics,
            'gat_metrics': gat_metrics,
            'analysis_names': analysis_names,
            'overall_gin_mean': float(np.mean(gin_metrics)),
            'overall_gat_mean': float(np.mean(gat_metrics)),
            'overall_gin_std': float(np.std(gin_metrics)),
            'overall_gat_std': float(np.std(gat_metrics))
        }

        summary_file = os.path.join(results_dir, 'comprehensive_analysis_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nFinal summary saved in '{summary_file}'")
    else:
        print("No results found for analysis.")