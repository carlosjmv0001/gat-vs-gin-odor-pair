import numpy as np  
import scipy.stats as stats  
import matplotlib.pyplot as plt  
import json  
import torch  
from collections import defaultdict  
import os  
import sys  
from statsmodels.stats.multitest import multipletests  
  
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
  
def calculate_cohens_d(group1, group2):  
    """Calculate Cohen's d effect size"""  
    n1, n2 = len(group1), len(group2)  
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +   
                         (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))  
    d = (np.mean(group1) - np.mean(group2)) / pooled_std  
    return d  
  
def calculate_effect_size_metrics(gin_scores, gat_scores):  
    """Calculate comprehensive effect size metrics"""  
    metrics = {}  
      
    # Cohen's d  
    metrics['cohens_d'] = calculate_cohens_d(gin_scores, gat_scores)  
      
    # Glass's delta  
    metrics['glass_delta'] = (np.mean(gat_scores) - np.mean(gin_scores)) / np.std(gin_scores)  
      
    # Hedges' g (small sample correction)  
    n = len(gin_scores) + len(gat_scores)  
    correction_factor = 1 - (3 / (4 * n - 9))  
    metrics['hedges_g'] = metrics['cohens_d'] * correction_factor  
      
    return metrics  
  
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
  
def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95):  
    """Calculate bootstrap confidence intervals"""  
    bootstrap_means = []  
    n = len(data)  
      
    for _ in range(n_bootstrap):  
        bootstrap_sample = np.random.choice(data, size=n, replace=True)  
        bootstrap_means.append(np.mean(bootstrap_sample))  
      
    bootstrap_means = np.array(bootstrap_means)  
    lower = np.percentile(bootstrap_means, (1 - confidence) * 100 / 2)  
    upper = np.percentile(bootstrap_means, 100 - (1 - confidence) * 100 / 2)  
      
    return np.mean(bootstrap_means), lower, upper  
  
def calculate_enhanced_confidence_intervals(scores, confidence=0.95):  
    """Calculate both parametric and bootstrap confidence intervals"""  
    # Original parametric CI  
    mean, margin, parametric_ci = calculate_confidence_intervals(scores, confidence)  
      
    # Bootstrap CI  
    bootstrap_mean, bootstrap_lower, bootstrap_upper = bootstrap_confidence_interval(scores, confidence=confidence)  
    bootstrap_ci = (bootstrap_lower, bootstrap_upper)  
      
    return {  
        'mean': mean,  
        'parametric_ci': parametric_ci,  
        'bootstrap_ci': bootstrap_ci,  
        'margin_parametric': margin,  
        'margin_bootstrap': max(bootstrap_mean - bootstrap_lower, bootstrap_upper - bootstrap_mean)  
    }  
  
def calculate_statistical_power(effect_size, n_samples, alpha=0.05):  
    """Calculate statistical power using t-test approximation"""  
    # Non-central t-distribution parameters  
    df = 2 * n_samples - 2  
    ncp = effect_size * np.sqrt(n_samples / 2)  
      
    # Critical t-value  
    t_critical = stats.t.ppf(1 - alpha/2, df)  
      
    # Power calculation  
    power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)  
    return power  
  
def power_analysis(gin_scores, gat_scores, target_power=0.8):  
    """Perform comprehensive power analysis"""  
    effect_size = abs(calculate_cohens_d(gin_scores, gat_scores))  
    n_current = len(gin_scores)  
      
    # Current power  
    current_power = calculate_statistical_power(effect_size, n_current)  
      
    # Required sample size for target power  
    n_required = None  
    for n in range(n_current, 1000):  
        if calculate_statistical_power(effect_size, n) >= target_power:  
            n_required = n  
            break  
      
    return {  
        'effect_size': effect_size,  
        'current_n': n_current,  
        'current_power': current_power,  
        'required_n': n_required,  
        'target_power': target_power  
    }  
  
def apply_multiple_comparison_corrections(p_values, method='fdr_bh'):  
    """Apply multiple comparison corrections"""  
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method=method)  
    return p_corrected, rejected  
  
def per_class_statistical_analysis(gin_aurocs, gat_aurocs, class_names):  
    """Perform per-class analysis with multiple comparison correction"""  
    p_values = []  
    effect_sizes = []  
      
    for i in range(len(gin_aurocs)):  
        # Wilcoxon signed-rank test for each class  
        stat, p_val = stats.wilcoxon(gin_aurocs[:, i], gat_aurocs[:, i])  
        p_values.append(p_val)  
          
        # Effect size for each class  
        effect_sizes.append(calculate_cohens_d(gin_aurocs[:, i], gat_aurocs[:, i]))  
      
    # Apply corrections  
    p_bonferroni, _ = apply_multiple_comparison_corrections(p_values, 'bonferroni')  
    p_fdr, _ = apply_multiple_comparison_corrections(p_values, 'fdr_bh')  
      
    return {  
        'p_values': p_values,  
        'p_bonferroni': p_bonferroni,  
        'p_fdr': p_fdr,  
        'effect_sizes': effect_sizes,  
        'significant_bonferroni': np.sum(p_bonferroni < 0.05),  
        'significant_fdr': np.sum(p_fdr < 0.05)  
    }  
  
def non_parametric_statistical_tests(gin_scores, gat_scores):  
    """Perform comprehensive non-parametric statistical tests"""  
    results = {}  
      
    # Wilcoxon signed-rank test (paired)  
    if len(gin_scores) == len(gat_scores):  
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(gin_scores, gat_scores)  
        results['wilcoxon'] = {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p}  
      
    # Mann-Whitney U test (independent)  
    mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(gin_scores, gat_scores, alternative='two-sided')  
    results['mannwhitney'] = {'statistic': mannwhitney_stat, 'p_value': mannwhitney_p}  
      
    # Kolmogorov-Smirnov test  
    ks_stat, ks_p = stats.ks_2samp(gin_scores, gat_scores)  
    results['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_p}  
      
    # Permutation test  
    n_permutations = 10000  
    observed_diff = np.mean(gat_scores) - np.mean(gin_scores)  
    combined = np.concatenate([gin_scores, gat_scores])  
    permutation_diffs = []  
      
    for _ in range(n_permutations):  
        np.random.shuffle(combined)  
        perm_gin = combined[:len(gin_scores)]  
        perm_gat = combined[len(gin_scores):]  
        permutation_diffs.append(np.mean(perm_gat) - np.mean(perm_gin))  
      
    permutation_p = np.sum(np.abs(permutation_diffs) >= np.abs(observed_diff)) / n_permutations  
    results['permutation'] = {'observed_diff': observed_diff, 'p_value': permutation_p}  
      
    return results  
  
def enhanced_comprehensive_statistical_analysis():  
    """Enhanced statistical analysis with all improvements"""  
    results = load_all_results()  
  
    print("=== ENHANCED COMPREHENSIVE STATISTICAL ANALYSIS ===\n")  
  
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
          
        # Enhanced analysis  
        effect_sizes = calculate_effect_size_metrics(cv['gin_aurocs'], cv['gat_aurocs'])  
        print(f"   Effect Size (Cohen's d): {effect_sizes['cohens_d']:.4f}")  
        print(f"   Effect Size (Hedges' g): {effect_sizes['hedges_g']:.4f}")  
          
        power_results = power_analysis(cv['gin_aurocs'], cv['gat_aurocs'])  
        print(f"   Statistical power: {power_results['current_power']:.4f}")  
          
        non_param_results = non_parametric_statistical_tests(cv['gin_aurocs'], cv['gat_aurocs'])  
        print(f"   Wilcoxon p-value: {non_param_results['wilcoxon']['p_value']:.4f}")  
        print(f"   Permutation p-value: {non_param_results['permutation']['p_value']:.4f}")  
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
        
        # Enhanced effect size analysis  
        effect_sizes = calculate_effect_size_metrics(gin_metrics, gat_metrics)  
        print(f"   Overall Effect Size (Cohen's d): {effect_sizes['cohens_d']:.4f}")  
        print(f"   Overall Effect Size (Hedges' g): {effect_sizes['hedges_g']:.4f}")  
        
        # Power analysis  
        power_results = power_analysis(gin_metrics, gat_metrics)  
        print(f"   Statistical power: {power_results['current_power']:.4f}")  
        
        # CORRECTED: Use actual cross-validation data for non-parametric tests  
        if 'cross_validation' in results:  
            cv = results['cross_validation']  
            non_param_results = non_parametric_statistical_tests(cv['gin_aurocs'], cv['gat_aurocs'])  
            print(f"   Wilcoxon p-value: {non_param_results['wilcoxon']['p_value']:.4f}")  
            print(f"   Permutation p-value: {non_param_results['permutation']['p_value']:.4f}")  
        else:  
            # Fallback to aggregated data if no cross-validation available  
            non_param_results = non_parametric_statistical_tests(gin_metrics, gat_metrics)  
            print(f"   Wilcoxon p-value: {non_param_results['wilcoxon']['p_value']:.4f}")  
            print(f"   Permutation p-value: {non_param_results['permutation']['p_value']:.4f}")  
        print()
  
    return gin_metrics, gat_metrics, analysis_names, results  
  
def create_enhanced_statistical_visualization(gin_metrics, gat_metrics, analysis_names, results, results_dir):  
    """Create enhanced visualizations with new statistical metrics"""  
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))  
      
    # Plot 1: Effect sizes  
    effect_sizes = [calculate_cohens_d([gin], [gat]) for gin, gat in zip(gin_metrics, gat_metrics)]  
    bars = ax1.bar(analysis_names, effect_sizes, alpha=0.7, color='purple')  
    ax1.set_ylabel("Cohen's d")  
    ax1.set_title('Effect Sizes by Analysis Type')  
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)  
    ax1.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Small effect')  
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')  
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large effect')  
    ax1.legend()  
      
    # Plot 2: Power analysis  
    power_results = [power_analysis([gin], [gat]) for gin, gat in zip(gin_metrics, gat_metrics)]  
    current_powers = [p['current_power'] for p in power_results]  
    ax2.bar(analysis_names, current_powers, alpha=0.7, color='orange')  
    ax2.set_ylabel('Statistical Power')  
    ax2.set_title('Statistical Power by Analysis Type')  
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target power (0.8)')  
    ax2.legend()  
      
    # Plot 3: Confidence intervals comparison - HYBRID VERSION  
    x = np.arange(len(analysis_names))  
    width = 0.35  
    
    gin_means = []  
    gin_errors = []  
    gat_means = []  
    gat_errors = []  
    
    for i, name in enumerate(analysis_names):  
        if name == 'Cross-Validation' and 'cross_validation' in results:  
            # Usar dados reais das folds  
            cv = results['cross_validation']  
            gin_ci = calculate_enhanced_confidence_intervals(cv['gin_aurocs'])  
            gat_ci = calculate_enhanced_confidence_intervals(cv['gat_aurocs'])  
            
            gin_means.append(gin_ci['mean'])  
            gin_errors.append(gin_ci['margin_bootstrap'])  
            gat_means.append(gat_ci['mean'])  
            gat_errors.append(gat_ci['margin_bootstrap'])  
        else:  
            # Métricas padronizadas - indicadores visuais pequenos  
            gin_means.append(gin_metrics[i])  
            gat_means.append(gat_metrics[i])  
            gin_errors.append(0.005)  # Indicador visual  
            gat_errors.append(0.005)  
    
    ax3.bar(x - width/2, gin_means, width, yerr=gin_errors,   
            label='GIN', alpha=0.8, color='blue', capsize=5)  
    ax3.bar(x + width/2, gat_means, width, yerr=gat_errors,   
            label='GAT', alpha=0.8, color='red', capsize=5)  
    
    ax3.set_ylabel('AUROC Score')  
    ax3.set_title('Confidence Intervals (Bootstrap for CV, Indicators for Std)')  
    ax3.set_xticks(x)  
    ax3.set_xticklabels(analysis_names, rotation=45, ha='right')  
    ax3.legend()
      
    # Plot 4: P-value bar chart (more appropriate)  
    if 'cross_validation' in results:  
        cv = results['cross_validation']  
        gin_aurocs = cv['gin_aurocs']  
        gat_aurocs = cv['gat_aurocs']  
        
        t_stat, p_val = stats.ttest_rel(gat_aurocs, gin_aurocs)  
        
        bars = ax4.bar(['Cross-Validation'], [p_val], alpha=0.7,   
                    color='red' if p_val < 0.05 else 'blue')  
        ax4.set_ylabel('P-value')  
        ax4.set_title('Statistical Significance')  
        ax4.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='α=0.05 threshold')  
        ax4.set_yscale('log')  
        ax4.legend()  
        ax4.grid(True, alpha=0.3)  
        
        # Add significance indicator and p-value text  
        if p_val < 0.05:  
            ax4.text(0, p_val * 1.1, '*', ha='center', va='bottom',   
                    fontsize=14, fontweight='bold')  
        ax4.text(0, p_val * 0.5, f'p = {p_val:.4f}', ha='center', va='center',   
                fontweight='bold', color='white')
          
    plt.tight_layout()  
    plt.savefig(os.path.join(results_dir, 'enhanced_statistical_analysis.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
def save_enhanced_results(gin_metrics, gat_metrics, analysis_names, results, results_dir):  
    """Save enhanced analysis results with all statistical metrics"""  
    enhanced_results = {  
        'gin_metrics': gin_metrics,  
        'gat_metrics': gat_metrics,  
        'analysis_names': analysis_names,  
        'overall_gin_mean': float(np.mean(gin_metrics)),  
        'overall_gat_mean': float(np.mean(gat_metrics)),  
        'overall_gin_std': float(np.std(gin_metrics)),  
        'overall_gat_std': float(np.std(gat_metrics)),  
        'effect_sizes': {},  
        'power_analysis': {},  
        'non_parametric_tests': {}  
    }  
      
    # Calculate and store effect sizes  
    effect_sizes = calculate_effect_size_metrics(gin_metrics, gat_metrics)  
    enhanced_results['effect_sizes'] = {  
        'cohens_d': float(effect_sizes['cohens_d']),  
        'hedges_g': float(effect_sizes['hedges_g']),  
        'glass_delta': float(effect_sizes['glass_delta'])  
    }  
      
    # Calculate and store power analysis  
    power_results = power_analysis(gin_metrics, gat_metrics)  
    enhanced_results['power_analysis'] = {  
        'effect_size': float(power_results['effect_size']),  
        'current_power': float(power_results['current_power']),  
        'required_n': power_results['required_n'],  
        'target_power': power_results['target_power']  
    }  
      
    # Calculate and store non-parametric tests  
    if 'cross_validation' in results:  
        cv = results['cross_validation']  
        non_param_results = non_parametric_statistical_tests(cv['gin_aurocs'], cv['gat_aurocs'])  
    else:  
        # Fallback to aggregated data if no cross-validation available  
        non_param_results = non_parametric_statistical_tests(gin_metrics, gat_metrics)
    enhanced_results['non_parametric_tests'] = {  
        'wilcoxon_p_value': float(non_param_results['wilcoxon']['p_value']),  
        'mannwhitney_p_value': float(non_param_results['mannwhitney']['p_value']),  
        'ks_p_value': float(non_param_results['kolmogorov_smirnov']['p_value']),  
        'permutation_p_value': float(non_param_results['permutation']['p_value'])  
    }  
      
    # Save enhanced results  
    enhanced_summary_file = os.path.join(results_dir, 'enhanced_statistical_analysis_summary.json')  
    with open(enhanced_summary_file, 'w') as f:  
        json.dump(enhanced_results, f, indent=2)  
      
    print(f"\nEnhanced results saved in '{enhanced_summary_file}'")  
    return enhanced_results  
  
if __name__ == "__main__":  
    print("Starting enhanced comparative statistical analysis...")  
  
    # Get project base directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
  
    # Create results directory using absolute path  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
  
    # Run enhanced statistical analysis  
    gin_metrics, gat_metrics, analysis_names, results = enhanced_comprehensive_statistical_analysis()  
  
    if gin_metrics and gat_metrics:  
        # Create enhanced visualizations  
        create_enhanced_statistical_visualization(gin_metrics, gat_metrics, analysis_names, results, results_dir)  
          
        # Save enhanced results  
        enhanced_results = save_enhanced_results(gin_metrics, gat_metrics, analysis_names, results, results_dir)  
          
        print(f"\nEnhanced analysis complete!")  
        print(f"Plot saved as '{os.path.join(results_dir, 'enhanced_statistical_analysis.png')}'")  
          
        # Summary of key findings  
        print("\n=== KEY STATISTICAL FINDINGS ===")  
        effect_sizes = enhanced_results['effect_sizes']  
        power_analysis = enhanced_results['power_analysis']  
        non_param = enhanced_results['non_parametric_tests']  
          
        print(f"Effect Size (Cohen's d): {effect_sizes['cohens_d']:.4f}")  
        print(f"Statistical Power: {power_analysis['current_power']:.4f}")  
        print(f"Wilcoxon p-value: {non_param['wilcoxon_p_value']:.4f}")  
        print(f"Permutation p-value: {non_param['permutation_p_value']:.4f}")  
          
        if power_analysis['required_n']:  
            print(f"Required sample size for 80% power: {power_analysis['required_n']}")  
          
    else:  
        print("No results found for enhanced analysis.")