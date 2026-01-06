import torch  
import torch_geometric as pyg  
import tqdm  
import sys  
import numpy as np  
import json  
from sklearn.metrics import roc_auc_score, classification_report  
import matplotlib.pyplot as plt  
import os  
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ks_2samp  
from statsmodels.stats.multitest import multipletests  
  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
  
from pairing.data import PairData, Dataset, loader  
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential  
from new1_main import MixturePredictor as GATMixturePredictor, GAT  
  
# Critical: Add the classes to the main module so PyTorch can find them during unpickling  
sys.modules['__main__'].MixturePredictor = GINMixturePredictor  
sys.modules['__main__'].GCN = GCN  
sys.modules['__main__'].GAT = GAT  
sys.modules['__main__'].make_sequential = make_sequential  
  
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
    import scipy.stats as stats  
      
    # Parametric CI  
    mean = np.mean(scores)  
    sem = stats.sem(scores)  
    t_critical = stats.t.ppf((1 + confidence) / 2, df=len(scores) - 1)  
    margin_of_error = t_critical * sem  
    parametric_ci = (mean - margin_of_error, mean + margin_of_error)  
      
    # Bootstrap CI  
    bootstrap_mean, bootstrap_lower, bootstrap_upper = bootstrap_confidence_interval(scores, confidence=confidence)  
    bootstrap_ci = (bootstrap_lower, bootstrap_upper)  
      
    return {  
        'mean': mean,  
        'parametric_ci': parametric_ci,  
        'bootstrap_ci': bootstrap_ci,  
        'margin_parametric': margin_of_error,  
        'margin_bootstrap': max(bootstrap_mean - bootstrap_lower, bootstrap_upper - bootstrap_mean)  
    }  
  
def calculate_statistical_power(effect_size, n_samples, alpha=0.05):  
    """Calculate statistical power using t-test approximation"""  
    import scipy.stats as stats  
      
    df = 2 * n_samples - 2  
    ncp = effect_size * np.sqrt(n_samples / 2)  
      
    t_critical = stats.t.ppf(1 - alpha/2, df)  
    power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)  
    return power  
  
def power_analysis(gin_scores, gat_scores, target_power=0.8):  
    """Perform comprehensive power analysis"""  
    effect_size = abs(calculate_cohens_d(gin_scores, gat_scores))  
    n_current = len(gin_scores)  
      
    current_power = calculate_statistical_power(effect_size, n_current)  
      
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
  
def non_parametric_statistical_tests(gin_scores, gat_scores):  
    """Perform comprehensive non-parametric statistical tests"""  
    results = {}  
      
    # Wilcoxon signed-rank test (paired)  
    if len(gin_scores) == len(gat_scores):  
        wilcoxon_stat, wilcoxon_p = wilcoxon(gin_scores, gat_scores)  
        results['wilcoxon'] = {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p}  
      
    # Mann-Whitney U test (independent)  
    mannwhitney_stat, mannwhitney_p = mannwhitneyu(gin_scores, gat_scores, alternative='two-sided')  
    results['mannwhitney'] = {'statistic': mannwhitney_stat, 'p_value': mannwhitney_p}  
      
    # Kolmogorov-Smirnov test  
    ks_stat, ks_p = ks_2samp(gin_scores, gat_scores)  
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
  
def get_gin_model():  
    """Load trained GIN model"""  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    model = torch.load(os.path.join(base_dir, "runs/1adc5394/model.pt"),  
                      map_location=device, weights_only=False)  
    model.eval()  
    return model  
  
def get_gat_model():  
    """Load trained GAT model"""  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
  
    # Get project base directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
  
    # Switch to GAT's MixturePredictor for loading  
    sys.modules['__main__'].MixturePredictor = GATMixturePredictor  
    model = torch.load(os.path.join(base_dir, "runs/702a60ea/model.pt"),  
                      map_location=device, weights_only=False)  
    model.eval()  
    return model  
  
def collate_test_gin(test_dataset):  
    """Evaluate GIN model using the standardized framework"""  
    model = get_gin_model()  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    test_loader = loader(test_dataset, batch_size=128)  
    preds = []  
    ys = []  
  
    print("Evaluating GIN model...")  
    for batch in tqdm.tqdm(test_loader):  
        batch.to(device)  
        with torch.no_grad():  
            pred = model(**batch.to_dict())  
  
        preds.append(pred.cpu())  
        ys.append(batch.y.cpu())  
  
    return torch.cat(preds, dim=0), torch.cat(ys, dim=0)  
  
def collate_test_gat(test_dataset):  
    """Evaluate GAT model using the standardized framework"""  
    model = get_gat_model()  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    test_loader = loader(test_dataset, batch_size=128)  
    preds = []  
    ys = []  
  
    print("Evaluating GAT model...")  
    for batch in tqdm.tqdm(test_loader):  
        batch.to(device)  
        with torch.no_grad():  
            pred = model(**batch.to_dict())  
  
        preds.append(pred.cpu())  
        ys.append(batch.y.cpu())  
  
    return torch.cat(preds, dim=0), torch.cat(ys, dim=0)  
  
def calculate_metrics(predictions, true_labels, model_name):  
    """Calculate detailed metrics for a model"""  
    # Apply sigmoid to get probabilities  
    pred_probs = torch.sigmoid(predictions).numpy()  
    true_labels_np = true_labels.numpy()  
  
    # AUROC micro-average  
    auroc_micro = roc_auc_score(true_labels_np, pred_probs, average='micro')  
  
    # AUROC macro-average  
    auroc_macro = roc_auc_score(true_labels_np, pred_probs, average='macro')  
  
    # AUROC per class  
    auroc_per_class = roc_auc_score(true_labels_np, pred_probs, average=None)  
  
    # Binary predictions (threshold = 0.5)  
    pred_binary = (pred_probs > 0.5).astype(int)  
  
    # Calculate additional metrics  
    metrics = {  
        'model_name': model_name,  
        'auroc_micro': auroc_micro,  
        'auroc_macro': auroc_macro,  
        'auroc_per_class': auroc_per_class.tolist(),  
        'num_samples': len(true_labels),  
        'num_classes': true_labels.shape[1],  
        'mean_predictions': pred_probs.mean(axis=0).tolist(),  
        'mean_true_labels': true_labels_np.mean(axis=0).tolist()  
    }  
  
    return metrics  
  
def compare_models():  
    """Run enhanced standardized comparison between models"""  
    print("=== ENHANCED STANDARDIZED MODEL EVALUATION ===")  
    print("Loading test dataset...")  
  
    test_dataset = Dataset(is_train=False)  
  
    # Evaluate GIN model  
    gin_preds, true_labels = collate_test_gin(test_dataset)  
    gin_metrics = calculate_metrics(gin_preds, true_labels, "GIN")  
  
    # Reset namespace for GAT  
    sys.modules['__main__'].MixturePredictor = GINMixturePredictor  
  
    # Evaluate GAT model  
    gat_preds, _ = collate_test_gat(test_dataset)  
    gat_metrics = calculate_metrics(gat_preds, true_labels, "GAT")  
  
    # Enhanced statistical analysis  
    print("\n=== ENHANCED STATISTICAL ANALYSIS ===")  
    print(f"GIN AUROC (micro): {gin_metrics['auroc_micro']:.4f}")  
    print(f"GAT AUROC (micro): {gat_metrics['auroc_micro']:.4f}")  
    print(f"GIN AUROC (macro): {gin_metrics['auroc_macro']:.4f}")  
    print(f"GAT AUROC (macro): {gat_metrics['auroc_macro']:.4f}")  
  
    # Performance difference  
    micro_diff = gat_metrics['auroc_micro'] - gin_metrics['auroc_micro']  
    macro_diff = gat_metrics['auroc_macro'] - gin_metrics['auroc_macro']  
  
    print(f"\nDifference GAT - GIN:")  
    print(f"AUROC (micro): {micro_diff:+.4f}")  
    print(f"AUROC (macro): {macro_diff:+.4f}")  
  
    # Per-class analysis  
    auroc_diff_per_class = np.array(gat_metrics['auroc_per_class']) - np.array(gin_metrics['auroc_per_class'])  
    classes_better_gat = np.sum(auroc_diff_per_class > 0)  
    classes_better_gin = np.sum(auroc_diff_per_class < 0)  
  
    print(f"\nPer-class analysis:")  
    print(f"Classes where GAT is better: {classes_better_gat}/{gin_metrics['num_classes']}")  
    print(f"Classes where GIN is better: {classes_better_gin}/{gin_metrics['num_classes']}")  
  
    # Skip effect size calculations for single-sample case  
    print(f"\n=== MICRO-AVERAGE STATISTICAL ANALYSIS ===")  
    print("Effect size calculations skipped: insufficient samples (need >1 per group)")  
      
    print(f"\n=== MACRO-AVERAGE STATISTICAL ANALYSIS ===")  
    print("Effect size calculations skipped: insufficient samples (need >1 per group)")  
  
    # Non-parametric tests for per-class differences  
    print(f"\n=== PER-CLASS NON-PARAMETRIC TESTS ===")  
    non_param_results = non_parametric_statistical_tests(gin_metrics['auroc_per_class'], gat_metrics['auroc_per_class'])  
    print(f"Wilcoxon signed-rank: p = {non_param_results['wilcoxon']['p_value']:.4f}")  
    print(f"Mann-Whitney U: p = {non_param_results['mannwhitney']['p_value']:.4f}")  
    print(f"Kolmogorov-Smirnov: p = {non_param_results['kolmogorov_smirnov']['p_value']:.4f}")  
    print(f"Permutation test: p = {non_param_results['permutation']['p_value']:.4f}")  
  
    # Create visualization  
    create_enhanced_comparison_visualization(gin_metrics, gat_metrics, auroc_diff_per_class,   
                                           None, None, non_param_results)  
  
    return gin_metrics, gat_metrics, None, None, non_param_results
  
def create_enhanced_comparison_visualization(gin_metrics, gat_metrics, auroc_diff_per_class,   
                                           micro_effect_sizes, macro_effect_sizes, non_param_results):  
    """Create enhanced comparison visualizations with statistical metrics"""  
    # Get project base directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
  
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))  
  
    # Plot 1: Overall AUROC comparison  
    models = ['GIN', 'GAT']  
    micro_scores = [gin_metrics['auroc_micro'], gat_metrics['auroc_micro']]  
    macro_scores = [gin_metrics['auroc_macro'], gat_metrics['auroc_macro']]  
  
    x = np.arange(len(models))  
    width = 0.35  
  
    bars1 = ax1.bar(x - width/2, micro_scores, width, label='AUROC Micro', alpha=0.8)  
    bars2 = ax1.bar(x + width/2, macro_scores, width, label='AUROC Macro', alpha=0.8)  
    ax1.set_ylabel('AUROC Score')  
    ax1.set_title('Overall AUROC Comparison')  
    ax1.set_xticks(x)  
    ax1.set_xticklabels(models)  
    ax1.legend()  
    ax1.set_ylim(0, 1)  
    ax1.grid(True, alpha=0.3)  
  
    # Add values to AUROC bars  
    for bar, value in zip(bars1, micro_scores):  
        height = bar.get_height()  
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,  
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)  
      
    for bar, value in zip(bars2, macro_scores):  
        height = bar.get_height()  
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,  
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)  
  
    # Plot 2: Per-class AUROC difference  
    classes = np.arange(len(auroc_diff_per_class))  
    colors = ['red' if diff < 0 else 'green' for diff in auroc_diff_per_class]  
  
    ax2.bar(classes, auroc_diff_per_class, color=colors, alpha=0.7)  
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)  
    ax2.set_xlabel('Class Index')  
    ax2.set_ylabel('AUROC Difference (GAT - GIN)')  
    ax2.set_title(f'Per-Class Performance Difference - Wilcoxon p={non_param_results["wilcoxon"]["p_value"]:.4f}')  
    ax2.grid(True, alpha=0.3)  
  
    # Plot 3: Prediction Calibration  
    gin_mean_preds = np.array(gin_metrics['mean_predictions'])  
    gat_mean_preds = np.array(gat_metrics['mean_predictions'])  
    true_mean = np.array(gin_metrics['mean_true_labels'])  
  
    ax3.scatter(true_mean, gin_mean_preds, alpha=0.6, label='GIN', s=30)  
    ax3.scatter(true_mean, gat_mean_preds, alpha=0.6, label='GAT', s=30)  
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')  
    ax3.set_xlabel('True Frequency')  
    ax3.set_ylabel('Mean Prediction')  
    ax3.set_title('Prediction Calibration')  
    ax3.legend()  
    ax3.grid(True, alpha=0.3)  
  
    # Plot 4: Histogram of differences
    ax4.hist(auroc_diff_per_class, bins=20, alpha=0.7, edgecolor='black')  
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No difference')  
    ax4.axvline(x=np.mean(auroc_diff_per_class), color='green', linestyle='-',  
                alpha=0.7, label=f'Mean: {np.mean(auroc_diff_per_class):.4f}')  
    ax4.set_xlabel('AUROC Difference (GAT - GIN)')  
    ax4.set_ylabel('Number of Classes')  
    ax4.set_title('Distribution of Per-Class Differences')  
    ax4.legend()  
    ax4.grid(True, alpha=0.3)   
  
    plt.tight_layout()  
  
    enhanced_comparison_path = os.path.join(results_dir, 'enhanced_standardized_model_comparison.png')  
    plt.savefig(enhanced_comparison_path, dpi=300, bbox_inches='tight')  
    plt.close()  
  
    print(f"\nEnhanced plot saved as '{enhanced_comparison_path}'")
  
def calculate_statistical_power(effect_size, n_samples, alpha=0.05):  
    """Calculate statistical power using t-test approximation"""  
    import scipy.stats as stats  
      
    df = 2 * n_samples - 2  
    ncp = effect_size * np.sqrt(n_samples / 2)  
      
    t_critical = stats.t.ppf(1 - alpha/2, df)  
    power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)  
    return power  
  
def power_analysis(gin_scores, gat_scores, target_power=0.8):  
    """Perform comprehensive power analysis"""  
    effect_size = abs(calculate_cohens_d(gin_scores, gat_scores))  
    n_current = len(gin_scores)  
      
    current_power = calculate_statistical_power(effect_size, n_current)  
      
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
  
def non_parametric_statistical_tests(gin_scores, gat_scores):  
    """Perform comprehensive non-parametric statistical tests"""  
    results = {}  
      
    # Wilcoxon signed-rank test (paired)  
    if len(gin_scores) == len(gat_scores):  
        wilcoxon_stat, wilcoxon_p = wilcoxon(gin_scores, gat_scores)  
        results['wilcoxon'] = {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p}  
      
    # Mann-Whitney U test (independent)  
    mannwhitney_stat, mannwhitney_p = mannwhitneyu(gin_scores, gat_scores, alternative='two-sided')  
    results['mannwhitney'] = {'statistic': mannwhitney_stat, 'p_value': mannwhitney_p}  
      
    # Kolmogorov-Smirnov test  
    ks_stat, ks_p = ks_2samp(gin_scores, gat_scores)  
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
  
if __name__ == "__main__":  
    print("Starting enhanced standardized evaluation...")  
  
    # Get project base directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
  
    # Create results directory using absolute path  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
  
    # Run enhanced standardized comparison  
    gin_metrics, gat_metrics, micro_effect_sizes, macro_effect_sizes, non_param_results = compare_models()  
  
    # Save enhanced results using absolute path  
    results = {  
        'gin_metrics': gin_metrics,  
        'gat_metrics': gat_metrics,  
        'comparison_summary': {  
            'gin_auroc_micro': gin_metrics['auroc_micro'],  
            'gat_auroc_micro': gat_metrics['auroc_micro'],  
            'gin_auroc_macro': gin_metrics['auroc_macro'],  
            'gat_auroc_macro': gat_metrics['auroc_macro'],  
            'micro_improvement': gat_metrics['auroc_micro'] - gin_metrics['auroc_micro'],  
            'macro_improvement': gat_metrics['auroc_macro'] - gin_metrics['auroc_macro']  
        },  
        # Enhanced statistical metrics  
        'micro_effect_sizes': micro_effect_sizes,  
        'macro_effect_sizes': macro_effect_sizes,  
        'non_parametric_tests': {  
            'wilcoxon_p_value': float(non_param_results['wilcoxon']['p_value']),  
            'mannwhitney_p_value': float(non_param_results['mannwhitney']['p_value']),  
            'ks_p_value': float(non_param_results['kolmogorov_smirnov']['p_value']),  
            'permutation_p_value': float(non_param_results['permutation']['p_value'])  
        }  
    }  
  
    results_file_path = os.path.join(results_dir, 'enhanced_standardized_evaluation_results.json')  
    with open(results_file_path, 'w') as f:  
        json.dump(results, f, indent=2)  
  
    print(f"\nEnhanced detailed results saved in '{results_file_path}'")