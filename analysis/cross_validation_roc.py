import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_curve, auc  
import json  
import os  
import sys  
from collections import defaultdict  
import scipy.stats as stats  
from torch_geometric.data import Batch  
from statsmodels.stats.multitest import multipletests  
  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
  
from pairing.data import Dataset, loader, PairData, to_pairdata, get_all_notes  
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential  
from new1_main import MixturePredictor as GATMixturePredictor, GAT  
  
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
  
def load_fold_data():  
    """Load cross-validation data from dataset/folds directory"""  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    folddir = os.path.join(base_dir, "dataset/folds")  
  
    if not os.path.exists(folddir):  
        print(f"Directory {folddir} not found. Run graph/coverage.py to generate folds.")  
        return []  
  
    fold_fnames = os.listdir(folddir)  
    folds = []  
  
    for fname in fold_fnames:  
        if not fname.endswith(".json"):  
            continue  
        with open(os.path.join(folddir, fname)) as f:  
            fold_data = json.load(f)  
            folds.append(fold_data)  
  
    print(f"Loaded {len(folds)} folds for cross-validation")  
    return folds  
  
def load_models():  
    """Load trained GIN and GAT models"""  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
  
    # Get project base directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
  
    # Load GIN model first  
    gin_model = torch.load(os.path.join(base_dir, "runs/1adc5394/model.pt"),  
                          map_location=device, weights_only=False)  
  
    # Switch to GAT's MixturePredictor for the GAT model  
    sys.modules['__main__'].MixturePredictor = GATMixturePredictor  
    gat_model = torch.load(os.path.join(base_dir, "runs/702a60ea/model.pt"),  
                          map_location=device, weights_only=False)  
  
    gin_model.eval()  
    gat_model.eval()  
  
    gin_model = gin_model.to(device)  
    gat_model = gat_model.to(device)  
  
    return gin_model, gat_model, device  
  
def create_fold_dataset_from_json(fold_data, is_test=True):  
    """Convert fold JSON data to Dataset format"""  
    # Get all valid notes  
    all_notes = get_all_notes()  
  
    # Choose train or test from fold  
    data_key = "test" if is_test else "train"  
    fold_edges = fold_data[data_key]  
  
    # Convert edges to PairData objects  
    data_list = []  
    for edge_data in fold_edges:  
        try:  
            # Extract edge information  
            edge = edge_data["edge"]  
            blend_notes = edge_data["blend_notes"]  
  
            # Create pairing in expected format  
            pairing = (tuple(edge), {note: 1 for note in blend_notes})  
  
            # Convert to PairData  
            pair_data = to_pairdata(pairing, all_notes)  
            data_list.append(pair_data)  
        except (KeyError, AttributeError, Exception):  
            continue  
  
    return data_list  
  
def evaluate_model_on_fold(model, fold_data, device, model_name, is_test=True):  
    """Evaluate a model on a specific fold using real fold data"""  
  
    # Create fold-specific dataset  
    fold_data_list = create_fold_dataset_from_json(fold_data, is_test)  
  
    if not fold_data_list:  
        print(f"Empty fold for {model_name}, skipping...")  
        return torch.tensor([]), torch.tensor([])  
  
    all_preds = []  
    all_true = []  
  
    print(f"Evaluating {model_name} on fold with {len(fold_data_list)} samples...")  
  
    # Use the existing loader function instead of manual batching  
    from torch_geometric.data import InMemoryDataset  
  
    # Create a temporary dataset  
    data, slices = InMemoryDataset.collate(fold_data_list)  
  
    # Create a temporary dataset class  
    class TempDataset:  
        def __init__(self, data, slices):  
            self.data = data  
            self.slices = slices  
  
        def __len__(self):  
            return len(fold_data_list)  
  
        def __getitem__(self, idx):  
            return fold_data_list[idx]  
  
    temp_dataset = TempDataset(data, slices)  
    test_loader = loader(temp_dataset, batch_size=128)  
  
    with torch.no_grad():  
        for batch in test_loader:  
            batch = batch.to(device)  
            pred = model(**batch.to_dict())  
            pred_probs = torch.sigmoid(pred)  
  
            all_preds.append(pred_probs.cpu())  
            all_true.append(batch.y.cpu())  
  
    if all_preds:  
        return torch.cat(all_preds, dim=0), torch.cat(all_true, dim=0)  
    else:  
        return torch.tensor([]), torch.tensor([])  
  
def calculate_roc_curves(predictions, true_labels, n_classes):  
    """Calculate ROC curves for each class"""  
    fpr = dict()  
    tpr = dict()  
    roc_auc = dict()  
  
    # Calculate ROC for each class  
    for i in range(n_classes):  
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predictions[:, i])  
        roc_auc[i] = auc(fpr[i], tpr[i])  
  
    # Calculate micro-average ROC  
    fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), predictions.ravel())  
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])  
  
    return fpr, tpr, roc_auc  
  
def perform_cross_validation():  
    """Run full cross-validation with real fold data"""  
    print("Loading models...")  
    gin_model, gat_model, device = load_models()  
  
    print("Loading fold data...")  
    folds = load_fold_data()  
  
    if not folds:  
        print("No folds found!")  
        return [], []  
  
    gin_results = []  
    gat_results = []  
  
    # For each fold, evaluate both models  
    for fold_idx, fold_data in enumerate(folds):  
        print(f"\n=== FOLD {fold_idx + 1} ===")  
  
        # Evaluate GIN on the specific fold  
        gin_preds, true_labels = evaluate_model_on_fold(  
            gin_model, fold_data, device, "GIN", is_test=True  
        )  
  
        if len(gin_preds) == 0:  
            print(f"Fold {fold_idx + 1} is empty, skipping...")  
            continue  
  
        gin_fpr, gin_tpr, gin_auc = calculate_roc_curves(  
            gin_preds.numpy(), true_labels.numpy(), gin_preds.shape[1]  
        )  
        gin_results.append({  
            'fpr': gin_fpr,  
            'tpr': gin_tpr,  
            'auc': gin_auc,  
            'predictions': gin_preds,  
            'true_labels': true_labels  
        })  
  
        # Evaluate GAT on the same fold  
        gat_preds, _ = evaluate_model_on_fold(  
            gat_model, fold_data, device, "GAT", is_test=True  
        )  
  
        gat_fpr, gat_tpr, gat_auc = calculate_roc_curves(  
            gat_preds.numpy(), true_labels.numpy(), gat_preds.shape[1]  
        )  
        gat_results.append({  
            'fpr': gat_fpr,  
            'tpr': gat_tpr,  
            'auc': gat_auc,  
            'predictions': gat_preds,  
            'true_labels': true_labels  
        })  
  
        print(f"GIN AUROC (micro): {gin_auc['micro']:.4f}")  
        print(f"GAT AUROC (micro): {gat_auc['micro']:.4f}")  
  
    return gin_results, gat_results  
  
def plot_roc_comparison(gin_results, gat_results):  
    """Create ROC comparison plots with confidence intervals"""  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
  
    # Collect AUROCs for statistical analysis  
    gin_aurocs = [result['auc']['micro'] for result in gin_results]  
    gat_aurocs = [result['auc']['micro'] for result in gat_results]  
  
    # Plot 1: Individual ROC curves  
    colors_gin = ['blue', 'lightblue', 'navy', 'darkblue', 'steelblue']  
    colors_gat = ['red', 'lightcoral', 'darkred', 'crimson', 'indianred']  
  
    for i, (gin_result, gat_result) in enumerate(zip(gin_results, gat_results)):  
        color_gin = colors_gin[i % len(colors_gin)]  
        color_gat = colors_gat[i % len(colors_gat)]  
  
        ax1.plot(gin_result['fpr']['micro'], gin_result['tpr']['micro'],  
                color=color_gin, alpha=0.7, linewidth=1.5,  
                label=f'GIN Fold {i+1} (AUC = {gin_result["auc"]["micro"]:.3f})')  
  
        ax1.plot(gat_result['fpr']['micro'], gat_result['tpr']['micro'],  
                color=color_gat, alpha=0.7, linewidth=1.5,  
                label=f'GAT Fold {i+1} (AUC = {gat_result["auc"]["micro"]:.3f})')  
  
    # Diagonal line (chance level)  
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance Level (AUC = 0.5)')  
  
    ax1.set_xlim([0.0, 1.0])  
    ax1.set_ylim([0.0, 1.05])  
    ax1.set_xlabel('False Positive Rate')  
    ax1.set_ylabel('True Positive Rate')  
    ax1.set_title('ROC Curves per Fold', pad=15)  
    ax1.legend(loc="lower right", fontsize=8)  
    ax1.grid(True, alpha=0.3)  
  
    # Plot 2: Statistical comparison with enhanced confidence intervals  
    gin_enhanced_ci = calculate_enhanced_confidence_intervals(gin_aurocs)  
    gat_enhanced_ci = calculate_enhanced_confidence_intervals(gat_aurocs)  
  
    models = ['GIN', 'GAT']  
    means = [gin_enhanced_ci['mean'], gat_enhanced_ci['mean']]  
    errors = [[gin_enhanced_ci['margin_bootstrap']], [gat_enhanced_ci['margin_bootstrap']]]  
  
    bars = ax2.bar(models, means, yerr=errors, capsize=5, alpha=0.7, color=['blue', 'red'])  
    ax2.set_ylabel('AUROC (Micro-Average)')  
    ax2.set_title('AUROC Comparison with Bootstrap 95% Confidence Intervals', pad=15)  
    ax2.set_ylim(0, 1)  
    ax2.grid(True, alpha=0.3)  
  
    # Add values to bars with enhanced confidence intervals  
    for bar, mean, ci in zip(bars, means, [gin_enhanced_ci, gat_enhanced_ci]):  
        height = bar.get_height()  
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,  
                f'{mean:.4f}\n[{ci["bootstrap_ci"][0]:.4f}, {ci["bootstrap_ci"][1]:.4f}]',  
                ha='center', va='bottom', fontsize=9)  
  
    plt.tight_layout()  
    plt.savefig(os.path.join(results_dir, 'cross_validation_roc_comparison.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
    return gin_aurocs, gat_aurocs  
  
def statistical_comparison(gin_aurocs, gat_aurocs):  
    """Perform enhanced statistical test between models"""  
    from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ks_2samp  
      
    print("\n=== ENHANCED STATISTICAL ANALYSIS ===")  
    print(f"GIN AUROC: {np.mean(gin_aurocs):.4f} ± {np.std(gin_aurocs):.4f}")  
    print(f"GAT AUROC: {np.mean(gat_aurocs):.4f} ± {np.std(gat_aurocs):.4f}")  
  
    if len(gin_aurocs) > 1:  
        # 1. Paired t-test (original)  
        t_stat, p_value = ttest_rel(gat_aurocs, gin_aurocs)  
        print(f"\nPaired t-test:")  
        print(f"t-statistic: {t_stat:.4f}")  
        print(f"p-value: {p_value:.4f}")  
  
        # 2. Effect size metrics  
        effect_sizes = calculate_effect_size_metrics(gin_aurocs, gat_aurocs)  
        print(f"\nEffect Sizes:")  
        print(f"Cohen's d: {effect_sizes['cohens_d']:.4f}")  
        print(f"Glass's delta: {effect_sizes['glass_delta']:.4f}")  
        print(f"Hedges' g: {effect_sizes['hedges_g']:.4f}")  
  
        # 3. Enhanced confidence intervals  
        gin_enhanced_ci = calculate_enhanced_confidence_intervals(gin_aurocs)  
        gat_enhanced_ci = calculate_enhanced_confidence_intervals(gat_aurocs)  
        print(f"\nEnhanced Confidence Intervals:")  
        print(f"GIN: {gin_enhanced_ci['mean']:.4f} ± {gin_enhanced_ci['margin_bootstrap']:.4f}")  
        print(f"     Parametric: [{gin_enhanced_ci['parametric_ci'][0]:.4f}, {gin_enhanced_ci['parametric_ci'][1]:.4f}]")  
        print(f"     Bootstrap: [{gin_enhanced_ci['bootstrap_ci'][0]:.4f}, {gin_enhanced_ci['bootstrap_ci'][1]:.4f}]")  
        print(f"GAT: {gat_enhanced_ci['mean']:.4f} ± {gat_enhanced_ci['margin_bootstrap']:.4f}")  
        print(f"     Parametric: [{gat_enhanced_ci['parametric_ci'][0]:.4f}, {gat_enhanced_ci['parametric_ci'][1]:.4f}]")  
        print(f"     Bootstrap: [{gat_enhanced_ci['bootstrap_ci'][0]:.4f}, {gat_enhanced_ci['bootstrap_ci'][1]:.4f}]")  
  
        # 4. Power analysis  
        power_results = power_analysis(gin_aurocs, gat_aurocs)  
        print(f"\nPower Analysis:")  
        print(f"Effect size: {power_results['effect_size']:.4f}")  
        print(f"Current power: {power_results['current_power']:.4f}")  
        print(f"Required n for 80% power: {power_results['required_n']}")  
  
        # 5. Non-parametric tests  
        non_param_results = non_parametric_statistical_tests(gin_aurocs, gat_aurocs)  
        print(f"\nNon-parametric Tests:")  
        print(f"Wilcoxon signed-rank: p = {non_param_results['wilcoxon']['p_value']:.4f}")  
        print(f"Mann-Whitney U: p = {non_param_results['mannwhitney']['p_value']:.4f}")  
        print(f"Kolmogorov-Smirnov: p = {non_param_results['kolmogorov_smirnov']['p_value']:.4f}")  
        print(f"Permutation test: p = {non_param_results['permutation']['p_value']:.4f}")  
  
        # 6. Interpretation  
        print(f"\nINTERPRETATION:")  
        if p_value < 0.05:  
            winner = "GAT" if t_stat > 0 else "GIN"  
            print(f"Statistically significant difference (p < 0.05): {winner} is superior")  
        else:  
            print("No statistically significant difference (p >= 0.05)")  
          
        # Effect size interpretation  
        abs_cohens_d = abs(effect_sizes['cohens_d'])  
        if abs_cohens_d < 0.2:  
            effect_interp = "negligible"  
        elif abs_cohens_d < 0.5:  
            effect_interp = "small"  
        elif abs_cohens_d < 0.8:  
            effect_interp = "medium"  
        else:  
            effect_interp = "large"  
        print(f"Effect size magnitude: {effect_interp}")  
          
    else:  
        print("Too few folds for robust statistical test")  
  
if __name__ == "__main__":  
    print("Starting cross-validation with ROC...")  
  
    # Get project base directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
  
    # Create results directory using relative path  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
  
    # Run cross-validation  
    gin_results, gat_results = perform_cross_validation()  
  
    if not gin_results or not gat_results:  
        print("No valid results obtained from folds.")  
        exit(1)  
  
    # Plot comparisons  
    gin_aurocs, gat_aurocs = plot_roc_comparison(gin_results, gat_results)  
  
    # Statistical analysis  
    statistical_comparison(gin_aurocs, gat_aurocs)  
  
    # Save enhanced results using absolute path  
    results = {  
        'gin_aurocs': gin_aurocs,  
        'gat_aurocs': gat_aurocs,  
        'gin_mean': float(np.mean(gin_aurocs)),  
        'gat_mean': float(np.mean(gat_aurocs)),  
        'gin_std': float(np.std(gin_aurocs)),  
        'gat_std': float(np.std(gat_aurocs)),  
        # Enhanced statistical metrics  
        'effect_sizes': calculate_effect_size_metrics(gin_aurocs, gat_aurocs),  
        'gin_enhanced_ci': calculate_enhanced_confidence_intervals(gin_aurocs),  
        'gat_enhanced_ci': calculate_enhanced_confidence_intervals(gat_aurocs),  
        'power_analysis': power_analysis(gin_aurocs, gat_aurocs),  
        'non_parametric_tests': non_parametric_statistical_tests(gin_aurocs, gat_aurocs)  
    }  
  
    results_file = os.path.join(results_dir, 'cross_validation_results.json')  
    with open(results_file, 'w') as f:  
        json.dump(results, f, indent=2)  
  
    print(f"\nResults saved in '{results_file}'")  
    print(f"Plot saved as '{os.path.join(results_dir, 'cross_validation_roc_comparison.png')}'")  