import torch  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
from sklearn.metrics import roc_auc_score  
import sys  
import json  
import os  
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, ks_2samp  
from statsmodels.stats.multitest import multipletests  
  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
  
from pairing.data import Dataset, loader  
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential  
from new1_main import MixturePredictor as GATMixturePredictor, GAT  
  
# Setup for model loading  
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
  
def load_models():  
    """Load GIN and GAT models"""  
    device = "cuda" if torch.cuda.is_available() else "cpu"  
  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
  
    gin_model = torch.load(os.path.join(base_dir, "runs/1adc5394/model.pt"),  
                          map_location=device, weights_only=False)  
  
    sys.modules['__main__'].MixturePredictor = GATMixturePredictor  
    gat_model = torch.load(os.path.join(base_dir, "runs/702a60ea/model.pt"),  
                          map_location=device, weights_only=False)  
  
    gin_model.eval()  
    gat_model.eval()  
  
    return gin_model.to(device), gat_model.to(device), device  
  
def get_class_names():  
    """Get olfactory note class names"""  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    with open(os.path.join(base_dir, "pairing/notes.csv"), "r") as f:  
        notes = f.read().split(",")  
    return [n.strip() for n in notes if n.strip()]  
  
def get_predictions_and_labels(model, dataset, device):  
    """Get predictions and true labels"""  
    test_loader = loader(dataset, batch_size=128)  
    all_preds = []  
    all_labels = []  
  
    with torch.no_grad():  
        for batch in test_loader:  
            batch = batch.to(device)  
            pred = model(**batch.to_dict())  
            pred_probs = torch.sigmoid(pred)  
  
            all_preds.append(pred_probs.cpu())  
            all_labels.append(batch.y.cpu())  
  
    return torch.cat(all_preds, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()  
  
def calculate_per_class_auroc(predictions, labels):  
    """Calculate per-class AUROC"""  
    aurocs = []  
    for i in range(predictions.shape[1]):  
        try:  
            auroc = roc_auc_score(labels[:, i], predictions[:, i])  
            aurocs.append(auroc)  
        except ValueError:  
            # Class with no positive examples  
            aurocs.append(0.0)  
    return np.array(aurocs)  
  
def analyze_class_frequency(labels, class_names):  
    """Analyze class frequency"""  
    frequencies = labels.sum(axis=0)  
    freq_df = pd.DataFrame({  
        'class': class_names,  
        'frequency': frequencies,  
        'percentage': frequencies / len(labels) * 100  
    })  
    return freq_df.sort_values('frequency', ascending=False)  
  
def create_performance_heatmap(gin_aurocs, gat_aurocs, class_names, results_dir):  
    """Create per-class performance heatmap"""  
    # Prepare data for heatmap  
    performance_data = np.array([gin_aurocs, gat_aurocs])  
  
    fig, ax = plt.subplots(figsize=(19, 8))  
  
    # Create heatmap  
    sns.heatmap(performance_data,  
                xticklabels=class_names,  
                yticklabels=['GIN', 'GAT'],  
                annot=True,  
                fmt='.3f',  
                cmap='RdYlBu_r',  
                center=0.5,  
                ax=ax)  
  
    plt.title('AUROC per Olfactory Note Class')  
    plt.xlabel('Note Classes')  
    plt.ylabel('Models')  
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()  
  
    heatmap_path = os.path.join(results_dir, 'class_performance_heatmap.png')  
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')  
    plt.close()  
    print(f"Heatmap saved as: {heatmap_path}")  
  
def create_difficulty_ranking(gin_aurocs, gat_aurocs, class_names, freq_df, results_dir):  
    """Create class difficulty ranking"""  
    avg_aurocs = (gin_aurocs + gat_aurocs) / 2  
  
    difficulty_df = pd.DataFrame({  
        'class': class_names,  
        'gin_auroc': gin_aurocs,  
        'gat_auroc': gat_aurocs,  
        'avg_auroc': avg_aurocs,  
        'frequency': freq_df.set_index('class').loc[class_names, 'frequency'].values  
    })  
  
    # Sort by difficulty (lower AUROC = harder)  
    difficulty_df = difficulty_df.sort_values('avg_auroc')  
  
    # Visualization  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  
  
    # Plot 1: Difficulty ranking  
    x = np.arange(len(class_names))  
    ax1.bar(x, difficulty_df['gin_auroc'], alpha=0.7, label='GIN', width=0.4)  
    ax1.bar(x + 0.4, difficulty_df['gat_auroc'], alpha=0.7, label='GAT', width=0.4)  
    ax1.set_xlabel('Classes (ordered by difficulty)')  
    ax1.set_ylabel('AUROC')  
    ax1.set_title('Class Difficulty Ranking')  
    ax1.set_xticks(x + 0.2)  
    ax1.set_xticklabels(difficulty_df['class'], rotation=45, ha='right')  
    ax1.legend()  
    ax1.grid(True, alpha=0.3)  
  
    # Plot 2: Frequency vs Performance  
    ax2.scatter(difficulty_df['frequency'], difficulty_df['gin_auroc'],  
                alpha=0.7, label='GIN', s=50)  
    ax2.scatter(difficulty_df['frequency'], difficulty_df['gat_auroc'],  
                alpha=0.7, label='GAT', s=50)  
    ax2.set_xlabel('Class Frequency')  
    ax2.set_ylabel('AUROC')  
    ax2.set_title('Frequency vs Performance')  
    ax2.legend()  
    ax2.grid(True, alpha=0.3)  
  
    plt.tight_layout()  
  
    difficulty_path = os.path.join(results_dir, 'difficulty_analysis.png')  
    plt.savefig(difficulty_path, dpi=300, bbox_inches='tight')  
    plt.close()  
    print(f"Difficulty analysis saved as: {difficulty_path}")  
  
    return difficulty_df  

def analyze_rare_vs_frequent(difficulty_df, results_dir):  
    """Analyze performance on rare vs frequent classes with statistical tests"""  
    # Define threshold for rare classes (bottom 25%)  
    freq_threshold = difficulty_df['frequency'].quantile(0.25)  
  
    rare_classes = difficulty_df[difficulty_df['frequency'] <= freq_threshold]  
    frequent_classes = difficulty_df[difficulty_df['frequency'] > freq_threshold]  
  
    print("=== RARE vs FREQUENT CLASSES ANALYSIS ===")  
    print(f"Frequency threshold: {freq_threshold:.0f}")  
    print(f"Rare classes: {len(rare_classes)}")  
    print(f"Frequent classes: {len(frequent_classes)}")  
  
    print("\nPerformance on Rare Classes:")  
    print(f"GIN mean AUROC: {rare_classes['gin_auroc'].mean():.4f}")  
    print(f"GAT mean AUROC: {rare_classes['gat_auroc'].mean():.4f}")  
  
    print("\nPerformance on Frequent Classes:")  
    print(f"GIN mean AUROC: {frequent_classes['gin_auroc'].mean():.4f}")  
    print(f"GAT mean AUROC: {frequent_classes['gat_auroc'].mean():.4f}")  
  
    # Statistical tests for rare classes  
    print("\n=== STATISTICAL TESTS FOR RARE CLASSES ===")  
    rare_effect_sizes = calculate_effect_size_metrics(rare_classes['gin_auroc'].values,   
                                                     rare_classes['gat_auroc'].values)  
    rare_statistical_tests = non_parametric_statistical_tests(rare_classes['gin_auroc'].values,   
                                                             rare_classes['gat_auroc'].values)  
      
    print(f"Effect Size (Cohen's d): {rare_effect_sizes['cohens_d']:.4f}")  
    print(f"Wilcoxon p-value: {rare_statistical_tests['wilcoxon']['p_value']:.4f}")  
    print(f"Mann-Whitney p-value: {rare_statistical_tests['mannwhitney']['p_value']:.4f}")  
    print(f"Permutation p-value: {rare_statistical_tests['permutation']['p_value']:.4f}")  
  
    # Statistical tests for frequent classes  
    print("\n=== STATISTICAL TESTS FOR FREQUENT CLASSES ===")  
    frequent_effect_sizes = calculate_effect_size_metrics(frequent_classes['gin_auroc'].values,   
                                                        frequent_classes['gat_auroc'].values)  
    frequent_statistical_tests = non_parametric_statistical_tests(frequent_classes['gin_auroc'].values,   
                                                                frequent_classes['gat_auroc'].values)  
      
    print(f"Effect Size (Cohen's d): {frequent_effect_sizes['cohens_d']:.4f}")  
    print(f"Wilcoxon p-value: {frequent_statistical_tests['wilcoxon']['p_value']:.4f}")  
    print(f"Mann-Whitney p-value: {frequent_statistical_tests['mannwhitney']['p_value']:.4f}")  
    print(f"Permutation p-value: {frequent_statistical_tests['permutation']['p_value']:.4f}")  
  
    # Plot 1: Performance comparison with statistical significance  
    fig1, ax1 = plt.subplots(figsize=(10, 6))  
      
    categories = ['Rare Classes', 'Frequent Classes']  
    gin_means = [rare_classes['gin_auroc'].mean(), frequent_classes['gin_auroc'].mean()]  
    gat_means = [rare_classes['gat_auroc'].mean(), frequent_classes['gat_auroc'].mean()]  
  
    x = np.arange(len(categories))  
    width = 0.35  
  
    bars1 = ax1.bar(x - width/2, gin_means, width, label='GIN', alpha=0.8)  
    bars2 = ax1.bar(x + width/2, gat_means, width, label='GAT', alpha=0.8)  
  
    ax1.set_ylabel('Mean AUROC')  
    ax1.set_title('Performance: Rare vs Frequent Classes')  
    ax1.set_xticks(x)  
    ax1.set_xticklabels(categories)  
    ax1.legend()  
    ax1.grid(True, alpha=0.3)  
  
    # Add values and significance to bars  
    for i, (gin, gat) in enumerate(zip(gin_means, gat_means)):  
        ax1.text(i - width/2, gin + 0.01, f'{gin:.3f}', ha='center', va='bottom')  
        ax1.text(i + width/2, gat + 0.01, f'{gat:.3f}', ha='center', va='bottom')  
          
        # Add significance indicators  
        if i == 0:  # Rare classes  
            p_val = rare_statistical_tests['wilcoxon']['p_value']  
        else:  # Frequent classes  
            p_val = frequent_statistical_tests['wilcoxon']['p_value']  
          
        if p_val < 0.05:  
            ax1.text(i, max(gin, gat) + 0.02, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')  
  
    plt.tight_layout()  
      
    performance_path = os.path.join(results_dir, 'rare_vs_frequent_performance.png')  
    plt.savefig(performance_path, dpi=300, bbox_inches='tight')  
    plt.close()  
    print(f"\nPerformance comparison plot saved as '{performance_path}'")  
  
    # Plot 2: Effect sizes comparison  
    fig2, ax2 = plt.subplots(figsize=(10, 6))  
      
    effect_metrics = ['Cohen\'s d', 'Glass\'s Δ', 'Hedges\' g']  
    rare_effects = [rare_effect_sizes['cohens_d'], rare_effect_sizes['glass_delta'], rare_effect_sizes['hedges_g']]  
    frequent_effects = [frequent_effect_sizes['cohens_d'], frequent_effect_sizes['glass_delta'], frequent_effect_sizes['hedges_g']]  
      
    x = np.arange(len(effect_metrics))  
    width = 0.35  
  
    ax2.bar(x - width/2, rare_effects, width, label='Rare Classes', alpha=0.8)  
    ax2.bar(x + width/2, frequent_effects, width, label='Frequent Classes', alpha=0.8)  
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)  
    ax2.set_ylabel('Effect Size')  
    ax2.set_title('Effect Size Comparison')  
    ax2.set_xticks(x)  
    ax2.set_xticklabels(effect_metrics)  
    ax2.legend()  
    ax2.grid(True, alpha=0.3)  
  
    plt.tight_layout()  
      
    effect_sizes_path = os.path.join(results_dir, 'rare_vs_frequent_effect_sizes.png')  
    plt.savefig(effect_sizes_path, dpi=300, bbox_inches='tight')  
    plt.close()  
    print(f"Effect sizes comparison plot saved as '{effect_sizes_path}'")  
  
    return rare_statistical_tests, frequent_statistical_tests, rare_effect_sizes, frequent_effect_sizes  

def main():  
    print("=== ENHANCED DETAILED CLASS PERFORMANCE ANALYSIS ===")  
  
    # Get project base directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
  
    # Create results directory using absolute path  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
  
    # Load models and data  
    gin_model, gat_model, device = load_models()  
    test_dataset = Dataset(is_train=False)  
    class_names = get_class_names()  
  
    print(f"Analyzing {len(class_names)} olfactory note classes...")  
  
    # Get predictions  
    gin_preds, labels = get_predictions_and_labels(gin_model, test_dataset, device)  
    gat_preds, _ = get_predictions_and_labels(gat_model, test_dataset, device)  
  
    # Calculate per-class AUROC  
    gin_aurocs = calculate_per_class_auroc(gin_preds, labels)  
    gat_aurocs = calculate_per_class_auroc(gat_preds, labels)  
  
    # Analyze frequencies  
    freq_df = analyze_class_frequency(labels, class_names)  
  
    # Create visualizations passing results_dir  
    create_performance_heatmap(gin_aurocs, gat_aurocs, class_names, results_dir)  
    difficulty_df = create_difficulty_ranking(gin_aurocs, gat_aurocs, class_names, freq_df, results_dir)  
      
    # Enhanced analysis with statistical tests  
    rare_statistical_tests, frequent_statistical_tests, rare_effect_sizes, frequent_effect_sizes = analyze_rare_vs_frequent(difficulty_df, results_dir)
  
    # Save enhanced results using absolute path  
    results = {  
        'class_names': class_names,  
        'gin_aurocs': gin_aurocs.tolist(),  
        'gat_aurocs': gat_aurocs.tolist(),  
        'difficulty_ranking': difficulty_df.to_dict('records'),  
        'frequency_analysis': freq_df.to_dict('records'),  
        # Enhanced statistical metrics  
        'rare_classes_statistical_tests': {  
            'effect_sizes': rare_effect_sizes,   
            'wilcoxon_p_value': float(rare_statistical_tests['wilcoxon']['p_value']),  
            'mannwhitney_p_value': float(rare_statistical_tests['mannwhitney']['p_value']),  
            'permutation_p_value': float(rare_statistical_tests['permutation']['p_value'])  
        },  
        'frequent_classes_statistical_tests': {  
            'effect_sizes': frequent_effect_sizes,  
            'wilcoxon_p_value': float(frequent_statistical_tests['wilcoxon']['p_value']),  
            'mannwhitney_p_value': float(frequent_statistical_tests['mannwhitney']['p_value']),  
            'permutation_p_value': float(frequent_statistical_tests['permutation']['p_value'])  
        }  
    }  
  
    results_file = os.path.join(results_dir, 'enhanced_detailed_performance_results.json')  
    with open(results_file, 'w') as f:  
        json.dump(results, f, indent=2)  
  
    print(f"\nEnhanced results saved in '{results_file}'")  
  
if __name__ == "__main__":  
    main()