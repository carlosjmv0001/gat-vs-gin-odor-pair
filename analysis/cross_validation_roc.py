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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairing.data import Dataset, loader, PairData, to_pairdata, get_all_notes
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential
from new1_main import MixturePredictor as GATMixturePredictor, GAT

# Critical: Add the classes to the main module so PyTorch can find them during unpickling
sys.modules['__main__'].MixturePredictor = GINMixturePredictor
sys.modules['__main__'].GCN = GCN
sys.modules['__main__'].GAT = GAT
sys.modules['__main__'].make_sequential = make_sequential

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

    # Plot 2: Statistical comparison
    gin_mean = np.mean(gin_aurocs)
    gat_mean = np.mean(gat_aurocs)

    # Calculate confidence intervals
    gin_ci = stats.t.interval(0.95, len(gin_aurocs)-1, loc=gin_mean, scale=stats.sem(gin_aurocs))
    gat_ci = stats.t.interval(0.95, len(gat_aurocs)-1, loc=gat_mean, scale=stats.sem(gat_aurocs))

    models = ['GIN', 'GAT']
    means = [gin_mean, gat_mean]

    # Correct error structure - should be [lower_errors, upper_errors]
    lower_errors = [gin_mean - gin_ci[0], gat_mean - gat_ci[0]]
    upper_errors = [gin_ci[1] - gin_mean, gat_ci[1] - gat_mean]
    errors = [lower_errors, upper_errors]

    bars = ax2.bar(models, means, yerr=errors, capsize=5, alpha=0.7, color=['blue', 'red'])
    ax2.set_ylabel('AUROC (Micro-Average)')
    ax2.set_title('AUROC Comparison with 95% Confidence Intervals', pad=15)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Add values to bars
    for bar, mean, ci in zip(bars, means, [gin_ci, gat_ci]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.4f}\n[{ci[0]:.4f}, {ci[1]:.4f}]',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cross_validation_roc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return gin_aurocs, gat_aurocs

def statistical_comparison(gin_aurocs, gat_aurocs):
    """Perform statistical test between models"""
    from scipy.stats import ttest_rel

    print("\n=== STATISTICAL ANALYSIS ===")
    print(f"GIN AUROC: {np.mean(gin_aurocs):.4f} ± {np.std(gin_aurocs):.4f}")
    print(f"GAT AUROC: {np.mean(gat_aurocs):.4f} ± {np.std(gat_aurocs):.4f}")

    # Paired t-test
    if len(gin_aurocs) > 1:
        t_stat, p_value = ttest_rel(gat_aurocs, gin_aurocs)
        print(f"\nPaired t-test:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("Statistically significant difference (p < 0.05)")
        else:
            print("No statistically significant difference (p >= 0.05)")
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

    # Save results using absolute path
    results = {
        'gin_aurocs': gin_aurocs,
        'gat_aurocs': gat_aurocs,
        'gin_mean': float(np.mean(gin_aurocs)),
        'gat_mean': float(np.mean(gat_aurocs)),
        'gin_std': float(np.std(gin_aurocs)),
        'gat_std': float(np.std(gat_aurocs))
    }

    results_file = os.path.join(results_dir, 'cross_validation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in '{results_file}'")
    print(f"Plot saved as '{os.path.join(results_dir, 'cross_validation_roc_comparison.png')}'")