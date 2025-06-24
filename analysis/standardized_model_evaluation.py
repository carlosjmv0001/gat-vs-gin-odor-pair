import torch
import torch_geometric as pyg
import tqdm
import sys
import numpy as np
import json
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairing.data import PairData, Dataset, loader
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential
from new1_main import MixturePredictor as GATMixturePredictor, GAT

# Critical: Add the classes to the main module so PyTorch can find them during unpickling
sys.modules['__main__'].MixturePredictor = GINMixturePredictor
sys.modules['__main__'].GCN = GCN
sys.modules['__main__'].GAT = GAT
sys.modules['__main__'].make_sequential = make_sequential

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
    """Run standardized comparison between models"""
    print("=== STANDARDIZED MODEL EVALUATION ===")
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

    # Compare results
    print("\n=== COMPARISON RESULTS ===")
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

    # Create visualization
    create_comparison_visualization(gin_metrics, gat_metrics, auroc_diff_per_class)

    return gin_metrics, gat_metrics

def create_comparison_visualization(gin_metrics, gat_metrics, auroc_diff_per_class):
    """Create comparison visualizations"""
    # Get project base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Overall AUROC comparison
    models = ['GIN', 'GAT']
    micro_scores = [gin_metrics['auroc_micro'], gat_metrics['auroc_micro']]
    macro_scores = [gin_metrics['auroc_macro'], gat_metrics['auroc_macro']]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, micro_scores, width, label='AUROC Micro', alpha=0.8)
    ax1.bar(x + width/2, macro_scores, width, label='AUROC Macro', alpha=0.8)
    ax1.set_ylabel('AUROC Score')
    ax1.set_title('Overall AUROC Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Add values to bars
    for i, (micro, macro) in enumerate(zip(micro_scores, macro_scores)):
        ax1.text(i - width/2, micro + 0.01, f'{micro:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, macro + 0.01, f'{macro:.3f}', ha='center', va='bottom')

    # Plot 2: Per-class AUROC difference
    classes = np.arange(len(auroc_diff_per_class))
    colors = ['red' if diff < 0 else 'green' for diff in auroc_diff_per_class]

    ax2.bar(classes, auroc_diff_per_class, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('AUROC Difference (GAT - GIN)')
    ax2.set_title('Per-Class Performance Difference')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mean prediction distribution
    gin_mean_preds = np.array(gin_metrics['mean_predictions'])
    gat_mean_preds = np.array(gat_metrics['mean_predictions'])
    true_mean = np.array(gin_metrics['mean_true_labels'])

    ax3.scatter(true_mean, gin_mean_preds, alpha=0.6, label='GIN', s=30)
    ax3.scatter(true_mean, gat_mean_preds, alpha=0.6, label='GAT', s=30)
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
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

    comparison_path = os.path.join(results_dir, 'standardized_model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved as '{comparison_path}'")

if __name__ == "__main__":
    print("Starting standardized evaluation...")

    # Get project base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create results directory using absolute path
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Run standardized comparison
    gin_metrics, gat_metrics = compare_models()

    # Save detailed results using absolute path
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
        }
    }

    results_file_path = os.path.join(results_dir, 'standardized_evaluation_results.json')
    with open(results_file_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved in '{results_file_path}'")