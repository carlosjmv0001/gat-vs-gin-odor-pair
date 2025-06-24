import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys
import json
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairing.data import Dataset, loader
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential
from new1_main import MixturePredictor as GATMixturePredictor, GAT

# Setup for model loading
sys.modules['__main__'].MixturePredictor = GINMixturePredictor
sys.modules['__main__'].GCN = GCN
sys.modules['__main__'].GAT = GAT
sys.modules['__main__'].make_sequential = make_sequential

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
    """Analyze performance on rare vs frequent classes"""
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

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Rare Classes', 'Frequent Classes']
    gin_means = [rare_classes['gin_auroc'].mean(), frequent_classes['gin_auroc'].mean()]
    gat_means = [rare_classes['gat_auroc'].mean(), frequent_classes['gat_auroc'].mean()]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, gin_means, width, label='GIN', alpha=0.8)
    ax.bar(x + width/2, gat_means, width, label='GAT', alpha=0.8)

    ax.set_ylabel('Mean AUROC')
    ax.set_title('Performance: Rare vs Frequent Classes')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add values to bars
    for i, (gin, gat) in enumerate(zip(gin_means, gat_means)):
        ax.text(i - width/2, gin + 0.01, f'{gin:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, gat + 0.01, f'{gat:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    analysis_path = os.path.join(results_dir, 'rare_vs_frequent_analysis.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rare vs frequent analysis saved as: {analysis_path}")

def main():
    print("=== DETAILED CLASS PERFORMANCE ANALYSIS ===")

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
    analyze_rare_vs_frequent(difficulty_df, results_dir)

    # Save results using absolute path
    results = {
        'class_names': class_names,
        'gin_aurocs': gin_aurocs.tolist(),
        'gat_aurocs': gat_aurocs.tolist(),
        'difficulty_ranking': difficulty_df.to_dict('records'),
        'frequency_analysis': freq_df.to_dict('records')
    }

    results_file = os.path.join(results_dir, 'detailed_performance_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in '{results_file}'")

if __name__ == "__main__":
    main()