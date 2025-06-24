import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
from scipy.stats import entropy
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairing.data import Dataset, loader
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential
from new1_main import MixturePredictor as GATMixturePredictor, GAT

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

def calculate_entropy(predictions):
    """Calculate the entropy of predictions (uncertainty measure)"""
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    # Entropy for each sample
    # For multi-label, we can sum the Bernoulli entropy for each class
    # Or treat as a joint probability distribution (more complex)
    # Here, we calculate the Bernoulli entropy for each class and sum

    # Bernoulli entropy: -p*log(p) - (1-p)*log(1-p)
    per_class_entropy = -predictions * np.log(predictions) - (1 - predictions) * np.log(1 - predictions)
    total_entropy_per_sample = np.sum(per_class_entropy, axis=1)

    return total_entropy_per_sample

def analyze_uncertainty():
    """Full uncertainty analysis"""
    print("=== UNCERTAINTY ANALYSIS ===")

    # Create results directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load models and data
    gin_model, gat_model, device = load_models()
    test_dataset = Dataset(is_train=False)

    # Get predictions
    gin_preds, labels = get_predictions_and_labels(gin_model, test_dataset, device)
    gat_preds, _ = get_predictions_and_labels(gat_model, test_dataset, device)

    # Calculate prediction entropy
    gin_entropy = calculate_entropy(gin_preds)
    gat_entropy = calculate_entropy(gat_preds)

    print(f"GIN - Mean Entropy: {np.mean(gin_entropy):.4f}, Std: {np.std(gin_entropy):.4f}")
    print(f"GAT - Mean Entropy: {np.mean(gat_entropy):.4f}, Std: {np.std(gat_entropy):.4f}")

    # Visualize entropy distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(gin_entropy, bins=50, alpha=0.7, label='GIN', density=True)
    ax.hist(gat_entropy, bins=50, alpha=0.7, label='GAT', density=True)
    ax.set_xlabel('Prediction Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Uncertainty Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'prediction_uncertainty_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {os.path.join(results_dir, 'prediction_uncertainty_distribution.png')}")

    # Ambiguous cases analysis (high uncertainty)
    # Define an entropy threshold (e.g., top 10% most uncertain)
    gin_uncertain_threshold = np.percentile(gin_entropy, 90)
    gat_uncertain_threshold = np.percentile(gat_entropy, 90)

    gin_uncertain_indices = np.where(gin_entropy >= gin_uncertain_threshold)[0]
    gat_uncertain_indices = np.where(gat_entropy >= gat_uncertain_threshold)[0]

    print(f"\nGIN - Number of high uncertainty samples (top 10%): {len(gin_uncertain_indices)}")
    print(f"GAT - Number of high uncertainty samples (top 10%): {len(gat_uncertain_indices)}")

    # Optionally: Save examples of uncertain samples for manual analysis
    # E.g.: save indices or predictions/labels

    # Save results
    results = {
        'gin_mean_entropy': float(np.mean(gin_entropy)),
        'gat_mean_entropy': float(np.mean(gat_entropy)),
        'gin_std_entropy': float(np.std(gin_entropy)),
        'gat_std_entropy': float(np.std(gat_entropy)),
        'gin_uncertain_samples_count': len(gin_uncertain_indices),
        'gat_uncertain_samples_count': len(gat_uncertain_indices)
    }

    with open(os.path.join(results_dir, 'uncertainty_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in '{os.path.join(results_dir, 'uncertainty_analysis_results.json')}'")

if __name__ == "__main__":
    analyze_uncertainty()