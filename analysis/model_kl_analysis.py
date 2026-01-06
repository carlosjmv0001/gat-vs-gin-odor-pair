import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import json
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairing.data import Dataset, loader, PairData
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential
from new1_main import MixturePredictor as GATMixturePredictor, GAT

sys.modules['__main__'].MixturePredictor = GINMixturePredictor  # For GIN model
sys.modules['__main__'].GCN = GCN
sys.modules['__main__'].GAT = GAT
sys.modules['__main__'].make_sequential = make_sequential

def load_models():
    """Load trained GIN and GAT models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gin_model = torch.load(os.path.join(base_dir, "runs/1adc5394/model.pt"),
                          map_location=device, weights_only=False)

    sys.modules['__main__'].MixturePredictor = GATMixturePredictor
    gat_model = torch.load(os.path.join(base_dir, "runs/702a60ea/model.pt"),
                          map_location=device, weights_only=False)

    gin_model.eval()
    gat_model.eval()

    gin_model = gin_model.to(device)
    gat_model = gat_model.to(device)

    return gin_model, gat_model, device

def get_model_predictions(model, dataset, model_name, device):
    """Get predictions from a model for the entire dataset"""
    test_loader = loader(dataset, batch_size=128)
    all_preds = []
    all_true = []

    print(f"Generating predictions for {model_name} using {device}...")

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(**batch.to_dict())
            pred_probs = torch.sigmoid(pred)

            all_preds.append(pred_probs.cpu())
            all_true.append(batch.y.cpu())

    return torch.cat(all_preds, dim=0), torch.cat(all_true, dim=0)

def calculate_prediction_similarity(predictions, true_labels):
    """Calculate direct similarity between prediction distributions"""
    pred_dist = predictions.mean(dim=0)
    true_dist = true_labels.mean(dim=0)

    pred_dist = pred_dist / pred_dist.sum()
    true_dist = true_dist / true_dist.sum()

    eps = 1e-12
    pred_norm = (pred_dist + eps) / (pred_dist.sum() + eps)
    true_norm = (true_dist + eps) / (true_dist.sum() + eps)

    kl_div = torch.sum(pred_norm * torch.log(pred_norm / true_norm))
    kl_similarity = torch.exp(-kl_div)

    return kl_similarity.item(), pred_dist, true_dist

def analyze_models():
    """Main function for comparative analysis"""
    print("Loading models...")
    gin_model, gat_model, device = load_models()

    print("Loading test dataset...")
    test_dataset = Dataset(is_train=False)

    gin_preds, true_labels = get_model_predictions(gin_model, test_dataset, "GIN", device)
    gat_preds, _ = get_model_predictions(gat_model, test_dataset, "GAT", device)

    print("Calculating KL similarities...")

    gin_similarity, gin_dist, true_dist = calculate_prediction_similarity(gin_preds, true_labels)
    gat_similarity, gat_dist, _ = calculate_prediction_similarity(gat_preds, true_labels)

    eps = 1e-12
    gin_norm = (gin_dist + eps) / (gin_dist.sum() + eps)
    gat_norm = (gat_dist + eps) / (gat_dist.sum() + eps)
    gin_gat_kl = torch.sum(gin_norm * torch.log(gin_norm / gat_norm))
    gin_gat_similarity = torch.exp(-gin_gat_kl).item()

    print("\n=== KL ANALYSIS RESULTS ===")
    print(f"GIN vs True Data Similarity: {gin_similarity:.4f}")
    print(f"GAT vs True Data Similarity: {gat_similarity:.4f}")
    print(f"GIN vs GAT Similarity: {gin_gat_similarity:.4f}")

    return {
        'gin_similarity': gin_similarity,
        'gat_similarity': gat_similarity,
        'gin_gat_similarity': gin_gat_similarity,
        'gin_dist': gin_dist,
        'gat_dist': gat_dist,
        'true_dist': true_dist
    }

if __name__ == "__main__":
    print("Starting KL analysis...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results = analyze_models()

    results_file = os.path.join(results_dir, 'kl_analysis_results.json')
    with open(results_file, 'w') as f:
        results_serializable = {
            'gin_similarity': results['gin_similarity'],
            'gat_similarity': results['gat_similarity'],
            'gin_gat_similarity': results['gin_gat_similarity'],
            'gin_dist': results['gin_dist'].tolist(),
            'gat_dist': results['gat_dist'].tolist(),
            'true_dist': results['true_dist'].tolist()
        }
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved in '{results_file}'")