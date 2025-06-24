import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
import sys
import json
from sklearn.metrics import roc_auc_score
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pairing.data import Dataset, loader

# Import model classes
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

    gin_model = torch.load(os.path.join(base_dir, "runs/1adc5394/model.pt"), map_location=device, weights_only=False)

    sys.modules['__main__'].MixturePredictor = GATMixturePredictor
    gat_model = torch.load(os.path.join(base_dir, "runs/702a60ea/model.pt"), map_location=device, weights_only=False)

    gin_model.eval()
    gat_model.eval()

    return gin_model.to(device), gat_model.to(device), device

def get_predictions_and_labels(model, dataset, device):
    """Get predictions and true labels"""
    test_loader = loader(dataset, batch_size=128)
    all_preds = []
    all_labels = []
    all_smiles_s = []
    all_smiles_t = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(**batch.to_dict())
            pred_probs = torch.sigmoid(pred)

            all_preds.append(pred_probs.cpu())
            all_labels.append(batch.y.cpu())
            all_smiles_s.extend(batch.smiles_s)
            all_smiles_t.extend(batch.smiles_t)

    return torch.cat(all_preds, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy(), all_smiles_s, all_smiles_t

def calculate_tanimoto_similarity(smiles1, smiles2):
    """Calculate Tanimoto similarity between two SMILES"""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return np.nan

    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

    return FingerprintSimilarity(fp1, fp2)

def analyze_molecular_pairs():
    """Full molecular pair analysis"""
    print("=== MOLECULAR PAIR ANALYSIS ===")

    # Create results directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    gin_model, gat_model, device = load_models()
    test_dataset = Dataset(is_train=False)

    # Get predictions, labels and SMILES
    gin_preds, labels, smiles_s, smiles_t = get_predictions_and_labels(gin_model, test_dataset, device)
    gat_preds, _, _, _ = get_predictions_and_labels(gat_model, test_dataset, device)

    # Calculate Tanimoto similarity for each pair
    tanimoto_similarities = []
    for i in range(len(smiles_s)):
        sim = calculate_tanimoto_similarity(smiles_s[i], smiles_t[i])
        tanimoto_similarities.append(sim)

    tanimoto_similarities = np.array(tanimoto_similarities)

    # Calculate per-sample AUROC (mean over classes)
    # Note: roc_auc_score expects 2D arrays for multi_class='ovr'
    gin_auroc_per_sample = np.array([roc_auc_score(labels[i:i+1, :], gin_preds[i:i+1, :], average='micro') for i in range(len(labels))])
    gat_auroc_per_sample = np.array([roc_auc_score(labels[i:i+1, :], gat_preds[i:i+1, :], average='micro') for i in range(len(labels))])

    # Create DataFrame for correlation
    correlation_data = pd.DataFrame({
        'tanimoto_similarity': tanimoto_similarities,
        'gin_auroc': gin_auroc_per_sample,
        'gat_auroc': gat_auroc_per_sample
    })

    # Remove NaNs (for invalid SMILES)
    correlation_data.dropna(inplace=True)

    # Plot correlation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=correlation_data, x='tanimoto_similarity', y='gin_auroc', alpha=0.6, label='GIN')
    sns.scatterplot(data=correlation_data, x='tanimoto_similarity', y='gat_auroc', alpha=0.6, label='GAT')
    plt.xlabel('Tanimoto Similarity between Pair Molecules')
    plt.ylabel('Per-Sample AUROC')
    plt.title('Correlation between Structural Similarity and Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'tanimoto_performance_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {os.path.join(results_dir, 'tanimoto_performance_correlation.png')}")

    # Synergy analysis (simplified example)
    # Identify pairs where performance is very high despite low similarity
    # Or where performance is low despite high similarity

    # Example: Pairs with low similarity but high performance
    low_sim_high_perf_gin = correlation_data[(correlation_data['tanimoto_similarity'] < 0.2) & (correlation_data['gin_auroc'] > 0.9)]
    low_sim_high_perf_gat = correlation_data[(correlation_data['tanimoto_similarity'] < 0.2) & (correlation_data['gat_auroc'] > 0.9)]

    print(f"\nGIN: {len(low_sim_high_perf_gin)} pairs with low similarity and high performance.")
    print(f"GAT: {len(low_sim_high_perf_gat)} pairs with low similarity and high performance.")

    results = {
        'mean_tanimoto_similarity': float(np.mean(tanimoto_similarities)),
        'gin_mean_auroc_per_sample': float(np.mean(gin_auroc_per_sample)),
        'gat_mean_auroc_per_sample': float(np.mean(gat_auroc_per_sample)),
        'low_sim_high_perf_gin_count': len(low_sim_high_perf_gin),
        'low_sim_high_perf_gat_count': len(low_sim_high_perf_gat)
    }

    with open(os.path.join(results_dir, 'molecular_pair_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in '{os.path.join(results_dir, 'molecular_pair_analysis_results.json')}'")

if __name__ == "__main__":
    analyze_molecular_pairs()