import torch  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd  
from rdkit import Chem  
from rdkit.Chem import Descriptors  
import sys  
import json  
import os  
from sklearn.metrics import log_loss, brier_score_loss  
from scipy import stats  
  
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
    """Get predictions and true labels with SMILES"""  
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
  
    return (torch.cat(all_preds, dim=0).numpy(),   
            torch.cat(all_labels, dim=0).numpy(),  
            all_smiles_s, all_smiles_t)  
  
def calculate_molecular_properties(smiles_list):  
    """Calculate molecular properties for SMILES list"""  
    properties = {}  
    smiles_processed = []  
      
    for smiles in smiles_list:  
        mol = Chem.MolFromSmiles(smiles)  
        if mol is None:  
            properties[smiles] = {'MolWt': np.nan}  
            smiles_processed.append(smiles)  
            continue  
          
        properties[smiles] = {  
            'MolWt': Descriptors.MolWt(mol)  
        }  
        smiles_processed.append(smiles)  
      
    return properties  
  
def calculate_per_pair_metrics(predictions, labels):  
    """Calculate error metrics for each pair"""  
    # BCE loss per sample  
    bce_per_sample = []  
    for i in range(len(predictions)):  
        sample_bce = log_loss(labels[i], predictions[i], labels=[0, 1])  
        bce_per_sample.append(sample_bce)  
      
    # Brier score per sample (average over classes)  
    brier_per_sample = []  
    for i in range(len(predictions)):  
        sample_brier = brier_score_loss(labels[i], predictions[i])  
        brier_per_sample.append(sample_brier)  
      
    # Entropy per sample  
    epsilon = 1e-10  
    predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)  
    entropy_per_sample = []  
    for i in range(len(predictions)):  
        sample_entropy = -np.sum(predictions_clipped[i] * np.log(predictions_clipped[i]) +   
                                (1 - predictions_clipped[i]) * np.log(1 - predictions_clipped[i]))  
        entropy_per_sample.append(sample_entropy)  
      
    return np.array(bce_per_sample), np.array(brier_per_sample), np.array(entropy_per_sample)  
  
def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):  
    """Calculate bootstrap confidence interval"""  
    n = len(data)  
    bootstrap_means = []  
      
    for _ in range(n_bootstrap):  
        bootstrap_sample = np.random.choice(data, size=n, replace=True)  
        bootstrap_means.append(np.mean(bootstrap_sample))  
      
    alpha = 1 - confidence  
    lower_percentile = (alpha / 2) * 100  
    upper_percentile = (1 - alpha / 2) * 100  
      
    ci_lower = np.percentile(bootstrap_means, lower_percentile)  
    ci_upper = np.percentile(bootstrap_means, upper_percentile)  
      
    return ci_lower, ci_upper  
  
def analyze_molecular_weight_error():  
    """Analyze error metrics vs molecular weight"""  
    print("=== MOLECULAR WEIGHT ERROR ANALYSIS ===")  
      
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
      
    # Load models and data  
    gin_model, gat_model, device = load_models()  
    test_dataset = Dataset(is_train=False)  
      
    # Get predictions  
    gin_preds, labels, smiles_s, smiles_t = get_predictions_and_labels(gin_model, test_dataset, device)  
    gat_preds, _, _, _ = get_predictions_and_labels(gat_model, test_dataset, device)  
      
    # Calculate molecular properties  
    all_smiles = list(set(smiles_s + smiles_t))  
    mol_properties = calculate_molecular_properties(all_smiles)  
      
    # Calculate per-pair metrics  
    gin_bce, gin_brier, gin_entropy = calculate_per_pair_metrics(gin_preds, labels)  
    gat_bce, gat_brier, gat_entropy = calculate_per_pair_metrics(gat_preds, labels)  
      
    # Calculate MW_total and MW_max for each pair  
    mw_total = []  
    mw_max = []  
    valid_indices = []  
      
    for i, (s, t) in enumerate(zip(smiles_s, smiles_t)):  
        mw_s = mol_properties.get(s, {}).get('MolWt', np.nan)  
        mw_t = mol_properties.get(t, {}).get('MolWt', np.nan)  
          
        if not np.isnan(mw_s) and not np.isnan(mw_t):  
            mw_total.append(mw_s + mw_t)  
            mw_max.append(max(mw_s, mw_t))  
            valid_indices.append(i)  
      
    # Filter valid data  
    valid_indices = np.array(valid_indices)  
    mw_total = np.array(mw_total)  
    mw_max = np.array(mw_max)  
      
    gin_bce_valid = gin_bce[valid_indices]  
    gat_bce_valid = gat_bce[valid_indices]  
    gin_brier_valid = gin_brier[valid_indices]  
    gat_brier_valid = gat_brier[valid_indices]  
    gin_entropy_valid = gin_entropy[valid_indices]  
    gat_entropy_valid = gat_entropy[valid_indices]  
      
    # Create bins for MW_total  
    n_bins = 10  
    mw_bins = np.percentile(mw_total, np.linspace(0, 100, n_bins + 1))  
    bin_centers = (mw_bins[:-1] + mw_bins[1:]) / 2  
      
    # Calculate binned statistics with confidence intervals  
    gin_bce_binned = []  
    gat_bce_binned = []  
    gin_bce_ci_lower = []  
    gin_bce_ci_upper = []  
    gat_bce_ci_lower = []  
    gat_bce_ci_upper = []  
      
    for i in range(n_bins):  
        mask = (mw_total >= mw_bins[i]) & (mw_total < mw_bins[i + 1])  
        if i == n_bins - 1:  # Include last bin boundary  
            mask = (mw_total >= mw_bins[i]) & (mw_total <= mw_bins[i + 1])  
          
        if np.sum(mask) > 0:  
            gin_bin_data = gin_bce_valid[mask]  
            gat_bin_data = gat_bce_valid[mask]  
              
            gin_bce_binned.append(np.mean(gin_bin_data))  
            gat_bce_binned.append(np.mean(gat_bin_data))  
              
            gin_ci_lower, gin_ci_upper = bootstrap_confidence_interval(gin_bin_data)  
            gat_ci_lower, gat_ci_upper = bootstrap_confidence_interval(gat_bin_data)  
              
            gin_bce_ci_lower.append(gin_ci_lower)  
            gin_bce_ci_upper.append(gin_ci_upper)  
            gat_bce_ci_lower.append(gat_ci_lower)  
            gat_bce_ci_upper.append(gat_ci_upper)  
        else:  
            gin_bce_binned.append(np.nan)  
            gat_bce_binned.append(np.nan)  
            gin_bce_ci_lower.append(np.nan)  
            gin_bce_ci_upper.append(np.nan)  
            gat_bce_ci_lower.append(np.nan)  
            gat_bce_ci_upper.append(np.nan)  
      
    # Create visualization  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
      
    # Plot 1: BCE vs MW_total with confidence intervals  
    ax1.errorbar(bin_centers, gin_bce_binned,   
                 yerr=[np.array(gin_bce_binned) - np.array(gin_bce_ci_lower),  
                       np.array(gin_bce_ci_upper) - np.array(gin_bce_binned)],  
                 fmt='o-', capsize=5, capthick=2, label='GIN', alpha=0.8)  
    ax1.errorbar(bin_centers, gat_bce_binned,  
                 yerr=[np.array(gat_bce_binned) - np.array(gat_bce_ci_lower),  
                       np.array(gat_bce_ci_upper) - np.array(gat_bce_binned)],  
                 fmt='o-', capsize=5, capthick=2, label='GAT', alpha=0.8)  
      
    ax1.set_xlabel('Total Molecular Weight (MW1 + MW2)')  
    ax1.set_ylabel('Binary Cross-Entropy Loss')  
    ax1.set_title('Per-Pair Log Loss Stratified by Molecular Weight Quantiles')  
    ax1.legend()  
    ax1.grid(True, alpha=0.3)  
      
    # Plot 2: Scatter plot with regression lines  
    ax2.scatter(mw_total, gin_bce_valid, alpha=0.5, s=20, label='GIN')  
    ax2.scatter(mw_total, gat_bce_valid, alpha=0.5, s=20, label='GAT')  
      
    # Add regression lines  
    z_gin = np.polyfit(mw_total, gin_bce_valid, 1)  
    p_gin = np.poly1d(z_gin)  
    z_gat = np.polyfit(mw_total, gat_bce_valid, 1)  
    p_gat = np.poly1d(z_gat)  
      
    ax2.plot(mw_total, p_gin(mw_total), "r--", alpha=0.8, label=f'GIN trend (slope={z_gin[0]:.6f})')  
    ax2.plot(mw_total, p_gat(mw_total), "g--", alpha=0.8, label=f'GAT trend (slope={z_gat[0]:.6f})')  
      
    ax2.set_xlabel('Total Molecular Weight (MW1 + MW2)')  
    ax2.set_ylabel('Binary Cross-Entropy Loss')  
    ax2.set_title('Per-Pair Error vs Molecular Weight')  
    ax2.legend()  
    ax2.grid(True, alpha=0.3)  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(results_dir, 'molecular_weight_error_analysis.png'),   
                dpi=300, bbox_inches='tight')  
    plt.close()  
      
    # Save results  
    results = {  
        'gin_bce_mean': float(np.mean(gin_bce_valid)),  
        'gat_bce_mean': float(np.mean(gat_bce_valid)),  
        'gin_brier_mean': float(np.mean(gin_brier_valid)),  
        'gat_brier_mean': float(np.mean(gat_brier_valid)),  
        'gin_entropy_mean': float(np.mean(gin_entropy_valid)),  
        'gat_entropy_mean': float(np.mean(gat_entropy_valid)),  
        'mw_correlation_gin': float(np.corrcoef(mw_total, gin_bce_valid)[0, 1]),  
        'mw_correlation_gat': float(np.corrcoef(mw_total, gat_bce_valid)[0, 1])  
    }  
      
    with open(os.path.join(results_dir, 'molecular_weight_error_results.json'), 'w') as f:  
        json.dump(results, f, indent=2)  
      
    print(f"GIN - BCE: {results['gin_bce_mean']:.4f}, MW correlation: {results['mw_correlation_gin']:.4f}")  
    print(f"GAT - BCE: {results['gat_bce_mean']:.4f}, MW correlation: {results['mw_correlation_gat']:.4f}")  
    print(f"\nResults saved in '{os.path.join(results_dir, 'molecular_weight_error_results.json')}'")  
  
if __name__ == "__main__":  
    analyze_molecular_weight_error()