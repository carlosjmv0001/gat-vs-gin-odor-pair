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
from collections import defaultdict
from sklearn.metrics import roc_auc_score

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

    # Get project base directory
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

def calculate_molecular_properties(smiles_list):
    """Calculate molecular properties for a list of SMILES"""
    properties = defaultdict(list)
    smiles_processed = [] # Store valid SMILES
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Handle invalid SMILES
            for desc in ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']:
                properties[desc].append(np.nan)
            smiles_processed.append(smiles) # Still add SMILES to keep alignment
            continue

        properties['MolWt'].append(Descriptors.MolWt(mol))
        properties['LogP'].append(Descriptors.MolLogP(mol))
        properties['NumHDonors'].append(Descriptors.NumHDonors(mol))
        properties['NumHAcceptors'].append(Descriptors.NumHAcceptors(mol))
        properties['TPSA'].append(Descriptors.TPSA(mol))
        smiles_processed.append(smiles)

    df = pd.DataFrame(properties)
    df['SMILES'] = smiles_processed
    return df

def analyze_molecular_properties():
    """Full molecular property analysis"""
    print("=== MOLECULAR PROPERTY ANALYSIS ===")

    # Create results directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    gin_model, gat_model, device = load_models()
    test_dataset = Dataset(is_train=False)

    # Get predictions, labels and SMILES
    gin_preds, labels, smiles_s, smiles_t = get_predictions_and_labels(gin_model, test_dataset, device)
    gat_preds, _, _, _ = get_predictions_and_labels(gat_model, test_dataset, device)

    # Calculate properties for both molecules in each pair
    all_smiles = list(set(smiles_s + smiles_t)) # Get unique SMILES
    mol_properties_df = calculate_molecular_properties(all_smiles)

    # Check if mol_properties_df is empty before proceeding
    if mol_properties_df.empty:
        print("No molecular property calculated. Check input SMILES.")
        # Save empty or error results to avoid full failure
        results = {
            'gin_auroc_per_sample_mean': np.nan,
            'gat_auroc_per_sample_mean': np.nan,
            'molecular_properties_summary': {}
        }
        with open(os.path.join(results_dir, 'molecular_property_analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved in '{os.path.join(results_dir, 'molecular_property_analysis_results.json')}'")
        return

    # Calculate per-class AUROC mean (not true per-sample AUROC)
    gin_auroc_per_sample = np.mean(roc_auc_score(labels, gin_preds, average=None, multi_class='ovr'), axis=0)
    gat_auroc_per_sample = np.mean(roc_auc_score(labels, gat_preds, average=None, multi_class='ovr'), axis=0)

    # Create a DataFrame for correlation
    correlation_data = pd.DataFrame({
        'smiles_s': smiles_s,
        'smiles_t': smiles_t,
        'gin_auroc': gin_auroc_per_sample,
        'gat_auroc': gat_auroc_per_sample
    })

    # Correct way to map properties:
    # First, create a SMILES-to-properties dict
    smiles_to_props = mol_properties_df.set_index('SMILES').to_dict('index')

    # Map properties to the correlation DataFrame
    correlation_data['molwt_s'] = correlation_data['smiles_s'].map(lambda s: smiles_to_props.get(s, {}).get('MolWt', np.nan))

    # Plot correlation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=correlation_data, x='molwt_s', y='gin_auroc', alpha=0.6, label='GIN')
    sns.scatterplot(data=correlation_data, x='molwt_s', y='gat_auroc', alpha=0.6, label='GAT')
    plt.xlabel('Molecular Weight (Molecule S)')
    plt.ylabel('Per-Sample AUROC')
    plt.title('Correlation between Molecular Weight and Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'molwt_performance_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {os.path.join(results_dir, 'molwt_performance_correlation.png')}")

    # Save results
    results = {
        'gin_auroc_per_sample_mean': float(np.mean(gin_auroc_per_sample)),
        'gat_auroc_per_sample_mean': float(np.mean(gat_auroc_per_sample)),
        'molecular_properties_summary': mol_properties_df.describe().to_dict()
    }

    with open(os.path.join(results_dir, 'molecular_property_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in '{os.path.join(results_dir, 'molecular_property_analysis_results.json')}'")

if __name__ == "__main__":
    analyze_molecular_properties()