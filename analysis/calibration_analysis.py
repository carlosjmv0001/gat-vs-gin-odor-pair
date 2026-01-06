import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.calibration import calibration_curve  
from sklearn.metrics import brier_score_loss  
import sys  
import json  
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
  
def calculate_ece(y_true, y_prob, n_bins=10):  
    """Calculate Expected Calibration Error"""  
    bin_boundaries = np.linspace(0, 1, n_bins + 1)  
    bin_lowers = bin_boundaries[:-1]  
    bin_uppers = bin_boundaries[1:]  
  
    ece = 0  
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):  
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)  
        prop_in_bin = in_bin.mean()  
  
        if prop_in_bin > 0:  
            accuracy_in_bin = y_true[in_bin].mean()  
            avg_confidence_in_bin = y_prob[in_bin].mean()  
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin  
  
    return ece  
  
def calculate_per_class_ece(y_true, y_prob, n_bins=10):  
    """Calculate ECE for each class individually"""  
    n_classes = y_true.shape[1]  
    ece_per_class = {}  
      
    for class_idx in range(n_classes):  
        class_true = y_true[:, class_idx]  
        class_prob = y_prob[:, class_idx]  
            
        if class_true.sum() == 0:  
            ece_per_class[class_idx] = 0.0  
            continue  
              
        ece = calculate_ece(class_true, class_prob, n_bins)  
        ece_per_class[class_idx] = ece  
      
    return ece_per_class  
  
def create_reliability_diagram(y_true, y_prob, model_name, n_bins=10):  
    """Create reliability diagram"""  
    fraction_of_positives, mean_predicted_value = calibration_curve(  
        y_true, y_prob, n_bins=n_bins, strategy='uniform'  
    )  
  
    fig, ax = plt.subplots(figsize=(8, 6))  
  
    ax.plot(mean_predicted_value, fraction_of_positives, "s-",  
            label=f'{model_name} (ECE = {calculate_ece(y_true, y_prob):.3f})')  
  
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")  
  
    ax.set_xlabel('Mean Predicted Confidence')  
    ax.set_ylabel('Fraction of Positives')  
    ax.set_title(f'Reliability Diagram - {model_name}')  
    ax.legend()  
    ax.grid(True, alpha=0.3)  
  
    return fig  
  
def analyze_calibration():  
    """Full calibration analysis"""  
    print("=== MODEL CALIBRATION ANALYSIS ===")  
  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
  
    gin_model, gat_model, device = load_models()  
    test_dataset = Dataset(is_train=False)  
  
    gin_preds, labels = get_predictions_and_labels(gin_model, test_dataset, device)  
    gat_preds, _ = get_predictions_and_labels(gat_model, test_dataset, device)  
  
    gin_preds_flat = gin_preds.flatten()  
    gat_preds_flat = gat_preds.flatten()  
    labels_flat = labels.flatten()  
  
    gin_ece = calculate_ece(labels_flat, gin_preds_flat)  
    gat_ece = calculate_ece(labels_flat, gat_preds_flat)  
  
    gin_brier = brier_score_loss(labels_flat, gin_preds_flat)  
    gat_brier = brier_score_loss(labels_flat, gat_preds_flat)  
  
    print(f"GIN - ECE: {gin_ece:.4f}, Brier Score: {gin_brier:.4f}")  
    print(f"GAT - ECE: {gat_ece:.4f}, Brier Score: {gat_brier:.4f}")  
  
    gin_ece_per_class = calculate_per_class_ece(labels, gin_preds)  
    gat_ece_per_class = calculate_per_class_ece(labels, gat_preds)  
  
    class_frequencies = labels.sum(axis=0)  
    freq_threshold = np.percentile(class_frequencies, 25) 
  
    rare_classes = np.where(class_frequencies <= freq_threshold)[0]  
    frequent_classes = np.where(class_frequencies > freq_threshold)[0]  
  
    gin_rare_ece = np.mean([gin_ece_per_class[i] for i in rare_classes])  
    gin_frequent_ece = np.mean([gin_ece_per_class[i] for i in frequent_classes])  
    gat_rare_ece = np.mean([gat_ece_per_class[i] for i in rare_classes])  
    gat_frequent_ece = np.mean([gat_ece_per_class[i] for i in frequent_classes])  
  
    print(f"GIN - ECE Classes Raras: {gin_rare_ece:.4f}, ECE Classes Frequentes: {gin_frequent_ece:.4f}")  
    print(f"GAT - ECE Classes Raras: {gat_rare_ece:.4f}, ECE Classes Frequentes: {gat_frequent_ece:.4f}")  
  
    fig1 = create_reliability_diagram(labels_flat, gin_preds_flat, "GIN")  
    plt.savefig(os.path.join(results_dir, 'gin_reliability_diagram.png'), dpi=300, bbox_inches='tight')  
    plt.close(fig1)  
  
    fig2 = create_reliability_diagram(labels_flat, gat_preds_flat, "GAT")  
    plt.savefig(os.path.join(results_dir, 'gat_reliability_diagram.png'), dpi=300, bbox_inches='tight')  
    plt.close(fig2)  
  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
  
    metrics = ['ECE', 'Brier Score']  
    gin_metrics = [gin_ece, gin_brier]  
    gat_metrics = [gat_ece, gat_brier]  
  
    x = np.arange(len(metrics))  
    width = 0.35  
  
    ax1.bar(x - width/2, gin_metrics, width, label='GIN', alpha=0.8)  
    ax1.bar(x + width/2, gat_metrics, width, label='GAT', alpha=0.8)  
    ax1.set_ylabel('Metric Value')  
    ax1.set_title('Calibration Metrics')  
    ax1.set_xticks(x)  
    ax1.set_xticklabels(metrics)  
    ax1.legend()  
    ax1.grid(True, alpha=0.3)  
  
    ax2.hist(gin_preds_flat, bins=50, alpha=0.7, label='GIN', density=True)  
    ax2.hist(gat_preds_flat, bins=50, alpha=0.7, label='GAT', density=True)  
    ax2.set_xlabel('Predicted Confidence')  
    ax2.set_ylabel('Density')  
    ax2.set_title('Confidence Distribution')  
    ax2.legend()  
    ax2.grid(True, alpha=0.3)  
  
    plt.tight_layout()  
    plt.savefig(os.path.join(results_dir, 'calibration_comparison.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
    fig3, ax = plt.subplots(figsize=(10, 6))  
    ax.scatter(class_frequencies,   
               [gin_ece_per_class[i] for i in range(len(class_frequencies))],   
               alpha=0.7, label='GIN', s=30)  
    ax.scatter(class_frequencies,   
               [gat_ece_per_class[i] for i in range(len(class_frequencies))],   
               alpha=0.7, label='GAT', s=30)  
    ax.set_xlabel('Class Frequency')  
    ax.set_ylabel('ECE')  
    ax.set_title('ECE vs Class Frequency')  
    ax.legend()  
    ax.grid(True, alpha=0.3)  
  
    plt.savefig(os.path.join(results_dir, 'ece_vs_frequency.png'), dpi=300, bbox_inches='tight')  
    plt.close(fig3)  
  
    results = {  
        'gin_ece': gin_ece,  
        'gat_ece': gat_ece,  
        'gin_brier': gin_brier,  
        'gat_brier': gat_brier,  
        'gin_ece_per_class': gin_ece_per_class,  
        'gat_ece_per_class': gat_ece_per_class,  
        'gin_rare_ece': gin_rare_ece,  
        'gin_frequent_ece': gin_frequent_ece,  
        'gat_rare_ece': gat_rare_ece,  
        'gat_frequent_ece': gat_frequent_ece  
    }  
  
    with open(os.path.join(results_dir, 'calibration_results.json'), 'w') as f:  
        json.dump(results, f, indent=2)  
  
    print(f"\nResults saved in '{os.path.join(results_dir, 'calibration_results.json')}'")  
  
if __name__ == "__main__":  
    analyze_calibration()