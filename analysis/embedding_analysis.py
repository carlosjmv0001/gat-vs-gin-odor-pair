import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import sys
import json
import seaborn as sns
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

def get_embeddings_and_labels(model, dataset, device):
    """Get embeddings and true labels"""
    test_loader = loader(dataset, batch_size=128)
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            if hasattr(model, 'gcn'): # GIN model
                emb_s = model.gcn(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.x_s_batch)
                emb_t = model.gcn(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.x_t_batch)
            elif hasattr(model, 'gat'): # GAT model
                emb_s = model.gat(batch.x_s, batch.edge_index_s, batch.edge_attr_s, batch.x_s_batch)
                emb_t = model.gat(batch.x_t, batch.edge_index_t, batch.edge_attr_t, batch.x_t_batch)
            else:
                raise AttributeError("Model has no 'gcn' or 'gat' attribute for embedding extraction.")

            embedding = torch.cat([emb_s, emb_t], dim=1)
            all_embeddings.append(embedding.cpu())
            all_labels.append(batch.y.cpu())

    return torch.cat(all_embeddings, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()

def visualize_embeddings(embeddings, labels, model_name, title_suffix="", results_dir="results"):
    """Visualize embeddings using t-SNE"""
    print(f"Reducing dimensionality for {model_name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                         c=clusters, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Cluster')

    ax.set_title(f't-SNE of Embeddings - {model_name} {title_suffix}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(results_dir, f'{model_name.lower()}_embeddings_tsne_{title_suffix.replace(" ", "_")}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as: {filename}")

def analyze_embeddings():
    """Full embedding analysis"""
    print("=== EMBEDDING ANALYSIS ===")

    # Get project base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Create results directory using absolute path
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    gin_model, gat_model, device = load_models()
    test_dataset = Dataset(is_train=False)

    gin_embeddings, labels = get_embeddings_and_labels(gin_model, test_dataset, device)
    gat_embeddings, _ = get_embeddings_and_labels(gat_model, test_dataset, device)

    print(f"GIN embeddings shape: {gin_embeddings.shape}")
    print(f"GAT embeddings shape: {gat_embeddings.shape}")

    visualize_embeddings(gin_embeddings, labels, "GIN", "by Cluster", results_dir)
    visualize_embeddings(gat_embeddings, labels, "GAT", "by Cluster", results_dir)

    n_clusters = 5
    kmeans_gin = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(gin_embeddings)
    kmeans_gat = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(gat_embeddings)

    print(f"\nKMeans for GIN: Inertia = {kmeans_gin.inertia_:.2f}")
    print(f"KMeans for GAT: Inertia = {kmeans_gat.inertia_:.2f}")
    print("Incompatible dimensions - separate analysis required")
    print(f"GIN: {gin_embeddings.shape[1]} dimensions vs GAT: {gat_embeddings.shape[1]} dimensions")

    results = {
        'gin_embedding_shape': gin_embeddings.shape,
        'gat_embedding_shape': gat_embeddings.shape,
        'gin_kmeans_inertia': kmeans_gin.inertia_,
        'gat_kmeans_inertia': kmeans_gat.inertia_,
        'note': 'Incompatible dimensions for direct centroid comparison'
    }

    results_file = os.path.join(results_dir, 'embedding_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved in '{results_file}'")

if __name__ == "__main__":
    analyze_embeddings()