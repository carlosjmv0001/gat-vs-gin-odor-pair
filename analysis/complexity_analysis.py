import torch  
import numpy as np  
import matplotlib.pyplot as plt  
import time  
import psutil  
import gc  
import json  
import os  
import sys  
from tqdm import tqdm  
  
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
  
def measure_training_time(model, train_loader, device, epochs=5):  
    """Measure training time per epoch"""  
    model.train()  
    loss_fn = torch.nn.BCEWithLogitsLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
      
    epoch_times = []  
      
    for epoch in range(epochs):  
        start_time = time.time()  
          
        for batch_data in train_loader:  
            batch_data.to(device)  
            optimizer.zero_grad()  
            pred = model(**batch_data.to_dict())  
            loss = loss_fn(pred, batch_data.y)  
            loss.backward()  
            optimizer.step()  
          
        epoch_time = time.time() - start_time  
        epoch_times.append(epoch_time)  
      
    return np.mean(epoch_times), np.std(epoch_times)  
  
def measure_memory_usage(model, test_loader, device):  
    """Measure GPU memory usage during inference"""  
    if not torch.cuda.is_available():  
        return 0.0, "CPU only"  
      
    model.eval()  
    torch.cuda.empty_cache()  
    gc.collect()  
      
    # Measure baseline memory  
    baseline_memory = torch.cuda.memory_allocated()  
      
    # Run inference and measure peak memory  
    peak_memory = baseline_memory  
      
    with torch.no_grad():  
        for batch_data in test_loader:  
            batch_data.to(device)  
            _ = model(**batch_data.to_dict())  
            current_memory = torch.cuda.memory_allocated()  
            peak_memory = max(peak_memory, current_memory)  
      
    memory_used = peak_memory - baseline_memory  
    memory_mb = memory_used / (1024 * 1024)  
      
    return memory_mb, f"{memory_mb:.2f} MB"  
  
def find_max_batch_size(model, device, max_batch_size=512):  
    """Find maximum batch size that fits in GPU memory"""  
    print("Finding maximum batch size...")  
      
    # Get actual data dimensions from a real sample  
    test_dataset = Dataset(is_train=False)  
    test_loader = loader(test_dataset, batch_size=1)  
    real_batch = next(iter(test_loader)).to(device)  
      
    # Use real dimensions for dummy data  
    num_nodes_s = real_batch.x_s.size(0)  
    num_nodes_t = real_batch.x_t.size(0)  
    num_edges_s = real_batch.edge_index_s.size(1)  
    num_edges_t = real_batch.edge_index_t.size(1)  
    num_features = real_batch.x_s.size(1)  
    num_edge_features = real_batch.edge_attr_s.size(1)  
    num_classes = real_batch.y.size(1)  
      
    max_size = 0  
    batch_size = 1  
      
    while batch_size <= max_batch_size:  
        try:  
            # Create dummy batch with correct dimensions  
            dummy_batch = {  
                'x_s': torch.randn(batch_size * num_nodes_s, num_features).to(device),  
                'x_t': torch.randn(batch_size * num_nodes_t, num_features).to(device),  
                'edge_index_s': torch.randint(0, batch_size * num_nodes_s, (2, batch_size * num_edges_s)).to(device),  
                'edge_index_t': torch.randint(0, batch_size * num_nodes_t, (2, batch_size * num_edges_t)).to(device),  
                'edge_attr_s': torch.randn(batch_size * num_edges_s, num_edge_features).to(device),  
                'edge_attr_t': torch.randn(batch_size * num_edges_t, num_edge_features).to(device),  
                'x_s_batch': torch.repeat_interleave(torch.arange(batch_size), num_nodes_s).to(device),  
                'x_t_batch': torch.repeat_interleave(torch.arange(batch_size), num_nodes_t).to(device),  
                'y': torch.randn(batch_size, num_classes).to(device)  
            }  
              
            _ = model(**dummy_batch)  
            max_size = batch_size  
            batch_size *= 2  
            torch.cuda.empty_cache()  
              
        except RuntimeError as e:  
            if "out of memory" in str(e):  
                break  
            else:  
                raise e  
      
    return max_size, f"Max batch size: {max_size}" 
  
def measure_convergence_rates(model, train_loader, device, rare_classes_indices):  
    """Measure convergence speed on rare classes"""  
    model.train()  
    loss_fn = torch.nn.BCEWithLogitsLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
      
    convergence_metrics = []  
    epochs_to_track = 20  
      
    for epoch in range(epochs_to_track):  
        epoch_losses = []  
        rare_class_losses = []  
          
        for batch_data in train_loader:  
            batch_data.to(device)  
            optimizer.zero_grad()  
            pred = model(**batch_data.to_dict())  
              
            # Overall loss  
            total_loss = loss_fn(pred, batch_data.y)  
              
            # Rare class loss  
            rare_pred = pred[:, rare_classes_indices]  
            rare_true = batch_data.y[:, rare_classes_indices]  
            rare_loss = loss_fn(rare_pred, rare_true)  
              
            total_loss.backward()  
            optimizer.step()  
              
            epoch_losses.append(total_loss.item())  
            rare_class_losses.append(rare_loss.item())  
          
        convergence_metrics.append({  
            'epoch': epoch,  
            'overall_loss': np.mean(epoch_losses),  
            'rare_class_loss': np.mean(rare_class_losses)  
        })  
      
    return convergence_metrics  
  
def analyze_complexity():  
    """Complete computational complexity analysis"""  
    print("=== COMPUTATIONAL COMPLEXITY ANALYSIS ===")  
      
    # Create results directory  
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    results_dir = os.path.join(base_dir, "results")  
    os.makedirs(results_dir, exist_ok=True)  
      
    # Load models and data  
    gin_model, gat_model, device = load_models()  
      
    # Load datasets  
    train_dataset = Dataset(is_train=True)  
    test_dataset = Dataset(is_train=False)  
      
    print(f"Using device: {device}")  
    print(f"Train dataset size: {len(train_dataset)}")  
    print(f"Test dataset size: {len(test_dataset)}")  
      
    # 1. Training Time Analysis  
    print("\n1. TRAINING TIME ANALYSIS:")  
      
    # Use smaller batch sizes for fair comparison  
    gin_batch_size = 32  
    gat_batch_size = 16  # Smaller due to attention overhead  
      
    gin_train_loader = loader(train_dataset, batch_size=gin_batch_size)  
    gat_train_loader = loader(train_dataset, batch_size=gat_batch_size)  
      
    gin_time_mean, gin_time_std = measure_training_time(gin_model, gin_train_loader, device)  
    gat_time_mean, gat_time_std = measure_training_time(gat_model, gat_train_loader, device)  
      
    print(f"GIN training time: {gin_time_mean:.3f} ± {gin_time_std:.3f} seconds/epoch")  
    print(f"GAT training time: {gat_time_mean:.3f} ± {gat_time_std:.3f} seconds/epoch")  
    print(f"GAT/GIN time ratio: {gat_time_mean/gin_time_mean:.2f}x")  
      
    # 2. Memory Usage Analysis  
    print("\n2. MEMORY USAGE ANALYSIS:")  
      
    test_loader = loader(test_dataset, batch_size=32)  
      
    gin_memory, gin_memory_str = measure_memory_usage(gin_model, test_loader, device)  
    gat_memory, gat_memory_str = measure_memory_usage(gat_model, test_loader, device)  
      
    print(f"GIN memory usage: {gin_memory_str}")  
    print(f"GAT memory usage: {gat_memory_str}")  
    print(f"GAT/GIN memory ratio: {gat_memory/gin_memory:.2f}x")  
      
    # 3. Maximum Batch Size Analysis  
    print("\n3. MAXIMUM BATCH SIZE ANALYSIS:")  
      
    gin_max_batch, gin_batch_str = find_max_batch_size(gin_model, device)  
    gat_max_batch, gat_batch_str = find_max_batch_size(gat_model, device)  
      
    print(f"GIN max batch size: {gin_batch_str}")  
    print(f"GAT max batch size: {gat_batch_str}")  
    print(f"GIN/GAT batch ratio: {gin_max_batch/gat_max_batch:.2f}x")  
      
    # 4. Convergence Analysis (simplified)  
    print("\n4. CONVERGENCE ANALYSIS:")  
      
    # Identify rare classes (bottom 25% by frequency)  
    class_counts = train_dataset.y.sum(dim=0)  
    rare_threshold = np.percentile(class_counts.cpu().numpy(), 25)  
    rare_classes = torch.where(class_counts <= rare_threshold)[0].cpu().numpy()  
      
    print(f"Rare classes identified: {len(rare_classes)} (threshold: {rare_threshold:.0f})")  
      
    # Use smaller dataset for convergence analysis  
    small_train_dataset = torch.utils.data.Subset(train_dataset, range(min(1000, len(train_dataset))))  
    small_train_loader = loader(small_train_dataset, batch_size=16)  
      
    gin_convergence = measure_convergence_rates(gin_model, small_train_loader, device, rare_classes)  
    gat_convergence = measure_convergence_rates(gat_model, small_train_loader, device, rare_classes)  
      
    # Calculate convergence speed (epochs to reach 90% of final performance)  
    def calculate_convergence_speed(convergence_data):  
        losses = [m['rare_class_loss'] for m in convergence_data]  
        final_loss = losses[-1]  
        target_loss = final_loss + 0.1 * (losses[0] - final_loss)  # 90% improvement  
          
        for i, loss in enumerate(losses):  
            if loss <= target_loss:  
                return i + 1  
        return len(losses)  
      
    gin_convergence_speed = calculate_convergence_speed(gin_convergence)  
    gat_convergence_speed = calculate_convergence_speed(gat_convergence)  
      
    print(f"GIN convergence epochs: {gin_convergence_speed}")  
    print(f"GAT convergence epochs: {gat_convergence_speed}")  
      
    # 5. Create Visualizations  
    create_complexity_visualizations(  
        gin_time_mean, gat_time_mean,  
        gin_memory, gat_memory,  
        gin_max_batch, gat_max_batch,  
        gin_convergence, gat_convergence,  
        results_dir  
    )  
      
    # 6. Save Results  
    results = {  
        'training_time': {  
            'gin_mean': gin_time_mean,  
            'gin_std': gin_time_std,  
            'gat_mean': gat_time_mean,  
            'gat_std': gat_time_std,  
            'ratio': gat_time_mean / gin_time_mean  
        },  
        'memory_usage': {  
            'gin_mb': gin_memory,  
            'gat_mb': gat_memory,  
            'ratio': gat_memory / gin_memory if gin_memory > 0 else 1.0  
        },  
        'batch_size': {  
            'gin_max': gin_max_batch,  
            'gat_max': gat_max_batch,  
            'ratio': gin_max_batch / gat_max_batch  
        },  
        'convergence': {  
            'gin_epochs': gin_convergence_speed,  
            'gat_epochs': gat_convergence_speed,  
            'rare_classes_count': len(rare_classes)  
        },  
        'device': str(device),  
        'dataset_sizes': {  
            'train': len(train_dataset),  
            'test': len(test_dataset)  
        }  
    }  
      
    results_file = os.path.join(results_dir, 'complexity_analysis_results.json')  
    with open(results_file, 'w') as f:  
        json.dump(results, f, indent=2)  
      
    print(f"\nResults saved in '{results_file}'")  
    return results  
  
def create_complexity_visualizations(gin_time, gat_time, gin_mem, gat_mem,   
                                   gin_batch, gat_batch, gin_conv, gat_conv, results_dir):  
    """Create comprehensive complexity analysis visualizations"""  
      
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))  
      
    # Plot 1: Training Time Comparison  
    models = ['GIN', 'GAT']  
    times = [gin_time, gat_time]  
      
    bars = ax1.bar(models, times, alpha=0.8, color=['blue', 'red'])  
    ax1.set_ylabel('Training Time (seconds/epoch)')  
    ax1.set_title('Training Time Comparison')  
    ax1.grid(True, alpha=0.3)  
      
    # Add ratio annotation  
    ratio = gat_time / gin_time  
    ax1.text(0.5, max(times) * 0.9, f'GAT/GIN = {ratio:.2f}x',   
             ha='center', transform=ax1.transData, fontsize=12, fontweight='bold')  
      
    # Add values to bars  
    for bar, time in zip(bars, times):  
        height = bar.get_height()  
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,  
                f'{time:.3f}s', ha='center', va='bottom')  
      
    # Plot 2: Memory Usage Comparison  
    memories = [gin_mem, gat_mem]  
      
    bars = ax2.bar(models, memories, alpha=0.8, color=['blue', 'red'])  
    ax2.set_ylabel('Memory Usage (MB)')  
    ax2.set_title('GPU Memory Usage Comparison')  
    ax2.grid(True, alpha=0.3)  
      
    # Add ratio annotation  
    mem_ratio = gat_mem / gin_mem if gin_mem > 0 else 1.0  
    ax2.text(0.5, max(memories) * 0.9, f'GAT/GIN = {mem_ratio:.2f}x',   
             ha='center', transform=ax2.transData, fontsize=12, fontweight='bold')  
      
    # Add values to bars  
    for bar, mem in zip(bars, memories):  
        height = bar.get_height()  
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(memories)*0.02,  
                f'{mem:.1f}MB', ha='center', va='bottom')  
      
    # Plot 3: Maximum Batch Size  
    batches = [gin_batch, gat_batch]  
      
    bars = ax3.bar(models, batches, alpha=0.8, color=['blue', 'red'])  
    ax3.set_ylabel('Maximum Batch Size')  
    ax3.set_title('Maximum Batch Size Comparison')  
    ax3.grid(True, alpha=0.3)  
      
    # Add ratio annotation  
    batch_ratio = gin_batch / gat_batch  
    ax3.text(0.5, max(batches) * 0.9, f'GIN/GAT = {batch_ratio:.2f}x',   
             ha='center', transform=ax3.transData, fontsize=12, fontweight='bold')  
      
    # Add values to bars  
    for bar, batch in zip(bars, batches):  
        height = bar.get_height()  
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(batches)*0.02,  
                f'{batch}', ha='center', va='bottom')  
      
    # Plot 4: Convergence Speed  
    gin_epochs = [m['epoch'] for m in gin_conv]  
    gin_losses = [m['rare_class_loss'] for m in gin_conv]  
    gat_epochs = [m['epoch'] for m in gat_conv]  
    gat_losses = [m['rare_class_loss'] for m in gat_conv]  
      
    ax4.plot(gin_epochs, gin_losses, 'o-', label='GIN', color='blue', linewidth=2)  
    ax4.plot(gat_epochs, gat_losses, 'o-', label='GAT', color='red', linewidth=2)  
    ax4.set_xlabel('Epoch')  
    ax4.set_ylabel('Rare Class Loss')  
    ax4.set_title('Convergence Speed on Rare Classes')  
    ax4.legend()  
    ax4.grid(True, alpha=0.3)  
      
    plt.tight_layout()  
      
    complexity_path = os.path.join(results_dir, 'computational_complexity_analysis.png')  
    plt.savefig(complexity_path, dpi=300, bbox_inches='tight')  
    plt.close()  
  
def main():  
    """Main complexity analysis function"""  
      
    return analyze_complexity()  
  
if __name__ == "__main__":  
    main()