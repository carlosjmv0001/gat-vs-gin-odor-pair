import torch    
import numpy as np    
import pandas as pd    
import matplotlib.pyplot as plt    
import seaborn as sns    
from scipy import stats    
import json    
import os    
import sys    
from rdkit import Chem    
from rdkit.Chem import Descriptors    
    
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
    
from pairing.data import Dataset, loader    
from main import MixturePredictor as GINMixturePredictor, GCN, make_sequential    
from new1_main import MixturePredictor as GATMixturePredictor, GAT    
    
# Setup for model loading  
sys.modules['__main__'].MixturePredictor = GINMixturePredictor    
sys.modules['__main__'].GCN = GCN    
sys.modules['__main__'].GAT = GAT    
sys.modules['__main__'].make_sequential = make_sequential    
    
class DiagnosticFramework:    
    """Unified framework for diagnostic analysis of GIN vs GAT"""    
        
    def __init__(self):    
        # Expanded structural markers  
        self.structural_markers = {    
            'vanilla': ['lactones', 'esters', 'aldehydes'],    
            'sulfurous': ['thiols', 'sulfides', 'disulfides'],    
            'citrus': ['limonene', 'citral', 'terpenes'],    
            'minty': ['menthol', 'carvone', 'terpenes'],    
            'camphoreous': ['camphor', 'borneol', 'terpenes'],    
            'alliaceous': ['allicin', 'diallyl_disulfide', 'organosulfur'],    
            'buttery': ['diacetyl', 'lactones', 'esters'],    
            'creamy': ['lactones', 'esters', 'fatty_acids'],    
            'caramellic': ['furans', 'pyrazines', 'sugars'],    
            'coffee': ['pyrazines', 'furans', 'phenols'],    
            'soapy': ['esters', 'alcohols', 'fatty_acids'],    
            'amber': ['terpenes', 'resins', 'benzoin'],    
            'berry': ['esters', 'ketones', 'lactones'],    
            'fermented': ['acids', 'esters', 'alcohols'],    
            'cooling': ['menthol', 'eucalyptol', 'camphor'],    
            'tropical': ['esters', 'lactones', 'terpenes'],    
            'nutty': ['pyrazines', 'lactones', 'aldehydes'],    
            'waxy': ['long_chain_alkanes', 'esters', 'alcohols'],    
            'herbal': ['terpenes', 'phenols', 'aldehydes'],    
            'animal': ['indoles', 'sulfur_compounds', 'steroids'],    
            'honey': ['sugars', 'furans', 'esters'],    
            'fatty': ['fatty_acids', 'esters', 'aldehydes'],    
            'ethereal': ['esters', 'aldehydes', 'acetates'],    
            'aldehydic': ['aldehydes', 'acetates'],    
            'balsamic': ['resins', 'esters', 'phenols']    
        }    
            
        # Expanded conceptual aggregates  
        self.conceptual_aggregates = {    
            'fruity': 'combination of sweet, sour, aromatic notes',    
            'floral': 'complex blend of volatile compounds',    
            'woody': 'mixture of terpenes, phenols, resins',    
            'earthy': 'geosmin, minerals, organic compounds',    
            'musk': 'macrocycle compounds, animalic notes',    
            'spicy': 'various pungent compounds',    
            'green': 'leaf aldehydes, grassy compounds',    
            'musty': 'aged, damp, moldy characteristics'    
        }    
        
    def load_models_and_data(self):    
        """Load models and data consistently with other analyses"""    
        device = "cuda" if torch.cuda.is_available() else "cpu"    
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
            
        # Load GIN model  
        gin_model = torch.load(os.path.join(base_dir, "runs/1adc5394/model.pt"),    
                              map_location=device, weights_only=False)    
            
        # Load GAT model  
        sys.modules['__main__'].MixturePredictor = GATMixturePredictor    
        gat_model = torch.load(os.path.join(base_dir, "runs/702a60ea/model.pt"),    
                              map_location=device, weights_only=False)    
            
        gin_model.eval()    
        gat_model.eval()    
            
        # Load test dataset  
        test_dataset = Dataset(is_train=False)    
            
        return gin_model.to(device), gat_model.to(device), device, test_dataset    
        
    def get_predictions_and_labels(self, model, dataset, device):    
        """Get predictions and labels - consistent pattern"""    
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
        
    def create_high_impact_table(self, results_file):    
        """Create high-impact table with requested metrics"""    
        with open(results_file, 'r') as f:    
            results = json.load(f)    
            
        df_data = []    
        for i, note in enumerate(results['class_names']):    
            gin_auroc = results['gin_aurocs'][i]    
            gat_auroc = results['gat_aurocs'][i]    
            delta = gat_auroc - gin_auroc    
                
            # Find frequency  
            freq_info = next((item for item in results['frequency_analysis']     
                            if item['class'] == note), None)    
            frequency = freq_info['frequency'] if freq_info else 0    
                
            # Expanded classification
            if note in self.structural_markers:    
                note_type = 'structural_marker'    
            elif note in self.conceptual_aggregates:    
                note_type = 'conceptual_aggregate'    
            else:    
                note_type = 'unknown'
                
            df_data.append({    
                'Note': note,    
                'AUROC_GIN': gin_auroc,    
                'AUROC_GAT': gat_auroc,    
                'Delta': delta,    
                'Frequency': frequency,    
                'Delta_Rank': 0,    
                'Type': note_type    
            })    
            
        df = pd.DataFrame(df_data)    
        df['Delta_Rank'] = df['Delta'].abs().rank(ascending=False)    
            
        return df.sort_values('Delta_Rank')    
        
    def analyze_inductive_bias(self, df):    
        """Analyze inductive bias patterns"""    
        # Statistical analysis  
        delta_stats = {    
            'mean_delta': df['Delta'].mean(),    
            'std_delta': df['Delta'].std(),    
            'large_positive_deltas': len(df[df['Delta'] > 0.1]),    
            'large_negative_deltas': len(df[df['Delta'] < -0.1]),    
            'statistical_significance': stats.ttest_1samp(df['Delta'], 0)    
        }    
            
        # Identify clusters  
        gat_dominated = df[df['Delta'] > 0.05].sort_values('Delta', ascending=False)    
        gin_dominated = df[df['Delta'] < -0.05].sort_values('Delta')    
            
        return {    
            'delta_statistics': delta_stats,    
            'gat_dominated_notes': gat_dominated,    
            'gin_dominated_notes': gin_dominated,    
            'high_impact_notes': df[df['Delta'].abs() > 0.1]    
        }    
        
    def design_conditional_strategy(self, bias_analysis):    
        """Design conditional selection strategy"""    
        GAT_THRESHOLD = 0.05    
        GIN_THRESHOLD = -0.05    
            
        strategy = {    
            'gat_preferred_notes': [],    
            'gin_preferred_notes': [],    
            'ambiguous_notes': []    
        }    
            
        for _, row in bias_analysis['high_impact_notes'].iterrows():    
            if row['Delta'] > GAT_THRESHOLD:    
                strategy['gat_preferred_notes'].append(row['Note'])    
            elif row['Delta'] < GIN_THRESHOLD:    
                strategy['gin_preferred_notes'].append(row['Note'])    
            else:    
                strategy['ambiguous_notes'].append(row['Note'])    
            
        return strategy    
        
    def simulate_conditional_performance(self, df, strategy):    
        """Simulate conditional strategy performance"""    
        conditional_aurocs = []    
            
        for _, row in df.iterrows():    
            if row['Note'] in strategy['gat_preferred_notes']:    
                conditional_aurocs.append(row['AUROC_GAT'])    
            elif row['Note'] in strategy['gin_preferred_notes']:    
                conditional_aurocs.append(row['AUROC_GIN'])    
            else:    
                conditional_aurocs.append((row['AUROC_GIN'] + row['AUROC_GAT']) / 2)    
            
        conditional_mean = np.mean(conditional_aurocs)    
        gin_mean = df['AUROC_GIN'].mean()    
        gat_mean = df['AUROC_GAT'].mean()    
            
        return {    
            'conditional_mean_auroc': conditional_mean,    
            'improvement_over_gin': conditional_mean - gin_mean,    
            'improvement_over_gat': conditional_mean - gat_mean    
        }    
        
    def create_visualizations(self, df, bias_analysis, results_dir):    
        """Create diagnostic visualizations"""    
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))    
            
        # 1. Delta plot by type  
        colors = {'structural_marker': 'blue', 'conceptual_aggregate': 'red', 'unknown': 'gray'}    
        for note_type in df['Type'].unique():    
            subset = df[df['Type'] == note_type]    
            ax1.scatter(subset.index, subset['Delta'],     
                       c=colors[note_type], label=note_type.replace('_', ' ').title(), alpha=0.7)    
            
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)    
        ax1.set_xlabel('Note Index')    
        ax1.set_ylabel('AUROC Difference (GAT - GIN)')    
        ax1.set_title('Performance Differences by Note Type')    
        ax1.legend()    
        ax1.grid(True, alpha=0.3)    
            
        # 2. High impact table visualization 
        high_impact = bias_analysis['high_impact_notes'].head(5) 
        y_pos = np.arange(len(high_impact))    
        ax2.barh(y_pos, high_impact['Delta'],     
                color=['green' if d > 0 else 'red' for d in high_impact['Delta']])    
        ax2.set_yticks(y_pos)    
        ax2.set_yticklabels(high_impact['Note'])    
        ax2.set_xlabel('AUROC Difference (GAT - GIN)')    
        ax2.set_title('Top 5 High-Impact Notes') 
        ax2.grid(True, alpha=0.3)    
            
        # 3. Architecture dominance  
        gat_count = len(bias_analysis['gat_dominated_notes'])    
        gin_count = len(bias_analysis['gin_dominated_notes'])    
        ambiguous_count = len(df) - gat_count - gin_count    
            
        ax3.pie([gat_count, gin_count, ambiguous_count],     
               labels=['GAT-Dominated', 'GIN-Dominated', 'Ambiguous'],    
               colors=['red', 'blue', 'gray'], autopct='%1.1f%%')    
        ax3.set_title('Architecture Dominance Distribution')    
            
        # 4. Frequency vs Delta correlation  
        ax4.scatter(df['Frequency'], df['Delta'], alpha=0.6)    
        ax4.set_xlabel('Class Frequency')    
        ax4.set_ylabel('AUROC Difference (GAT - GIN)')    
        ax4.set_title('Frequency vs Performance Difference')    
        ax4.grid(True, alpha=0.3)    
            
        # Add trend line  
        z = np.polyfit(df['Frequency'], df['Delta'], 1)    
        p = np.poly1d(z)    
        ax4.plot(df['Frequency'], p(df['Frequency']), "r--", alpha=0.8)    
            
        plt.tight_layout()    
        plt.savefig(os.path.join(results_dir, 'diagnostic_framework_analysis.png'),     
                   dpi=300, bbox_inches='tight')    
        plt.close()    
        
    def run_complete_analysis(self):    
        """Execute complete diagnostic analysis"""    
        print("=== DIAGNOSTIC FRAMEWORK ANALYSIS ===")    
            
        # Setup  
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
        results_dir = os.path.join(base_dir, "results")    
        os.makedirs(results_dir, exist_ok=True)    
            
        # Load existing data  
        results_file = os.path.join(results_dir, 'enhanced_detailed_performance_results.json') 
        if not os.path.exists(results_file):    
            print("Run detailed_performance_analysis.py first!")    
            return    
            
        # 1. Create high-impact table  
        df = self.create_high_impact_table(results_file)    
            
        # 2. Analyze inductive bias  
        bias_analysis = self.analyze_inductive_bias(df)    
            
        # 3. Design conditional strategy  
        strategy = self.design_conditional_strategy(bias_analysis)    
            
        # 4. Simulate performance  
        conditional_results = self.simulate_conditional_performance(df, strategy)    
            
        # 5. Create visualizations  
        self.create_visualizations(df, bias_analysis, results_dir)    
            
        # 6. Save results  
        results = {    
            'high_impact_table': df.to_dict('records'),    
            'inductive_bias_analysis': {    
                'delta_statistics': bias_analysis['delta_statistics'],    
                'gat_dominated_count': len(bias_analysis['gat_dominated_notes']),    
                'gin_dominated_count': len(bias_analysis['gin_dominated_notes'])    
            },    
            'conditional_strategy': strategy,    
            'conditional_simulation': conditional_results    
        }    
            
        output_file = os.path.join(results_dir, 'diagnostic_framework_results.json')    
        with open(output_file, 'w') as f:    
            json.dump(results, f, indent=2)    
            
        # Print summary  
        print(f"\n=== DIAGNOSTIC ANALYSIS SUMMARY ===")    
        print(f"Notes analyzed: {len(df)}")    
        print(f"GAT-dominated: {len(bias_analysis['gat_dominated_notes'])}")    
        print(f"GIN-dominated: {len(bias_analysis['gin_dominated_notes'])}")    
        print(f"Expected improvement with conditional strategy: {conditional_results['improvement_over_gin']:.4f}")    
        print(f"\nResults saved in: {output_file}")    
            
        return results    
    
if __name__ == "__main__":    
    framework = DiagnosticFramework()    
    framework.run_complete_analysis()