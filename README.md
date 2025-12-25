# GAT vs GIN for Odor Pair Prediction  
  
This project compares Graph Attention Network (GAT) and Graph Isomorphism Network (GIN) architectures for molecular pair prediction in odor classification tasks.  
  
## Attribution and Code Base  
  
This project is based on the original [odor-pair](https://github.com/odor-pair/odor-pair) repository.  
  
### Unmodified Files  
- `data/`, `dataset/`, `graph/`: Copied directly from the original project  
- `crawl.py`: Maintained to document the original dataset generation process (not used in execution)  
  
### Modified Files  
- `pairing/data.py`: Adapted from the original with modifications to support GAT and GIN models  
- `main.py`: Based on the original with adjustments for GIN model implementation (`MixturePredictor` with `GCN`)  
  
### New Code  
- `new1_main.py`: GAT model implementation following the logical structure of `main.py`  
- `analysis/`: Complete analysis framework developed for GAT vs GIN comparison  
  
## Project Structure

```
├── README.md # Project documentation  
├── LICENSE # MIT License
├── analysis/ # Comparative analysis framework
│ ├── standardized_model_evaluation.py # Standardized evaluation
│ ├── detailed_performance_analysis.py # Per-class analysis
│ ├── cross_validation_roc.py # Cross-validation
│ ├── embedding_analysis.py # Embedding analysis
│ ├── molecular_pair_analysis.py # Molecular pair analysis
│ ├── molecular_property_analysis.py # Molecular property analysis
│ ├── calibration_analysis.py # Model calibration analysis
│ ├── uncertainty_analysis.py # Prediction uncertainty analysis
│ ├── model_kl_analysis.py # KL divergence analysis
│ └── statistical_analysis_comparison.py # Statistical meta-analysis
├── pairing/ # Data pipeline (modified)
├── data/, dataset/, graph/ # Original data (unmodified)
├── main.py # GIN model (modified)
├── new1_main.py # GAT model (new)
└── crawl.py # Original web scraping script (documentation)
```

## Implemented Models  
  
### GIN (Graph Isomorphism Network)  
- File: `main.py`  
- Architecture: `MixturePredictor` with `GCN` component  
- Trained model: `runs/1adc5394/model.pt`  
  
### GAT (Graph Attention Network)    
- File: `new1_main.py`  
- Architecture: `MixturePredictor` with `GAT` component  
- Trained model: `runs/702a60ea/model.pt`  
  
## Analysis Framework  
  
The project includes a comprehensive analysis framework with multiple dimensions:  
  
### Performance Analysis  
- AUROC micro and macro-average  
- Per-class olfactory note analysis  
- Cross-validation with confidence intervals  
  
### Molecular Analysis  
- Tanimoto similarity between molecular pairs  
- Molecular properties (molecular weight, LogP, TPSA)  
- Embedding analysis with t-SNE and clustering  

### Model Behavior Analysis  
- Prediction calibration and reliability  
- Uncertainty quantification    
- KL divergence between model distributions

### Statistical Analysis  
- Paired t-tests for significance  
- Meta-analysis aggregating multiple results  
- Comparative visualizations  
  
## Usage  
  
### Model Training  
```bash  
# GIN model
python main.py  
  
# GAT model
python new1_main.py  
```

### Comparative Analysis  
  
#### Performance Analysis  
```bash  
# Standardized evaluation
python analysis/standardized_model_evaluation.py  
  
# Detailed per-class analysis
python analysis/detailed_performance_analysis.py  
  
# Cross-validation with ROC curves
python analysis/cross_validation_roc.py
```

#### Molecular Analysis
```bash  
# Embedding analysis with t-SNE
python analysis/embedding_analysis.py  
  
# Molecular pair similarity analysis
python analysis/molecular_pair_analysis.py  
  
# Molecular property correlation analysis
python analysis/molecular_property_analysis.py
```

#### Model Behavior Analysis
```bash  
# Model calibration analysis
python analysis/calibration_analysis.py  
  
# Prediction uncertainty analysis
python analysis/uncertainty_analysis.py  
  
# KL divergence analysis
python analysis/model_kl_analysis.py
```

#### Statistical Analysis
```bash  
# Comprehensive statistical comparison
python analysis/statistical_analysis_comparison.py
```

## Results  
  
Analysis results are saved in `results/` with structured JSON files and PNG visualizations for each analysis type.  
  
## Dependencies  
  
- PyTorch  
- PyTorch Geometric  
- RDKit  
- scikit-learn  
- matplotlib  
- seaborn  
- pandas  
- numpy  
  
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.



