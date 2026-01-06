# GAT vs GIN for Odor Pair Prediction

This project compares Graph Attention Network (GAT) and Graph Isomorphism Network (GIN) architectures for molecular pair prediction in odor classification tasks with comprehensive statistical analysis.

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
- `analysis/`: Enhanced analysis framework with advanced statistical methods for GAT vs GIN comparison

## Project Structure

```
├── README.md # Project documentation
├── LICENSE # MIT License
├── analysis/ # Enhanced comparative analysis framework
│ ├── standardized_model_evaluation.py # Standardized evaluation with advanced statistics
│ ├── detailed_performance_analysis.py # Per-class analysis with statistical tests
│ ├── cross_validation_roc.py # Cross-validation with effect sizes and bootstrap
│ ├── embedding_analysis.py # Embedding analysis
│ ├── molecular_pair_analysis.py # Molecular pair analysis
│ ├── molecular_property_analysis.py # Molecular property analysis
│ ├── calibration_analysis.py # Model calibration analysis
│ ├── uncertainty_analysis.py # Prediction uncertainty analysis
│ ├── model_kl_analysis.py # KL divergence analysis
│ ├── statistical_analysis_comparison.py # Statistical meta-analysis
│ └── diagnostic_framework.py # Diagnostic analysis framework
├── pairing/ # Data pipeline (modified)
├── data/, dataset/, graph/ # Original data (unmodified)
├── main.py # GIN model (modified)
├── new1_main.py # GAT model (new)
└── crawl.py # Original web scraping script
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

## Enhanced Analysis Framework

The project includes a comprehensive analysis framework with advanced statistical methods across multiple dimensions:

### Performance Analysis
- **Enhanced AUROC Analysis**: Micro and macro-average with confidence intervals
- **Per-class Analysis**: Detailed olfactory note performance with statistical significance
- **Cross-validation**: Bootstrap confidence intervals and effect sizes
- **Statistical Tests**: Cohen's d, Hedges' g, Glass's delta, Wilcoxon, Mann-Whitney, permutation tests

### Molecular Analysis
- Tanimoto similarity between molecular pairs
- Molecular properties (molecular weight, LogP, TPSA)
- Embedding analysis with t-SNE and clustering

### Model Behavior Analysis
- Prediction calibration and reliability
- Uncertainty quantification
- KL divergence between model distributions

### Advanced Statistical Analysis
- **Effect Size Metrics**: Cohen's d, Glass's delta, Hedges' g
- **Bootstrap Confidence Intervals**: Non-parametric confidence estimation
- **Power Analysis**: Statistical power and sample size requirements
- **Non-parametric Tests**: Wilcoxon signed-rank, Mann-Whitney U, Kolmogorov-Smirnov
- **Permutation Testing**: Robust significance testing
- **Meta-analysis**: Aggregating results across multiple evaluation types

### Diagnostic Framework
- Rare vs frequent class performance analysis
- Inductive bias detection
- Conditional performance strategies

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
# Standardized evaluation with advanced statistics
python analysis/standardized_model_evaluation.py

# Detailed per-class analysis with statistical tests
python analysis/detailed_performance_analysis.py

# Cross-validation with ROC curves and effect sizes
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

#### Statistical and Diagnostic Analysis
```bash
# Comprehensive statistical comparison with meta-analysis
python analysis/statistical_analysis_comparison.py

# Diagnostic framework analysis
python analysis/diagnostic_framework.py
```

## Results

Enhanced analysis results are saved in `results/` with structured JSON files and PNG visualizations:

### Performance Results
- `enhanced_standardized_evaluation_results.json` - Standardized evaluation with statistical metrics
- `cross_validation_results.json` - Cross-validation results with confidence intervals
- `enhanced_detailed_performance_results.json` - Per-class analysis with statistical tests

### Statistical Results
- `enhanced_statistical_analysis_summary.json` - Meta-analysis results
- `enhanced_statistical_analysis.png` - Comprehensive statistical visualization

### Visualizations
- `enhanced_standardized_model_comparison.png` - Multi-panel performance comparison
- `cross_validation_roc_comparison.png` - ROC curves with confidence intervals
- `rare_vs_frequent_performance.png` - Rare vs frequent class analysis
- `rare_vs_frequent_effect_sizes.png` - Effect size visualization

## Dependencies

- PyTorch
- PyTorch Geometric
- RDKit
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- scipy
- statsmodels

## Statistical Analysis

The framework includes comprehensive statistical analysis:

- **Effect Size Calculation**: Cohen's d, Glass's delta, Hedges' g for magnitude assessment
- **Bootstrap Methods**: Non-parametric confidence intervals (1000 bootstrap samples)
- **Power Analysis**: Statistical power calculation and required sample size estimation
- **Multiple Testing Correction**: Bonferroni and FDR corrections for multiple comparisons
- **Non-parametric Validation**: Wilcoxon, Mann-Whitney, and permutation tests
- **Robust Statistics**: Median absolute deviation and trimmed means

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
```
