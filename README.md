# Machine Learning Case Studies


A collection of machine learning case studies focusing on clustering, feature selection, and medical image analysis. This repository contains implementations and analyses of various ML techniques applied to real-world problems.

## Case Studies

### 1. Model Selection for Clustering in Digital Pathology
- Implementation of clustering algorithms for cancer tissue analysis
- Comparison of K-means and Hierarchical clustering
- Evaluation of dimensionality reduction techniques (PCA vs UMAP)
- Performance analysis using multiple metrics

**Key Features:**
- Pre-trained models comparison (ResNet50, InceptionV3, VGG16)
- Dimensionality reduction techniques
- Cluster validation metrics
- Visual result analysis

### 2. CNP Prediction from EEG Data
- Prediction of Central Neuropathic Pain in Spinal Cord Injury patients
- Feature engineering and selection techniques
- Classification using multiple models
- Leave-one-group-out cross-validation

**Key Features:**
- EEG data processing
- Feature selection methods (Wrapper & Embedded)
- Multiple classification models (KNN, SVM)
- Performance evaluation metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/username/ml-case-studies.git
cd ml-case-studies

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
ml-case-studies/
│
├── clustering/
│   ├── models/
│   ├── data_processing/
│   ├── evaluation/
│   └── visualization/
│
├── feature_selection/
│   ├── wrapper_methods/
│   ├── embedded_methods/
│   ├── evaluation/
│   └── visualization/
│
├── data/
│   ├── pathology/
│   └── eeg/
│
├── notebooks/
│   ├── clustering_analysis.ipynb
│   └── feature_selection_analysis.ipynb
│
└── requirements.txt
```

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib
- Seaborn
- UMAP

## Key Results

### Clustering Analysis
- UMAP projection outperformed PCA across all clustering methods
- Best performing model: ResNet50 UMAP with Hierarchical clustering
  - Silhouette score: 0.67
  - V-Measure: 0.56
- K-means with ResNet50 UMAP showed strong performance
  - Silhouette score: 0.61
  - V-Measure: 0.69

### Feature Selection
- Wrapper method (Forward Selection) improved model accuracy up to 94%
- Embedded methods increased accuracy to 80%
- KNN and SVM baseline performance: 75-80%
- L1 regularization showed significant improvement in feature selection

## Usage

### Clustering Analysis
```python
from clustering.models import ClusteringModel
from clustering.evaluation import evaluate_clusters

# Initialize and run clustering
model = ClusteringModel(
    method='hierarchical',
    n_clusters=20,
    projection='umap'
)
clusters = model.fit_predict(data)

# Evaluate results
scores = evaluate_clusters(clusters, ground_truth)
```

### Feature Selection
```python
from feature_selection.wrapper_methods import ForwardSelection
from feature_selection.evaluation import evaluate_model

# Initialize feature selector
selector = ForwardSelection(
    estimator='svm',
    n_features='auto'
)

# Select features and evaluate
selected_features = selector.fit_transform(X, y)
scores = evaluate_model(X_selected, y, cv=logo_cv)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

- Ansh Goyal
- Aditya Tripathi
- Raj Singh
- Rizwana Yasmin Hashim
- Kapil Arora

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Digital Pathology research community
- EEG analysis tools and frameworks
- Scikit-learn community for machine learning tools
