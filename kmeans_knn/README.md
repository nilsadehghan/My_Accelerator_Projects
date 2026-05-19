


# Letter Image Recognition - KNN & K-Means from Scratch

## 📌 Project Overview

This project implements two fundamental machine learning algorithms from scratch - **K-Nearest Neighbors (KNN)** for supervised classification and **K-Means** for unsupervised clustering - on the **Letter Image Recognition** dataset from UCI Machine Learning Repository.

The goal is to deeply understand how these algorithms work, the impact of data preprocessing, and parameter selection, without using any ready-made machine learning libraries like scikit-learn.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository
- **Samples**: 20,000 samples
- **Features**: 16 numerical features (x-box, y-box, width, high, onpix, x-bar, y-bar, etc.)
- **Classes**: 26 uppercase English letters (A-Z)
- **File**: `letter-recognition.data`

> Note: The dataset is designed for supervised classification (handwriting recognition).

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.10 | Core programming language |
| NumPy | Numerical computations |
| Pandas | Data loading & CSV output |
| Matplotlib | Plotting |
| Seaborn | Statistical visualizations |
| Time | Performance measurement |

> ⚠️ **No machine learning libraries (scikit-learn, etc.) were used** - all algorithms implemented manually.

---

## 📁 Repository Structure


HW1_StudentNumber_FullName.zip/
│
├── HW1_part1/                          # KNN Implementation
│   ├── knn.py                          # Main KNN code
│   ├── letter-recognition.data         # Dataset
│   ├── output_Euclidean_K1.csv         # Predictions (Euclidean, K=1)
│   ├── output_Euclidean_K3.csv         # Predictions (Euclidean, K=3)
│   ├── output_Euclidean_K5.csv         # Predictions (Euclidean, K=5)
│   ├── output_Euclidean_K9.csv         # Predictions (Euclidean, K=9)
│   ├── output_Manhattan_K1.csv         # Predictions (Manhattan, K=1)
│   ├── output_Manhattan_K3.csv         # Predictions (Manhattan, K=3)
│   ├── output_Manhattan_K5.csv         # Predictions (Manhattan, K=5)
│   ├── output_Manhattan_K9.csv         # Predictions (Manhattan, K=9)
│   └── result/                         # Saved plots
│
└── HW1_part2/                          # K-Means Implementation
    ├── kmeans.py                       # Main K-Means code
    ├── letter-recognition.data         # Dataset
    ├── kmeans_labels_iter*.csv         # Cluster labels for each max_iter
    ├── kmeans_labels_history_iter*.csv # Label history per iteration
    ├── kmeans_iteration_analysis.csv   # Iteration performance summary
    ├── kmeans_cluster_stats_K2.csv     # Mean & std per feature per cluster
    ├── kmeans_clusters_K2.csv          # Final cluster labels
    ├── kmeans_confusion_matrix_K2.csv  # Confusion matrix
    ├── kmeans_evaluation_metrics_K2.csv # Accuracy, Precision, Recall, F1
    └── result/                         # Saved plots




## 🚀 Part 1: K-Nearest Neighbors (Supervised Learning)

### Algorithm Steps

1. Calculate distance between test point and all training points
2. Select `k` closest neighbors
3. Predict the majority label among those neighbors

### Implementation Details

| Component | Implementation |
|-----------|----------------|
| Distance Metrics | Euclidean, Manhattan (both implemented manually) |
| K Values Tested | 1, 3, 5, 9 |
| Train-Test Split | 80% training, 20% testing (randomized) |
| Normalization | Min-Max scaling (all features → [0,1]) |
| Voting | `Counter` from collections module |

### Key Functions

python
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def knn_predict(X_train, y_train, x_test, k, distance_func):
    # Calculate distances, find k nearest, vote


### Results

| Distance | K=1 | K=3 | K=5 | K=9 |
|----------|-----|-----|-----|-----|
| **Euclidean** | 95.65% | **95.93%** | 95.90% | 95.53% |
| **Manhattan** | 95.03% | 95.70% | 95.70% | 95.63% |

**Conclusion:** Best result achieved with **Euclidean distance + K=3** at **95.93% accuracy**.

---

## 🔬 Part 2: K-Means Clustering (Unsupervised Learning)

### Algorithm Steps

1. Randomly initialize `k` centroids from data points
2. Assign each point to the nearest centroid
3. Update centroids as mean of points in each cluster
4. Repeat steps 2-3 until convergence

### Optimization Techniques Used

| Technique | Description | Benefit |
|-----------|-------------|---------|
| **Two-Stage Search** | Quick evaluation (5 iterations) → Fine search (15 iterations) | 35% faster |
| **Vectorized Silhouette** | NumPy operations instead of Python loops | 3-5x faster |
| **Fast K-Means** | No history storage (for search phase) | Memory & time efficient |

### Part A: Finding Optimal K

Tested K = 1 to 26 using Silhouette Score:

| K | Silhouette Score |
|---|------------------|
| 1 | 0.0000 |
| **2** | **0.1752** (Best) |
| 3 | 0.1586 |
| 4 | 0.1525 |
| ... | ... |

**Finding:** Despite knowing there are 26 letters, Silhouette Score suggests K=2. This demonstrates that **Silhouette alone cannot determine the true number of clusters** for this dataset.

### Part B: Minimum Good Iterations

| max_iter | actual_iterations | Time (s) | Silhouette |
|----------|-------------------|----------|------------|
| 1 | 1 | 0.19 | 0.0877 |
| 2 | 2 | 0.39 | 0.1500 |
| 3 | 3 | 0.58 | 0.1650 |
| 5 | 5 | 1.07 | 0.1704 |
| 7 | 7 | 1.70 | 0.1754 |
| 10 | 10 | 1.92 | 0.1746 |

**Conclusion:** Minimum good iterations = **5** (reaches ~97% of best score)

### Part C: Cluster Statistics (K=2)

**Mean values per cluster:**

| Feature | Cluster 0 | Cluster 1 | Difference |
|---------|-----------|-----------|------------|
| F1 | 0.3478 | 0.1712 | 0.1766 |
| F2 | 0.6177 | 0.2879 | **0.3298** |
| F3 | 0.4241 | 0.2407 | 0.1834 |
| F4 | 0.4566 | 0.2382 | 0.2184 |
| F5 | 0.3224 | 0.1256 | 0.1968 |

> Feature F2 (y-bar: mean y-coordinate of bright pixels) shows the largest difference between clusters.

### Final Evaluation (with mapping to letters)

| Metric | Value |
|--------|-------|
| Accuracy | 5.24% |
| Mean Precision | 0.41% |
| Mean Recall | 5.21% |
| Macro F1-Score | 0.76% |

**Interpretation:** Unsupervised clustering performs poorly on this dataset (designed for supervised learning).

---

## 📊 Visualizations (All 16 Features)

### Implemented Plots

| Plot | Description | Centroids Marked |
|------|-------------|------------------|
| **Scatter Plot Matrix** | 16×16 grid showing relationships between all feature pairs | ✅ Black 'X' |
| **Pair Plot** | Automated version of scatter plot matrix with histograms on diagonal | ✅ Black 'X' |
| **Strip Plot** | 4×4 grid showing distribution of each feature across clusters | ✅ Red dashed line |
| **Swarm Plot** | Same as strip plot but points arranged to avoid overlap | ✅ Red dashed line |
| **Centroids Comparison** | Line plot comparing cluster centers across all 16 features | ✅ Line + circles |

### Balanced Sampling

- For visualizations, **50 samples per class** (KNN) and **50 samples per cluster** (K-Means) were randomly selected
- For Pair Plot, reduced to **15 samples per cluster** for better centroid visibility
- **Important:** Sampling was only for visualization, all computations used the full 20,000 samples

---

## 📈 Evaluation Metrics (Manual Implementation)

| Metric | How It's Computed |
|--------|-------------------|
| **Accuracy** | `(TP + TN) / Total` → `np.mean(y_true == y_pred)` |
| **Precision** | `TP / (TP + FP)` (averaged across all classes) |
| **Recall** | `TP / (TP + FN)` (averaged across all classes) |
| **Confusion Matrix** | 26×26 matrix built manually with nested loops |
| **Silhouette Score** | `(b - a) / max(a, b)` (implemented with vectorized NumPy) |

> All metrics were implemented manually without using `sklearn.metrics`.

---

## ⚡ Performance Optimizations

| Challenge | Solution |
|-----------|----------|
| K-Means took >1 hour to find optimal K | Two-stage search (quick + fine) reduced time to ~15 minutes |
| Silhouette calculation was too slow | Vectorized NumPy operations instead of Python loops (3-5x faster) |
| Swarm plot warnings (dense points) | try-except fallback to strip plot |
| Plots too crowded to see centroids | Balanced sampling (50 samples per cluster, 15 for pair plot) |

---

## 📝 Key Findings

1. **KNN (Supervised)** achieves **~96% accuracy** on this dataset
2. **K-Means (Unsupervised)** achieves only **~5% accuracy** - the dataset is not suitable for clustering
3. **Silhouette Score suggests K=2**, while we know there are 26 letters - demonstrating its limitations
4. **Euclidean distance** outperforms Manhattan for this dataset (95.93% vs 95.70% at K=3)
5. **Minimum good iterations** for K-Means on this dataset is **5**
6. **Feature F2 (y-bar)** shows the largest difference between clusters (0.3298)

---

## 🎯 Conclusion

This project successfully demonstrates:

- **Full manual implementation** of KNN and K-Means without ML libraries
- **Importance of normalization** when features have different scales
- **Difference between supervised and unsupervised learning**
- **Limitations of Silhouette Score** for determining optimal K
- **Not all datasets are suitable for clustering** - this one was designed for classification

---

## 🧪 How to Run

### Prerequisites


pip install numpy pandas matplotlib seaborn


### Run Part 1 (KNN)


cd HW1_part1
python knn.py


### Run Part 2 (K-Means)


cd HW1_part2
python kmeans.py


---

## 📁 Output Files

### KNN Outputs
- output_{Euclidean/Manhattan}_K{1,3,5,9}.csv - True labels, predicted labels, and all 16 features

### K-Means Outputs
- kmeans_labels_iter{1,2,3,5,7,10}.csv - Final cluster labels for each max_iter
- kmeans_labels_history_iter{1,2,3,5,7,10}.csv - Label history per iteration
- kmeans_iteration_analysis.csv - Iteration performance summary
- kmeans_cluster_stats_K2.csv - Mean and standard deviation per feature per cluster
- kmeans_clusters_K2.csv - Final cluster labels
- kmeans_confusion_matrix_K2.csv - Confusion matrix (after mapping)
- kmeans_evaluation_metrics_K2.csv - Accuracy, Precision, Recall, F1

---

## 👨‍💻 Author

Student project for **Hardware Accelerators** course   
Spring 2025

---

## 📜 License

This project is for educational purposes only.
```

