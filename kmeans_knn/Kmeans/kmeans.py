# =========================
# Part 2: K-Means Clustering
# =========================
# Unsupervised learning - no labels used
# Distance metric: Euclidean only

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# =========================
# Load dataset
# =========================
data = pd.read_csv("letter-recognition.data", header=None)

X = data.iloc[:, 1:].values  # features only (NO labels)


# =========================
# Normalize features (Min-Max)
# =========================
def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


X = normalize(X)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print("Distance metric: Euclidean\n")


# =========================
# Euclidean distance
# =========================
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# =========================
# Initialize centroids randomly
# =========================
def initialize_centroids(X, k):
    n_samples = len(X)
    indices = np.random.choice(n_samples, k, replace=False)
    return X[indices].copy()


# =========================
# Assign each point to nearest centroid
# =========================
def assign_clusters(X, centroids):
    n_samples = len(X)
    k = len(centroids)
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        min_dist = float('inf')
        min_idx = 0
        for j in range(k):
            dist = euclidean(X[i], centroids[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        labels[i] = min_idx
    return labels


# =========================
# Update centroids (mean of points in each cluster)
# =========================
def update_centroids(X, labels, k):
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features))

    for j in range(k):
        points = X[labels == j]
        if len(points) > 0:
            new_centroids[j] = np.mean(points, axis=0)
        else:
            new_centroids[j] = X[np.random.choice(len(X))]
    return new_centroids


# =========================
# Check convergence
# =========================
def has_converged(old, new, tolerance=1e-4):
    for j in range(len(old)):
        if np.linalg.norm(old[j] - new[j]) > tolerance:
            return False
    return True


# =========================
# K-Means Fast
# =========================
def kmeans_fast(X, k, max_iterations=10):

    centroids = initialize_centroids(X, k)

    for iteration in range(max_iterations):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        if has_converged(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels


def kmeans(X, k, max_iterations=10, tolerance=1e-4):
    centroids = initialize_centroids(X, k)
    iteration_times = []
    iteration_labels = []

    for iteration in range(max_iterations):
        start = time.time()

        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)

        iteration_times.append(time.time() - start)
        iteration_labels.append(labels.copy())

        if has_converged(centroids, new_centroids, tolerance):
            break

        centroids = new_centroids

    return labels, centroids, iteration_times, iteration_labels







# =========================
# Silhouette Score Optimized
# =========================
def silhouette_score_optimized(X, labels):
    n_samples = len(X)
    n_clusters = len(np.unique(labels))

    if n_clusters == 1:
        return 0

    sil_values = np.zeros(n_samples)

    for i in range(n_samples):
        current_cluster = labels[i]

        # a(i)
        same_mask = labels == current_cluster
        same_points = X[same_mask]
        n_same = len(same_points)

        if n_same > 1:
            dists = np.sqrt(np.sum((same_points - X[i]) ** 2, axis=1))
            a_i = np.sum(dists[dists > 0]) / (n_same - 1)
        else:
            a_i = 0

        # b(i)
        b_i = float('inf')
        for c in range(n_clusters):
            if c != current_cluster:
                other_points = X[labels == c]
                if len(other_points) > 0:
                    mean_dist = np.mean(np.sqrt(np.sum((other_points - X[i]) ** 2, axis=1)))
                    if mean_dist < b_i:
                        b_i = mean_dist

        if max(a_i, b_i) > 0:
            sil_values[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(sil_values)


# ============================================================================

# ============================================================================
# ============================================================================
# PART A: Finding Optimal K (Complete but Efficient)
# ============================================================================
print("=" * 70)
print("PART A: Finding Optimal K (Complete Search - K=1 to 26)")
print("=" * 70)


print("\n--- Step 1: Quick evaluation for all K (1 to 26) ---")
k_range_all = range(1, 27)
quick_scores = []
quick_times = []

for k in k_range_all:
    print(f"Testing K = {k}...", end=" ")
    start = time.time()
    labels = kmeans_fast(X, k, max_iterations=5)
    score = silhouette_score_optimized(X, labels)
    elapsed = time.time() - start
    quick_scores.append(score)
    quick_times.append(elapsed)
    print(f"Score: {score:.4f} (time: {elapsed:.2f}s)")


top_k_indices = np.argsort(quick_scores)[-3:]
candidate_ks = [k_range_all[i] for i in sorted(top_k_indices)]
print(f"\nCandidate Ks from quick search: {candidate_ks}")


print("\n--- Step 2: Fine search on candidates ---")
fine_scores = []
for k in candidate_ks:
    print(f"Testing K = {k} (detailed)...")
    start = time.time()
    labels = kmeans_fast(X, k, max_iterations=15)
    score = silhouette_score_optimized(X, labels)
    elapsed = time.time() - start
    fine_scores.append((k, score))
    print(f"  Score: {score:.4f} (time: {elapsed:.2f}s)")


best_k, best_score = max(fine_scores, key=lambda x: x[1])
print("\n" + "=" * 70)
print(f"✅ Best K: {best_k} (Silhouette Score: {best_score:.4f})")
print("=" * 70)


# ============================================================================
# PART B: Finding Minimum Good Iterations
# ============================================================================
print("\n" + "=" * 70)
print("PART B: K-Means with Different Max Iterations")
print("=" * 70)

print(f"\nUsing K = {best_k} (best from Silhouette Score)")

max_iter_values = [1, 2, 3, 5, 7, 10]
iteration_results = []

for max_iter in max_iter_values:
    print(f"\n{'=' * 50}")
    print(f"Testing max_iterations = {max_iter}")
    print(f"{'=' * 50}")

    start_total = time.time()


    labels, _, iter_times, labels_history = kmeans(X, best_k, max_iterations=max_iter)

    total_time = time.time() - start_total

    sil_score = silhouette_score_optimized(X, labels)

    converged = len(iter_times) < max_iter


    labels_output = pd.DataFrame({
        "Sample_Index": range(len(labels)),
        "Cluster_Label": labels
    })
    labels_output.to_csv(f"kmeans_labels_iter{max_iter}.csv", index=False)
    print(f"  ✅ Saved labels to: kmeans_labels_iter{max_iter}.csv")


    history_df = pd.DataFrame()
    for iter_idx, hist_labels in enumerate(labels_history):
        history_df[f'Iteration_{iter_idx+1}'] = hist_labels
    history_df.to_csv(f"kmeans_labels_history_iter{max_iter}.csv", index=False)
    print(f"  ✅ Saved labels history to: kmeans_labels_history_iter{max_iter}.csv")


    iteration_results.append({
        'max_iterations': max_iter,
        'actual_iterations': len(iter_times),
        'converged': 'Yes' if converged else 'No',
        'total_time_sec': total_time,
        'avg_time_per_iter_ms': np.mean(iter_times) * 1000,
        'silhouette_score': sil_score,
    })

    print(f"  Best K used: {best_k}")
    print(f"  Actual iterations performed: {len(iter_times)}")
    print(f"  Converged? {'Yes ✅' if converged else 'No ❌'}")
    print(f"  Total time: {total_time:.4f} seconds")
    print(f"  Avg time per iteration: {np.mean(iter_times) * 1000:.3f} ms")
    print(f"  Final Silhouette Score: {sil_score:.4f}")


results_df = pd.DataFrame(iteration_results)
print("\n" + "=" * 70)
print("📊 Iteration Performance Summary:")
print("=" * 70)
print(results_df[['max_iterations', 'actual_iterations', 'converged',
                  'total_time_sec', 'silhouette_score']].to_string(index=False))

results_df.to_csv("kmeans_iteration_analysis.csv", index=False)
print(f"\n✅ Saved iteration analysis to: kmeans_iteration_analysis.csv")

# ============================================================================
# PART C: Mean and Std Deviation per feature per cluster
# ============================================================================
print("\n" + "=" * 70)
print("PART C: Cluster Statistics (Mean & Std per Feature)")
print("=" * 70)

print(f"\nRunning final K-Means with K = {best_k}, max_iterations=10...")
final_labels, final_centroids, final_times, _ = kmeans(X, best_k, max_iterations=10)


feature_names = [f"F{i + 1}" for i in range(X.shape[1])]


cluster_stats = []

for cluster_id in range(best_k):

    points = X[final_labels == cluster_id]

    if len(points) > 0:

        means = np.mean(points, axis=0)

        stds = np.std(points, axis=0)


        row = {'Cluster': cluster_id, 'Size': len(points)}
        for i, name in enumerate(feature_names):
            row[f'{name}_mean'] = means[i]
            row[f'{name}_std'] = stds[i]
        cluster_stats.append(row)


stats_df = pd.DataFrame(cluster_stats)


print("\n" + "=" * 70)
print("📊 TABLE 1: MEAN values per feature per cluster")
print("=" * 70)
mean_cols = ['Cluster', 'Size'] + [f'{f}_mean' for f in feature_names]
print(stats_df[mean_cols].to_string(index=False))


print("\n" + "=" * 70)
print("📊 TABLE 2: STD values per feature per cluster")
print("=" * 70)
std_cols = ['Cluster'] + [f'{f}_std' for f in feature_names]
print(stats_df[std_cols].to_string(index=False))


stats_df.to_csv(f"kmeans_cluster_stats_K{best_k}.csv", index=False)
print(f"\n✅ Saved cluster statistics to: kmeans_cluster_stats_K{best_k}.csv")

# ============================================================================

# ============================================================================
print("\n" + "=" * 70)
print("📊 FINAL SUMMARY - K-MEANS CLUSTERING")
print("=" * 70)
print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"Distance metric: Euclidean")
print(f"Optimal K (Silhouette Score): {best_k} (Score: {best_score:.4f})")
print(f"Total time for final run: {sum(final_times):.4f} seconds")

print("\n📁 Output files generated:")
print(f"  - kmeans_labels_iter*.csv (cluster labels for each max_iter)")
print(f"  - kmeans_labels_history_iter*.csv (labels history for each max_iter)")
print(f"  - kmeans_iteration_analysis.csv (iteration performance summary)")
print(f"  - kmeans_cluster_stats_K{best_k}.csv (mean & std per feature per cluster)")

print("\n📊 Cluster sizes:")
sizes = [np.sum(final_labels == i) for i in range(best_k)]
for i, size in enumerate(sizes):
    print(f"  Cluster {i}: {size} samples ({size / len(X) * 100:.1f}%)")

# ============================================================================
# PART D: Visualizations and Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("PART D: Visualizations and Evaluation (All 16 Features)")
print("=" * 70)

# ============================================================================
# First, run final K-Means with best_k to get labels
# ============================================================================
print("\nRunning final K-Means for visualization...")
final_labels, final_centroids, final_times, _ = kmeans(X, best_k, max_iterations=10)

# Save cluster labels to CSV
cluster_output = pd.DataFrame({"Cluster_Label": final_labels})
cluster_output.to_csv(f"kmeans_clusters_K{best_k}.csv", index=False)
print(f"✅ Saved cluster labels to: kmeans_clusters_K{best_k}.csv")

# ============================================================================
# Balanced Sampling (like KNN - 50 samples per cluster)
# ============================================================================
print("\n" + "=" * 70)
print("Balanced Sampling (50 samples per cluster)")
print("=" * 70)

unique_clusters = np.unique(final_labels)
samples_per_cluster = 50

sample_idx = []
for c in unique_clusters:
    cluster_idx = np.where(final_labels == c)[0]
    chosen = np.random.choice(cluster_idx, min(samples_per_cluster, len(cluster_idx)), replace=False)
    sample_idx.extend(chosen)

sample_idx = np.array(sample_idx)
X_sample = X[sample_idx]
y_sample = final_labels[sample_idx]  # cluster labels for coloring

print(f"Total samples after balanced sampling: {len(sample_idx)}")
print(f"Unique clusters in sample: {np.unique(y_sample)}")

# ============================================================================
# 1) Scatter Plot Matrix (16x16, skip diagonal) WITH CENTROIDS
# ============================================================================
print("\n1) Scatter Plot Matrix - 16x16 features WITH CENTROIDS")
n_features = X_sample.shape[1]  # 16
fig, axes = plt.subplots(n_features, n_features, figsize=(25, 25))

for i in range(n_features):
    for j in range(n_features):
        ax = axes[i, j]

        if i == j:
            ax.axis('off')  # Skip diagonal as requested
        else:

            ax.scatter(X_sample[:, j], X_sample[:, i],
                       c=y_sample, cmap='tab20',
                       alpha=0.5, s=5)


            for cluster_id in range(best_k):
                center_x = final_centroids[cluster_id, j]
                center_y = final_centroids[cluster_id, i]
                ax.scatter(center_x, center_y,
                           marker='X', s=200, c='black',
                           edgecolors='white', linewidth=2)
            # ===================================================

        if i == n_features - 1:
            ax.set_xlabel(f'F{j + 1}')
        if j == 0:
            ax.set_ylabel(f'F{i + 1}')

        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle(f"1) Scatter Plot Matrix with Centroids (K={best_k})", fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# 2) Pair Plot - All 16 features WITH CENTROIDS (Low Density)
# ============================================================================
print("\n2) Pair Plot - All 16 features WITH CENTROIDS (Low Density)")

samples_per_cluster_for_pairplot = 15

sampled_indices_pair = []
for c in unique_clusters:
    cluster_idx = np.where(final_labels == c)[0]
    n_samples = min(samples_per_cluster_for_pairplot, len(cluster_idx))
    chosen = np.random.choice(cluster_idx, n_samples, replace=False)
    sampled_indices_pair.extend(chosen)

sampled_indices_pair = np.array(sampled_indices_pair)
X_sample_pair = X[sampled_indices_pair]
y_sample_pair = final_labels[sampled_indices_pair]

print(f"Samples for Pair Plot: {len(sampled_indices_pair)} (from {len(unique_clusters)} clusters)")
# ================================================

sample_df = pd.DataFrame(X_sample_pair)
sample_df.columns = [f"F{i + 1}" for i in range(16)]
sample_df["Cluster"] = y_sample_pair
sample_df["Type"] = "Data Point"

centroids_df = pd.DataFrame(final_centroids)
centroids_df.columns = [f"F{i + 1}" for i in range(16)]
centroids_df["Cluster"] = range(best_k)
centroids_df["Type"] = "Centroid"


combined_df = pd.concat([sample_df, centroids_df], ignore_index=True)


pair_plot = sns.pairplot(combined_df,
                         hue="Cluster",
                         palette='tab20',
                         diag_kind='hist',
                         plot_kws={'alpha': 0.7, 's': 8})


n_features = 16
for i in range(n_features):
    for j in range(n_features):
        if i != j:
            ax = pair_plot.axes[i, j]
            for cluster_id in range(best_k):
                ax.scatter(final_centroids[cluster_id, j],
                          final_centroids[cluster_id, i],
                          marker='X', s=300, c='black',
                          edgecolors='white', linewidth=2.5,
                          zorder=10)  # zorder=10 یعنی بالاترین لایه

plt.suptitle(f"2) Pair Plot with Centroids (K={best_k}, {samples_per_cluster_for_pairplot} samples/cluster)", y=1.02)
plt.show()





# ============================================================================
# 3) Strip Plot - All 16 features (4x4 grid)
# ============================================================================
print("\n3) Strip Plot - All 16 features by Cluster")
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
axes = axes.flatten()

for i in range(16):
    ax = axes[i]
    df_plot = pd.DataFrame({"Feature": X_sample[:, i], "Cluster": y_sample})
    sns.stripplot(x="Cluster", y="Feature", data=df_plot, ax=ax, alpha=0.5, size=2)


    for cluster_id in range(best_k):
        center_value = final_centroids[cluster_id, i]

        ax.axhline(y=center_value, xmin=cluster_id / best_k, xmax=(cluster_id + 1) / best_k,
                   color='red', linestyle='--', linewidth=2, alpha=0.7)

    # ====================================================================

    ax.set_title(f"Feature {i + 1} (Centroids: red dashed line)")
    ax.set_xlabel("Cluster ID")
    ax.tick_params(axis='x', rotation=0)
    if i % 4 != 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Value")

plt.suptitle(f"3) Strip Plot with Centroids (K={best_k})", fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# 4) Swarm Plot - All 16 features (4x4 grid with fallback) WITH CENTROIDS
# ============================================================================
print("\n4) Swarm Plot - All 16 features by Cluster WITH CENTROIDS")
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
axes = axes.flatten()

for i in range(16):
    ax = axes[i]
    df_plot = pd.DataFrame({"Feature": X_sample[:, i], "Cluster": y_sample})

    try:
        sns.swarmplot(x="Cluster", y="Feature", data=df_plot, ax=ax, alpha=0.5, size=2)
    except:
        sns.stripplot(x="Cluster", y="Feature", data=df_plot, ax=ax, alpha=0.5, size=2)
        ax.set_title(f"Feature {i + 1} (Strip fallback)")
    else:
        ax.set_title(f"Feature {i + 1}")


    for cluster_id in range(best_k):
        center_value = final_centroids[cluster_id, i]

        ax.axhline(y=center_value, xmin=cluster_id / best_k, xmax=(cluster_id + 1) / best_k,
                   color='red', linestyle='--', linewidth=2, alpha=0.7)
    # ====================================================================

    ax.set_xlabel("Cluster ID")
    if i % 4 != 0:
        ax.set_ylabel("")

plt.suptitle(f"4) Swarm Plot with Centroids (K={best_k})", fontsize=16)
plt.tight_layout()
plt.show()

# ============================================================================
# 5) Additional: Centroids Comparison (explicit centroids plot)
# ============================================================================
print("\n5) Centroids Comparison - All 16 features (Explicit)")
plt.figure(figsize=(14, 8))

feature_indices = np.arange(16)
for cluster_id in range(best_k):
    plt.plot(feature_indices, final_centroids[cluster_id],
             marker='o', linewidth=2, markersize=8,
             label=f'Cluster {cluster_id} (Center)')

plt.xlabel('Feature Number')
plt.ylabel('Normalized Feature Value')
plt.title(f'K-Means Centroids (Cluster Centers) - All 16 Features (K={best_k})')
plt.xticks(feature_indices, [f'F{i + 1}' for i in range(16)], rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# PART E: Confusion Matrix and Evaluation Metrics (Manual Implementation)
# ============================================================================
print("\n" + "=" * 70)
print("PART E: Confusion Matrix and Evaluation Metrics (No sklearn)")
print("=" * 70)

y_actual = data.iloc[:, 0].values
print(f"\nActual labels shape: {y_actual.shape}")
print(f"Unique actual letters: {np.unique(y_actual)}")

# ============================================================================
# Step 1: Map each cluster to the most common actual letter
# ============================================================================
print("\n" + "-" * 50)
print("Step 1: Mapping clusters to actual letters")
print("-" * 50)

from collections import Counter

cluster_to_letter = {}
mapping_accuracy = 0

for cluster_id in range(best_k):

    cluster_indices = np.where(final_labels == cluster_id)[0]

    cluster_letters = y_actual[cluster_indices]

    most_common = Counter(cluster_letters).most_common(1)[0]
    cluster_to_letter[cluster_id] = most_common[0]
    print(f"  Cluster {cluster_id} -> Letter '{most_common[0]}' ({most_common[1]} samples)")


y_pred_mapped = np.array([cluster_to_letter[label] for label in final_labels])

# ============================================================================
# Step 2: Manual Confusion Matrix
# ============================================================================
print("\n" + "-" * 50)
print("Step 2: Building Confusion Matrix (Manual)")
print("-" * 50)

all_letters = np.unique(y_actual)
n_letters = len(all_letters)


letter_to_idx = {letter: idx for idx, letter in enumerate(all_letters)}

confusion_matrix = np.zeros((n_letters, n_letters), dtype=int)

for true_label, pred_label in zip(y_actual, y_pred_mapped):
    i = letter_to_idx[true_label]
    j = letter_to_idx[pred_label]
    confusion_matrix[i, j] += 1

print(f"✅ Confusion matrix built: {n_letters}x{n_letters}")

# ============================================================================
# Step 3: Calculate Accuracy, Precision, Recall (Manual)
# ============================================================================
print("\n" + "-" * 50)
print("Step 3: Calculating Metrics")
print("-" * 50)

# Accuracy
accuracy = np.sum(y_actual == y_pred_mapped) / len(y_actual)
print(f"✅ Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# Precision and Recall per class
precisions = []
recalls = []

for letter in all_letters:
    idx = letter_to_idx[letter]

    # True Positive: predicted = letter AND actual = letter
    tp = confusion_matrix[idx, idx]

    # False Positive: predicted = letter BUT actual != letter
    fp = np.sum(confusion_matrix[:, idx]) - tp

    # False Negative: actual = letter BUT predicted != letter
    fn = np.sum(confusion_matrix[idx, :]) - tp

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)

# Mean metrics
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (
                                                                                              mean_precision + mean_recall) > 0 else 0

print(f"✅ Mean Precision: {mean_precision:.4f}")
print(f"✅ Mean Recall: {mean_recall:.4f}")
print(f"✅ Macro F1-Score: {f1_score:.4f}")

# ============================================================================
# Step 4: Visualize Confusion Matrix (Heatmap)
# ============================================================================
print("\n" + "-" * 50)
print("Step 4: Visualizing Confusion Matrix")
print("-" * 50)

plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix,
            cmap='Blues',
            xticklabels=all_letters,
            yticklabels=all_letters,
            annot=False,  # True if you want numbers (but becomes crowded)
            cbar=True,
            square=True)
plt.title(f'Confusion Matrix - K-Means (K={best_k}) after mapping to letters', fontsize=14)
plt.xlabel('Predicted Letter', fontsize=12)
plt.ylabel('Actual Letter', fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================================================
# Step 5: Save confusion matrix to CSV (without sklearn)
# ============================================================================
print("\n" + "-" * 50)
print("Step 5: Saving Confusion Matrix to CSV")
print("-" * 50)

cm_df = pd.DataFrame(confusion_matrix,
                     index=[f'Actual_{l}' for l in all_letters],
                     columns=[f'Pred_{l}' for l in all_letters])
cm_df.to_csv(f"kmeans_confusion_matrix_K{best_k}.csv")
print(f"✅ Saved confusion matrix to: kmeans_confusion_matrix_K{best_k}.csv")

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Mean Precision', 'Mean Recall', 'Macro F1-Score'],
    'Value': [accuracy, mean_precision, mean_recall, f1_score]
})
metrics_df.to_csv(f"kmeans_evaluation_metrics_K{best_k}.csv", index=False)
print(f"✅ Saved evaluation metrics to: kmeans_evaluation_metrics_K{best_k}.csv")

# ============================================================================
# Step 6: Print per-class precision and recall
# ============================================================================
print("\n" + "-" * 50)
print("Step 6: Per-Class Precision and Recall")
print("-" * 50)

print("\n{:<8} {:<12} {:<12} {:<12}".format('Letter', 'Precision', 'Recall', 'F1-Score'))
print("-" * 50)
for i, letter in enumerate(all_letters):
    p = precisions[i]
    r = recalls[i]
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
    print("{:<8} {:<12.4f} {:<12.4f} {:<12.4f}".format(letter, p, r, f1))

# ============================================================================
# Update FINAL SUMMARY with evaluation metrics
# ============================================================================
print("\n" + "=" * 70)
print("📊 FINAL SUMMARY - K-MEANS CLUSTERING (With Evaluation)")
print("=" * 70)
print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
print(f"Distance metric: Euclidean")
print(f"Optimal K (Silhouette Score): {best_k} (Score: {best_score:.4f})")
print(f"\n📈 Evaluation Metrics (after mapping clusters to letters):")
print(f"  - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"  - Mean Precision: {mean_precision:.4f}")
print(f"  - Mean Recall: {mean_recall:.4f}")
print(f"  - Macro F1-Score: {f1_score:.4f}")

print("\n📁 Output files generated:")
print(f"  - kmeans_labels_iter*.csv (cluster labels for each max_iter)")
print(f"  - kmeans_labels_history_iter*.csv (labels history for each max_iter)")
print(f"  - kmeans_iteration_analysis.csv (iteration performance summary)")
print(f"  - kmeans_cluster_stats_K{best_k}.csv (mean & std per feature per cluster)")
print(f"  - kmeans_clusters_K{best_k}.csv (final cluster labels)")
print(f"  - kmeans_confusion_matrix_K{best_k}.csv (confusion matrix)")
print(f"  - kmeans_evaluation_metrics_K{best_k}.csv (accuracy, precision, recall, f1)")

print("\n📊 Cluster sizes:")
sizes = [np.sum(final_labels == i) for i in range(best_k)]
for i, size in enumerate(sizes):
    print(f"  Cluster {i}: {size} samples ({size / len(X) * 100:.1f}%)")


























































































