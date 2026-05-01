import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# =========================
# Load dataset
# =========================
data = pd.read_csv("letter-recognition.data", header=None)

X = data.iloc[:, 1:].values   # features
y = data.iloc[:, 0].values    # labels

# =========================
# Normalize features (Min-Max)
# =========================
def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X = normalize(X)

# =========================
# Train-Test Split (80/20)
# =========================
def train_test_split(X, y, test_size=0.2):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    split = int(len(X) * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# =========================
# Distance functions
# =========================
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

# =========================
# KNN prediction
# =========================
def knn_predict(X_train, y_train, x_test, k, distance_func):
    distances = []

    for i in range(len(X_train)):
        dist = distance_func(x_test, X_train[i])
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    labels = [label for _, label in neighbors]

    return Counter(labels).most_common(1)[0][0]

# =========================
# Predict all
# =========================
def predict_all(X_train, y_train, X_test, k, distance_func):
    preds = []

    for x in X_test:
        preds.append(knn_predict(X_train, y_train, x, k, distance_func))

    return np.array(preds)

# =========================
# Metrics
# =========================
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall(y_true, y_pred):
    classes = np.unique(y_true)

    precisions = []
    recalls = []

    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))

        #1e-9---> prevent Division by Zero
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)

# =========================
# Confusion Matrix
# =========================
def confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    matrix = pd.DataFrame(0, index=labels, columns=labels)

    for t, p in zip(y_true, y_pred):
        matrix.loc[t, p] += 1

    return matrix

# =========================
# Experiment setup
# =========================
ks = [1, 3, 5, 9]

distance_methods = {
    "Euclidean": euclidean,
    "Manhattan": manhattan
}

results = {}

# =========================
# Run experiments
# =========================
for dist_name, dist_func in distance_methods.items():

    results[dist_name] = []

    print("\n========================")
    print("Distance:", dist_name)
    print("========================")

    for k in ks:

        y_pred = predict_all(X_train, y_train, X_test, k, dist_func)

        acc = accuracy(y_test, y_pred)
        prec, rec = precision_recall(y_test, y_pred)

        results[dist_name].append(acc)

        print(f"K = {k}")
        print("Accuracy =", acc)
        print("Precision =", prec)
        print("Recall =", rec)
        print("-------------------")




        df_out = pd.DataFrame(X_test, columns=[f"feat_{i + 1}" for i in range(X_test.shape[1])])
        df_out.insert(0, "True", y_test)
        df_out.insert(1, "Predicted", y_pred)
        df_out.to_csv(f"output_{dist_name}_K{k}.csv", index=False)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, cmap="Blues")
        plt.title(f"Confusion Matrix ({dist_name}, K={k})")
        plt.show()

# =========================
# Compare results
# =========================
plt.figure()

for dist_name in distance_methods.keys():
    plt.plot(ks, results[dist_name], marker='o', label=dist_name)

plt.xlabel("K value")
plt.ylabel("Accuracy")
plt.title("KNN Comparison (Euclidean vs Manhattan)")
plt.legend()
plt.show()



# =========================
# Balanced Sampling
# =========================
unique_classes = np.unique(y_test)
samples_per_class = 50

sample_idx = []
for c in unique_classes:
    class_idx = np.where(y_test == c)[0]
    chosen = np.random.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
    sample_idx.extend(chosen)

sample_idx = np.array(sample_idx)
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]

# =========================
# Scatter Plot
# =========================
n_features = X_sample.shape[1]  # 16
fig, axes = plt.subplots(n_features, n_features, figsize=(25, 25))

for i in range(n_features):
    for j in range(n_features):
        ax = axes[i, j]

        if i == j:

            ax.axis('off')

        else:

            ax.scatter(X_sample[:, j], X_sample[:, i],
                       c=pd.factorize(y_sample)[0], cmap='tab20',
                       alpha=0.5, s=5)


        if i == n_features - 1:
            ax.set_xlabel(f'F{j + 1}')
        if j == 0:
            ax.set_ylabel(f'F{i + 1}')


        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Scatter Plot Matrix - All 16 Features vs All Features", fontsize=16)
plt.tight_layout()
plt.show()

# =========================
# Pair Plot
# =========================
print("Pair Plot")

sample_df = pd.DataFrame(X_sample)
sample_df["label"] = y_sample

sns.pairplot(sample_df, hue="label")
plt.title("Pair Plot")
plt.show()

# =========================
# 3) Strip Plot - All 16 Features
# =========================
# Create a 4x4 grid for 16 features
fig, axes = plt.subplots(4, 4, figsize=(24, 20))
axes = axes.flatten()

for i in range(16):  # Loop over all 16 features
    ax = axes[i]

    # Create dataframe with current feature and labels
    df_plot = pd.DataFrame({
        "Feature": X_sample[:, i],
        "Label": y_sample
    })

    # Draw strip plot
    sns.stripplot(x="Label", y="Feature", data=df_plot, ax=ax, alpha=0.5, size=2)

    # Set title for each subplot
    ax.set_title(f"Feature {i + 1}")

    # Rotate x-axis labels for better readability (26 letters)
    ax.tick_params(axis='x', rotation=90)

    # Show y-axis label only on first column of each row (i % 4 == 0)
    if i % 4 != 0:
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Value")

plt.suptitle("3) Strip Plot - All 16 Features Distribution by Class", fontsize=16)
plt.tight_layout()
plt.show()

# =========================
# 4) Swarm Plot - All 16 Features
# =========================
# Note: Swarm plot may fail when points are too dense (warning shown)
# try-except will automatically fall back to strip plot

fig, axes = plt.subplots(4, 4, figsize=(24, 20))
axes = axes.flatten()

for i in range(16):  # Loop over all 16 features
    ax = axes[i]

    # Create dataframe with current feature and labels
    df_plot = pd.DataFrame({
        "Feature": X_sample[:, i],
        "Label": y_sample
    })

    try:
        # Try to draw swarm plot (better visualization but may fail)
        sns.swarmplot(x="Label", y="Feature", data=df_plot, ax=ax, alpha=0.5, size=2)
    except:
        # Fallback to strip plot if swarm plot fails (too many points)
        sns.stripplot(x="Label", y="Feature", data=df_plot, ax=ax, alpha=0.5, size=2)
        ax.set_title(f"Feature {i + 1} (Strip alternative)")
    else:
        ax.set_title(f"Feature {i + 1}")

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=90)

    # Hide y-axis labels for non-first columns
    if i % 4 != 0:
        ax.set_ylabel("")

plt.suptitle("4) Swarm Plot - All 16 Features by Class", fontsize=16)
plt.tight_layout()
plt.show()









