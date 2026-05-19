import numpy as np
import pandas as pd


# =========================================================
# 1. DATA LOADING & PREPROCESSING
# =========================================================
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    return X_train, y_train, X_test, y_test


def min_max_scale(X_train, X_test):
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    range_ = X_max - X_min
    range_[range_ == 0] = 1
    X_train_scaled = (X_train - X_min) / range_
    X_test_scaled = (X_test - X_min) / range_
    return X_train_scaled, X_test_scaled


# =========================================================
# 2. ACTIVATION & HELPER FUNCTIONS
# =========================================================
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exps = np.exp(shift_z)
    return exps / np.sum(exps, axis=1, keepdims=True)



# =========================================================
# Cross Entropy Loss Function
# =========================================================
def cross_entropy_loss(y_pred, y_true):
    """
    Explicit categorical cross entropy loss.
    """

    epsilon = 1e-15

    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / y_true.shape[0]

    return loss

def one_hot(y, num_classes=10):
    encoded = np.zeros((y.shape[0], num_classes))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded


# =========================================================
# 3. FIXED-POINT SIMULATION LOGIC (KEY ADDITION)
# =========================================================
def simulate_fixed_point(array, bits=16, frac_bits=8):
    """
    Simulates the quantization error that happens in Hardware.
    """
    step = 1.0 / (1 << frac_bits)
    min_val = -(1 << (bits - frac_bits - 1))
    max_val = (1 << (bits - frac_bits - 1)) - step

    # Quantization process: Scale -> Round -> Clip -> Rescale
    quantized = np.round(array * (1 << frac_bits)) / (1 << frac_bits)
    return np.clip(quantized, min_val, max_val)


def float_to_hex(val, bits=16, frac_bits=8):
    scaled = int(round(val * (1 << frac_bits)))
    min_val = -(1 << (bits - 1))
    max_val = (1 << (bits - 1)) - 1
    scaled = max(min_val, min(scaled, max_val))
    if scaled < 0:
        scaled = (1 << bits) + scaled
    return f"{scaled:04X}"


# =========================================================
# 4. MLP MODEL
# =========================================================
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.z2

    def backward(self, X, y_true):
        n = X.shape[0]
        dZ2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, dZ2) / n
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / n
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=50, batch_size=32):

        y_one_hot = one_hot(y)

        n = X.shape[0]

        for epoch in range(epochs):

            idx = np.random.permutation(n)

            X, y_one_hot = X[idx], y_one_hot[idx]

            total_loss = 0

            for i in range(0, n, batch_size):
                X_batch = X[i:i + batch_size]

                y_batch = y_one_hot[i:i + batch_size]

                # Forward pass
                self.forward(X_batch)

                # Explicit Cross Entropy Loss
                loss = cross_entropy_loss(self.a2, y_batch)

                total_loss += loss

                # Backpropagation
                self.backward(X_batch, y_batch)

            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    # --- Quantization Methods ---
    def apply_quantization(self):
        """Applies fixed-point simulation to the current weights."""
        self.W1 = simulate_fixed_point(self.W1)
        self.b1 = simulate_fixed_point(self.b1)
        self.W2 = simulate_fixed_point(self.W2)
        self.b2 = simulate_fixed_point(self.b2)

    def save_all(self):
        for name, arr in [("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2)]:
            flat = arr.flatten()
            hex_data = [float_to_hex(v) for v in flat]
            with open(f"{name}.mem", "w") as f:
                f.write("\n".join(hex_data))
        print("Hardware .mem files saved.")

    # =========================================================
    # Save Predictions
    # =========================================================
    def save_predictions(self, X_test, y_test, filename):

        predictions = self.predict(X_test)

        with open(filename, "w") as f:
            # Header
            f.write("Predicted True\n")

            # Save prediction and ground truth
            for pred, true_label in zip(predictions, y_test):
                f.write(f"{pred} {true_label}\n")

        print(f"{filename} saved successfully.")


# =========================================================
# 5. MAIN EXECUTION
# =========================================================
def main():
    X_train, y_train, X_test, y_test = load_data("train.csv", "test.csv")
    X_train, X_test = min_max_scale(X_train, X_test)

    model = MLP(784, 128, 10, lr=0.01)

    print("Step 1: Training standard Float64 model...")
    model.train(X_train, y_train, epochs=30)

    acc_float = model.accuracy(X_test, y_test)
    print(f"Original Float64 Test Accuracy: {acc_float * 100:.2f}%")

    print("\nStep 2: Simulating Hardware Quantization (Fixed-Point 16-bit)...")
    # Also quantize the test input to be realistic
    X_test_q = simulate_fixed_point(X_test)
    model.apply_quantization()

    acc_quant = model.accuracy(X_test_q, y_test)
    print(f"Quantized (Simulated HW) Test Accuracy: {acc_quant * 100:.2f}%")
    print(f"Accuracy Drop: {(acc_float - acc_quant) * 100:.2f}%")

    # Step 3: Save files for Hardware
    model.save_all()
    # =====================================================
    # Save Software Prediction Results
    # =====================================================

    model.save_predictions(
        X_test,
        y_test,
        "software_predictions.txt"
    )

    model.save_predictions(
        X_test_q,
        y_test,
        "quantized_predictions.txt"
    )


if __name__ == "__main__":
    main()