# Hardware-Efficient MLP Inference Engine on FPGA (Xilinx Artix-7)

This repository contains a complete, from-scratch implementation of a **Multi-Layer Perceptron (MLP)** neural network trained in Python (using only NumPy) and deployed as a highly optimized, sequential hardware inference engine in **Verilog HDL** using Xilinx Vivado.

The project demonstrates the entire pipeline of digital hardware design for AI: from custom training and fixed-point quantization (Q8.8) to Finite State Machine (FSM) control logic and hardware behavioral simulation.

---

## 🚀 Key Features

* **Zero High-Level AI Frameworks:** Built completely from scratch in Python using only `NumPy`. No TensorFlow, PyTorch, or Keras.
* **Sequential Hardware Architecture:** Optimized for minimal FPGA resource utilization. Uses a single Multiply-Accumulate (MAC) unit steered by an FSM instead of massive parallel multipliers, making it highly area-efficient.
* **16-bit Fixed-Point Precision (Q8.8):** Eliminates heavy floating-point hardware. Implements a dedicated quantization downshift inside the Verilog matrix multiplication loop.
* **Hardware ReLU Activation:** Implemented on-the-fly using a zero-overhead bit-sign check (`acc[31]`).
* **Optimized Argmax (No Softmax Needed):** Uses a hardware comparator loop on raw output logits, bypassing resource-heavy exponential computations since Softmax preserves monotonicity.

---

## 📐 Network Architecture

* **Input Layer:** 784 neurons (Flattened 28x28 MNIST-like images)
* **Hidden Layer:** 128 neurons with custom ReLU activation
* **Output Layer:** 10 neurons (Representing classification scores for digits 0-9)

---

