# Transformer-Implementation-from-Scratch
**(Academic Coursework Project)**

## 📘 Project Type
**University Academic Coursework**

This project was completed as part of a university-level deep learning / natural language processing course.  
It is intended **strictly for academic and educational purposes** and implements core components of the **Transformer architecture** from scratch using **PyTorch**, without relying on high-level Transformer APIs.

---

## 📌 Project Overview

This coursework implements a **full Transformer model** (Encoder–Decoder architecture) based on the paper  
**“Attention Is All You Need” (Vaswani et al., 2017)**.

The implementation is modular and pedagogical, focusing on understanding and building:

- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Encoder & Decoder Blocks
- Masked Attention
- Layer Normalization
- Feedforward Networks

The model is applied to a **toy sequence-to-sequence task (arithmetic addition / subtraction)** to demonstrate end-to-end functionality.

---

## 🧠 Learning Objectives

This coursework demonstrates understanding of:

- Transformer architecture fundamentals
- Scaled dot-product attention
- Attention masking
- Encoder–decoder interaction
- Residual connections and LayerNorm
- Sequence modeling with PyTorch
- Custom loss functions (Cross-Entropy, Label Smoothing)
- Dataset and DataLoader design for sequence tasks

---


---

## 🏗️ Core Components Implemented

### 🔹 Tokenization & Preprocessing
- Vocabulary dictionary creation
- Digit-level tokenization for numbers
- Handling of special tokens (e.g. BOS, EOS)

### 🔹 Attention Mechanisms
- Scaled dot-product attention:
  - Single sequence (two-loop)
  - Batched (two-loop)
  - Batched (no-loop, matrix-based)
- Optional masking for decoder self-attention

### 🔹 Self-Attention & Multi-Head Attention
- Learnable linear projections for Q, K, V
- Parallel attention heads
- Concatenation and projection back to embedding space

### 🔹 Positional Encoding
- Simple positional encoding
- Sinusoidal positional encoding (as in original paper)

### 🔹 Encoder
- Stacked EncoderBlocks
- Each block includes:
  - Multi-Head Attention
  - Feedforward Network
  - Residual connections
  - Layer Normalization
  - Dropout

### 🔹 Decoder
- Masked self-attention
- Cross-attention with encoder output
- Feedforward layers
- Residual connections and normalization

### 🔹 Full Transformer Model
- Shared embedding layer
- Encoder–Decoder architecture
- Autoregressive decoding with masking

---

## 🧪 Dataset & Task

### Task Type
**Sequence-to-Sequence Learning**

- Input: Arithmetic expressions as token sequences  
  - Example: `BOS POSITIVE 0333 add POSITIVE 0696 EOS`
- Output: Result sequence  
  - Example: `BOS POSITIVE 1029 EOS`

### Dataset Class
- Custom `torch.utils.data.Dataset` (`AddSubDataset`)
- Provides:
  - Tokenized input sequence
  - Positional encodings
  - Tokenized target sequence

---

## 📉 Loss Functions Implemented

- **CrossEntropyLoss**
- **LabelSmoothingLoss**
  - Improves generalization
  - Reduces overconfidence

---

## ⚠️ Important Notes (Course Constraints)

- No use of `.cuda()` or `.to(device)` inside implementation blocks
- Attention and normalization layers implemented manually
- Designed for **learning clarity**, not production efficiency
- Many functions contain `TODO` sections as part of coursework requirements

---

## 🧾 Results & Takeaways

- Successfully implements a Transformer architecture from first principles
- Demonstrates correct data flow through encoder and decoder
- Reinforces understanding of attention, masking, and residual learning
- Highlights how high-level Transformer APIs abstract complex internals

📌 **Final Conclusion:**  
This coursework provides a **ground-up, educational implementation** of the Transformer model, emphasizing conceptual clarity and architectural understanding rather than library abstraction.

---

## 🛠️ Requirements

```bash
pip install torch numpy


## 📂 Project Structure

