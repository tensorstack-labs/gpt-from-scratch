# 🧠 Chapter 4: Building the 124M GPT Architecture in PyTorch

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Masterclass-red)](https://youtu.be/3HoO7wiMvNY)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_04_gpt_architecture/Chapter4_Interactive_Notebook.ipynb)

In Chapter 4, we assemble the complete 124-Million Parameter GPT-2 Model Architecture in pure PyTorch from first principles:

1. **Layer Normalization (`LayerNorm`)**: Normalizing feature coordinates per token to stabilize deep 12-layer residual streams.
2. **GELU Non-Linearity**: Smooth Gaussian Error Linear Unit activation replacing sharp, dying ReLUs.
3. **Feed-Forward Network (FFN)**: 4x dimension expansion ($768 \rightarrow 3072 \rightarrow 768$) acting as associative memory banks.
4. **Residual Shortcut Connections**: Dual skip connections ($x + F(x)$) creating uninterrupted backpropagation superhighways.
5. **Pre-LayerNorm `TransformerBlock`**: Stacking Multi-Head Attention, LayerNorm, and FFN.
6. **Complete `GPTModel` (124M)**: Token embeddings + Positional embeddings $\rightarrow 12 \times$ Transformer Blocks $\rightarrow$ Final LayerNorm $\rightarrow$ LM Output Head.
7. **Autoregressive Text Generation**: Greedy decoding loop generating new tokens one by one.

---

## 🚀 Quickstart

Run the complete GPT model architecture test suite:

```bash
python gpt_architecture.py
```
