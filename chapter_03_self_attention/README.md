# 🧠 Chapter 3: Coding Multi-Head Self-Attention in PyTorch

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Masterclass-red)](https://youtu.be/80MwvIFL_T4)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_03_self_attention/Chapter3_Interactive_Notebook.ipynb)

In Chapter 3, we implement the complete attention mechanism that powers modern Large Language Models:

1. **Simplified Dot-Product Self-Attention**: Pure similarity matrices without weights.
2. **Softmax Normalization & $\sqrt{d_k}$ Scaling**: Preventing vanishing gradients in high-dimensional spaces.
3. **Trainable Projections ($W_q, W_k, W_v$)**: Custom Query, Key, and Value linear transformations.
4. **Causal Attention Masking**: The lower-triangular $-\infty$ matrix to prevent future token leaks.
5. **Dropout Regularization**: Robust weight regularizations.
6. **Multi-Head Attention**: Splitting $d_{\text{model}} \rightarrow (h \times d_k)$ parallel attention heads in single GPU tensor operations.

---

## 🚀 Quickstart

Run the self-attention test suite:

```bash
python multi_head_attention.py
```
