# 🧠 Chapter 1: How Large Language Models Actually Work

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Masterclass-red)](https://youtu.be/rI20fupI2Sk)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_01_llm_overview/Chapter1_Interactive_Architecture_Demo.ipynb)

In Chapter 1, we unpack the foundational architecture of modern Generative AI and Transformers from first principles.

---

## 🎯 What You Learn
1. **The Evolution of NLP**: Why Rule-Based Systems and RNNs hit a computational bottleneck until the 2017 Transformer breakthrough.
2. **Autoregressive Next-Token Prediction**: How LLMs generate language token-by-token using causal masking.
3. **The 3 Stages of LLM Development**:
   - **Stage 1 (Architecture)**: Tokenizers, Embedding layers, Multi-Head Attention, and Residual Feed-Forward networks.
   - **Stage 2 (Pretraining)**: Self-supervised learning across billions of parameters to minimize cross-entropy loss.
   - **Stage 3 (Instruction Fine-Tuning & Alignment)**: SFT & RLHF converting raw base text generators into helpful AI assistants.

---

## 💻 Running the Code
```bash
# Run local script
python chapter_01_llm_overview/01_tokenizer_vs_embedding_intuition.py
```
Or launch the 1-click Google Colab notebook above!
