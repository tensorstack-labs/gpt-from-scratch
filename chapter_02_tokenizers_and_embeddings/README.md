# 🧠 Chapter 2: Building Tokenizers and Vector Embeddings in PyTorch

[![YouTube Video](https://img.shields.io/badge/YouTube-Watch%20Masterclass-red)](https://youtu.be/E2Ix9RbryxQ)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_02_tokenizers_and_embeddings/Chapter2_Interactive_Notebook.ipynb)

In Chapter 2, we build the complete text ingestion and continuous representation pipeline in pure PyTorch from scratch.

---

## 🚀 Key Modules in this Directory

| File | Description | How to Run |
| :--- | :--- | :--- |
| **`01_simple_tokenizer.py`** | Custom Regex Tokenizer (V1 & V2) with `<\|unk\|>` and `<\|endoftext\|>` | `python 01_simple_tokenizer.py` |
| **`02_bpe_tiktoken_demo.py`** | OpenAI `tiktoken` BPE benchmark (`gpt2` vs `cl100k_base`) | `python 02_bpe_tiktoken_demo.py` |
| **`03_sliding_window_dataset.py`** | PyTorch `GPTDatasetV1` & `DataLoader` with +1 Target Shift | `python 03_sliding_window_dataset.py` |
| **`04_token_position_embeddings.py`** | `nn.Embedding` Token + Positional 3D Tensor Assembly | `python 04_token_position_embeddings.py` |
| **`Chapter2_Interactive_Notebook.ipynb`** | 1-Click Interactive Google Colab Notebook | Run via Colab button above |

---

## 📐 Mathematical Formulation

### 1. Autoregressive Input-Target Pairs
Given a sequence of token IDs $\mathbf{t} = (t_1, t_2, \dots, t_N)$:
$$\text{Input Chunk: } \mathbf{x} = (t_i, t_{i+1}, \dots, t_{i+L-1})$$
$$\text{Target Chunk: } \mathbf{y} = (t_{i+1}, t_{i+2}, \dots, t_{i+L})$$

### 2. Master Embedding Superposition
$$\mathbf{X} = \text{Dropout}(\mathbf{X}_{\text{token}} + \mathbf{X}_{\text{pos}}) \in \mathbb{R}^{\text{Batch} \times \text{Seq\_Len} \times d_{\text{model}}}$$
where:
- $\mathbf{X}_{\text{token}} = \mathbf{W}_{\text{token}}[\mathbf{x}] \in \mathbb{R}^{B \times L \times 768}$
- $\mathbf{X}_{\text{pos}} = \mathbf{W}_{\text{pos}}[\mathbf{p}] \in \mathbb{R}^{L \times 768}$
