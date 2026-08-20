# 🧠 TensorStack: Build a Large Language Model from Scratch in PyTorch

<p align="center">
  <img src="https://raw.githubusercontent.com/tensorstack-labs/gpt-from-scratch/main/assets/tensorstack_banner.png" alt="TensorStack Banner" width="100%">
</p>

<p align="center">
  <a href="https://www.youtube.com/playlist?list=PLbb--db9n7PA"><img src="https://img.shields.io/badge/YouTube-TensorStack%20Masterclass-red?style=for-the-badge&logo=youtube" alt="YouTube Course"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License"></a>
</p>

---

## 📖 About TensorStack
Welcome to the official repository for **TensorStack: Build a Large Language Model from Scratch in PyTorch**.

Unlike high-level courses that rely on black-box model wrappers (`AutoModelForCausalLM`), this masterclass builds **every single layer, tokenizer, multi-head self-attention block, and pretraining loop from first principles using pure PyTorch (`import torch`, `torch.nn`)**.

---

## 🗺️ Masterclass Curriculum & Interactive Notebooks

| Chapter | Topic | YouTube Masterclass | Run in Google Colab | Local Source Code |
| :--- | :--- | :--- | :--- | :--- |
| **Chapter 01** | **How Large Language Models Actually Work** | [▶ Watch Episode 1](https://youtu.be/rI20fupI2Sk) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_01_llm_overview/Chapter1_Interactive_Architecture_Demo.ipynb) | [Chapter 1 Code](./chapter_01_llm_overview) |
| **Chapter 02** | **Tokenizers & Vector Embeddings from Scratch** | [▶ Watch Episode 2](https://youtu.be/E2Ix9RbryxQ) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_02_tokenizers_and_embeddings/Chapter2_Interactive_Notebook.ipynb) | [Chapter 2 Code](./chapter_02_tokenizers_and_embeddings) |
| **Chapter 03** | **Coding Multi-Head Self-Attention in PyTorch** | *⏳ Dropping Soon* | *⏳ In Progress* | *⏳ Dropping Soon* |
| **Chapter 04** | **124M GPT Architecture & Transformer Blocks** | *⏳ Dropping Soon* | *⏳ In Progress* | *⏳ Dropping Soon* |
| **Chapter 05** | **Pretraining Loops & Loss Optimization** | *⏳ Dropping Soon* | *⏳ In Progress* | *⏳ Dropping Soon* |
| **Chapter 06** | **Instruction Fine-Tuning & Alignment** | *⏳ Dropping Soon* | *⏳ In Progress* | *⏳ Dropping Soon* |

---

## 🚀 How to Run the Code

### 🌐 Option A: Run in Google Colab (Zero Installation)
You can run and test every chapter directly in your web browser with 1 click:
- **Chapter 1 Colab Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_01_llm_overview/Chapter1_Interactive_Architecture_Demo.ipynb)
- **Chapter 2 Colab Notebook**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorstack-labs/gpt-from-scratch/blob/main/chapter_02_tokenizers_and_embeddings/Chapter2_Interactive_Notebook.ipynb)

---

### 💻 Option B: Run Locally on Your Machine

#### 1. Clone the Repository
```bash
git clone https://github.com/tensorstack-labs/gpt-from-scratch.git
cd gpt-from-scratch
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run Chapter 1 Code
```bash
# High-level architecture mental models and tensor intuition demo
python chapter_01_llm_overview/01_tokenizer_vs_embedding_intuition.py
```

#### 4. Run Chapter 2 Code
```bash
# 1. Custom First-Principles Regex Tokenizer (V1 & V2)
python chapter_02_tokenizers_and_embeddings/01_simple_tokenizer.py

# 2. Production Tiktoken & BPE Subwords Demo
python chapter_02_tokenizers_and_embeddings/02_bpe_tiktoken_demo.py

# 3. PyTorch Sliding Window Dataset & DataLoader (+1 Target Shift)
python chapter_02_tokenizers_and_embeddings/03_sliding_window_dataset.py

# 4. Token + Positional Embedding Layers (3D Input Tensor Assembly)
python chapter_02_tokenizers_and_embeddings/04_token_position_embeddings.py
```

---

## 📂 Repository Directory Structure

```bash
gpt-from-scratch/
│
├── README.md                           # Master course guide with Colab badges & instructions
├── requirements.txt                    # Minimal dependencies (torch, tiktoken, numpy)
├── LICENSE                             # Open-source MIT License
│
├── chapter_01_llm_overview/
│   ├── README.md                       # Chapter 1 notes & architecture breakdown
│   ├── 01_tokenizer_vs_embedding_intuition.py # Discrete vs Continuous tensor demo
│   └── Chapter1_Interactive_Architecture_Demo.ipynb # 1-Click Colab Notebook
│
└── chapter_02_tokenizers_and_embeddings/
    ├── README.md                       # Chapter 2 deep-dive guide & formulas
    ├── 01_simple_tokenizer.py          # Custom regex tokenizer from first principles
    ├── 02_bpe_tiktoken_demo.py         # Production OpenAI Tiktoken & BPE demo
    ├── 03_sliding_window_dataset.py    # PyTorch Dataset & DataLoader pipeline
    ├── 04_token_position_embeddings.py # nn.Embedding Token + Positional pipeline
    └── Chapter2_Interactive_Notebook.ipynb # 1-Click Colab Notebook
```

---

## 🤝 Community & Support
- **YouTube Masterclass**: [Subscribe to TensorStack on YouTube](https://www.youtube.com/@TensorStack)
- **Discussion & Questions**: Leave comments on our YouTube videos or open a GitHub Issue here!
- **Star the Repo**: If you find this course helpful, star ⭐ this repository to support open-source AI education!

---
## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
