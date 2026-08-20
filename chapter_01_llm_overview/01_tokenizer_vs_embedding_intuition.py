"""
01_tokenizer_vs_embedding_intuition.py
TensorStack - Chapter 1: How LLMs Actually Work

Demonstrates:
1. The fundamental difference between Rule-Based Spam Filters vs Statistical ML vs Deep Learning.
2. Conceptual breakdown of Discrete String -> Integer ID -> 768-dim Vector Projection.
3. The 3 Stages of Building a Large Language Model (Architecture -> Pretraining -> Fine-Tuning).
"""

import torch
import torch.nn as nn


def demonstrate_llm_architecture_blueprint():
    print("==================================================")
    print("🚀 TENSORSTACK - CHAPTER 1: LLM BLUEPRINT DEMO")
    print("==================================================")
    
    # 1. Discrete Text Input
    text = "Machine learning models learn patterns"
    words = text.split()
    print(f"📝 Raw Text Input: \"{text}\"")
    print(f"🔤 Discrete Word Tokens: {words}")
    
    # 2. Integer ID Mapping (Tokenizer Step)
    vocab = {word: idx for idx, word in enumerate(words)}
    token_ids = torch.tensor([vocab[w] for w in words], dtype=torch.long)
    print(f"🔢 Integer Token IDs:   {token_ids.tolist()}")
    
    # 3. Continuous Vector Embedding Step (768 Dimensions)
    d_model = 768
    embedding_table = nn.Embedding(len(vocab), d_model)
    continuous_vectors = embedding_table(token_ids)
    
    print(f"\n🌟 Continuous Embedding Tensor Shape: {continuous_vectors.shape}")
    print(f"  • Number of Tokens: {continuous_vectors.shape[0]}")
    print(f"  • Hidden Dimension: d_model = {continuous_vectors.shape[1]}")
    print(f"  • First Token Vector (first 5 floats): {continuous_vectors[0, :5].tolist()}...")
    
    print("\n--------------------------------------------------")
    print("🏛️ THE 3 STAGES OF BUILDING AN LLM:")
    print("  1. Architecture Blueprint: Code Tokenizers, Embeddings & Transformer Blocks (Untrained)")
    print("  2. Pretraining: Train on Trillions of Web Tokens to predict next token (Foundation Model)")
    print("  3. Instruction Fine-Tuning: Q&A Alignment + RLHF (ChatGPT / Claude / LLaMA)")
    print("==================================================")


if __name__ == "__main__":
    demonstrate_llm_architecture_blueprint()
