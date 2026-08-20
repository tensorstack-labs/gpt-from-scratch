"""
02_bpe_tiktoken_demo.py
TensorStack - Chapter 2: Byte Pair Encoding & Production Tiktoken

Demonstrates:
1. The Byte Pair Encoding (BPE) subword concept.
2. Production tokenization using OpenAI's fast Rust-based tiktoken library.
3. Comparative tokenization across GPT-2 ('gpt2') and GPT-4 ('cl100k_base').
"""

import tiktoken
from typing import List


def run_tiktoken_benchmarks():
    print("==================================================")
    print("🚀 TENSORSTACK: PRODUCTION TIKTOKEN BENCHMARK")
    print("==================================================")
    
    # 1. Load GPT-2 Tokenizer (Vocab size: 50,257)
    enc_gpt2 = tiktoken.get_encoding("gpt2")
    print(f"✔ Loaded 'gpt2' Tokenizer (Vocab Size: {enc_gpt2.n_vocab:,})")
    
    # 2. Load GPT-4 Tokenizer (Vocab size: 100,277)
    enc_gpt4 = tiktoken.get_encoding("cl100k_base")
    print(f"✔ Loaded 'cl100k_base' Tokenizer (Vocab Size: {enc_gpt4.n_vocab:,})")
    
    # 3. Test Strings
    test_cases = [
        "Hello, world! Welcome to TensorStack LLM Masterclass.",
        "def self_attention(query, key, value, mask=None):",
        "Akiro Kurosawa's 1954 masterpiece: 七人の侍 (Seven Samurai)",
        "supercalifragilisticexpialidocious"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n--------------------------------------------------")
        print(f"📌 TEST CASE {i}: \"{text}\"")
        
        # GPT-2 Encoding
        ids_gpt2 = enc_gpt2.encode(text)
        subwords_gpt2 = [enc_gpt2.decode([t]) for t in ids_gpt2]
        
        # GPT-4 Encoding
        ids_gpt4 = enc_gpt4.encode(text)
        subwords_gpt4 = [enc_gpt4.decode([t]) for t in ids_gpt4]
        
        print(f"  [GPT-2] Tokens: {len(ids_gpt2)} | IDs: {ids_gpt2[:6]}...")
        print(f"          Subwords: {subwords_gpt2}")
        print(f"  [GPT-4] Tokens: {len(ids_gpt4)} | IDs: {ids_gpt4[:6]}...")
        print(f"          Subwords: {subwords_gpt4}")
        
    # 4. Handling Special Tokens in Tiktoken
    special_text = "Document A <|endoftext|> Document B"
    special_ids = enc_gpt2.encode(special_text, allowed_special={"<|endoftext|>"})
    print("\n--------------------------------------------------")
    print("🔒 SPECIAL TOKEN ENCODING:")
    print(f"  Input:  \"{special_text}\"")
    print(f"  IDs:    {special_ids}")
    print(f"  Token 50256 decoded: \"{enc_gpt2.decode([50256])}\"")
    print("==================================================")


if __name__ == "__main__":
    run_tiktoken_benchmarks()
