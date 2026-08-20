"""
01_simple_tokenizer.py
TensorStack - Chapter 2: Building Tokenizers from First Principles

Implements:
1. SimpleTokenizerV1: Basic regex token splitter and vocabulary indexer.
2. SimpleTokenizerV2: Enhanced tokenizer with special tokens (<|endoftext|>, <|unk|>).
"""

import re
from typing import List, Dict


class SimpleTokenizerV1:
    """
    A foundational word-level tokenizer that splits on whitespace and punctuation.
    """
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        # Split text on punctuation, dashes, and whitespace
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        token_ids = [self.str_to_int[token] for token in tokens]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.int_to_str[i] for i in token_ids]
        text = " ".join(tokens)
        # Clean up spaces before punctuation marks
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        text = re.sub(r'\s+--\s+', '--', text)
        return text


class SimpleTokenizerV2:
    """
    Enhanced tokenizer that supports special tokens:
    - <|unk|>: Handles Out-Of-Vocabulary words without throwing KeyError
    - <|endoftext|>: Marks document boundaries for autoregressive training
    """
    def __init__(self, vocab: Dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> List[int]:
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token.strip() for token in tokens if token.strip()]
        # Map unseen words to <|unk|> ID
        token_ids = [
            self.str_to_int.get(token, self.str_to_int["<|unk|>"])
            for token in tokens
        ]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        tokens = [self.int_to_str[i] for i in token_ids]
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        text = re.sub(r'\s+--\s+', '--', text)
        return text


def build_vocab_from_text(raw_text: str) -> Dict[str, int]:
    """Extracts unique tokens and assigns integer IDs."""
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    tokens = [token.strip() for token in tokens if token.strip()]
    unique_tokens = sorted(list(set(tokens)))
    
    # Add special tokens
    unique_tokens.extend(["<|endoftext|>", "<|unk|>"])
    
    vocab = {token: idx for idx, token in enumerate(unique_tokens)}
    return vocab


if __name__ == "__main__":
    sample_corpus = (
        "Hello, world. This is a first-principles demonstration of building a "
        "tokenizer from scratch in Python for TensorStack's LLM course!"
    )
    
    print("==================================================")
    print("🚀 TENSORSTACK: CUSTOM REGEX TOKENIZER DEMO")
    print("==================================================")
    
    vocab = build_vocab_from_text(sample_corpus)
    print(f"✔ Vocabulary Size: {len(vocab)} unique tokens")
    
    tokenizer = SimpleTokenizerV2(vocab)
    
    # Test encoding
    test_phrase = "Hello, world! Building LLMs is exciting."
    encoded_ids = tokenizer.encode(test_phrase)
    print(f"\n📝 Raw Input:  \"{test_phrase}\"")
    print(f"🔢 Encoded IDs: {encoded_ids}")
    
    # Test decoding
    decoded_phrase = tokenizer.decode(encoded_ids)
    print(f"🔄 Decoded:    \"{decoded_phrase}\"")
    print("==================================================")
