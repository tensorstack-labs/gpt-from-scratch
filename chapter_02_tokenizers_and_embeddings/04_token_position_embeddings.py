"""
04_token_position_embeddings.py
TensorStack - Chapter 2: Token and Positional Embedding Layers

Implements:
1. Token Embedding Layer: nn.Embedding(vocab_size, d_model).
2. Positional Embedding Layer: nn.Embedding(context_length, d_model).
3. Final Master Input Tensor Assembly: X = Token_Embeddings + Pos_Embeddings.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MasterEmbeddingPipeline(nn.Module):
    """
    Combines token embeddings and absolute learnable positional embeddings
    into a single 3D continuous tensor ready for Transformer Multi-Head Attention.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 1024,
        d_model: int = 768,
        drop_rate: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length

        # 1. Token Embedding Table [vocab_size, d_model]
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # 2. Absolute Learnable Positional Embedding Table [context_length, d_model]
        self.position_embeddings = nn.Embedding(context_length, d_model)

        # 3. Regularization Dropout Layer
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            in_idx: 2D integer tensor of shape [batch_size, num_tokens]
        Returns:
            3D continuous float tensor of shape [batch_size, num_tokens, d_model]
        """
        batch_size, num_tokens = in_idx.shape
        assert num_tokens <= self.context_length, (
            f"Input sequence length ({num_tokens}) exceeds maximum context length ({self.context_length})"
        )

        # Step 1: Lookup Token Embeddings -> [batch_size, num_tokens, d_model]
        tok_embeds = self.token_embeddings(in_idx)

        # Step 2: Generate Positional IDs [0, 1, ..., num_tokens-1]
        pos_ids = torch.arange(num_tokens, device=in_idx.device)

        # Step 3: Lookup Positional Embeddings -> [num_tokens, d_model]
        pos_embeds = self.position_embeddings(pos_ids)

        # Step 4: Element-wise Broadcast Addition (Superposition)
        # [batch_size, num_tokens, d_model] + [num_tokens, d_model]
        input_tensor = tok_embeds + pos_embeds

        # Step 5: Dropout
        output_tensor = self.dropout(input_tensor)
        return output_tensor


if __name__ == "__main__":
    print("==================================================")
    print("🚀 TENSORSTACK: MASTER EMBEDDING PIPELINE DEMO")
    print("==================================================")
    
    vocab_size = 50257
    context_length = 1024
    d_model = 768
    batch_size = 4
    seq_len = 8
    
    pipeline = MasterEmbeddingPipeline(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        drop_rate=0.1
    )
    
    print(f"✔ Initialized MasterEmbeddingPipeline:")
    print(f"  • Token Embedding Table:     {pipeline.token_embeddings.weight.shape}")
    print(f"  • Positional Embedding Table: {pipeline.position_embeddings.weight.shape}")
    print(f"  • Embedding Dimension:       d_model = {d_model}")
    print(f"  • Context Window Capacity:   max_length = {context_length}\n")
    
    # Simulate a batched input tensor from DataLoader [Batch=4, Seq_Len=8]
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"📦 Simulated Input Batch Tensor (x):")
    print(f"  Shape: {dummy_input_ids.shape}   # [Batch_Size={batch_size}, Seq_Len={seq_len}]")
    print(f"  Values:\n{dummy_input_ids}\n")
    
    # Forward pass
    output_master_tensor = pipeline(dummy_input_ids)
    print(f"🌟 OUTPUT MASTER TENSOR (Ready for Attention Block):")
    print(f"  Shape: {output_master_tensor.shape}   # [Batch, Seq_Len, d_model]")
    print(f"  Data Type: {output_master_tensor.dtype}")
    print(f"  Device:    {output_master_tensor.device}")
    print("==================================================")
