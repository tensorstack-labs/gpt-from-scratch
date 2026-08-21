"""
TensorStack Chapter 3: Coding Multi-Head Self-Attention in Pure PyTorch
From First Principles (Sebastian Raschka's 'Build a Large Language Model (From Scratch)')
"""
import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        return attn_weights @ values

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        return attn_weights @ values

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout_p=0.1):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -float("inf"))
        attn_weights = self.dropout(torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1))
        return attn_weights @ values

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout=0.1, num_heads=12):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False)
        self.W_key   = nn.Linear(d_in, d_out, bias=False)
        self.W_value = nn.Linear(d_in, d_out, bias=False)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (queries @ keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -float("inf"))
        attn_weights = self.dropout(torch.softmax(scores, dim=-1))

        context = (attn_weights @ values).transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)

if __name__ == "__main__":
    torch.manual_seed(123)
    inputs = torch.randn(2, 6, 768)
    mha = MultiHeadAttention(d_in=768, d_out=768, context_len=6, dropout=0.1, num_heads=12)
    outputs = mha(inputs)
    print(f"✔ MultiHeadAttention Output shape: {outputs.shape}")
    assert outputs.shape == (2, 6, 768)
    print("🎉 Chapter 3 PyTorch Unit Tests Passed!")
