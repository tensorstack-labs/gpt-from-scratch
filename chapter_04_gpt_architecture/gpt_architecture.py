"""
TensorStack Chapter 4: Building the 124M GPT Architecture in Pure PyTorch
From First Principles (Sebastian Raschka's 'Build a Large Language Model (From Scratch)')
"""
import torch
import torch.nn as nn

# 1. Custom Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# 2. GELU Non-Linear Activation
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# 3. Feed-Forward Network (FFN)
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"])
        )

    def forward(self, x):
        return self.layers(x)

# 4. Multi-Head Attention (from Chapter 3)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout, num_heads):
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

# 5. Transformer Block (Pre-LayerNorm)
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_len=cfg["context_length"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

# 6. Complete 124M GPT-2 Model Architecture
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        b, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# 7. Autoregressive Text Generation Function
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✔ 124M GPT-2 Model Instantiated Successfully!")
    print(f"✔ Total Trainable Parameters: {total_params:,}")
    
    # Test Forward Pass
    sample_tokens = torch.randint(0, 50257, (2, 6))
    out_logits = model(sample_tokens)
    print(f"✔ Input shape:  {sample_tokens.shape}")
    print(f"✔ Logits shape: {out_logits.shape}")
    assert out_logits.shape == (2, 6, 50257)
    
    # Test Generation Loop
    gen_tokens = generate_text_simple(model, sample_tokens, max_new_tokens=5, context_size=1024)
    print(f"✔ Generated shape: {gen_tokens.shape}")
    assert gen_tokens.shape == (2, 11)
    print("🎉 All Chapter 4 Model & Generation Tests Passed!")
