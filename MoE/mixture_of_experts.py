import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ['TRANSFORMERS_OFFLINE']   = '1'
os.environ['HF_DATASETS_OFFLINE']    = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 1. hyperparams
num_layers    = 4
num_experts   = 4
num_heads     = 8
batch_size    = 16
vocab_size    = 50257
d_model       = 512
embd_dim      = 512
block_size    = 256
head_dim      = d_model // num_heads
learning_rate = 3e-4
epochs        = 3
aux_loss_coef = 0.1    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. model archi
class PositionalEmbedding(nn.Module): 
    def __init__(self):
        super().__init__()
        pe  = torch.zeros([block_size, d_model])
        pos = torch.arange(block_size).unsqueeze(1)
        i   = torch.arange(0, d_model, 2)
        div = torch.pow(10000, -i / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, T):
        return self.pe[:T, :]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(embd_dim, head_dim)
        self.key   = nn.Linear(embd_dim, head_dim)
        self.value = nn.Linear(embd_dim, head_dim)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C          = x.shape
        q                = self.query(x)
        k                = self.key(x)
        v                = self.value(x)
        attention        = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
        attention_masked = attention.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attention_score  = F.softmax(attention_masked, dim=-1) @ v
        return attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads    = nn.ModuleList([Head() for _ in range(num_heads)])
        self.lin_proj = nn.Linear(embd_dim, embd_dim)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.lin_proj(x)


class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand   = nn.Linear(embd_dim, embd_dim * 4)
        self.compress = nn.Linear(embd_dim * 4, embd_dim)

    def forward(self, x):
        return self.compress(F.gelu(self.expand(x)))


class Router(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.weights = nn.Linear(embd_dim, num_experts, bias=True)
        self.noise   = nn.Linear(embd_dim, num_experts, bias=True)

    def forward(self, x):
        B, T, C      = x.shape
        x_flat       = x.view(B * T, C)
        clean_logits = self.weights(x_flat)
        if self.training:
            epsilon      = torch.randn_like(clean_logits)
            noise_scale  = F.softplus(self.noise(x_flat))
            noisy_logits = clean_logits + epsilon * noise_scale
        else:
            noisy_logits = clean_logits
        probs = F.softmax(noisy_logits, dim=-1)
        return probs


class MoELayer(nn.Module):
    def __init__(self, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts   = num_experts
        self.top_k         = top_k
        self.experts       = nn.ModuleList([FFNN() for _ in range(num_experts)])

    def forward(self, x, probs):
        B, T, C  = x.shape
        x_flat   = x.view(-1, C)
        output   = torch.zeros(B * T, C, dtype=x.dtype, device=x.device)

        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)

        
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-20)

        for i in range(self.num_experts):
            row_indices, column_indices = (top_indices == i).nonzero(as_tuple=True)
            if row_indices.numel() == 0:
                continue
            expert_output = self.experts[i](x_flat[row_indices])
            gated_scores  = top_probs[row_indices, column_indices].unsqueeze(-1)
            expert_output = expert_output * gated_scores
            output.index_add_(0, row_indices, expert_output)

        return output.view(B, T, C), aux_loss


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention   = MultiHeadAttention()
        self.router      = Router(num_experts=num_experts)
        self.experts     = MoELayer(num_experts=num_experts, top_k=2)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x                    = self.layer_norm1(self.attention(x) + x)
        expert_probs         = self.router(x)
        moe_output           = self.experts(x, expert_probs)
        x                    = self.layer_norm2(x + moe_output)
        return x,


class MoEGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings      = nn.Embedding(vocab_size, embd_dim)
        self.positional_embeddings = PositionalEmbedding()
        self.attention_layers      = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.lm_head               = nn.Linear(embd_dim, vocab_size)

    def forward(self, x):
        B, T           = x.shape
        x              = self.token_embeddings(x) + self.positional_embeddings(T)
        for block in self.attention_layers:
            x = block(x)
        logits = self.lm_head(x)
        return logits


# 3. Dataset and DataLoaders

# 4. Training Loop

# 5. Metrics 