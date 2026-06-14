import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE']  = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ========================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ========================================================================
num_layers      = 4
num_experts     = 4
num_heads       = 8
batch_size      = 16
vocab_size      = 50257
d_model         = 512
embd_dim        = 512
block_size      = 256
head_dim        = d_model // num_heads
learning_rate   = 3e-4
epochs          = 3
aux_loss_coef   = 1e-2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================================================================
# 2. MODEL ARCHITECTURE (CUSTOM MOE GPT)
# ========================================================================
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

    def forward(self):
        return self.pe[:block_size, :]


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(embd_dim, head_dim)
        self.key   = nn.Linear(embd_dim, head_dim)
        self.value = nn.Linear(embd_dim, head_dim)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
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
        B, T, C  = x.shape
        x_flat   = x.view(B * T, C)
        clean_logits = self.weights(x_flat)
        if self.training:
            epsilon     = torch.randn_like(clean_logits)
            noise_scale = F.softplus(self.noise(x_flat))
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
        # expert specialization tracking
        self.track_experts = False
        self.expert_counts = torch.zeros(num_experts)

    def forward(self, x, probs):
        B, T, C  = x.shape
        x_flat   = x.view(-1, C)
        output   = torch.zeros(B * T, C, dtype=x.dtype, device=x.device)

        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)

        # auxiliary loss
        token_fractions = probs.mean(dim=0)
        expert_usage    = torch.bincount(top_indices[:, 0], minlength=self.num_experts).float()
        expert_usage   /= expert_usage.sum()
        aux_loss        = self.num_experts * torch.dot(token_fractions, expert_usage)

        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-20)

        for i in range(self.num_experts):
            row_indices, column_indices = (top_indices == i).nonzero(as_tuple=True)
            if row_indices.numel() == 0:
                continue
            expert_output = self.experts[i](x_flat[row_indices])
            gated_scores  = top_probs[row_indices, column_indices].unsqueeze(-1)
            expert_output = expert_output * gated_scores
            output.index_add_(0, row_indices, expert_output)

            # track which tokens go to which expert
            if self.track_experts:
                self.expert_counts[i] += row_indices.numel()

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
        x            = self.layer_norm1(self.attention(x) + x)
        expert_probs = self.router(x)
        moe_output, aux_loss = self.experts(x, expert_probs)
        x            = self.layer_norm2(x + moe_output)
        return x, aux_loss


class MoEGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings      = nn.Embedding(vocab_size, embd_dim)
        self.positional_embeddings = PositionalEmbedding()
        self.attention_layers      = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.lm_head               = nn.Linear(embd_dim, vocab_size)

    def forward(self, x):
        x             = self.token_embeddings(x) + self.positional_embeddings()
        total_aux_loss = 0.0
        for block in self.attention_layers:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        logits = self.lm_head(x)
        return logits, total_aux_loss


# ========================================================================
# 3. EXPERT SPECIALIZATION ANALYSIS
# ========================================================================
def analyze_expert_specialization(model, tokenizer, device):
    test_cases = {
        'numbers'    : "1 2 3 4 5 6 7 8 9 10 100 1000",
        'verbs'      : "running jumping eating sleeping walking talking",
        'nouns'      : "cat dog house tree book chair table",
        'adjectives' : "happy sad big small fast slow red blue",
        'story'      : "Once upon a time there was a little girl who loved to read",
        'dialogue'   : "She said hello and he replied how are you today",
    }

    model.eval()

    # enable tracking on all MoE layers
    for block in model.attention_layers:
        block.experts.track_experts = True

    expert_activation = {}

    for category, text in test_cases.items():
        # reset counts
        for block in model.attention_layers:
            block.experts.expert_counts = torch.zeros(num_experts)

        inputs    = tokenizer(text, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            model(input_ids)

        # sum counts across all layers
        total_counts = torch.zeros(num_experts)
        for block in model.attention_layers:
            total_counts += block.experts.expert_counts

        expert_activation[category] = total_counts

    # disable tracking
    for block in model.attention_layers:
        block.experts.track_experts = False

    # print results
    print("\n" + "=" * 65)
    print(" EXPERT SPECIALIZATION ANALYSIS")
    print("=" * 65)
    header = f"{'Category':<15}"
    for i in range(num_experts):
        header += f"  Expert{i}"
    print(header)
    print("-" * 65)

    for category, counts in expert_activation.items():
        total = counts.sum()
        pct   = counts / total * 100 if total > 0 else counts
        row   = f"{category:<15}"
        for p in pct:
            row += f"  {p:6.1f}%"
        print(row)

    print("=" * 65)


# ========================================================================
# 4. DATA PREPARATION
# ========================================================================
print("Loading tokenizer and dataset...")
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset    = load_dataset("roneneldan/TinyStories")
train_data = dataset['train'].select(range(50000))
val_data   = dataset['validation'].select(range(5000))

def tokenize_func(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=block_size + 1,
        padding='max_length',
        return_tensors="pt"
    )

print("Tokenizing splits...")
tokenized_train = train_data.map(tokenize_func, batched=True, remove_columns=['text'], num_proc=1)
tokenized_val   = val_data.map(tokenize_func,   batched=True, remove_columns=['text'], num_proc=1)

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    return x, y

train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(tokenized_val,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model     = MoEGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ========================================================================
# 5. TRAINING LOOP
# ========================================================================
print("Beginning training execution...")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    total_train_aux  = 0.0

    for step, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits, aux_loss = model(x_batch)

        B, T, C = logits.shape
        lm_loss = F.cross_entropy(logits.view(B * T, C), y_batch.view(B * T))
        loss    = lm_loss + aux_loss_coef * aux_loss

        loss.backward()
        optimizer.step()

        total_train_loss += lm_loss.item()
        total_train_aux  += aux_loss.item()

        if step % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | "
                  f"LM Loss: {lm_loss.item():.4f} | Aux Loss: {aux_loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_aux  = total_train_aux  / len(train_loader)

    # validation
    model.eval()
    total_val_loss = 0.0
    total_val_aux  = 0.0

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val   = x_val.to(device), y_val.to(device)
            val_logits, val_aux = model(x_val)
            B_v, T_v, C_v  = val_logits.shape
            val_lm_loss    = F.cross_entropy(val_logits.view(B_v * T_v, C_v), y_val.view(B_v * T_v))
            total_val_loss += val_lm_loss.item()
            total_val_aux  += val_aux.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_aux  = total_val_aux  / len(val_loader)

    print("\n" + "=" * 60)
    print(f" EPOCH {epoch+1} METRICS SUMMARY:")
    print(f" --> Avg Train LM Loss      : {avg_train_loss:.4f}")
    print(f" --> Avg Train Aux Loss     : {avg_train_aux:.4f}")
    print(f" --> Avg Val LM Loss        : {avg_val_loss:.4f}")
    print(f" --> Avg Val Aux Loss       : {avg_val_aux:.4f}")
    print("=" * 60 + "\n")

    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss'                : avg_val_loss,
    }, f"/home/maligireddy.s/jobs/checkpoints/moe_gpt_epoch_{epoch+1}.pt")

# ========================================================================
# 6. EXPERT SPECIALIZATION ANALYSIS (runs after training)
# ========================================================================
print("\nRunning expert specialization analysis...")
analyze_expert_specialization(model, tokenizer, device)

print("\nTraining complete.")