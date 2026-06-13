import torch
import torch.nn as nn
import torch.nn.functional as F

num_layers = 6
num_experts = 4
num_heads = 8
batch_size = 4
vocab_size = 10000
d_model,embd_dim = 512, 512
block_size = 256
head_dim = d_model//num_heads

class PositionalEmbedding(nn.Module): #sinosudial embeds
    def __init__(self):
        super().__init__()
        pe = torch.zeros([block_size,d_model])
        pos = torch.arange(block_size).unsqueeze(1)
        i = torch.arange(0,d_model,2)
        div = torch.pow(10000, -i / d_model)
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe',pe)

    def forward(self):
        return self.pe[:block_size,:]

class Head(nn.Module): #Individual head
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(embd_dim,head_dim) #B,embd_dim,head_dim
        self.key   = nn.Linear(embd_dim,head_dim) #B,embd_dim,head_dim
        self.value = nn.Linear(embd_dim,head_dim) #B,embd_dim,head_dim
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        #x : B,block_size,embd_dim
        B,T,C = x.shape
        q = self.query(x)  #B,block_size,head_dim
        k = self.key(x)
        v = self.value(x)

        #apply mask for attention
        attention = q @ k.transpose(-2,-1) * (head_dim**-0.5)
        attention_masked = attention.masked_fill(self.tril[:block_size,:block_size]==0,float('-inf'))
        attention_score = F.softmax(attention_masked,dim=-1) @ v
        return attention_score

class MutliHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(num_heads)])
        self.lin_proj = nn.Linear(embd_dim,embd_dim)

    def forward(self,x):
        x = torch.concat([h(x) for h in self.heads],dim=-1)
        return self.lin_proj(x)

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(embd_dim,embd_dim*4)
        self.compress = nn.Linear(embd_dim*4,embd_dim)
    def forward(self,x):
        return self.compress(F.gelu(self.expand(x)))
    
class Router(nn.Module):
    def __init__(self,num_experts=4):
        super().__init__()
        self.weights = nn.Linear(embd_dim,num_experts,bias=True)
        self.noise = nn.Linear(embd_dim,num_experts,bias=True)
    def forward(self,x):
        B,T,C = x.shape
        x = x.view(B*T,C)
        clean_logits = self.weights(x)

        if self.training: #add noise
            epsilon = torch.randn_like()
            noise_scale = F.softplus(self.noise(x))
            noisy_logits = clean_logits + epsilon * noise_scale
        else:
            noisy_logits = clean_logits
        probs = F.softmax(noisy_logits,dim=-1)
        return probs #gives probabalities of experts
    
class MoELayer(nn.Module):
    def __init__(self,num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([FFNN() for _ in range(num_experts)])
    def forward(self,x,probs):
        #pick top-k and call FFNN for that particular experts
        pass

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MutliHeadAttention()
        self.router = Router()
        self.experts = MoELayer()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self,x):
        x = self.layer_norm1(self.attention(x)+x)
        expert_probs = self.router(x)
        moe_output = self.experts(x,expert_probs)
        x = self.layer_norm2(x + moe_output)
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeddings   = nn.Embedding(vocab_size,embd_dim)
        self.positional_embeddings = PositionalEmbedding()
        self.attention_layers   = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.lm_head            = nn.Linear(embd_dim, vocab_size)
    
    def forward(self,x):
        x = self.token_embeddings(x) + self.positional_embeddings()
        for block in self.attention_layers:
            x = block(x)
        return self.lm_head(x)

        
model = GPT()
x = torch.randint(low=0,high=vocab_size-10,size=(batch_size,block_size))
output = model(x)
print(output.shape)
