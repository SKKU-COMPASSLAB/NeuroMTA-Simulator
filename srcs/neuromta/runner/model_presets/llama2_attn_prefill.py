import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Llama2AttentionPrefill(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Llama-2 uses no bias in its linear layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x):
        bsz, seq_len, _ = x.size()
        
        # 1. Q, K, V Projections and Reshaping
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # (Note: RoPE - Rotary Position Embedding is typically applied here in Llama-2)
        
        # 2. Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 3. Causal Masking (to prevent looking at future tokens)
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 4. Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

# Initialize Module
MODULE = Llama2AttentionPrefill().eval()

# Dummy Inputs: [Batch Size: 1, Sequence Length: 128, Hidden Size: 4096]
INPUTS = [torch.randn(1, 128, 4096)]