import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Llama2AttentionDecode(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, x, past_key_value):
        """
        x: Current token tensor (bsz, 1, hidden_size)
        past_key_value: Tuple of past K and V tensors (past_k, past_v)
        """
        bsz, seq_len, _ = x.size() # seq_len is typically 1 in decode phase
        
        # 1. Q, K, V Projections for the NEW token
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. KV Cache Concatenation
        past_k, past_v = past_key_value
        k = torch.cat([past_k, k], dim=2) # Append along sequence dimension
        v = torch.cat([past_v, v], dim=2)
        
        # 3. Scaled Dot-Product Attention (No causal mask needed for seq_len=1)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 4. Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_output)

# Initialize Module
MODULE = Llama2AttentionDecode().eval()

# Dummy Inputs:
# 1. Current Token: [Batch Size: 1, Sequence Length: 1, Hidden Size: 4096]
# 2. KV Cache from past 127 tokens: Tuple([1, 32, 127, 128], [1, 32, 127, 128])
INPUTS = [
    torch.randn(1, 1, 4096),
    (torch.randn(1, 32, 127, 128), torch.randn(1, 32, 127, 128))
]