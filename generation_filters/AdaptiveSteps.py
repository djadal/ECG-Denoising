import torch
import torch.nn as nn
from .utils import default
from einops import rearrange, einsum

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) l -> b h c l', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum(q, k, 'b h d i, b h d j -> b h i j')
        attn = sim.softmax(dim=-1)
        out = einsum(attn, v, 'b h i j, b h d j -> b h i d')

        out = rearrange(out, 'b h l d -> b (h d) l')
        
        return self.to_out(out)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads=num_heads, dim_head=head_size)
        self.dropout1 = nn.Dropout(dropout)
        
        ff_dim = default(ff_dim, dim * 4)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Conv1d(dim, ff_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, dim, kernel_size=1)
        )
        
    def forward(self, x):
        norm_x = self.norm1(x)
        norm_x_perm = norm_x.permute(0, 2, 1)  
        attn_output = self.attn(norm_x_perm)
        attn_output = attn_output.permute(0, 2, 1)
        x1 = x + self.dropout1(attn_output)
        
        norm_x1 = self.norm2(x1)
        norm_x1_perm = norm_x1.permute(0, 2, 1)
        ff_output = self.ff(norm_x1_perm).permute(0, 2, 1)
        
        return x1 + ff_output
    
class AdaptiveStepScheduler(nn.Module):
    def __init__(self, num_layers, dim, head_size, num_heads, ff_dim, dropout=0):
        super(AdaptiveStepScheduler, self).__init__()
        self.in_conv = nn.Conv1d(1, dim, kernel_size=1)
        self.layers = nn.ModuleList([
            TransformerEncoder(dim, head_size, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.out_mlp = nn.Linear(dim * 512, 1)  # Output layer for step prediction
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.in_conv(x)
        x = x.permute(0, 2, 1)
        
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1)
        return self.activation(self.out_mlp(x))