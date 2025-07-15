import torch
import torch.nn as nn
from util import default

class TransformerEncoder(nn.Module):
    def __init__(self, dim, head_size, num_heads, ff_dim, dropout=0):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim * num_heads, num_heads, dropout=dropout, batch_first=True)
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
        # x : (batch_size, seq_len, dim)
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x1 = x + self.dropout1(attn_output)
        
        norm_x1 = self.norm2(x1)
        norm_x1_perm = norm_x1.permute(0, 2, 1)
        ff_output = self.ff(norm_x1_perm).permute(0, 2, 1)
        
        return x1 + ff_output
    
class AdaptiveStepScheduler(nn.Module):
    def __init__(self, num_layers, dim, head_size, num_heads, ff_dim, dropout=0):
        super(AdaptiveStepScheduler, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoder(dim, head_size, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.out_mlp = nn.Linear(dim, 1)  # Output layer for step prediction
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        for layer in self.layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        return self.activation(self.out_mlp(x))