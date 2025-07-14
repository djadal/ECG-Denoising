import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from math import log as ln
import copy


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level=noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding
  
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x
  

class ResidualBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, 80, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, 80)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv1d(80, 80, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, 80)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += identity
        return out

        
class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoding = FeatureWiseAffine(input_size, hidden_size, use_affine_level=1)
        self.input_conv = Conv1d(input_size, input_size, 3, padding=1, padding_mode='reflect')
        self.output_conv = Conv1d(input_size, hidden_size, 3, padding=1, padding_mode='reflect')
    
    def forward(self, x, noise_embed):
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        return self.output_conv(x)


class QualityAssignmentPruner:
    """
    Quality Assignment Pruning based on Algorithm 1
    Uses dual criteria: L2norm and Redundancy
    """
    def __init__(self, compression_rate=0.5):
        self.compression_rate = compression_rate
    
    def compute_l2_norm(self, filter_weights):
        """Compute L2 norm of filter weights"""
        return torch.norm(filter_weights.view(-1), p=2)
    
    def compute_cosine_similarity(self, filter1, filter2):
        """Compute cosine similarity between two filters"""
        f1_flat = filter1.view(-1)
        f2_flat = filter2.view(-1)
        return F.cosine_similarity(f1_flat.unsqueeze(0), f2_flat.unsqueeze(0))
    
    def compute_redundancy(self, filter_weights, all_filters):
        """Compute redundancy score for a filter"""
        similarities = []
        for other_filter in all_filters:
            if not torch.equal(filter_weights, other_filter):
                sim = self.compute_cosine_similarity(filter_weights, other_filter)
                similarities.append(sim.item())
        
        return np.mean(similarities) if similarities else 0.0
    
    def dual_criteria_pruning(self, layer_filters):
        """
        Apply dual-criteria quality assignment pruning
        Based on Algorithm 1 from the paper
        """
        num_filters = layer_filters.size(0)
        l2_norms = []
        redundancies = []
        
        # Step 1-3: Compute L2 norm and redundancy for each filter
        for i in range(num_filters):
            filter_i = layer_filters[i]
            l2_norm = self.compute_l2_norm(filter_i)
            redundancy = self.compute_redundancy(filter_i, layer_filters)
            
            l2_norms.append(l2_norm.item())
            redundancies.append(redundancy)
        
        # Step 5-6: Compute mean and threshold
        mean_l2_norms = np.mean(l2_norms)
        threshold = 0.1 * mean_l2_norms
        
        # Step 7-8: Sort filters
        l2_sorted_indices = np.argsort(l2_norms)[::-1]  # Descending order
        redundancy_sorted_indices = np.argsort(redundancies)  # Ascending order
        
        # Step 9: Select top (1-r) × l(i) filters from each sorted list
        num_keep = int((1 - self.compression_rate) * num_filters)
        top_l2_indices = set(l2_sorted_indices[:num_keep])
        top_redundancy_indices = set(redundancy_sorted_indices[:num_keep])
        
        # Step 10-14: Apply pruning decision
        pruning_mask = torch.ones(num_filters, dtype=torch.bool)
        
        for i in range(num_filters):
            # Prune if L2norm < threshold AND redundancy > 0 AND not in top lists
            if (l2_norms[i] < threshold and 
                redundancies[i] > 0 and 
                i in top_redundancy_indices):
                pruning_mask[i] = False  # Mark for pruning
        
        return pruning_mask
    
    def prune_conv_layer(self, conv_layer):
        """Prune a convolutional layer using quality assignment"""
        if conv_layer.weight.size(0) <= 8:  # Don't prune if too few filters
            return conv_layer
        
        with torch.no_grad():
            # Get filter weights
            filters = conv_layer.weight.data
            
            # Apply dual-criteria pruning
            keep_mask = self.dual_criteria_pruning(filters)
            
            # Create new layer with pruned filters
            num_keep = keep_mask.sum().item()
            if num_keep == 0:
                num_keep = 1  # Keep at least one filter
                keep_mask[0] = True
            
            # Create new conv layer
            new_conv = nn.Conv1d(
                conv_layer.in_channels,
                num_keep,
                conv_layer.kernel_size,
                conv_layer.stride,
                conv_layer.padding,
                conv_layer.dilation,
                conv_layer.groups,
                conv_layer.bias is not None,
                conv_layer.padding_mode
            )
            
            # Copy kept weights
            new_conv.weight.data = filters[keep_mask]
            if conv_layer.bias is not None:
                new_conv.bias.data = conv_layer.bias.data[keep_mask]
            
            return new_conv
    

class ConditionalModel(nn.Module):
    def __init__(self, feats=64):
        super(ConditionalModel, self).__init__()
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            ResidualBlock(feats),
            ResidualBlock(80),
            ResidualBlock(80),
            ResidualBlock(80),
            ResidualBlock(80),
        ])
        
        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            ResidualBlock(feats),
            ResidualBlock(80),
            ResidualBlock(80),
            ResidualBlock(80),
            ResidualBlock(80),
        ])
        
        self.embed = PositionalEncoding(feats)
        
        self.bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
        ])
        
        self.conv_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')
        
    def forward(self, x, cond, noise_scale):
        noise_embed = self.embed(noise_scale)
        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed)) 
        
        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond)+x
        
        return self.conv_out(cond)
    
    def apply_quality_assignment_pruning(self, compression_rate=0.3):
        """Apply quality assignment pruning to the model"""
        pruner = QualityAssignmentPruner(compression_rate)
        
        print(f"Applying Quality Assignment Pruning with compression rate: {compression_rate}")
        
        # Count original parameters
        original_params = sum(p.numel() for p in self.parameters())
        print(f"Original parameters: {original_params:,}")
        
        # Prune convolutional layers in residual blocks
        for stream_name, stream in [("stream_x", self.stream_x), ("stream_cond", self.stream_cond)]:
            for i, layer in enumerate(stream):
                if isinstance(layer, ResidualBlock):
                    # Prune conv layers in residual block
                    layer.conv1 = pruner.prune_conv_layer(layer.conv1)
                    layer.conv2 = pruner.prune_conv_layer(layer.conv2)
                    
                    # Update GroupNorm accordingly
                    layer.gn1 = nn.GroupNorm(min(8, layer.conv1.out_channels), layer.conv1.out_channels)
                    layer.gn2 = nn.GroupNorm(min(8, layer.conv2.out_channels), layer.conv2.out_channels)
                    
                    print(f"Pruned {stream_name}[{i}] - ResidualBlock")
        
        # Prune bridge layers
        for i, bridge in enumerate(self.bridge):
            bridge.input_conv = pruner.prune_conv_layer(bridge.input_conv)
            bridge.output_conv = pruner.prune_conv_layer(bridge.output_conv)
            print(f"Pruned bridge[{i}]")
        
        # Count pruned parameters
        pruned_params = sum(p.numel() for p in self.parameters())
        compression_ratio = (original_params - pruned_params) / original_params
        
        print(f"Pruned parameters: {pruned_params:,}")
        print(f"Parameter reduction: {compression_ratio:.2%}")
        print(f"Compression achieved: {original_params/pruned_params:.2f}x")


if __name__ == "__main__":    
    # Create and test the model
    print("=== Original Model ===")
    net = ConditionalModel(80).to('cpu')
    x = torch.randn(10,1,512).to('cpu')
    y = torch.randn(10,1,512).to('cpu')
    noise_scale = torch.randn(10,1).to('cpu')
    
    # Test original model
    z = net(x, y, noise_scale)
    print(f"Original output shape: {z.shape}")
    print(f"Original total parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    print("\n=== Applying Quality Assignment Pruning ===")
    # Apply pruning
    net.apply_quality_assignment_pruning(compression_rate=0.3)
    
    # Test pruned model
    z_pruned = net(x, y, noise_scale)
    print(f"\nPruned output shape: {z_pruned.shape}")
    print(f"Pruned total parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Verify model still works
    print(f"\nModel output difference (should be small): {torch.mean(torch.abs(z - z_pruned)).item():.6f}")
