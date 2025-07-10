import torch
import torch.nn as nn

class SSDLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sum((y_pred - y_true)**2, dim=-1).mean()

class CombinedSSDMSELoss(nn.Module):
    def forward(self, y_pred, y_true):
        mse_term = torch.mean((y_true - y_pred)**2, dim=-1) * 500
        ssd_term = torch.sum((y_true - y_pred)**2, dim=-1)
        return (mse_term + ssd_term).mean()

class CombinedSSDMADLoss(nn.Module):
    def forward(self, y_pred, y_true):
        mad_term = torch.amax((y_true - y_pred)**2, dim=-1) * 50
        ssd_term = torch.sum((y_true - y_pred)**2, dim=-1)
        return (mad_term + ssd_term).mean()

class SADLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.sum(torch.abs(y_pred - y_true), dim=-1).mean()

class MADLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.amax((y_pred - y_true)**2, dim=-1).mean()