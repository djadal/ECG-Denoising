import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class HuberFreqLoss(nn.Module):
    def __init__(self, delta=0.05, sample_rate=360):
        super(HuberFreqLoss, self).__init__()
        self.delta = delta
        self.sample_rate = sample_rate
        
    def forward(self, y_pred, y_true):
        sig_len = y_true.shape[-1]

        window = torch.hann_window(sig_len, device=y_true.device)
        windowed_true = y_true * window
        windowed_pred = y_pred * window

        fft_true = torch.fft.rfft(windowed_true, dim=-1)
        fft_pred = torch.fft.rfft(windowed_pred, dim=-1)

        power_true = torch.abs(fft_true)**2
        power_pred = torch.abs(fft_pred)**2

        power_true = power_true / self.sample_rate
        power_pred = power_pred / self.sample_rate
        
        flat_true = power_true.reshape(power_true.shape[0], -1)
        flat_pred = power_pred.reshape(power_pred.shape[0], -1)

        mean_true = torch.mean(flat_true, dim=1, keepdim=True)
        mean_pred = torch.mean(flat_pred, dim=1, keepdim=True)
        std_true = torch.std(flat_true, dim=1, keepdim=True)
        std_pred = torch.std(flat_pred, dim=1, keepdim=True)
        
        corr = torch.mean((flat_true - mean_true) * (flat_pred - mean_pred), dim=1) / (std_true.squeeze(-1) * std_pred.squeeze(-1) + 1e-8)
        similarity = torch.mean(corr)

        frequency_weights = torch.exp(1 - torch.abs(similarity))

        diff = y_true - y_pred
        squared_loss = torch.square(diff)
        linear_loss = self.delta * (torch.abs(diff) - 0.5 * self.delta)
        
        weighted_loss = frequency_weights * torch.where(torch.abs(diff) <= self.delta, squared_loss, linear_loss)

        cos_loss = 1 - F.cosine_similarity(y_pred, y_true, dim=-1)
        
        return weighted_loss.mean() + cos_loss.mean()