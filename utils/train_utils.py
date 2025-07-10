import torch
import numpy as np

class LRScheduler:
    def __init__(self, config=None):
        self.config = config or {}
        self.scheduler = None
        
    def initialize(self, optimizer):
        scheduler_type = self.config.get("type", "")
        
        if scheduler_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.get("mode", "min"),
                factor=self.config.get("factor", 0.5),
                patience=self.config.get("patience", 2),
                min_lr=self.config.get("min_lr", 1e-10),
                verbose=self.config.get("verbose", True),
            )
        elif scheduler_type == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.config.get("step_size", 150),
                gamma=self.config.get("gamma", 0.1),
                verbose=self.config.get("verbose", True)
            )
        
    def step(self, metrics=None):
        if self.scheduler is None:
            return
            
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
            
    def get_last_lr(self):
        if hasattr(self.scheduler, "get_last_lr"):
            return self.scheduler.get_last_lr()
        elif hasattr(self.scheduler, "_last_lr"):
            return self.scheduler._last_lr
        return None


class EarlyStopping:
    def __init__(self, config=None):
        self.config = config or {}
        self.min_delta = self.config.get("min_delta", 0.0)
        self.patience = self.config.get("patience", 10)
        self.verbose = self.config.get("verbose", False)
        self.mode = self.config.get("mode", "min")
        self.best = None
        self.counter = 0
        self.should_stop = False
        
        if self.mode == "min":
            self.monitor_op = lambda a, b: a < b - self.min_delta
            self.best = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b + self.min_delta
            self.best = -float('inf')
            
    def step(self, metric):
        if self.monitor_op(metric, self.best):
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.should_stop