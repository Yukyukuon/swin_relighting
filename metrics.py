import torch

def si_mse(pred, target):
    return torch.mean((pred - target) ** 2)