import torch

def classification(x, y, device, dtype):
    return x.to(device = device, dtype = dtype), y.to(device = device)

def regression(x, y, device, dtype):
    return x.to(device = device, dtype = dtype), y.to(device = device, dtype = dtype)
