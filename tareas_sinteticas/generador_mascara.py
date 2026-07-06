import torch

def generar_mascara(n):
    return torch.tril(torch.ones(n,n)).unsqueeze(0).unsqueeze(0)