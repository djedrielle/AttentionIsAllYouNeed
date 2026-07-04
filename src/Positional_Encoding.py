import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)                # (max_len, d_model), lo llenaremos
        pos = torch.arange(0, max_len).unsqueeze(1)       # (max_len, 1): 0,1,2,...
        # frecuencias: 1 / 10000^(2i/d_model) para cada par de dims i
        div_term = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))   # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)                              # (1, max_len, d_model), para sumar por batch
        self.register_buffer('pe', pe)                    # tensor FIJO, no entrenable (ver nota)

    def forward(self, x):                                 # x: (batch, n, d_model)
        x = x + self.pe[:, :x.size(1)] # Sumarle a cada embedding su vector posicion respectivo
        return x
