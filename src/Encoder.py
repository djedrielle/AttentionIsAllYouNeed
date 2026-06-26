import torch.nn as nn
from MultiHead_Attention import MultiHeadAttention
from Feed_Forward import PositionwiseFeedForward
from Add_and_Norm import AddAndNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)
        
    def forward(self, x, mask=None):
        x = self.add_norm1(x, self.attn(x, x, x, mask)[0]) # Pasamos Q, K, V al multihead attention. s_MHA[0] como parametro para omitir los pesos de la salida de MHA
        x = self.add_norm2(x, self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        
    def forward(self, x, mask=None):
        for capa in self.layers:
            x = capa(x, mask)
        return x