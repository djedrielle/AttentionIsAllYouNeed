import torch.nn as nn
from MultiHead_Attention import MultiHeadAttention
from Feed_Forward import PositionwiseFeedForward
from Add_and_Norm import AddAndNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, h)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.AddAndNorm_MHA = AddAndNorm(d_model, dropout)
        self.AddAndNorm_FF = AddAndNorm(d_model, dropout)
        
    def forward(self, x, mask=None):
        s_MHA = self.attn(x, x, x, mask) # Pasamos Q, K, V al multihead attention
        s_MHA_AN = self.AddAndNorm_MHA(x, s_MHA[0]) # s_MHA[0] como parametro para omitir los pesos de la salida de MHA
        s_MHA_AN_FF = self.ff(s_MHA_AN) # Pasamos por el Feed Forward
        s_MHA_AN_FF_AN = self.AddAndNorm_FF(s_MHA_AN, s_MHA_AN_FF) # Se envuelve la salida del FF en su AddAndNorm correspondiente
        return s_MHA_AN_FF_AN

class Encoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        
    def forward(self, x, mask=None):
        for capa in self.layers:
            x = capa(x, mask)
        return x