import torch.nn as nn
from MultiHead_Attention import MultiHeadAttention
from Feed_Forward import PositionwiseFeedForward
from Add_and_Norm import AddAndNorm
# Lo mismo que el Encoder solo que agregando el MaskedMultiHeadAttention, el attn del Encoder se cambia acá por CrossAttn
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.MaskedAttn = MultiHeadAttention(d_model, h)
        self.CrossAttn = MultiHeadAttention(d_model, h)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.AddAndNorm_MaskedMHA = AddAndNorm(d_model, dropout)
        self.AddAndNorm_CrossMHA = AddAndNorm(d_model, dropout)
        self.AddAndNorm_FF = AddAndNorm(d_model, dropout)
        
    def forward(self, x, memory, MaskedAttnMask=None, crossAttnMask=None): # memory permite traer K y V del encoder
        s_maskedMHA = self.MaskedAttn(x, x, x, MaskedAttnMask) # Pasamos Q, K, V al masked multihead attention
        s_maskedMHA_AN = self.AddAndNorm_MaskedMHA(x, s_maskedMHA[0])
        
        s_maskedMHA_AN_Cross = self.CrossAttn(s_maskedMHA_AN, memory, memory, crossAttnMask) # Q viene del decoder, K y V viene del encoder
        s_maskedMHA_AN_Cross_AN = self.AddAndNorm_CrossMHA(s_maskedMHA_AN, s_maskedMHA_AN_Cross[0])
        
        s_maskedMHA_AN_Cross_AN_FF = self.ff(s_maskedMHA_AN_Cross_AN) # Pasamos por el Feed Forward
        s_maskedMHA_AN_Cross_AN_FF_AN = self.AddAndNorm_FF(s_maskedMHA_AN_Cross_AN, s_maskedMHA_AN_Cross_AN_FF) # Se envuelve la salida del FF en su AddAndNorm correspondiente
        
        return s_maskedMHA_AN_Cross_AN_FF_AN

class Decoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        
    def forward(self, x, memory, MaskedAttnMask=None, crossAttnMask=None):
        for capa in self.layers:
            x = capa(x, memory, MaskedAttnMask, crossAttnMask)
        return x