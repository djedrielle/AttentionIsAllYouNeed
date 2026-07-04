import torch.nn as nn
from MultiHead_Attention import MultiHeadAttention
from Feed_Forward import PositionwiseFeedForward
from Add_and_Norm import AddAndNorm
# Lo mismo que el Encoder solo que agregando el MaskedMultiHeadAttention, el attn del Encoder se cambia acá por cross_attn
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_norm1 = AddAndNorm(d_model, dropout)
        self.add_norm2 = AddAndNorm(d_model, dropout)
        self.add_norm3 = AddAndNorm(d_model, dropout)
        
    def forward(self, x, memory, tgt_mask=None, src_mask=None): # memory permite traer K y V del encoder
        x = self.add_norm1(x, self.masked_attn(x, x, x, tgt_mask)[0]) # tgt_mask es True, ya que es la que es la que entra a masked attention
        x = self.add_norm2(x, self.cross_attn(x, memory, memory, src_mask)[0])
        x = self.add_norm3(x, self.ff(x))
        return x

class Decoder(nn.Module):
    def __init__(self, N, d_model, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        
    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        for capa in self.layers:
            x = capa(x, memory, tgt_mask, src_mask)
        return x