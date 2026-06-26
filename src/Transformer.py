import torch.nn as nn
from Embedding import Embeddings
from Positional_Encoding import PositionalEncoding
from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N, d_model, h, d_ff, dropout):
        super().__init__()
        self.src_embed = Embeddings(src_vocab, d_model)
        self.tgt_embed = Embeddings(tgt_vocab, d_model)
        self.PE = PositionalEncoding(d_model)
        self.Encoder = Encoder(N, d_model, h, d_ff, dropout)
        self.Decoder = Decoder(N, d_model, h, d_ff, dropout)
        self.Linear = nn.Linear(d_model, tgt_vocab) # Hay que convertir la salida del Decoder al tamanno del vocabulario de salida
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.Encoder(self.PE(self.src_embed(src)), src_mask)
        dec = self.Decoder(self.PE(self.tgt_embed(tgt)), memory, tgt_mask, src_mask)
        return self.Linear(dec)
