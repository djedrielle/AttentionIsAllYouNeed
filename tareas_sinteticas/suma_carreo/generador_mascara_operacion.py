import torch

def generar_mascara_operacion(src, tgt_in):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt_in != 0).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.tril(torch.ones(tgt_in.size(1),tgt_in.size(1))).unsqueeze(0).unsqueeze(0)
    tgt_mask = (tgt_mask & causal_mask.bool())
    return src_mask, tgt_mask