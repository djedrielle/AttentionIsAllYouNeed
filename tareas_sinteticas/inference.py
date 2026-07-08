import torch
from generador_datos import generar_datos
from generador_mascara import generar_mascara
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from Transformer import Transformer
import torch

# Setup
src_vocab = tgt_vocab = 13
N = 2
d_model = 64
h = 4
d_ff = 256
dropout = 0.1
tarea = input("¿Qué tarea quieres que el modelo haga? (copiar/invertir/ordenar) -> ")

model = Transformer(src_vocab, tgt_vocab, N, d_model, h, d_ff, dropout)
model.load_state_dict(torch.load(f'{tarea}.pt'))
model.eval()

# Inferencia
src, _, _ = generar_datos(1, 10, tarea) # Generar la entrada al modelo
respuesta = torch.full((src.size(0), 1), 1, dtype=torch.long) # Inicializar la respuesta con BOS
for _ in range(src.size(1) + 1):
    tgt_mask = generar_mascara(respuesta.size(1))
    logits = model(src, respuesta, None, tgt_mask)
    respuesta = torch.cat([respuesta, logits[:,-1].argmax(dim=-1, keepdim=True)], dim=1)

print("entrada :", src[0].tolist())
print("Respuesta del modelo   :", respuesta[0, 1:].tolist())