from generador_datos import generar_datos
from generador_mascara import generar_mascara
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from Transformer import Transformer
import torch
import torch.nn as nn

src_vocab = tgt_vocab = 13
N = 6
d_model = 512
h = 8
d_ff = 2048
dropout = 0.1

# Instanciar el modelo y el optimizador
model = Transformer(src_vocab, tgt_vocab, N, d_model, h, d_ff, dropout)
model.train()
optimizador = torch.optim.Adam(model.parameters(), lr=1e-3)
criterio = nn.CrossEntropyLoss(ignore_index = 0)

for epoca in range(2000):
    # Setup Parametros
    src, tgt_in, tgt_out = generar_datos(32, 10)
    tgt_mask = generar_mascara(tgt_in.size(1))
    # forward
    logits = model(src, tgt_in, None, tgt_mask)
    # Calcular el loss
    logits = torch.reshape(logits, (-1,13))
    tgt_out = torch.reshape(tgt_out, (-1,))
    loss = criterio(logits, tgt_out)
    # Limpiar pesos, calcular el grad, aplicar correcciones
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()
    if epoca % 100 == 0:
        predicciones = logits.argmax(dim=-1)
        aciertos = (predicciones == tgt_out).float()
        accuracy = aciertos.mean().item()         
        print(f'Paso {epoca:>4} | loss -> {loss.item():.4f} | accuracy -> {accuracy:.2%}')

