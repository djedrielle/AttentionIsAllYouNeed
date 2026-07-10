from generador_operacion import generar_operacion
from generador_mascara_operacion import generar_mascara_operacion
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from Transformer import Transformer
import torch
import torch.nn as nn

src_vocab = tgt_vocab = 14
N = 2
d_model = 64
h = 4
d_ff = 256
dropout = 0.1

# tarea = input("¿Qué tarea quieres que el modelo aprenda? (copiar/invertir/ordenar) -> ")
# if tarea != "copiar" and tarea != "invertir" and tarea != "ordenar":
    # print("Debe ingresar una opción valida... (copiar/invertir/ordenar), se le enseñará a copiar al modelo... este es el default.")
    # tarea = "copiar"

# Instanciar el modelo y el optimizador
model = Transformer(src_vocab, tgt_vocab, N, d_model, h, d_ff, dropout)
model.train()
optimizador = torch.optim.Adam(model.parameters(), lr=1e-3)
criterio = nn.CrossEntropyLoss(ignore_index = 0)

for epoca in range(2000):
    # Setup Parametros
    src, tgt_in, tgt_out = generar_operacion(32)
    src_mask, tgt_mask = generar_mascara_operacion(src, tgt_in)
    # forward
    logits = model(src, tgt_in, src_mask, tgt_mask)
    # Calcular el loss
    logits = torch.reshape(logits, (-1, src_vocab))
    tgt_out = torch.reshape(tgt_out, (-1,))
    loss = criterio(logits, tgt_out)
    # Limpiar pesos, calcular el grad, aplicar correcciones
    optimizador.zero_grad()
    loss.backward()
    optimizador.step()
    if epoca % 100 == 0:
        predicciones = logits.argmax(dim=-1)          # (N,) símbolo predicho por casilla
        reales = (tgt_out != 0)                        # True donde NO es PAD
        aciertos = (predicciones == tgt_out) & reales  # acierto solo si coincide Y no es PAD
        accuracy = aciertos.sum().float() / reales.sum()   # aciertos / total de casillas reales
        print(f'Paso {epoca:>5} | loss -> {loss.item():.4f} | accuracy -> {accuracy.item():.2%}')

torch.save(model.state_dict(), 'suma.pt')