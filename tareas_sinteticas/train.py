from generador_datos import generar_datos
from generador_mascara import generar_mascara
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from Transformer import Transformer
import torch
import torch.nn as nn

src_vocab = tgt_vocab = 13
N = 2
d_model = 64
h = 4
d_ff = 256
dropout = 0.1

# Instanciar el modelo y el optimizador
model = Transformer(src_vocab, tgt_vocab, N, d_model, h, d_ff, dropout)
model.train()
optimizador = torch.optim.Adam(model.parameters(), lr=1e-3)
criterio = nn.CrossEntropyLoss(ignore_index = 0)

@torch.no_grad()                          # no calculamos gradientes: solo miramos
def generar_copia(model, src, bos=1):
    model.eval()                          # apaga el dropout (predicciones estables)
    # 1. El encoder procesa src UNA sola vez -> memory
    memory = model.Encoder(model.PE(model.src_embed(src)), None)
    # 2. Arrancamos la respuesta con solo BOS
    ys = torch.full((src.size(0), 1), bos, dtype=torch.long)
    # 3. Generamos casilla por casilla, L+1 pasos (los símbolos + EOS)
    for _ in range(src.size(1) + 1):
        tgt_mask = generar_mascara(ys.size(1))
        dec = model.Decoder(model.PE(model.tgt_embed(ys)), memory, tgt_mask, None)
        logits = model.Linear(dec[:, -1])            # solo la ÚLTIMA posición
        siguiente = logits.argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, siguiente], dim=1)       # la anexamos y repetimos
    model.train()                         # volvemos a modo entrenamiento
    return ys

for epoca in range(3000):
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
        muestra_src, _, _ = generar_datos(1, 10)      # un ejemplo nuevo
        generado = generar_copia(model, muestra_src)
        print(f'  src      -> {muestra_src[0].tolist()}')
        print(f'  generado -> {generado[0, 1:].tolist()}')   # [1:] quita el BOS
