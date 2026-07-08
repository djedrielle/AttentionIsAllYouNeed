import random as rndm
import json
import os
import torch

directorio_base = os.path.dirname(os.path.abspath(__file__))
ruta_json = os.path.join(directorio_base, 'vocab.json')

with open(ruta_json, 'r', encoding='utf-8') as file:
    vocab = json.load(file)

def generar_datos(batch, longitud, tarea):
    array = []
    for _ in range(batch):
        sec = []
        for _ in range(longitud):
            indice = rndm.randint(3, len(vocab)-1)
            sec.append(indice)
        array.append(sec)
    src = torch.tensor(array)
    
    if tarea == "copiar":
        for batch in array:
            batch.insert(0, 1)
    elif tarea == "invertir":
        for batch in array:
            batch.reverse()
            batch.insert(0, 1)
    elif tarea == "ordenar":
        for batch in array:
            batch.sort()
            batch.insert(0, 1)
    tgt_in = torch.tensor(array)
    # Eliminar BOS al inicio y agregar EOS al final del array
    for batch in array:
        del batch[0]
        batch.append(2)
    tgt_out = torch.tensor(array)

    return src, tgt_in, tgt_out
