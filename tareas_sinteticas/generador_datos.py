import random as rndm
import json
import os

directorio_base = os.path.dirname(os.path.abspath(__file__))
ruta_json = os.path.join(directorio_base, 'vocab.json')

with open(ruta_json, 'r', encoding='utf-8') as file:
    vocab = json.load(file)

def generador_datos(batch, longitud):
    array = []
    for _ in range(batch):
        sec = []
        for _ in range(longitud):
            indice = rndm.randint(3, len(vocab))
            sec.append(indice)
        array.append(sec)
    print(array)
    src = array # src listo