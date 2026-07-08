import random as rndm
import json
import os
import torch

directorio_base = os.path.dirname(os.path.abspath(__file__))
ruta_json = os.path.join(directorio_base, 'vocab.json')

with open(ruta_json, 'r', encoding='utf-8') as file:
    vocab = json.load(file)


def generar_operacion(batch):
    src_array = []
    tgt_in_array = []
    tgt_out_array = []
    for _ in range(batch):
        sumando1 = []
        sumando2 = []
        # Se hacen bucles distintos para que los sumandos queden de distinta longitud
        for _ in range(rndm.randint(1, 4)):
            sumando1.append(rndm.randint(1, 9))
        for _ in range(rndm.randint(1, 4)):
            sumando2.append(rndm.randint(1, 9))
        s1 = int(''.join(map(str, sumando1)))
        s2 = int(''.join(map(str, sumando2)))
        suma_total = s1 + s2
        # Convertir a una lista de enteros
        suma_total = list(str(suma_total))
        suma_total = [int(x) for x in suma_total]
        # Mapear e invertir el orden
        sumando1 = [4 + x for x in sumando1][::-1]
        sumando2 = [4 + x for x in sumando2][::-1]
        suma_total = [4 + x for x in suma_total][::-1]
        # Agregar el src del batch a la lista
        src_array.append(sumando1 + [3] + sumando2) # De momento solo nos enfocamos en sumar
        # Agregar el tgt_in del batch a la lista
        suma_total.insert(0, 1) # Agregar BOS al inicio
        tgt_in_array.append(suma_total.copy())
        # Agregar el tgt_out del batch a la lista
        del suma_total[0] # Eliminar el BOS
        suma_total.append(2) # Agregar el EOS al final
        tgt_out_array.append(suma_total)
    
    # Rellenar con PAD
    max_longitud = max(len(batch) for batch in src_array) # Rellenar src_array
    for batch in src_array:
        cant = max_longitud - len(batch)
        batch.extend([0] * cant)

    max_longitud = max(len(batch) for batch in tgt_in_array) # Rellenar tgt_in_array
    for batch in tgt_in_array:
        cant = max_longitud - len(batch)
        batch.extend([0] * cant)

    max_longitud = max(len(batch) for batch in tgt_out_array) # Rellenar tgt_out_array
    for batch in tgt_out_array:
        cant = max_longitud - len(batch)
        batch.extend([0] * cant)
        
    return torch.tensor(src_array), torch.tensor(tgt_in_array), torch.tensor(tgt_out_array)