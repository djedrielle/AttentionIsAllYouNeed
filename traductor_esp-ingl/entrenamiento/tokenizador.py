from itertools import islice

def construir_vocab(frases):
    # frases: lista de strings de UN idioma
    caracteres = set()
    for f in frases:
        caracteres.update(f)              
    caracteres = sorted(caracteres)       

    
    char_a_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for i, c in enumerate(caracteres, start=4):
        char_a_id[c] = i

    id_a_char = {i: c for c, i in char_a_id.items()}
    return char_a_id, id_a_char

frases_en = []
frases_es = []

with open('../spa-eng/spa.txt', 'r', encoding='utf-8') as archivo:
    primeras_lineas = islice(archivo, 1000)

    for i, linea in enumerate(primeras_lineas):
        partes = linea.strip().split('\t')
        frases_es.append(partes[1])
        frases_en.append(partes[0])

# Uno por idioma (usa idealmente solo las frases de ENTRENAMIENTO)
en_char_a_id, en_id_a_char = construir_vocab(frases_en)
es_char_a_id, es_id_a_char = construir_vocab(frases_es)

def encode(frase, idioma):
    # frase debe ser un string e idioma debe ser "es" o "en"
    ids = []
    if (idioma == "es"):
        for caracter in frase:
            ids.append(es_char_a_id.get(caracter, es_char_a_id["<UNK>"]))
    elif (idioma == "en"):
        for caracter in frase:
            ids.append(es_char_a_id.get(caracter, es_char_a_id["<UNK>"]))
    return ids

def decode(ids, idioma):
    tabla = es_id_a_char if idioma == "es" else en_id_a_char
    frase = ""
    for id in ids:
        if id == 2:      # EOS -> terminar
            break
        if id in (0, 1): # PAD / BOS -> ignorar
            continue
        frase += tabla[id]
    return frase
