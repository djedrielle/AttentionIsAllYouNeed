import torch.nn as nn
import attention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, h): #d_model es la dimension de cada token, h es la cantidad de cabezas
        super().__init__()
        assert d_model % h == 0, "d_model debe ser divisible entre h"
        self.h = h
        self.d_k = d_model // h #d_k es la dimension en la que trabaja cada cabeza

        # Las 4 proyecciones aprendibles (los únicos pesos de este módulo):
        self.W_q = nn.Linear(d_model, d_model)   # vista "Query" de cada token
        self.W_k = nn.Linear(d_model, d_model)   # vista "Key"   de cada token
        self.W_v = nn.Linear(d_model, d_model)   # vista "Value" de cada token
        self.W_o = nn.Linear(d_model, d_model)   # mezcla final tras concatenar las cabezas

    def forward(self, Q, K, V, mask=None): # Entran los 3 embbedings (batch, n, d_model)
        batch = Q.size(0)

        # Se les hace una proyeccion (parte aprendible del flujo, aca el modelo ajusta pesos para aprender)
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        # Cada proyeccion es dividida en h cabezas
        Q = Q.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch, -1, self.h, self.d_k).transpose(1, 2)
 
        # Q/K/V quedo dividido para h cabezas independientes, cada cabeza ve n tokens
        # pero ahora cada token tiene su trozo de 64 dims de las 512 dims de los tokens originales 
        
        salida, pesos = attention.scaled_dot_product_attention(Q, K, V, mask) # Atencion
        salida = salida.transpose(1, 2).contiguous().view(batch, -1, self.h * self.d_k) # Unificamos las cabezas
        salida = self.W_o(salida) # Ultima proyeccion del Multi-Head

        return salida, pesos