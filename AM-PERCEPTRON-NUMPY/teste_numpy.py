import numpy as np

#Criar os vetores
#Array do numpy vai fazer os calculos transformando objeto python em c (calculo em linguagem de baixo nivel) e devolver o resultado para o python

x = np.array([8.0, 7.0]) #Entradas(x1, x2)
w = np.array([0.8, 0.3]) #Pesos (w1, w2)
bias = -7.0

#Calculo de produto escalar
z_dot = np.dot(x, w) + bias

#Outra forma de fazer com operador @
z_operador = (x @ w) + bias

print("Z com np.dot", z_dot)
print("Z com operador @", z_operador)