import numpy as np

def softmax(x):
    """
    Función softmax que normaliza un vector de valores para que sumen 1.
    La entrada x es una matriz de afinidad, y la salida es una matriz de pesos de atención.
    """
    max_fila = np.max(x, axis=-1, keepdims=True) 
    exp = np.exp(x - max_fila) 
    suma_filas = np.sum(exp, axis=-1, keepdims=True)
    return exp / suma_filas


T = 4 # Tiempo o número de palabras que se le pasa
dk = 8 # Dimensionalidad de la Q y K

# Query
Q = np.random.randint(0, 10, (T, dk))
# Key
K = np.random.randint(0, 10, (T, dk))
# Value
V = np.random.randint(0, 10, (T, dk))

# Se traspone la matriz K para poder hacer el producto punto con Q
Kt = K.T

# Se multiplica la matriz Query con la Key(traspuesta para poder realizar correctamente el producto de matrices al ser las dos iguales no era posible)
# Se obtiene la matriz de afinidad que nos indica la relación entre cada palabra con las demás
afinidad = np.dot(Q, Kt) 

# Al realmente la multiplicacion ser un sumatorio de varianzas 0 la varianza del resultado es dk, por lo que se divide entre la raiz cuadrada de la dimensionalidad de la Q y K para normalizar el resultado
normalizado = afinidad / np.sqrt(dk)

# Se aplica la función softmax para obtener los pesos de atención
softmax_result = softmax(normalizado)

# Se multiplica la matriz de pesos de atención con la matriz Value para obtener la salida final
salida = np.dot(softmax_result, V)

#Mostramos la matriz de salida final
print("Matriz de salida final:")
print(salida)

