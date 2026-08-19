import numpy as np

class PerceptronNumPy:
    # construtor
    def __init__(self, pesos, bias): 
        self.pesos = np.array(pesos, dtype=float)
        self.bias = float(bias)

    def predict_single(self,entradas):
        entradas_arr = np.array(entradas, dtype=float)
        z = np.dot(entradas_arr, self.pesos) + self.bias

        return 1 if z >=0 else 0

    def predict(self, entradas_lote):
        X = np.array(entradas_lote, dtype=float)

        Z = np.dot(X, self.pesos) + self.bias

        return np.where(Z >= 0, 1, 0)

if __name__ == "__main__":
    print("Teste de classe Perceptron com Numpy")


    # Pesos e bias de exemplo
    pesos_ex = [0.5, -1.0, 0.2] 
    bias_ex = 0.5

    modelo = PerceptronNumPy(pesos=pesos_ex, bias=bias_ex)

    #Exemplo de uso predict_single
    amostra_unica = [2.0, 0.5, 1.0]

    predicao_unica = modelo.predict_single(amostra_unica)
    print(f"Predicao para amostra unica {amostra_unica}: {predicao_unica}")

    # Lote de teste : 4 amostras

    X_teste = [
        [ 1.0, 2.0, 3.0 ],
        [ 0.0, 1.0, 0.0 ],
        [ 2.0, 0.5, 1.0 ],
        [ 0.0, 0.0, 0.0 ],
    ]

    # Gerando a resposta em lote
    predicoes = modelo.predict(X_teste)
    print("Entradas:")
    print(np.array(X_teste))
    print("\nPredicoes do lote: ", predicoes)