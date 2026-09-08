#Perceptron é uma uma rede neural simples, que toma decisões binárias

#Numpy: Numerical Python, biblioteca básica para computação científica
import numpy as np

class PerceptronNumPy:
    # construtor
    # Inicializar o Perceptron já com pesos e viés (bias) pré-definidos,
    # sem precisar treiná-lo do zero
    def __init__(self, pesos, bias): 
        #recebe uma lista de pesos
        self.pesos = np.array(pesos, dtype=float)
        #recebe o valor de bias (viés)
        self.bias = float(bias)

    def predict_single(self,entradas):
        #converte uma lista de entradas recebidas
        entradas_arr = np.array(entradas, dtype=float)
        #Faz o produto escalar (multiplica cada entrada pelo
        # seu respectivo peso e soma todos os resultados) + soma valor do bias
        z = np.dot(entradas_arr, self.pesos) + self.bias

        # Se a soma ponderada $z$ for maior ou igual a zero, o neurônio ativa e retorna $1$
        #Caso contrario, retorna $0$ (não ativa)
        return 1 if z >=0 else 0

    #versão vetorizada em lote do predict_single: usa os recursos do NumPypara calcular
    # as respostas de várias amostras ao mesmo tempo em uma única operação matricial
    def predict(self, entradas_lote):
        #converte a lista de listas de entradas recebidas em um array 2D do NumPy
        X = np.array(entradas_lote, dtype=float)
        #Realiza a multiplicação de matriz por vetor + broadcasting (bias)
        Z = np.dot(X, self.pesos) + self.bias
        #Aplica a função de ativação degrau em todo o vetor $Z$ simultaneamente
        return np.where(Z >= 0, 1, 0)

#Executando uma predição:
# Garante que este código só rode quando o arquivo for executado diretamente
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