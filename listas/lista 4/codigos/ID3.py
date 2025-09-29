import pandas as pd
import numpy as np

# ====================================
# Função para calcular a entropia
# ====================================
def entropia(dados):
    """Calcula a entropia de uma série de classes."""
    classes, freq = np.unique(dados, return_counts=True)
    proporcoes = freq / len(dados)
    return -np.sum(proporcoes * np.log2(proporcoes))

# ====================================
# Função para calcular o ganho de informação
# ====================================
def info_gain(coluna, y_alvo):
    """Calcula o ganho de informação de um atributo específico."""
    entropia_inicial = entropia(y_alvo)
    valores, contagem_valores = np.unique(coluna, return_counts=True)

    entropia_ponderada = 0
    for valor, qtd in zip(valores, contagem_valores):
        subconjunto = y_alvo[coluna == valor]
        entropia_ponderada += (qtd / len(coluna)) * entropia(subconjunto)
    
    return entropia_inicial - entropia_ponderada

# ====================================
# Implementação do algoritmo ID3
# ====================================
def construir_arvore(dados, rotulos, atributos_disponiveis):
    """
    Constrói uma árvore de decisão usando o algoritmo ID3.
    """
    # Caso base 1: todas as classes são iguais
    if len(np.unique(rotulos)) == 1:
        return np.unique(rotulos)[0]
    
    # Caso base 2: sem atributos restantes
    if len(atributos_disponiveis) == 0:
        # Retorna a classe mais frequente
        return np.bincount(rotulos).argmax()
    
    # Escolhe o atributo com maior ganho de informação
    ganhos = [info_gain(dados[atributo], rotulos) for atributo in atributos_disponiveis]
    melhor_idx = np.argmax(ganhos)
    melhor_atributo = atributos_disponiveis[melhor_idx]
    
    arvore = {melhor_atributo: {}}
    
    # Cria os ramos da árvore para cada valor possível do atributo escolhido
    for valor_unico in np.unique(dados[melhor_atributo]):
        subset_dados = dados[dados[melhor_atributo] == valor_unico]
        subset_rotulos = rotulos[dados[melhor_atributo] == valor_unico]
        
        # Remove o atributo já utilizado
        novos_atributos = [attr for attr in atributos_disponiveis if attr != melhor_atributo]
        
        arvore[melhor_atributo][valor_unico] = construir_arvore(subset_dados, subset_rotulos, novos_atributos)
    
    return arvore

# ====================================
# Carregar os dados
# ====================================
dados_csv = pd.read_csv('../test.csv')

# Considerando que a última coluna seja o rótulo
atributos = list(dados_csv.columns[:-1])
classe_alvo = dados_csv.iloc[:, -1].values

# Construir a árvore
arvore_resultado = construir_arvore(dados_csv[atributos], classe_alvo, atributos)

# Exibir a árvore final
print("Árvore de decisão gerada (ID3):")
print(arvore_resultado)
