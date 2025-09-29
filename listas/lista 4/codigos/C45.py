import pandas as pd
import numpy as np
from collections import Counter

# =======================================
# Classe do nó da árvore
# =======================================
class NoArvore:
    def __init__(self, atributo=None, corte=None, ramos=None, classe=None, folha=False):
        self.atributo = atributo      # índice do atributo
        self.corte = corte            # valor do limiar
        self.ramos = ramos or {}      # filhos do nó
        self.classe = classe          # classe final (se for folha)
        self.folha = folha            # indica se é folha

    def __str__(self, nivel=0):
        ident = "  " * nivel
        if self.folha:
            return f"{ident}Classe: {self.classe}\n"
        saida = f"{ident}[Atributo {self.atributo} < {self.corte}]\n"
        for condicao, ramo in self.ramos.items():
            saida += f"{ident}  → {condicao}:\n{ramo.__str__(nivel + 2)}"
        return saida


# =======================================
# Função para calcular entropia
# =======================================
def calcular_entropia(labels):
    contagem = Counter(labels)
    total = len(labels)
    return -sum((qtd / total) * np.log2(qtd / total) for qtd in contagem.values())


# =======================================
# Função para ganho de informação e gain ratio
# =======================================
def ganho_info_e_ratio(coluna, rotulos, corte):
    """Divide o conjunto em dois lados e calcula ganho de informação e gain ratio"""
    lado_esq = [rotulos[i] for i in range(len(rotulos)) if coluna[i] < corte]
    lado_dir = [rotulos[i] for i in range(len(rotulos)) if coluna[i] >= corte]

    if not lado_esq or not lado_dir:
        return 0, 0

    entropia_inicial = calcular_entropia(rotulos)
    entropia_esq = calcular_entropia(lado_esq)
    entropia_dir = calcular_entropia(lado_dir)

    p_esq = len(lado_esq) / len(rotulos)
    p_dir = len(lado_dir) / len(rotulos)

    entropia_apos_split = p_esq * entropia_esq + p_dir * entropia_dir
    info_gain = entropia_inicial - entropia_apos_split

    # Split info (normalização usada pelo C4.5)
    split_info = -sum(p * np.log2(p) for p in [p_esq, p_dir] if p > 0)
    gain_ratio = info_gain / split_info if split_info != 0 else 0

    return info_gain, gain_ratio


# =======================================
# Identifica o melhor atributo e corte
# =======================================
def encontrar_melhor_divisao(X, y):
    num_atributos = len(X[0])
    melhor_attr = None
    melhor_corte = None
    melhor_gain_ratio = -float("inf")

    for idx_attr in range(num_atributos):
        valores_attr = [amostra[idx_attr] for amostra in X]
        valores_unicos = sorted(set(valores_attr))

        # Pontos médios como candidatos
        cortes = [(valores_unicos[i] + valores_unicos[i + 1]) / 2
                  for i in range(len(valores_unicos) - 1)]

        for corte in cortes:
            _, ratio = ganho_info_e_ratio(valores_attr, y, corte)
            if ratio > melhor_gain_ratio:
                melhor_gain_ratio = ratio
                melhor_attr = idx_attr
                melhor_corte = corte

    return melhor_attr, melhor_corte


# =======================================
# Função recursiva para montar a árvore
# =======================================
def treinar_arvore(X, y, profundidade_max=None, nivel=0):
    # Caso 1: todas as classes são iguais
    if len(set(y)) == 1:
        return NoArvore(classe=y[0], folha=True)

    # Caso 2: não há atributos ou limite de profundidade atingido
    if not X or (profundidade_max is not None and nivel >= profundidade_max):
        classe_comum = Counter(y).most_common(1)[0][0]
        return NoArvore(classe=classe_comum, folha=True)

    # Encontrar melhor divisão
    atributo, corte = encontrar_melhor_divisao(X, y)
    if atributo is None:
        classe_comum = Counter(y).most_common(1)[0][0]
        return NoArvore(classe=classe_comum, folha=True)

    # Dividir dataset
    X_esq = [linha for linha in X if linha[atributo] < corte]
    y_esq = [y[i] for i in range(len(y)) if X[i][atributo] < corte]

    X_dir = [linha for linha in X if linha[atributo] >= corte]
    y_dir = [y[i] for i in range(len(y)) if X[i][atributo] >= corte]

    # Caso em que não é possível dividir mais
    if not y_esq or not y_dir:
        classe_comum = Counter(y).most_common(1)[0][0]
        return NoArvore(classe=classe_comum, folha=True)

    # Recursão
    ramo_esq = treinar_arvore(X_esq, y_esq, profundidade_max, nivel + 1)
    ramo_dir = treinar_arvore(X_dir, y_dir, profundidade_max, nivel + 1)

    return NoArvore(
        atributo=atributo,
        corte=corte,
        ramos={'<': ramo_esq, '>=': ramo_dir},
        folha=False
    )


# =======================================
# Função para prever
# =======================================
def prever_amostra(amostra, arvore):
    if arvore.folha:
        return arvore.classe
    valor = amostra[arvore.atributo]
    if valor < arvore.corte:
        return prever_amostra(amostra, arvore.ramos['<'])
    else:
        return prever_amostra(amostra, arvore.ramos['>='])


# =======================================
# Avaliação da árvore
# =======================================
def calcular_acuracia(X, y, arvore):
    predicoes = [prever_amostra(x, arvore) for x in X]
    acertos = sum(1 for real, prev in zip(y, predicoes) if real == prev)
    return acertos / len(y)


# =======================================
# Execução principal
# =======================================
dados = pd.read_csv('./teste.csv')

# Última coluna = rótulo
X = dados.iloc[:, :-1].values.tolist()
y = dados.iloc[:, -1].values.tolist()

# Construção da árvore
modelo_c45 = treinar_arvore(X, y)

# Exibir a estrutura
print("Árvore de decisão construída (C4.5):")
print(modelo_c45)

# Avaliar no próprio conjunto
acuracia = calcular_acuracia(X, y, modelo_c45)
print(f"Acurácia no treino: {acuracia:.2f}")
