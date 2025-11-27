# ============================================
# 1. Importações
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter

# ============================================
# 2. Carregamento da base
# ============================================
df = pd.read_csv("creditcard.csv")

print("--- Primeiras linhas ---")
display(df.head())

print("\n--- Dimensões ---")
print(df.shape)

print("\n--- Info ---")
print(df.info())

print("\n--- Estatísticas ---")
display(df.describe())

# ============================================
# 3. Distribuição das Classes
# ============================================
contagem = df['Class'].value_counts()
proporcao = contagem / len(df) * 100

print("\n--- Contagem de Classes ---")
print(contagem)

print("\n--- Proporção (%) ---")
print(proporcao)

# Gráfico (matplotlib)
plt.figure(figsize=(6,4))
plt.bar(contagem.index, contagem.values)
plt.title("Distribuição da Variável Alvo (Class)")
plt.xlabel("Classe")
plt.ylabel("Quantidade")

for i,v in enumerate(contagem.values):
    plt.text(i, v + 1000, str(v), ha='center')
    plt.text(i, v / 2, f"{proporcao.iloc[i]:.4f}%", ha='center', color='white')

plt.show()

# ============================================
# 4. Correlação
# ============================================
corr = df.corr()

plt.figure(figsize=(14,12))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.title("Matriz de Correlação")
plt.show()

print("\n--- Correlação com Class ---")
corr_class = corr['Class'].sort_values(ascending=False)
print(corr_class)

plt.figure(figsize=(10,6))
plt.barh(corr_class.drop("Class").index, corr_class.drop("Class").values)
plt.xlabel("Correlação")
plt.title("Correlação com a Classe (Class)")
plt.show()

# ============================================
# 5. Remoção de colunas pouco relevantes
# ============================================
colunas_remover = ['V28', 'V27', 'V23', 'V22', 'V25', 'V13', 'V15']

df = df.drop(columns=colunas_remover)

print("\n--- Colunas restantes ---")
print(df.shape)

# ============================================
# 6. Nulos e Duplicados
# ============================================
print("\n--- Valores ausentes ---")
print(df.isnull().sum())

print("\nDuplicados:", df.duplicated().sum())

df = df.drop_duplicates()

# ============================================
# 7. Boxplots (matplotlib)
# ============================================
plt.figure(figsize=(6,4))
plt.boxplot(df['Amount'])
plt.title("Boxplot - Amount")
plt.ylabel("Valor")
plt.show()

print(df['Amount'].describe())

plt.figure(figsize=(6,4))
plt.boxplot(df['Time'])
plt.title("Boxplot - Time")
plt.ylabel("Tempo")
plt.show()

print(df['Time'].describe())

# ============================================
# 8. Divisão Treino/Teste
# ============================================
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\n--- Proporções após estratificação ---")
print(y_train.value_counts(normalize=True)*100)
print(y_test.value_counts(normalize=True)*100)

# ============================================
# 9. Escalonamento
# ============================================
scaler = StandardScaler()
cols_scale = ['Time', 'Amount']

X_train[cols_scale] = scaler.fit_transform(X_train[cols_scale])
X_test[cols_scale] = scaler.transform(X_test[cols_scale])

# ============================================
# 10. SMOTE (apenas treino)
# ============================================
print("\n--- Antes SMOTE ---")
print(Counter(y_train))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n--- Depois SMOTE ---")
print(Counter(y_train_res))

# ============================================
# 11. Rejuntando para Clustering
# ============================================
X_scaled_all = pd.concat([X_train, X_test])

print("\nConjunto total escalonado:", X_scaled_all.shape)

# ============================================
# 12. K-Means
# ============================================
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
labels_kmeans = kmeans.fit_predict(X_scaled_all)

sil_kmeans = silhouette_score(X_scaled_all, labels_kmeans)

print("\n--- K-Means ---")
print("Tamanho dos clusters:")
print(pd.Series(labels_kmeans).value_counts())
print(f"Silhueta: {sil_kmeans:.4f}")

# ============================================
# 13. DBSCAN
# ============================================
dbscan = DBSCAN(eps=2, min_samples=10)
labels_db = dbscan.fit_predict(X_scaled_all)

clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)

print("\n--- DBSCAN ---")
print("Clusters encontrados:", clusters_db)
print(pd.Series(labels_db).value_counts())

if clusters_db > 1:
    mask = labels_db != -1
    sil_db = silhouette_score(X_scaled_all[mask], labels_db[mask])
    print("Silhueta:", sil_db)
else:
    print("Não foi possível calcular Silhueta (apenas 1 cluster)")
