# apriori_print_ck_lk.py
from itertools import combinations

# ===== BASE (da imagem) =====
data = [
    {"Leite":0,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  #1
    {"Leite":1,"Café":0,"Cerveja":1,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  #2
    {"Leite":0,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  #3
    {"Leite":1,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  #4
    {"Leite":0,"Café":1,"Cerveja":1,"Pão":0,"Manteiga":1,"Arroz":0,"Feijão":0},  #5
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":1,"Manteiga":0,"Arroz":1,"Feijão":0},  #6
    {"Leite":0,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":0,"Arroz":0,"Feijão":1},  #7
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":0,"Manteiga":0,"Arroz":0,"Feijão":0},  #8
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":0,"Manteiga":0,"Arroz":1,"Feijão":1},  #9
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":0,"Manteiga":0,"Arroz":1,"Feijão":0},  #10
]

MIN_SUP = 0.30  # suporte mínimo (fração)

cols = list(data[0].keys())
transactions = [tuple(k for k in cols if row[k]==1) for row in data]
N = len(transactions)

def sup_count(itemset):
    s = set(itemset)
    return sum(1 for t in transactions if s.issubset(t))

def sup_frac(itemset):
    return sup_count(itemset) / N

def print_level(title, mapping):
    print(f"\n{title}")
    for iset, cnt in sorted(mapping.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {iset}  | sup={cnt/N:.3f}  (count={cnt})")

def apriori(min_sup=0.3):
    min_count = int(min_sup * N + 1e-9)

    # C1 (candidatos 1-item)
    C = {}
    C[1] = {(i,): sup_count([i]) for i in cols}
    print_level("C1 (candidatos)", C[1])

    # L1 (frequentes 1-item)
    L = {}
    L[1] = {k:v for k,v in C[1].items() if v >= min_count}
    print_level("L1 (frequentes)", L[1])

    k = 2
    while True:
        if not L[k-1]:
            break
        prev = list(L[k-1].keys())

        # Geração de Ck (join + prune)
        Ck = set()
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                cand = tuple(sorted(set(prev[i]) | set(prev[j])))
                if len(cand) == k:
                    # prune: todos os (k-1)-subconjuntos devem estar em L_{k-1}
                    if all(tuple(sorted(s)) in L[k-1] for s in combinations(cand, k-1)):
                        Ck.add(cand)

        # Contagem de Ck
        C[k] = {c: sup_count(c) for c in Ck}
        if not C[k]:
            break
        print_level(f"C{k} (candidatos)", C[k])

        # Filtra para Lk
        L[k] = {c:cnt for c,cnt in C[k].items() if cnt >= min_count}
        if not L[k]:
            print(f"\nNão há itemsets frequentes em L{k} com sup >= {min_sup:.2f}. Encerrando.")
            break
        print_level(f"L{k} (frequentes)", L[k])
        k += 1

    return C, L

if __name__ == "__main__":
    Cmap, Lmap = apriori(MIN_SUP)
    # (Opcional) resumo final dos frequentes
    print("\n=== RESUMO: TODOS OS ITEMSETS FREQUENTES ===")
    for k in sorted(Lmap):
        for iset, cnt in sorted(Lmap[k].items(), key=lambda x: (-x[1], x[0])):
            print(f"{iset} | sup={cnt/N:.3f}")
