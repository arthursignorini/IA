# apriori_leite_cafe.py
# Suporte mínimo = 0.3, confiança = 0.8
# Base "Leite, Café, Cerveja, Pão, Manteiga, Arroz, Feijão" (10 transações)

from itertools import combinations

# 0 = Não, 1 = Sim (base da imagem)
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

MIN_SUP = 0.30
MIN_CONF = 0.80

cols = list(data[0].keys())
transactions = [tuple(k for k,v in row.items() if v==1) for row in data]
N = len(transactions)

def sup_count(itemset):
    s = set(itemset)
    return sum(1 for t in transactions if s.issubset(t))

def sup_frac(itemset):
    return sup_count(itemset) / N

def apriori(min_sup=0.3):
    min_count = int(min_sup * N + 1e-9)
    # L1
    L = {1: {}}
    for i in cols:
        c = sup_count([i])
        if c >= min_count:
            L[1][(i,)] = c
    k = 2
    while True:
        prev = list(L[k-1].keys())
        if not prev:
            break
        # join + prune
        Ck = set()
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                cand = tuple(sorted(set(prev[i]) | set(prev[j])))
                if len(cand) == k:
                    # all (k-1)-subsets must be frequent
                    if all(tuple(sorted(s)) in L[k-1] for s in combinations(cand, k-1)):
                        Ck.add(cand)
        if not Ck:
            break
        # count + filter
        Lk = {}
        for c in Ck:
            cnt = sup_count(c)
            if cnt >= min_count:
                Lk[c] = cnt
        if not Lk:
            break
        L[k] = Lk
        k += 1
    return L

def association_rules(Lmap, min_conf=0.8):
    rules = []
    for size, d in Lmap.items():
        if size < 2:  # precisa de pelo menos 2 itens
            continue
        for iset, cnt in d.items():
            items = tuple(sorted(iset))
            s_iset = cnt / N
            for r in range(1, len(items)):
                for ante in combinations(items, r):
                    cons = tuple(sorted(set(items) - set(ante)))
                    s_ante = sup_frac(ante)
                    if s_ante == 0: 
                        continue
                    conf = s_iset / s_ante
                    if conf >= min_conf:
                        rules.append((ante, cons, s_iset, conf))
    return rules

if __name__ == "__main__":
    L = apriori(MIN_SUP)
    # contagens pedidas
    n_L1 = len(L.get(1, {}))
    n_L2 = len(L.get(2, {}))
    n_L3 = len(L.get(3, {}))
    rules = association_rules(L, MIN_CONF)
    print(f"L1 = {n_L1}, L2 = {n_L2}, L3 = {n_L3}, Regras(conf>={MIN_CONF}) = {len(rules)}\n")

    # listar itemsets
    for k in sorted(L):
        print(f"L{k} (frequentes, sup>={MIN_SUP}):")
        for iset, cnt in sorted(L[k].items(), key=lambda x:(-x[1], x[0])):
            print(f"  {iset}  sup={cnt/N:.3f}  count={cnt}")
        print()

    # listar regras
    print(f"Regras (conf>={MIN_CONF}):")
    for ante, cons, s, conf in sorted(rules, key=lambda x:(-x[3], -x[2], x[0])):
        a = ", ".join(ante); c = ", ".join(cons)
        print(f"  {a} -> {c} | sup={s:.3f} conf={conf:.3f}")
