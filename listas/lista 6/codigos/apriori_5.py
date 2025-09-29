# apriori_5.py
# Sem dependências externas. Salva resultados em CSV.

from itertools import combinations
from csv import DictWriter

# ===== 1) BASE (a da imagem) =====
# 0 = "Não", 1 = "Sim"
data = [
    # N°, Leite, Café, Cerveja, Pão, Manteiga, Arroz, Feijão
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

# ===== 2) PARÂMETROS =====
MIN_SUP = 0.30   # suporte mínimo (fração)
MIN_CONF = 0.60  # confiança mínima (fração)

# ===== 3) TRANSFORMAÇÃO EM TRANSAÇÕES =====
cols = list(data[0].keys())
transactions = [tuple(k for k in cols if row[k]==1) for row in data]
N = len(transactions)

def support_count(itemset):
    s = set(itemset)
    return sum(1 for t in transactions if s.issubset(t))

def support_frac(itemset):
    return support_count(itemset) / N

# ===== 4) APRIORI =====
def apriori(min_sup=0.3):
    min_count = int(min_sup * N + 1e-9)
    # L1
    L = {}
    L[1] = {}
    for item in cols:
        cnt = support_count([item])
        if cnt >= min_count:
            L[1][(item,)] = cnt

    k = 2
    while True:
        prev = list(L[k-1].keys())
        # join
        Ck = set()
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                cand = tuple(sorted(set(prev[i]) | set(prev[j])))
                if len(cand) == k:
                    # prune
                    if all(tuple(sorted(s)) in L[k-1] for s in combinations(cand, k-1)):
                        Ck.add(cand)
        # count + filter
        Lk = {}
        for c in Ck:
            cnt = support_count(c)
            if cnt >= min_count:
                Lk[c] = cnt
        if not Lk:
            break
        L[k] = Lk
        k += 1
    return L

# ===== 5) REGRAS =====
def association_rules(Lmap, min_conf=0.6):
    rules = []
    for size, d in Lmap.items():
        if size < 2: 
            continue
        for iset, cnt in d.items():
            sup_iset = cnt / N
            items = tuple(sorted(iset))
            for r in range(1, len(items)):
                for ante in combinations(items, r):
                    cons = tuple(sorted(set(items) - set(ante)))
                    sup_ante = support_frac(ante)
                    if sup_ante == 0: 
                        continue
                    conf = sup_iset / sup_ante
                    if conf >= min_conf:
                        sup_cons = support_frac(cons)
                        lift = conf / sup_cons if sup_cons > 0 else float("inf")
                        rules.append({
                            "antecedent": ante,
                            "consequent": cons,
                            "support": round(sup_iset, 3),
                            "confidence": round(conf, 3),
                            "lift": round(lift, 3)
                        })
    # ordena por confiança, depois suporte
    rules.sort(key=lambda r: (r["confidence"], r["support"]), reverse=True)
    return rules

# ===== 6) EXECUÇÃO =====
Lmap = apriori(MIN_SUP)

# imprime itemsets
print("\n=== ITEMSETS FREQUENTES (min_sup =", MIN_SUP, ") ===")
freq_rows = []
for k in sorted(Lmap):
    for iset, cnt in sorted(Lmap[k].items(), key=lambda x: (-x[1], x[0])):
        sup = cnt / N
        print(f"{iset} | sup={sup:.3f} | count={cnt}")
        freq_rows.append({"itemset": " & ".join(iset), "size": k, "support": round(sup,3), "count": cnt})

# imprime regras
rules = association_rules(Lmap, MIN_CONF)
print("\n=== REGRAS (min_conf =", MIN_CONF, ") ===")
for r in rules:
    a = ", ".join(r["antecedent"])
    c = ", ".join(r["consequent"])
    print(f"{a}  =>  {c} | sup={r['support']:.3f} conf={r['confidence']:.3f} lift={r['lift']:.3f}")

# ===== 7) SALVA CSVs =====
with open("frequent_itemsets.csv","w",newline="",encoding="utf-8") as f:
    w = DictWriter(f, fieldnames=["itemset","size","support","count"])
    w.writeheader()
    for row in freq_rows:
        w.writerow(row)

with open("association_rules.csv","w",newline="",encoding="utf-8") as f:
    w = DictWriter(f, fieldnames=["antecedent","consequent","support","confidence","lift"])
    w.writeheader()
    for r in rules:
        w.writerow({
            "antecedent": " & ".join(r["antecedent"]),
            "consequent": " & ".join(r["consequent"]),
            "support": r["support"],
            "confidence": r["confidence"],
            "lift": r["lift"],
        })

print("\nArquivos salvos: frequent_itemsets.csv, association_rules.csv")
