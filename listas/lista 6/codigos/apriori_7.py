# apriori_negativos.py
# Regras com ausência de item: ex "quem NÃO leva Cerveja leva Pão"

from itertools import combinations

# ===== BASE (a da imagem) =====
# 1 = "Sim", 0 = "Não"
data = [
    {"Leite":0,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  # 1
    {"Leite":1,"Café":0,"Cerveja":1,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  # 2
    {"Leite":0,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  # 3
    {"Leite":1,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":1,"Arroz":0,"Feijão":0},  # 4
    {"Leite":0,"Café":1,"Cerveja":1,"Pão":0,"Manteiga":1,"Arroz":0,"Feijão":0},  # 5
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":1,"Manteiga":0,"Arroz":1,"Feijão":0},  # 6
    {"Leite":0,"Café":1,"Cerveja":0,"Pão":1,"Manteiga":0,"Arroz":0,"Feijão":1},  # 7
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":0,"Manteiga":0,"Arroz":0,"Feijão":0},  # 8
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":0,"Manteiga":0,"Arroz":1,"Feijão":1},  # 9
    {"Leite":0,"Café":0,"Cerveja":0,"Pão":0,"Manteiga":0,"Arroz":1,"Feijão":0},  # 10
]

# ===== PARÂMETROS =====
MIN_SUP = 0.30     # suporte mínimo
MIN_CONF = 0.60    # confiança mínima
ONLY_NEG_ANTECEDENT = True  # mostra só regras cujo antecedente tem pelo menos um "não X"

# ===== PREP =====
cols = list(data[0].keys())
N = len(data)
# transação como dicionário {item: 0 ou 1}
transactions = data

# universo de literais: (item, True) significa "leva item"
#                       (item, False) significa "não leva item"
LITERALS = [(c, True) for c in cols] + [(c, False) for c in cols]

def literal_str(lit):
    item, val = lit
    return item if val else f"não {item}"

def has_conflict(itemset):
    """evita itemset com A e não A ao mesmo tempo"""
    seen = {}
    for it, val in itemset:
        if it in seen and seen[it] != val:
            return True
        seen[it] = val
    return False

def support_count(itemset):
    """quantas transações satisfazem todos os literais"""
    def match(t):
        for it, val in itemset:
            if t[it] != (1 if val else 0):
                return False
        return True
    return sum(1 for t in transactions if match(t))

def support_frac(itemset):
    return support_count(itemset) / N

def print_level(title, mapping):
    print(f"\n{title}")
    for iset, cnt in sorted(mapping.items(), key=lambda x: (-x[1], tuple(literal_str(l) for l in x[0]))):
        sup = cnt / N
        s = ", ".join(literal_str(l) for l in iset)
        print(f"  {{{s}}} | sup={sup:.3f}  count={cnt}")

# ===== APRIORI COM LITERAIS POS E NEG =====
def apriori(min_sup=0.3):
    min_count = int(min_sup * N + 1e-9)

    # C1
    C = {}
    C[1] = { (lit,): support_count([lit]) for lit in LITERALS }
    print_level("C1 (candidatos)", C[1])

    # L1
    L = {}
    L[1] = { k:v for k,v in C[1].items() if v >= min_count }
    print_level("L1 (frequentes)", L[1])

    k = 2
    while True:
        if not L.get(k-1):
            break
        prev = list(L[k-1].keys())

        # join
        Ck = set()
        for i in range(len(prev)):
            for j in range(i+1, len(prev)):
                cand = tuple(sorted(set(prev[i]) | set(prev[j])))
                if len(cand) == k and not has_conflict(cand):
                    # prune: todos os (k-1) subconjuntos devem estar em L_{k-1}
                    if all(tuple(sorted(s)) in L[k-1] for s in combinations(cand, k-1)):
                        Ck.add(cand)

        # contar
        C[k] = { c: support_count(c) for c in Ck }
        if not C[k]:
            break
        print_level(f"C{k} (candidatos)", C[k])

        # filtrar
        L[k] = { c:cnt for c,cnt in C[k].items() if cnt >= min_count }
        if not L[k]:
            print(f"\nSem frequentes em L{k} com sup >= {min_sup:.2f}.")
            break
        print_level(f"L{k} (frequentes)", L[k])
        k += 1

    return C, L

# ===== REGRAS =====
def association_rules(Lmap, min_conf=0.6):
    rules = []
    def fmt(lits):
        return ", ".join(literal_str(l) for l in lits)

    for size, d in Lmap.items():
        if size < 2:
            continue
        for iset, cnt in d.items():
            items = tuple(sorted(iset))
            sup_iset = cnt / N
            for r in range(1, len(items)):
                for ante in combinations(items, r):
                    cons = tuple(sorted(set(items) - set(ante)))
                    sup_ante = support_frac(ante)
                    if sup_ante == 0:
                        continue
                    conf = sup_iset / sup_ante
                    if conf >= min_conf:
                        # filtro: antecedente deve ter ao menos um literal negativo, se ativado
                        if ONLY_NEG_ANTECEDENT and not any(not v for _, v in ante):
                            continue
                        sup_cons = support_frac(cons)
                        lift = conf / sup_cons if sup_cons > 0 else float("inf")
                        rules.append({
                            "antecedent": ante,
                            "consequent": cons,
                            "support": sup_iset,
                            "confidence": conf,
                            "lift": lift,
                            "ante_str": fmt(ante),
                            "cons_str": fmt(cons),
                        })
    rules.sort(key=lambda r: (r["confidence"], r["support"]), reverse=True)
    return rules

# ===== RUN =====
if __name__ == "__main__":
    Cmap, Lmap = apriori(MIN_SUP)

    print("\n=== REGRAS COM AUSÊNCIA NO ANTECEDENTE ===")
    rules = association_rules(Lmap, MIN_CONF)
    if not rules:
        print("Nenhuma regra passou os limiares atuais.")
    else:
        for r in rules:
            print(f"Se {{{r['ante_str']}}} => {{{r['cons_str']}}}  | "
                  f"sup={r['support']:.3f}  conf={r['confidence']:.3f}  lift={r['lift']:.3f}")
