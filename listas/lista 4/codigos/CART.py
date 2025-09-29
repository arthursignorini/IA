# -*- coding: utf-8 -*-
"""
Árvore CART personalizada para o dataset Titanic (train.csv)
- Critério: índice Gini
- Splits binários: numéricos por limiar ótimo; categóricos por prefixo após ordenação por taxa de positivos
- Partição: 80/20 estratificada (seed ajustável)
- Gera: arquivo com árvore, regras e relatório em Markdown
"""

import argparse
import os
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import pandas as pd

# -------------------------
# Estrutura do nó (estilo diferente)
# -------------------------
class NodeCART:
    def __init__(
        self,
        feat_idx: Optional[int] = None,
        feat_name: Optional[str] = None,
        thresh: Optional[float] = None,
        left_categories: Optional[Set[Any]] = None,
        categorical: bool = False,
        children: Optional[Dict[str, 'NodeCART']] = None,
        prediction: Optional[Any] = None,
        leaf: bool = False
    ):
        self.feat_idx = feat_idx
        self.feat_name = feat_name
        self.thresh = thresh
        self.left_categories = left_categories or set()
        self.categorical = categorical
        self.children = children or {}
        self.prediction = prediction
        self.leaf = leaf

    def __str__(self, depth=0) -> str:
        pad = "  " * depth
        if self.leaf:
            return f"{pad}Leaf -> {self.prediction}\n"
        if self.categorical:
            cond_txt = f"{self.feat_name} ∈ {sorted(list(self.left_categories))}"
            s = f"{pad}{cond_txt}\n"
            s += f"{pad}  → IN:\n{self.children['in'].__str__(depth+2)}"
            s += f"{pad}  → OUT:\n{self.children['out'].__str__(depth+2)}"
            return s
        else:
            s = f"{pad}[{self.feat_name} < {self.thresh:.6g}]\n"
            s += f"{pad}  → < :\n{self.children['<'].__str__(depth+2)}"
            s += f"{pad}  → >= :\n{self.children['>='].__str__(depth+2)}"
            return s

    def rules(self, prefix: Optional[List[str]] = None) -> List[str]:
        prefix = prefix or []
        out: List[str] = []
        if self.leaf:
            cond = " AND ".join(prefix) if prefix else "(TRUE)"
            out.append(f"{cond} => predict {self.prediction}")
            return out
        if self.categorical:
            left_cond = f"{self.feat_name} ∈ {{{', '.join(map(str, sorted(list(self.left_categories))))}}}"
            right_cond = f"{self.feat_name} ∉ {{{', '.join(map(str, sorted(list(self.left_categories))))}}}"
            out += self.children['in'].rules(prefix + [left_cond])
            out += self.children['out'].rules(prefix + [right_cond])
            return out
        else:
            left_cond = f"{self.feat_name} < {self.thresh:.6g}"
            right_cond = f"{self.feat_name} >= {self.thresh:.6g}"
            out += self.children['<'].rules(prefix + [left_cond])
            out += self.children['>='].rules(prefix + [right_cond])
            return out

# -------------------------
# Métricas auxiliares
# -------------------------
def gini(y: np.ndarray) -> float:
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return 0.0
    cnt = Counter(y)
    return 1.0 - sum((v / n) ** 2 for v in cnt.values())

def acc(y_true: np.ndarray, y_pred: List[Any]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def confusion(y_true: np.ndarray, y_pred: List[Any]):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [0, 1]
    mat = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        mat[labels.index(t), labels.index(p)] += 1
    return mat

# -------------------------
# Splits numéricos: varre midpoints
# -------------------------
def find_best_numeric(col: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[Optional[float], float]:
    vals = col
    uniq = np.unique(vals)
    if uniq.size < 2:
        return None, np.inf
    candidates = (uniq[1:] + uniq[:-1]) / 2.0
    best_thr, best_g = None, np.inf
    n = len(y)
    for thr in candidates:
        left_mask = vals < thr
        nL = left_mask.sum()
        nR = n - nL
        if nL < min_leaf or nR < min_leaf:
            continue
        g = (nL / n) * gini(y[left_mask]) + (nR / n) * gini(y[~left_mask])
        if g < best_g:
            best_g = g
            best_thr = thr
    return best_thr, best_g

# -------------------------
# Splits categóricos: ordena por taxa de positivos e testa prefixos
# -------------------------
def find_best_categorical(col: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[Optional[Set[Any]], float]:
    cats, inv = np.unique(col, return_inverse=True)
    if cats.size < 2:
        return None, np.inf
    rates = []
    for i, c in enumerate(cats):
        mask = (inv == i)
        yi = y[mask]
        rate = (yi == 1).mean() if yi.size > 0 else 0.0
        rates.append((c, rate))
    rates.sort(key=lambda x: x[1])
    order = [c for c, _ in rates]
    best_set, best_g = None, np.inf
    n = len(y)
    for k in range(1, len(order)):
        left_set = set(order[:k])
        mask_left = np.isin(col, list(left_set))
        nL = mask_left.sum()
        nR = n - nL
        if nL < min_leaf or nR < min_leaf:
            continue
        g = (nL / n) * gini(y[mask_left]) + (nR / n) * gini(y[~mask_left])
        if g < best_g:
            best_g = g
            best_set = left_set
    return best_set, best_g

# -------------------------
# Seleção do melhor atributo (CART)
# -------------------------
def choose_best_split(
    X: np.ndarray, y: np.ndarray,
    feature_names: List[str], categorical_feats: Set[str],
    min_samples_leaf: int
) -> Tuple[Optional[int], Optional[float], Optional[Set[Any]], bool]:
    n, d = X.shape
    best_idx, best_thr, best_set, is_cat = None, None, None, False
    best_g = np.inf
    for j in range(d):
        name = feature_names[j]
        col = X[:, j]
        if name in categorical_feats:
            left_set, g = find_best_categorical(col, y, min_samples_leaf)
            if g < best_g:
                best_g = g
                best_idx = j
                best_thr = None
                best_set = left_set
                is_cat = True
        else:
            thr, g = find_best_numeric(col.astype(float), y, min_samples_leaf)
            if g < best_g:
                best_g = g
                best_idx = j
                best_thr = thr
                best_set = None
                is_cat = False
    return best_idx, best_thr, best_set, is_cat

def most_common(y: np.ndarray):
    return Counter(y).most_common(1)[0][0]

# -------------------------
# Construção recursiva da árvore
# -------------------------
def build_cart(
    X: np.ndarray, y: np.ndarray,
    feature_names: List[str], categorical_feats: Set[str],
    max_depth: Optional[int] = None, depth: int = 0,
    min_samples_split: int = 2, min_samples_leaf: int = 1
) -> NodeCART:
    if len(set(y)) == 1:
        return NodeCART(prediction=y[0], leaf=True)
    if len(y) == 0 or (max_depth is not None and depth >= max_depth) or len(y) < min_samples_split:
        return NodeCART(prediction=most_common(y), leaf=True)

    idx, thr, left_set, is_cat = choose_best_split(X, y, feature_names, categorical_feats, min_samples_leaf)
    if idx is None:
        return NodeCART(prediction=most_common(y), leaf=True)

    fname = feature_names[idx]
    if is_cat:
        left_mask = np.isin(X[:, idx], list(left_set))
        right_mask = ~left_mask
        if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
            return NodeCART(prediction=most_common(y), leaf=True)
        left_node = build_cart(X[left_mask], y[left_mask], feature_names, categorical_feats, max_depth, depth+1, min_samples_split, min_samples_leaf)
        right_node = build_cart(X[right_mask], y[right_mask], feature_names, categorical_feats, max_depth, depth+1, min_samples_split, min_samples_leaf)
        return NodeCART(
            feat_idx=idx,
            feat_name=fname,
            left_categories=left_set,
            categorical=True,
            children={'in': left_node, 'out': right_node},
            prediction=most_common(y),
            leaf=False
        )
    else:
        left_mask = X[:, idx].astype(float) < float(thr)
        right_mask = ~left_mask
        if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
            return NodeCART(prediction=most_common(y), leaf=True)
        left_node = build_cart(X[left_mask], y[left_mask], feature_names, categorical_feats, max_depth, depth+1, min_samples_split, min_samples_leaf)
        right_node = build_cart(X[right_mask], y[right_mask], feature_names, categorical_feats, max_depth, depth+1, min_samples_split, min_samples_leaf)
        return NodeCART(
            feat_idx=idx,
            feat_name=fname,
            thresh=thr,
            categorical=False,
            children={'<': left_node, '>=': right_node},
            prediction=most_common(y),
            leaf=False
        )

# -------------------------
# Predição e avaliação
# -------------------------
def predict_cart(x: np.ndarray, tree: NodeCART) -> Any:
    node = tree
    while not node.leaf:
        if node.categorical:
            if x[node.feat_idx] in node.left_categories:
                node = node.children['in']
            else:
                node = node.children['out']
        else:
            if float(x[node.feat_idx]) < float(node.thresh):
                node = node.children['<']
            else:
                node = node.children['>=']
    return node.prediction

def evaluate_cart(X: np.ndarray, y: np.ndarray, tree: NodeCART) -> float:
    preds = [predict_cart(X[i], tree) for i in range(len(X))]
    return acc(y, preds)

# -------------------------
# Titanic preprocessing
# -------------------------
REQ_COLS = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def load_titanic(path: str) -> Tuple[np.ndarray, np.ndarray, List[str], Set[str]]:
    df = pd.read_csv(path)
    df = df[REQ_COLS].copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])
    categorical_feats = {'Sex', 'Embarked'}
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    cols = []
    for f in feature_names:
        if f in categorical_feats:
            cols.append(df[f].astype(object).to_numpy())
        else:
            cols.append(df[f].astype(float).to_numpy())
    X = np.column_stack(cols)
    y = df['Survived'].astype(int).to_numpy()
    return X, y, feature_names, categorical_feats

def stratified_indices(y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    test_idx = []
    for c in np.unique(y):
        cls_idx = idx[y == c]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(test_size * len(cls_idx))))
        test_idx.extend(cls_idx[:n_test])
    test_idx = np.array(sorted(test_idx))
    train_idx = np.array([i for i in idx if i not in set(test_idx)])
    return train_idx, test_idx

# -------------------------
# Save outputs
# -------------------------
def dump_tree_and_rules(tree: NodeCART, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "tree_cart.txt"), "w", encoding="utf-8") as f:
        f.write(str(tree))
    with open(os.path.join(outdir, "rules_cart.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tree.rules()))

def dump_report(outdir: str, acc_tr: float, acc_te: float, cm_tr, cm_te):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"""# CART (Gini) — Titanic

**Configuração**
- Atributos usados: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- Tratamento de faltantes: Age/Fare (mediana), Embarked (moda)
- Partição: 80/20 estratificada

**Resultados**
- Acurácia (treino): {acc_tr:.4f}
- Acurácia (teste) : {acc_te:.4f}

**Matriz de confusão (treino)**
{cm_tr}

**Matriz de confusão (teste)**
{cm_te}
""")

# -------------------------
# CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Caminho para train.csv (Titanic)")
    parser.add_argument("--out", required=True, help="Diretório de saída")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    args = parser.parse_args()

    X, y, feat_names, cat_feats = load_titanic(args.data)
    tr_idx, te_idx = stratified_indices(y, 0.2, args.seed)

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    tree = build_cart(
        X_tr, y_tr,
        feature_names=feat_names,
        categorical_feats=cat_feats,
        max_depth=args.max_depth,
        depth=0,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )

    acc_tr = evaluate_cart(X_tr, y_tr, tree)
    acc_te = evaluate_cart(X_te, y_te, tree)
    cm_tr = confusion(y_tr, [predict_cart(x, tree) for x in X_tr])
    cm_te = confusion(y_te, [predict_cart(x, tree) for x in X_te])

    dump_tree_and_rules(tree, args.out)
    dump_report(args.out, acc_tr, acc_te, cm_tr, cm_te)

    print(">>> CART finalizado")
    print(tree)
    print(f"Train ACC: {acc_tr:.4f}")
    print(f"Test  ACC: {acc_te:.4f}")
    print("Outputs saved to:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()
