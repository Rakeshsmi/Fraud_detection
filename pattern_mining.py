"""Association-rule / frequent-pattern mining of suspicious claims.

A small self-contained Apriori implementation (no extra dependency).
Each claim is turned into a set of readable 'items' (binned/boolean features);
we mine itemsets that are frequent among fraudulent claims and rank them by
lift = P(fraud | pattern) / P(fraud).
"""
from itertools import combinations

import numpy as np
import pandas as pd


def to_items(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean item matrix (rows = claims, columns = human-readable items)."""
    I = pd.DataFrame(index=df.index)
    I["high amount/coverage ratio (>0.6)"] = df["amount_to_coverage"] > 0.6
    I["new customer (<12m)"] = df["new_customer"] == 1
    I["late report (>14d)"] = df["late_report"] == 1
    I["coverage changed recently"] = df["coverage_changed_recently"] == 1
    I["no police report"] = df["police_report_filed"] == 0
    I["no witnesses"] = df["no_witness"] == 1
    I["night incident"] = df["night_incident"] == 1
    I["repeat claimant (>=3 prior)"] = df["previous_claims"] >= 3
    I["theft/fire claim"] = df["claim_type"].isin(["theft", "fire"])
    I["text: missing documents"] = df["kw_missing_docs"] == 1
    I["text: cash/urgent demand"] = (df["kw_cash"] == 1) | (df["kw_urgent"] == 1)
    I["text: total loss"] = df["kw_total_loss"] == 1
    I["text: no witnesses mentioned"] = df["kw_no_witness"] == 1
    return I.astype(bool)


def apriori(items: pd.DataFrame, min_support: float, max_len: int = 3) -> pd.DataFrame:
    X = items.values
    n = len(X)
    cols = list(items.columns)
    freq, level = [], []
    for j, c in enumerate(cols):
        s = X[:, j].mean()
        if s >= min_support:
            level.append((frozenset([j]), s))
    freq += level
    k = 2
    while level and k <= max_len:
        prev = [s for s, _ in level]
        cands = {a | b for a, b in combinations(prev, 2) if len(a | b) == k}
        level = []
        for c in cands:
            s = X[:, list(c)].all(axis=1).mean()
            if s >= min_support:
                level.append((c, s))
        freq += level
        k += 1
    return pd.DataFrame(
        [{"pattern": " + ".join(cols[j] for j in sorted(s)), "size": len(s),
          "support_in_fraud": sup, "_idx": sorted(s)} for s, sup in freq])


def mine_fraud_patterns(df: pd.DataFrame, min_support: float = 0.10,
                        min_lift: float = 1.5, top: int = 15) -> pd.DataFrame:
    items = to_items(df)
    y = df["is_fraud"].values.astype(bool)
    base_rate = y.mean()
    pats = apriori(items[y].reset_index(drop=True), min_support)
    rows = []
    for _, r in pats.iterrows():
        mask = items.values[:, r["_idx"]].all(axis=1)
        n_match = int(mask.sum())
        conf = y[mask].mean() if n_match else 0.0       # P(fraud | pattern)
        rows.append({
            "pattern": r["pattern"], "size": r["size"],
            "support_in_fraud": round(r["support_in_fraud"], 3),
            "claims_matching": n_match,
            "fraud_rate_when_present": round(conf, 3),
            "lift": round(conf / base_rate, 2),
        })
    out = pd.DataFrame(rows)
    out = out[(out["lift"] >= min_lift) & (out["size"] >= 2)]
    return out.sort_values(["lift", "support_in_fraud"], ascending=False).head(top).reset_index(drop=True)
