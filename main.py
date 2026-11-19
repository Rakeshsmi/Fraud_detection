"""End-to-end pipeline: data -> clean -> features -> models -> evaluation -> patterns -> alerts.

Run:  python main.py
"""
import json
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_generation import generate_claims
from src.modeling import (FEATURE_COLS, build_alerts, build_models,
                          cross_validate_threshold, evaluate)
from src.pattern_mining import mine_fraud_patterns
from src.preprocessing import TARGET, clean, engineer_features
from src import visualize as viz

DATA_PATH = "data/insurance_claims.csv"
for d in ["data", "models", "outputs/figures"]:
    os.makedirs(d, exist_ok=True)

# 1. Data ------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    generate_claims().to_csv(DATA_PATH, index=False)
raw = pd.read_csv(DATA_PATH)
print(f"[1] Loaded {raw.shape[0]} rows, {raw.shape[1]} columns")
print("    Missing values per column:\n", raw.isna().sum()[raw.isna().sum() > 0].to_string())

# 2. Preprocessing + feature engineering -----------------------------------
df = engineer_features(clean(raw))
print(f"[2] After cleaning: {len(df)} rows | fraud rate = {df[TARGET].mean():.3%}")
viz.eda_plots(df)

# 3. Train / test split ----------------------------------------------------
X, y = df[FEATURE_COLS], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
df_te = df.loc[X_te.index]

# 4. Train + evaluate all models -------------------------------------------
models, results = build_models(), []
for name, model in models.items():
    model.fit(X_tr, y_tr)
    thr = 0.5 if name.startswith("Baseline") else cross_validate_threshold(model, X_tr, y_tr)
    results.append(evaluate(name, model, X_te, y_te, thr))
    print(f"[3] trained {name:<28} (threshold {thr:.2f})")

res = pd.DataFrame(results).set_index("model")
res.round(3).to_csv("outputs/model_comparison.csv")
print("\n=== MODEL COMPARISON (held-out test set) ===")
print(res[["threshold", "roc_auc", "pr_auc", "precision", "recall", "f1"]].round(3).to_string())

best = res.drop(index="Baseline (majority class)")["pr_auc"].idxmax()
thr = float(res.loc[best, "threshold"])
print(f"\nBest model by PR-AUC: {best} (alert threshold = {thr:.2f})")

# 5. Visualisations + feature importance -----------------------------------
viz.model_plots(models, X_te, y_te, best, thr)
top_features = viz.importance_plot(models[best], X_te, y_te, list(X_te.columns))

# 6. Pattern mining --------------------------------------------------------
patterns = mine_fraud_patterns(df)          # mined on full labelled history
patterns.to_csv("outputs/fraud_patterns.csv", index=False)
viz.pattern_plot(patterns)
print("\n=== RECURRING FRAUD PATTERNS (Apriori, ranked by lift) ===")
print(patterns[["pattern", "support_in_fraud", "fraud_rate_when_present", "lift"]].head(10).to_string(index=False))

# 7. Predictions + risk alerts --------------------------------------------
alerts = build_alerts(df_te, models[best].predict_proba(X_te)[:, 1], thr)
alerts = alerts.merge(df_te[["claim_id", TARGET]], on="claim_id").rename(columns={TARGET: "actual_fraud"})
alerts.to_csv("outputs/fraud_predictions.csv", index=False)
alerts[alerts["risk_level"] == "HIGH"].to_csv("outputs/high_risk_alerts.csv", index=False)
viz.risk_plot(alerts)
print("\n=== RISK ALERTS (test set) ===")
print(alerts["risk_level"].value_counts().to_string())
print("\nTop 5 high-risk claims:")
print(alerts.head(5)[["claim_id", "claim_type", "claim_amount", "fraud_probability", "risk_level", "reasons"]]
      .to_string(index=False))

# 8. Save artefacts --------------------------------------------------------
joblib.dump({"model": models[best], "threshold": thr, "name": best}, "models/best_model.joblib")
summary = {"best_model": best, "threshold": thr,
           "metrics": res.loc[best].round(4).to_dict(),
           "baseline_pr_auc": round(float(res.loc["Baseline (majority class)", "pr_auc"]), 4),
           "top_features": top_features}
json.dump(summary, open("outputs/summary.json", "w"), indent=2, default=float)
print("\nDone. Outputs in ./outputs, model in ./models/best_model.joblib")
