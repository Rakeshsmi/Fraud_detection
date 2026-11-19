"""Model training, evaluation, thresholding and risk alerts."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_COLUMN

SEED = 42
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_COLUMN]


def _preprocessor(scale: bool):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=300,
                                stop_words="english"), TEXT_COLUMN),
    ], sparse_threshold=0)  # dense output (required by HistGradientBoosting)


def build_models() -> dict:
    """Baseline + three candidate models (same features for all)."""
    return {
        "Baseline (majority class)": Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", DummyClassifier(strategy="prior"))]),
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor(True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]),
        "Random Forest": Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                           class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                                   max_iter=250, class_weight="balanced",
                                                   random_state=SEED))]),
    }


def best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[np.argmax(f1)])


def evaluate(name, model, X_test, y_test, threshold=0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "model": name, "threshold": round(threshold, 3),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def cross_validate_threshold(model, X_train, y_train) -> float:
    """Pick the F1-optimal threshold using out-of-fold predictions (no test leakage)."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    return best_f1_threshold(y_train, oof)


# ---------------- Risk alerts ----------------
RISK_RULES = {
    "Claim amount > 60% of coverage": lambda r: r["amount_to_coverage"] > 0.6,
    "New customer (< 12 months)": lambda r: r["new_customer"] == 1,
    "Reported > 14 days after incident": lambda r: r["late_report"] == 1,
    "Coverage changed recently": lambda r: r["coverage_changed_recently"] == 1,
    "No police report": lambda r: r["police_report_filed"] == 0,
    "No witnesses": lambda r: r["no_witness"] == 1,
    "Incident at night (23:00-04:59)": lambda r: r["night_incident"] == 1,
    "Multiple previous claims (>=3)": lambda r: r["previous_claims"] >= 3,
    "Suspicious wording in description": lambda r: r["text_flag_count"] >= 2,
}


def risk_level(p: float, threshold: float) -> str:
    """HIGH = flagged as fraud; MEDIUM = watch-list (>= half the threshold); else LOW."""
    if p >= threshold:
        return "HIGH"
    if p >= threshold / 2:
        return "MEDIUM"
    return "LOW"


def build_alerts(df: pd.DataFrame, proba, threshold: float) -> pd.DataFrame:
    """Attach fraud probability, risk level and human-readable reasons."""
    out = df[["claim_id", "customer_id", "claim_type", "claim_amount"]].copy()
    out["fraud_probability"] = np.round(proba, 4)
    out["risk_level"] = [risk_level(p, threshold) for p in proba]
    out["predicted_fraud"] = (proba >= threshold).astype(int)
    out["reasons"] = [
        "; ".join(k for k, f in RISK_RULES.items() if f(r)) or "No rule triggered"
        for _, r in df.iterrows()
    ]
    return out.sort_values("fraud_probability", ascending=False).reset_index(drop=True)
