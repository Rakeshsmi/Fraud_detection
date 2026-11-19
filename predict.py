"""Score a single new claim with the saved model (used by demo.py and app.py)."""
import joblib
import pandas as pd

from .modeling import FEATURE_COLS, RISK_RULES, risk_level
from .preprocessing import clean, engineer_features


def load_model(path="models/best_model.joblib"):
    return joblib.load(path)


def score_claim(bundle: dict, claim: dict) -> dict:
    raw = pd.DataFrame([{
        "claim_id": claim.get("claim_id", "NEW"), "customer_id": claim.get("customer_id", 0),
        "incident_date": claim.get("incident_date", "2024-01-01"), **claim}])
    df = engineer_features(clean(raw))
    p = float(bundle["model"].predict_proba(df[FEATURE_COLS])[:, 1][0])
    row = df.iloc[0]
    reasons = [k for k, f in RISK_RULES.items() if f(row)]
    return {"fraud_probability": round(p, 4),
            "risk_level": risk_level(p, bundle["threshold"]),
            "predicted_fraud": p >= bundle["threshold"],
            "reasons": reasons or ["No rule triggered"]}
