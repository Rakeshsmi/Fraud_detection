"""Cleaning + feature engineering for structured and text data."""
import numpy as np
import pandas as pd

TEXT_RED_FLAGS = {
    "kw_no_witness": ["no witnesses"],
    "kw_cash": ["cash settlement"],
    "kw_urgent": ["urgent", "immediately"],
    "kw_missing_docs": ["receipts lost", "bills unavailable", "unable to provide",
                        "records not available", "will submit later"],
    "kw_total_loss": ["total loss", "completely destroyed"],
    "kw_night": ["late at night", "overnight", "nobody was home"],
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix inconsistent values, remove duplicates, impute missing data."""
    df = df.copy()
    df = df.drop_duplicates(subset="claim_id").reset_index(drop=True)

    # Standardise categoricals
    df["claim_type"] = df["claim_type"].astype(str).str.strip().str.lower()
    for col in ["police_report_filed", "coverage_changed_recently"]:
        df[col] = df[col].astype(str).str.strip().str.lower().map(
            {"yes": 1, "y": 1, "no": 0, "n": 0}).fillna(0).astype(int)
    df["region"] = df["region"].fillna("Unknown")

    # Invalid values -> NaN -> impute
    df.loc[df["claim_amount"] <= 0, "claim_amount"] = np.nan
    df.loc[~df["customer_age"].between(18, 90), "customer_age"] = np.nan
    df.loc[df["report_delay_days"] < 0, "report_delay_days"] = np.nan

    # Flag missingness (can itself be a signal), then median-impute
    for col in ["customer_age", "claim_amount", "report_delay_days"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

    df["claim_description"] = df["claim_description"].fillna("").str.lower()
    df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain features (ratios, history, time, text flags)."""
    df = df.copy()
    df["amount_to_coverage"] = df["claim_amount"] / df["policy_coverage"]
    df["log_claim_amount"] = np.log1p(df["claim_amount"])
    df["new_customer"] = (df["policy_tenure_months"] < 12).astype(int)
    df["late_report"] = (df["report_delay_days"] > 14).astype(int)
    df["night_incident"] = ((df["incident_hour"] >= 23) | (df["incident_hour"] <= 4)).astype(int)
    df["no_witness"] = (df["num_witnesses"] == 0).astype(int)
    df["high_amount_ratio"] = (df["amount_to_coverage"] > 0.6).astype(int)
    df["claims_per_year"] = df["previous_claims"] / (df["policy_tenure_months"] / 12).clip(lower=0.5)
    df["month"] = df["incident_date"].dt.month.fillna(0).astype(int)
    df["weekday"] = df["incident_date"].dt.weekday.fillna(0).astype(int)

    # customer history: number of claims by this customer in the dataset
    df["customer_claim_count"] = df.groupby("customer_id")["claim_id"].transform("count")

    # text features
    df["desc_length"] = df["claim_description"].str.split().str.len()
    df["desc_missing"] = (df["claim_description"] == "").astype(int)
    for name, kws in TEXT_RED_FLAGS.items():
        df[name] = df["claim_description"].apply(lambda t: int(any(k in t for k in kws)))
    df["text_flag_count"] = df[list(TEXT_RED_FLAGS)].sum(axis=1)
    return df


NUMERIC_FEATURES = [
    "customer_age", "policy_tenure_months", "previous_claims", "policy_coverage",
    "claim_amount", "log_claim_amount", "report_delay_days", "num_witnesses",
    "incident_hour", "police_report_filed", "coverage_changed_recently",
    "amount_to_coverage", "new_customer", "late_report", "night_incident",
    "no_witness", "claims_per_year", "month", "weekday", "customer_claim_count",
    "desc_length", "desc_missing", "text_flag_count",
    "customer_age_missing", "claim_amount_missing", "report_delay_days_missing",
] + list(TEXT_RED_FLAGS)
CATEGORICAL_FEATURES = ["claim_type", "region"]
TEXT_COLUMN = "claim_description"
TARGET = "is_fraud"
