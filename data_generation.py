"""Documented synthetic insurance-claim dataset generator.

Why synthetic? Real claim data is confidential. This generator creates realistic
structured fields + free-text descriptions, with fraud driven by several
interacting risk factors (plus noise), and deliberately injects missing and
inconsistent values so the preprocessing step is meaningful.
"""
import numpy as np
import pandas as pd

SEED = 42

CLAIM_TYPES = ["auto", "home", "health", "theft", "fire"]
REGIONS = ["North", "South", "East", "West", "Central"]

LEGIT_PHRASES = [
    "minor collision at traffic signal, police report filed",
    "rear-ended while stopped, dashcam footage available",
    "water pipe burst, plumber invoice attached",
    "routine hospital treatment, medical bills enclosed",
    "windshield cracked by road debris, repair estimate provided",
    "witnesses present and contact details shared",
    "damage assessed by authorised surveyor",
    "claim reported promptly after the incident",
    "fender damage in parking lot, CCTV available",
    "storm damage to roof, photos attached",
]
FRAUD_PHRASES = [
    "total loss, vehicle completely destroyed, no witnesses",
    "stolen overnight, no cctv, urgent settlement needed",
    "fire broke out while nobody was home, receipts lost",
    "expensive jewellery and electronics missing, original bills unavailable",
    "requesting cash settlement immediately",
    "unable to provide documents, will submit later",
    "accident happened late at night on an empty road",
    "multiple injuries claimed, hospital records not available",
    "recently increased coverage before the incident",
    "claimant insists on a specific repair shop",
]
NEUTRAL_PHRASES = [
    "incident occurred as described in the form",
    "please process the claim at the earliest",
    "damage was noticed in the morning",
    "vehicle was parked outside the house",
    "details of the event are attached",
]


def _description(is_fraud: bool, rng) -> str:
    # Signal is noisy: some frauds look legit and vice versa.
    p_fraud_phrase = 0.65 if is_fraud else 0.10
    parts = []
    for _ in range(rng.integers(2, 4)):
        r = rng.random()
        if r < p_fraud_phrase:
            parts.append(rng.choice(FRAUD_PHRASES))
        elif r < p_fraud_phrase + 0.45:
            parts.append(rng.choice(LEGIT_PHRASES))
        else:
            parts.append(rng.choice(NEUTRAL_PHRASES))
    return ". ".join(parts).capitalize() + "."


def generate_claims(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = rng.integers(1000, 1000 + n // 2, n)  # repeat customers
    age = np.clip(rng.normal(40, 12, n), 18, 80).round()
    tenure = rng.integers(1, 180, n)  # months with insurer
    prev_claims = rng.poisson(1.0, n)
    claim_type = rng.choice(CLAIM_TYPES, n, p=[0.35, 0.2, 0.2, 0.15, 0.1])
    region = rng.choice(REGIONS, n)
    coverage = rng.choice([50_000, 100_000, 250_000, 500_000], n, p=[.3, .35, .25, .1])
    claim_amount = np.clip(rng.lognormal(9.3, 0.9, n), 500, None)
    claim_amount = np.minimum(claim_amount, coverage * rng.uniform(0.3, 1.0, n))
    report_delay = rng.exponential(5, n).round()
    police_report = rng.random(n) < 0.6
    witnesses = rng.poisson(1.2, n)
    coverage_changed_recently = rng.random(n) < 0.12
    incident_hour = rng.integers(0, 24, n)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")

    # Hidden fraud logit built from several interacting risk factors
    logit = (
        -4.2
        + 1.1 * (claim_amount / coverage > 0.6)
        + 0.9 * (tenure < 12)
        + 0.35 * np.clip(prev_claims - 1, 0, 4)
        + 0.8 * (report_delay > 14)
        + 0.9 * coverage_changed_recently
        + 0.7 * (~police_report)
        + 0.6 * (witnesses == 0)
        + 0.6 * ((incident_hour >= 23) | (incident_hour <= 4))
        + 0.5 * np.isin(claim_type, ["theft", "fire"])
        + rng.normal(0, 0.6, n)
    )
    is_fraud = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "claim_id": [f"CLM{100000 + i}" for i in range(n)],
        "customer_id": customer_id,
        "customer_age": age,
        "policy_tenure_months": tenure,
        "previous_claims": prev_claims,
        "claim_type": claim_type,
        "region": region,
        "policy_coverage": coverage,
        "claim_amount": claim_amount.round(2),
        "report_delay_days": report_delay,
        "police_report_filed": np.where(police_report, "Yes", "No"),
        "num_witnesses": witnesses,
        "coverage_changed_recently": np.where(coverage_changed_recently, "Yes", "No"),
        "incident_hour": incident_hour,
        "incident_date": dates.strftime("%Y-%m-%d"),
        "claim_description": [_description(f, rng) for f in is_fraud],
        "is_fraud": is_fraud.astype(int),
    })

    # ---- Inject realistic data-quality problems ----
    m = lambda p: rng.random(n) < p
    df.loc[m(0.04), "customer_age"] = np.nan
    df.loc[m(0.03), "claim_amount"] = np.nan
    df.loc[m(0.05), "report_delay_days"] = np.nan
    df.loc[m(0.03), "claim_description"] = np.nan
    df.loc[m(0.02), "region"] = np.nan
    neg = m(0.01)
    df.loc[neg, "claim_amount"] = -df.loc[neg, "claim_amount"].abs()   # invalid negatives
    df.loc[m(0.01), "customer_age"] = 150                              # impossible age
    messy = m(0.05)
    df.loc[messy, "claim_type"] = df.loc[messy, "claim_type"].str.upper() + " "  # casing/spaces
    df.loc[m(0.03), "police_report_filed"] = "Y"
    df.loc[m(0.03), "police_report_filed"] = "n"
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = generate_claims()
    out.to_csv("data/insurance_claims.csv", index=False)
    print(out.shape, "fraud rate: %.3f" % out["is_fraud"].mean())
