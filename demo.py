"""Command-line demo: input claim -> processing -> fraud prediction + risk alert.

Run:  python demo.py     (after `python main.py` has created the model)
"""
from src.predict import load_model, score_claim

SAMPLES = {
    "Suspicious theft claim": dict(
        customer_age=29, policy_tenure_months=4, previous_claims=3, claim_type="theft",
        region="East", policy_coverage=100000, claim_amount=82000, report_delay_days=21,
        police_report_filed="No", num_witnesses=0, coverage_changed_recently="Yes",
        incident_hour=2,
        claim_description="Stolen overnight, no cctv, urgent settlement needed. "
                          "Expensive jewellery missing, original bills unavailable. Requesting cash settlement immediately."),
    "Routine auto claim": dict(
        customer_age=45, policy_tenure_months=96, previous_claims=0, claim_type="auto",
        region="North", policy_coverage=250000, claim_amount=6000, report_delay_days=1,
        police_report_filed="Yes", num_witnesses=2, coverage_changed_recently="No",
        incident_hour=15,
        claim_description="Rear-ended while stopped, dashcam footage available. Police report filed and witnesses present."),
}

bundle = load_model()
print(f"Model: {bundle['name']} | alert threshold: {bundle['threshold']:.2f}\n")
for title, claim in SAMPLES.items():
    r = score_claim(bundle, claim)
    print(f"--- {title} ---")
    print(f"Input    : {claim['claim_type']} claim of {claim['claim_amount']:,} "
          f"(coverage {claim['policy_coverage']:,}), tenure {claim['policy_tenure_months']} months")
    print(f"Fraud probability : {r['fraud_probability']:.1%}")
    print(f"Risk alert        : {r['risk_level']}  (predicted fraud: {r['predicted_fraud']})")
    print("Risk factors      : " + "; ".join(r["reasons"]) + "\n")
