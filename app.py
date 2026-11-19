"""Streamlit dashboard.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.predict import load_model, score_claim

st.set_page_config(page_title="Insurance Fraud Detector", layout="wide")
st.title("Insurance Claim Fraud & Risk Prediction")
bundle = load_model()
tab1, tab2, tab3 = st.tabs(["Score a claim", "Risk alerts", "Model & patterns"])

with tab1:
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Customer age", 18, 90, 35)
    tenure = c1.number_input("Policy tenure (months)", 0, 240, 6)
    prev = c1.number_input("Previous claims", 0, 20, 1)
    ctype = c2.selectbox("Claim type", ["auto", "home", "health", "theft", "fire"])
    region = c2.selectbox("Region", ["North", "South", "East", "West", "Central"])
    coverage = c2.selectbox("Policy coverage", [50000, 100000, 250000, 500000], 1)
    amount = c2.number_input("Claim amount", 0, 1_000_000, 50000, step=1000)
    delay = c3.number_input("Report delay (days)", 0, 90, 3)
    hour = c3.slider("Incident hour", 0, 23, 14)
    witnesses = c3.number_input("Witnesses", 0, 10, 0)
    police = c3.checkbox("Police report filed", True)
    changed = c3.checkbox("Coverage changed recently", False)
    desc = st.text_area("Claim description", "Stolen overnight, no cctv, urgent settlement needed.")
    if st.button("Analyse claim"):
        r = score_claim(bundle, dict(
            customer_age=age, policy_tenure_months=tenure, previous_claims=prev, claim_type=ctype,
            region=region, policy_coverage=coverage, claim_amount=amount, report_delay_days=delay,
            police_report_filed="Yes" if police else "No", num_witnesses=witnesses,
            coverage_changed_recently="Yes" if changed else "No", incident_hour=hour,
            claim_description=desc))
        st.metric("Fraud probability", f"{r['fraud_probability']:.1%}")
        {"HIGH": st.error, "MEDIUM": st.warning, "LOW": st.success}[r["risk_level"]](f"Risk level: {r['risk_level']}")
        st.write("**Risk factors:**", "; ".join(r["reasons"]))

with tab2:
    alerts = pd.read_csv("outputs/fraud_predictions.csv")
    lvl = st.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"], ["HIGH"])
    st.dataframe(alerts[alerts["risk_level"].isin(lvl)], use_container_width=True)

with tab3:
    st.subheader("Model comparison")
    st.dataframe(pd.read_csv("outputs/model_comparison.csv"), use_container_width=True)
    st.subheader("Recurring fraud patterns")
    st.dataframe(pd.read_csv("outputs/fraud_patterns.csv"), use_container_width=True)
    for f in ["03_roc_pr_curves", "05_feature_importance", "06_fraud_patterns"]:
        st.image(f"outputs/figures/{f}.png")
