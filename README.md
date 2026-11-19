# Intelligent Insurance Claim Fraud & Risk Prediction

## 1. Problem Statement
Insurance companies receive thousands of claims and can only investigate a few. This project builds a
machine-learning application that analyses claim records, customer history, claim amounts and free-text
claim descriptions to **flag potentially fraudulent claims**, **estimate fraud probability / risk level**, and
**mine recurring patterns** that characterise suspicious claims.

## 2. Objectives
- Clean and engineer features from structured *and* text data.
- Train and compare classification models against a baseline on an imbalanced target (~6% fraud).
- Produce fraud predictions with explainable **risk alerts** (LOW / MEDIUM / HIGH + reason codes).
- Use **pattern mining (Apriori association rules)** to find recurring suspicious claim characteristics.
- Provide a working demo (CLI + Streamlit dashboard).

## 3. Dataset (documented synthetic)
Real claim data is confidential, so `src/data_generation.py` generates a reproducible synthetic dataset
(`seed=42`, saved to `data/insurance_claims.csv`, 6,040 rows incl. 40 duplicates).

| Group | Columns |
|---|---|
| Customer / history | `customer_id, customer_age, policy_tenure_months, previous_claims, region` |
| Policy / claim | `claim_type, policy_coverage, claim_amount, coverage_changed_recently` |
| Incident | `incident_date, incident_hour, report_delay_days, police_report_filed, num_witnesses` |
| Text | `claim_description` (free text) |
| Target | `is_fraud` (1 = fraud) |

Fraud is generated from a hidden logistic function of several interacting risk factors (high amount/coverage
ratio, new customer, late reporting, no police report, night incident, etc.) **plus noise**, and text
descriptions mix fraud-style, legit-style and neutral phrases. Data-quality problems are deliberately injected:
missing values, negative amounts, impossible ages, inconsistent casing (`"AUTO "`), `Y/n` vs `Yes/No`, duplicates.

> Limitation: because the data is synthetic, absolute scores (especially the strong text signal) are optimistic
> compared with real insurance data. To use a real dataset (e.g. Kaggle *Auto Insurance Claims*), map its columns to
> the schema above and drop the CSV into `data/`.

## 4. Data Processing (`src/preprocessing.py`)
- Remove duplicate claim IDs; standardise categorical text (case, whitespace, Y/N -> 1/0).
- Treat invalid values (amount <= 0, age outside 18-90) as missing; add `*_missing` indicator flags;
  median-impute numeric gaps; fill missing region with `Unknown` and missing text with empty string.
- **Engineered features:** amount-to-coverage ratio, log amount, new-customer flag, late-report flag,
  night-incident flag, no-witness flag, claims per year, customer claim count (history), month/weekday,
  description length, **text red-flag keywords** (cash demand, urgency, missing documents, total loss, ...),
  plus **TF-IDF (1-2 grams)** of the description inside the model pipeline.

## 5. Algorithms
| Component | Technique |
|---|---|
| Baseline | Majority-class `DummyClassifier` |
| Predictive models | Logistic Regression, Random Forest, Histogram Gradient Boosting (class-imbalance weighted) |
| Text | TF-IDF + keyword flags |
| Threshold | F1-optimal threshold chosen from 5-fold out-of-fold predictions on the training set (no test leakage) |
| Pattern mining | Apriori (own implementation) on binary claim "items"; ranked by **lift** = P(fraud ∣ pattern) / P(fraud) |
| Explainability | Rule-based reason codes + permutation feature importance |

Risk levels: **HIGH** = probability >= tuned threshold (flagged as fraud), **MEDIUM** = >= half the threshold (watch-list), **LOW** otherwise.

## 6. Installation
```bash
git clone <your-repo-url> && cd fraud-detection
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Execution
```bash
python main.py                 # generate data -> preprocess -> train -> evaluate -> patterns -> alerts
python demo.py                 # CLI demo scoring two sample claims
streamlit run app.py           # interactive dashboard (score a claim, browse alerts, view results)
jupyter notebook notebooks/demo.ipynb   # notebook demonstration (input -> processing -> output)
```

### Project structure
```
main.py  demo.py  app.py  requirements.txt  notebooks/demo.ipynb
src/   data_generation.py preprocessing.py modeling.py pattern_mining.py visualize.py predict.py
data/insurance_claims.csv
models/best_model.joblib
outputs/  model_comparison.csv  fraud_predictions.csv  high_risk_alerts.csv  fraud_patterns.csv  summary.json  figures/
```

## 8. Sample Results (held-out 20% test set, stratified)
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (majority) | 0.500 | 0.061 | 0.000 | 0.000 | 0.000 |
| Logistic Regression | 0.955 | 0.610 | 0.639 | 0.630 | 0.634 |
| Random Forest | 0.952 | 0.589 | 0.423 | 0.712 | 0.531 |
| **Gradient Boosting** | **0.961** | **0.615** | 0.543 | 0.699 | 0.611 |

PR-AUC and recall/precision are emphasised over accuracy because only ~6% of claims are fraudulent
(a model predicting "never fraud" is 94% accurate but useless). Best model flags 94 of 1,200 test claims as HIGH risk,
catching ~70% of fraud (51 of 73) with 43 false alarms.

**Top recurring fraud patterns (Apriori, by lift):**
| Pattern | Fraud rate when present | Lift |
|---|---|---|
| missing documents wording + cash/urgent demand | 75.9% | 12.5x |
| coverage changed recently + missing documents wording | 48.5% | 8.0x |
| no police report + no witnesses + missing documents wording | 47.3% | 7.8x |

(Exact numbers are in `outputs/`; they are reproducible via the fixed seed.)

### Visualizations (`outputs/figures/`)
1. `01_eda_overview.png` - class imbalance, amount ratio by class, fraud rate by claim type
2. `02_correlation_heatmap.png` - feature correlations
3. `03_roc_pr_curves.png` - ROC and precision-recall comparison vs baseline
4. `04_confusion_matrix.png` - best model at tuned threshold
5. `05_feature_importance.png` - permutation importance
6. `06_fraud_patterns.png` - top patterns by lift
7. `07_risk_alerts.png` - alert level counts and probability distribution

## 9. Limitations & Future Work
Synthetic data; no temporal validation split; customer-history features use counts within the dataset only.
Future work: real data, SHAP explanations, graph analysis of linked claimants/repair shops, cost-sensitive
thresholds (investigation cost vs. fraud loss), sentence-embedding text features.

## 10. Team
_Add team members and roles here._

## 11. Submitting via GitHub
```bash
git init && git add . && git commit -m "Insurance fraud detection project"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
