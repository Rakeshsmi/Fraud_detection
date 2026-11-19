"""All figures saved to outputs/figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

sns.set_theme(style="whitegrid")
FIG = "outputs/figures"


def _save(name):
    plt.tight_layout()
    plt.savefig(f"{FIG}/{name}.png", dpi=140)
    plt.close()


def eda_plots(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    df["is_fraud"].map({0: "Legit", 1: "Fraud"}).value_counts().plot.bar(
        ax=ax[0], color=["#4C72B0", "#C44E52"], rot=0)
    ax[0].set_title("Class distribution (imbalanced)")
    sns.boxplot(data=df.assign(cls=df["is_fraud"].map({0: "Legit", 1: "Fraud"})), x="cls", y="amount_to_coverage",
                hue="cls", palette=["#4C72B0", "#C44E52"], legend=False, ax=ax[1], order=["Legit", "Fraud"], hue_order=["Legit", "Fraud"])
    ax[1].set_title("Claim amount / coverage by class"); ax[1].set_xlabel("")
    df.groupby("claim_type")["is_fraud"].mean().sort_values().plot.barh(ax=ax[2], color="#DD8452")
    ax[2].set_title("Fraud rate by claim type")
    _save("01_eda_overview")

    cols = ["is_fraud", "amount_to_coverage", "policy_tenure_months", "previous_claims",
            "report_delay_days", "num_witnesses", "police_report_filed",
            "coverage_changed_recently", "night_incident", "text_flag_count"]
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature correlation heatmap")
    _save("02_correlation_heatmap")


def model_plots(models, X_test, y_test, best_name, threshold):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for name, m in models.items():
        if name.startswith("Baseline"):
            continue
        p = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[0].plot(fpr, tpr, label=name)
        ax[1].plot(rc, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", label="Baseline")
    ax[0].set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].axhline(y_test.mean(), color="k", ls="--", label="Baseline")
    ax[1].set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
    ax[0].legend(); ax[1].legend()
    _save("03_roc_pr_curves")

    p = models[best_name].predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, (p >= threshold).astype(int))
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion matrix - {best_name}\n(threshold={threshold:.2f})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    _save("04_confusion_matrix")


def importance_plot(model, X_test, y_test, feature_cols, top=15):
    r = permutation_importance(model, X_test, y_test, scoring="average_precision",
                               n_repeats=5, random_state=42, n_jobs=-1)
    idx = np.argsort(r.importances_mean)[-top:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_cols)[idx], r.importances_mean[idx],
             xerr=r.importances_std[idx], color="#55A868")
    plt.title("Permutation feature importance (PR-AUC drop)")
    _save("05_feature_importance")
    return dict(zip(np.array(feature_cols)[idx][::-1], r.importances_mean[idx][::-1].round(4)))


def pattern_plot(patterns):
    if patterns.empty:
        return
    d = patterns.head(10).iloc[::-1]
    plt.figure(figsize=(10, 5))
    plt.barh(d["pattern"], d["lift"], color="#8172B2")
    plt.axvline(1, color="k", ls="--")
    plt.title("Top recurring fraud patterns (lift vs. base fraud rate)")
    plt.xlabel("Lift")
    _save("06_fraud_patterns")


def risk_plot(alerts):
    order = ["LOW", "MEDIUM", "HIGH"]
    c = alerts["risk_level"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c.plot.bar(ax=ax[0], color=["#55A868", "#DD8452", "#C44E52"], rot=0)
    ax[0].set_title("Risk alert levels (test set)")
    sns.histplot(alerts["fraud_probability"], bins=30, ax=ax[1], color="#4C72B0")
    ax[1].set_title("Predicted fraud probability")
    _save("07_risk_alerts")
