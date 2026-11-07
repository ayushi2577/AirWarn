import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from config import ROOT, AQI_CUTOFF
from models import (load_model_table, model_columns, make_logreg, make_rbf_svm,
                    validation_scores, svm_validation_scores)
from evaluate import metrics_at, pick_threshold, full_metrics

RESULTS = ROOT / "outputs" / "results"
FIGS = ROOT / "outputs" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

SVM_PARAMS = {"C": 4.0, "gamma": 0.003}   # settled in compare.py

table = load_model_table()
cols = model_columns(table)

# 1. Validation scores (4 folds stacked)
lr_val = validation_scores(table, make_logreg, cols)
svm_val = svm_validation_scores(
    table, lambda c: make_rbf_svm(c, **SVM_PARAMS), cols)

# Platt scaling, fitted on POOLED validation scores only
platt = LogisticRegression(C=1e6)
platt.fit(svm_val[["score"]].values, svm_val["poor_tomorrow"].astype(int).values)
svm_val["prob"] = platt.predict_proba(svm_val[["score"]].values)[:, 1]
lr_val["prob"] = lr_val["score"]

y = lr_val["poor_tomorrow"].astype(int).values
models = {"Logistic regression": lr_val["prob"].values,
          "RBF SVM (Platt)": svm_val["prob"].values}

# Persistence = a single yes/no point
pers_pred = (lr_val["AQI"] > AQI_CUTOFF).astype(int).values
pers = metrics_at(y, pers_pred)

# 2. Threshold sweep
thresholds = np.round(np.arange(0.05, 0.9501, 0.01), 2)
rows = []
for name, s in models.items():
    for t in thresholds:
        m = metrics_at(y, (s >= t).astype(int))
        rows.append({"model": name, "threshold": t, **m})
sweep = pd.DataFrame(rows)
sweep.to_csv(RESULTS / "threshold_sweep_validation.csv", index=False)

# 3. Plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in models:
    d = sweep[sweep["model"] == name]
    axes[0].plot(d["threshold"], d["precision"], label=f"{name} precision")
    axes[0].plot(d["threshold"], d["recall"], "--", label=f"{name} recall")
axes[0].set_xlabel("threshold"); axes[0].set_ylabel("value")
axes[0].set_title("Precision and recall vs threshold (validation)")
axes[0].legend(fontsize=7)

for name, s in models.items():
    p, r, _ = precision_recall_curve(y, s)
    axes[1].plot(r, p, label=name)
axes[1].scatter([pers["recall"]], [pers["precision"]], color="black",
                zorder=5, label="Persistence (single point)")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].set_title("Precision-recall curve (validation)")
axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIGS / "threshold_and_pr_validation.png", dpi=150)

# 4. Freeze each model's threshold (highest precision with recall >= 80%)
frozen = {}
summary = []
for name, s in models.items():
    t = pick_threshold(y, s, 0.80)
    frozen[name] = round(t, 4)
    m = full_metrics(y, s, (s >= t).astype(int))
    summary.append({"model": name, "threshold": round(t, 4),
                    "precision": m["precision"], "recall": m["recall"],
                    "f1": m["f1"], "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"]})
summary.append({"model": "Persistence", "threshold": "AQI>200",
                "precision": pers["precision"], "recall": pers["recall"],
                "f1": pers["f1"], "pr_auc": np.nan, "roc_auc": np.nan})
summary = pd.DataFrame(summary)
summary.to_csv(RESULTS / "frozen_summary_validation.csv", index=False)

with open(RESULTS / "frozen_settings.json", "w") as f:
    json.dump({"thresholds": frozen,
               "svm_params": SVM_PARAMS,
               "platt_coef": float(platt.coef_[0][0]),
               "platt_intercept": float(platt.intercept_[0])}, f, indent=2)

print(summary.round(3).to_string(index=False))
print("\nSaved: outputs/results/ and outputs/figures/")

def block_bootstrap(rows, y, score, pred, n_boot=500, seed=0):
    """Resample whole calendar weeks (with replacement) and recompute metrics.
    rows needs a 'Date' column. Returns {metric: (low, high)} 95% intervals."""
    rng = np.random.default_rng(seed)
    week = rows["Date"].dt.to_period("W").astype(str).values
    y, score, pred = np.asarray(y), np.asarray(score), np.asarray(pred)
    idx_by_week = {w: np.where(week == w)[0] for w in np.unique(week)}
    weeks = list(idx_by_week)
    out = {"precision": [], "recall": [], "pr_auc": []}
    for _ in range(n_boot):
        pick = rng.choice(len(weeks), size=len(weeks), replace=True)
        idx = np.concatenate([idx_by_week[weeks[k]] for k in pick])
        if y[idx].sum() == 0:
            continue
        m = metrics_at(y[idx], pred[idx])
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
        out["pr_auc"].append(average_precision_score(y[idx], score[idx]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items()}