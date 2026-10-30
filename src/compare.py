import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from models import (load_model_table, model_columns, make_logreg, make_rbf_svm,
                    validation_scores, svm_validation_scores, tune,
                    summarize)
from evaluate import pick_threshold, metrics_at

table = load_model_table()
cols = model_columns(table)

# 1. Widen the RBF grid once (the previous best sat on the grid edge)
print("Widened RBF grid...")
grid = [{"C": c, "gamma": g} for c in [4.0, 16.0, 64.0] for g in [0.001, 0.003]]
rbf = tune(table, make_rbf_svm, cols, grid)
print(rbf.round(4).to_string(index=False), "\n")
best = rbf.iloc[0]
best_params = {"C": float(best["C"]), "gamma": float(best["gamma"])}
print("Best RBF:", best_params, "\n")

# 2. Per-fold PR-AUC and precision at 80% recall: LR vs best RBF SVM
lr_val = validation_scores(table, make_logreg, cols)
svm_val = svm_validation_scores(
    table, lambda c: make_rbf_svm(c, **best_params), cols)


def per_fold(val):
    out = {}
    for f, g in val.groupby("fold"):
        y = g["poor_tomorrow"].astype(int).values
        s = g["score"].values
        t = pick_threshold(y, s, 0.80)
        out[f] = (average_precision_score(y, s),
                  metrics_at(y, (s >= t).astype(int))["precision"])
    return out


lr_f, svm_f = per_fold(lr_val), per_fold(svm_val)
rows = []
for f in sorted(lr_f):
    rows.append({"fold": f,
                 "LR_prauc": lr_f[f][0], "SVM_prauc": svm_f[f][0],
                 "diff_prauc(LR-SVM)": lr_f[f][0] - svm_f[f][0],
                 "LR_p@80r": lr_f[f][1], "SVM_p@80r": svm_f[f][1],
                 "diff_p@80r(LR-SVM)": lr_f[f][1] - svm_f[f][1]})
cmp = pd.DataFrame(rows)
print(cmp.round(4).to_string(index=False), "\n")

# 3. Apply the model-choice rule written before results
for col in ["diff_prauc(LR-SVM)", "diff_p@80r(LR-SVM)"]:
    d = cmp[col].values
    mean, spread = d.mean(), d.std(ddof=1)
    svm_wins = int((d < 0).sum())
    print(f"{col}: mean {mean:+.4f}, spread (std) {spread:.4f}, "
          f"SVM better in {svm_wins}/4 folds")
    clearly = svm_wins >= 3 and abs(mean) > spread and mean < 0
    print("  ->", "choose SVM" if clearly else "choose logistic regression", "\n")

# 4. Platt scaling: logistic curve fitted on POOLED validation SVM scores only
platt = LogisticRegression(C=1e6)
platt.fit(svm_val[["score"]].values, svm_val["poor_tomorrow"].astype(int).values)
svm_val["platt_prob"] = platt.predict_proba(svm_val[["score"]].values)[:, 1]
print("Platt scaling fitted. Example probabilities:",
      svm_val["platt_prob"].describe().round(3).to_dict())