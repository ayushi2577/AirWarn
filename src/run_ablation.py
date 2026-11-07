import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from config import ROOT, FOLDS
from splits import get_fold
from pipeline import load_model_table, model_columns, get_Xy, make_preprocessor
from features import group_of
from evaluate import pick_threshold, metrics_at

RESULTS = ROOT / "outputs" / "results"
DOCS_DATA = ROOT / "docs" / "data"

table = load_model_table()
all_cols = model_columns(table)


def group_for(c):
    if c == "City":
        return "city"
    return group_of(c)


groups = {}
for c in all_cols:
    groups.setdefault(group_for(c), []).append(c)
print({g: len(v) for g, v in groups.items()}, "\n")


def evaluate(cols):
    """Mean PR-AUC and mean recall at the pooled-validation 80%-recall threshold."""
    parts = []
    for i in range(len(FOLDS)):
        tr, va = get_fold(table, i)
        X_tr, y_tr = get_Xy(tr)
        X_va, y_va = get_Xy(va)
        m = Pipeline([("prep", make_preprocessor(cols)),
                      ("clf", LogisticRegression(C=1.0, max_iter=2000))])
        m.fit(X_tr[cols], y_tr)
        parts.append((y_va.values, m.predict_proba(X_va[cols])[:, 1]))
    y_all = np.concatenate([p[0] for p in parts])
    s_all = np.concatenate([p[1] for p in parts])
    t = pick_threshold(y_all, s_all, 0.80)
    pr = [average_precision_score(y, s) for y, s in parts]
    rec = [metrics_at(y, (s >= t).astype(int))["recall"] for y, s in parts]
    prec = metrics_at(y_all, (s_all >= t).astype(int))["precision"]
    return float(np.mean(pr)), float(np.mean(rec)), float(prec)


rows = []
full = evaluate(all_cols)
rows.append({"setting": "all features", "mean_pr_auc": full[0],
             "mean_recall": full[1], "pooled_precision": full[2]})
for g, gcols in groups.items():
    keep = [c for c in all_cols if c not in gcols]
    r = evaluate(keep)
    rows.append({"setting": f"without {g}", "mean_pr_auc": r[0],
                 "mean_recall": r[1], "pooled_precision": r[2]})
hist = [c for c in all_cols if group_for(c) == "recent_history"]
r = evaluate(hist)
rows.append({"setting": "recent history only", "mean_pr_auc": r[0],
             "mean_recall": r[1], "pooled_precision": r[2]})

res = pd.DataFrame(rows)
res["d_pr_auc"] = res["mean_pr_auc"] - full[0]
res.to_csv(RESULTS / "ablation_validation.csv", index=False)
print(res.round(4).to_string(index=False))
json.dump(rows, open(DOCS_DATA / "ablation.json", "w"), indent=1)
print("\nSaved ablation csv and docs/data/ablation.json")