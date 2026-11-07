import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from config import ROOT, FOLDS
from splits import get_fold
from pipeline import (load_model_table, model_columns, get_Xy, make_preprocessor)
from features import group_of

RESULTS = ROOT / "outputs" / "results"
DOCS_DATA = ROOT / "docs" / "data"
DOCS_DATA.mkdir(parents=True, exist_ok=True)

table = load_model_table()
cols = model_columns(table)


def make(penalty, C):
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    return Pipeline([
        ("prep", make_preprocessor(cols)),
        ("clf", LogisticRegression(l1_ratio=l1_ratio, C=C, solver="saga",
                                   max_iter=5000)),
    ])


def mean_prauc(penalty, C):
    scores = []
    for i in range(len(FOLDS)):
        tr, va = get_fold(table, i)
        X_tr, y_tr = get_Xy(tr)
        X_va, y_va = get_Xy(va)
        m = make(penalty, C).fit(X_tr[cols], y_tr)
        scores.append(average_precision_score(y_va, m.predict_proba(X_va[cols])[:, 1]))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


# 1. Choose the penalty strength on the 4 forward-chaining folds
rows = []
for penalty in ["l2", "l1"]:
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        mean, sd = mean_prauc(penalty, C)
        rows.append({"penalty": penalty, "C": C, "mean_pr_auc": mean, "fold_std": sd})
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "regularization_validation.csv", index=False)
print(res.round(4).to_string(index=False))

best = res.sort_values("mean_pr_auc", ascending=False).groupby("penalty").head(1)
print("\nBest per penalty:")
print(best.round(4).to_string(index=False))

# 2. Which features does Lasso set to zero? Fit on ALL training years (2015-2018)
#    with the best L1 strength. No validation or test rows involved.
best_l1_C = float(best[best["penalty"] == "l1"]["C"].iloc[0])
train_rows = pd.concat([get_fold(table, len(FOLDS) - 1)[0],
                        get_fold(table, len(FOLDS) - 1)[1]])
X, y = get_Xy(train_rows)
model = make("l1", best_l1_C).fit(X[cols], y)
names = model.named_steps["prep"].get_feature_names_out()
coef = model.named_steps["clf"].coef_[0]


def group_for(n):
    if n.startswith("City_"):
        return "city"
    for base in ["month", "dayofweek", "season"]:
        if n.startswith(base + "_"):
            return "calendar"
    return group_of(n)


coefs = pd.DataFrame({"feature": names, "coef": coef,
                      "group": [group_for(n) for n in names]})
coefs["zero"] = coefs["coef"] == 0
coefs.sort_values("coef", key=abs, ascending=False).to_csv(
    RESULTS / "lasso_coefficients.csv", index=False)

print(f"\nLasso (C={best_l1_C}) kept {int((~coefs['zero']).sum())} of {len(coefs)} columns")
print(coefs.groupby("group")["zero"].agg(zeroed="sum", total="count").to_string())
print("\nTop 10 features by size:")
print(coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
      .head(10)[["feature", "coef", "group"]].round(3).to_string(index=False))

# 3. Chart data for the website
json.dump({"curve": rows,
           "lasso_by_group": coefs.groupby("group")["zero"].agg(
               zeroed="sum", total="count").reset_index().to_dict("records")},
          open(DOCS_DATA / "regularization.json", "w"), indent=1)
print("\nSaved regularization csvs and docs/data/regularization.json")