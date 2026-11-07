import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from config import ROOT, AQI_CUTOFF, TRAIN
from splits import get_split, between, get_eval_rows, seen_cities
from pipeline import load_model_table, model_columns, get_Xy
from models import make_logreg, make_rbf_svm
from evaluate import metrics_at, change_day_mask, block_bootstrap

RESULTS = ROOT / "outputs" / "results"
DOCS_DATA = ROOT / "docs" / "data"
cfg = json.load(open(RESULTS / "frozen_settings.json"))
T = cfg["thresholds"]

table = load_model_table()
cols = model_columns(table)
seen = seen_cities(table)

# Refit on ALL training years (2015-2018 minus the embargo day), frozen settings
train = between(get_eval_rows(table), *TRAIN)
X_tr, y_tr = get_Xy(train)
lr = make_logreg(cols).fit(X_tr[cols], y_tr)
svm = make_rbf_svm(cols, **cfg["svm_params"]).fit(X_tr[cols], y_tr)


def score_rows(rows):
    X, _ = get_Xy(rows)
    s_lr = lr.predict_proba(X[cols])[:, 1]
    d = svm.decision_function(X[cols])
    s_svm = 1 / (1 + np.exp(-(cfg["platt_coef"] * d + cfg["platt_intercept"])))
    return s_lr, s_svm


def report(name, rows):
    rows = rows.reset_index(drop=True)
    y = rows["poor_tomorrow"].astype(int).values
    s_lr, s_svm = score_rows(rows)
    pers = (rows["AQI"] > AQI_CUTOFF).astype(int).values
    chg = change_day_mask(rows)
    out = []
    for model, s, pred in [
        ("Logistic regression", s_lr, (s_lr >= T["Logistic regression"]).astype(int)),
        ("RBF SVM (Platt)", s_svm, (s_svm >= T["RBF SVM (Platt)"]).astype(int)),
        ("Persistence", pers.astype(float), pers),
    ]:
        m = metrics_at(y, pred)
        m["pr_auc"] = average_precision_score(y, s) if y.sum() else np.nan
        m["roc_auc"] = roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
        m["change_day_recall"] = metrics_at(y[chg], pred[chg])["recall"] if chg.any() else np.nan
        if y.sum() >= 30:
            ci = block_bootstrap(rows, y, s, pred)
            m["precision_CI"] = f"{ci['precision'][0]:.3f} to {ci['precision'][1]:.3f}"
            m["recall_CI"] = f"{ci['recall'][0]:.3f} to {ci['recall'][1]:.3f}"
        out.append({"part": name, "model": model, "rows": len(rows),
                    "positives": int(y.sum()), **m})
    return out, rows, s_lr, s_svm, pers


results = []
test = get_split(table, "test")
for label, subset in [("2019 all cities", test),
                      ("2019 SEEN cities (headline)", test[test["City"].isin(seen)]),
                      ("2019 NEW cities", test[~test["City"].isin(seen)])]:
    results += report(label, subset)[0]

for key, label in [("pre_lockdown", "2020 pre-lockdown"), ("lockdown", "2020 lockdown")]:
    s2020 = get_split(table, key)
    results += report(label + " seen", s2020[s2020["City"].isin(seen)])[0]
    results += report(label + " new", s2020[~s2020["City"].isin(seen)])[0]

res = pd.DataFrame(results)
res.to_csv(RESULTS / "final_test_results.csv", index=False)
show = ["part", "model", "rows", "positives", "precision", "recall", "f1",
        "pr_auc", "change_day_recall", "precision_CI", "recall_CI"]
print(res[show].round(3).to_string(index=False))

# Per-city 2019 (seen cities), logistic regression
t_seen = test[test["City"].isin(seen)].reset_index(drop=True)
y = t_seen["poor_tomorrow"].astype(int).values
s_lr, s_svm = score_rows(t_seen)
t_seen["pred"] = (s_lr >= T["Logistic regression"]).astype(int)
city_rows = []
for c, g in t_seen.groupby("City"):
    yy = g["poor_tomorrow"].astype(int).values
    m = metrics_at(yy, g["pred"].values)
    city_rows.append({"city": c, "rows": len(g), "positives": int(yy.sum()),
                      "precision": m["precision"], "recall": m["recall"],
                      "reliable": int(yy.sum()) >= 30})
city = pd.DataFrame(city_rows).sort_values("positives", ascending=False)
city.to_csv(RESULTS / "per_city_test_2019.csv", index=False)
print("\nPer-city 2019 (seen cities, LR):")
print(city.round(3).to_string(index=False))

# Add the test view to the slider data (exploration only)
sweep = {}
for name, s in [("Logistic regression", s_lr), ("RBF SVM (Platt)", s_svm)]:
    rows = []
    for t in np.round(np.arange(0.05, 0.9501, 0.01), 2):
        m = metrics_at(y, (s >= t).astype(int))
        rows.append({"threshold": float(t), **{k: m[k] for k in
                     ["TP", "FP", "FN", "TN", "precision", "recall"]}})
    sweep[name] = rows
tj = json.load(open(DOCS_DATA / "thresholds.json"))
tj["test_2019_seen"] = sweep
pm = metrics_at(y, (t_seen["AQI"] > AQI_CUTOFF).astype(int).values)
tj["persistence_test_2019_seen"] = {k: pm[k] for k in ["TP", "FP", "FN", "TN", "precision", "recall"]}
json.dump(tj, open(DOCS_DATA / "thresholds.json", "w"), indent=1)
print("\nSaved final_test_results.csv, per_city_test_2019.csv and updated thresholds.json")