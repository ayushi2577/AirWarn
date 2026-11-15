import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from config import ROOT, AQI_CUTOFF, TRAIN
from splits import get_split, between, get_eval_rows, seen_cities
from pipeline import load_model_table, model_columns, get_Xy
from models import make_logreg, make_rbf_svm
from evaluate import metrics_at

RESULTS = ROOT / "outputs" / "results"
cfg = json.load(open(RESULTS / "frozen_settings.json"))

table = load_model_table()
cols = model_columns(table)
seen = seen_cities(table)

# Same frozen refit as the final test
train = between(get_eval_rows(table), *TRAIN)
X_tr, y_tr = get_Xy(train)
lr = make_logreg(cols).fit(X_tr[cols], y_tr)
svm = make_rbf_svm(cols, **cfg["svm_params"]).fit(X_tr[cols], y_tr)

test = get_split(table, "test")
test = test[test["City"].isin(seen)].reset_index(drop=True)
X, _ = get_Xy(test)
y = test["poor_tomorrow"].astype(int).values
s_lr = lr.predict_proba(X[cols])[:, 1]
s_svm = svm.decision_function(X[cols])
pers = (test["AQI"] > AQI_CUTOFF).astype(int).values


def prec_at(y, s, target):
    """Best precision among thresholds with recall >= target."""
    p, r, _ = precision_recall_curve(y, s)
    ok = np.where(r[:-1] >= target)[0]
    return float(p[:-1][ok].max()) if len(ok) else np.nan


pm = metrics_at(y, pers)
print(f"2019 seen cities. Persistence: precision {pm['precision']:.3f}, recall {pm['recall']:.3f}")

week = test["Date"].dt.to_period("W").astype(str).values
idx_by_week = {w: np.where(week == w)[0] for w in np.unique(week)}
weeks = list(idx_by_week)
rng = np.random.default_rng(0)

rows = []
for name, s in [("Logistic regression", s_lr), ("RBF SVM", s_svm),
                ("Tuned single feature (AQI)", test["AQI"].values)]:
    point = prec_at(y, s, pm["recall"])
    diffs = []
    for _ in range(500):
        pick = rng.choice(len(weeks), size=len(weeks), replace=True)
        idx = np.concatenate([idx_by_week[weeks[k]] for k in pick])
        if y[idx].sum() == 0:
            continue
        pb = metrics_at(y[idx], pers[idx])
        diffs.append(prec_at(y[idx], s[idx], pb["recall"]) - pb["precision"])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    rows.append({"model": name,
                 "precision_at_persistence_recall": round(point, 3),
                 "persistence_precision": round(pm["precision"], 3),
                 "gain": round(point - pm["precision"], 3),
                 "gain_95%_interval": f"{lo:+.3f} to {hi:+.3f}"})

res = pd.DataFrame(rows)
res.to_csv(RESULTS / "equal_recall_test_2019.csv", index=False)
print(res.to_string(index=False))
print("\nIf the interval excludes 0, the gain over persistence is bigger than the noise.")