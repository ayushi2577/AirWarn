import json
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from config import ROOT, TRAIN
from splits import get_split, between, get_eval_rows, seen_cities
from pipeline import load_model_table, model_columns, get_Xy
from models import (make_logreg, make_rbf_svm, validation_scores,
                    svm_validation_scores)

RESULTS = ROOT / "outputs" / "results"
OUT = ROOT / "docs" / "data"
OUT.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(RESULTS / "frozen_settings.json"))
T = cfg["thresholds"]

table = load_model_table()
cols = model_columns(table)
seen = seen_cities(table)

# 1. final test table (from the one-time run)
pd.read_csv(RESULTS / "final_test_results.csv").to_json(
    OUT / "final_results.json", orient="records", indent=1)

# 2. per-city tables
per_city = {
    "validation": json.loads(pd.read_csv(RESULTS / "per_city_validation.csv")
                             .to_json(orient="records")),
    "test_2019": json.loads(pd.read_csv(RESULTS / "per_city_test_2019.csv")
                            .to_json(orient="records")),
}
json.dump(per_city, open(OUT / "per_city.json", "w"), indent=1)

# 3. reliability diagram data (validation)
lr_val = validation_scores(table, make_logreg, cols)
svm_val = svm_validation_scores(
    table, lambda c: make_rbf_svm(c, **cfg["svm_params"]), cols)
svm_p = 1 / (1 + np.exp(-(cfg["platt_coef"] * svm_val["score"].values
                          + cfg["platt_intercept"])))
y = lr_val["poor_tomorrow"].astype(int).values
rel = {}
for name, s in [("Logistic regression", lr_val["score"].values),
                ("RBF SVM (Platt)", svm_p)]:
    frac, mean = calibration_curve(y, s, n_bins=10)
    rel[name] = [{"pred": float(m), "actual": float(f)} for m, f in zip(mean, frac)]
json.dump(rel, open(OUT / "reliability.json", "w"), indent=1)

# 4. 2019 timeline (frozen logistic regression refit, same as the final test)
train = between(get_eval_rows(table), *TRAIN)
X_tr, y_tr = get_Xy(train)
lr = make_logreg(cols).fit(X_tr[cols], y_tr)
test = get_split(table, "test").reset_index(drop=True)
X, _ = get_Xy(test)
test["p"] = lr.predict_proba(X[cols])[:, 1]
cities = {}
for c, g in test.groupby("City"):
    g = g.sort_values("Date")
    cities[c] = {"seen": c in seen,
                 "dates": g["Date"].dt.strftime("%Y-%m-%d").tolist(),
                 "p": [round(float(v), 3) for v in g["p"]],
                 "y": g["poor_tomorrow"].astype(int).tolist()}
json.dump({"threshold": T["Logistic regression"], "cities": cities},
          open(OUT / "timeline.json", "w"), separators=(",", ":"))
print("Saved final_results.json, per_city.json, reliability.json, timeline.json")