import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import ROOT, AQI_CUTOFF
from models import (load_model_table, model_columns, make_logreg, make_rbf_svm,
                    validation_scores, svm_validation_scores)
from evaluate import block_bootstrap, metrics_at

RESULTS = ROOT / "outputs" / "results"
cfg = json.load(open(RESULTS / "frozen_settings.json"))

table = load_model_table()
cols = model_columns(table)
lr_val = validation_scores(table, make_logreg, cols)
svm_val = svm_validation_scores(
    table, lambda c: make_rbf_svm(c, **cfg["svm_params"]), cols)
svm_val["prob"] = 1 / (1 + np.exp(-(cfg["platt_coef"] * svm_val["score"]
                                    + cfg["platt_intercept"])))

y = lr_val["poor_tomorrow"].astype(int).values
entries = {
    "Logistic regression": (lr_val, lr_val["score"].values,
                            cfg["thresholds"]["Logistic regression"]),
    "RBF SVM (Platt)": (svm_val, svm_val["prob"].values,
                        cfg["thresholds"]["RBF SVM (Platt)"]),
}

rows_out = []
for name, (df, s, t) in entries.items():
    ci = block_bootstrap(df, y, s, (s >= t).astype(int))
    rows_out.append({"model": name, **{k: f"{v[0]:.3f} to {v[1]:.3f}" for k, v in ci.items()}})

pers = (lr_val["AQI"] > AQI_CUTOFF).astype(int).values
ci = block_bootstrap(lr_val, y, pers.astype(float), pers)
rows_out.append({"model": "Persistence", **{k: f"{v[0]:.3f} to {v[1]:.3f}" for k, v in ci.items()}})

res = pd.DataFrame(rows_out)
res.to_csv(RESULTS / "bootstrap_ci_validation.csv", index=False)
print(res.to_string(index=False))