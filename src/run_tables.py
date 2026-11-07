import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from config import ROOT, AQI_CUTOFF
from models import (load_model_table, model_columns, make_logreg, make_rbf_svm,
                    validation_scores, svm_validation_scores)
from evaluate import metrics_at, change_day_mask

RESULTS = ROOT / "outputs" / "results"
FIGS = ROOT / "outputs" / "figures"
DOCS_DATA = ROOT / "docs" / "data"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
DOCS_DATA.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(RESULTS / "frozen_settings.json"))

table = load_model_table()
cols = model_columns(table)
lr = validation_scores(table, make_logreg, cols).reset_index(drop=True)
svm = svm_validation_scores(
    table, lambda c: make_rbf_svm(c, **cfg["svm_params"]), cols).reset_index(drop=True)
svm["score"] = 1 / (1 + np.exp(-(cfg["platt_coef"] * svm["score"]
                                 + cfg["platt_intercept"])))

y = lr["poor_tomorrow"].astype(int).values
T = cfg["thresholds"]
pers = (lr["AQI"] > AQI_CUTOFF).astype(int).values
models = {"Logistic regression": (lr, lr["score"].values),
          "RBF SVM (Platt)": (svm, svm["score"].values)}

# 1. Precision at persistence's recall (the fair comparison)
pers_m = metrics_at(y, pers)


def precision_at_recall(y, s, target):
    p, r, _ = precision_recall_curve(y, s)
    ok = np.where(r[:-1] >= target)[0]
    return float(p[:-1][ok].max()) if len(ok) else float("nan")


print(f"Persistence: precision {pers_m['precision']:.3f} at recall {pers_m['recall']:.3f}")
for name, (_, s) in models.items():
    print(f"{name}: precision {precision_at_recall(y, s, pers_m['recall']):.3f} "
          f"at the same recall")

# 2. Change days
chg = change_day_mask(lr)
print(f"\nChange days: {chg.sum()} of {len(y)} ({chg.mean():.1%})")
print(f"(of which clean->bad: {int(y[chg].sum())})")
rows = [{"model": "Persistence", "caught": int(pers[chg][y[chg] == 1].sum()),
         "change_days_correct": float((pers[chg] == y[chg]).mean())}]
for name, (_, s) in models.items():
    pred_m = (s >= T[name]).astype(int)
    rows.append({"model": name, "caught": int(pred_m[chg][y[chg] == 1].sum()),
                 "change_days_correct": float((pred_m[chg] == y[chg]).mean())})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# 3. Per-city results (logistic regression, frozen threshold)
pred = (lr["score"].values >= T["Logistic regression"]).astype(int)
lr_r = lr.copy()
lr_r["pred"] = pred
city_rows = []
for c, g in lr_r.groupby("City"):
    yy = g["poor_tomorrow"].astype(int).values
    m = metrics_at(yy, g["pred"].values)
    city_rows.append({"city": c, "rows": len(g), "positives": int(yy.sum()),
                      "precision": m["precision"], "recall": m["recall"],
                      "reliable": int(yy.sum()) >= 30})
city = pd.DataFrame(city_rows).sort_values("positives", ascending=False)
city.to_csv(RESULTS / "per_city_validation.csv", index=False)
print("\nPer-city (reliable = at least 30 positives):")
print(city.round(3).to_string(index=False))

# 4. Reliability diagram
plt.figure(figsize=(5, 5))
for name, (_, s) in models.items():
    frac, mean = calibration_curve(y, s, n_bins=10)
    plt.plot(mean, frac, marker="o", label=name)
plt.plot([0, 1], [0, 1], "k--", label="perfect")
plt.xlabel("predicted probability")
plt.ylabel("actual share of bad days")
plt.title("Reliability diagram (validation)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / "reliability_validation.png", dpi=150)

# 5. Slider data (validation view; test is added in Part 8)
sweep = pd.read_csv(RESULTS / "threshold_sweep_validation.csv")
out = {"frozen_thresholds": T,
       "persistence": {k: pers_m[k] for k in ["TP", "FP", "FN", "TN", "precision", "recall"]},
       "validation": {name: sweep[sweep["model"] == name]
                      [["threshold", "TP", "FP", "FN", "TN", "precision", "recall"]]
                      .to_dict("records") for name in models}}
json.dump(out, open(DOCS_DATA / "thresholds.json", "w"), indent=1)
print("\nSaved per_city csv, reliability plot, docs/data/thresholds.json")