import json
import numpy as np
import pandas as pd

from config import ROOT, AQI_CUTOFF
from models import (load_model_table, model_columns, make_logreg,
                    validation_scores)
from evaluate import metrics_at, change_day_mask

RESULTS = ROOT / "outputs" / "results"
DOCS_DATA = ROOT / "docs" / "data"
cfg = json.load(open(RESULTS / "frozen_settings.json"))
T = cfg["thresholds"]["Logistic regression"]

table = load_model_table()
cols = model_columns(table)
val = validation_scores(table, make_logreg, cols).reset_index(drop=True)

val["y"] = val["poor_tomorrow"].astype(int)
val["pred"] = (val["score"] >= T).astype(int)
val["error"] = np.where((val.y == 1) & (val.pred == 0), "FN",
                np.where((val.y == 0) & (val.pred == 1), "FP", "ok"))
val["change_day"] = change_day_mask(val)
val["month"] = val["Date"].dt.month
val["year"] = val["Date"].dt.year

print(f"Frozen LR threshold: {T}")
print(val["error"].value_counts().to_string(), "\n")

# 1. What do missed bad days (FN) look like vs caught ones (TP)?
bad = val[val.y == 1]
print("Bad days: share caught vs missed, by whether today was already bad")
bad = bad.assign(today_bad=(bad["AQI"] > AQI_CUTOFF))
print(bad.groupby("today_bad")["pred"].agg(rows="count", caught="mean")
      .round(3).to_string(), "\n")
print("Average values on missed (FN) vs caught (TP) bad days")
cmp_cols = ["AQI", "AQI_lag1", "AQI_chg1", "PM2.5", "PM10"]
print(bad.groupby("pred")[cmp_cols].mean().round(1)
      .rename(index={0: "missed", 1: "caught"}).to_string(), "\n")

# 2. Errors by change day vs ordinary day
print("By change day (flip) vs ordinary day")
g = val.groupby("change_day").apply(
    lambda d: pd.Series(metrics_at(d.y.values, d.pred.values)))
print(g[["TP", "FP", "FN", "TN", "precision", "recall"]].round(3).to_string(), "\n")

# 3. Errors by month (Oct-Nov = Diwali and crop burning)
print("By month: bad days, missed, false alarms")
m = val.groupby("month").apply(lambda d: pd.Series({
    "rows": len(d), "bad_days": int(d.y.sum()),
    "FN": int((d.error == "FN").sum()), "FP": int((d.error == "FP").sum()),
    "recall": metrics_at(d.y.values, d.pred.values)["recall"]}))
print(m.round(3).to_string(), "\n")

# 4. Heavy imputation: rows right after a long gap (lags were blank before filling)
raw = table.set_index(["City", "Date"])
val = val.merge(table[["City", "Date", "AQI_lag1", "AQI_lag3", "PM10_missing"]]
                .rename(columns={"AQI_lag1": "l1", "AQI_lag3": "l3"}),
                on=["City", "Date"], how="left", suffixes=("", "_t"))
val["gappy"] = val["l1"].isna() | val["l3"].isna() | (val["PM10_missing"] == 1)
print("Rows that needed imputation (blank lag or missing PM10) vs complete rows")
g = val.groupby("gappy").apply(
    lambda d: pd.Series({"rows": len(d), "bad_days": int(d.y.sum()),
                         **{k: v for k, v in metrics_at(d.y.values, d.pred.values).items()
                            if k in ("precision", "recall")}}))
print(g.round(3).to_string(), "\n")

# 5. Worst individual misses (large gap between prediction and outcome)
fn = val[val.error == "FN"].sort_values("score").head(10)
print("10 most confident misses (bad day predicted as fine)")
print(fn[["City", "Date", "AQI", "AQI_lag1", "score"]].round(2).to_string(index=False), "\n")
fp = val[val.error == "FP"].sort_values("score", ascending=False).head(10)
print("10 most confident false alarms")
print(fp[["City", "Date", "AQI", "AQI_lag1", "score"]].round(2).to_string(index=False))

val[["City", "Date", "AQI", "y", "score", "pred", "error", "change_day"]].to_csv(
    RESULTS / "validation_predictions.csv", index=False)
json.dump(m.reset_index().round(4).to_dict("records"),
          open(DOCS_DATA / "errors_by_month.json", "w"), indent=1)
print("\nSaved validation_predictions.csv and docs/data/errors_by_month.json")