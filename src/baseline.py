"""
get_fold gives each validation fold. We use validation only, never the 2019 test.
Three baselines make predictions: always “not Poor”, persistence, and “AQI > c”.
pick_threshold finds c on the pooled validation rows, as the highest precision with recall at least 80%.
full_metrics scores each baseline with precision, recall, F1, PR-AUC and ROC-AUC.
change_day_mask marks days where the yes/no status flips. Persistence has recall 0 on these by construction, and the table shows it.

majority(rows): Always predicts 0 (“not Poor”). Shows why accuracy misleads.

persistence(rows): Predicts 1 if today’s AQI is above 200 (“tomorrow = today”). This is the main bar to beat.

single_feature(rows, c): Predicts 1 if today’s AQI is above a tuned cutoff c. Shows how much the extra features add beyond today’s AQI.
The main block in baselines.py (runs only when you execute the file):

Loads the table and collects the 4 validation folds.
Uses pick_threshold on the pooled validation rows to find the cutoff c.
For each fold and for the pooled set, runs the three baselines and scores them with full_metrics.
Also computes recall on change days only.
Prints one results table.
"""

import numpy as np
import pandas as pd
from config import AQI_CUTOFF, FOLDS
from splits import get_fold
from evaluate import full_metrics, pick_threshold, change_day_mask, metrics_at


def majority(rows):
    """Always 'not Poor'."""
    return np.zeros(len(rows), dtype=int)


def persistence(rows):
    """Tomorrow is bad if today's observed AQI > 200."""
    return (rows["AQI"] > AQI_CUTOFF).astype(int).values


def single_feature(rows, c):
    """Tomorrow is bad if today's AQI > c (c tuned on validation)."""
    return (rows["AQI"] > c).astype(int).values


if __name__ == "__main__":
    from pipeline import load_model_table

    table = load_model_table()
    vals = [get_fold(table, i)[1] for i in range(len(FOLDS))]
    pooled = pd.concat(vals)
    c = pick_threshold(pooled["poor_tomorrow"].astype(int), pooled["AQI"], 0.80)
    print(f"tuned cutoff c (recall >= 80% on pooled validation): {c:.1f}\n")

    results = []
    for name, rows in [(f"fold{i+1}", v) for i, v in enumerate(vals)] + [("pooled", pooled)]:
        y = rows["poor_tomorrow"].astype(int).values
        chg = change_day_mask(rows)
        for model, pred, score in [
            ("majority", majority(rows), np.zeros(len(rows))),
            ("persistence", persistence(rows), persistence(rows)),
            ("single_feature", single_feature(rows, c), rows["AQI"].values),
        ]:
            m = full_metrics(y, score, pred)
            m["change_day_recall"] = metrics_at(y[chg], pred[chg])["recall"]
            m.update(part=name, model=model)
            results.append(m)

    df = pd.DataFrame(results)
    cols = ["part", "model", "precision", "recall", "f1", "pr_auc", "roc_auc", "change_day_recall"]
    print(df[cols].round(3).to_string(index=False))