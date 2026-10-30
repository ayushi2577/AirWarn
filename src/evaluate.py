"""
metrics_at(y, pred): Takes the true answers and yes/no predictions. Returns the confusion matrix counts (TP, FP, FN, TN) plus precision, recall and F1.

full_metrics(y, score, pred): Calls metrics_at, then adds PR-AUC and ROC-AUC, which use the model’s raw scores across all thresholds. This is the main scoring function.

change_day_mask(rows): Marks the days where the yes/no status flips (today’s status differs from tomorrow’s). Persistence is wrong on every one of these days.

pick_threshold(y, score, min_recall): Searches thresholds and returns the one with the highest precision while recall stays at least 80%. Use it on validation data only, then freeze the value.

baselines.py (the three simple guesses)

majority(rows): Always predicts 0 (“not Poor”). Shows why accuracy misleads.

persistence(rows): Predicts 1 if today’s AQI is above 200 (“tomorrow = today”). This is the main bar to beat.

single_feature(rows, c): Predicts 1 if today’s AQI is above a tuned cutoff c. Shows how much the extra features add beyond today’s AQI.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)
from config import AQI_CUTOFF


def _div(a, b):
    return a / b if b else 0.0


def metrics_at(y, pred):
    """Confusion matrix, precision, recall, F1 for yes/no predictions."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": p, "recall": r, "f1": _div(2 * p * r, p + r)}


def full_metrics(y, score, pred):
    """Threshold-free metrics (PR-AUC, ROC-AUC) plus the metrics at `pred`."""
    out = metrics_at(y, pred)
    out["pr_auc"] = average_precision_score(y, score)
    out["roc_auc"] = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    return out


def change_day_mask(rows):
    """True where the yes/no status flips: today's status != tomorrow's."""
    today_bad = (rows["AQI"] > AQI_CUTOFF).astype(int).values
    return today_bad != rows["poor_tomorrow"].astype(int).values


def pick_threshold(y, score, min_recall=0.80):
    """Highest precision among thresholds with recall >= min_recall.
    Choose this on VALIDATION data only, then freeze it."""
    prec, rec, thr = precision_recall_curve(y, score)
    ok = np.where(rec[:-1] >= min_recall)[0]
    if len(ok) == 0:
        return float(thr.min())
    best = ok[np.argmax(prec[:-1][ok])]
    return float(thr[best])