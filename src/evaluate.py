"""
metrics_at(y, pred): Takes the true answers and yes/no predictions. Returns the confusion matrix counts (TP, FP, FN, TN) plus precision, recall and F1.

full_metrics(y, score, pred): Calls metrics_at, then adds PR-AUC and ROC-AUC, which use the model’s raw scores across all thresholds. This is the main scoring function.

change_day_mask(rows): Marks the days where the yes/no status flips (today’s status differs from tomorrow’s). Persistence is wrong on every one of these days.

pick_threshold(y, score, min_recall): Searches thresholds and returns the one with the highest precision while recall stays at least 80%. Use it on validation data only, then freeze the value.
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

def block_bootstrap(rows, y, score, pred, n_boot=500, seed=0):
    """Resample whole calendar weeks (with replacement) and recompute metrics.
    rows needs a 'Date' column. Returns {metric: (low, high)} 95% intervals."""
    rng = np.random.default_rng(seed)
    week = rows["Date"].dt.to_period("W").astype(str).values
    y, score, pred = np.asarray(y), np.asarray(score), np.asarray(pred)
    idx_by_week = {w: np.where(week == w)[0] for w in np.unique(week)}
    weeks = list(idx_by_week)
    out = {"precision": [], "recall": [], "pr_auc": []}
    for _ in range(n_boot):
        pick = rng.choice(len(weeks), size=len(weeks), replace=True)
        idx = np.concatenate([idx_by_week[weeks[k]] for k in pick])
        if y[idx].sum() == 0:
            continue
        m = metrics_at(y[idx], pred[idx])
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
        out["pr_auc"].append(average_precision_score(y[idx], score[idx]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items()}