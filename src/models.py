import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import FOLDS
from splits import get_fold
from pipeline import load_model_table, model_columns, get_Xy, make_preprocessor
from evaluate import full_metrics, metrics_at, pick_threshold, change_day_mask

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


def make_logreg(columns, C=1.0):
    """Preprocessing + logistic regression in one Pipeline.
    Light fixed L2 (C=1.0), no class weights. L1 vs L2 comes later (Part 5)."""
    return Pipeline([
        ("prep", make_preprocessor(columns)),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])

def make_linear_svm(columns, C=1.0):
    """Linear SVM (squared hinge loss, the LinearSVC default)."""
    return Pipeline([
        ("prep", make_preprocessor(columns)),
        ("clf", LinearSVC(C=C, max_iter=20000)),
    ])


def make_rbf_svm(columns, C=1.0, gamma="scale"):
    """Exact RBF SVM. One fit takes about 3 seconds, so no Nystroem."""
    return Pipeline([
        ("prep", make_preprocessor(columns)),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma)),
    ])


def svm_validation_scores(table, make_model, columns):
    """Same as validation_scores, but SVMs give decision scores, not probabilities."""
    parts = []
    for i in range(len(FOLDS)):
        tr, va = get_fold(table, i)
        X_tr, y_tr = get_Xy(tr)
        X_va, _ = get_Xy(va)
        model = make_model(columns)
        model.fit(X_tr[columns], y_tr)
        va = va.copy()
        va["score"] = model.decision_function(X_va[columns])
        va["fold"] = i + 1
        parts.append(va)
    return pd.concat(parts)


def tune(table, make_model, columns, grid):
    """Try each setting on the 4 folds. Returns a table of pooled PR-AUC
    (the average of the 4 per-fold PR-AUC values) for each setting."""
    from sklearn.metrics import average_precision_score
    rows = []
    for params in grid:
        val = svm_validation_scores(table, lambda c, p=params: make_model(c, **p), columns)
        per_fold = [average_precision_score(g["poor_tomorrow"].astype(int), g["score"])
                    for _, g in val.groupby("fold")]
        rows.append({**params, "mean_pr_auc": sum(per_fold) / len(per_fold)})
    return pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)

def validation_scores(table, make_model, columns):
    """For each forward-chaining fold: fit on the fold's training rows only,
    predict probabilities on its validation rows. Returns all validation rows
    stacked, with 'score' and 'fold' columns added."""
    parts = []
    for i in range(len(FOLDS)):
        tr, va = get_fold(table, i)
        X_tr, y_tr = get_Xy(tr)
        X_va, _ = get_Xy(va)
        model = make_model(columns)
        model.fit(X_tr[columns], y_tr)
        va = va.copy()
        va["score"] = model.predict_proba(X_va[columns])[:, 1]
        va["fold"] = i + 1
        parts.append(va)
    return pd.concat(parts)


def summarize(val, score_col="score", min_recall=0.80):
    """Per-fold and pooled metrics. The frozen threshold is chosen on the
    pooled validation scores; 'prec@80rec' uses each part's own threshold
    (this is the number the model-choice rule compares across folds)."""
    y_all = val["poor_tomorrow"].astype(int).values
    t = pick_threshold(y_all, val[score_col].values, min_recall)
    rows = []
    groups = [(f"fold{f}", g) for f, g in val.groupby("fold")] + [("pooled", val)]
    for name, g in groups:
        y = g["poor_tomorrow"].astype(int).values
        s = g[score_col].values
        pred = (s >= t).astype(int)
        m = full_metrics(y, s, pred)
        own_t = pick_threshold(y, s, min_recall)
        m["prec@80rec"] = metrics_at(y, (s >= own_t).astype(int))["precision"]
        chg = change_day_mask(g)
        m["change_day_recall"] = metrics_at(y[chg], pred[chg])["recall"]
        m["part"] = name
        rows.append(m)
    cols = ["part", "precision", "recall", "f1", "pr_auc", "roc_auc",
            "prec@80rec", "change_day_recall"]
    return t, pd.DataFrame(rows)[cols]
