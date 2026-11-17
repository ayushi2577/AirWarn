import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import build_dataset
from features import build_features, feature_columns, FLAG_COLS

CAT_COLS = ["City", "season", "month", "dayofweek"]
FLAG_NAMES = [f"{c}_missing" for c in FLAG_COLS]


def load_model_table():
    """Features + target + is_eval in one table (all rows, all splits).
    Use splits.get_split / get_fold on this table to pick rows."""
    data = build_dataset()
    feats = build_features(data)
    target = data[["City", "Date", "is_eval", "poor_tomorrow"]]
    return feats.merge(target, on=["City", "Date"], how="left")


def model_columns(table):
    """Columns the model may see: engineered features + City. Never the target."""
    return feature_columns(table.drop(columns=["is_eval", "poor_tomorrow"])) + ["City"]


def get_Xy(rows):
    X = rows[model_columns(rows)]
    y = rows["poor_tomorrow"].astype(int)
    return X, y


class PercentileCapper(BaseEstimator, TransformerMixin):
    """Learns an upper cap (default 99.5th percentile) per column from the
    TRAINING data, then clips any later data to it. NaNs are left alone."""

    def __init__(self, upper=99.5):
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        caps = []
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            caps.append(np.percentile(col, self.upper) if len(col) else np.inf)
        self.caps_ = np.array(caps)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.minimum(np.asarray(X, dtype=float), self.caps_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(columns):
    """Preprocessing for the given input columns (pass a subset for ablation).
    Everything learned (caps, medians, scaling, city list) comes from the
    training rows only, because this sits inside a Pipeline."""
    cat = [c for c in columns if c in CAT_COLS]
    flags = [c for c in columns if c in FLAG_NAMES]
    num = [c for c in columns if c not in cat and c not in flags]

    num_pipe = Pipeline([
        ("cap", PercentileCapper(99.5)),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", num_pipe, num),
         ("flag", "passthrough", flags),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        verbose_feature_names_out=False,
    )


if __name__ == "__main__":
    from splits import get_fold

    table = load_model_table()
    cols = model_columns(table)
    print("input columns:", len(cols))

    # Sanity check on fold 1 only. The 2019 test set is NOT touched here.
    tr, va = get_fold(table, 0)
    X_tr, y_tr = get_Xy(tr)
    X_va, y_va = get_Xy(va)

    pre = make_preprocessor(cols)
    A = pre.fit_transform(X_tr)
    B = pre.transform(X_va)
    print("train matrix:", A.shape, "| val matrix:", B.shape)
    print("NaNs left (train, val):", int(np.isnan(A).sum()), int(np.isnan(B).sum()))

    new_in_val = set(va["City"]) - set(tr["City"])
    print("cities in validation but not in training:", sorted(new_in_val))
    if new_in_val:
        rows = va["City"].isin(new_in_val)
        city_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith("City_")]
        print("city columns all zero for those rows:", bool((B[rows.values][:, city_cols] == 0).all()))