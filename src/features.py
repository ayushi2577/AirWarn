"""
build_features sorts by city and date, so each city’s rows are in day order.
It forward-fills short gaps within each city, then adds “was missing” flags from the raw values.
It adds lags and rolling means (both look backwards only) and the change since yesterday.
It adds month, day of week and season.
group_of labels each column with its feature group, which you’ll need for Lasso and the ablation.
"""
import pandas as pd

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
              "O3", "Benzene", "Toluene"]          # Xylene dropped
BASE = POLLUTANTS + ["AQI"]
FFILL_LIMIT = 3                                    # fill gaps of at most 3 days
LAG_COLS = ["AQI", "PM2.5", "PM10"]
LAGS = [1, 2, 3, 7]
ROLL_COLS = ["AQI", "PM2.5", "PM10"]
ROLL_WINDOWS = [3, 7]
FLAG_COLS = ["PM10", "NH3", "Toluene", "Benzene"]  # many gaps -> add "was missing" flag
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "summer", 4: "summer",
           5: "summer", 6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
           10: "postmonsoon", 11: "postmonsoon"}


def build_features(df):
    """Past-only features for every city-day. Needs the explicit calendar
    (one row per city per day), so shift(k) means exactly k days back.
    Returns City, Date plus feature columns (no target)."""
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    out = df[["City", "Date"]].copy()

    # 1. forward-fill within each city (past values only, short gaps only)
    filled = df.groupby("City")[BASE].ffill(limit=FFILL_LIMIT)
    out[BASE] = filled

    # 2. "was missing today" flags, from the RAW values
    for c in FLAG_COLS:
        out[f"{c}_missing"] = df[c].isna().astype(int)

    by_city = filled.groupby(df["City"])

    # 3. lags: value k days ago
    for c in LAG_COLS:
        for k in LAGS:
            out[f"{c}_lag{k}"] = by_city[c].shift(k)

    # 4. rolling means that END at day t (backward window, never centered)
    for c in ROLL_COLS:
        for w in ROLL_WINDOWS:
            out[f"{c}_roll{w}"] = by_city[c].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean())

    # 5. change since yesterday (rising or falling?)
    out["AQI_chg1"] = out["AQI"] - out["AQI_lag1"]

    # 6. calendar
    out["month"] = df["Date"].dt.month
    out["dayofweek"] = df["Date"].dt.day_of_week
    out["season"] = df["Date"].dt.month.map(SEASONS)
    return out


def group_of(col):
    """Which of the five feature groups a column belongs to (City is added later by one-hot)."""
    base = col.split("_")[0]
    if base in ("PM2.5", "PM10"):
        return "particulates"
    if base == "AQI":
        return "recent_history"
    if base in ("month", "dayofweek", "season"):
        return "calendar"
    return "gases"


def feature_columns(feats):
    return [c for c in feats.columns if c not in ("City", "Date")]


if __name__ == "__main__":
    from data import build_dataset
    f = build_features(build_dataset())
    print(f.shape)
    print(f.isna().mean().sort_values(ascending=False).head(8))