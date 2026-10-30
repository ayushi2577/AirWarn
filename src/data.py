"""
load_raw: reads the CSV and sorts it by city and date.
make_explicit_calendar: guarantees one row per city per day. Data check says none are missing, but this makes the code safe anyway.
add_target: adds 2 column poor_tomorrow and is_eval ... finds tomorrow’s AQI by date and marks which rows are usable (is_eval). It doesn’t delete anything yet, because lags and rolling means in features.py need the blank days too. We filter to is_eval only when training and scoring.
build_dataset: runs steps 1 to 3 in order.
print_counts: shows the counts to compare with your proposal.
"""

import pandas as pd
from config import RAW_CSV, PROCESSED_DIR, AQI_CUTOFF


def load_raw(path=RAW_CSV):
    """Read city_day.csv, parse dates, sort by city then date."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def make_explicit_calendar(df):
    """Make sure every city has one row for EVERY calendar day between its
    first and last date. Missing days get blank values. This way 'next row'
    always means 'next day'."""
    parts = []
    for city, g in df.groupby("City"):
        full_days = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        g = g.set_index("Date").reindex(full_days)
        g["City"] = city
        g.index.name = "Date"
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def add_target(df):
    """Look up tomorrow's AQI BY DATE (merge on Date + 1 day), then build:
       - AQI_tomorrow
       - is_eval: True only if today's AND tomorrow's AQI are both observed
       - poor_tomorrow: 1 if AQI_tomorrow > 200 else 0 (only on eval rows)"""
    tomorrow = df[["City", "Date", "AQI"]].copy()
    tomorrow["Date"] = tomorrow["Date"] - pd.Timedelta(days=1)  # shift tomorrow's value onto today's date
    tomorrow = tomorrow.rename(columns={"AQI": "AQI_tomorrow"})

    df = df.merge(tomorrow, on=["City", "Date"], how="left")
    df["is_eval"] = df["AQI"].notna() & df["AQI_tomorrow"].notna()
    df["poor_tomorrow"] = (df["AQI_tomorrow"] > AQI_CUTOFF).astype(int)
    df.loc[~df["is_eval"], "poor_tomorrow"] = pd.NA
    return df


def build_dataset():
    """Full flow: load -> explicit calendar -> target. Returns ALL rows
    (with an is_eval flag), because features later need the blank days too."""
    return add_target(make_explicit_calendar(load_raw()))


def print_counts(df):
    """Print the numbers that must match your proposal."""
    has_tomorrow = df["AQI_tomorrow"].notna()
    ev = df[df["is_eval"]]
    print("total rows:", len(df))
    print("rows with tomorrow's AQI:", has_tomorrow.sum())
    print("today blank but tomorrow present (dropped):", (has_tomorrow & df["AQI"].isna()).sum())
    print("evaluation rows:", len(ev))
    print("positives:", int(ev["poor_tomorrow"].sum()),
          f"({ev['poor_tomorrow'].mean():.1%})")
    print("\nrows and positives per year:")
    print(ev.groupby(ev["Date"].dt.year)["poor_tomorrow"].agg(rows="count", positives="sum"))


if __name__ == "__main__":    # if directly ran
    data = build_dataset()
    print_counts(data)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(PROCESSED_DIR / "full_dataset.parquet", index=False)