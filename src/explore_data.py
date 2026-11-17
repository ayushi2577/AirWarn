"""
Part 0 data checks for: Next-Day Unhealthy Air Warning for Indian Cities.

Downloads the Kaggle dataset "Air Quality Data in India (2015-2020)"
(rohanrao/air-quality-data-in-india), loads city_day.csv, and writes a
summary of everything needed for the proposal and Part 1.

Usage:
    pip install pandas numpy matplotlib kagglehub
    python explore_data.py                      # downloads with kagglehub
    python explore_data.py --csv path/to/city_day.csv   # use a file you already downloaded

Outputs (in ./outputs/):
    data_summary.md          main report: counts, missingness, target, split sizes
    missing_by_city.csv      % missing per pollutant per city
    city_coverage.csv        first/last date, rows and gaps per city
    positive_rate_by_city.csv, positive_rate_by_year.csv, positive_rate_by_month.csv
    split_counts.csv         rows and positives per split and per validation fold
    plots/*.png              coverage heatmap, missingness, positive rate by month
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KAGGLE_SLUG = "rohanrao/air-quality-data-in-india"
OUT = "outputs"
PLOTS = os.path.join(OUT, "plots")

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
              "Benzene", "Toluene", "Xylene"]

# Official CPCB AQI categories
BUCKETS = [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 200, "Moderate"),
           (201, 300, "Poor"), (301, 400, "Very Poor"), (401, np.inf, "Severe")]
POOR_CUTOFF = 200  # target: tomorrow's AQI > 200

# Split dates (from the plan). The day before each cutoff is dropped (embargo),
# because its label is tomorrow's AQI, which lies in the next period.
# 2020 is split at India's national lockdown (from 2020-03-25); 2020-03-24 is
# an embargo day because its label falls inside the lockdown period.
SPLITS = {
    "train (2015-2018)": ("2015-01-01", "2018-12-30"),
    "test (2019)": ("2019-01-01", "2019-12-30"),
    "stress: 2020 pre-lockdown": ("2020-01-01", "2020-03-23"),
    "stress: 2020 lockdown": ("2020-03-25", "2020-12-31"),
}
TRAIN_END = "2018-12-30"
FOLDS = [  # forward-chaining validation folds inside the training years
    ("fold 1", "2016-12-30", "2017-01-01", "2017-06-30"),
    ("fold 2", "2017-06-29", "2017-07-01", "2017-12-31"),
    ("fold 3", "2017-12-30", "2018-01-01", "2018-06-30"),
    ("fold 4", "2018-06-29", "2018-07-01", "2018-12-30"),
]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_csv(user_path):
    if user_path:
        return user_path
    # Use city_day.csv if it sits next to this script or in the current folder
    for folder in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        local = os.path.join(folder, "city_day.csv")
        if os.path.exists(local):
            print("Using local file:", local)
            return local
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "kagglehub is not installed. Either `pip install kagglehub`, or download "
            "city_day.csv from https://www.kaggle.com/datasets/" + KAGGLE_SLUG +
            " and run: python explore_data.py --csv path/to/city_day.csv")
    print("Downloading from Kaggle (needs a Kaggle login/API token if prompted)...")
    folder = kagglehub.dataset_download(KAGGLE_SLUG)
    hits = glob.glob(os.path.join(folder, "**", "city_day.csv"), recursive=True)
    if not hits:
        raise SystemExit(f"city_day.csv not found in {folder}")
    print("Found:", hits[0])
    return hits[0]


def bucket_of(aqi):
    if pd.isna(aqi):
        return np.nan
    for lo, hi, name in BUCKETS:
        if aqi <= hi:
            return name
    return "Severe"


# ----------------------------------------------------------------------------
# Main checks
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="path to city_day.csv (skips the download)")
    args = ap.parse_args()

    os.makedirs(PLOTS, exist_ok=True)
    df = pd.read_csv(find_csv(args.csv))
    df["Date"] = pd.to_datetime(df["Date"])
    pollutants = [c for c in POLLUTANTS if c in df.columns]

    lines = ["# Data summary: city_day.csv", ""]
    add = lines.append

    # --- 1. Basic shape -----------------------------------------------------
    add("## 1. Basic facts")
    add(f"- Rows: **{len(df):,}**, columns: **{df.shape[1]}**")
    add(f"- Columns: {', '.join(df.columns)}")
    add(f"- Date range: **{df['Date'].min().date()} to {df['Date'].max().date()}**")
    add(f"- Cities: **{df['City'].nunique()}** ({', '.join(sorted(df['City'].unique()))})")
    dups = df.duplicated(["City", "Date"]).sum()
    add(f"- Duplicate city-date rows: **{dups}**")
    add("")

    # --- 2. Coverage per city ----------------------------------------------
    cov = df.groupby("City")["Date"].agg(first="min", last="max", rows="count")
    cov["calendar_days"] = (cov["last"] - cov["first"]).dt.days + 1
    cov["missing_days"] = cov["calendar_days"] - cov["rows"]
    cov["aqi_present"] = df.groupby("City")["AQI"].apply(lambda s: s.notna().sum())
    cov = cov.sort_values("first")
    cov.to_csv(os.path.join(OUT, "city_coverage.csv"))
    add("## 2. Coverage per city (first date, last date, rows)")
    add("```")
    add(cov.to_string())
    add("```")
    add("")

    # --- 3. Missing values ---------------------------------------------------
    miss = df[pollutants + ["AQI", "AQI_Bucket"]].isna().mean().mul(100).round(1)
    add("## 3. Missing values (% of rows)")
    add("```")
    add(miss.sort_values(ascending=False).to_string())
    add("```")
    miss_city = (df.groupby("City")[pollutants + ["AQI"]]
                   .apply(lambda g: g.isna().mean().mul(100).round(1)))
    miss_city.to_csv(os.path.join(OUT, "missing_by_city.csv"))
    fully_missing = {c: list(miss_city.index[miss_city[c] == 100]) for c in pollutants}
    fully_missing = {c: v for c, v in fully_missing.items() if v}
    if fully_missing:
        add("- Pollutants never measured in some cities:")
        for c, v in fully_missing.items():
            add(f"  - {c}: {', '.join(v)}")
    add("")

    # --- 4. Impossible / suspicious values ---------------------------------
    add("## 4. Value checks")
    neg = (df[pollutants + ["AQI"]] < 0).sum()
    neg = neg[neg > 0]
    add(f"- Negative values: {'none' if neg.empty else neg.to_dict()}")
    desc = df[pollutants + ["AQI"]].describe(percentiles=[.5, .99]).T[["min", "50%", "99%", "max"]]
    add("```")
    add(desc.round(1).to_string())
    add("```")
    add("")

    # --- 5. AQI_Bucket consistency -----------------------------------------
    add("## 5. Does AQI_Bucket match the AQI value?")
    both = df.dropna(subset=["AQI", "AQI_Bucket"])
    computed = both["AQI"].apply(bucket_of)
    mismatch = (computed != both["AQI_Bucket"]).sum()
    add(f"- Rows with both: {len(both):,}; mismatches: **{mismatch}**")
    add(f"- Bucket counts: {df['AQI_Bucket'].value_counts().to_dict()}")
    add("")

    # --- 6. Build the target by date (never by 'next row') ------------------
    nxt = df[["City", "Date", "AQI"]].copy()
    nxt["Date"] = nxt["Date"] - pd.Timedelta(days=1)  # row for t+1 shifted onto t
    nxt = nxt.rename(columns={"AQI": "AQI_tomorrow"})
    d = df.merge(nxt, on=["City", "Date"], how="left")
    d["today_poor"] = (d["AQI"] > POOR_CUTOFF).where(d["AQI"].notna())
    d["poor_tomorrow"] = (d["AQI_tomorrow"] > POOR_CUTOFF).where(d["AQI_tomorrow"].notna())
    usable = d.dropna(subset=["poor_tomorrow"]).copy()
    usable["poor_tomorrow"] = usable["poor_tomorrow"].astype(int)

    add("## 6. Target: poor_tomorrow = 1 if AQI on day t+1 > 200")
    no_next_row = (~df.set_index(["City", "Date"]).index.isin(
        nxt.set_index(["City", "Date"]).index)).sum()
    add(f"- Rows whose next calendar day is absent from the file: {no_next_row:,}")
    add(f"- Rows whose next day exists but has no AQI: {d['AQI_tomorrow'].isna().sum() - no_next_row:,}")
    add(f"- Rows with tomorrow's AQI (target defined): {len(usable):,}")

    # Evaluation-rows rule: today's AND tomorrow's AQI must both be observed,
    # so persistence always uses a real reading. Same rows for every model.
    today_blank = usable["today_poor"].isna().sum()
    usable = usable.dropna(subset=["today_poor"]).copy()
    usable["today_poor"] = usable["today_poor"].astype(int)
    add(f"- Of those, today's AQI is blank on **{today_blank:,}** rows -> excluded by the evaluation-rows rule")
    add(f"- **Evaluation rows (today's and tomorrow's AQI both observed): {len(usable):,}**")
    add(f"- **Positives: {usable['poor_tomorrow'].sum():,} "
        f"({usable['poor_tomorrow'].mean():.1%})**")

    usable["change_day"] = (usable["today_poor"] != usable["poor_tomorrow"]).astype(int)
    same = 1 - usable["change_day"].mean()
    add(f"- Persistence check: tomorrow's yes/no status equals today's on **{same:.1%}** of "
        f"evaluation rows (this is the persistence baseline's accuracy)")
    add(f"- Change days (binary flip, <=200 vs >200): **{usable['change_day'].sum():,}** "
        f"(clean->bad: {((usable['today_poor'] == 0) & (usable['poor_tomorrow'] == 1)).sum():,}, "
        f"bad->clean: {((usable['today_poor'] == 1) & (usable['poor_tomorrow'] == 0)).sum():,})")
    add("")

    usable["Year"] = usable["Date"].dt.year
    usable["Month"] = usable["Date"].dt.month
    by_city = usable.groupby("City")["poor_tomorrow"].agg(rows="count", positives="sum", rate="mean")
    by_year = usable.groupby("Year")["poor_tomorrow"].agg(rows="count", positives="sum", rate="mean")
    by_month = usable.groupby("Month")["poor_tomorrow"].agg(rows="count", positives="sum", rate="mean")
    by_city.sort_values("rate", ascending=False).to_csv(os.path.join(OUT, "positive_rate_by_city.csv"))
    by_year.to_csv(os.path.join(OUT, "positive_rate_by_year.csv"))
    by_month.to_csv(os.path.join(OUT, "positive_rate_by_month.csv"))
    for title, tab in [("by year", by_year), ("by city", by_city.sort_values("rate", ascending=False))]:
        add(f"### Positive rate {title}")
        add("```")
        add(tab.assign(rate=tab["rate"].round(3)).to_string())
        add("```")
    add("")

    # --- 7. Split and fold sizes --------------------------------------------
    seen_cities = set(usable.loc[usable["Date"] <= TRAIN_END, "City"])
    rows = []
    for name, (a, b) in SPLITS.items():
        s = usable[(usable["Date"] >= a) & (usable["Date"] <= b)]
        rows.append([name, a, b, len(s), int(s["poor_tomorrow"].sum()),
                     round(s["poor_tomorrow"].mean(), 3) if len(s) else np.nan,
                     s["City"].nunique()])
        if not name.startswith("train"):
            for label, mask in [("  seen cities", s["City"].isin(seen_cities)),
                                ("  new cities", ~s["City"].isin(seen_cities))]:
                g = s[mask]
                rows.append([label, a, b, len(g), int(g["poor_tomorrow"].sum()),
                             round(g["poor_tomorrow"].mean(), 3) if len(g) else np.nan,
                             g["City"].nunique()])
    for name, tr_end, va_start, va_end in FOLDS:
        tr = usable[usable["Date"] <= tr_end]
        va = usable[(usable["Date"] >= va_start) & (usable["Date"] <= va_end)]
        rows.append([f"{name} train", "2015-01-01", tr_end, len(tr), int(tr["poor_tomorrow"].sum()),
                     round(tr["poor_tomorrow"].mean(), 3) if len(tr) else np.nan, tr["City"].nunique()])
        rows.append([f"{name} validate", va_start, va_end, len(va), int(va["poor_tomorrow"].sum()),
                     round(va["poor_tomorrow"].mean(), 3) if len(va) else np.nan, va["City"].nunique()])
    splits = pd.DataFrame(rows, columns=["part", "from", "to", "rows", "positives", "rate", "cities"])
    splits.to_csv(os.path.join(OUT, "split_counts.csv"), index=False)
    add("## 7. Split and fold sizes (embargo days already excluded)")
    add("```")
    add(splits.to_string(index=False))
    add("```")
    small = splits[(splits["positives"] < 100) & splits["part"].str.contains("validate")]
    if not small.empty:
        add("- **Warning:** these validation folds have fewer than 100 positives; "
            "consider adjusting fold dates: " + ", ".join(small["part"]))
    new_all = sorted(set(usable["City"]) - seen_cities)
    add(f"- Cities never in training ({len(new_all)}): {', '.join(new_all)}")
    add("- New-city groups with fewer than 50 positives should be reported but flagged as low-reliability.")
    add("")

    # --- 8. Correlations among pollutants -----------------------------------
    corr = df[pollutants + ["AQI"]].corr().round(2)
    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
                 .sort_values(ascending=False))
    add("## 8. Most correlated pairs (|r| > 0.7)")
    add("```")
    add(pairs[pairs.abs() > 0.7].to_string() if (pairs.abs() > 0.7).any() else "none")
    add("```")
    add("")

    # --- 9. Plots --------------------------------------------------------------
    cy = df.assign(Year=df["Date"].dt.year).pivot_table(
        index="City", columns="Year", values="AQI", aggfunc=lambda s: s.notna().sum(), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(cy.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cy.columns)), cy.columns)
    ax.set_yticks(range(len(cy.index)), cy.index)
    ax.set_title("Days with an AQI value, by city and year")
    fig.colorbar(im, ax=ax, label="days")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "coverage_city_year.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    miss.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("% missing"); ax.set_title("Missing values per column")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "missing_per_column.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_month.index, by_month["rate"])
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylabel("share of Poor-or-worse tomorrow")
    ax.set_title("Positive rate by month (seasonality)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "positive_rate_by_month.png"), dpi=120); plt.close(fig)

    with open(os.path.join(OUT, "data_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\nFull report: {OUT}/data_summary.md  |  plots: {PLOTS}/")


if __name__ == "__main__":
    main()