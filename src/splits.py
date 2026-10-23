"""
between: picks rows in a date range.
get_eval_rows: keeps only the usable rows (is_eval).
seen_cities: finds which cities appear in the training years, so we can tag “new” cities later.
get_split: returns the train, test or 2020 rows.
get_fold: returns the training and validation rows for one forward-chaining fold.
print_split_counts: prints the counts you need to check.
"""
import pandas as pd
from config import TRAIN, TEST, PRE_LOCKDOWN, LOCKDOWN, FOLDS


def between(df, start, end):
    """Rows whose Date is within [start, end]."""
    return df[(df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))]


def get_eval_rows(df):
    """Only rows where today's AND tomorrow's AQI are observed."""
    return df[df["is_eval"]].copy()


def seen_cities(df):
    """Cities with at least one evaluation row in the training years."""
    return set(between(get_eval_rows(df), *TRAIN)["City"].unique())


def get_split(df, name):
    """name: 'train', 'test', 'pre_lockdown', 'lockdown'."""
    ranges = {"train": TRAIN, "test": TEST,
              "pre_lockdown": PRE_LOCKDOWN, "lockdown": LOCKDOWN}
    return between(get_eval_rows(df), *ranges[name])


def get_fold(df, i):
    """Returns (train_rows, val_rows) for fold i (0-based).
    Train = everything from 2015 up to train_end; validation = the next half-year."""
    train_end, val_start, val_end = FOLDS[i]
    ev = get_eval_rows(df)
    train = between(ev, TRAIN[0], train_end)
    val = between(ev, val_start, val_end)
    return train, val


def print_split_counts(df):
    seen = seen_cities(df)
    print("Seen cities (have eval rows in 2015-2018):", len(seen))
    print("\nSplit               rows  positives  new-city rows  new-city positives")
    for name in ["train", "test", "pre_lockdown", "lockdown"]:
        s = get_split(df, name)
        new = s[~s["City"].isin(seen)]
        print(f"{name:<18}{len(s):>6}{int(s['poor_tomorrow'].sum()):>11}"
              f"{len(new):>15}{int(new['poor_tomorrow'].sum()):>20}")
    print("\nFold  train_rows  val_rows  val_positives  val_pos_rate")
    for i in range(len(FOLDS)):
        tr, va = get_fold(df, i)
        print(f"{i+1:>4}{len(tr):>12}{len(va):>10}"
              f"{int(va['poor_tomorrow'].sum()):>15}{va['poor_tomorrow'].mean():>14.1%}")


if __name__ == "__main__":
    from data import build_dataset
    print_split_counts(build_dataset())