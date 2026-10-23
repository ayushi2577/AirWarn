from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "city_day.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

AQI_CUTOFF = 200   # official CPCB "Poor" boundary, not tuned

# Fixed split (all cities share the same dates). Dropped embargo days:
# 2018-12-31, 2019-12-31 and 2020-03-24.
TRAIN = ("2015-01-01", "2018-12-30")
TEST = ("2019-01-01", "2019-12-30")
PRE_LOCKDOWN = ("2020-01-01", "2020-03-23")
LOCKDOWN = ("2020-03-25", "2020-07-01")

# Forward-chaining folds: (train_end, val_start, val_end). The day between
# train_end and val_start is the embargo day (its label falls in validation).
FOLDS = [
    ("2016-12-30", "2017-01-01", "2017-06-30"),
    ("2017-06-29", "2017-07-01", "2017-12-31"),
    ("2017-12-30", "2018-01-01", "2018-06-30"),
    ("2018-06-29", "2018-07-01", "2018-12-30"),
]