import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import build_dataset
from features import build_features, feature_columns

DATA = build_dataset()
FULL = build_features(DATA)


def test_features_ignore_future():
    cols = feature_columns(FULL)
    for cutoff in ["2016-06-15", "2018-03-10", "2019-07-04"]:
        cut = pd.Timestamp(cutoff)
        truncated = build_features(DATA[DATA["Date"] <= cut])
        expected = FULL[FULL["Date"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected[cols], truncated[cols])
        