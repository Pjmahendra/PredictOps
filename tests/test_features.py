"""Feature engineering tests — the key one proves features use PAST data only."""
import numpy as np
import pandas as pd
import pytest

from predictops.config import project_path
from predictops.features.build import LABEL_COLS, aggregate_minutes, build_features, feature_columns

PROCESSED = project_path("data/processed/metropt3_labeled.parquet")
needs_data = pytest.mark.skipif(not PROCESSED.exists(), reason="run scripts/build_processed.py first")


@pytest.fixture(scope="module")
def slice10s():
    df = pd.read_parquet(PROCESSED)
    return df[df.timestamp.between("2020-06-04", "2020-06-06")].reset_index(drop=True)


@needs_data
def test_features_are_past_only(slice10s):
    """Features at time <= T must be IDENTICAL whether or not data after T exists."""
    cut = pd.Timestamp("2020-06-05 12:00")
    full = build_features(slice10s)
    prefix = build_features(slice10s[slice10s.timestamp <= cut])
    cols = feature_columns(full)
    a = full[full.timestamp <= cut].set_index("timestamp")[cols]
    b = prefix.set_index("timestamp")[cols]
    common = a.index.intersection(b.index)[:-1]          # last minute may be partial in the prefix
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common], check_exact=False, rtol=1e-4)


@needs_data
def test_no_label_or_id_in_features(slice10s):
    cols = set(feature_columns(build_features(slice10s)))
    assert not cols & (set(LABEL_COLS) | {"timestamp", "segment_id", "minutes_to_onset"})


def test_minute_aggregation_right_closed_interval():
    t = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:10", "2020-01-01 00:00:50", "2020-01-01 00:01:00"])
    d = pd.DataFrame({"timestamp": t, "segment_id": 0})
    for c in ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current", "COMP", "DV_eletric",
              "Towers", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses", "Reservoirs"]:
        d[c] = [1.0, 2.0, 3.0, 4.0]
    m = aggregate_minutes(d)
    # 00:00:00 belongs to the minute ending 00:00; 00:00:10..00:01:00 to the minute ending 00:01
    assert m.timestamp.tolist() == [pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 00:01")]
    assert m.TP3.tolist() == [1.0, 3.0]


@needs_data
def test_rolling_never_crosses_segments(slice10s):
    m = build_features(slice10s)
    first = m.groupby("segment_id").head(1)
    assert (first["minutes_in_segment"] == 0).all()
    assert first["TP3_diff_5min"].isna().all()           # no lag from a previous segment
