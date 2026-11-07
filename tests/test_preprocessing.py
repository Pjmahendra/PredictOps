"""Stage 4b tests: targets, splits, leakage safety, cleaning, saved artifacts."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from predictops.config import load_config, project_path
from predictops.data.ingestion import validate_frame
from predictops.preprocessing.labels import FORBIDDEN_AS_FEATURES, add_targets
from predictops.preprocessing.pipeline import clean_sensor_frame, make_preprocessor

CFG = load_config()
PROCESSED = project_path(CFG["paths"]["processed_dir"]) / "metropt3_labeled.parquet"
needs_data = pytest.mark.skipif(not PROCESSED.exists(), reason="run scripts/build_processed.py first")


# ---------------------------------------------------------------- synthetic unit tests
def _toy_cfg():
    return {"failure_events": [{"id": 1, "start": "2020-01-01 12:00:00", "end": "2020-01-01 13:00:00"}]}


def test_warning_target_only_before_onset():
    t = pd.date_range("2020-01-01 08:00", "2020-01-01 14:00", freq="10min")
    df = pd.DataFrame({"timestamp": t})
    df["in_failure_window"] = df.timestamp.between("2020-01-01 12:00", "2020-01-01 13:00").astype(int)
    out = add_targets(df, [2], _toy_cfg())
    pos = out[out.y_warn_2h == 1].timestamp
    assert pos.min() >= pd.Timestamp("2020-01-01 10:00") and pos.max() < pd.Timestamp("2020-01-01 12:00")
    assert out.loc[out.in_failure_window == 1, "y_warn_2h"].isna().all()   # undefined while failing
    assert (out.loc[out.timestamp > "2020-01-01 13:00", "y_warn_2h"] == 0).all()


def test_future_columns_are_forbidden_as_features():
    assert {"minutes_to_onset", "in_failure_window", "event_id", "timestamp"} <= FORBIDDEN_AS_FEATURES


def test_clean_fills_only_short_gaps_and_handles_inf():
    cfg = CFG
    t = pd.date_range("2020-01-01", periods=20, freq="10s")
    d = {"timestamp": t.astype(str)}
    for c in cfg["dataset"]["analog_sensors"]:
        d[c] = np.arange(20, dtype=float)
    for c in cfg["dataset"]["digital_sensors"]:
        d[c] = 1.0
    df = pd.DataFrame(d)
    df.loc[3, "TP2"] = np.nan                    # 1 missing reading -> filled
    df.loc[8:17, "TP3"] = np.nan                 # 100 s missing -> only first 6 (60 s) filled
    df.loc[5, "H1"] = np.inf
    out = clean_sensor_frame(df.sample(frac=1, random_state=0), cfg)   # shuffled input
    assert out.timestamp.is_monotonic_increasing
    assert out.TP2.isna().sum() == 0
    assert out.TP3.isna().sum() == 4
    assert out.loc[5, "H1"] == out.loc[4, "H1"]   # inf -> NaN -> forward-filled


def test_preprocessor_learns_from_fit_data_only():
    train = pd.DataFrame({"a": [0.0, 2.0, 4.0]})
    test = pd.DataFrame({"a": [100.0, 200.0]})
    pre = make_preprocessor(["a"]).fit(train)
    sc = pre.named_transformers_["num"].named_steps["scale"]
    assert sc.mean_[0] == pytest.approx(2.0)          # test data had no influence
    assert pre.transform(test)["a"].iloc[0] > 10      # test values are far out-of-range, not re-centred


# ---------------------------------------------------------------- real-data checks
@needs_data
def test_splits_are_chronological_and_disjoint():
    df = pd.read_parquet(PROCESSED, columns=["timestamp", "split"])
    tr, va, te = (df[df.split == s].timestamp for s in ["train", "validation", "test"])
    assert tr.max() < va.min() and va.max() < te.min()
    emb = pd.Timedelta(hours=CFG["splits"]["embargo_hours"])
    assert va.min() - tr.max() >= emb and te.min() - va.max() >= emb


@needs_data
def test_test_split_holds_only_event_4():
    df = pd.read_parquet(PROCESSED, columns=["event_id", "split"])
    assert set(df.loc[(df.split == "test") & (df.event_id > 0), "event_id"]) == {4}
    assert 4 not in set(df.loc[df.split != "test", "event_id"])


@needs_data
def test_flags_consistent():
    df = pd.read_parquet(PROCESSED, columns=["is_frozen", "uncertain", "usable", "in_failure_window", "y_warn_12h"])
    assert ((df.uncertain == 1) & (df.in_failure_window == 1)).sum() == 0
    assert (df.usable == ((df.is_frozen == 0) & (df.uncertain == 0))).all()
    assert df.loc[df.in_failure_window == 1, "y_warn_12h"].isna().all()


@needs_data
def test_forward_chaining_folds_never_train_on_future():
    from predictops.preprocessing.splits import event_folds
    df = pd.read_parquet(PROCESSED, columns=["timestamp", "split"])
    for _, tr, va in event_folds(df, CFG):
        assert df.loc[tr, "timestamp"].max() < df.loc[va, "timestamp"].min()


@needs_data
def test_saved_preprocessor_matches_train_statistics():
    meta = json.loads(project_path("models/metadata/preprocessing.json").read_text())
    pre = joblib.load(project_path("models/preprocessing/base_preprocessor.joblib"))
    df = pd.read_parquet(project_path("data/train/sensors.parquet"))
    tr = df[df.usable == 1]
    sc = pre.named_transformers_["num"].named_steps["scale"]
    np.testing.assert_allclose(sc.mean_, tr[meta["numeric_features"]].mean().to_numpy(), rtol=1e-4)


@needs_data
@pytest.mark.parametrize("name", ["event4_2020-07-15.csv", "normal_day_2020-08-10.csv", "small_2h_2020-08-10.csv"])
def test_sample_files_are_valid_raw_format(name):
    df = pd.read_csv(project_path(CFG["paths"]["sample_dir"]) / name)
    assert validate_frame(df).ok and len(df) > 0
