"""Unit tests for data ingestion & validation (Phase: setup)."""
import pandas as pd
import pytest

from predictops.config import load_config, sensor_columns
from predictops.data.ingestion import add_segments, validate_frame

CFG = load_config()


def make_frame(n=5):
    d = {"timestamp": pd.date_range("2020-02-01", periods=n, freq="10s").astype(str)}
    for c in CFG["dataset"]["analog_sensors"]:
        d[c] = [1.0] * n
    for c in CFG["dataset"]["digital_sensors"]:
        d[c] = [0] * n
    return pd.DataFrame(d)


def test_valid_frame_passes():
    assert validate_frame(make_frame()).ok


def test_empty_file_rejected():
    assert not validate_frame(make_frame(0)).ok


def test_missing_column_rejected():
    rep = validate_frame(make_frame().drop(columns=["TP2"]))
    assert not rep.ok and "TP2" in rep.errors[0]


def test_unexpected_column_is_warning_only():
    df = make_frame(); df["foo"] = 1
    rep = validate_frame(df)
    assert rep.ok and any("foo" in w for w in rep.warnings)


def test_invalid_timestamp_rejected():
    df = make_frame(); df.loc[2, "timestamp"] = "not-a-date"
    assert not validate_frame(df).ok


def test_non_numeric_sensor_rejected():
    df = make_frame(); df["TP2"] = df["TP2"].astype(object); df.loc[1, "TP2"] = "abc"
    assert not validate_frame(df).ok


def test_digital_must_be_binary():
    df = make_frame(); df.loc[0, "LPS"] = 2
    assert not validate_frame(df).ok


def test_infinite_rejected():
    df = make_frame(); df.loc[0, "TP3"] = float("inf")
    assert not validate_frame(df).ok


def test_missing_values_warn():
    df = make_frame(); df.loc[0, "TP3"] = None
    rep = validate_frame(df)
    assert rep.ok and any("missing" in w for w in rep.warnings)


def test_segments_split_on_gap():
    t = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:10", "2020-01-01 02:00:00", "2020-01-01 02:00:10"])
    out = add_segments(pd.DataFrame({"timestamp": t}), gap_seconds=600)
    assert out.segment_id.tolist() == [0, 0, 1, 1]


def test_sensor_list_matches_documentation():
    assert len(sensor_columns()) == 15
