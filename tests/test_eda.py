"""Tests for EDA helpers that later phases reuse (operating state, frozen runs, flags)."""
import pandas as pd

from predictops import eda


def _frame(n, **cols):
    d = {"timestamp": pd.date_range("2020-04-17 23:00", periods=n, freq="10s"), "segment_id": 0}
    for c in eda.ANALOG:
        d[c] = cols.get(c, 1.0)
    return pd.DataFrame(d)


def test_operating_state_thresholds():
    df = pd.DataFrame({"Motor_current": [0.04, 3.8, 6.0, 9.0]})
    assert eda.operating_state(df).astype(str).tolist() == ["off", "offloaded", "loaded", "starting"]


def test_frozen_run_detected_only_when_long():
    df = _frame(400)  # all analogs constant for 400 rows (> 360)
    assert len(eda.frozen_runs(df)) == 1
    assert len(eda.frozen_runs(_frame(100))) == 0


def test_jittering_signal_not_frozen():
    df = _frame(400)
    df["TP3"] = [8.0 + (i % 2) * 0.002 for i in range(400)]
    assert len(eda.frozen_runs(df)) == 0


def test_failure_flags_match_config_window():
    df = _frame(10)  # 23:00 .. 23:01:30 on 17 Apr -> before event 1
    df = pd.concat([df, _frame(5).assign(timestamp=pd.date_range("2020-04-18 01:00", periods=5, freq="10s"))])
    out = eda.add_failure_flags(df.reset_index(drop=True))
    assert out.in_failure_window.tolist() == [0] * 10 + [1] * 5
    assert set(out.event_id[out.in_failure_window == 1]) == {1}
