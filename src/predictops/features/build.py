"""Time-series feature engineering (shared by training AND inference).

Two steps
---------
1. `aggregate_minutes`: 10-s rows -> 1-minute rows. Each minute row is labelled with
   the END of its interval, (t-1min, t], so a feature at time t only uses data <= t.
   Done per segment: a minute never mixes two recording segments.
2. `add_rolling_features`: past-only rolling statistics on the minute series,
   computed within segments (except the 24 h baseline — see below).

Why 1-minute resolution?  ~250k rows instead of 1.5M makes SVM / LOF / SHAP feasible
on a laptop, while 1 min is still far finer than our shortest horizon (2 h).

Leakage rules (enforced in tests/test_features.py)
  * pandas time-based rolling windows are right-closed, i.e. they include the current
    row and the PAST only; we never use center=True or negative shifts.
  * Labels/targets are carried along but are never in `feature_columns()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predictops.config import load_config

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
WINDOWS = {"5min": 5, "30min": 30, "2h": 120}          # name -> minutes
LABEL_COLS = ["in_failure_window", "event_id", "uncertain", "is_frozen", "usable", "y_detect",
              "y_warn_2h", "y_warn_6h", "y_warn_12h", "minutes_to_onset", "split"]


# --------------------------------------------------------------------- step 1
def aggregate_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """10 s rows -> 1 min rows (per segment). Works with or without label columns."""
    d = df.copy()
    d["t"] = d.timestamp.dt.floor("min") + pd.Timedelta(minutes=1)   # interval end
    d.loc[d.timestamp == d.timestamp.dt.floor("min"), "t"] = d.timestamp  # exact minute -> belongs to (t-1, t]
    # raw-level helpers
    d["load_start"] = (d.groupby("segment_id")["DV_eletric"].diff() == 1).astype("int8")
    loaded = d.Motor_current.between(5.5, 8.5)
    d["oil_loaded"] = d.Oil_temperature.where(loaded)
    d["tp3_minus_res"] = d.TP3 - d.Reservoirs if "Reservoirs" in d else np.nan

    g = d.groupby(["segment_id", "t"], sort=True)
    agg = {c: "mean" for c in ANALOG + DIGITAL}
    agg.update({"load_start": "sum", "oil_loaded": "mean", "tp3_minus_res": "mean", "timestamp": "size"})
    out = g.agg(agg).rename(columns={"timestamp": "n_readings"})
    for c in ["TP3", "Motor_current", "Oil_temperature"]:
        out[f"{c}_min1"] = g[c].min()
        out[f"{c}_max1"] = g[c].max()

    present = [c for c in LABEL_COLS if c in d]
    if present:
        last = g[present].last()                                   # label at time t
        if "usable" in present:
            last["usable"] = g["usable"].min()                    # minute usable only if all rows are
        out = out.join(last)
    out = out.reset_index().rename(columns={"t": "timestamp"})
    return out


# --------------------------------------------------------------------- step 2
def _run_length(mask: pd.Series, seg: pd.Series) -> pd.Series:
    """Number of consecutive minutes (ending now) where mask is True, within a segment."""
    m = mask.astype(int)
    breaks = ((m == 0) | (seg != seg.shift())).cumsum()
    return m.groupby(breaks).cumsum().astype("float32")


def add_rolling_features(m: pd.DataFrame) -> pd.DataFrame:
    m = m.sort_values("timestamp").reset_index(drop=True)
    seg = m.segment_id
    idx = m.set_index("timestamp")
    by_seg = idx.groupby("segment_id", sort=False)
    feats: dict[str, pd.Series] = {}

    def roll(col, win, fn, minp_frac=0.5):
        r = by_seg[col].rolling(win, min_periods=max(1, int(WINDOWS[win] * minp_frac)))
        return getattr(r, fn)().reset_index(level=0, drop=True).to_numpy()

    for c in ANALOG:
        for w in WINDOWS:
            feats[f"{c}_mean_{w}"] = roll(c, w, "mean")
            feats[f"{c}_std_{w}"] = roll(c, w, "std")
            if w != "5min":
                mx, mn = roll(c, w, "max"), roll(c, w, "min")
                feats[f"{c}_max_{w}"], feats[f"{c}_min_{w}"] = mx, mn
                feats[f"{c}_range_{w}"] = mx - mn
        feats[f"{c}_median_2h"] = roll(c, "2h", "median")
    for c in ["TP3", "Oil_temperature", "Motor_current", "TP2"]:
        feats[f"{c}_skew_2h"] = roll(c, "2h", "skew")
        feats[f"{c}_kurt_2h"] = roll(c, "2h", "kurt")

    # compressor behaviour
    for w in ["30min", "2h"]:
        feats[f"duty_cycle_{w}"] = roll("DV_eletric", w, "mean")
        feats[f"load_cycles_per_h_{w}"] = roll("load_start", w, "sum") * 60 / WINDOWS[w]
        feats[f"oil_temp_loaded_{w}"] = roll("oil_loaded", w, "mean", minp_frac=0.0)
        feats[f"lps_share_{w}"] = roll("LPS", w, "mean")
    feats["minutes_continuously_loaded"] = _run_length(m.DV_eletric > 0.95, seg).to_numpy()
    feats["minutes_h1_near_zero"] = _run_length(m.H1 < 0.5, seg).to_numpy()
    feats["minutes_in_segment"] = m.groupby("segment_id").cumcount().astype("float32").to_numpy()

    # lags / rate of change (within segment; past only)
    for c in ["TP3", "Oil_temperature", "Motor_current"]:
        for lag in (5, 30):
            prev = m.groupby("segment_id")[c].shift(lag)
            feats[f"{c}_diff_{lag}min"] = (m[c] - prev).to_numpy()
            feats[f"{c}_pctchg_{lag}min"] = ((m[c] - prev) / prev.abs().replace(0, np.nan)).to_numpy()

    # deviation from a 24 h past baseline (crosses gaps on purpose: it is a slow-moving
    # reference of PAST behaviour, used to cancel the seasonal oil-temperature drift, docs/02 §9)
    base = idx[["Oil_temperature", "TP3", "Motor_current"]].rolling("24h", min_periods=60).median()
    feats["oil_temp_vs_24h_baseline"] = feats["Oil_temperature_mean_30min"] - base.Oil_temperature.to_numpy()
    feats["tp3_vs_24h_baseline"] = feats["TP3_mean_30min"] - base.TP3.to_numpy()
    feats["motor_current_vs_24h_baseline"] = feats["Motor_current_mean_30min"] - base.Motor_current.to_numpy()

    f = pd.DataFrame(feats, index=m.index).astype("float32")
    f = f.replace([np.inf, -np.inf], np.nan)
    return pd.concat([m, f], axis=1)


def feature_columns(m: pd.DataFrame) -> list[str]:
    """All model-input columns: everything except identifiers and labels."""
    exclude = set(LABEL_COLS) | {"timestamp", "segment_id", "n_readings", "load_start", "oil_loaded"}
    return [c for c in m.columns if c not in exclude]


def build_features(df10s: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline used identically by training and by the API."""
    return add_rolling_features(aggregate_minutes(df10s))
