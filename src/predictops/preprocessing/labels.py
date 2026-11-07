"""Quality flags and target construction.

Everything here is derived from (a) the sensor data and (b) the official failure
windows in configs/config.yaml. No label is invented: rows whose status cannot be
known are marked `uncertain` and excluded from training/evaluation, never relabelled.

Columns added
-------------
is_frozen          1 if all analog sensors repeat an identical value for >= 1 h (stale logger)
operating_state    off / offloaded / loaded / starting (from motor current, PDF levels)
in_failure_window  1 inside a reported failure window (ground truth, verbatim from PDF)
event_id           1..4 inside a window, else 0
uncertain          1 if the failure signature is sustained OUTSIDE reported windows
                   (possible unreported failure — status unknown)
y_detect           TARGET A "failure in progress": = in_failure_window
y_warn_{H}h        TARGET B "a reported failure starts within the next H hours":
                   1 for rows in [start - H, start); 0 for normal rows;
                   NaN (undefined) for rows already inside a window
minutes_to_onset   minutes until next reported onset (for lead-time analysis ONLY —
                   it uses the future and must NEVER be used as a model input)
usable             1 if the row may be used for training/evaluation (not frozen, not uncertain)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from predictops import eda
from predictops.config import load_config

FORBIDDEN_AS_FEATURES = {
    "in_failure_window", "event_id", "uncertain", "usable", "minutes_to_onset",
    "raw_index", "timestamp", "segment_id", "dt_seconds",
}


def add_frozen_flag(df: pd.DataFrame, min_rows: int = 360) -> pd.DataFrame:
    df["is_frozen"] = np.int8(0)
    for r in eda.frozen_runs(df, min_rows=min_rows).itertuples():
        df.loc[df.timestamp.between(r.start, r.end), "is_frozen"] = 1
    return df


def add_uncertain_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Sustained failure signature outside reported windows -> uncertain."""
    df["uncertain"] = np.int8(0)
    for r in eda.signature_episodes(df).itertuples():
        m = df.timestamp.between(r.start, r.end) & (df.in_failure_window == 0)
        df.loc[m, "uncertain"] = 1
    return df


def add_targets(df: pd.DataFrame, horizons_h: list[int], cfg: dict) -> pd.DataFrame:
    df["y_detect"] = df["in_failure_window"].astype("int8")
    onsets = sorted(pd.Timestamp(ev["start"]) for ev in cfg["failure_events"])

    # minutes until the next onset (analysis only, never a feature)
    t = df.timestamp.to_numpy()
    on = np.array(onsets, dtype="datetime64[ns]")
    idx = np.searchsorted(on, t, side="right")          # next onset strictly after t
    nxt = np.where(idx < len(on), on[np.minimum(idx, len(on) - 1)], np.datetime64("NaT", "ns"))
    df["minutes_to_onset"] = ((nxt - t) / np.timedelta64(1, "m")).astype("float32")

    for h in horizons_h:
        col = f"y_warn_{h}h"
        y = ((df.minutes_to_onset > 0) & (df.minutes_to_onset <= h * 60)).astype("float32")
        y[df.in_failure_window == 1] = np.nan             # already failing: undefined
        df[col] = y
    return df


def build_labeled(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = eda.add_failure_flags(df, cfg)
    df["operating_state"] = eda.operating_state(df).astype(str)
    df = add_frozen_flag(df)
    df = add_uncertain_flag(df)
    df = add_targets(df, cfg["targets"]["warning_horizons_hours"], cfg)
    df["usable"] = ((df.is_frozen == 0) & (df.uncertain == 0)).astype("int8")
    return df
