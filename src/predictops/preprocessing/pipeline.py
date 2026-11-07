"""Leakage-safe preprocessing pipeline (scikit-learn), shared by training and inference.

* `clean_sensor_frame` — row-level cleaning applied identically at train and inference
  time: sort, drop duplicate timestamps, coerce numerics, replace ±inf with NaN,
  clip digital sensors to {0,1}, and fill ONLY short gaps (<= max_fill_seconds) by
  forward-fill within a segment. Longer gaps stay NaN and go to the imputer.
* `make_preprocessor` — SimpleImputer(median) + scaler. It is an *unfitted*
  object: it must be `.fit()` on TRAINING rows only, then saved with joblib and
  re-used unchanged on validation/test/inference data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from predictops.config import load_config
from predictops.data.ingestion import add_segments


def clean_sensor_frame(df: pd.DataFrame, cfg: dict | None = None, max_fill_seconds: int = 60) -> pd.DataFrame:
    cfg = cfg or load_config()
    analog, digital = cfg["dataset"]["analog_sensors"], cfg["dataset"]["digital_sensors"]
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
    for c in analog + digital:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    for c in digital:
        df[c] = df[c].clip(0, 1).round()
    if "segment_id" not in df:
        df = add_segments(df.reset_index(drop=True), cfg["dataset"]["segment_gap_seconds"])
    limit = max(1, max_fill_seconds // cfg["dataset"]["nominal_sampling_seconds"])
    df[analog + digital] = df.groupby("segment_id")[analog + digital].ffill(limit=limit)
    return df.reset_index(drop=True)


def make_preprocessor(numeric_cols: list[str], binary_cols: list[str] | None = None,
                      scaler: str = "standard") -> ColumnTransformer:
    """Imputer + scaler for numeric features; imputer only for binary ones.

    StandardScaler is the default. RobustScaler (median/IQR) was tried first and
    REJECTED: TP2 and DV_pressure sit at ~0 bar for >50% of the time, so their IQR is
    0.004 bar and a normal 10-bar reading would be scaled to ~2,500 (docs/03 §5)."""
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    parts = [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", sc)]), numeric_cols)]
    if binary_cols:
        parts.append(("bin", SimpleImputer(strategy="most_frequent"), binary_cols))
    return ColumnTransformer(parts, verbose_feature_names_out=False).set_output(transform="pandas")
