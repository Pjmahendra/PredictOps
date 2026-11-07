"""Data ingestion and validation for MetroPT-3.

Responsibilities
----------------
1. Read the raw CSV with memory-efficient dtypes (float32 analogs, int8 digitals).
2. Validate the schema against the documented 15 sensors (no invented columns).
3. Parse timestamps, verify chronological order, detect duplicates.
4. Split the series into *segments* wherever the logger stopped for longer
   than `segment_gap_seconds`. Later rolling windows never cross a segment.
5. Save a typed Parquet copy to data/intermediate/ (raw CSV stays untouched).

The same `validate_frame` function is reused by the API to validate uploads,
so training and inference apply identical rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from predictops.config import load_config, project_path


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "n_rows": self.n_rows}


def _dtypes(cfg: dict) -> dict[str, str]:
    d = {c: "float32" for c in cfg["dataset"]["analog_sensors"]}
    d.update({c: "float32" for c in cfg["dataset"]["digital_sensors"]})  # cast to int8 after NaN check
    return d


def validate_frame(df: pd.DataFrame, cfg: dict | None = None) -> ValidationReport:
    """Check a DataFrame of sensor readings. Used for raw data AND user uploads."""
    cfg = cfg or load_config()
    ts = cfg["dataset"]["timestamp_column"]
    analog, digital = cfg["dataset"]["analog_sensors"], cfg["dataset"]["digital_sensors"]
    rep = ValidationReport(ok=True, n_rows=len(df))

    if len(df) == 0:
        rep.errors.append("File contains no rows.")
    missing = [c for c in [ts, *analog, *digital] if c not in df.columns]
    if missing:
        rep.errors.append(f"Missing required columns: {missing}")
    extra = [c for c in df.columns if c not in {ts, *analog, *digital, cfg['dataset']['index_column_raw']}]
    if extra:
        rep.warnings.append(f"Unexpected columns will be ignored: {extra}")
    if rep.errors:
        rep.ok = False
        return rep

    t = pd.to_datetime(df[ts], errors="coerce")
    if t.isna().any():
        rep.errors.append(f"{int(t.isna().sum())} invalid timestamps.")
    else:
        if not t.is_monotonic_increasing:
            rep.warnings.append("Timestamps are not sorted; they will be sorted chronologically.")
        if t.duplicated().any():
            rep.warnings.append(f"{int(t.duplicated().sum())} duplicate timestamps.")

    for c in [*analog, *digital]:
        v = pd.to_numeric(df[c], errors="coerce")
        bad = v.isna() & df[c].notna()
        if bad.any():
            rep.errors.append(f"Column '{c}' has {int(bad.sum())} non-numeric values.")
        if v.isna().any():
            rep.warnings.append(f"Column '{c}' has {int(v.isna().sum())} missing values.")
        if np.isinf(v.to_numpy(dtype="float64", na_value=np.nan)).any():
            rep.errors.append(f"Column '{c}' contains infinite values.")
    for c in digital:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if not v.isin([0, 1]).all():
            rep.errors.append(f"Digital sensor '{c}' has values other than 0/1.")

    rep.ok = not rep.errors
    return rep


def add_segments(df: pd.DataFrame, gap_seconds: int) -> pd.DataFrame:
    """Label continuous recording segments; a new segment starts after a gap > gap_seconds."""
    dt = df["timestamp"].diff().dt.total_seconds()
    df["dt_seconds"] = dt.fillna(0).astype("float32")
    df["segment_id"] = (dt > gap_seconds).cumsum().astype("int32")
    return df


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    """Read the raw CSV (read-only) into an optimised, validated DataFrame."""
    cfg = cfg or load_config()
    path = project_path(cfg["paths"]["raw_csv"])
    df = pd.read_csv(path, dtype=_dtypes(cfg))
    rep = validate_frame(df, cfg)
    if not rep.ok:
        raise ValueError(f"Raw data failed validation: {rep.errors}")
    df = df.rename(columns={cfg["dataset"]["index_column_raw"]: "raw_index"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for c in cfg["dataset"]["digital_sensors"]:
        df[c] = df[c].astype("int8")
    df["raw_index"] = df["raw_index"].astype("int64")
    return add_segments(df, cfg["dataset"]["segment_gap_seconds"])


def build_intermediate(cfg: dict | None = None) -> Path:
    """Raw CSV -> data/intermediate/metropt3.parquet (typed, segmented)."""
    cfg = cfg or load_config()
    df = load_raw(cfg)
    out = project_path(cfg["paths"]["intermediate_dir"]) / "metropt3.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load_intermediate(cfg: dict | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_parquet(project_path(cfg["paths"]["intermediate_dir"]) / "metropt3.parquet", columns=columns)
