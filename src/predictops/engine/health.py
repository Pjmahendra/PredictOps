"""Equipment Health & Failure Risk Engine.

Combines four evidence sources per minute into DISTINCT, explainable outputs:

  evidence                          source
  ------------------------------    -------------------------------------------------
  A anomaly (main)                  One-Class SVM score, severity, persistent alert
  D failure-signature detector      XGBoost P(failure in progress) vs its threshold
  S sensor checks                   key 30-min features outside the normal range
                                    (0.5–99.5% of training-normal data) + LPS active
  H history                         number of anomaly alerts in the past 24 h

Outputs
  anomaly_status   Normal / Anomalous      (Anomalous = persistent anomaly alert)
  failure_risk     Low / Medium / High     (rule-based, see `classify_risk`)
  health_score     0–100                   HEURISTIC index (100 = healthy); NOT a probability
  health_status    Healthy / Warning / Critical
  reasons          list of human-readable explanations
  recommendations  maintenance actions mapped from the reasons

Weights and cut-offs live in configs/config.yaml -> health_engine and are checked
against validation data in scripts/evaluate_health.py (docs/07).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEVERITY_LEVEL = {"normal": 0.0, "low": 0.5, "medium": 0.8, "high": 1.0}

# Sensor checks: feature -> (human name, unit)
CHECKS = {
    "TP3_mean_30min": ("panel pressure TP3 (30-min mean)", "bar"),
    "Oil_temperature_mean_30min": ("oil temperature (30-min mean)", "°C"),
    "Motor_current_mean_30min": ("motor current (30-min mean)", "A"),
    "DV_pressure_mean_30min": ("dryer discharge pressure DV (30-min mean)", "bar"),
    "duty_cycle_30min": ("compressor duty cycle (30 min)", "share loaded"),
    "oil_temp_vs_24h_baseline": ("oil temperature vs 24 h baseline", "°C"),
    "tp3_vs_24h_baseline": ("TP3 vs 24 h baseline", "bar"),
}

RECOMMENDATIONS = {
    "air_leak_pattern": "Inspect the air circuit for leaks: the compressor is running loaded almost continuously.",
    "low_pressure": "Check pneumatic panel / reservoir pressure and look for leaks (pressure below normal range).",
    "lps": "Low-pressure switch active (<7 bar): inspect immediately for major leaks or valve faults.",
    "oil_hot": "Check oil level and cooling (oil temperature above its normal range / recent baseline).",
    "motor": "Investigate motor-current behaviour (outside normal operating range).",
    "dv": "Check the air-dryer towers and discharge valve (DV pressure abnormal).",
    "detector": "Pattern matches previously reported air-leak failures: schedule an inspection.",
    "persistent_anomaly": "Persistent abnormal behaviour: schedule an equipment inspection.",
    "history": "Repeated anomaly alerts in the last 24 h: plan preventive maintenance.",
    "ok": "Continue normal monitoring.",
}


def sensor_checks(m: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """Return per-row booleans: is the feature below/above its normal range."""
    out = {}
    for f in CHECKS:
        lo, hi = ranges[f]["p005"], ranges[f]["p995"]
        out[f"{f}__low"] = m[f] < lo
        out[f"{f}__high"] = m[f] > hi
    out["lps_active"] = m.get("LPS", pd.Series(0, index=m.index)) > 0.5
    return pd.DataFrame(out, index=m.index)


def health_score(anom_level, det_ratio, sensor_share, history_level, w: dict) -> np.ndarray:
    """100 − weighted penalties. Each component is in [0, 1]."""
    pen = (w["anomaly"] * anom_level + w["detector"] * np.clip(det_ratio, 0, 1)
           + w["sensors"] * sensor_share + w["history"] * history_level)
    return np.round(np.clip(100 * (1 - pen), 0, 100), 1)


def classify_status(score, alert, det_hit, cut: dict) -> np.ndarray:
    st = np.where(score >= cut["healthy_min"], "Healthy", np.where(score >= cut["warning_min"], "Warning", "Critical"))
    st = np.where((alert | det_hit) & (st == "Healthy"), "Warning", st)       # any confirmed alert -> at least Warning
    st = np.where(alert & det_hit, "Critical", st)                             # both engines agree -> Critical
    return st


def classify_risk(alert, severity, det_ratio, sensor_any) -> np.ndarray:
    high = (det_ratio >= 1) | (alert & np.isin(severity, ["medium", "high"]))
    medium = alert | (severity != "normal") | (det_ratio >= 0.5) | sensor_any
    return np.where(high, "High", np.where(medium, "Medium", "Low"))


def build_reasons(row: pd.Series, checks_row: pd.Series, det_threshold: float) -> tuple[list[str], list[str]]:
    reasons, keys = [], []
    if row.anomaly_alert:
        reasons.append(f"Persistent anomaly: score above the normal-operation threshold for ≥10 of the last 30 min "
                       f"(severity {row.anomaly_severity}).")
        keys.append("persistent_anomaly")
    elif row.anomaly_severity != "normal":
        reasons.append(f"Unusual reading (severity {row.anomaly_severity}), not yet persistent.")
    if row.detector_prob >= det_threshold:
        reasons.append(f"Failure-signature model: P={row.detector_prob:.2f} ≥ threshold {det_threshold:.2f} "
                       "(matches previously reported air-leak pattern).")
        keys.append("detector")
    for f, (name, unit) in CHECKS.items():
        v = row.get(f, np.nan)
        if checks_row[f"{f}__low"]:
            reasons.append(f"{name} is below its normal range ({v:.2f} {unit}).")
        if checks_row[f"{f}__high"]:
            reasons.append(f"{name} is above its normal range ({v:.2f} {unit}).")
    if checks_row["lps_active"]:
        reasons.append("Low-pressure switch (LPS) is active."); keys.append("lps")
    hi = lambda f: bool(checks_row[f"{f}__high"]); lo = lambda f: bool(checks_row[f"{f}__low"])
    if hi("duty_cycle_30min") or (hi("DV_pressure_mean_30min") and row.get("H1_mean_30min", 9) < 0.5):
        keys.append("air_leak_pattern")
    if lo("TP3_mean_30min") or lo("tp3_vs_24h_baseline"):
        keys.append("low_pressure")
    if hi("Oil_temperature_mean_30min") or hi("oil_temp_vs_24h_baseline"):
        keys.append("oil_hot")
    if hi("Motor_current_mean_30min") or lo("Motor_current_mean_30min"):
        keys.append("motor")
    if hi("DV_pressure_mean_30min"):
        keys.append("dv")
    if row.alerts_last_24h >= 2:
        reasons.append(f"{int(row.alerts_last_24h)} anomaly alerts in the last 24 h."); keys.append("history")
    if not reasons:
        reasons.append("All indicators within normal operating ranges.")
    recs = [RECOMMENDATIONS[k] for k in dict.fromkeys(keys)] or [RECOMMENDATIONS["ok"]]
    return reasons, recs
