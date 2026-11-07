"""Data-drift and prediction-drift monitoring.

Compares a CURRENT window of minutes against a fixed REFERENCE sample (training-period
normal operation, saved at training time).

* KS test (two-sample Kolmogorov–Smirnov): max distance between the two empirical CDFs.
  With thousands of minutes almost any difference is "significant", so we report the
  statistic (effect size) and use thresholds on it, not just the p-value.
* PSI (Population Stability Index): sum over bins of (cur% − ref%)·ln(cur%/ref%).
  Rule of thumb: < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 major shift.
* Prediction drift: share of minutes flagged by the anomaly model vs the 0.5% expected
  by construction on reference data.

Data drift ≠ model failure: drift says inputs changed (e.g. summer is hotter); only
new ground-truth labels can show whether performance actually degraded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                                   # (near-)constant reference
        edges = np.array([-np.inf, np.median(ref), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        a, b = reference[f].dropna().to_numpy(), current[f].dropna().to_numpy()
        if len(a) < 10 or len(b) < 10:
            continue
        ks = ks_2samp(a, b)
        p = psi(a, b)
        rows.append({"feature": f, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue), "psi": p,
                     "ref_median": float(np.median(a)), "cur_median": float(np.median(b)),
                     "status": "major" if p > 0.25 else ("moderate" if p > 0.1 else "stable")})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
