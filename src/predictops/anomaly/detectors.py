"""Anomaly detection (semi-supervised / novelty detection).

Idea: learn what NORMAL operation looks like from known-normal training minutes only,
then flag minutes that do not fit. No failure labels are used to train or to choose
features or thresholds — labels are used only afterwards, to evaluate.

Detectors
  isolation_forest  isolates points with random splits; anomalies need fewer splits
  lof               Local Outlier Factor (novelty mode): density relative to neighbours
  ocsvm             One-Class SVM: smallest region (in kernel space) holding normal data
  robust_z          statistical: max |x - median| / (1.4826 * MAD) over features

Threshold: the q-quantile of each detector's scores on the training-normal data
(default 99.5% -> by construction ~0.5% of normal minutes are flagged). Severity
levels use higher quantiles. Persistence rules turn flagged minutes into alerts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from predictops.preprocessing.pipeline import make_preprocessor

# Label-free, physically motivated features (NOT the supervised selection, which used labels)
ANOMALY_FEATURES = [
    "TP2_mean_30min", "TP3_mean_30min", "H1_mean_30min", "DV_pressure_mean_30min",
    "Oil_temperature_mean_30min", "Motor_current_mean_30min",
    "TP3_std_30min", "Oil_temperature_std_30min", "Motor_current_std_30min", "DV_pressure_std_30min",
    "duty_cycle_30min", "load_cycles_per_h_2h", "oil_temp_loaded_2h",
    "oil_temp_vs_24h_baseline", "tp3_vs_24h_baseline", "motor_current_vs_24h_baseline",
    "TP3_range_30min", "lps_share_30min",
]
SEED = 42
SEVERITY_QUANTILES = {"low": 0.995, "medium": 0.999, "high": 0.9999}


@dataclass
class AnomalyDetector:
    kind: str
    features: list[str] = field(default_factory=lambda: list(ANOMALY_FEATURES))
    max_train_rows: int | None = None
    pre: object = None
    model: object = None
    median_: np.ndarray | None = None
    mad_: np.ndarray | None = None
    thresholds: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- fit
    def fit(self, normal: pd.DataFrame, calib: pd.DataFrame | None = None) -> "AnomalyDetector":
        """Fit on known-normal rows. Thresholds come from `calib` (a later, separate
        slice of normal data) so they are not computed on the very rows the model memorised."""
        X = normal[self.features]
        if self.max_train_rows and len(X) > self.max_train_rows:
            X = X.sample(self.max_train_rows, random_state=SEED)
        self.pre = make_preprocessor(self.features).fit(X)
        Z = self.pre.transform(X).to_numpy()
        if self.kind == "isolation_forest":
            self.model = IsolationForest(n_estimators=300, max_samples=4096, random_state=SEED, n_jobs=-1).fit(Z)
        elif self.kind == "lof":
            self.model = LocalOutlierFactor(n_neighbors=50, novelty=True, n_jobs=-1).fit(Z)
        elif self.kind == "ocsvm":
            self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(Z)
        elif self.kind == "robust_z":
            self.median_ = np.nanmedian(Z, axis=0)
            mad = np.nanmedian(np.abs(Z - self.median_), axis=0) * 1.4826
            # features that are constant most of the time have MAD≈0 -> fall back to std
            self.mad_ = np.where(mad < 1e-3, np.nanstd(Z, axis=0) + 1e-6, mad)
        else:
            raise ValueError(self.kind)
        ref = calib if calib is not None else normal
        ref_scores = self.score(ref)
        self.thresholds = {lvl: float(np.nanquantile(ref_scores, q)) for lvl, q in SEVERITY_QUANTILES.items()}
        return self

    # ---------------------------------------------------------------- score (higher = more anomalous)
    def _z(self, df):
        Z = self.pre.transform(df[self.features]).to_numpy()
        return (Z - self.median_) / self.mad_ if self.kind == "robust_z" else Z

    def score(self, df: pd.DataFrame) -> np.ndarray:
        Z = self._z(df)
        if self.kind == "robust_z":
            return np.nanmax(np.abs(Z), axis=1)
        return -self.model.score_samples(Z)

    def severity(self, scores: np.ndarray) -> np.ndarray:
        sev = np.full(len(scores), "normal", dtype=object)
        for lvl in ["low", "medium", "high"]:
            sev[scores >= self.thresholds[lvl]] = lvl
        return sev

    def top_deviating_features(self, df: pd.DataFrame, k: int = 3) -> list[list[tuple[str, float]]]:
        """Per row: the k features furthest from the normal median (in robust z units).
        A simple, model-agnostic explanation of WHICH measurements look abnormal."""
        if self.median_ is None:  # compute robust stats lazily for any detector
            Z0 = self.pre.transform(df[self.features]).to_numpy()
            self.median_ = np.zeros(Z0.shape[1]); self.mad_ = np.ones(Z0.shape[1])
        Z = self.pre.transform(df[self.features]).to_numpy()
        z = (Z - self.median_) / self.mad_
        out = []
        for row in z:
            order = np.argsort(-np.abs(np.nan_to_num(row)))[:k]
            out.append([(self.features[i], float(row[i])) for i in order])
        return out


def persistent_mask(timestamps: pd.Series, flagged: np.ndarray, min_minutes: int = 10, window_minutes: int = 30) -> np.ndarray:
    """Persistent anomaly: >= min_minutes flagged within the trailing window (past-only).
    Removes isolated point anomalies (single odd minutes) from alerting."""
    s = pd.Series(np.asarray(flagged, dtype=float), index=pd.to_datetime(timestamps))
    return (s.rolling(f"{window_minutes}min").sum() >= min_minutes).to_numpy()
