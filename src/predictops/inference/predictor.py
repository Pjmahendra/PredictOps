"""Single inference pipeline used by the API, the dashboard, replay and batch jobs.

raw 10-s sensor rows ──clean_sensor_frame──▶ build_features (same code as training)
   ──▶ anomaly detector + failure-signature detector ──▶ health engine ──▶ results

Artifacts are loaded ONCE (constructor) and never retrained during requests.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from predictops.anomaly.detectors import persistent_mask
from predictops.config import load_config, project_path
from predictops.data.ingestion import validate_frame
from predictops.engine import health
from predictops.evaluation.metrics import alert_episodes
from predictops.features.build import build_features
from predictops.preprocessing.pipeline import clean_sensor_frame


class ArtifactError(RuntimeError):
    pass


class PredictOpsPredictor:
    def __init__(self, models_dir: str | Path | None = None):
        self.cfg = load_config()
        md = Path(models_dir) if models_dir else project_path(self.cfg["paths"]["models_dir"])
        try:
            self.anom_meta = json.loads((md / "metadata" / "anomaly.json").read_text())
            self.anomaly = joblib.load(md / "trained" / f"anomaly_{self.anom_meta['selected_detector']}.joblib")
            sel = json.loads((md / "metadata" / "ml_selection.json").read_text())["selected_model"]
            self.det_meta = json.loads((md / "metadata" / f"ml_detect_{sel}.json").read_text())
            self.detector = joblib.load(md / "trained" / f"detect_{sel}.joblib")
            self.ranges = json.loads((md / "metadata" / "sensor_ranges.json").read_text())
        except FileNotFoundError as ex:
            raise ArtifactError(f"Missing model artifact: {Path(ex.filename).name}. Run the training scripts.") from ex
        self.det_features = self.det_meta["features"]
        self.det_threshold = float(self.det_meta["threshold"])
        self.hcfg = self.cfg["health_engine"]
        self._shap = None

    # ------------------------------------------------------------------ info
    @property
    def model_versions(self) -> dict:
        return {"failure_detector": self.det_meta["model_version"], "anomaly_detector": self.anom_meta["model_version"]}

    # ------------------------------------------------------------------ main entry points
    def predict_raw(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw 10-s sensor rows (CSV format) -> per-minute assessment."""
        rep = validate_frame(raw, self.cfg)
        if not rep.ok:
            raise ValueError("; ".join(rep.errors))
        df = clean_sensor_frame(raw, self.cfg)
        return self.predict_features(build_features(df))

    def predict_features(self, m: pd.DataFrame) -> pd.DataFrame:
        m = m.sort_values("timestamp").reset_index(drop=True)
        out = pd.DataFrame({"timestamp": m.timestamp})
        # A — anomaly
        a = self.anomaly.score(m)
        out["anomaly_score"] = a
        out["anomaly_severity"] = self.anomaly.severity(a)
        flagged = a >= self.anomaly.thresholds["low"]
        out["anomaly_alert"] = persistent_mask(m.timestamp, flagged,
                                               self.hcfg["alert_min_minutes"], self.hcfg["alert_window_minutes"])
        out["anomaly_status"] = np.where(out.anomaly_alert, "Anomalous", "Normal")
        # D — failure-signature detector
        p = self.detector.predict_proba(m[self.det_features])[:, 1]
        out["detector_prob"] = p
        det_ratio = p / self.det_threshold
        # S — sensor checks
        chk = health.sensor_checks(m, self.ranges)
        sensor_share = chk.mean(axis=1).to_numpy() * len(chk.columns) / max(self.hcfg["sensor_saturation"], 1)
        sensor_share = np.clip(sensor_share, 0, 1)
        # H — history: alert episodes started in the past 24 h (past-only)
        starts = out.anomaly_alert & ~out.anomaly_alert.shift(1, fill_value=False)
        out["alerts_last_24h"] = pd.Series(starts.astype(float).to_numpy(), index=m.timestamp).rolling("24h").sum().to_numpy()
        hist_level = np.clip(out.alerts_last_24h / 3, 0, 1)
        anom_level = np.maximum(out.anomaly_severity.map(health.SEVERITY_LEVEL).to_numpy(), out.anomaly_alert.astype(float))
        # combine
        out["health_score"] = health.health_score(anom_level, det_ratio, sensor_share, hist_level, self.hcfg["weights"]).round(1)
        out["health_status"] = health.classify_status(out.health_score.to_numpy(), out.anomaly_alert.to_numpy(),
                                                      det_ratio >= 1, self.hcfg["cutoffs"])
        out["failure_risk"] = health.classify_risk(out.anomaly_alert.to_numpy(), out.anomaly_severity.to_numpy(),
                                                   det_ratio, chk.any(axis=1).to_numpy())
        reasons, recs = [], []
        mm = m.assign(**{c: out[c] for c in ["anomaly_alert", "anomaly_severity", "detector_prob", "alerts_last_24h"]})
        for i in range(len(mm)):
            r, rc = health.build_reasons(mm.iloc[i], chk.iloc[i], self.det_threshold)
            reasons.append(r); recs.append(rc)
        out["reasons"] = reasons; out["recommendations"] = recs
        for c in ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current", "LPS"]:
            if c in m:
                out[c] = m[c].to_numpy()
        out["model_version_detector"] = self.det_meta["model_version"]
        out["model_version_anomaly"] = self.anom_meta["model_version"]
        out["_row"] = np.arange(len(out))
        self._last_features = m
        return out

    # ------------------------------------------------------------------ explanations
    def explain(self, m: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """SHAP (failure-signature model) + top deviating features (anomaly model)."""
        if self._shap is None:
            from predictops.explainability.shap_explain import TreeShapExplainer
            self._shap = TreeShapExplainer(self.detector)
        shap_part = self._shap.contributions(m[self.det_features], top_k)
        dev = self.anomaly.top_deviating_features(m, k=top_k)
        return [{"detector_shap": s, "anomaly_top_deviations": [{"feature": f, "robust_z": z} for f, z in d]}
                for s, d in zip(shap_part, dev)]

    @staticmethod
    def alert_list(results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate persistent anomaly minutes into alert episodes (for the alerts table)."""
        ep = alert_episodes(results.timestamp, results.anomaly_alert.to_numpy(), merge_gap_min=30, min_duration_min=5)
        rows = []
        for e in ep.itertuples():
            seg = results[(results.timestamp >= e.start) & (results.timestamp <= e.end)]
            worst = seg.loc[seg.health_score.idxmin()]
            rows.append({"start": e.start, "end": e.end, "minutes": e.minutes,
                         "max_severity": max(seg.anomaly_severity, key=lambda s: health.SEVERITY_LEVEL[s]),
                         "min_health_score": float(seg.health_score.min()), "status": worst.health_status,
                         "reasons": worst.reasons, "recommendations": worst.recommendations})
        return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def get_predictor() -> PredictOpsPredictor:
    """Process-wide cached instance (models loaded once)."""
    return PredictOpsPredictor()
