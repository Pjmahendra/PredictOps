"""Model information & performance (read from saved metadata / evaluation reports)."""
from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import ModelVersion
from predictops.config import project_path

META = project_path("models/metadata")
REPORTS = project_path("reports")


def _csv(path) -> list[dict]:
    p = REPORTS / path
    return pd.read_csv(p).astype(object).where(lambda d: d.notna(), None).to_dict(orient="records") if p.exists() else []


def model_info() -> dict:
    sel = json.loads((META / "ml_selection.json").read_text())
    det = json.loads((META / f"ml_detect_{sel['selected_model']}.json").read_text())
    an = json.loads((META / "anomaly.json").read_text())
    return {
        "failure_detector": {"version": det["model_version"], "algorithm": det["model"], "task": det["task"],
                             "threshold": det["threshold"], "n_features": len(det["features"]), "features": det["features"],
                             "trained_on": det["trained_on"], "train_period": det["train_period"],
                             "selection": sel["criterion"], "decided_before_test": sel["decided_before_test"]},
        "anomaly_detector": {"version": an["model_version"], "algorithm": an["selected_detector"],
                             "features": an["features"], "thresholds": an["thresholds"], "fit_period": an["fit_period"],
                             "calibration_period": an["calibration_period"], "alert_rule": an["alert_rule"]},
        "deep_learning": {"status": "experimental — not deployed (unstable across seeds, see docs/07)"},
        "limitations": ["Single compressor, 4 reported failures (air leaks).",
                        "Failure-signature model missed test event 4 (different pattern); anomaly model detected it.",
                        "Detection of ongoing failures, not hours-ahead prediction.",
                        "Health score is a heuristic index, not a calibrated probability."],
    }


def performance() -> dict:
    return {
        "ml_validation_selection": _csv("ml/validation_selection.csv"),
        "ml_test_detection": _csv("ml/test_results_detect.csv"),
        "ml_test_warning": _csv("ml/test_results_warning.csv"),
        "ml_tuning": _csv("ml/tuning_results.csv"),
        "anomaly": _csv("anomaly/anomaly_results.csv"),
        "deep_learning": _csv("dl/dl_results.csv"),
        "deep_learning_seeds": _csv("dl/dl_seed_stability.csv"),
        "health_engine": _csv("health/health_status_by_truth.csv"),
        "feature_engineering": _csv("feature_study/fe_comparison.csv"),
        "imbalance": _csv("feature_study/imbalance_comparison.csv"),
    }


def register_versions(db: Session):
    """Record deployed model versions in the DB (idempotent)."""
    info = model_info()
    for name, key, task, art in [("failure_detector", "failure_detector", "y_detect", "detect"),
                                 ("anomaly_detector", "anomaly_detector", "novelty detection", "anomaly")]:
        v = info[key]["version"]
        if db.scalar(select(ModelVersion).where(ModelVersion.version == v)) is None:
            db.add(ModelVersion(name=name, version=v, task=task, artifact=f"models/trained/{art}_{info[key]['algorithm']}.joblib",
                                metrics={"see": "GET /models/performance"}))
    db.commit()


def list_versions(db: Session) -> list[dict]:
    return [{"name": m.name, "version": m.version, "task": m.task, "artifact": m.artifact, "active": m.active,
             "registered_at": m.registered_at} for m in db.scalars(select(ModelVersion).order_by(ModelVersion.registered_at))]
