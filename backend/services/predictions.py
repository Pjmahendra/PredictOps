"""Business logic: run the inference pipeline, persist runs / predictions / alerts."""
from __future__ import annotations

import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.db import Alert, Dataset, Prediction, PredictionRun
from backend.services.datasets import default_equipment
from backend.settings import Settings
from predictops.inference.predictor import PredictOpsPredictor


def _clip_window(raw: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is None and end is None:
        return raw
    t = pd.to_datetime(raw["timestamp"], errors="coerce")
    keep = pd.Series(True, index=raw.index)
    if start is not None:
        keep &= t >= pd.Timestamp(start) - pd.Timedelta(hours=2)   # 2 h warm-up for rolling features
    if end is not None:
        keep &= t <= pd.Timestamp(end)
    return raw[keep]


def run_and_store(raw: pd.DataFrame, predictor: PredictOpsPredictor, db: Session, settings: Settings,
                  source: str, dataset: Dataset | None = None, start=None, end=None):
    raw = _clip_window(raw, start, end)
    try:
        res = predictor.predict_raw(raw)
    except ValueError as ex:                                     # validation errors -> 422 with reasons
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Invalid sensor data: {ex}")
    if start is not None:
        res = res[res.timestamp >= pd.Timestamp(start)]
    if res.empty:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "No complete minutes could be assessed (too little data in the selected window).")
    if len(res) > settings.max_predict_minutes:
        raise HTTPException(413,
                            f"Request covers {len(res)} minutes; limit is {settings.max_predict_minutes}.")
    eq = default_equipment(db)
    summary = {
        "status_counts": res.health_status.value_counts().to_dict(),
        "risk_counts": res.failure_risk.value_counts().to_dict(),
        "anomalous_minutes": int((res.anomaly_status == "Anomalous").sum()),
        "min_health_score": float(res.health_score.min()), "mean_health_score": float(res.health_score.mean()),
    }
    run = PredictionRun(dataset_id=dataset.id if dataset else None, equipment_id=eq.id, source=source, n_minutes=len(res),
                        start_time=res.timestamp.min().to_pydatetime(), end_time=res.timestamp.max().to_pydatetime(),
                        model_version_detector=predictor.model_versions["failure_detector"],
                        model_version_anomaly=predictor.model_versions["anomaly_detector"], summary=summary)
    db.add(run); db.flush()
    db.bulk_insert_mappings(Prediction, [
        {"run_id": run.id, "equipment_id": eq.id, "timestamp": r.timestamp.to_pydatetime(), "health_score": float(r.health_score),
         "health_status": r.health_status, "failure_risk": r.failure_risk, "anomaly_status": r.anomaly_status,
         "anomaly_score": float(r.anomaly_score), "anomaly_severity": r.anomaly_severity,
         "detector_prob": float(r.detector_prob), "reasons": r.reasons, "recommendations": r.recommendations}
        for r in res.itertuples()])
    alerts = []
    for a in predictor.alert_list(res).itertuples():
        al = Alert(run_id=run.id, equipment_id=eq.id, start_time=a.start.to_pydatetime(), end_time=a.end.to_pydatetime(),
                   minutes=int(a.minutes), severity=a.max_severity, status=a.status, min_health_score=a.min_health_score,
                   reasons=a.reasons, recommendations=a.recommendations)
        db.add(al); alerts.append(al)
    db.commit(); db.refresh(run)
    for al in alerts:
        db.refresh(al)
    return run, res, alerts


def latest_assessment(res: pd.DataFrame) -> dict:
    r = res.iloc[-1]
    return {k: (r[k].to_pydatetime() if k == "timestamp" else r[k]) for k in
            ["timestamp", "health_score", "health_status", "failure_risk", "anomaly_status", "anomaly_score",
             "anomaly_severity", "detector_prob", "reasons", "recommendations"]}


def get_run(db: Session, run_id: int) -> PredictionRun:
    run = db.get(PredictionRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found.")
    return run


def predictions_frame(db: Session, run_id: int | None = None, start=None, end=None, limit: int = 20_000) -> pd.DataFrame:
    q = select(Prediction)
    if run_id is not None:
        q = q.where(Prediction.run_id == run_id)
    if start is not None:
        q = q.where(Prediction.timestamp >= start)
    if end is not None:
        q = q.where(Prediction.timestamp <= end)
    rows = db.scalars(q.order_by(Prediction.timestamp.desc()).limit(limit)).all()
    cols = ["run_id", "timestamp", "health_score", "health_status", "failure_risk", "anomaly_status", "anomaly_score",
            "anomaly_severity", "detector_prob", "reasons", "recommendations"]
    return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows], columns=cols).iloc[::-1].reset_index(drop=True)


def delete_run(db: Session, run_id: int):
    get_run(db, run_id)
    db.execute(delete(PredictionRun).where(PredictionRun.id == run_id)); db.commit()
