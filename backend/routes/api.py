"""HTTP routes. Thin layer: validate input -> call a service -> return a schema.

Groups (tags): system, datasets, predictions, explainability, alerts, dashboard, monitoring, reports.
"""
from __future__ import annotations

import io
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from backend.db import Alert, Dataset, Prediction, PredictionRun, utcnow
from backend.dependencies import get_db, predictor_dep
from backend.schemas import (AckIn, AlertOut, DatasetOut, ExplainRequest, HealthCheck, ObservationsIn,
                             PredictRequest, PredictResponse, RunOut)
from backend.services import datasets as ds_svc
from backend.services import models_info
from backend.services import predictions as pred_svc
from backend.settings import Settings, get_settings
from predictops.config import load_config, project_path
from predictops.features.build import build_features
from predictops.inference.predictor import PredictOpsPredictor
from predictops.preprocessing.pipeline import clean_sensor_frame

router = APIRouter()
SENSORS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current",
           "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]


@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    return pd.read_parquet(project_path("data/processed/history_predictions.parquet"))


def _json_df(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df.replace({np.nan: None}).to_dict(orient="records")


# ======================================================================== system
@router.get("/health", response_model=HealthCheck, tags=["system"])
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1")); dbs = "ok"
    except Exception as ex:
        dbs = f"error: {type(ex).__name__}"
    try:
        from predictops.inference.predictor import get_predictor
        mv, ok = get_predictor().model_versions, True
    except Exception:
        mv, ok = {}, False
    return HealthCheck(status="ok" if ok and dbs == "ok" else "degraded", models_loaded=ok, model_versions=mv, database=dbs)


@router.get("/models", tags=["system"])
def models():
    return models_info.model_info()


@router.get("/models/performance", tags=["system"])
def models_performance():
    return models_info.performance()


@router.get("/models/versions", tags=["system"])
def model_versions(db: Session = Depends(get_db)):
    return models_info.list_versions(db)


# ======================================================================== datasets
@router.post("/datasets/upload", response_model=DatasetOut, status_code=201, tags=["datasets"])
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db),
                         settings: Settings = Depends(get_settings)):
    """Upload a CSV in MetroPT-3 format. The file is validated; invalid files are stored with their
    validation report (`validation.ok = false`) so the user can see what is wrong."""
    ds, _ = await ds_svc.save_upload(file, db, settings)
    return ds


@router.get("/datasets", response_model=list[DatasetOut], tags=["datasets"])
def list_datasets(db: Session = Depends(get_db)):
    return db.scalars(select(Dataset).order_by(Dataset.uploaded_at.desc())).all()


@router.get("/datasets/samples", tags=["datasets"])
def list_samples():
    return {"samples": sorted(ds_svc.SAMPLES)}


@router.get("/datasets/{dataset_id}/sensors", tags=["datasets"])
def dataset_sensors(dataset_id: int, resample: str = Query("1min", pattern="^(10s|1min|5min|15min|1h)$"),
                    db: Session = Depends(get_db)):
    """Sensor time series of an uploaded/sample dataset (for charts), optionally down-sampled."""
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(404, "Dataset not found")
    raw = ds_svc.load_dataset_frame(ds)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    cols = [c for c in SENSORS if c in raw]
    out = raw[cols].apply(pd.to_numeric, errors="coerce")
    if resample != "10s":
        out = out.resample(resample).mean().dropna(how="all")
    return _json_df(out.reset_index())


# ======================================================================== predictions
@router.post("/predict", response_model=PredictResponse, tags=["predictions"])
def predict(req: PredictRequest, db: Session = Depends(get_db), predictor: PredictOpsPredictor = Depends(predictor_dep),
            settings: Settings = Depends(get_settings)):
    """Batch assessment of an uploaded dataset (`dataset_id`) or a bundled sample (`sample_name`)."""
    if (req.dataset_id is None) == (req.sample_name is None):
        raise HTTPException(422, "Provide exactly one of dataset_id or sample_name.")
    if req.dataset_id is not None:
        ds = db.get(Dataset, req.dataset_id)
        if ds is None:
            raise HTTPException(404, "Dataset not found")
        if not ds.validation.get("ok"):
            raise HTTPException(422, f"Dataset failed validation: {ds.validation.get('errors')}")
        raw, source = ds_svc.load_dataset_frame(ds), "upload"
    else:
        ds, raw = ds_svc.register_sample(req.sample_name, db); source = "sample"
    run, res, alerts = pred_svc.run_and_store(raw, predictor, db, settings, source, ds, req.start, req.end)
    return {"run": run, "latest": pred_svc.latest_assessment(res), "alerts": alerts}


@router.post("/predict/observations", response_model=PredictResponse, tags=["predictions"])
def predict_observations(body: ObservationsIn, db: Session = Depends(get_db),
                         predictor: PredictOpsPredictor = Depends(predictor_dep), settings: Settings = Depends(get_settings)):
    """Individual / streaming-style assessment from raw 10-s readings (JSON).
    Rolling features use up to 2 h of history; with less data, long-window features are imputed
    and the assessment is less reliable."""
    raw = pd.DataFrame([r.model_dump() for r in body.readings])
    raw["timestamp"] = raw["timestamp"].astype(str)
    if not body.store:
        try:
            res = predictor.predict_raw(raw)
        except ValueError as ex:
            raise HTTPException(422, str(ex))
        latest = pred_svc.latest_assessment(res)
        fake = {"id": 0, "dataset_id": None, "source": "observations", "n_minutes": len(res),
                "start_time": res.timestamp.min(), "end_time": res.timestamp.max(),
                "model_version_detector": predictor.model_versions["failure_detector"],
                "model_version_anomaly": predictor.model_versions["anomaly_detector"], "summary": {}, "created_at": utcnow()}
        return {"run": fake, "latest": latest, "alerts": []}
    # store raw readings as a dataset file so the run can be explained / replotted later
    ds = ds_svc.save_frame(raw, db, settings, source="observations")
    run, res, alerts = pred_svc.run_and_store(raw, predictor, db, settings, "observations", ds)
    return {"run": run, "latest": pred_svc.latest_assessment(res), "alerts": alerts}


@router.get("/runs", response_model=list[RunOut], tags=["predictions"])
def list_runs(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    return db.scalars(select(PredictionRun).order_by(PredictionRun.created_at.desc()).limit(limit)).all()


@router.delete("/runs/{run_id}", status_code=204, tags=["predictions"])
def delete_run(run_id: int, db: Session = Depends(get_db)):
    pred_svc.delete_run(db, run_id)


@router.get("/predictions", tags=["predictions"])
def get_predictions(run_id: int | None = None, start: datetime | None = None, end: datetime | None = None,
                    limit: int = Query(20_000, ge=1, le=100_000), db: Session = Depends(get_db)):
    """Historical prediction records (per minute) filtered by run and/or time range."""
    return _json_df(pred_svc.predictions_frame(db, run_id, start, end, limit))


@router.get("/predictions/download", tags=["predictions"])
def download_predictions(run_id: int, db: Session = Depends(get_db)):
    run = pred_svc.get_run(db, run_id)
    df = pred_svc.predictions_frame(db, run_id, limit=1_000_000)
    df["reasons"] = df.reasons.map(" | ".join); df["recommendations"] = df.recommendations.map(" | ".join)
    df["model_version_detector"] = run.model_version_detector; df["model_version_anomaly"] = run.model_version_anomaly
    return StreamingResponse(io.BytesIO(df.to_csv(index=False).encode()), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=predictops_run{run_id}.csv"})


# ======================================================================== explainability
def _run_features(run: PredictionRun, db: Session) -> pd.DataFrame:
    if run.dataset_id is None:
        raise HTTPException(422, "This run has no stored sensor file (observations run with store=false).")
    ds = db.get(Dataset, run.dataset_id)
    return build_features(clean_sensor_frame(ds_svc.load_dataset_frame(ds), load_config()))


@router.post("/explain", tags=["explainability"])
def explain(req: ExplainRequest, db: Session = Depends(get_db), predictor: PredictOpsPredictor = Depends(predictor_dep)):
    """SHAP contributions (failure-signature model) + most-deviating features (anomaly model) for one minute."""
    run = pred_svc.get_run(db, req.run_id)
    m = _run_features(run, db)
    t = pd.Timestamp(req.timestamp)
    idx = (m.timestamp - t).abs().idxmin()
    row = m.loc[[idx]]
    ex = predictor.explain(row, req.top_k)[0]
    pred = db.scalar(select(Prediction).where(Prediction.run_id == run.id, Prediction.timestamp == row.timestamp.iloc[0].to_pydatetime()))
    return {"timestamp": row.timestamp.iloc[0].isoformat(), "run_id": run.id,
            "detector_probability": float(predictor.detector.predict_proba(row[predictor.det_features])[:, 1][0]),
            "detector_threshold": predictor.det_threshold,
            "anomaly_score": float(predictor.anomaly.score(row)[0]), "anomaly_threshold": predictor.anomaly.thresholds["low"],
            "health_status": pred.health_status if pred else None, "reasons": pred.reasons if pred else None,
            **ex, "note": "SHAP explains the model's output, not the physical cause of a failure."}


@router.get("/explain/global", tags=["explainability"])
def explain_global():
    p = project_path("reports/shap/xgb_global_mean_abs_shap.csv")
    if not p.exists():
        raise HTTPException(404, "Run scripts/explain.py first.")
    d = pd.read_csv(p).rename(columns={"Unnamed: 0": "feature"})
    return {"model": "failure_detector", "metric": "mean |SHAP| (validation sample)", "features": _json_df(d)}


# ======================================================================== alerts
@router.get("/alerts", response_model=list[AlertOut], tags=["alerts"])
def list_alerts(acknowledged: bool | None = None, run_id: int | None = None, limit: int = Query(200, le=2000),
                db: Session = Depends(get_db)):
    q = select(Alert)
    if acknowledged is not None:
        q = q.where(Alert.acknowledged == acknowledged)
    if run_id is not None:
        q = q.where(Alert.run_id == run_id)
    return db.scalars(q.order_by(Alert.start_time.desc()).limit(limit)).all()


@router.post("/alerts/{alert_id}/acknowledge", response_model=AlertOut, tags=["alerts"])
def acknowledge(alert_id: int, body: AckIn, db: Session = Depends(get_db)):
    al = db.get(Alert, alert_id)
    if al is None:
        raise HTTPException(404, "Alert not found")
    al.acknowledged, al.acknowledged_at, al.note = True, utcnow(), body.note
    db.commit(); db.refresh(al)
    return al


# ======================================================================== dashboard / history
@router.get("/dashboard/stats", tags=["dashboard"])
def dashboard_stats(db: Session = Depends(get_db)):
    h = _history()
    last_run = db.scalar(select(PredictionRun).order_by(PredictionRun.created_at.desc()).limit(1))
    latest = None
    if last_run:
        p = db.scalar(select(Prediction).where(Prediction.run_id == last_run.id).order_by(Prediction.timestamp.desc()).limit(1))
        latest = {k: getattr(p, k) for k in ["timestamp", "health_score", "health_status", "failure_risk", "anomaly_status",
                                              "reasons", "recommendations"]} | {"run_id": last_run.id}
    return {
        "dataset": {"observations_10s": 1_516_948, "minutes_assessed": int(len(h)),
                    "start": str(h.timestamp.min()), "end": str(h.timestamp.max()), "reported_failures": 4},
        "history_status_counts": h.health_status.value_counts().to_dict(),
        "history_anomalous_minutes": int((h.anomaly_status == "Anomalous").sum()),
        "runs": db.scalar(select(func.count(PredictionRun.id))) or 0,
        "alerts_total": db.scalar(select(func.count(Alert.id))) or 0,
        "alerts_open": db.scalar(select(func.count(Alert.id)).where(Alert.acknowledged.is_(False))) or 0,
        "latest_assessment": latest,
    }


@router.get("/history", tags=["dashboard"])
def history(start: datetime, end: datetime, every: int = Query(1, ge=1, le=60),
            columns: str | None = Query(None, description="comma-separated subset")):
    """Precomputed assessment of the historical dataset (used by charts and historical replay).
    `data_role` says whether a minute was in-sample (training/validation) or unseen (test)."""
    h = _history()
    d = h[(h.timestamp >= pd.Timestamp(start)) & (h.timestamp <= pd.Timestamp(end))].iloc[::every]
    if len(d) > 50_000:
        raise HTTPException(413, "Range too large; increase `every` or narrow the range.")
    if columns:
        d = d[["timestamp"] + [c for c in columns.split(",") if c in d and c != "timestamp"]]
    return _json_df(d)


@router.get("/history/events", tags=["dashboard"])
def history_events():
    """Reported failure windows + history-wide alert episodes (for timelines)."""
    h = _history()
    ep = PredictOpsPredictor.alert_list(h.assign(reasons=h.reasons.str.split(" \\| "),
                                                 recommendations=h.recommendations.str.split(" \\| ")))
    roles = h.set_index("timestamp").data_role
    ep["data_role"] = [roles.asof(s) for s in ep.start]
    return {"reported_failures": load_config()["failure_events"], "alert_episodes": _json_df(ep)}


# ======================================================================== monitoring
@lru_cache(maxsize=1)
def _features_all() -> pd.DataFrame:
    """Full feature table if present (development), else the slim monitoring copy (deployment)."""
    full = project_path("data/processed/features_1min.parquet")
    return pd.read_parquet(full if full.exists() else project_path("data/processed/features_monitoring.parquet"))


@lru_cache(maxsize=1)
def _reference():
    return (pd.read_parquet(project_path("models/metadata/reference_features.parquet")),
            pd.read_parquet(project_path("models/metadata/reference_scores.parquet")).anomaly_score.to_numpy())


def _drift_payload(cur: pd.DataFrame, predictor: PredictOpsPredictor) -> dict:
    from predictops.anomaly.detectors import ANOMALY_FEATURES
    from predictops.monitoring.drift import drift_report, psi
    ref, ref_scores = _reference()
    cur = cur[cur.get("usable", 1) == 1] if "usable" in cur else cur
    if len(cur) < 60:
        raise HTTPException(422, "Need at least 60 minutes of data for drift analysis.")
    feats = [f for f in ANOMALY_FEATURES if f in cur]
    rep = drift_report(ref, cur, feats)
    scores = predictor.anomaly.score(cur)
    thr = predictor.anomaly.thresholds["low"]
    return {
        "minutes": int(len(cur)), "reference": "training-period normal operation (Feb–May 2020, 20k minutes)",
        "feature_drift": _json_df(rep),
        "summary": {"features_major": int((rep.status == "major").sum()), "features_moderate": int((rep.status == "moderate").sum()),
                    "features_stable": int((rep.status == "stable").sum())},
        "prediction_drift": {"flagged_share_reference": float((ref_scores >= thr).mean()),
                             "flagged_share_current": float((scores >= thr).mean()),
                             "anomaly_score_psi": psi(ref_scores, scores)},
        "note": "Drift means the inputs changed; it does not by itself mean the model is wrong. "
                "Performance can only be re-measured when new failure labels are available.",
    }


@router.get("/monitoring/drift", tags=["monitoring"])
def drift(start: datetime | None = None, end: datetime | None = None, run_id: int | None = None,
          db: Session = Depends(get_db), predictor: PredictOpsPredictor = Depends(predictor_dep)):
    """Feature drift (KS, PSI) and anomaly-rate drift of a historical window or an uploaded run vs the training reference."""
    if run_id is not None:
        cur = _run_features(pred_svc.get_run(db, run_id), db)
    elif start is not None and end is not None:
        f = _features_all()
        cur = f[(f.timestamp >= pd.Timestamp(start)) & (f.timestamp <= pd.Timestamp(end))]
    else:
        raise HTTPException(422, "Provide run_id, or start and end.")
    return _drift_payload(cur, predictor)


@router.get("/monitoring/monthly", tags=["monitoring"])
def drift_monthly(predictor: PredictOpsPredictor = Depends(predictor_dep)):
    """Per-month drift summary across the whole dataset (cached report)."""
    p = project_path("reports/monitoring/monthly_drift.csv")
    if not p.exists():
        raise HTTPException(404, "Run scripts/monitoring_report.py first.")
    return _json_df(pd.read_csv(p))


# ======================================================================== curves & replay
@router.get("/models/curves", tags=["system"])
def model_curves(points: int = Query(200, ge=20, le=2000)):
    """ROC and precision-recall curves of every supervised model on the TEST split (event 4)."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    tp = pd.read_parquet(project_path("data/processed/test_predictions.parquet"))
    out = {}
    for c in [c for c in tp if c.startswith("score_")]:
        fpr, tpr, _ = roc_curve(tp.y_detect, tp[c]); pr, rc, _ = precision_recall_curve(tp.y_detect, tp[c])
        i = np.linspace(0, len(fpr) - 1, min(points, len(fpr))).astype(int)
        j = np.linspace(0, len(pr) - 1, min(points, len(pr))).astype(int)
        out[c[6:]] = {"roc": {"fpr": fpr[i].tolist(), "tpr": tpr[i].tolist()},
                      "pr": {"recall": rc[j].tolist(), "precision": pr[j].tolist()}}
    return {"split": "test (event 4)", "base_rate": float(tp.y_detect.mean()), "curves": out}


@lru_cache(maxsize=1)
def _raw10s() -> pd.DataFrame:
    cols = ["timestamp"] + SENSORS
    return pd.read_parquet(project_path("data/intermediate/metropt3.parquet"), columns=cols)


@router.get("/replay/step", tags=["dashboard"])
def replay_step(t: datetime, lookback_hours: int = Query(25, ge=3, le=48),
                predictor: PredictOpsPredictor = Depends(predictor_dep)):
    """HISTORICAL REPLAY (not live monitoring): take the recorded raw 10-s readings up to time `t`
    (with `lookback_hours` of history for rolling features), run the full inference pipeline on them,
    and return the assessment of the latest minute plus its raw readings."""
    raw = _raw10s()
    t = pd.Timestamp(t)
    w = raw[(raw.timestamp > t - pd.Timedelta(hours=lookback_hours)) & (raw.timestamp <= t)]
    if len(w) < 12:
        raise HTTPException(422, "No recorded data near this time (logger offline).")
    w = w.assign(timestamp=w.timestamp.astype(str))
    res = predictor.predict_raw(w)
    last = res.iloc[-1]
    return {"mode": "historical replay", "requested_time": t.isoformat(), "minute": last.timestamp.isoformat(),
            "assessment": {k: (float(last[k]) if isinstance(last[k], (np.floating, float)) else
                               (bool(last[k]) if isinstance(last[k], (np.bool_, bool)) else last[k]))
                           for k in ["health_score", "health_status", "failure_risk", "anomaly_status", "anomaly_score",
                                     "anomaly_severity", "detector_prob", "reasons", "recommendations"]},
            "sensors": {k: float(last[k]) for k in ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current", "LPS"]},
            "model_versions": predictor.model_versions}


# ======================================================================== reports
@router.get("/reports/run/{run_id}.pdf", tags=["reports"])
def run_report(run_id: int, db: Session = Depends(get_db), predictor: PredictOpsPredictor = Depends(predictor_dep)):
    """Downloadable PDF maintenance report for one prediction run (all content from stored results)."""
    from predictops.reports.pdf import build_run_report
    from backend.schemas import AlertOut, RunOut
    run = pred_svc.get_run(db, run_id)
    preds = pred_svc.predictions_frame(db, run_id, limit=1_000_000)
    preds["timestamp"] = pd.to_datetime(preds.timestamp)
    alerts = [AlertOut.model_validate(a).model_dump(mode="json")
              for a in db.scalars(select(Alert).where(Alert.run_id == run_id).order_by(Alert.start_time))]
    sensors, expl = pd.DataFrame(), None
    if run.dataset_id:
        m = _run_features(run, db)
        sensors = m[["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current"]]
        worst_t = preds.loc[preds.health_score.idxmin(), "timestamp"]
        row = m.loc[[(m.timestamp - worst_t).abs().idxmin()]]
        expl = predictor.explain(row, 6)[0]
    pdf = build_run_report(RunOut.model_validate(run).model_dump(mode="json"), preds, alerts, sensors, expl,
                           "MetroPT-3 APU compressor (metro train air production unit)")
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename=predictops_report_run{run_id}.pdf"})
