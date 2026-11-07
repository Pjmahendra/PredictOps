"""Integration tests: FastAPI + inference pipeline + SQLite (temporary database)."""
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictops.config import project_path

pytestmark = pytest.mark.skipif(not project_path("data/processed/history_predictions.parquet").exists(),
                                reason="run training + scripts/precompute_history.py first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    import os
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp / 'test.db'}"
    os.environ["UPLOAD_DIR"] = str(tmp / "uploads")
    os.environ["MAX_UPLOAD_MB"] = "5"
    os.environ.pop("API_KEY", None)
    from backend.settings import get_settings
    get_settings.cache_clear()
    from backend.main import app
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


SMALL = project_path("data/sample/small_2h_2020-08-10.csv")


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["models_loaded"] and r["database"] == "ok"


def test_models_info_and_performance(client):
    info = client.get("/models").json()
    assert info["failure_detector"]["decided_before_test"] is True
    perf = client.get("/models/performance").json()
    assert len(perf["ml_test_detection"]) == 5 and perf["anomaly"]


def test_upload_validate_predict_explain_download(client):
    r = client.post("/datasets/upload", files={"file": ("x.csv", SMALL.read_bytes(), "text/csv")})
    assert r.status_code == 201, r.text
    ds = r.json(); assert ds["validation"]["ok"]
    p = client.post("/predict", json={"dataset_id": ds["id"]})
    assert p.status_code == 200, p.text
    body = p.json(); run_id = body["run"]["id"]
    assert body["latest"]["health_status"] in {"Healthy", "Warning", "Critical"}
    assert body["run"]["model_version_detector"].startswith("xgboost")
    preds = client.get("/predictions", params={"run_id": run_id}).json()
    assert len(preds) == body["run"]["n_minutes"]
    ex = client.post("/explain", json={"run_id": run_id, "timestamp": preds[-1]["timestamp"]}).json()
    assert ex["detector_shap"]["contributions"] and ex["anomaly_top_deviations"]
    csv = client.get("/predictions/download", params={"run_id": run_id})
    assert csv.status_code == 200 and "health_score" in csv.text.splitlines()[0]
    sens = client.get(f"/datasets/{ds['id']}/sensors", params={"resample": "5min"}).json()
    assert "TP3" in sens[0]


def test_sample_prediction_creates_alerts_and_ack(client):
    r = client.post("/predict", json={"sample_name": "event4_2020-07-15.csv"}).json()
    assert r["alerts"], "event-4 sample should produce at least one alert"
    aid = r["alerts"][0]["id"]
    a = client.post(f"/alerts/{aid}/acknowledge", json={"note": "inspected"}).json()
    assert a["acknowledged"] and a["note"] == "inspected"
    assert all(x["acknowledged"] is False for x in client.get("/alerts", params={"acknowledged": False}).json())


def test_observations_endpoint(client):
    df = pd.read_csv(SMALL).head(400).drop(columns=["Unnamed: 0"])
    r = client.post("/predict/observations", json={"readings": df.to_dict(orient="records"), "store": True})
    assert r.status_code == 200, r.text


def test_observations_out_of_range_rejected(client):
    df = pd.read_csv(SMALL).head(10).drop(columns=["Unnamed: 0"])
    df.loc[0, "TP3"] = 999
    r = client.post("/predict/observations", json={"readings": df.to_dict(orient="records")})
    assert r.status_code == 422


@pytest.mark.parametrize("name,content,code", [
    ("bad.txt", b"a,b\n1,2", 415),                        # wrong extension
    ("empty.csv", b"", 422),                              # empty file
])
def test_upload_rejections(client, name, content, code):
    assert client.post("/datasets/upload", files={"file": (name, content, "text/csv")}).status_code == code


def test_upload_too_large(client):
    big = b"x" * (6 * 1024 * 1024)
    assert client.post("/datasets/upload", files={"file": ("big.csv", big, "text/csv")}).status_code == 413


def test_invalid_columns_stored_but_not_predicted(client):
    bad = pd.read_csv(SMALL).drop(columns=["TP3"]).to_csv(index=False).encode()
    ds = client.post("/datasets/upload", files={"file": ("bad.csv", bad, "text/csv")}).json()
    assert ds["validation"]["ok"] is False and "TP3" in ds["validation"]["errors"][0]
    assert client.post("/predict", json={"dataset_id": ds["id"]}).status_code == 422


def test_path_traversal_filename_is_neutralised(client):
    r = client.post("/datasets/upload", files={"file": ("../../etc/evil.csv", SMALL.read_bytes(), "text/csv")})
    assert r.status_code == 201 and r.json()["original_filename"] == "evil.csv"


def test_history_dashboard_monitoring(client):
    h = client.get("/history", params={"start": "2020-07-15T00:00:00", "end": "2020-07-15T23:59:00", "every": 5}).json()
    assert h and {"health_score", "data_role"} <= set(h[0])
    assert client.get("/dashboard/stats").json()["dataset"]["reported_failures"] == 4
    d = client.get("/monitoring/drift", params={"start": "2020-08-01T00:00:00", "end": "2020-08-03T00:00:00"}).json()
    assert d["feature_drift"] and "prediction_drift" in d
    assert client.get("/history/events").json()["reported_failures"]


def test_unknown_resources(client):
    assert client.get("/predictions/download", params={"run_id": 99999}).status_code == 404
    assert client.post("/alerts/99999/acknowledge", json={}).status_code == 404
    assert client.post("/predict", json={"sample_name": "nope.csv"}).status_code == 404
    assert client.post("/predict", json={}).status_code == 422


def test_curves_and_replay(client):
    c = client.get("/models/curves").json()
    assert set(c["curves"]) == {"logreg", "tree", "rf", "svm", "xgboost"}
    r = client.get("/replay/step", params={"t": "2020-07-15T18:30:00"})
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "historical replay" and r.json()["assessment"]["health_status"] == "Critical"
    assert client.get("/replay/step", params={"t": "2020-04-26T12:00:00"}).status_code == 422   # logger offline gap


def test_pdf_report(client):
    run = client.post("/predict", json={"sample_name": "event4_2020-07-15.csv"}).json()["run"]
    r = client.get(f"/reports/run/{run['id']}.pdf")
    assert r.status_code == 200 and r.content[:4] == b"%PDF" and len(r.content) > 3000
