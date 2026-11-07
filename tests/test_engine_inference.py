"""Stage 6 tests: health engine rules, inference pipeline on real sample files, SHAP, artifacts."""
import numpy as np
import pandas as pd
import pytest

from predictops.config import load_config, project_path
from predictops.engine import health

CFG = load_config()
HAVE_MODELS = project_path("models/metadata/sensor_ranges.json").exists()
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="run training scripts first")


def test_health_score_bounds_and_monotonic():
    w = CFG["health_engine"]["weights"]
    s_ok = health.health_score(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), w)
    s_bad = health.health_score(np.array([1.0]), np.array([5.0]), np.array([1.0]), np.array([1.0]), w)
    assert s_ok[0] == 100 and s_bad[0] == 0
    assert abs(sum(w.values()) - 1) < 1e-9


def test_status_escalation_rules():
    cut = CFG["health_engine"]["cutoffs"]
    st = health.classify_status(np.array([95, 95, 95]), np.array([False, True, True]), np.array([False, False, True]), cut)
    assert st.tolist() == ["Healthy", "Warning", "Critical"]


def test_risk_levels():
    r = health.classify_risk(np.array([False, False, True]), np.array(["normal", "low", "high"]),
                             np.array([0.1, 0.1, 0.1]), np.array([False, False, False]))
    assert r.tolist() == ["Low", "Medium", "High"]


@pytest.fixture(scope="module")
def predictor():
    from predictops.inference.predictor import PredictOpsPredictor
    return PredictOpsPredictor()


@needs_models
def test_predict_raw_on_event4_sample(predictor):
    raw = pd.read_csv(project_path("data/sample/event4_2020-07-15.csv"))
    r = predictor.predict_raw(raw)
    assert len(r) > 1000
    assert set(r.health_status) <= {"Healthy", "Warning", "Critical"}
    assert set(r.failure_risk) <= {"Low", "Medium", "High"}
    assert r.health_score.between(0, 100).all()
    during = r[r.timestamp.between("2020-07-15 17:30", "2020-07-15 19:00")]
    assert (during.health_status == "Critical").any()            # event 4 is flagged
    assert all(isinstance(x, list) and x for x in r.recommendations)


@needs_models
def test_normal_day_mostly_healthy(predictor):
    r = predictor.predict_raw(pd.read_csv(project_path("data/sample/normal_day_2020-08-10.csv")))
    assert (r.health_status == "Healthy").mean() > 0.9


@needs_models
def test_invalid_upload_rejected(predictor):
    raw = pd.read_csv(project_path("data/sample/small_2h_2020-08-10.csv")).drop(columns=["TP3"])
    with pytest.raises(ValueError, match="TP3"):
        predictor.predict_raw(raw)


@needs_models
def test_explanations_shape(predictor):
    raw = pd.read_csv(project_path("data/sample/small_2h_2020-08-10.csv"))
    predictor.predict_raw(raw)
    ex = predictor.explain(predictor._last_features.tail(3), top_k=4)
    assert len(ex) == 3 and len(ex[0]["detector_shap"]["contributions"]) == 4
    assert len(ex[0]["anomaly_top_deviations"]) == 4


@needs_models
def test_shap_additivity(predictor):
    """SHAP values + base value must reproduce the model's logit (exactness of TreeSHAP)."""
    from predictops.explainability.shap_explain import TreeShapExplainer
    m = pd.read_parquet(project_path("data/processed/features_1min.parquet")).query("split == 'test'").head(50)
    tx = TreeShapExplainer(predictor.detector)
    ex = tx.explain(m[predictor.det_features])
    p = predictor.detector.predict_proba(m[predictor.det_features])[:, 1]
    logit = np.log(p / (1 - p))
    np.testing.assert_allclose(ex.values.sum(1) + ex.base_values, logit, atol=1e-3)


def test_missing_artifacts_raise_clear_error(tmp_path):
    from predictops.inference.predictor import ArtifactError, PredictOpsPredictor
    with pytest.raises(ArtifactError, match="Missing model artifact"):
        PredictOpsPredictor(models_dir=tmp_path)
