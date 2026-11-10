"""Synthetic live stream: cycles healthy → degrading → failure and the pipeline reacts."""
import pandas as pd
import pytest

from predictops.config import load_config, project_path
from predictops.simulation.simulator import SENSORS, StreamSimulator

RAW = project_path("data/intermediate/metropt3.parquet")
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="typed parquet not available")


@pytest.fixture(scope="module")
def run():
    from predictops.inference.predictor import get_predictor
    raw = pd.read_parquet(RAW, columns=["timestamp"] + SENSORS)
    sim = StreamSimulator(raw, load_config()["failure_events"], get_predictor(), seed=1)
    for _ in range(40):
        sim.step(10)
    return sim.snapshot(720)


def test_phases_cycle_in_order(run):
    names = [p["phase"] for p in run["phases"]]
    assert names[:4] == ["healthy", "degrading", "failure", "maintenance"]
    assert run["mode"].startswith("SIMULATED")


def test_pipeline_separates_healthy_and_failure(run):
    d = pd.DataFrame(run["rows"])
    healthy, failure = d[d.sim_phase == "healthy"], d[d.sim_phase == "failure"]
    assert (healthy.health_status == "Healthy").mean() > 0.9
    assert (failure.health_status == "Critical").mean() > 0.7
    assert healthy.health_score.mean() > failure.health_score.mean() + 40

