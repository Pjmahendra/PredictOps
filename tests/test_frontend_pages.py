"""End-to-end UI smoke tests: every dashboard page renders against the RUNNING API without exceptions.
Requires: uvicorn backend.main:app on 127.0.0.1:8000."""
import sys

import httpx
import pytest

from predictops.config import project_path

FRONT = project_path("frontend")
try:
    API_UP = httpx.get("http://127.0.0.1:8000/health", timeout=2).status_code == 200
except Exception:
    API_UP = False
pytestmark = pytest.mark.skipif(not API_UP, reason="start the API first")
PAGES = ["overview", "sensors", "prediction", "anomaly", "explain", "health", "maintenance", "performance", "monitoring", "replay", "simulation"]


@pytest.fixture(autouse=True, scope="module")
def _path():
    sys.path.insert(0, str(FRONT))
    yield
    sys.path.remove(str(FRONT))


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(FRONT / "views" / f"{page}.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


def test_prediction_flow_runs_sample():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(FRONT / "views" / "prediction.py"), default_timeout=180).run()
    at.selectbox[0].set_value("event4_2020-07-15.csv").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("run #" in m.value for m in at.markdown)
