"""PredictOps launcher.

  python run.py            start API (port 8000) + dashboard (port 8501)
  python run.py pipeline   rebuild everything from the raw CSV (data -> models -> history)
  python run.py test       run the test suite
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PIPELINE = ["build_intermediate", "inspect_dataset", "run_eda", "build_processed", "feature_study", "train_ml",
            "train_anomaly", "train_dl", "explain", "evaluate_health", "precompute_history", "monitoring_report"]


def pipeline():
    for s in PIPELINE:
        print(f"\n=== {s} ===", flush=True)
        subprocess.run([PY, str(ROOT / "scripts" / f"{s}.py")], cwd=ROOT, check=True)


def serve():
    api = subprocess.Popen([PY, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT)
    time.sleep(3)
    ui = subprocess.Popen([PY, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501"], cwd=ROOT)
    print("\nAPI:       http://127.0.0.1:8000/docs\nDashboard: http://localhost:8501\nCtrl+C to stop.")
    try:
        ui.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in (ui, api):
            p.terminate()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    {"pipeline": pipeline, "test": lambda: subprocess.run([PY, "-m", "pytest", "-q"], cwd=ROOT), "serve": serve}[cmd]()
