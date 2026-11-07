"""Stage 21 — performance measurements on this machine. Run: .venv/bin/python scripts/benchmark.py"""
import platform, resource, time, json
import httpx, pandas as pd
from predictops.config import project_path
from predictops.inference.predictor import PredictOpsPredictor

out = {"machine": f"{platform.machine()} {platform.system()} {platform.release()}", "python": platform.python_version()}
t = time.time(); P = PredictOpsPredictor(); out["model_load_s"] = round(time.time() - t, 2)
raw = pd.read_csv(project_path("data/sample/event4_2020-07-15.csv"))
for n in [1, 3]:
    t = time.time()
    for _ in range(n): r = P.predict_raw(raw)
    out[f"predict_1day_s"] = round((time.time() - t) / n, 2)
out["rows_1day_10s"] = len(raw); out["minutes_1day"] = len(r)
m = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
t = time.time(); P.predict_features(m); out["predict_full_dataset_252k_min_s"] = round(time.time() - t, 1)
t = time.time(); P.explain(m.tail(100)); out["shap_100_rows_s"] = round(time.time() - t, 2)
for path, params in [("/health", {}), ("/replay/step", {"t": "2020-07-15T18:30:00"}), ("/history", {"start": "2020-07-14", "end": "2020-07-16"})]:
    try:
        ts = []
        for _ in range(5):
            t = time.time(); httpx.get(f"http://127.0.0.1:8000{path}", params=params, timeout=60); ts.append(time.time() - t)
        out[f"api{path}_median_ms"] = round(sorted(ts)[2] * 1000)
    except Exception as ex:
        out[f"api{path}"] = f"API not running ({type(ex).__name__})"
out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)   # macOS reports bytes
sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in project_path("models/trained").glob("*")}
out["artifact_sizes_mb"] = sizes
project_path("reports/benchmark.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
