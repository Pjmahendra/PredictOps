# 09 — Testing, Performance and Deployment (Stages 20, 21, 23)

---

## 1. Testing strategy (pytest) — **69 tests, all passing**
`.venv/bin/pytest -q` → `69 passed` (≈12 s). UI tests need the API running.

| File | # | Level | What is verified |
|---|---|---|---|
| `test_ingestion.py` | 11 | unit | schema validation: missing/unexpected columns, invalid timestamps, non-numeric, non-binary digital, ±inf, missing values, empty file, segment splitting |
| `test_intermediate_data.py` | 3 | integration (real data) | 1,516,948 rows, sorted/unique, **raw CSV unchanged (SHA-256)** |
| `test_eda.py` | 4 | unit | operating-state thresholds, frozen-run detection, failure-window flags |
| `test_preprocessing.py` | 12 | unit + real data | warning labels only before onset, forbidden future columns, short-gap-only filling, scaler fitted on train only, chronological splits + embargo, test holds only event 4, forward-chaining folds never train on the future, saved scaler = train statistics, sample files valid |
| `test_features.py` | 4 | unit + real data | **features are past-only** (prefix-invariance), no labels in features, right-closed minute bins, no rolling across segments |
| `test_engine_inference.py` | 9 | unit + integration | health score bounds, status escalation, risk rules, end-to-end raw CSV → assessment (event 4 → Critical, normal day > 90% Healthy), invalid upload, explanation shape, **SHAP additivity**, clear error on missing artifacts |
| `test_api.py` | 15 | integration (FastAPI + SQLite temp DB) | health, model info, upload → validate → predict → explain → CSV download → sensors, alerts + acknowledgement, observations, out-of-range values, wrong type / empty / oversize uploads, invalid columns, **path traversal**, history/dashboard/drift, 404s, curves, replay (incl. logger-gap 422), PDF report |
| `test_frontend_pages.py` | 11 | end-to-end UI (Streamlit AppTest against live API) | all 10 pages render without exceptions; the Failure-Prediction flow runs a sample end-to-end |

**Edge cases covered:** missing values, unexpected columns, invalid timestamps, empty files, large uploads, out-of-range
measurements, invalid data types, missing model artifacts, unknown IDs, logger-offline replay times.
**Not automated:** database outage (SQLite file is local; `/health` reports `database: error` if unreachable — manually checked by design, not by test).

### End-to-end user journey (verified manually with screenshots in `reports/screenshots/`)
open app → upload valid CSV (validation shown) → view sensor trends → run prediction → anomaly page → SHAP explanation →
equipment health → maintenance alerts (acknowledge) → download CSV and PDF. The same journey is covered by `test_api.py` +
`test_frontend_pages.py`.

### Bugs found by testing (and fixed)
| Bug | Found by | Fix |
|---|---|---|
| NaN in evaluation tables broke JSON responses | `test_models_info_and_performance` | convert NaN → null |
| Health score 1e-14 instead of 0 | `test_health_score_bounds_and_monotonic` | round to 0.1 |
| `datetime64` NaT unit deprecation | pytest warning | explicit `"ns"` unit |
| Charts drew straight lines across logger gaps (fabricated-looking data) | screenshot review | gap rows break lines |
| Legend overlapped titles; clipped labels | screenshot review | layout fixes |
| Leave-one-event-out CV used future data | design review (Stage 4b) | forward-chaining folds + test |

## 2. Performance (Stage 21) — measured on this machine (`scripts/benchmark.py` → `reports/benchmark.json`)
Apple Silicon (arm64) Mac, 8 GB RAM, 8 cores, Python 3.12.

| Operation | Time / size |
|---|---|
| Raw CSV (218 MB) → typed Parquet | 5 s; 91 MB in RAM (float32/int8), 29 MB on disk |
| 1-min feature build, full dataset (1.5 M rows → 252,719 minutes × 132 features) | ≈2 s |
| Model loading at API start | 0.07 s |
| Assess one day of raw data (8,661 readings → 1,433 minutes) | **0.17 s** |
| Assess the full 7-month history | 23 s |
| SHAP for 100 minutes (TreeExplainer) | 0.52 s |
| API `/health` · `/history` (2 days) · `/replay/step` | 5 ms · 95 ms · 162 ms (median) |
| Peak memory of a full inference process | ≈0.8 GB; API container ≈370 MB |
| Training: XGBoost / One-Class SVM / LSTM / RF tuning | 3.5 s / 0.2 s / 15 s (MPS) / ≈4 min |
| Deployed model artifacts | XGBoost 0.5 MB, One-Class SVM 0.03 MB |

**Optimisations used:** Parquet + float32/int8 (≈4× less memory); raw CSV read once; 1-minute aggregation (6× fewer rows);
vectorised pandas rolling windows; SVM / LOF / OCSVM trained on stratified or random subsamples; models loaded once and cached;
history precomputed once for charts; `st.cache_data` for API GETs; charts down-sample long ranges (`every`); Plotly WebGL for long series.

**Requirements:** any 64-bit machine with ≥ 4 GB RAM can run the app; ≥ 8 GB recommended to retrain (full pipeline ≈ 15 min).
Deep learning runs on CPU in a few minutes (MPS/GPU optional).

## 3. Deployment (Stage 23)

### Files
| File | Purpose |
|---|---|
| `Dockerfile` | slim Python 3.12 image, runtime deps only (`requirements-deploy.txt`, no torch/MLflow/Jupyter), non-root user |
| `.dockerignore` | excludes raw data, training splits, DB, logs, tests, the 96 MB full feature table |
| `docker-compose.yml` | `api` + `dashboard` services, persistent DB volume |
| `render.yaml` | Render blueprint: two web services, auto-generated `API_KEY`, health check `/health` |
| `run.py` | local launcher (`serve`, `pipeline`, `test`) |

Runtime data shipped in the image (≈90 MB): models, metadata, evaluation reports, sample CSVs, the typed Parquet
(for replay), precomputed history, test predictions and a 19.5 MB monitoring feature table (`features_monitoring.parquet`).

### Verified local deployment (Docker)
```bash
docker compose up --build        # http://localhost:8501  (API on :8000)
```
Verified on 2026-11-06: image builds; `/health` ok with both models loaded; `/predict` on the event-4 sample → 1,204 Healthy /
41 Warning / 188 Critical minutes and 2 alerts; PDF report 200; drift endpoint works; the containerised dashboard renders
with data from the containerised API. Memory: API ≈373 MB, dashboard ≈71 MB.

### Cloud deployment (Render) — prepared, needs your account
1. Commit the runtime artifacts (see `.gitignore` notes in README) and push to GitHub.
2. Render → *New → Blueprint* → select the repo → it reads `render.yaml` and creates both services.
3. Open `https://predictops-dashboard.onrender.com`.
I did not deploy to the cloud: it requires your Render/GitHub accounts and publishing the code — your decision.

**Free-tier limitations:** 512 MB RAM (API fits at ≈370 MB, little headroom); services sleep after inactivity (first request
≈30–60 s); ephemeral disk → SQLite history and uploads reset on redeploy (use Render PostgreSQL via `DATABASE_URL` to persist);
CPU-only. Streamlit Community Cloud could host the dashboard alone, but the API still needs a host.

### Troubleshooting
| Symptom | Cause / fix |
|---|---|
| Dashboard: "Cannot reach the PredictOps API" | start the API first; check `API_BASE_URL`; in Docker run `docker compose down && docker compose up` (stale network) |
| `/health` → `degraded`, `models_loaded: false` | artifacts missing → `python run.py pipeline` |
| `XGBoostError: libomp` on macOS | `brew install libomp` |
| 401 from API | `API_KEY` is set — send header `X-API-Key` (dashboard reads `API_KEY` env) |
| 413 on upload | file larger than `MAX_UPLOAD_MB` |
| scikit-learn "InconsistentVersionWarning" | install the exact versions in `requirements.lock` / `requirements-deploy.txt` |
