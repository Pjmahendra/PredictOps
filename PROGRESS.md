# PredictOps — Progress Log

One entry per stage. Every "verified" item has a command that proves it.

## Stage 1–2 — Requirements & dataset investigation ✅ (2026-11-06)
- Inspected `MetroPT3(AirCompressor).csv` (1,516,948 × 17) and `Data Description_Metro.pdf`.
- Findings: no missing values, ~10 s sampling (not 1 Hz), 230 gaps > 10 min, **no label column**,
  4 air-leak failure windows (1.97% of rows), PDF table inconsistencies.
- Correction logged: an early quick script read the wrong column for LPS; the generated report
  (`docs/01`) is authoritative — LPS never fires during events 1 and 2.
- Doc: `docs/01_dataset_inspection.md`, `docs/data_dictionary.md`.

## Stage 3 — Architecture & setup ✅ (2026-11-06)
**Completed & verified**
- Python 3.12.16 venv, all dependencies installed and importable (`requirements.lock`).
- Folder structure, `configs/config.yaml`, `.env.example`, `.gitignore`, `pyproject.toml`, Git repo initialised (no commit yet).
- Raw data copied to `data/raw/`, write-protected, SHA-256 recorded.
- `src/predictops/data/ingestion.py`: validation (reused later for uploads), dtype optimisation, segmentation.
- `data/intermediate/metropt3.parquet`: 1,516,948 rows, 231 segments, 91 MB RAM, 29 MB disk.
- Tests: `pytest -q` → 14 passed.

**Files created:** requirements.txt, requirements.lock, pyproject.toml, .gitignore, .env.example, .env,
configs/config.yaml, src/predictops/{__init__,config}.py, src/predictops/data/{__init__,ingestion}.py,
scripts/{build_intermediate,inspect_dataset}.py, tests/{test_ingestion,test_intermediate_data}.py,
docs/{00_setup_and_architecture,01_dataset_inspection,data_dictionary}.md, README.md, PROGRESS.md.

**Issues hit:** XGBoost failed to import (missing OpenMP) → fixed with `brew install libomp`.

## Stage 4a — Exploratory Data Analysis ✅ (2026-11-06)
**Completed & verified**
- `src/predictops/eda.py` (reusable stats/plots + data-quality detectors), `scripts/run_eda.py`.
- 16 figures in `reports/figures/eda/`, 9 tables in `reports/eda_stats/`.
- `notebooks/02_exploratory_analysis.ipynb` executed end-to-end without errors.
- `docs/02_eda.md`: every chart explained + decisions for the next stage.
- Tests: `pytest -q` → 18 passed.

**Key results**
- Failure windows have a strong signature (continuous load, H1≈0, DV_pressure≈2 bar, oil +13 °C).
- Events 1–3: abrupt onset, no visible multi-day precursor. Event 4: gradual oil-temp rise / pressure sag.
- 3 unreported failure-like episodes (12 Apr, 13 May, 19–20 May) → label noise; marked "uncertain", not relabelled.
- 7 frozen-sensor runs (3.32% of rows) → to be flagged and excluded.
- TP3 ≡ Reservoirs (r=1.000); COMP/MPG/DV_eletric redundant.

## Stage 4b — Preprocessing, targets & splits ✅ (2026-11-06)
**Completed & verified**
- `src/predictops/preprocessing/{labels,splits,pipeline}.py`, `scripts/build_processed.py`.
- Flags: `is_frozen` (50,418 rows), `uncertain` (17,459 rows), `usable`, `operating_state`.
- Targets: `y_detect` (in window) and `y_warn_{2,6,12}h` (onset within H hours; NaN inside windows).
- Splits: train (event 1) / validation (events 2, 3) / test (event 4), 12 h embargo; forward-chaining CV folds.
- Base preprocessor (median imputer + StandardScaler) fitted on train∩usable only → `models/preprocessing/`.
- 3 raw-format sample CSVs from the test period in `data/sample/`.
- `docs/03_preprocessing.md`. Tests: `pytest -q` → 30 passed.

**Decisions changed during the stage (with evidence)**
- RobustScaler → StandardScaler (IQR of TP2/DV_pressure = 0.004 bar would blow values up ~2,500×).
- Leave-one-event-out CV → forward-chaining CV (LOEO would use June to predict May).

**Limitations found**
- Event 1's 12 h pre-onset is fully frozen → only events 2, 3 provide early-warning training signal.

## Stage 4c — Features, selection, imbalance ✅ (2026-11-06)
- `src/predictops/features/build.py`: 1-min aggregation + 132 past-only features; selection → 33 (train only).
- Detection near-perfect on validation; 6 h warning at chance; SMOTE/ADASYN tested and rejected (no benefit).
- `docs/04_features_and_imbalance.md`; feature leakage test (prefix invariance) passes.

## Stage 5 — Supervised ML ✅ (2026-11-06)
- LR / DT / RF / SVM / XGBoost tuned on forward folds; XGBoost selected BEFORE test (OOF PR-AUC 0.9992).
- **Test (event 4): all models miss at validation thresholds** (recall 0); XGB/RF ROC-AUC 0.985/0.988. Early warning ≤ chance.
- Artifacts + metadata + MLflow runs (`mlruns.db`). `docs/05_ml_models.md`.

## Stage 6 — Anomaly detection, DL, SHAP, health engine ✅ (2026-11-06)
- One-Class SVM (normal-only training) selected on validation: **2/2 validation events, test event 4 detected**, 0.16 false alarms/day. LOF / robust-z failures documented.
- LSTM / 1D-CNN: perfect on validation, unstable on test across seeds → not deployed.
- SHAP: TreeExplainer (live) + KernelSHAP for the anomaly model; explains why event 4 was missed.
- Health engine + single inference pipeline (`src/predictops/inference/predictor.py`): test 99.3% normal minutes Healthy, event 4 → Critical, 0.10 false Critical alarms/day.
- Docs: `06_anomaly_detection.md`, `07_dl_shap_health.md`. Tests: 43 passed.

## Stage 7 — Backend, database, dashboard ✅ (2026-11-06)
- FastAPI (layered routes/services/db/inference), 25+ endpoints; SQLite schema with FK/cascades/indexes; API-key + CORS.
- Streamlit dashboard: 10 pages, dark theme, validated palette, gap-aware charts; screenshots in `reports/screenshots/`.
- Historical replay re-runs the full pipeline on raw readings; PDF + CSV reports; KS/PSI drift monitoring.
- Docs: `08_application.md`. Bugs found by screenshots/tests fixed (NaN JSON, gap lines, legend overlap).

## Stages 8–10 — Testing, performance, deployment, documentation ✅ (2026-11-06)
- 69 tests pass (unit / integration / API / UI end-to-end). Benchmarks in `reports/benchmark.json`.
- Docker image + compose verified end-to-end; Render blueprint prepared (cloud deploy needs your account).
- Notebooks 01–10 executed. Docs 09 (testing/perf/deploy), 10 (project report), 11 (diagrams), 12 (PPT/demo/viva), 13 (acceptance).

## Remaining (your decisions)
- Fill `[ ]` placeholders in docs/10; assemble slides from docs/12 Part A.
- Commit to Git / push to GitHub; optionally deploy via Render blueprint.
