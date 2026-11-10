# PredictOps — Intelligent Equipment Health & Failure Prediction

An end-to-end predictive-maintenance platform for the **MetroPT-3** metro-train air compressor
(Air Production Unit) dataset: anomaly detection, failure-window prediction, explainability (SHAP),
an equipment health engine, a FastAPI backend and an interactive dashboard.

> Status: **All stages complete** — see the acceptance report [docs/13_acceptance_report.md](docs/13_acceptance_report.md). See [`PROGRESS.md`](PROGRESS.md) for the phase log.

## Documentation (read in order)
| Doc | Contents |
|---|---|
| [docs/00_setup_and_architecture.md](docs/00_setup_and_architecture.md) | Environment, tech stack decisions, architecture, folder layout |
| [docs/01_dataset_inspection.md](docs/01_dataset_inspection.md) | Auto-generated facts about the real data, label investigation, feasibility |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Every column, unit, dtype, range and meaning |
| [docs/02_eda.md](docs/02_eda.md) | EDA: every chart explained, data-quality findings, decisions |
| [docs/03_preprocessing.md](docs/03_preprocessing.md) | Flags, target definitions, chronological splits, leakage checks |
| [docs/04_features_and_imbalance.md](docs/04_features_and_imbalance.md) | Feature engineering, selection, SMOTE/ADASYN study |
| [docs/05_ml_models.md](docs/05_ml_models.md) | 5 ML models, tuning, selection, honest test results |
| [docs/06_anomaly_detection.md](docs/06_anomaly_detection.md) | Isolation Forest / LOF / One-Class SVM / statistical; main detector |
| [docs/07_dl_shap_health.md](docs/07_dl_shap_health.md) | LSTM & 1D-CNN, SHAP explanations, health & risk engine |
| [docs/08_application.md](docs/08_application.md) | API, database, dashboard, replay, reports, monitoring, security |
| [docs/09_testing_performance_deployment.md](docs/09_testing_performance_deployment.md) | 73 tests, benchmarks, Docker/Render deployment, troubleshooting |
| [docs/14_live_simulation.md](docs/14_live_simulation.md) | synthetic live stream that fails at random times (demo only) |
| [docs/10_project_report.md](docs/10_project_report.md) | Full academic project report |
| [docs/11_diagrams.md](docs/11_diagrams.md) | Architecture, DFD, ML pipeline, use-case, sequence, ER, deployment diagrams |
| [docs/12_presentation_demo_viva.md](docs/12_presentation_demo_viva.md) | Slide outline, demo script, 45 viva Q&A |
| [docs/13_acceptance_report.md](docs/13_acceptance_report.md) | Acceptance checklist & audit |

## Dataset
MetroPT-3, UCI ML Repository — https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
Place `MetroPT3(AirCompressor).csv` and `Data Description_Metro.pdf` in `data/raw/`
(the CSV is not committed to Git because of its size). Verify with:
```bash
cd data/raw && shasum -a 256 -c SHA256SUMS
```

## Quick start (macOS)
```bash
brew install python@3.12 libomp
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install -e .
cp .env.example .env
python scripts/build_intermediate.py
python scripts/inspect_dataset.py
python scripts/run_eda.py
python scripts/build_processed.py
pytest -q
```

## Run the application
```bash
python run.py            # API http://127.0.0.1:8000/docs + dashboard http://localhost:8501
python run.py pipeline   # rebuild data, models, history from the raw CSV (~15 min)
python run.py test       # 73 tests (UI tests need the API running)
docker compose up --build   # containerised version
```

## Headline results (honest)
- One-Class SVM (trained on normal data only) detected **all 3 evaluated failures incl. the unseen test failure**, ~0.16 false alarms/day.
- Supervised models: ~0.999 PR-AUC on validation but **missed the unseen test failure** (different pattern) — explained with SHAP.
- Hours-ahead early warning and RUL are **not supported** by this data and are not claimed.

## Git
```bash
git add -A && git commit -m "Stage 3: project setup, ingestion, dataset inspection"
gh repo create PredictOps --private --source=. --push    # or create on github.com and `git remote add origin ...`
```

## Honest scope
- Single compressor, ~7 months, **4 documented failure events** → results are reported per event.
- "Live" monitoring in the app is a **replay of historical data**; the *Live Simulation* page streams **synthetic** data (docs/14).
- No remaining-useful-life (RUL) prediction is claimed.
