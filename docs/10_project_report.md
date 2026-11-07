# PredictOps — Intelligent Equipment Health & Failure Prediction Platform

**A Project Report submitted in partial fulfilment of the requirements for the degree of
Bachelor of Engineering / Technology in Computer Science and Engineering**

| | |
|---|---|
| Submitted by | [Student name] — [Roll / Register number] |
| Under the guidance of | [Guide name, designation] |
| Department | Department of Computer Science and Engineering |
| Institution | [College name, University] |
| Academic year | [2026–2027] |

> Placeholders in `[ ]` are to be filled in by the student. All results in this report were produced by the code in this
> repository on the MetroPT-3 dataset; the command that regenerates each number is given in the referenced `docs/` file.

---

## Abstract
Unplanned failures of railway air compressors interrupt service and are expensive to repair. This project builds
**PredictOps**, an end-to-end machine-learning platform that monitors the Air Production Unit (APU) of a metro train using the
public MetroPT-3 dataset (1,516,948 multivariate sensor readings, February–September 2020). The dataset is unlabeled; the only
ground truth is four company-reported air-leak failure windows. We built a leakage-safe pipeline (validation, gap-aware
segmentation, frozen-sensor detection, 132 past-only time-series features, chronological event-aware splits) and compared
five supervised classifiers, four anomaly detectors and two deep-learning sequence models. All supervised models detected the two
validation failures almost perfectly (out-of-fold PR-AUC up to 0.999), but **missed the unseen test failure**, which had a
different sensor signature; SHAP explanations show why. A One-Class SVM trained only on normal operation detected **all three
evaluated failures, including the test failure**, with about 0.16 false-alarm episodes per day. Early warning several hours in
advance was tested and found **not supported** by this data. The detectors feed an explainable health and risk engine, a
FastAPI backend with SQLite storage, and a ten-page Streamlit dashboard with historical replay, maintenance alerts, PDF/CSV reports
and drift monitoring. The work shows that with very few failure events, anomaly detection plus honest evaluation is more reliable
than supervised failure prediction.

**Keywords:** predictive maintenance, anomaly detection, time-series feature engineering, One-Class SVM, XGBoost, SHAP, data leakage, MLOps.

---

## 1. Introduction
Industrial equipment produces continuous sensor data. Predictive maintenance (PdM) uses that data to detect deterioration
early, so maintenance can be planned instead of reacting to breakdowns. A metro train's APU compressor supplies compressed air for
braking, suspension and doors. An air leak forces the compressor to run almost continuously, which overheats the oil and can
lead to service failures. PredictOps analyses recorded APU sensor data to assess equipment health, detect abnormal behaviour,
explain it, and recommend maintenance actions.

## 2. Problem statement
Given multivariate sensor recordings from one metro APU compressor and four reported failure windows, design and evaluate a
system that (a) detects abnormal and failure behaviour, (b) estimates failure risk and equipment health, (c) explains its outputs
in terms of sensor measurements, and (d) presents results in an interactive application. It must use scientifically valid
evaluation (no data leakage, no tuning on test data), and must not overstate capabilities the data can't support.

## 3. Motivation
- Reactive maintenance causes service disruption; fixed-interval maintenance wastes resources.
- Many student and academic PdM projects report inflated accuracy because of random splits on time series, resampling before splitting,
  or accuracy on heavily imbalanced data. This project aims to show a **correct** methodology and an honest result.
- Explainability is required for maintenance engineers to trust and act on alerts.

## 4. Objectives
1. Inspect and document the real dataset, including its labeling limitations.
2. Build a reusable, leakage-safe preprocessing and time-series feature pipeline.
3. Construct defensible targets from the reported failure windows; assess supervised failure detection and early warning.
4. Compare classical ML (LR, DT, RF, SVM, XGBoost), anomaly detection (IF, LOF, OCSVM, statistical) and deep learning (LSTM, 1D-CNN).
5. Study class imbalance (SMOTE, ADASYN, sampling, class weights) without contaminating evaluation.
6. Explain models with SHAP; build an explainable health and risk engine with maintenance recommendations.
7. Deliver a FastAPI + SQLite + Streamlit application with replay, reports and monitoring; test, containerise and document it.

## 5. Existing system and its limitations
| Existing practice | Limitation |
|---|---|
| Fixed-interval (preventive) maintenance | ignores actual condition; over- or under-maintains |
| Single-sensor threshold alarms (e.g. the APU's own low-pressure switch, LPS) | late, noisy; in this data LPS never fired during failures 1 and 2 |
| Supervised failure classifiers trained on few events | learn the signature of past failures; may miss new failure shapes (shown in this project) |
| Black-box models | no explanation, so low operator trust |

## 6. Proposed system
PredictOps combines a **normal-behaviour anomaly model** (main detector), a **failure-signature classifier** (recognises
previously seen air-leak patterns), **sensor range checks** and **alert history** into an explainable health score. It serves results
through an API and a dashboard, with traceable model versions, reports and drift monitoring. The system describes itself honestly:
it detects ongoing abnormal behaviour; it does not claim hours-ahead prediction or remaining useful life (RUL).

## 7. Literature review
- **Dataset and APU failures.** Veloso et al. [2] describe the MetroPT data collection and APU failure types. Davari et al. [1]
  propose deep-learning anomaly detection for the APU and evaluate against reported failures. Barros et al. [3] study failure detection
  of the APU in an operational context. These works motivate treating the problem as **anomaly detection evaluated against
  failure reports**, which this project follows.
- **Ensembles.** Random Forests [4] average decorrelated trees; XGBoost [5] is a regularised gradient-boosting system widely used
  on tabular data. SVMs [6] maximise the class margin.
- **Anomaly detection.** Isolation Forest [7] isolates anomalies with random partitions; LOF [8] compares local densities;
  One-Class SVM [9] estimates the support of the normal distribution.
- **Imbalance.** SMOTE [10] and ADASYN [11] synthesise minority samples by interpolation. Their independence assumptions are
  questionable for autocorrelated time series. For imbalanced evaluation, PR curves are more informative than ROC [12].
- **Sequences.** LSTM networks [13] model long-range temporal dependencies.
- **Explainability.** SHAP [14] attributes predictions with Shapley values; TreeSHAP [15] computes them exactly for tree ensembles.
- **Time-series validation.** Random K-fold leaks information on dependent data; blocked or forward-chaining validation is preferred [16].

## 8. Dataset description
MetroPT-3 (UCI ML Repository, id 791): `MetroPT3(AirCompressor).csv` (1,516,948 rows × 17 columns) plus `Data Description_Metro.pdf`.
- **15 sensors**: 7 analog (TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current) and 8 digital (COMP, DV_eletric,
  Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses). Full data dictionary: `docs/data_dictionary.md`.
- **Sampling ≈ 10 s** in the provided file, although the PDF says 1 Hz and 15.2 M points (the file is a 1-in-10 subsample).
- No missing values or duplicate timestamps; **230 logger-off gaps > 10 min** (≈ 901 h), which split the data into 231 segments.
- **7 frozen-sensor periods** (3.3% of rows) where all analog values repeat exactly for 1–51 h.
- **No label column.** Four reported air-leak windows (18 Apr; 29–30 May; 5–7 Jun; 15 Jul 2020) cover 1.97% of rows.
  The official table has numbering and maintenance-date inconsistencies, which are documented and not altered.
- EDA found **three unreported periods** with the same sustained failure signature (12 Apr, 13 May, 19–20 May).

## 9. System requirements
| | Minimum | Used |
|---|---|---|
| Hardware | 64-bit CPU, 4 GB RAM (app); 8 GB to retrain | Apple-silicon Mac, 8 cores, 8 GB |
| OS | macOS / Linux / Windows (Docker) | macOS |
| Software | Python 3.12, pip, (Docker optional) | Python 3.12.16, Docker 28.4 |

## 10. Technology stack
Python, NumPy, pandas + PyArrow (Parquet), SciPy, scikit-learn, imbalanced-learn, XGBoost, PyTorch, SHAP, MLflow (local),
FastAPI + Pydantic, SQLAlchemy + SQLite, Streamlit + Plotly, reportlab, pytest, Docker. Rationale per component: `docs/00_setup_and_architecture.md` §2.

## 11. System architecture
See `docs/11_diagrams.md` (system architecture, data flow, ML pipeline, use-case, sequence, ER and deployment diagrams).
In short: raw data → validation → typed Parquet → features → {anomaly model, signature model} → health engine →
FastAPI → SQLite + dashboard → reports. Training and inference share **one** feature and inference code path.

## 12. Data preprocessing
Timestamp parsing and sorting; schema and range validation (reused for uploads); dtype optimisation (float32/int8, 4× less RAM);
gap segmentation (no interpolation across gaps; forward-fill ≤ 60 s only at inference); frozen-sensor and uncertain-episode flags
(rows kept, excluded from training and metrics); outliers kept (they are often the failure). Scaling uses StandardScaler fitted on
training rows only. RobustScaler was rejected because TP2's interquartile range is only 0.004 bar. Details: `docs/03_preprocessing.md`.

**Targets.** `y_detect` = inside a reported window; `y_warn_{2,6,12}h` = a reported window starts within H hours (undefined inside
windows). **Splits.** Train Feb–20 May (event 1), validation 21 May–30 Jun (events 2, 3), test Jul–Aug (event 4), with a 12 h
embargo. Model selection uses forward-chaining folds only.

## 13. Feature engineering
1-minute aggregation with right-closed intervals; 132 past-only features over 5 min / 30 min / 2 h windows: rolling
mean/std/min/max/range/median/skew/kurtosis, duty cycle, load cycles per hour, oil temperature while loaded, consecutive-minute
run lengths, lags and rates of change, deviations from a 24 h baseline, TP3 − Reservoirs. Selection on training data only:
variance filter → correlation filter (|r| > 0.98) → XGBoost gain gives 33 features. A prefix-invariance test proves no feature uses
future data. Details: `docs/04_features_and_imbalance.md`.

## 14. Class imbalance handling
Failure minutes are 1.1% of training minutes (detection) and 0.27% (6 h warning), but come from only **one** training event each.
SMOTE, ADASYN, random over- and undersampling and class weights were compared **inside training folds only** (imblearn pipeline).
Detection PR-AUC was 0.97–0.999 whatever the method; warning stayed at chance. Synthetic resampling gave no benefit and was not
adopted. Class weights plus validation-tuned thresholds were used instead.

## 15. Machine learning algorithms
Logistic Regression, Decision Tree, Random Forest, RBF-SVM (20k stratified subsample) and XGBoost, each in a
Pipeline(imputer + scaler → model), tuned by random search on forward-chaining folds (PR-AUC). The threshold was chosen on
out-of-fold validation predictions, and the model was selected **before** the test set was opened. Details: `docs/05_ml_models.md`.

## 16. Deep learning implementation
PyTorch LSTM (2 × 32 units) and 1D-CNN (2 conv blocks) on 60-minute × 13-channel sequences built within segments; normalisation
from training data; class-weighted BCE; Adam; early stopping on validation loss. Sequences can't span partitions (12 h embargo).
Results were checked across 4 random seeds. Details: `docs/07_dl_shap_health.md` Part A.

## 17. Anomaly detection
Isolation Forest, LOF (novelty mode), One-Class SVM and a robust-z statistical detector were trained only on known-normal
February–April minutes, using 18 label-free physical features. Thresholds are the 99.5/99.9/99.99% quantiles of a separate
normal calibration period. An alert requires ≥ 10 flagged minutes within 30 minutes. Selection was made on validation.
Details: `docs/06_anomaly_detection.md`.

## 18. Explainable AI
TreeSHAP for the XGBoost signature model (exact; additivity verified by a test); KernelSHAP for the One-Class SVM on selected
minutes; robust-z deviations for live anomaly explanations. Importance, contribution, correlation and causation are explicitly
distinguished; SHAP is never presented as proof of cause.

## 19. Equipment health assessment
Health score = 100 × (1 − [0.40·anomaly + 0.30·signature + 0.20·sensor-range + 0.10·history]), with Healthy ≥ 75, Warning 45–75,
Critical < 45 and escalation rules. Failure risk is Low/Medium/High by rule; anomaly status is Normal/Anomalous; reasons and
recommendations are generated from the evidence. It's a transparent heuristic, validated on events 2–3 before testing.
Details: `docs/07_dl_shap_health.md` Part C.

## 20. Backend implementation
FastAPI with layered routes, services, data access and inference; 25+ endpoints (upload, batch and observation prediction, history,
explain, alerts and acknowledgement, replay, drift, PDF/CSV reports, model info); Pydantic validation; structured errors; request
logging; optional API key; CORS allow-list. Details: `docs/08_application.md`.

## 21. Frontend implementation
A ten-page Streamlit dashboard (Overview, Sensor Monitoring, Historical Replay, Failure Prediction, Anomaly Detection,
Explainable AI, Equipment Health, Maintenance, Model Performance, Data & Model Monitoring) with a custom dark industrial theme,
a colour-blind-validated palette, status shown as icon plus label, and charts that never bridge data gaps.

## 22. Database design
SQLite (via SQLAlchemy, so it can be swapped for PostgreSQL): equipment, datasets (file references, not copies), prediction runs
(with model versions), per-minute predictions, alerts (with acknowledgement), model versions; foreign keys with cascades and
indexes. ER diagram: `docs/11_diagrams.md` §6.

## 23. Experimental setup
| Item | Setting |
|---|---|
| Unit of evaluation | 1-minute rows (`usable` only) + **event level** |
| Development data | train split (event 1) + validation split (events 2, 3) |
| Model selection | forward-chaining folds A (→ event 2) and B (→ event 3) |
| Test | test split, event 4, evaluated **once** |
| Event detection rule | alert episode overlapping [onset − 12 h, end]; others are false alarms |
| Seeds | 42 (DL additionally 1, 2, 3) |
| Hardware | Apple-silicon Mac, 8 GB RAM; PyTorch on MPS |

## 24. Evaluation metrics
Precision, recall, F1, PR-AUC (primary, suited to rare positives), ROC-AUC, balanced accuracy, MCC, Cohen's κ, FPR/FNR, Brier
score and calibration, plus **events detected, detection delay and false-alarm episodes per day**. Accuracy is reported but
never used for selection: "always normal" scores ≈ 99.6%.

## 25. Results and discussion

### 25.1 Supervised detection
| Model | Validation OOF PR-AUC | Test PR-AUC | Test ROC-AUC | Test recall @ threshold |
|---|---|---|---|---|
| XGBoost (selected) | 0.9992 | 0.480 | 0.985 | 0.00 |
| Logistic Regression | 0.9992 | 0.040 | 0.581 | 0.00 |
| SVM (RBF) | 0.9987 | 0.003 | 0.200 | 0.00 |
| Random Forest | 0.9939 | 0.163 | 0.988 | 0.00 |
| Decision Tree | 0.9880 | 0.046 | 0.686 | 0.00 |

Test base rate 0.0037 (270 of 72,651 minutes). Validation looked near-perfect, but every model missed event 4 at its validation
threshold. SHAP shows why: the models depend on DV_pressure ≈ 2 bar, the signature of events 1–3. Event 4 showed low TP3,
LPS active and hot oil instead. XGBoost and RF still *ranked* event-4 minutes higher (ROC-AUC ≈ 0.99), but their scores stayed below
the thresholds learned on other events.

### 25.2 Early warning
For 2/6/12 h horizons, test PR-AUC was 0.0008–0.0067 against random baselines of 0.0017–0.0099. Hours-ahead prediction is
**not supported** by four events; the application therefore doesn't offer it.

### 25.3 Anomaly detection (main result)
| Detector | Validation events | Test events | Test false alarms/day | Test minute recall / precision |
|---|---|---|---|---|
| **One-Class SVM (selected)** | 2/2 | **1/1** | 0.16 | 0.61 / 0.29 |
| Isolation Forest | 2/2 | 1/1 | 0.26 | 0.45 / 0.23 |
| LOF | 0/2 | 0/1 | 0 | 0 |
| Robust z | 2/2 | 0/1 | 0 | 0 |

Alerts arrived 16–34 minutes after the reported onsets, so this is **detection, not prediction**. The OCSVM score rose steadily for
about 14 h before event 4 without crossing the threshold, a promising but single-event observation. The OCSVM also flagged 98.8% of
the three unreported episodes. LOF failed because duplicate "compressor-off" minutes make the local densities extreme; robust-z
failed because of seasonal drift.

### 25.4 Deep learning
Every LSTM and CNN run reached validation PR-AUC 1.0. On the test set the LSTM detected event 4 in 3 of 4 seeds, at about 0.5 false
alarms per day in two of them, and the CNN in 0 of 4. Because the results are unstable and validation can't separate good from bad runs,
DL was **not deployed**.

### 25.5 Health engine
On test, 99.3% of normal minutes are Healthy and 61.5% of event-4 minutes Critical; event 4 is detected with 0.10 false Critical
episodes per day (validation: 2/2 events, 0.07/day).

### 25.6 Engineering
69 automated tests pass (unit, integration, API, UI). One day of raw data is assessed in 0.17 s; API replay step 162 ms; the API
container uses ≈ 370 MB. A Docker deployment was verified end to end.

### 25.7 Discussion
The central lesson is the gap between validation and test performance when there are few events. Validation events resembled the
training event, so validation scores overstated generalisation. Keeping a truly unseen test event, never tuning on it, and reporting
its failure honestly is what exposed this. It also motivated the hybrid design, in which a label-free normal-behaviour model is the
primary detector.

## 26. Screenshots
`reports/screenshots/`: `overview.png`, `sensors.png`, `replay.png`, `prediction.png`, `anomaly.png`, `explain.png`, `health.png`,
`maintenance.png`, `performance.png`, `monitoring.png` (captured from the running application).

## 27. Limitations
- One compressor, seven months, four reported failures of one type (air leak); results can't be claimed to generalise to other machines or failure types.
- Labels are approximate (window boundaries) and incomplete (three unreported failure-like episodes).
- No hours-ahead early warning and no RUL estimation, because the data doesn't support them.
- Health-score weights are expert-set and validated on two events only.
- Historical replay only; no live sensor integration.
- Drift thresholds (PSI) need calibration to operating variability.

## 28. Future scope
- More failure events and maintenance logs (other MetroPT trains) to re-test supervised and early-warning models.
- Use the anomaly-score *trend* (seen before event 4) as an explicit early-warning feature once more events are available.
- Online learning and threshold adaptation for seasonal drift; operating-regime-specific anomaly models.
- Live MQTT/OPC-UA ingestion; PostgreSQL/TimescaleDB; user accounts and roles.
- Cost-sensitive alerting tuned with maintenance staff.

## 29. Conclusion
PredictOps delivers a complete, tested and documented predictive-maintenance application on real industrial data. Its most
important contribution is methodological honesty: by preventing leakage, holding out a genuinely unseen failure, and reporting
negative results (supervised generalisation, early warning, deep learning instability), the project identified the approach that
actually works on this data, a normal-behaviour One-Class SVM inside an explainable health engine, which detected every evaluated
failure with about one false alarm per week.

## References
1. N. Davari, B. Veloso, R. P. Ribeiro, P. M. Pereira, J. Gama, "Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry," *IEEE 8th Int. Conf. on Data Science and Advanced Analytics (DSAA)*, 2021. doi:10.1109/DSAA53316.2021.9564181
2. B. Veloso, R. P. Ribeiro, P. M. Pereira, J. Gama, "The MetroPT dataset for predictive maintenance," *Scientific Data*, 9, 764, 2022. doi:10.1038/s41597-022-01877-3
3. M. Barros, B. Veloso, P. M. Pereira, R. P. Ribeiro, J. Gama, "Failure detection of an air production unit in the operational context," in *IoT Streams for Data-Driven Predictive Maintenance and IoT, Edge, and Mobile for Embedded Machine Learning*, Springer, 2020. doi:10.1007/978-3-030-66770-2_5
4. L. Breiman, "Random Forests," *Machine Learning*, 45(1), 2001.
5. T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *Proc. 22nd ACM SIGKDD*, 2016.
6. C. Cortes, V. Vapnik, "Support-vector networks," *Machine Learning*, 20(3), 1995.
7. F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation Forest," *IEEE Int. Conf. on Data Mining (ICDM)*, 2008.
8. M. M. Breunig, H.-P. Kriegel, R. T. Ng, J. Sander, "LOF: Identifying Density-Based Local Outliers," *ACM SIGMOD*, 2000.
9. B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, R. C. Williamson, "Estimating the support of a high-dimensional distribution," *Neural Computation*, 13(7), 2001.
10. N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, 16, 2002.
11. H. He, Y. Bai, E. A. Garcia, S. Li, "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," *IEEE IJCNN*, 2008.
12. T. Saito, M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, 10(3), 2015.
13. S. Hochreiter, J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 9(8), 1997.
14. S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.
15. S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, 2, 2020.
16. C. Bergmeir, J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Information Sciences*, 191, 2012.
17. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, 12, 2011.
18. MetroPT-3 Dataset, UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
