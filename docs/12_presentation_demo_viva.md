# 12 — Presentation, Demonstration Script & Viva Preparation

---

## Part A — Slide-by-slide outline (≈18 slides, 12–15 min)
| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | Title | PredictOps — Intelligent Equipment Health & Failure Prediction; name, guide, college | dashboard Overview screenshot |
| 2 | Introduction | metro APU compressor; why air leaks matter; predictive vs preventive maintenance | APU photo/diagram (cite source) |
| 3 | Problem statement | detect abnormal / failure behaviour from sensors, explain it, recommend actions — honestly | one-sentence statement |
| 4 | Motivation | reactive cost; inflated results in many PdM projects (leakage, accuracy on imbalance) | 99.6% "accuracy" of always-normal |
| 5 | Objectives | 7 objectives (report §4) | checklist |
| 6 | Dataset | MetroPT-3: 1.52 M readings, 15 sensors, ~10 s, 230 gaps, **no labels**, 4 reported failures | `01_failure_timeline.png`, `02_data_coverage.png` |
| 7 | Key EDA findings | failure signature; abrupt vs gradual (event 4); 3 unreported episodes; frozen sensors | `12_event3_zoom.png`, `12_event4_zoom.png` |
| 8 | Methodology | flags → targets → chronological splits + embargo → past-only features → models → event-level eval | ML pipeline diagram (docs/11 §3) |
| 9 | Architecture | offline pipeline + online app | system architecture diagram (docs/11 §1) |
| 10 | Algorithms | 5 ML + 4 anomaly + 2 DL, one line each; SMOTE/ADASYN study | small table |
| 11 | Results 1 — validation vs test | supervised: 0.999 validation → missed event 4 | `test_event4_scores.png` + table |
| 12 | Results 2 — anomaly detection | OCSVM 3/3 events, 0.16 false alarms/day, precursor trend | `scores_event4.png` |
| 13 | SHAP explanations | why event 4 was missed (DV_pressure vs TP3) | `xgb_waterfall_event4_missed.png` |
| 14 | Health engine | score formula, statuses, 99.3% normal Healthy, event 4 Critical | `health_event4.png` |
| 15 | Dashboard | 4 screenshots: Overview, Anomaly, Explainable AI, Maintenance | `reports/screenshots/` |
| 16 | Advantages | leakage-safe, explainable, honest evaluation, full app + tests + Docker | bullets |
| 17 | Limitations & future scope | 4 events, one machine, no early warning/RUL; more events, online learning, live ingestion | bullets |
| 18 | Conclusion | anomaly + explainable health engine is what works on this data; methodology > headline accuracy | key numbers |

Tip: build the slides as you like (PowerPoint/Google Slides); every figure above already exists in `reports/figures/` or `reports/screenshots/`.

---

## Part B — Live demonstration script (≈8 min)
**Before the viva:** `cd PredictOps && source .venv/bin/activate && python run.py` (or `docker compose up`). Open
http://localhost:8501 and http://127.0.0.1:8000/docs in two tabs. Do one prediction run beforehand so the Overview cards are filled.

1. **Overview (45 s).** "1.5 M real readings from a metro compressor, Feb–Aug 2020. Health trend over 7 months; the dotted line
   separates data used for training from the **unseen** test period. Red bands are the 4 company-reported failures."
2. **Sensor Monitoring (60 s).** Jump to → *Reported failure #4*. Show TP3 falling and oil temperature rising; red = reported failure,
   amber = our alert. "Each sensor has its own panel — no misleading dual axes; gaps in the logger are shown as gaps."
3. **Failure Prediction (90 s).** Sample → `event4_2020-07-15.csv` → *Run batch prediction*. Explain the horizon note: "it detects
   failures in progress; hours-ahead prediction was tested and not supported, so we don't offer it." Move the minute slider to
   ~18:30 → SHAP bar: "low TP3 pushes toward failure, but DV_pressure ≈ 0 pushes against — the supervised model learned events 1–3."
   Download the PDF report and open it.
4. **Anomaly Detection (60 s).** Same range. "Model trained only on normal data. The score climbs ~14 h before event 4 and
   crosses the threshold at the event; alert raised." Point to severity levels and the alert table.
5. **Explainable AI (45 s).** Global importance; waterfall for the highest-risk minute; anomaly deviations (LPS active, TP3 low, oil hot).
   "SHAP explains the model, not the physical cause."
6. **Equipment Health (45 s).** Bands; select the worst minute → reasons. "Heuristic index, not a probability."
7. **Maintenance (30 s).** Acknowledge an alert with a note → it moves to history.
8. **Historical Replay (60 s).** Scenario *Event 4 build-up* → Start. "Each step re-runs the complete pipeline on the raw recorded
   readings — this is replay, not live monitoring."
9. **Model Performance & Monitoring (45 s).** Validation vs test bars; confusion matrices; monthly drift: "summer oil temperature drift
   — drift ≠ failure."
10. **API (15 s).** `/docs` → try `GET /health`. "Same models serve the dashboard, reports and API; 69 automated tests."

**If something fails:** API unreachable → start `uvicorn backend.main:app --port 8000`; slow first load → caches warm in a few seconds;
have `reports/screenshots/` ready as a backup.

---

## Part C — Viva questions and answers

### Predictive maintenance & dataset
**Q1. What is predictive maintenance?** Scheduling maintenance from the *measured condition* of equipment instead of fixed intervals
(preventive) or after breakdown (reactive). It aims to intervene just before failure.

**Q2. What is MetroPT-3?** Public UCI dataset from the Air Production Unit compressor of a Porto metro train: 15 analog/digital
signals (pressures, oil temperature, motor current, valve signals), Feb–Sep 2020, plus a table of 4 reported air-leak failures.

**Q3. Is the dataset labeled?** No. We constructed labels only from the 4 official failure windows, and flagged periods whose status
is unknown (unreported failure-like episodes) as *uncertain* rather than inventing labels.

**Q4. What was surprising in the data?** The file is sampled every ~10 s (not 1 Hz as documented); 230 logger gaps; 7 frozen-sensor
periods; event 4 behaves differently from events 1–3; three failure-like episodes were never reported.

### Time series, preprocessing, leakage
**Q5. Why not a random train/test split?** Consecutive readings are nearly identical; a random split puts near-copies in train and test,
inflating scores. We split chronologically by events with a 12 h embargo.

**Q6. What is data leakage? Give examples you prevented.** Using information at training time that would not be available at prediction
time. Prevented: random splits, scalers fitted on all data, resampling before splitting, rolling windows using future values, the
`minutes_to_onset` column as a feature, tuning on the test set, and leave-one-event-out CV that trains on the future.

**Q7. How did you prove your features don't use the future?** A test builds features on data cut at time T and on the full data;
all features at or before T are identical.

**Q8. What is time-series (forward-chaining) validation?** Each fold trains only on data before its validation block — like how the
model is used in reality.

**Q9. How did you handle missing data and gaps?** No missing values in the raw data. Logger gaps start a new segment and are never
interpolated; at inference only gaps ≤ 60 s are forward-filled; remaining NaN go to a training-fitted median imputer.

**Q10. Why StandardScaler and not RobustScaler?** TP2 and DV_pressure are ~0 more than half the time, so their IQR is 0.004 bar and
RobustScaler would turn a normal 10 bar reading into ~2,500.

### Feature engineering
**Q11. Which features did you engineer and why?** Rolling statistics (5 min/30 min/2 h), duty cycle and load cycles (a leak makes the
compressor run continuously), oil temperature while loaded (regime-aware), deviations from a 24 h baseline (removes seasonal drift),
lags and rates of change, run-lengths of abnormal readings, TP3 − Reservoirs.

**Q12. How did you choose window sizes?** From the real 10 s sampling and EDA: 5 min ≈ one load cycle, 30 min ≈ the time for the
failure signature to become sustained, 2 h ≈ the shortest warning horizon.

**Q13. How was feature selection done without leakage?** Using training rows only: variance filter, correlation filter (|r| > 0.98),
then XGBoost gain → 33 features.

### Imbalance, SMOTE, ADASYN
**Q14. How imbalanced is the data?** ≈1:87 minutes for detection in training (1.1%), ≈1:373 for 6 h warning — but the real scarcity is
**events**: one training event.

**Q15. What do SMOTE and ADASYN do?** SMOTE creates synthetic minority samples by interpolating between a minority point and its minority
neighbours. ADASYN does the same but generates more samples near hard-to-learn (boundary) points.

**Q16. Did you use them? Why not?** We tested them inside training folds only. They gave no PR-AUC improvement: synthetic minutes
interpolated from one autocorrelated event add no new information and can create physically impossible feature combinations. We used
class weights and validation-tuned thresholds instead.

**Q17. Why never resample validation/test data?** Evaluation must reflect the real class distribution; resampling it would fake performance.

### Algorithms
**Q18. Decision Tree?** Recursive yes/no splits that maximise class purity (Gini); interpretable, prone to overfitting → depth limits.

**Q19. Random Forest?** Many trees on bootstrap samples with random feature subsets; averaging reduces variance.

**Q20. SVM?** Finds the separating boundary with maximum margin; the RBF kernel allows non-linear boundaries. We trained it on a 20k
stratified subsample because training cost grows ~quadratically.

**Q21. XGBoost?** Gradient boosting: trees added sequentially, each fitting the errors (gradients) of the current ensemble, with
regularisation, shrinkage and subsampling.

**Q22. Why did XGBoost miss test event 4 despite 0.999 validation PR-AUC?** It learned the events 1–3 signature (DV_pressure ≈ 2 bar).
Event 4 presented as falling TP3 and hot oil with DV_pressure ≈ 0; SHAP shows these opposing contributions. Its scores rose
(ROC-AUC 0.985) but stayed below the threshold learned on other events.

**Q23. Why not just lower the threshold?** That would use test labels to tune — the test would no longer be an honest estimate.

### Deep learning
**Q24. LSTM?** A recurrent network with input/forget/output gates and a cell state that carries information over long sequences,
mitigating vanishing gradients.

**Q25. 1D-CNN?** Convolution filters slide along time to detect local patterns (e.g. loss of the on/off oscillation); pooling summarises them.

**Q26. How did you avoid sequence leakage?** Sequences built within one segment and one split; splits separated by 12 h > sequence
length; normalisation from training data; label = last minute only.

**Q27. Why wasn't DL deployed?** Perfect on validation but unstable on test across random seeds (LSTM detected event 4 in 3/4 seeds,
CNN in 0/4), and validation could not tell good runs from bad. A stable, simpler model was preferred.

### Anomaly detection
**Q28. Isolation Forest?** Random trees isolate points; anomalies are isolated in fewer splits (shorter path length).

**Q29. One-Class SVM?** Learns a boundary enclosing (1−ν) of normal data in kernel space; points outside are anomalous. ν = 0.01.

**Q30. Why did anomaly detection succeed where supervised failed?** It needs no failure labels and models normal behaviour, so any
deviation — including a new failure shape — is flagged.

**Q31. How were anomaly thresholds chosen without labels?** As 99.5/99.9/99.99% quantiles of scores on a separate normal calibration
period; alerts need persistence (≥10 of 30 min) to avoid one-off spikes.

**Q32. Is every anomaly a failure?** No. Unusual reading → abnormal behaviour (persistent) → confirmed failure (only reported events).

### Explainability
**Q33. What is SHAP?** Shapley-value attribution: each feature's average marginal contribution to the prediction over all feature
orderings; base value + Σ SHAP = model output (log-odds for XGBoost — verified by a test).

**Q34. Feature importance vs contribution vs correlation vs causation?** Importance = average reliance (global); contribution = effect on
one prediction (local); correlation = data association; causation = mechanism — SHAP gives none of the last.

### Metrics
**Q35. Precision / recall / F1?** Precision = TP/(TP+FP) — how many alarms were right; recall = TP/(TP+FN) — how many failures were
caught; F1 = harmonic mean.

**Q36. ROC-AUC vs PR-AUC?** ROC-AUC: ranking quality using TPR vs FPR — looks optimistic with rare positives. PR-AUC: precision vs recall —
its random baseline equals the positive rate (0.0037 on test), so it reflects rare-event performance better.

**Q37. Why event-level metrics?** Engineers care whether each failure was caught, how early, and how many false alarms per day — not
per-minute counts.

**Q38. Trade-off between false alarms and missed failures?** Lower threshold → more failures caught but more false alarms (alarm fatigue);
higher → fewer alarms but missed failures. We choose on validation and report both.

### Application, MLOps, monitoring
**Q39. Why FastAPI?** Typed validation with Pydantic, automatic OpenAPI docs, async, fast; clean separation from the UI.

**Q40. How is the same preprocessing guaranteed in training and deployment?** One shared package (`src/predictops`): the API, replay and
training all call the same `build_features` and `PredictOpsPredictor`; models are pipelines that include their scaler.

**Q41. What does the database store?** Datasets (file references), runs with model versions, per-minute predictions, alerts with
acknowledgement, model versions — not copies of the sensor data.

**Q42. How is model versioning done?** Versioned metadata JSON per model (params, threshold, features, train period, dataset hash,
metrics), MLflow runs, and the version stored with every prediction run.

**Q43. Data drift vs concept drift vs degradation?** Data drift: input distribution changes (summer oil temperature). Concept drift: the
input→failure relation changes. Degradation: worse accuracy — measurable only with new labels.

**Q44. Is the dashboard real-time?** No — it replays historical data through the real pipeline; this is stated in the UI.

**Q45. What would you do with more time/data?** More failure events from other trains, revisit early warning using anomaly-score trends,
regime-specific models, online threshold adaptation, live data ingestion.
