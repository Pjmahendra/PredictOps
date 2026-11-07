# 07 — Deep Learning, SHAP Explainability & Health Engine (Stage 6b–6d)

---

## Part A — Deep learning experiments (LSTM, 1D-CNN)
**Reproduce:** `scripts/train_dl.py` (~30 s on Apple MPS), `scripts/dl_seed_stability.py` (3 extra seeds).
**Code:** `src/predictops/models/deep.py`. **Results:** `reports/dl/*.csv`, `reports/figures/dl/loss_curves.png`.

### Setup
| Item | Choice | Why |
|---|---|---|
| Input | 60-minute sequences × 13 channels (1-min means of 6 analog sensors + 7 digital shares), **no hand-made features** | tests whether the network can learn temporal patterns itself |
| Target | `y_detect` of the **last** minute | no look-ahead |
| Sequences | inside one segment, all minutes usable; train stride 5 min (less redundancy), eval stride 1 | no crossing of recording gaps |
| Partition leakage | sequences are built **separately per split**; splits are separated by a 12 h embargo, longer than the 60 min sequences, so no sequence overlaps two partitions | |
| Normalisation | mean/std of **train** minutes only | |
| Imbalance | `BCEWithLogitsLoss(pos_weight = neg/pos)`, **no SMOTE** (it would break temporal order) | |
| Training | Adam lr 1e-3, batch 256, max 30 epochs, **early stopping** (patience 5) on validation loss | prevents overfitting (`loss_curves.png`) |
| LSTM | 2 layers × 32 units → linear (14,497 params) | learns order and long-range dependencies |
| 1D-CNN | 2 conv blocks (32 filters, kernel 5) → global average pool (7,425 params) | learns local shapes, e.g. loss of the on/off oscillation |

Sequence counts (sequences, positives): train 23,895 (271), validation 39,090 (3,135), test 69,297 (270).

### Results
| Run | Val PR-AUC | Test PR-AUC | Test recall | Event 4 detected | False alarms/day | Train time |
|---|---|---|---|---|---|---|
| LSTM seed 42 | 1.000 | 0.33 | 0.29 | yes | 0.00 | 15 s |
| LSTM seed 1 | 1.000 | 0.07 | 0.97 | yes | 0.52 | – |
| LSTM seed 2 | 1.000 | 0.01 | 0.00 | **no** | 0.42 | – |
| LSTM seed 3 | 1.000 | 0.32 | 1.00 | yes | 0.56 | – |
| 1D-CNN (4 seeds) | 1.000 | 0.002–0.19 | 0.00 | **no** (0/4) | 0.00 | 14 s |

**Interpretation:**
- On validation every network is perfect, so **validation can't tell good runs from bad ones**.
- On test the LSTM sometimes catches event 4, but whether it does depends on the random seed, and when it does it brings ~0.5 false alarms per day.
  The 1D-CNN never does. Like the tabular models, it locks onto the abrupt pattern of events 1–3.
- **Conclusion: deep learning is kept as a documented experiment and NOT deployed.** With ~270 positive training sequences
  from a single event, the networks can't learn a stable, general failure representation. The One-Class SVM is simpler,
  needs no failure labels, gives the same result every run, and generalises better here.
- **Cost:** ~15 s per model on MPS vs ~0.2 s for the One-Class SVM and ~3.5 s for XGBoost.

---

## Part B — Explainable AI with SHAP
**Reproduce:** `scripts/explain.py`. **Code:** `src/predictops/explainability/shap_explain.py`.
**Figures:** `reports/figures/shap/*.png`. **Data:** `reports/shap/*.{csv,json}`.

### Concepts (viva-ready)
| Term | Meaning |
|---|---|
| **Feature importance** (global) | how much a model relies on a feature *on average* (here: mean absolute SHAP value) |
| **Feature contribution** (local) | how much a feature pushed *this one prediction* up or down (one SHAP value) |
| **Correlation** | two variables move together in the data, independent of any model |
| **Causation** | changing X *makes* Y change. **SHAP cannot show this**: it explains the model, not the machine |

SHAP values come from cooperative game theory (Shapley values). Each feature is a "player", and its value is its average marginal
contribution over all orders in which features could be added. For XGBoost the result is in **log-odds**:
`base value + Σ SHAP values = model logit`. A test (`test_shap_additivity`) checks this holds exactly.

### Explainers used
| Model | Explainer | Why |
|---|---|---|
| XGBoost detector | `TreeExplainer` | exact for tree ensembles, milliseconds per row, so it's used live in the API |
| One-Class SVM anomaly detector | `KernelExplainer` (50 background normal minutes, 300 samples) | model-agnostic; slow, so used offline for selected minutes |
| Anomaly detector (live) | top robust-z deviations | instant, model-agnostic description of *which* measurements are unusual |

### Global importance (XGBoost, validation sample: 1,500 failure + 3,000 normal minutes)
| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | DV_pressure (current minute) | 3.59 |
| 2 | DV_pressure_mean_30min | 1.41 |
| 3 | DV_pressure_mean_5min | 1.26 |
| 4 | DV_pressure_max_30min | 1.25 |
| 5 | TP2_mean_5min | 1.01 |
| 6 | DV_pressure_max_2h | 0.88 |
| 7 | TP3_mean_5min | 0.68 |
| 8 | H1_min_30min | 0.62 |

Figures: `xgb_summary_beeswarm.png` (each dot is one minute; colour = feature value; position = contribution),
`xgb_global_bar.png`. **Reading:** the supervised model is essentially a "DV_pressure ≈ 2 bar" detector, plus TP2/TP3/H1.

### Local explanations (`xgb_waterfall_*.png`, `anomaly_waterfall_*.png`)
| Case | Model output | Main contributions |
|---|---|---|
| Event 3, 6 Jun 09:52 (high risk) | P = 0.99998 | DV_pressure 2.13 bar → +2.40; DV 30-min mean → +2.03; DV 30-min max → +1.71 |
| Normal minute, 22 May 10:40 | P = 0.00003 | DV_pressure −0.02 bar → **−4.14**; DV means → −1.1, −0.9; TP3 9.24 bar → −0.88 |
| **Event 4, 15 Jul 18:31 (missed)** | P = 0.10 | TP3 5-min mean **6.54 bar → +2.53** (low pressure points toward failure) but DV_pressure ≈ 0 → −1.67, −1.39, −1.36, −1.11 |
| Event 4, anomaly model (OCSVM) | score above threshold | LPS active 100% → +0.51; TP3 30-min mean 6.87 bar → +0.50; TP3 vs 24 h baseline −2.1 bar → +0.36; H1 ≈ 0 → +0.29 |
| Event 4 mid-window, anomaly model | at threshold | TP2 8.6 bar → +0.52; H1 ≈ 0 → +0.34; oil 84.9 °C → +0.24 |

**This explains the event 4 miss:** the supervised model did see the low pressure, but it had learned that an air leak *also*
means DV_pressure ≈ 2 bar (true for events 1–3). Event 4's leak showed up as falling pressure, an open low-pressure switch and hot oil,
which the anomaly model recognises as abnormal. SHAP turned a "the model failed" result into an understandable reason.
It's still an explanation of the model, not a proven root cause of the failure.

---

## Part C — Equipment Health & Failure Risk Engine
**Code:** `src/predictops/engine/health.py`, `src/predictops/inference/predictor.py` (the single inference path for API, dashboard and replay).
**Config:** `configs/config.yaml → health_engine`. **Evaluation:** `scripts/evaluate_health.py` → `reports/health/`, `reports/figures/health/health_event4.png`.

### Evidence combined each minute
| Code | Evidence | Source | Component in [0, 1] |
|---|---|---|---|
| A | anomaly | One-Class SVM: severity (normal 0 / low 0.5 / medium 0.8 / high 1), or 1 if a persistent alert | weight **0.40** |
| D | failure signature | XGBoost P / its validation threshold (capped at 1) | weight **0.30** |
| S | sensor checks | 7 key features outside the 0.5–99.5% range of training-normal data (+ LPS active); 4 or more checks = full penalty | weight **0.20** |
| H | history | anomaly alerts started in the past 24 h ÷ 3 | weight **0.10** |

`health_score = 100 × (1 − Σ weight × component)`: **a heuristic index, not a failure probability**.

### Outputs and rules
| Output | Values | Rule |
|---|---|---|
| `anomaly_status` | Normal / Anomalous | Anomalous = persistent anomaly alert (≥ 10 of the last 30 minutes flagged) |
| `health_status` | Healthy / Warning / Critical | score ≥ 75 → Healthy, 45–75 → Warning, < 45 → Critical; any alert (A or D) → at least Warning; A **and** D → Critical |
| `failure_risk` | Low / Medium / High | High: D over threshold, or persistent alert with medium+ severity. Medium: any alert, unusual reading, D ≥ 50% of threshold, or any sensor check. Otherwise Low |
| `reasons` | text list | which evidence fired, with values and units |
| `recommendations` | text list | mapped from reasons (leak inspection, pressure check, LPS, oil and cooling, motor, dryer/DV, inspection scheduling, or "Continue normal monitoring") |

Weights reflect the evidence from earlier stages: the anomaly model generalised best (0.40), the signature model is very precise but
narrow (0.30), sensor ranges are simple and transparent (0.20), and history adds persistence (0.10). The cut-offs were then
**checked on validation** (below) before the single test evaluation.

### Validation of the thresholds (minutes, usable data)
| Split | Group | Healthy | Warning | Critical | Mean score | Risk: High |
|---|---|---|---|---|---|---|
| validation | normal (38,054) | **99.3%** | 0.6% | 0.1% | 98.8 | 0.3% |
| validation | failure (3,253) | 0.0% | 1.8% | **98.3%** | 15.5 | **100%** |
| test | normal (72,381) | **99.3%** | 0.6% | 0.2% | 99.0 | 0.2% |
| test | failure, event 4 (270) | 31.9% | 6.7% | **61.5%** | 50.9 | 29.3% |

Event level ("Critical" used as the alarm): **validation 2/2 events, 0.07 false alarms/day; test 1/1, 0.10 false alarms/day**.
With "Warning or worse" as the alarm: 2/2 at 0.24/day and 1/1 at 0.20/day.

**Example output** (test, 15 Jul 18:31, inside event 4): score **27.5 → Critical, risk High**.
Reasons: persistent anomaly (high severity); TP3 30-min mean 6.87 bar below normal; oil 80.4 °C above normal; TP3 −2.14 bar vs 24 h baseline;
LPS active; 2 alerts in the past 24 h. Recommendations: schedule an inspection; LPS active, so inspect for major leaks or valve faults;
check panel/reservoir pressure; check oil level and cooling; plan preventive maintenance.

### Limitations
- Weights are expert-set and validated on 2 events, not learned. They're a transparent heuristic.
- Early in event 4, before the persistent rule triggers, 32% of its minutes still score Healthy. Detection, not prediction.
- Sensor ranges come from February–May; summer drift may shift them (Stage 18 monitoring).
