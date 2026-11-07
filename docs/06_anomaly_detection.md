# 06 — Anomaly Detection (Stage 6a)

**Reproduce:** `.venv/bin/python scripts/train_anomaly.py` (~10 s).
**Code:** `src/predictops/anomaly/detectors.py`. **Outputs:** `models/trained/anomaly_*.joblib`,
`models/metadata/anomaly.json`, `reports/anomaly/anomaly_results.csv`, `reports/figures/anomaly/*.png`.

> **Headline:** a **One-Class SVM**, trained only on normal data and selected on validation, **detects all three
> evaluated failures, including test event 4 that every supervised model missed**, with ~0.15 false-alarm episodes per day.
> Its score also **starts rising ~14 h before event 4**, though it stays below the alert threshold until the event begins.
> This is the main detection engine of PredictOps.

---

## 1. Why anomaly detection (and how it differs from failure prediction)
- **Supervised models** learn "what past failures looked like", so they miss failures that look different (docs/05 §4).
- **Anomaly detectors** learn "what *normal* looks like" and flag deviations of any kind. They need no failure labels.
- **Three different things the app keeps separate:**
  | Term | Meaning in PredictOps |
  |---|---|
  | *Unusual measurement* | one minute with a high anomaly score (point anomaly) |
  | *Abnormal behaviour* | a **persistent** anomaly: ≥ 10 flagged minutes within 30 min, which becomes an **alert** |
  | *Confirmed failure* | only the 4 company-reported events |
  An alert is **never** presented as a confirmed failure.

## 2. Setup (no failure labels used for training, features or thresholds)
| Item | Choice | Reason |
|---|---|---|
| Training data | train split, usable, outside failure windows, **1 Feb → 24 Apr** | known-normal behaviour (frozen and uncertain rows excluded) |
| Calibration data | same conditions, **25 Apr → 20 May** | thresholds from data the model did *not* fit (avoids optimistic in-sample scores) |
| Features | 18 **label-free** features (`ANOMALY_FEATURES`): 30-min means and stds of the sensors, duty cycle, load cycles per hour, oil temperature while loaded, 24 h baseline deviations, LPS share | chosen by physics, not by the supervised importance ranking (which used labels) |
| Thresholds | quantiles of calibration scores: low 99.5%, medium 99.9%, high 99.99% | by design about 0.5% of normal minutes are "unusual"; severity grows with rarity |
| Alerting | persistent: ≥ 10 flagged minutes in the trailing 30 min; merge within 30 min; min 5 min | turns single odd minutes (point anomalies) into a few meaningful alerts |
| Event match | an alert overlapping [start − 12 h, end] detects the event; any other alert is a false alarm | event-level evaluation |

### Detectors
| Detector | Idea | Training rows |
|---|---|---|
| Isolation Forest | random splits isolate anomalies in fewer steps | all fit rows (98,261 minutes) |
| Local Outlier Factor (novelty) | a point is anomalous if its local density is much lower than its neighbours' | 30k sample |
| One-Class SVM (RBF, ν = 0.01) | learns the smallest region in kernel space containing ~99% of normal data | 10k sample (cost) |
| Robust z-score | statistical: max over features of the absolute (x − median)/MAD | all fit rows |

## 3. Results
**Selection was done on validation (events 2, 3) only:** most events detected, then fewest false alarms per day, then PR-AUC.

| Detector | Set | Events detected | Alert timing vs reported start | False alarms/day | Minute recall | Minute precision | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| **One-Class SVM** ✔ | validation | **2/2** | +16, +20 min after | 0.14 | 0.99 | 0.94 | 0.87 | 0.991 |
| **One-Class SVM** ✔ | **test** | **1/1 (event 4)** | +34 min after | **0.16** (8 alerts in ~50 days) | 0.61 | 0.29 | 0.37 | 0.996 |
| Isolation Forest | validation | 2/2 | +9, +16 min | 0.28 | 0.03 | 0.32 | 0.76 | 0.983 |
| Isolation Forest | test | 1/1 | +29 min | 0.26 | 0.45 | 0.23 | 0.24 | 0.995 |
| LOF | validation / test | 0/2, 0/1 | – | 0 | 0 | – | 0.85 / 0.15 | 0.99 |
| Robust z | validation / test | 2/2, 0/1 | ~0 min | 0.14 / 0 | 0.01 / 0 | – | 0.43 / 0.01 | 0.94 / 0.71 |

("Lead time" in the CSV is negative when the alert came *after* the reported start.)
Figures: `scores_event2.png`, `scores_event3.png`, `scores_event4.png`, `test_period_timeline.png`.

**Interpretation:**
- **One-Class SVM** is best on validation and holds up on the unseen, *differently shaped* event 4.
  Detection comes 16–34 min after the reported start. Since the reported boundaries are approximate (docs/02 §5),
  this is **early detection of an ongoing failure, not prediction**.
- **Precursor on event 4** (`scores_event4.png`): the OCSVM score climbs steadily from about 00:00 on 15 July, roughly 14 h before the
  reported start, matching the gradual oil-temperature rise seen in EDA. It crosses the alert threshold only at the event.
  This is evidence that anomaly *trends* carry early information. With one event it is a case study, not a validated early warning.
  The dashboard shows the score trend so an operator can see it.
- **False alarms:** 8 alert episodes in ~50 test days (0.16/day, roughly one a week). Some may be genuine
  unusual behaviour (e.g. the alert early on 14 July). We can't know without maintenance records.
- **Unreported episodes:** OCSVM flags **98.8%** of the minutes in the 3 "uncertain" episodes (12 Apr, 13 May, 19–20 May),
  so to a normal-behaviour model they look like the reported failures. This supports, but doesn't prove, that they were unreported problems.

### Why LOF and robust-z did not work (documented, not tuned on events)
- **LOF:** many minutes when the compressor is off are *identical*, so neighbour distances are 0 and the local
  density becomes extreme. LOF scores reach ~10⁷ and the quantile thresholds become meaningless. A fix (de-duplication or
  adding jitter) is possible future work. Tuning it now with event knowledge would bias the comparison.
- **Robust z:** per-feature z-scores relative to February–April are inflated by the **seasonal drift** (e.g. oil temperature)
  and by near-constant features. Single-feature statistical thresholds are fragile; multivariate models handle it better.

## 4. Anomaly types covered
| Type | How PredictOps captures it |
|---|---|
| Point | single minute above the 'low' threshold (shown as an "unusual measurement", no alert) |
| Contextual | regime-aware features (oil temperature *while loaded*, deviation from the 24 h baseline) |
| Collective | 30-min / 2 h window features (loss of normal on/off cycling) |
| Persistent | alert rule ≥ 10 flagged minutes in 30 min |
| Multivariate | IF / OCSVM operate on all 18 features jointly |

## 5. Explanation of an anomaly
`AnomalyDetector.top_deviating_features()` returns, for each minute, the 3 features furthest from their normal
median (in robust-z units). Example output: *"oil_temp_vs_24h_baseline +6.1, duty_cycle_30min +4.3, TP3_mean_30min −3.2"*.
The app turns this into text such as "oil running hotter than its recent baseline; compressor loaded much more than usual".
It describes **which measurements are unusual**, not *why* the machine failed.

## 6. Severity
`severity()` maps the score to normal / low / medium / high using the 99.5 / 99.9 / 99.99% calibration quantiles.
Severity is a measure of **rarity relative to normal operation**, not a failure probability.

## 7. Limitations
- Thresholds assume the calibration period (late April to May) is representative of normal operation. Seasonal drift can shift scores (monitoring in Stage 18).
- One-Class SVM is trained on a 10k-minute sample for cost reasons.
- Detection, not prediction: alerts arrive after the reported onset (16–34 min).
