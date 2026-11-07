# 05 — Supervised Machine Learning (Stage 5)

**Reproduce:** `.venv/bin/python scripts/train_ml.py` (~12 min on an 8-core M-series Mac; RF tuning takes most of the time).
**Code:** `src/predictops/models/zoo.py` (models and search spaces), `scripts/train_ml.py` (protocol).
**Outputs:** `models/trained/detect_*.joblib`, `models/metadata/ml_*.json`, `reports/ml/*.csv`,
`reports/figures/ml/*.png`, `data/processed/test_predictions.parquet`, MLflow runs in `mlruns.db`.

> **Headline results (honest)**
> - On **validation** (events 2, 3), all five models detect an ongoing failure almost perfectly:
>   out-of-fold PR-AUC 0.988–0.999, F1 ≈ 0.99. XGBoost was **selected before the test was opened**.
> - On the **test** (event 4, never seen), **every model misses the event at its validation-chosen
>   threshold** (recall 0). XGBoost and RF still *rank* failure minutes higher (ROC-AUC 0.985 / 0.988,
>   PR-AUC 0.48 / 0.16 against a base rate of 0.004), but their scores peak around 0.1, far below the thresholds.
> - **Why:** events 1–3 share an abrupt signature (DV_pressure ≈ 2 bar, H1 ≈ 0). Event 4 is a slow
>   pressure decay with rising oil temperature (docs/02 §5). The models learned the *signature of three
>   similar events*, not "air leak" in general. With 3 training events this is the expected limitation.
> - **Early warning** (2/6/12 h) on event 4 is **at or below chance** for all horizons.
> - We do **not** re-tune on the test set to "fix" this. The next stage (anomaly detection) is designed
>   for exactly this case: behaviour that differs from normal *and* from previously seen failures.

---

## 1. Algorithms: what they are and why they're included
| Model | How it works (one-paragraph intuition) | Why it's here |
|---|---|---|
| **Logistic Regression** | Weighted sum of the features passed through a sigmoid to give a probability. Learns one weight per feature. | Simple, fast, interpretable **baseline**: any complex model must beat it. |
| **Decision Tree** | Asks a sequence of yes/no questions ("DV_pressure_mean_5min > 0.9?"), each split chosen to make the groups purer (Gini). | Human-readable rules; shows how "easy" the split is. Prone to overfitting, so depth is limited. |
| **Random Forest** | Many trees, each trained on a bootstrap sample and random feature subsets; votes are averaged. | Averaging reduces a single tree's variance; robust to noisy features. |
| **SVM (RBF kernel)** | Finds the boundary with the **maximum margin** between classes; the RBF kernel allows curved boundaries. | A different, geometry-based idea of non-linearity. |
| **XGBoost** | Trees are built **sequentially**, each correcting the errors of the previous ones (gradient boosting), with regularisation. | Usually the strongest on tabular data; fast; exact SHAP explanations. |

All models run inside `Pipeline(imputer+scaler → model)`, so preprocessing is re-fitted on each training fold.
Imbalance is handled with `class_weight="balanced"` / `scale_pos_weight`, without SMOTE (docs/04 §5).

## 2. Protocol (how leakage and test-tuning are avoided)
1. **Tuning:** random search (`ParameterSampler`, 5–15 configs per model) scored by mean PR-AUC over the
   **forward-chaining folds** A (train before event 2 → validate on event 2) and B (train before event 3 → validate on event 3).
   Search spaces: `zoo.py::SEARCH_SPACES`. Full log: `reports/ml/tuning_results.csv`.
2. **Threshold:** maximum F1 on the best config's **out-of-fold validation predictions** (never test).
3. **Selection:** highest out-of-fold PR-AUC (tie-break MCC). It's written to `models/metadata/ml_selection.json`
   with `decided_before_test: true` **before** any test prediction is made.
4. **Final fit:** all usable development rows (train + validation, events 1–3; 166k minutes).
5. **Test:** each final model is evaluated **once** on the test split (event 4; 72.6k usable minutes ≈ 50 days).
6. **SVM cost:** RBF-SVM training time grows roughly with rows², so it's trained on a stratified subsample of 20,000 rows
   (all positives plus random negatives). That's practical on a laptop, and it's documented here.

## 3. Validation results (out-of-fold, events 2 + 3), used for selection
| Model | Best hyperparameters | OOF PR-AUC | MCC | F1 | Recall | Precision | Threshold |
|---|---|---|---|---|---|---|---|
| **XGBoost (selected)** | depth 2, 600 trees, lr 0.03, subsample 0.7, min_child_weight 20 | **0.9992** | 0.989 | 0.991 | 0.987 | 0.995 | 0.516 |
| Logistic Regression | C = 0.001 | 0.9992 | 0.990 | 0.992 | 0.989 | 0.995 | 0.773 |
| SVM (RBF) | C = 10, gamma = 0.01 | 0.9987 | 0.988 | 0.991 | 0.989 | 0.993 | 0.006 |
| Random Forest | 100 trees, unlimited depth, min_leaf 50, sqrt features | 0.9939 | 0.989 | 0.991 | 0.987 | 0.995 | 0.374 |
| Decision Tree | depth 4, min_leaf 500 | 0.9880 | 0.980 | 0.984 | 0.987 | 0.981 | 0.971 |

XGBoost and LR tie on PR-AUC to 4 decimals; XGBoost wins on the unrounded value. Note that a
**depth-4 decision tree** is almost as good. That says the validation task is easy.
Calibration plot: `reports/figures/ml/calibration_oof.png`.

## 4. Test results (event 4, evaluated once)
| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 | MCC | Brier | False alarms/day | Train time | Inference |
|---|---|---|---|---|---|---|---|---|---|---|
| **XGBoost (selected)** | **0.480** | 0.985 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0035 | 0.00 | 3.5 s | 0.004 ms/row |
| Random Forest | 0.163 | **0.988** | 0.00 | 0.00 | 0.00 | 0.00 | 0.0035 | 0.00 | 255 s | 0.001 ms/row |
| Decision Tree | 0.046 | 0.686 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0039 | 0.00 | 1.5 s | |
| Logistic Regression | 0.040 | 0.581 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0038 | 0.00 | 0.7 s | |
| SVM (RBF) | 0.003 | 0.200 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0037 | 0.02 | 0.6 s | 0.005 ms/row |

Test set: 72,651 usable minutes, **270 failure minutes** (event 4); base rate (random PR-AUC) = **0.0037**. Full table: `reports/ml/test_results_detect.csv`.
Figures: `test_comparison.png`, `test_confusion_matrices.png`, `test_roc_pr.png`, `test_event4_scores.png`.

**How to read this:**
- **Accuracy would be about 99.6% for every model (270 of 72,651 minutes are failures)**, simply by predicting "normal" everywhere. That's why accuracy is never used to choose a model.
- **ROC-AUC 0.985 with recall 0** isn't a contradiction. ROC-AUC measures *ranking* (failure minutes get higher
  scores than normal ones), while recall depends on the *threshold*. In `test_event4_scores.png`, XGBoost and RF
  rise from ~0 to ~0.1 in the second half of event 4, clearly above their normal level but below the 0.37–0.52 thresholds
  learned on events 2–3.
- **Zero false alarms** over 50 test days is good, but here it comes with zero detections. **False alarms vs missed failures:**
  a lower threshold would catch event 4 but generate more false alarms. Choosing it *now* would mean tuning on the test set,
  so we don't. The trade-off is shown in the precision–recall curve and exposed as a configurable threshold in the app.
- Linear and kernel models (LR, SVM) generalise worst: they lean almost entirely on the DV_pressure/H1 pattern.

## 5. Early-warning targets (trained on onsets 2 and 3, tested on onset 4)
| Horizon | Model | Test positives | Random PR-AUC | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| 2 h | LR / XGB | 120 | 0.0017 | 0.0008 / 0.0017 | 0.02 / 0.53 |
| 6 h | LR / XGB | 361 | 0.0050 | 0.0027 / 0.0028 | 0.16 / 0.19 |
| 12 h | LR / XGB | 720 | 0.0099 | 0.0053 / 0.0067 | 0.10 / 0.32 |

PR-AUC is at or below chance, and ROC-AUC is often **below 0.5**. The hours before event 4 look *less* like a failure
than ordinary operation does to these models. **Conclusion: with this dataset, supervised early warning is not
supported. PredictOps therefore does not claim to predict failures hours in advance.**

## 6. What goes into the application
- The **XGBoost detector** (selected on validation) is used as a "failure-in-progress" **risk score**, with its known limitation:
  it recognises the events-1–3 type of air leak, and can miss differently shaped failures.
- Its score is **combined with the anomaly detector** (Stage 6) in the health engine (Stage 6/11), so a failure that
  doesn't match the learned signature can still raise a warning.
- Each model's artifact, metadata (version, features, thresholds, train period, dataset fingerprint, test metrics)
  and an MLflow run are saved, so every prediction can be traced to a model version.

## 7. Viva-ready talking points
- *"Why not accuracy?"* At a 0.37% failure rate, "always normal" scores 99.6% accuracy and catches nothing.
- *"Why did validation look perfect but test fail?"* Validation events 2–3 resemble training event 1; test event 4 doesn't.
  With 4 events, generalisation to new failure types can't be guaranteed. This is a data limitation, demonstrated honestly.
- *"Why not just lower the threshold?"* That decision would use test labels. It would turn the test into a second validation set and make the result meaningless.

## 8. Limitations
- 3 training events of one failure type (air leak) on one compressor.
- Thresholds transfer poorly between events, because probabilities aren't calibrated across failure types.
- The SVM was trained on a subsample (cost).

_Brier scores (~0.0035) look excellent only because almost every minute is normal and predicted near 0. With rare events, Brier score is dominated by the majority class and is not a useful selection metric._
