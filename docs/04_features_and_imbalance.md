# 04 — Feature Engineering, Feature Selection & Class Imbalance (Stage 4c)

**Reproduce:** `.venv/bin/python scripts/feature_study.py` (~1 min).
**Code:** `src/predictops/features/build.py` (features, used by training *and* the API),
`src/predictops/evaluation/metrics.py` (shared metrics), `scripts/feature_study.py` (experiments).
**Outputs:** `data/processed/features_1min.parquet`, `models/metadata/selected_features.json`,
`reports/feature_study/*.csv`, `reports/figures/features/*.png`. **No test data is used in this stage.**

> **Summary**
> - 132 past-only features at 1-minute resolution, reduced to **33 selected features**.
> - **Detecting a failure in progress** (`y_detect`) is near-perfect on validation (PR-AUC ≈ 0.99–0.999),
>   whatever the features or imbalance method. Engineered features lift XGBoost from 0.982 to 0.999.
> - **Early warning 6 h ahead** (`y_warn_6h`, trained on event 2, validated on event 3) shows
>   **no usable signal**: PR-AUC 0.04–0.07 against a random baseline of 0.058, and ROC-AUC ≈ 0.5–0.65.
> - **SMOTE / ADASYN / resampling give no real benefit** for either target. Decision: no synthetic
>   resampling; use class weighting where helpful plus a validation-chosen decision threshold.

---

## 1. From 10-second rows to 1-minute rows
`aggregate_minutes()` groups the 10 s readings into 1-minute intervals **(t − 1 min, t]**, labelled by
the interval **end** t, within each recording segment. A feature stamped at t therefore uses
only readings at or before t.

- **Why:** 252,719 rows instead of 1.5 M make SVM, LOF and SHAP feasible on a laptop. One minute is still
  much finer than the shortest horizon (2 h).
- **Kept per minute:** the mean of each analog sensor, the share of time each digital signal was 1, the min/max
  of TP3, motor current and oil temperature, the number of load starts, oil temperature *while loaded*, and TP3 − Reservoirs.
- **Labels per minute:** the label at time t (last reading). A minute is `usable` only if *all* its
  readings are usable.

## 2. Engineered features (132 candidates)
Windows are **5 min, 30 min, 2 h**, chosen from the real ~10 s sampling and EDA: 5 min ≈ one
load/unload cycle, 30 min ≈ the window where the failure signature becomes sustained (docs/02 §6),
2 h ≈ the shortest warning horizon. Minimum coverage is 50% of the window, otherwise NaN (imputed later).

| Group | Features | Why (physical or statistical meaning) |
|---|---|---|
| Statistical rolling | mean, std, min, max, range, median (2 h), skew and kurtosis (2 h) of 6 analog sensors | level, variability and shape of each signal |
| Compressor behaviour | `duty_cycle_{30min,2h}`, `load_cycles_per_h_{30min,2h}`, `minutes_continuously_loaded` | an air leak makes the compressor run continuously and stop cycling (docs/02 §5) |
| Regime-aware | `oil_temp_loaded_{30min,2h}` | oil temperature compared only within the loaded state (docs/02 §10) |
| Persistence | `minutes_h1_near_zero` | consecutive abnormal readings (no filter discharge) |
| Lags / rate of change | `{TP3, Oil_temperature, Motor_current}_diff_{5,30}min`, `_pctchg_` | trends and sudden changes |
| Baseline deviation | `oil_temp_vs_24h_baseline`, `tp3_vs_24h_baseline`, `motor_current_vs_24h_baseline` | removes the seasonal oil-temperature drift (docs/02 §9) |
| Cross-sensor | `tp3_minus_res` (TP3 − Reservoirs) | the two should be equal; a gap would mean a pneumatic fault |
| Context | `minutes_in_segment` | warm-up after the train starts |

**Leakage safety (tests/test_features.py):**
- `test_features_are_past_only`: features are rebuilt on data **cut at time T** and compared with
  features built on the full data. All values at ≤ T are identical, so no feature can depend on the future.
- `test_rolling_never_crosses_segments`: lags and run-lengths reset at every recording gap.
- `test_no_label_or_id_in_features`: labels, `minutes_to_onset`, timestamps and IDs are never inputs.

## 3. Feature selection (training split only)
| Step | Method | Features left |
|---|---|---|
| 0 | all engineered | 132 |
| 1 | remove (near-)constant: variance < 1e-6 | 132 |
| 2 | remove redundant: greedy, drop a feature with abs(r) > 0.98 to one already kept | 106 |
| 3 | rank by XGBoost **gain** on the train split (y_detect), keep top 30 + each analog sensor's 30-min mean (for interpretability) | **33** |

The ranking is dominated by **`DV_pressure_mean_5min`** (gain 8,910 vs 120 for the next feature). During
failures DV_pressure stays at ~2 bar instead of ~0 (docs/02 §5). Then come TP3 level/shape (`TP3_mean_30min`,
`TP3_kurt_2h`, `tp3_vs_24h_baseline`), TP2 and oil temperature. This matches the physics of an air leak.
Caution: a model that relies on one feature can break if that sensor behaves differently, and event 4
**does** behave differently (docs/02 §5). The test result in Stage 5 will show whether that matters.

Figure: `reports/figures/features/xgb_gain_top30.png`.

## 4. Does feature engineering help? (validation PR-AUC, class-weighted models)
Experimental setup:
- **y_detect:** train split → validation split (event 1 → events 2, 3); 124,955 train minutes (1,416 positive), 41,307 validation (3,253 positive).
- **y_warn_6h:** forward fold B: train on everything before the event-3 block (includes event 2's onset: **360 positive minutes**), validate on event 3 ± 3 days (6,233 minutes, 360 positive; random-guess PR-AUC = 0.058).

| Feature set | n | LR y_detect | XGB y_detect | LR y_warn_6h | XGB y_warn_6h |
|---|---|---|---|---|---|
| raw sensor 1-min means (no FE) | 13 | 0.989 | 0.982 | 0.056 | 0.042 |
| all engineered | 132 | 0.990 | **0.999** | 0.074 | 0.057 |
| after correlation filter | 106 | 0.988 | **0.999** | 0.067 | 0.052 |
| **selected** | 33 | 0.973 | **0.999** | 0.070 | 0.059 |

Figure: `reports/figures/features/fe_comparison_prauc.png`. Full metrics: `fe_comparison.csv`.

**Interpretation:**
- For detection, feature engineering mainly helps XGBoost, and the 33-feature set loses nothing
  compared with 132. It is selected because it's faster, and SHAP explanations are readable.
- XGBoost with raw features has high PR-AUC but recall of only 0.35 at the 0.5 threshold: ranking is good but
  probabilities are poorly calibrated. **Thresholds will be tuned on validation** (Stage 5).
- For early warning, no feature set gets meaningfully above chance. With 1 training onset and 1
  validation onset, the minutes before event 3 don't look like the minutes before event 2 in any learnable
  way. This agrees with the EDA (no gradual pattern before events 1–3).

## 5. Class-imbalance study (33 selected features)
### 5.1 Is the target imbalanced?
| Target | Train positives | Ratio |
|---|---|---|
| y_detect (train split) | 1,416 / 124,955 minutes (1.1%) | ≈ 1 : 87 |
| y_warn_6h (fold B) | 360 / 134,558 minutes (0.27%) | ≈ 1 : 373 |

Both are imbalanced. **The real problem is not the number of minority rows but the number of independent
events**: all 1,416 positive minutes come from **one** failure (event 1), and the 360 warning minutes from **one** onset.

### 5.2 Is SMOTE/ADASYN appropriate here? (checked before applying)
- SMOTE creates new rows by interpolating between a minority row and its nearest minority neighbours.
  Our minority rows are consecutive minutes of the *same* event and are highly autocorrelated, so synthetic
  rows are near-copies of existing minutes. They add **no new failure information**.
- Interpolating rolling-window features can produce combinations that can't occur physically
  (e.g. a 5-min mean inconsistent with the 30-min mean of the same instant).
- For sequences (LSTM/CNN), row-wise SMOTE would break temporal order, so it isn't used there (Stage 6).
- It was **still tested** on the tabular features, as required, **inside the training fold only** (imblearn
  `Pipeline`: preprocess → resample → model, so validation rows are never resampled).

### 5.3 Results (validation)
| Method | LR y_detect PR-AUC | XGB y_detect PR-AUC | XGB y_detect F1@0.5 | LR y_warn_6h PR-AUC | LR y_warn_6h recall / FPR |
|---|---|---|---|---|---|
| none | 0.999 | 0.999 | 0.976 | 0.069 | 0.00 / 0.002 |
| class weight | 0.973 | 0.999 | 0.956 | 0.070 | 0.50 / 0.365 |
| random oversampling | 0.973 | 0.998 | **0.987** | 0.070 | 0.49 / 0.363 |
| random undersampling | 0.997 | 0.998 | 0.987 | 0.063 | 0.49 / 0.411 |
| SMOTE | 0.991 | 0.998 | 0.976 | 0.070 | 0.47 / 0.357 |
| ADASYN | 0.999 | 0.999 | 0.957 | 0.073 | 0.31 / 0.287 |

Figure: `reports/figures/features/imbalance_prauc.png`. Full table: `imbalance_comparison.csv`.

**Interpretation:**
- **y_detect:** PR-AUC is between 0.97 and 0.999 for every method. Resampling changes the *threshold-0.5*
  F1 (it shifts predicted probabilities) but not the model's ability to rank failures above normal.
  Tuning the threshold on validation gets the same effect without synthetic data.
- **y_warn_6h:** class weighting or oversampling raises recall to ~0.5, but only by flagging **36% of
  normal minutes** (FPR 0.365). That's equivalent to guessing, and PR-AUC stays at chance.
- **Decision:** **no SMOTE/ADASYN in the final pipeline**. They add complexity and synthetic data without
  measurable benefit. We use class weighting for linear/kernel models, `scale_pos_weight` for XGBoost, and
  a validation-tuned threshold. Imbalance is judged by PR-AUC, recall, MCC and event-level metrics, never accuracy.

## 6. Consequences for Stage 5 (machine learning)
1. **Main supervised task = failure-in-progress detection (`y_detect`)**, with 5 algorithms compared,
   hyperparameters tuned on forward-chaining folds, threshold chosen on validation, and a single final test on event 4.
2. **Early warning (`y_warn_{2,6,12}h`)** will be reported transparently as a negative result
   (trained on event 2, validated on 3, tested on 4). This is a legitimate scientific finding, not a failure of the project.
3. **Anomaly detection (Stage 6)** becomes the main way to catch *unusual behaviour before or outside*
   reported failures. Event 4's gradual oil-temperature rise (docs/02 §5) is the best case for showing that.

## 7. Limitations
- Selection, imbalance and FE conclusions rest on 1 training event (detection) or 1 training onset (warning).
- XGBoost depends heavily on `DV_pressure`. Robustness to an event with a different pattern is tested in Stage 5.
- The F1 values at a fixed 0.5 threshold are for comparison only. Real thresholds are tuned in Stage 5.
