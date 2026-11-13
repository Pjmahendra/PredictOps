# PredictOps — Viva Preparation Guide (Simple English)

Read this together as a group. All numbers here come from our own project results.
If you get a question that is not in this guide, remember three things:
**we did not let future data leak in, we tested honestly, and we had very few failures to learn from.**

---

## 0. The 1-minute summary (all 4 of you must be able to say this)

> "Our project is called PredictOps. It watches the air compressor of a metro train in Porto, Portugal.
> We used a public dataset called MetroPT-3. It has 1.5 million sensor readings from February to September 2020.
> The data has **no labels**. The company only told us about **4 times the machine failed** (air leaks).
> We tried 5 normal ML models, 4 anomaly detection models and 2 deep learning models.
> The normal ML models looked perfect on validation (score 0.999), but they **missed the new failure in the test data**,
> because that failure looked different. SHAP showed us why.
> A **One-Class SVM**, trained only on normal data, **caught all 3 failures we tested**, with about 1 false alarm per week.
> We put this into a health score, an API, and a dashboard with replay, alerts, reports and a live simulation."

**Main lesson in one line:** when you have very few failures, it is better to learn what *normal* looks like and test honestly,
than to try to learn what failure looks like.

---

## 1. The Dataset (most questions will come from here)

### 1.1 Basic facts
| Item | Value |
|---|---|
| Where it comes from | UCI Machine Learning Repository, dataset number 791. Paper: Veloso et al., 2022 |
| Files | `MetroPT3(AirCompressor).csv` and a PDF that describes the data |
| Size | **1,516,948 rows and 17 columns** (row number + time + 15 sensors), about 210 MB |
| Time period | 1 Feb 2020 to 1 Sep 2020 (7 months) |
| Machine | One air compressor (called APU, Air Production Unit) on one metro train in Porto |
| How often readings come | **About every 10 seconds.** The PDF says every 1 second, but the file we got keeps only 1 out of every 10 readings |
| Missing values | None |
| Repeated timestamps | None |
| Labels | **None.** The PDF itself says "the dataset is unlabeled" |

### 1.2 What the machine does (basic idea)
- The compressor makes compressed air. The train uses this air for **brakes, suspension and doors**.
- The air is dried by **two drying towers**. One tower dries the air while the other one cleans itself.
- Normal working: the compressor **pumps air (loaded)** until pressure reaches about 10 bar, then **rests (offloaded)**.
  When pressure drops to about 8.2 bar, it starts pumping again. This on/off cycle repeats.
- **Air leak:** air keeps escaping, so the compressor has to run almost all the time.
  Because of this, **oil gets hot, pressure drops**, and the normal on/off cycle breaks.

### 1.3 The 15 sensors (7 analog + 8 digital). Learn this table.
Analog = gives a number. Digital = gives only 0 or 1.

| Sensor | Type | What it measures |
|---|---|---|
| TP2 | analog (bar) | Pressure at the compressor |
| TP3 | analog (bar) | Pressure at the main air panel. Normally 8–10 bar. **Most important sensor** |
| H1 | analog (bar) | Pressure drop when the filter releases water |
| DV_pressure | analog (bar) | Pressure drop when the drying towers release air. **About 0 when the compressor is pumping** |
| Reservoirs | analog (bar) | Pressure in the air tanks. Should be close to TP3 |
| Oil_temperature | analog (°C) | Temperature of the compressor oil (15–89 °C in the data) |
| Motor_current | analog (A) | Motor current: about 0 A = off, about 4 A = resting, about 7 A = pumping, about 9 A = starting |
| COMP | digital | Air intake valve. It is 1 when **no** air is coming in (compressor off or resting) |
| DV_eletric | digital | Output valve. It is 1 when pumping (the spelling mistake is in the original data) |
| Towers | digital | Which drying tower is working (0 = tower 1, 1 = tower 2) |
| MPG | digital | Starts the compressor when pressure falls below 8.2 bar |
| LPS | digital | **Low-pressure switch.** Turns on when pressure falls below 7 bar |
| Pressure_switch | digital | Detects air release in the drying towers |
| Oil_level | digital | Turns on when oil is low |
| Caudal_impulses | digital | Counts air flow going to the tanks |

Strongly related sensors: TP2 with DV_eletric (+0.947) and TP2 with MPG (−0.941), because all three show whether the compressor is pumping or not.

### 1.4 The 4 reported failures (all are air leaks)
| Failure | When | Rows | Hours | Used for |
|---|---|---|---|---|
| 1 | 18 Apr 2020, 00:00 → 23:59 | 8,657 | 24.0 | training |
| 2 | 29 May 23:30 → 30 May 06:00 | 2,360 | 6.6 | validation |
| 3 | 5 Jun 10:00 → 7 Jun 14:30 | 17,315 | 48.1 | validation |
| 4 | 15 Jul 14:30 → 19:00 | 1,622 | 4.5 | **testing (never seen before)** |

- 29,954 rows are inside a failure window. That is **1.97%** of the data, about 1 failure row for every 50 rows.
- **The LPS alarm did not turn on at all during failures 1 and 2.** So we could not use it as a label.
- **The company's failure table has mistakes.** We wrote them down but did not change them:
  - two rows are both numbered "#1";
  - failure 2 says "maintenance on 30 April", which is *before* the failure happened (it probably belongs to failure 1).

### 1.5 Problems we found in the data
- **230 gaps** longer than 10 minutes where the logger was off (about 901 hours in total). These split the data into **231 pieces (segments)**.
- **7 frozen periods** (3.3% of rows): every analog value stays exactly the same for 1 to 51 hours. This is a logger fault, not the machine.
- **3 periods that look exactly like failures but were never reported:** 12 April, 13 May and 19–20 May. So the labels are **incomplete**.
- **Summer effect:** oil is hotter in summer. That is a change in the data, but it is **not** a failure.
- **Failure 4 looks different from the others:**
  - failure 4 showed low TP3, LPS turned on, and hot oil, with DV_pressure ≈ 0;
  - failures 1–3 showed DV_pressure ≈ 2 bar.
  - **This one difference explains all our results.**

### 1.6 Tricky dataset questions
- **"Is this real data?"** Yes. It is real data from the Porto metro, shared publicly. Only the *Live Simulation* page uses fake data, and the page clearly says so.
- **"Why didn't you use the 1-second data?"** The public file is already at 10 seconds. We wrote down this difference instead of hiding it.
- **"How many failures do you have?"** 4. Count *failures*, not rows. 30,000 failure rows are still only 4 real examples.
- **"Why not predict remaining useful life (RUL)?"** For RUL you need to watch machines slowly wear out until they break. We only have 4 short leaks that were repaired quickly, so there is no wear-out curve.
- **"Did you change any labels?"** No. We marked the 3 unreported look-alike periods as *uncertain* and left them out of training and scoring.

---

## 2. Preprocessing (cleaning the data)

| Step | What we did and why |
|---|---|
| Sort by time | Everything after this depends on correct time order |
| Check columns and values | The same checker is used when a user uploads a file, so bad files are rejected |
| Smaller number types | float32 and int8 use about 4 times less memory. We saved the data as Parquet (faster than CSV) |
| **Split at gaps** | A new segment starts after any gap longer than 10 minutes. We **never fill in values across a gap**, because that would be inventing data. While predicting, we only fill tiny gaps of 60 seconds or less |
| Mark frozen periods | Kept in the data, but not used for training or scoring |
| Mark uncertain periods | The 3 unreported look-alikes, handled the same way |
| Outliers | **Kept**, because an outlier is often the failure itself |
| Scaling | StandardScaler, **learned from training data only**, inside the model pipeline |

### Targets (what the model is asked)
- `y_detect`: "Is this minute inside a reported failure?" (detection)
- `y_warn_2h / 6h / 12h`: "Will a failure start in the next 2, 6 or 12 hours?" (early warning)

### How we split the data (by time, by failure)
| Part | Period | Failures inside |
|---|---|---|
| Training | Feb → 20 May | failure 1 |
| Validation | 21 May → 30 Jun | failures 2 and 3 |
| Test | Jul → Aug | failure 4, **opened only once, at the very end** |

- We left a **12-hour gap (embargo)** between the parts, so no window of data touches two parts.
- To pick the model, we used **forward-chaining**: fold A trains on data before failure 2 and tests on failure 2; fold B trains on data before failure 3 and tests on failure 3.

---

## 3. Feature Engineering (making useful inputs)
- We turned the 10-second readings into **1-minute rows**.
- From these we made **132 features**. Each one uses only past data, over **5-minute, 30-minute and 2-hour** windows:
  - average, spread (std), min, max, range, median, skew and kurtosis;
  - **duty cycle** (how much of the time the compressor was pumping) and **load cycles per hour** (a leak makes it pump nonstop);
  - **oil temperature while pumping**;
  - how many minutes in a row a reading stayed abnormal;
  - lags (earlier values) and how fast a value is changing;
  - **difference from the last 24 hours**, which removes the summer effect;
  - TP3 minus Reservoirs (a pressure difference that points to a leak).
- Why these window sizes:
  - 5 minutes ≈ one on/off cycle;
  - 30 minutes ≈ the time a leak pattern takes to become clear;
  - 2 hours = the shortest warning time we tested.
- **Choosing features (using training data only):** remove features that barely change → remove one of each pair that are almost the same (correlation > 0.98) → rank by XGBoost importance. This leaves **33 features**.
- The anomaly models use a separate set of **18 simple physical features**.
- **Proof that no feature uses the future:** we build features on data cut at time T, and again on the full data. Every value at or before T must be exactly the same. An automatic test checks this.

---

## 4. Class Imbalance and SMOTE (a common trap)
- In training, failure minutes are 1.1% of all minutes (about 1 in 87). For the 6-hour warning target it is 0.27% (about 1 in 373).
- The *real* problem is that there is **only one failure in training**.
- We tried SMOTE, ADASYN, random oversampling, random undersampling and class weights, **only inside the training folds**.
- Result: the scores were about the same (0.97–0.999) whatever we used. Early warning stayed at random level.
- **Final choice: NO SMOTE.** We used class weights and a threshold chosen on validation data.

**Q: "Your accuracy is 99.6%. Did you use SMOTE to get that?"** No. Accuracy is high only because 99.63% of test minutes are normal. A model that always says "normal" would also get about 99.6%. That is why we never use accuracy to choose models.

**Q: "What does SMOTE do?"** It takes a failure point, picks one of its nearby failure points, and makes a new fake point somewhere on the line between them.

**Q: "What about ADASYN?"** Same as SMOTE, but it makes more fake points near the failure points that are hardest to classify.

**Q: "Why is SMOTE bad for time-series data?"**
- Back-to-back minutes are almost copies of each other, so mixing them creates nothing new.
- It can create sensor combinations that are physically impossible.
- If done before splitting, fake copies of test-like points end up in training (leakage).

**Q: "Why not resample the test data?"** The test must show the real-world mix of normal and failure. Changing it would fake the results.

---

## 5. Data Leakage (they WILL ask this; remember all 7)
**Leakage** means the model sees, during training, information it would not have when making real predictions.
We stopped these 7 kinds:
1. a random train/test split on time data (we split by time instead);
2. a scaler or imputer learned on all the data (ours learn from training data only);
3. SMOTE or resampling before the split (ours happened only inside folds);
4. rolling windows that look into the future (ours look only backwards, and the test proves it);
5. using "minutes until failure" as a feature (we removed it);
6. tuning on the test set (the model and threshold were fixed before we opened the test);
7. leave-one-failure-out cross-validation, which trains on future data (we used forward-chaining instead).

**Q: "Why not random K-fold?"** Readings 10 seconds apart are almost the same. A random split puts near-copies of training rows in the test, so the score looks much better than it really is.

---

## 6. Supervised Models (models that learn from labels)
Each model is: fill missing values → scale → model. We tuned each one with random search on the forward-chaining folds, using PR-AUC as the score.

| Model | Simple idea | Validation PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|
| **XGBoost (chosen)** | Builds trees one after another; each new tree fixes the mistakes of the earlier ones | 0.9992 | 0.480 | 0.985 |
| Logistic Regression | A straight-line model that outputs a probability | 0.9992 | 0.040 | 0.581 |
| SVM (RBF) | Finds the widest gap between the classes. Trained on 20,000 rows because it is slow on big data | 0.9987 | 0.003 | 0.200 |
| Random Forest | Many trees on random parts of the data; takes the vote | 0.9939 | 0.163 | 0.988 |
| Decision Tree | One tree of yes/no questions; depth limited so it doesn't memorise | 0.9880 | 0.046 | 0.686 |

- In the test data, only 0.37% of minutes are failures (270 out of 72,651). A random guess would get PR-AUC ≈ 0.0037, so XGBoost's 0.48 is about 130 times better than random.
- **But at the threshold chosen on validation, every model caught 0 failure minutes in the test (recall = 0, so precision = 0).**

**Q: "ROC-AUC is 0.985 but recall is 0. How is that possible?"** ROC-AUC only checks *ordering*. Failure minutes did get higher scores than normal minutes. But all those scores were still *below the threshold* we had set from failures 1–3, so nothing was flagged.

**Q: "Why not just lower the threshold?"** That means using the test answers to tune the model. Then the test is no longer fair.

**Q: "Why did it miss failure 4?"** SHAP shows the model learned "failure = DV_pressure around 2 bar", which was true for failures 1–3. At 15 July 18:31:
- low TP3 (6.54 bar) pushed the prediction **toward** failure (+2.53);
- DV_pressure ≈ 0 pushed it **away** from failure (−1.67, −1.39, …);
- the final probability was only 0.10, below the threshold.

**Q: "So was validation useless?"** No. But failures 2 and 3 looked like failure 1, so validation made the model look better than it was. Keeping a truly new failure for testing is what showed this.

### Early warning (predicting 2, 6 or 12 hours ahead)
Test PR-AUC was 0.0008–0.0067, while random guessing gives 0.0017–0.0099. **No better than guessing.** So our app does **not** offer early warning, and it says so.

---

## 7. Anomaly Detection (our main result)
All these models learned **only from normal data** (February–April), using 18 physical features.

| Model | Simple idea | Validation failures caught | Test failures caught | False alarms per day | Test minute recall / precision |
|---|---|---|---|---|---|
| **One-Class SVM (chosen)** | Draws a boundary around normal data; anything outside is unusual. ν = 0.01 | 2/2 | **1/1** | **0.16** | 0.61 / 0.29 |
| Isolation Forest | Splits data randomly; unusual points get separated in fewer splits | 2/2 | 1/1 | 0.26 | 0.45 / 0.23 |
| LOF | Compares how crowded a point's area is with its neighbours' areas | 0/2 | 0/1 | 0 | 0 |
| Robust z-score | How far each value is from the normal median | 2/2 | 0/1 | 0 | 0 |

- **Setting thresholds without labels:** we took the 99.5%, 99.9% and 99.99% levels of the scores on a *separate normal period*. These give the low, medium and high severity levels.
- **Persistence rule:** an alert is raised only if **at least 10 of the last 30 minutes** are flagged. This stops one-off spikes.
- **When a failure counts as caught:** an alert overlaps the time from 12 hours before the failure until its end. Any other alert counts as a false alarm.
- Alerts came **16–34 minutes after the failure started**, so this is **detection, not prediction**.
- The One-Class SVM score **rose slowly for about 14 hours before failure 4** but did not cross the line. That is interesting, but it is only one example.
- It also flagged 98.8% of the 3 unreported look-alike periods. This suggests they were real problems.
- **Why LOF failed:** there are many identical "compressor off" minutes, which confuses its idea of how crowded an area is.
- **Why z-score failed on the test:** the summer temperature change.
- **0.16 false alarms per day ≈ 1 false alarm per week.**

**Q: "Why did anomaly detection work when supervised models failed?"** It does not need failure labels. It learns what normal looks like, so *anything* different, even a new type of failure, gets flagged.

**Q: "Is every anomaly a failure?"** No. The steps are: an unusual reading → abnormal behaviour (if it keeps going) → a confirmed failure (only the reported ones).

**Q: "What is ν (nu)?"** Roughly the share of training points the model is allowed to leave outside the boundary (here about 1%).

---

## 8. Deep Learning
- **LSTM:** 2 layers of 32 units, about 14,500 parameters. It has "gates" that decide what to remember and what to forget, so it can remember long sequences.
- **1D-CNN:** 2 convolution blocks. Small filters slide over time and spot local patterns, like the on/off cycle disappearing.
- **Input:** 60 minutes × 13 sensors at a time, always taken from inside one segment and one split. The label is the **last minute** only.
- **Training:** scaling learned from training data, a weighted loss (BCE), the Adam optimiser, and early stopping when validation loss stopped improving. We ran it 4 times with different random seeds.
- **Results:**
  - every run scored a perfect 1.0 on validation;
  - on test, the LSTM caught failure 4 in 3 of 4 runs, and the CNN in 0 of 4 runs.
- **Not used in the final app:** the results change with the random seed, and validation cannot tell us which run is the good one.

---

## 9. Explainability (SHAP)
- **What SHAP does:** it tells you how much each feature pushed a prediction up or down. It comes from game theory (Shapley values): each feature gets "credit" based on its average contribution.
- **The parts always add up:** starting value + all SHAP values = model output. For XGBoost the output is in log-odds, and a test checks this.
- **TreeSHAP** is exact and fast for tree models. We used it for XGBoost.
- **KernelSHAP** works for any model but is slow. We used it for the One-Class SVM on a few selected minutes only.
- **For live anomaly explanations** we show how far each feature is from normal, e.g. "TP3 is 2.1 bar below normal".
- **Know these differences:**
  - **importance** = how much the model uses a feature overall;
  - **contribution** = how much a feature affected *one* prediction;
  - **correlation** = two things move together in the data;
  - **causation** = one thing actually causes the other.
  - **SHAP never proves cause.**

---

## 10. Health Score (Health Engine)
`health = 100 × (1 − [0.40·A + 0.30·D + 0.20·S + 0.10·H])`

| Part | What it means |
|---|---|
| A (anomaly) | One-Class SVM level: normal = 0, low = 0.5, medium = 0.8, high = 1. If there is an ongoing alert, A = 1 |
| D (failure pattern) | XGBoost probability ÷ its threshold, at most 1 |
| S (sensor checks) | How many of 7 key features are outside their normal range, plus whether LPS is on. 4 or more = full penalty |
| H (history) | Number of alerts in the last 24 hours ÷ 3 |

- **Status:**
  - 75 or above = Healthy;
  - 45 to 75 = Warning;
  - below 45 = Critical.
- **Extra rules:** any alert means at least Warning. If A **and** D both fire, it is Critical.
- **Risk:**
  - **High:** D is over its threshold, or there is an ongoing alert of medium or higher severity.
  - **Medium:** any alert, an unusual reading, D at half its threshold or more, or any failed sensor check.
  - Otherwise **Low**.
- **Why these weights:** the anomaly model worked best on new data (0.40); the failure-pattern model is accurate but only knows old failures (0.30); sensor checks are simple and easy to understand (0.20); history shows whether the problem keeps happening (0.10).
- **This score is an index, not a probability.** We checked it on validation before testing.
- **Results:**
  - test: 99.3% of normal minutes were Healthy and 61.5% of failure-4 minutes were Critical. Failure 4 was caught with 0.10 false Critical alarms per day;
  - validation: 2/2 failures caught, 0.07 false alarms per day;
  - example: at 15 July 18:31 the score was 27.5 → Critical, risk High.

---

## 11. Metrics (they will try to confuse you here)
- **Precision** = TP / (TP + FP): of all the alarms, how many were correct.
- **Recall** = TP / (TP + FN): of all the failures, how many we caught.
- **F1** = a balance (harmonic mean) of precision and recall.
- **PR-AUC is our main score.** A random guess scores the same as the failure rate (0.0037). ROC-AUC looks too good when failures are rare, because there are so many normal minutes.
- **Others we report:** balanced accuracy, MCC, Cohen's kappa, false positive and false negative rates, and Brier score.
- **Failure-level scores:** how many failures were caught, how late, and how many false alarms per day. This is what an engineer actually cares about.
- **Trade-off:** a lower threshold catches more failures but gives more false alarms. Too many false alarms and people start ignoring them.

### Trap questions
| Question | Answer |
|---|---|
| "99.6% accuracy is great, right?" | No. Always saying "normal" also gives 99.6%. Accuracy means nothing when only 0.37% are failures |
| "Precision of 0.29 is bad, isn't it?" | Counted by failures, we caught all of them with about 1 false alarm per week. The alarms are clustered around real failures |
| "Recall of 0.61 means you missed 39% of the failure?" | That counts *minutes*. The *failure itself* was caught. Most missed minutes are the 16–34 minute delay at the start and the edges of the window |
| "Isn't 4 failures too few?" | Yes, and we say so openly. We report each failure separately and don't claim it works on other machines |

---

## 12. Application and Engineering
- **FastAPI backend:** more than 25 endpoints (upload, predict, history, explain, alerts, replay, drift, PDF/CSV reports, model info, simulation). It checks inputs, gives clear errors, keeps logs, and has an optional API key. Auto-generated docs are at `/docs`.
- **Database: SQLite with SQLAlchemy** (easy to switch to PostgreSQL).
  - Tables: equipment, datasets (we store only the file location, not a copy), prediction runs (with model version), per-minute predictions, alerts (with acknowledgement) and model versions.
- **Streamlit dashboard with 11 pages:** Overview, Sensor Monitoring, Historical Replay, Failure Prediction, Anomaly Detection, Explainable AI, Equipment Health, Maintenance, Model Performance, Data & Model Monitoring, Live Simulation.
  - Colours are safe for colour-blind users, and charts never draw lines across gaps.
- **One shared code for everything:** training, the API, replay and the simulation all use the same feature code and predictor, so the model always gets inputs made the same way.
- **Model versions:** each model has a file with its settings, threshold, features, training period, data fingerprint (hash) and scores. Every prediction run saves which model version it used.
- **Drift monitoring:**
  - The **KS test** and **PSI** compare each month's data with the training data.
  - The usual PSI limits (0.1 and 0.25) were too strict for this data, so we compare drift against normal month-to-month changes.
  - The biggest summer drift was oil temperature.
  - Data drift (the inputs change) is not the same as concept drift (the link between inputs and failure changes), and neither is the same as the model getting worse (we can only measure that with new labels).
- **Testing: 73 automatic tests** with pytest. They cover functions, the API (including bad uploads and path-traversal attacks), the UI pages and the simulation.
  - In the report: cyclomatic complexity V(G) = 5 for `classify_status` and 3 for `classify_risk`. V(G) = number of decisions + 1 = number of independent paths.
- **Speed:**
  - one day of data is checked in 0.17 seconds;
  - one replay step takes about 162 ms;
  - the API uses about 370 MB of memory.
- **Docker:** `docker compose up` starts the API (port 8000) and the dashboard (port 8501).

### Live Simulation (make it clear that it is fake data)
- **The cycle repeats:**
  - **healthy** for 45–150 simulated minutes (the first one is 30–120 minutes, after 180 minutes of warm-up);
  - **degrading** for 20–45 minutes;
  - **failure** for 30–60 minutes;
  - **maintenance** for 15–30 minutes, with no readings, so a new segment starts;
  - then healthy again.
- So the failure starts at a random time that nobody announces.
- **How the readings are made:**
  - healthy readings are pieces of *real* normal data (far from any failure, with no gaps and no frozen periods);
  - failure readings are pieces of real failure data;
  - while degrading, the two are mixed. The mix goes slowly from 0% to 100% failure, and a little random noise (1%) is added.
- Every new minute goes through the real pipeline. The page is labelled "SIMULATED" and is **never used for any result**.
- **Q: "Is this real-time monitoring?"** No. Replay and simulation only show how the pipeline works. Connecting real live sensors is future work.

---

## 13. Limitations (say these before the examiner does)
- only one compressor, 7 months and 4 failures, all of them air leaks;
- the failure start and end times are approximate, and some failures were not reported;
- no early warning and no RUL;
- the health weights were chosen by us and checked on only 2 failures;
- only replay, no live sensors;
- the drift (PSI) limits need tuning.

## 14. Future Work
- get more failures from other metro trains;
- use the slowly rising anomaly score (seen 14 hours before failure 4) as an early warning;
- let thresholds adjust automatically for seasons, and build separate models for different operating modes;
- connect live sensors, use a bigger database, and add user logins;
- tune alarms based on what a missed failure costs compared with a false alarm.

---

## 15. Quick Tricky Questions
1. **Supervised vs unsupervised here?** Supervised learns from failure labels (XGBoost). Our anomaly models learn only from normal data. This is called *novelty detection* (semi-supervised).
2. **Novelty vs outlier detection?** In novelty detection the training data is clean and you check new points (our case, so LOF uses `novelty=True`). In outlier detection the training data itself contains the strange points.
3. **Bagging vs boosting?** Bagging builds trees separately and averages them; it reduces variance (Random Forest). Boosting builds trees one after another, each fixing the last one's mistakes; it reduces bias (XGBoost).
4. **Gini vs entropy?** Both measure how mixed a group is. Gini = 1 − Σp². Entropy = −Σ p·log p.
5. **Is this overfitting?** Validation 0.999 vs test 0.48 is mostly because the test failure was a **different type**, not classic overfitting to noise.
6. **Bias and variance?** A single decision tree changes a lot with the data (high variance); Random Forest fixes this by averaging. Logistic Regression is too simple (high bias) because it is a straight line.
7. **Kernel trick?** It lets SVM draw curved boundaries without actually moving the data to a bigger space. RBF kernel: K(x, y) = exp(−γ‖x − y‖²).
8. **Why not use 0.5 as the threshold?** With class weights and rare failures, the probabilities are not balanced around 0.5, so we chose the threshold on validation data.
9. **How did you tune hyperparameters?** Random search on the forward-chaining folds, scored by PR-AUC.
10. **What is an embargo?** A 12-hour empty gap between train, validation and test so data windows don't overlap.
11. **Why 1-minute rows?** Less noise, 6 times less data, matches how fast failures develop, and gives one common unit for scoring.
12. **Why Parquet instead of CSV?** It stores column types, is compressed, and loads much faster.
13. **Vanishing gradient?** In long sequences the learning signal gets smaller and smaller and disappears. LSTM's memory cell helps stop this.
14. **Why Adam?** It adjusts the learning rate for each weight automatically and works well without much tuning.
15. **Early stopping?** Stop training when validation loss stops improving, to avoid overfitting.
16. **Why class weights instead of SMOTE?** They give extra importance to failure rows without making fake data.
17. **What is MCC?** A score from −1 to +1 that uses all 4 boxes of the confusion matrix. It works well even when classes are very uneven.
18. **Brier score?** The average of (predicted probability − real answer)². It tells you how good the probabilities are.
19. **Concept drift in our data?** Failure 4 is a good example: same type of failure (air leak), but different sensor signs.
20. **Why not add failure 4 to training?** Then we would have nothing new left to test on. With more failures, that is exactly what we would do.
21. **No labels, so how did you label?** We used the company's reported failure times for `y_detect`. The unreported look-alikes were marked uncertain and left out.
22. **Correlation vs causation example?** Hot oil goes together with failure, but the real cause is the leak that makes the compressor run nonstop.
23. **What is PSI?** Population Stability Index. It measures how much a feature's distribution has shifted: Σ (new% − old%) × ln(new% / old%).
24. **KS test?** The biggest gap between two cumulative distributions.
25. **How do you know the test wasn't tuned?** We opened the test only **once**, after the model, thresholds and health weights were all fixed.

---

## 16. "Why did you do it this way and not another way?"

For every "why not X?" question, answer in this order:
**what we chose → what the other option was → why the other option is worse for *our* data.**
Never say "because everyone uses it".

### 16.1 How we framed the problem
| We chose | Instead of | Why |
|---|---|---|
| **Anomaly detection as the main detector** | Only supervised models | Only 4 failures, 1 in training. A supervised model only learns "what old failures looked like". Failure 4 looked different, so it was missed. Anomaly detection learns normal and flags anything different |
| Keep XGBoost as a **second helper** | Throw it away | It is very accurate on the known leak pattern and gives SHAP explanations. In the health score it adds extra evidence |
| **Detection** (failure is happening now) | Prediction hours ahead | We tested 2, 6 and 12 hours ahead; it was no better than guessing, so claiming it would be dishonest |
| **No RUL** | RUL with LSTM or regression | RUL needs data of machines slowly wearing out. We only have short leaks that were fixed |
| Labels from reported failures | Using the LPS alarm as the label | LPS did not turn on at all in failures 1 and 2 |
| Leave out unreported look-alikes | Calling them failure or normal | Calling them failure = making up labels. Calling them normal = teaching the model that failure is normal |

### 16.2 Cleaning the data
| We chose | Instead of | Why |
|---|---|---|
| **Split at gaps, no filling** | Filling gaps (interpolation) | Some gaps are many hours long. Filling them invents readings and fake smooth trends |
| Fill only gaps of 60 seconds or less | Fill everything / delete rows | Tiny gaps are just logger delay and safe to fill. Long gaps are real missing data |
| **Keep outliers** | Remove outliers | In machine monitoring the outlier is often the failure. Removing it removes what we are looking for |
| Leave out frozen periods | Keep them as normal | Values stuck for hours are a logger fault. The model would learn "flat = normal" |
| **StandardScaler** | RobustScaler | TP2 is about 0 most of the time, so its IQR is only 0.004 bar. RobustScaler would turn a normal 10 bar into about 2,500 |
| StandardScaler | MinMaxScaler | MinMax depends on the single biggest and smallest value, which are often failures |
| Scaler learned on training data only | Scaling all data first | Otherwise test data information leaks into training |
| **Parquet** | CSV | Faster, smaller, keeps the data types |
| Median for missing values | Mean / delete rows | The median is not pulled by extreme values. Deleting rows breaks the time order |

### 16.3 Splitting the data
| We chose | Instead of | Why |
|---|---|---|
| **Split by time** | Random split | Readings 10 seconds apart are near-copies. A random split puts copies in the test and makes scores look fake-good. In real life you always predict the future from the past |
| **Forward-chaining** | K-fold | K-fold trains on future data to predict the past |
| Forward-chaining | Leave-one-failure-out | It would also train on failures that happen later |
| **12-hour gap** | No gap | Windows near the edge would contain data from the other part |
| Open the test only once | Checking the test many times | Every look at the test that changes something makes the test score unfair |
| Keep failure 4 for testing | Train on all 4 failures | Then nothing new would be left to test, and we would never have found the main result |

### 16.4 Features
| We chose | Instead of | Why |
|---|---|---|
| **1-minute rows** | Raw 10-second rows | 6 times less data, less noise, and failures develop over minutes to hours |
| 1-minute | 1-hour rows | Failure 4 lasted only 4.5 hours, so we would get only about 4 failure rows |
| **Features based on how the machine works** | Only raw sensor values | A leak shows up as *behaviour* (nonstop pumping, pressure not recovering), not in one raw value |
| Our own features | Auto tools like tsfresh (thousands of features) | Thousands of features with 1 training failure = easy overfitting, and very hard to explain |
| **Windows that look only backwards** | Windows centred on the minute | A centred window uses future readings, which is impossible in real use |
| **Difference from the last 24 hours** | Raw values only | Removes the summer heat effect, so a hot summer is not flagged as a failure |
| Choosing features by filters + XGBoost | PCA | PCA mixes features into new ones that nobody can explain |
| 18 separate features for anomaly models | The 33 supervised features | The 33 were chosen using failure 1's labels, so they would push the anomaly model toward old failures |

### 16.5 Imbalance
| We chose | Instead of | Why |
|---|---|---|
| **Class weights** | SMOTE / ADASYN | We tested them and they did not help. Mixing near-identical minutes adds nothing new |
| Class weights | Undersampling | It throws away over 98% of the normal data, but we need that variety to avoid false alarms |
| Class weights | Oversampling (copying) | Copying the same failure many times makes the model memorise it |

### 16.6 Models
| We chose | Instead of | Why |
|---|---|---|
| **XGBoost** | Random Forest | Same on validation, but better on test (0.48 vs 0.16). It has built-in class weights and fast SHAP |
| XGBoost | Neural networks | For table data with little data, boosted trees are usually best, cheaper and easier to explain |
| Compare 5 models | Only XGBoost | When *all* 5 missed failure 4, it showed the problem was the data, not one algorithm |
| SVM on 20,000 rows | SVM on all rows | SVM gets very slow on big data; it would not run in 8 GB RAM |
| **One-Class SVM** | Isolation Forest | Both caught all failures, but Isolation Forest had more false alarms (0.26 vs 0.16 per day) |
| One-Class SVM | LOF | LOF caught 0 failures |
| One-Class SVM | z-score | Simple, but it missed failure 4 because of the summer effect |
| One-Class SVM | Autoencoder | Needs more data and tuning; our deep models were already unstable. Good future work |
| Random search | Grid search | Similar quality with far fewer runs |
| Random search | Optuna (smart search) | With only 2 folds and few failures, extra fine-tuning just overfits validation |
| **Deep learning not used in the app** | Using the LSTM | Results changed with the random seed, and we couldn't know in advance which run would work |
| LSTM / CNN | Transformer | Transformers need much more data than 1 training failure |

### 16.7 Thresholds, alerts and scores
| We chose | Instead of | Why |
|---|---|---|
| **Threshold chosen on validation** | 0.5 | With rare failures, probabilities are not centred at 0.5 |
| Anomaly threshold from a separate normal period | Using failure labels | Keeps the anomaly model completely label-free, and directly controls false alarms |
| A separate normal period | The training data | Scores on training data are too low, so thresholds would be too tight and cause false alarms |
| **10 out of 30 minutes rule** | Alarm on every flagged minute | Single spikes (like motor start-up) would cause too many false alarms |
| **PR-AUC** | Accuracy | Saying "always normal" already gives 99.6% |
| PR-AUC | ROC-AUC | ROC-AUC looks good even with many false alarms, because normal minutes are so many |
| **Counting failures caught** | Only counting minutes | Engineers act on failures, not minutes. One failure has hundreds of minutes |

### 16.8 Explanations and health score
| We chose | Instead of | Why |
|---|---|---|
| **SHAP** | XGBoost's built-in importance | Built-in importance only gives an overall ranking. SHAP explains each single prediction, with direction |
| SHAP | LIME | LIME can give different answers each run. TreeSHAP is exact |
| TreeSHAP for XGBoost, KernelSHAP only offline | KernelSHAP everywhere | KernelSHAP is too slow for live use |
| Distance-from-normal for live explanations | KernelSHAP live | Instant and easy to read: "TP3 is 2.1 bar below normal" |
| **Rule-based health score** | A trained health model | There are no health labels to train on. A formula is clear and engineers can check it |
| Weights 0.40 / 0.30 / 0.20 / 0.10 | Equal weights | Based on which evidence worked best on new data |
| Call it an index | Call it a probability | It is not calibrated, so calling it a probability would be wrong |

### 16.9 Software
| We chose | Instead of | Why |
|---|---|---|
| **FastAPI** | Flask | Input checking and API docs are built in; it is faster and uses type hints |
| FastAPI | Django | Django is too big for a simple ML API |
| API and dashboard separate | Everything inside Streamlit | The same API serves the dashboard, reports and other apps, and each part can be tested on its own |
| **Streamlit** | React | We work in Python; React would double the code for the same charts |
| **Plotly** | Matplotlib in the dashboard | Plotly lets you zoom and hover. Matplotlib was used for report images |
| **SQLite** | PostgreSQL | No setup and one file, enough for one machine. Switching to PostgreSQL is easy later |
| Store file locations + results in the DB | Store the raw sensor data in the DB | The raw data is already in Parquet; copying it wastes space |
| SQLite | MongoDB | Our data is connected (runs → predictions → alerts), which suits SQL tables |
| **One shared code for training and the app** | Separate scripts | Separate code can make inputs slightly different and silently give wrong predictions |
| Model version tracking | No tracking | We can trace every prediction to the exact model and data |
| **Docker** | Manual install | Works the same on every computer; one command starts everything |
| **73 automatic tests** | Manual testing | Can be repeated anytime and catches new bugs |
| Replay of old data | Claiming real-time monitoring | We had no live sensor feed. Replay is honest |
| Simulation built from real data pieces | Random or sine-wave fake data | Random data doesn't behave like a compressor, so the demo would mean nothing |
| KS + PSI drift checks | Only checking accuracy | Accuracy needs new labels, which are rare. Input drift can be seen right away |

### 16.10 If they ask "why not X?" and you didn't prepare
1. Say our limits: "We had only 4 failures, 1 in training, time data where neighbours are near-copies, and no labels."
2. Say what X needs that we didn't have: more labels, more data, future information, or more computing power.
3. Say what you would expect, and call it future work if it makes sense.

**Example: "Why not GRU?"** → "GRU is like LSTM with fewer gates. Our LSTM showed the problem is having only one training failure, not the model type, so GRU would probably be unstable too. We would test it if we had more failures."

**Be honest:** we did NOT test autoencoders, Transformers, Optuna, GRU, tsfresh, LIME, MongoDB or React. If asked, say
"we didn't test it, and this is why it doesn't suit our data". Never say "we tried it".

---

## 17. Code Map: what each folder and file does

The examiner may open the code and ask "what does this file do?" or "show me where you prevent leakage".
This section answers that.

### 17.1 The big picture: how the code is organised
```
PredictOps/
├── src/predictops/   ← THE CORE LIBRARY. All the real ML logic lives here. Everything else calls it.
├── scripts/          ← One script per project stage. They USE src/ to build data, train models and make reports.
├── backend/          ← FastAPI web server (the API). Uses src/ to make predictions and saves results in the database.
├── frontend/         ← Streamlit dashboard. Only talks to the backend through HTTP; it has no ML code.
├── tests/            ← 73 automatic tests (pytest).
├── configs/          ← config.yaml: all settings in one place (paths, failure times, splits, health weights).
├── data/             ← The data at every stage (raw → cleaned → features → predictions).
├── models/           ← Saved trained models + their metadata (settings, thresholds, scores).
├── reports/          ← Results: figures, tables, SHAP plots, screenshots.
├── database/         ← predictops.db, the SQLite database used by the app.
├── notebooks/        ← 8 Jupyter notebooks, one per stage, for showing the work step by step.
├── docs/             ← Documentation of every stage (00 to 14) + data dictionary.
├── deliverables/     ← Final report, presentation and this guide.
├── mlruns/, mlruns.db← MLflow experiment tracking (records of every training run).
├── run.py            ← One launcher for everything (start app / rebuild pipeline / run tests).
├── Dockerfile, docker-compose.yml ← Run the whole app in containers.
├── render.yaml       ← Settings to deploy the app on the Render cloud (API + dashboard).
└── requirements*.txt, pyproject.toml ← Python libraries needed.
```

**Key design idea (say this if asked about the structure):** all the ML logic is written **once** in `src/predictops`.
The training scripts, the API, the replay and the simulation all call the **same** functions. That way, the data is
prepared exactly the same way during training and in the app. If they were different, the app would silently give wrong answers
(this problem is called **training–serving skew**).

**Layers:**
- `frontend` → (HTTP) → `backend/routes` → `backend/services` → `src/predictops/inference` → models;
- `backend/db.py` saves the results in SQLite.

### 17.2 `run.py` — the launcher
| Command | What it does |
|---|---|
| `python run.py` | Starts the API (port 8000) and the dashboard (port 8501) together |
| `python run.py pipeline` | Rebuilds **everything** from the raw CSV by running 12 scripts in order (list in 17.4) |
| `python run.py test` | Runs all the tests |

### 17.3 `src/predictops/` — the core library (most important)
| File | What it handles | Main functions / classes |
|---|---|---|
| `config.py` | Reads `configs/config.yaml` so every file uses the same settings | `load_config`, `project_path`, `sensor_columns` |
| `data/ingestion.py` | **Reading and checking the data.** Loads the raw CSV, checks columns, types and value ranges (digital must be 0/1, no infinity, valid timestamps), converts to small types, adds `segment_id` at gaps > 10 min, saves Parquet. The same checker is used for user uploads | `validate_frame`, `ValidationReport`, `add_segments`, `load_raw`, `build_intermediate`, `load_intermediate` |
| `eda.py` | **Exploratory analysis.** Marks failure windows, works out whether the compressor is off/resting/pumping, daily stats, and draws all the EDA charts | `add_failure_flags`, `operating_state`, `daily_operation_stats`, `fig_*` (chart functions) |
| `preprocessing/labels.py` | **Flags and targets.** Marks frozen-sensor periods and the 3 uncertain periods, and creates `y_detect` and `y_warn_2h/6h/12h` | `add_frozen_flag`, `add_uncertain_flag`, `add_targets`, `build_labeled` |
| `preprocessing/splits.py` | **Splitting by time.** Assigns each row to train/validation/test with the 12 h gap, and makes the forward-chaining folds A and B | `assign_split`, `event_folds` |
| `preprocessing/pipeline.py` | **Cleaning + scaling, without leakage.** Fills only short gaps (≤ 60 s), fixes infinity values, and builds the imputer + StandardScaler pipeline that learns from training data only | `clean_sensor_frame`, `make_preprocessor` |
| `features/build.py` | **Feature engineering** (shared by training AND the app). Turns 10-s rows into 1-minute rows, then makes the 132 past-only rolling features. Rolling windows never cross a segment gap | `aggregate_minutes`, `add_rolling_features`, `_run_length`, `feature_columns`, `build_features` |
| `models/zoo.py` | **Supervised model definitions.** Creates LR, DT, RF, SVM and XGBoost, each wrapped in a pipeline, plus the hyperparameter ranges for random search | `make_estimator`, `make_pipeline` |
| `models/deep.py` | **Deep learning** (PyTorch). Builds 60-minute sequences inside segments, defines the LSTM and 1D-CNN networks, and handles training with early stopping | `make_sequences`, `LSTMNet`, `CNN1D`, `train_model`, `predict` |
| `anomaly/detectors.py` | **Anomaly detection.** One class wraps OCSVM / Isolation Forest / LOF / robust-z: fit on normal data, give scores, set the 3 severity thresholds from the calibration period. Also has the "10 of 30 minutes" persistence rule | `AnomalyDetector`, `persistent_mask` |
| `evaluation/metrics.py` | **All scoring**, shared by every experiment: precision, recall, PR-AUC, MCC and the rest, the best threshold, grouping alerts into episodes, and failure-level results (caught? how late? false alarms per day?) | `classification_report_dict`, `best_f1_threshold`, `alert_episodes`, `event_level_report` |
| `explainability/shap_explain.py` | **SHAP.** TreeSHAP for XGBoost and KernelSHAP for the One-Class SVM | `TreeShapExplainer`, `AnomalyShapExplainer` |
| `engine/health.py` | **Health engine.** Sensor range checks, the health score formula (0.40/0.30/0.20/0.10), the Healthy/Warning/Critical status, the Low/Medium/High risk, and the written reasons and recommendations | `sensor_checks`, `health_score`, `classify_status`, `classify_risk`, `build_reasons` |
| `inference/predictor.py` | **The single prediction pipeline** used by the API, dashboard, replay and simulation. It loads all saved models once | `PredictOpsPredictor`, `get_predictor`, `ArtifactError` |
| `monitoring/drift.py` | **Drift checks.** PSI and KS test per feature, compared with the training data | `psi`, `drift_report` |
| `reports/pdf.py` | **PDF maintenance report** for one prediction run (made with reportlab) | `build_run_report` |
| `simulation/simulator.py` | **Live simulation** (fake data made from real pieces). Phases healthy → degrading → failure → maintenance; runs in a background thread | `StreamSimulator`, `SignalSource`, `Phase`, `_contiguous_blocks` |

**What `predict_raw` does, step by step** (`inference/predictor.py`). Learn this flow:
1. `validate_frame`: reject bad input.
2. `clean_sensor_frame`: fill only tiny gaps.
3. `build_features`: make the 1-minute rows and the 132 features.
4. **A**: the One-Class SVM score → severity → persistent alert → Normal/Anomalous.
5. **D**: the XGBoost probability ÷ its threshold.
6. **S**: the sensor range checks.
7. **H**: alerts in the last 24 h.
8. `health_score`, then `classify_status` and `classify_risk`, then `build_reasons` (reasons + recommendations).
9. Output: one row per minute.

### 17.4 `scripts/` — one script per project stage (run in this order by `run.py pipeline`)
| # | Script | What it does | Output |
|---|---|---|---|
| 1 | `build_intermediate.py` | Raw CSV → checked, typed, segmented Parquet | `data/intermediate/metropt3.parquet` |
| 2 | `inspect_dataset.py` | Dataset inspection: sizes, gaps, ranges, failure table | `docs/01_dataset_inspection.md`, `docs/data_dictionary.md` |
| 3 | `run_eda.py` | All EDA charts and statistics | `reports/figures/`, `reports/eda_stats/` |
| 4 | `build_processed.py` | Flags, targets, splits, 1-minute features | `data/processed/*.parquet`, `data/train|validation|test/` |
| 5 | `feature_study.py` | Feature selection (132 → 33), before/after feature engineering, and the SMOTE/ADASYN/class-weight study | `reports/feature_study/` |
| 6 | `train_ml.py` | Tunes the 5 supervised models on the folds, picks the best, then **opens the test once** | `models/trained/detect_*.joblib`, `models/metadata/ml_*.json`, `reports/ml/` |
| 7 | `train_anomaly.py` | Trains the 4 anomaly models on normal data, sets the thresholds, scores failure-level results | `models/trained/anomaly_*.joblib`, `reports/anomaly/` |
| 8 | `train_dl.py` | LSTM and CNN experiments | `reports/dl/`, `models/metadata/dl.json` |
| 9 | `explain.py` | SHAP: overall importance and single-minute explanations (e.g. why failure 4 was missed) | `reports/shap/` |
| 10 | `evaluate_health.py` | Learns the normal sensor ranges and tests the health engine on validation and test | `reports/health/` |
| 11 | `precompute_history.py` | Runs the final pipeline over the whole dataset for the dashboard timelines and replay | `data/processed/history_predictions.parquet` |
| 12 | `monitoring_report.py` | Monthly drift report | `reports/monitoring/` |
| extra | `dl_seed_stability.py` | Re-runs the LSTM/CNN with 4 seeds to check stability | `reports/dl/` |
| extra | `benchmark.py` | Speed and memory measurements | `reports/benchmark.json` |
| extra | `screenshots.py` | Takes dashboard screenshots with Chrome | `reports/screenshots/` |

### 17.5 `backend/` — the FastAPI server
| File | What it handles |
|---|---|
| `main.py` | Creates the FastAPI app: CORS, request logging, error handlers, creates the database tables at start-up, and starts the simulation if `SIMULATION_AUTOSTART` is set |
| `settings.py` | Reads settings from environment variables / `.env` (database URL, API key, CORS, upload size limit). **No passwords are written in the code** |
| `dependencies.py` | Things each request needs: a database session, the shared predictor (loaded once and cached), and the optional API-key check |
| `db.py` | **Database tables** (SQLAlchemy): `Equipment`, `Dataset`, `PredictionRun`, `Prediction`, `Alert`, `ModelVersion`, with foreign keys and indexes |
| `schemas.py` | **Pydantic models:** the exact shape of every request and response, so bad input is rejected automatically (`ObservationsIn`, `PredictRequest`, `MinuteAssessment`, `AlertOut` and others) |
| `routes/api.py` | **All the URLs (endpoints).** Kept thin: check the input → call a service → return a schema. Covers health, models, datasets/upload, predict, runs, explain, alerts/acknowledge, dashboard, history, replay, drift, reports, simulation |
| `services/datasets.py` | **Uploads:** safe file names (stops path-traversal attacks), size limit, validation, no duplicate files (checked by hash), sample files |
| `services/predictions.py` | **Main business logic:** runs the predictor, then saves the run, the per-minute predictions and the alerts to the database |
| `services/models_info.py` | Reads model metadata and scores for the "Model Performance" page, and registers model versions |

### 17.6 `frontend/` — the Streamlit dashboard
| File | What it handles |
|---|---|
| `app.py` | Entry point. Registers all 11 pages in the **top navigation bar** (Overview + drop-down menus Monitor / Intelligence / Models & Data) |
| `lib/api.py` | **The only way the dashboard gets data.** A small HTTP client for the backend, with caching, error handling and breaking chart lines at gaps (`break_gaps`) |
| `lib/ui.py` | The look: light white / light-blue theme, the `subnav` section buttons used inside every page, colours, cards, badges, notes, the Plotly chart style, shading of failure windows |
| `lib/common.py` | Shared helpers: the failure list, the date-range picker, the "which data split is this" note |
| `views/overview.py` | Overview page: key numbers, the 7-month health trend, the line between the training period and the unseen test period |
| `views/sensors.py` | Sensor Monitoring: one chart per sensor, with failure windows and our alerts marked |
| `views/replay.py` | Historical Replay: steps through old data minute by minute through the real pipeline |
| `views/prediction.py` | Failure Prediction: upload or choose a file → run prediction → results, SHAP for one minute, PDF/CSV download |
| `views/anomaly.py` | Anomaly Detection: anomaly score, thresholds, severity, alert table |
| `views/explain.py` | Explainable AI: overall SHAP importance, waterfall chart for one minute, anomaly deviations |
| `views/health.py` | Equipment Health: health score bands, status, reasons for the worst minute |
| `views/maintenance.py` | Maintenance: open alerts, acknowledge them with a note, alert history |
| `views/performance.py` | Model Performance: validation vs test scores, confusion matrices, curves |
| `views/monitoring.py` | Data & Model Monitoring: monthly drift, drift check on any upload, model versions |
| `views/simulation.py` | Live Simulation: refreshes every 2 seconds, shows the fake phases as shaded areas |

**Why a top bar + section buttons:** with 11 pages, a top bar keeps the full screen width for charts, and the section buttons show one part of a page at a time, so nobody has to scroll through long pages. It also loads faster, because only the chosen section is drawn.

**Why the frontend has no ML code:** the frontend only shows data. Any other app (for example a mobile app) could use the same API.

### 17.7 `tests/` — 73 automatic tests
| File | What it checks |
|---|---|
| `test_ingestion.py` | The data checker: empty file, missing columns, bad timestamps, text in number columns, digital values not 0/1, infinity, segments split at gaps |
| `test_intermediate_data.py` | The real processed data: row count matches the raw data, sorted, no duplicates, raw file unchanged |
| `test_eda.py` | Operating-state thresholds, frozen-period detection, failure flags match the config |
| `test_preprocessing.py` | **Leakage tests:** warning target only before onset, future columns banned as features, scaler learns from training data only, splits are in time order and don't overlap, test holds only failure 4 |
| `test_features.py` | **The key test:** features use **past data only** (prefix test); no label or ID columns in the features; 1-minute grouping is right-closed; rolling never crosses a gap |
| `test_engine_inference.py` | Health score stays between 0 and 100 and goes down as evidence goes up; status and risk rules; prediction on the real failure-4 sample; a normal day is mostly Healthy; **SHAP values add up**; clear error if model files are missing |
| `test_api.py` | Full API with a temporary database: upload → validate → predict → explain → download; alerts and acknowledge; bad uploads; files too large; **path traversal**; replay across a gap; PDF report |
| `test_frontend_pages.py` | Every dashboard page opens without errors (needs the API running) |
| `test_simulation.py` | The phases come in the right order, and the pipeline scores healthy and failure phases differently |

### 17.8 Data, model and report folders
| Folder | What is inside |
|---|---|
| `data/raw/` | The original CSV (not uploaded to GitHub because it is too big), the description PDF, and `SHA256SUMS` (fingerprint to prove the raw file was never changed) |
| `data/intermediate/` | `metropt3.parquet`: checked, typed data with segments |
| `data/processed/` | `metropt3_labeled.parquet` (flags + targets), `features_1min.parquet` (the 132 features), `features_monitoring.parquet`, `history_predictions.parquet` (predictions for the whole dataset), `test_predictions.parquet` |
| `data/train/`, `data/validation/`, `data/test/` | The sensor data for each split, kept separate |
| `data/sample/` | Small CSV files to try in the app: the failure-4 day, a normal day, a 2-hour clip |
| `models/trained/` | Saved models (`.joblib`): the 5 supervised models and the 4 anomaly models |
| `models/preprocessing/` | `base_preprocessor.joblib`: the fitted imputer and scaler |
| `models/metadata/` | One JSON file per model: settings, threshold, features used, training period, data hash, scores. `ml_selection.json` records why XGBoost was chosen |
| `reports/` | `figures/` (all charts), `ml/`, `anomaly/`, `dl/`, `shap/`, `health/`, `monitoring/`, `feature_study/`, `eda_stats/`, `preprocessing_stats/` (result tables), `screenshots/`, `benchmark.json` |
| `configs/config.yaml` | Sections: `project`, `paths`, `dataset` (sensor lists, ranges), `failure_events` (the 4 failure times, copied exactly from the PDF), `targets`, `splits` (dates + 12 h gap), `health_engine` (weights, cut-offs, alert rule) |
| `database/predictops.db` | The app's SQLite database |
| `notebooks/` | 01 inspection, 02 EDA, 03 preprocessing, 04 features, 05 imbalance, 06 ML training, 07 anomaly, 08 deep learning |

### 17.9 "Show me where…" (quick answers when the examiner points at the code)
| Examiner asks | Open this |
|---|---|
| Where do you stop future data leaking into features? | `src/predictops/features/build.py` + the test in `tests/test_features.py` (`test_features_are_past_only`) |
| Where is the scaler fitted on training data only? | `src/predictops/preprocessing/pipeline.py` (`make_preprocessor`) + `tests/test_preprocessing.py` |
| Where are the splits and the 12 h gap? | `src/predictops/preprocessing/splits.py` and `configs/config.yaml → splits` |
| Where are the failure times? | `configs/config.yaml → failure_events` |
| Where is SMOTE tested? | `scripts/feature_study.py` (inside the training folds only) |
| Where is the model chosen and the test opened? | `scripts/train_ml.py` |
| Where is the One-Class SVM? | `src/predictops/anomaly/detectors.py` and `scripts/train_anomaly.py` |
| Where is the health formula? | `src/predictops/engine/health.py` (`health_score`, `classify_status`, `classify_risk`) |
| Where is the code shared between training and the app? | `src/predictops/inference/predictor.py` (`predict_raw`) |
| Where is the input validation? | `src/predictops/data/ingestion.py` (`validate_frame`) and `backend/schemas.py` |
| Where are the database tables? | `backend/db.py` |
| Where are the API endpoints? | `backend/routes/api.py` (live list at `http://127.0.0.1:8000/docs`) |
| Where is SHAP? | `src/predictops/explainability/shap_explain.py` and `scripts/explain.py` |
| Where is the simulation? | `src/predictops/simulation/simulator.py` and `frontend/views/simulation.py` |

---

## 18. Who studies what (matches the slides)
| Speaker | Slides | Must know well |
|---|---|---|
| 1 | 1–6 | Introduction, how the compressor works, problem, objectives, **the whole dataset section (Section 1)** |
| 2 | 7–10 | Data findings, cleaning, leakage, splitting, features, SMOTE (Sections 2–5) |
| 3 | 11–15 | Models, results, anomaly detection, deep learning, SHAP (Sections 6–9 and 11) |
| 4 | 16–21 | Architecture, health score, app, testing, simulation, limitations, conclusion (Sections 10 and 12–14) |

**Everyone** must know: Section 0, the failure table (1.4), leakage (5), the failure-4 story, Section 16, and "no SMOTE, accuracy means nothing here".

**Golden rule:** if you don't know an answer, say "we didn't test that, but from what we saw, I would expect…". **Never make up a number.**
