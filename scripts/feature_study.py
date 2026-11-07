"""Stage 4c — feature selection, before/after-FE comparison, and class-imbalance study.

Experimental protocol (no test data is touched here):
  * y_detect : train split  -> validation split           (events 1 -> 2, 3)
  * y_warn_6h: forward fold B -> train on dev rows before the event-3 block (contains
               event 2 onset), validate on event 3 ± 3 days.  (Train split alone has
               zero usable warning positives — docs/03 §3.)
  * Only `usable` rows; warning rows inside failure windows (NaN target) are dropped.
  * All selection decisions use TRAINING rows only.

Outputs: data/processed/features_1min.parquet, models/metadata/selected_features.json,
         reports/feature_study/*.csv, reports/figures/features/*.png
Run:  .venv/bin/python scripts/feature_study.py   (~3–6 min)
"""
from __future__ import annotations

import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from predictops.config import load_config, project_path
from predictops.evaluation.metrics import classification_report_dict
from predictops.features.build import ANALOG, DIGITAL, build_features, feature_columns
from predictops.preprocessing.pipeline import make_preprocessor
from predictops.preprocessing.splits import event_folds

warnings.filterwarnings("ignore")
SEED = 42
cfg = load_config()
OUT = project_path("reports/feature_study"); OUT.mkdir(parents=True, exist_ok=True)
FIG = project_path("reports/figures/features"); FIG.mkdir(parents=True, exist_ok=True)
t0 = time.time()

# ------------------------------------------------------------------ build features
df10 = pd.read_parquet(project_path("data/processed/metropt3_labeled.parquet"))
m = build_features(df10)
m.to_parquet(project_path("data/processed/features_1min.parquet"), index=False)
ALL = feature_columns(m)
RAW = [c for c in ANALOG + DIGITAL]          # 1-min means of the sensors = "no feature engineering"
print(f"features built: {m.shape}, {len(ALL)} candidate features")

# ------------------------------------------------------------------ experiment data
use = m[m.usable == 1]
det_tr = use[use.split == "train"]
det_va = use[use.split == "validation"]
mi = m.set_index(pd.RangeIndex(len(m)))
fold = {e: (tr, va) for e, tr, va in event_folds(mi.assign(split=m.split), cfg)}
tr_idx, va_idx = fold[3]
wtr = mi.loc[tr_idx]; wtr = wtr[(wtr.usable == 1) & wtr.y_warn_6h.notna()]
wva = mi.loc[va_idx]; wva = wva[(wva.usable == 1) & wva.y_warn_6h.notna()]
TASKS = {"y_detect": (det_tr, det_va, "y_detect"), "y_warn_6h": (wtr, wva, "y_warn_6h")}
pd.DataFrame([{"task": k, "train_rows": len(a), "train_pos": int(a[y].sum()), "val_rows": len(b),
               "val_pos": int(b[y].sum())} for k, (a, b, y) in TASKS.items()]).to_csv(OUT / "task_sizes.csv", index=False)

# ------------------------------------------------------------------ feature selection (train only)
X = det_tr[ALL]
vt = VarianceThreshold(1e-6).fit(X.fillna(X.median()))
step1 = [c for c, keep in zip(ALL, vt.get_support()) if keep]
corr = X[step1].sample(min(len(X), 60_000), random_state=SEED).corr().abs()
# greedy: keep a feature unless it is >0.98 correlated with an already-kept one
# (order: raw sensors first, then shorter windows -> the simpler feature survives)
step2 = []
for c in step1:
    if all(corr.loc[c, k] <= 0.98 for k in step2):
        step2.append(c)
# model-based ranking on TRAIN only (XGBoost gain, y_detect); warning target has no
# positives in the train split, so the ranking is shared.
xgb_rank = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=8, random_state=SEED, eval_metric="aucpr")
xgb_rank.fit(det_tr[step2], det_tr.y_detect)
imp = pd.Series(xgb_rank.get_booster().get_score(importance_type="gain")).reindex(step2).fillna(0).sort_values(ascending=False)
imp.to_csv(OUT / "xgb_gain_importance_train.csv", header=["gain"])
TOP = 30
step3 = imp.index[:TOP].tolist()
# guarantee interpretable physical coverage: each raw analog sensor's 30-min mean stays in
for c in [f"{s}_mean_30min" for s in ANALOG]:
    if c in step2 and c not in step3:
        step3.append(c)

sel = {"all_engineered": ALL, "after_variance": step1, "after_correlation": step2, "selected": step3, "raw_sensor_means": RAW}
pd.DataFrame([{"stage": k, "n_features": len(v)} for k, v in sel.items()]).to_csv(OUT / "selection_stages.csv", index=False)
print("selection:", {k: len(v) for k, v in sel.items()})


# ------------------------------------------------------------------ helpers
def make_model(kind, pos_weight=None):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if pos_weight else None)
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         n_jobs=8, random_state=SEED, eval_metric="aucpr",
                         scale_pos_weight=pos_weight if pos_weight else 1.0)


def run(features, task, kind, sampler=None, weighted=False):
    tr, va, y = TASKS[task]
    pw = (len(tr) - tr[y].sum()) / max(tr[y].sum(), 1) if weighted else None
    steps = [("pre", make_preprocessor(features))]
    if sampler is not None:
        steps.append(("sample", sampler))
    steps.append(("model", make_model(kind, pw)))
    pipe = ImbPipeline(steps)
    t = time.time()
    pipe.fit(tr[features], tr[y].astype(int))
    fit_s = time.time() - t
    s = pipe.predict_proba(va[features])[:, 1]
    r = classification_report_dict(va[y].astype(int), s, 0.5)
    r.update({"fit_seconds": round(fit_s, 1)})
    return r, s


# ------------------------------------------------------------------ before vs after FE
rows = []
for task in TASKS:
    for fs in ["raw_sensor_means", "all_engineered", "after_correlation", "selected"]:
        for kind in ["logreg", "xgboost"]:
            r, _ = run(sel[fs], task, kind, weighted=True)
            rows.append({"task": task, "feature_set": fs, "n_features": len(sel[fs]), "model": kind,
                         **{k: r[k] for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "fit_seconds"]}})
            print(rows[-1])
fe = pd.DataFrame(rows).round(4); fe.to_csv(OUT / "fe_comparison.csv", index=False)

# ------------------------------------------------------------------ imbalance study
SAMPLERS = {
    "none": (None, False),
    "class_weight": (None, True),
    "random_over": (RandomOverSampler(random_state=SEED), False),
    "random_under": (RandomUnderSampler(random_state=SEED), False),
    "smote": (SMOTE(random_state=SEED, k_neighbors=5), False),
    "adasyn": (ADASYN(random_state=SEED, n_neighbors=5), False),
}
rows = []
for task in TASKS:
    for kind in ["logreg", "xgboost"]:
        for name, (smp, w) in SAMPLERS.items():
            try:
                r, _ = run(step3, task, kind, sampler=smp, weighted=w)
                rows.append({"task": task, "model": kind, "method": name,
                             **{k: r[k] for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "fpr", "fit_seconds"]}})
            except Exception as ex:  # e.g. ADASYN finds no hard neighbours
                rows.append({"task": task, "model": kind, "method": name, "error": str(ex)[:120]})
            print(rows[-1])
imb = pd.DataFrame(rows).round(4); imb.to_csv(OUT / "imbalance_comparison.csv", index=False)

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(figsize=(9, 8))
imp.head(30)[::-1].plot.barh(ax=ax, color="#1f77b4")
ax.set(title="Top-30 features by XGBoost gain (train split, y_detect)", xlabel="gain")
fig.tight_layout(); fig.savefig(FIG / "xgb_gain_top30.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, task in zip(axes, TASKS):
    p = fe[fe.task == task].pivot(index="feature_set", columns="model", values="pr_auc")
    p.loc[["raw_sensor_means", "all_engineered", "after_correlation", "selected"]].plot.bar(ax=ax, rot=15)
    ax.set(title=f"Validation PR-AUC — {task}", ylabel="PR-AUC", ylim=(0, 1.05))
fig.tight_layout(); fig.savefig(FIG / "fe_comparison_prauc.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, task in zip(axes, TASKS):
    p = imb[imb.task == task].pivot(index="method", columns="model", values="pr_auc")
    p.loc[[k for k in SAMPLERS if k in p.index]].plot.bar(ax=ax, rot=15)
    ax.set(title=f"Validation PR-AUC by imbalance method — {task}", ylabel="PR-AUC", ylim=(0, 1.05))
fig.tight_layout(); fig.savefig(FIG / "imbalance_prauc.png", dpi=110); plt.close(fig)

json.dump({"selected_features": step3, "after_correlation": step2, "selection_data": "train split, usable rows",
           "corr_threshold": 0.98, "top_k_by_xgb_gain": TOP},
          open(project_path("models/metadata/selected_features.json"), "w"), indent=2)
print(f"done in {time.time()-t0:.0f}s")
