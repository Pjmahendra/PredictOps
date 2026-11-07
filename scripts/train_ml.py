"""Stage 5 — supervised ML: tuning, selection, single final test.

Protocol
  1. Tuning: for each model, sample N_ITER hyperparameter sets; score each by mean PR-AUC
     over the two forward-chaining folds (A: validate on event 2, B: validate on event 3).
  2. Threshold: from the best config's OUT-OF-FOLD validation predictions (max F1).
  3. Model selection: by out-of-fold PR-AUC, then MCC — written to disk BEFORE testing.
  4. Final fit: on all development rows (train + validation, events 1–3).
  5. Test: evaluate every final model ONCE on the test split (event 4) — row + event level.
  6. Early warning (y_warn_2h/6h/12h): train on dev (onsets 2, 3), test on event 4 — reported as is.

Outputs: models/trained/*.joblib, models/metadata/ml_*.json, reports/ml/*.csv,
         reports/figures/ml/*.png, data/processed/test_predictions.parquet, mlruns/ (MLflow)
Run:  .venv/bin/python scripts/train_ml.py   (~5–15 min)
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import ParameterSampler

from predictops.config import load_config, project_path
from predictops.evaluation.metrics import best_f1_threshold, classification_report_dict, event_level_report
from predictops.models.zoo import N_ITER, SEARCH_SPACES, SEED, SVM_MAX_ROWS, make_pipeline
from predictops.preprocessing.splits import event_folds

warnings.filterwarnings("ignore")
cfg = load_config()
OUT = project_path("reports/ml"); OUT.mkdir(parents=True, exist_ok=True)
FIG = project_path("reports/figures/ml"); FIG.mkdir(parents=True, exist_ok=True)
MOD = project_path("models/trained"); MOD.mkdir(parents=True, exist_ok=True)
META = project_path("models/metadata")
FEATS = json.load(open(META / "selected_features.json"))["selected_features"]
Y = "y_detect"
t_start = time.time()

m = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
m = m.reset_index(drop=True)
folds = {e: (tr, va) for e, tr, va in event_folds(m, cfg)}
usable = m.usable == 1
dataset_hash = hashlib.sha256(open(project_path(cfg["paths"]["raw_csv"]), "rb").read()).hexdigest()[:16]


def subsample_for_svm(df, y, rng):
    if len(df) <= SVM_MAX_ROWS:
        return df
    pos = df[df[y] == 1]
    neg = df[df[y] == 0].sample(SVM_MAX_ROWS - len(pos), random_state=rng)
    return pd.concat([pos, neg]).sort_index()


def fit(name, params, train_df, y=Y):
    if name == "svm":
        train_df = subsample_for_svm(train_df, y, SEED)
    pw = float((train_df[y] == 0).sum() / max((train_df[y] == 1).sum(), 1))
    pipe = make_pipeline(name, params, FEATS, pw)
    t = time.time()
    pipe.fit(train_df[FEATS], train_df[y].astype(int))
    return pipe, time.time() - t


# ------------------------------------------------------------------ 1. tuning
tuning_rows, best = [], {}
for name, space in SEARCH_SPACES.items():
    cands = list(ParameterSampler(space, n_iter=min(N_ITER[name], np.prod([len(v) for v in space.values()])),
                                  random_state=SEED))
    for params in cands:
        scores, oof = [], []
        for ev, (tr, va) in folds.items():
            trd = m.loc[tr][usable.loc[tr]]
            vad = m.loc[va][usable.loc[va]]
            pipe, secs = fit(name, params, trd)
            s = pipe.predict_proba(vad[FEATS])[:, 1]
            r = classification_report_dict(vad[Y], s)
            scores.append(r["pr_auc"]); oof.append(pd.DataFrame({"idx": vad.index, "y": vad[Y].values, "score": s}))
        row = {"model": name, "params": json.dumps(params, default=str), "pr_auc_foldA": scores[0],
               "pr_auc_foldB": scores[1], "pr_auc_mean": float(np.mean(scores))}
        tuning_rows.append(row)
        if name not in best or row["pr_auc_mean"] > best[name]["pr_auc_mean"]:
            best[name] = {**row, "params_obj": params, "oof": pd.concat(oof)}
    print(f"tuned {name}: best mean PR-AUC={best[name]['pr_auc_mean']:.4f} {best[name]['params']}")
pd.DataFrame(tuning_rows).round(4).to_csv(OUT / "tuning_results.csv", index=False)

# ------------------------------------------------------------------ 2+3. threshold & selection (no test data)
sel_rows = []
for name, b in best.items():
    oof = b["oof"]
    thr = best_f1_threshold(oof.y, oof.score)
    r = classification_report_dict(oof.y, oof.score, thr)
    b["threshold"] = thr
    sel_rows.append({"model": name, "oof_pr_auc": r["pr_auc"], "oof_mcc": r["mcc"], "oof_f1": r["f1"],
                     "oof_recall": r["recall"], "oof_precision": r["precision"], "threshold": thr,
                     "params": b["params"]})
sel = pd.DataFrame(sel_rows).sort_values(["oof_pr_auc", "oof_mcc"], ascending=False).round(4)
sel.to_csv(OUT / "validation_selection.csv", index=False)
chosen = sel.iloc[0]["model"]
(META / "ml_selection.json").write_text(json.dumps({
    "selected_model": chosen, "criterion": "out-of-fold PR-AUC (forward folds A,B), tie-break MCC",
    "decided_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "decided_before_test": True, "validation_table": sel.to_dict(orient="records")}, indent=2, default=str))
print("SELECTED (before test):", chosen)

# ------------------------------------------------------------------ 4+5. final fit on dev, single test
dev = m[m.split.isin(["train", "validation"]) & usable]
test = m[(m.split == "test") & usable].copy()
test_hours = len(test) / 60
ev4 = [e for e in cfg["failure_events"] if e["id"] == 4]

try:
    import mlflow
    mlflow.set_tracking_uri("sqlite:///" + str(project_path("mlruns.db")))
    mlflow.set_experiment("predictops-failure-detection")
except Exception as ex:  # tracking is optional; training must not fail because of it
    mlflow = None
    print("MLflow disabled:", ex)

test_rows, preds = [], {"timestamp": test.timestamp.values, "y_detect": test[Y].values}
for name, b in best.items():
    pipe, secs = fit(name, b["params_obj"], dev)
    t = time.time(); s = pipe.predict_proba(test[FEATS])[:, 1]; infer_ms = (time.time() - t) / len(test) * 1000
    r = classification_report_dict(test[Y], s, b["threshold"])
    evr = event_level_report(test.timestamp, s >= b["threshold"], ev4, test_hours, lookback_h=12)
    e4 = evr["events"][0]
    row = {"model": name, **{k: r[k] for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "balanced_accuracy",
                                                "mcc", "cohen_kappa", "accuracy", "brier", "fpr", "fnr", "tp", "fp", "fn", "tn"]},
           "threshold": b["threshold"], "event4_detected": e4["detected"], "event4_lead_time_min": e4["lead_time_min"],
           "false_alarm_episodes": evr["false_alarm_episodes"], "false_alarms_per_day": evr["false_alarms_per_day"],
           "train_seconds": round(secs, 1), "inference_ms_per_row": round(infer_ms, 4), "selected": name == chosen}
    test_rows.append(row); preds[f"score_{name}"] = s
    version = f"{name}-v1-{datetime.now(timezone.utc):%Y%m%d}"
    joblib.dump(pipe, MOD / f"detect_{name}.joblib")
    (META / f"ml_detect_{name}.json").write_text(json.dumps({
        "model_version": version, "model": name, "task": "y_detect (failure in progress)", "features": FEATS,
        "params": b["params_obj"], "threshold": b["threshold"], "trained_on": "train+validation usable rows (events 1-3)",
        "train_period": [str(dev.timestamp.min()), str(dev.timestamp.max())], "train_rows": len(dev),
        "dataset_sha256_16": dataset_hash, "validation_oof_pr_auc": float(sel.set_index("model").loc[name, "oof_pr_auc"]),
        "test_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in row.items()},
        "artifact": f"models/trained/detect_{name}.joblib", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, default=str))
    if mlflow:
        with mlflow.start_run(run_name=version):
            mlflow.log_params({**{f"hp_{k}": v for k, v in b["params_obj"].items()}, "model": name, "n_features": len(FEATS),
                               "threshold": round(b["threshold"], 4), "dataset_sha256_16": dataset_hash})
            mlflow.log_metrics({"val_oof_pr_auc": float(sel.set_index("model").loc[name, "oof_pr_auc"]),
                                **{f"test_{k}": float(row[k]) for k in ["pr_auc", "roc_auc", "f1", "recall", "precision", "mcc", "false_alarms_per_day"]}})
            mlflow.log_artifact(str(MOD / f"detect_{name}.joblib"))
    print(f"test {name}: PR-AUC={r['pr_auc']:.3f} F1={r['f1']:.3f} rec={r['recall']:.3f} prec={r['precision']:.3f} "
          f"ev4={e4['detected']} lead={e4['lead_time_min']} FA/day={evr['false_alarms_per_day']:.2f}")
res = pd.DataFrame(test_rows).round(4); res.to_csv(OUT / "test_results_detect.csv", index=False)
pd.DataFrame(preds).to_parquet(project_path("data/processed/test_predictions.parquet"), index=False)

# ------------------------------------------------------------------ 6. early warning
warn_rows = []
for h in cfg["targets"]["warning_horizons_hours"]:
    y = f"y_warn_{h}h"
    d = dev[dev[y].notna()]; te = test[test[y].notna()]
    for name in ["logreg", "xgboost"]:
        pipe, _ = fit(name, best[name]["params_obj"], d, y=y)
        s = pipe.predict_proba(te[FEATS])[:, 1]
        r = classification_report_dict(te[y], s)
        warn_rows.append({"horizon_h": h, "model": name, "train_onsets": "events 2, 3", "test_onset": "event 4",
                          "test_positives": int(te[y].sum()), "test_rows": len(te),
                          "random_pr_auc": round(te[y].mean(), 4), "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"]})
warn = pd.DataFrame(warn_rows).round(4); warn.to_csv(OUT / "test_results_warning.csv", index=False)
print(warn.to_string(index=False))

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(figsize=(11, 4))
res.set_index("model")[["pr_auc", "f1", "recall", "precision", "mcc"]].plot.bar(ax=ax, rot=0)
ax.set(title="Test (event 4) — failure-in-progress detection", ylim=(0, 1.05)); ax.legend(ncol=5, loc="lower center")
fig.tight_layout(); fig.savefig(FIG / "test_comparison.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, b) in zip(axes, best.items()):
    ConfusionMatrixDisplay.from_predictions(test[Y], preds[f"score_{name}"] >= b["threshold"], ax=ax,
                                            display_labels=["normal", "failure"], colorbar=False)
    ax.set_title(f"{name} (thr={b['threshold']:.2f})")
fig.tight_layout(); fig.savefig(FIG / "test_confusion_matrices.png", dpi=110); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name in best:
    RocCurveDisplay.from_predictions(test[Y], preds[f"score_{name}"], ax=axes[0], name=name)
    PrecisionRecallDisplay.from_predictions(test[Y], preds[f"score_{name}"], ax=axes[1], name=name)
axes[0].set_title("ROC — test"); axes[1].set_title("Precision-Recall — test")
fig.tight_layout(); fig.savefig(FIG / "test_roc_pr.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))
for name, b in best.items():
    pt, pp = calibration_curve(b["oof"].y, b["oof"].score, n_bins=10, strategy="quantile")
    ax.plot(pp, pt, marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set(title="Calibration (out-of-fold validation)", xlabel="predicted", ylabel="observed")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "calibration_oof.png", dpi=110); plt.close(fig)

s4, e4t = pd.Timestamp(ev4[0]["start"]), pd.Timestamp(ev4[0]["end"])
w = (test.timestamp >= s4 - pd.Timedelta(days=1)) & (test.timestamp <= e4t + pd.Timedelta(hours=12))
fig, ax = plt.subplots(figsize=(14, 4))
for name in best:
    ax.plot(test.timestamp[w], preds[f"score_{name}"][w.values], lw=.8, label=name)
ax.axvspan(s4, e4t, color="#d62728", alpha=.2, label="event 4 window")
ax.set(title="Predicted failure score around event 4 (test)", ylabel="score"); ax.legend(ncol=6)
fig.tight_layout(); fig.savefig(FIG / "test_event4_scores.png", dpi=110); plt.close(fig)

print(f"done in {(time.time()-t_start)/60:.1f} min")
