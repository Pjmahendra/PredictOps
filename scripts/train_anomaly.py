"""Stage 6 — anomaly detection: fit on known-normal data, evaluate against failure windows.

Data (no failure labels used for fitting or thresholds):
  fit   : train split, usable, y_detect==0, before 2020-04-25
  calib : train split, usable, y_detect==0, 2020-04-25 -> end of train (thresholds = score quantiles)
Evaluation: validation (events 2,3) for SELECTION; test (event 4) evaluated once.
Alerts: flagged minute = score >= 'low' threshold (99.5% normal quantile);
        alert = persistent (>=10 flagged minutes in trailing 30 min), merged within 30 min.
Run: .venv/bin/python scripts/train_anomaly.py
"""
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

from predictops.anomaly.detectors import AnomalyDetector, persistent_mask
from predictops.config import load_config, project_path
from predictops.evaluation.metrics import classification_report_dict, event_level_report

warnings.filterwarnings("ignore")
cfg = load_config()
OUT = project_path("reports/anomaly"); OUT.mkdir(parents=True, exist_ok=True)
FIG = project_path("reports/figures/anomaly"); FIG.mkdir(parents=True, exist_ok=True)
CAL = pd.Timestamp("2020-04-25")
t0 = time.time()

m = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
normal = m[(m.split == "train") & (m.usable == 1) & (m.y_detect == 0)]
fit_df, cal_df = normal[normal.timestamp < CAL], normal[normal.timestamp >= CAL]
DETECTORS = {
    "isolation_forest": AnomalyDetector("isolation_forest"),
    "lof": AnomalyDetector("lof", max_train_rows=30_000),
    "ocsvm": AnomalyDetector("ocsvm", max_train_rows=10_000),
    "robust_z": AnomalyDetector("robust_z"),
}
evs = {e["id"]: e for e in cfg["failure_events"]}
SETS = {"validation": ([2, 3], m[(m.split == "validation") & (m.usable == 1)]),
        "test": ([4], m[(m.split == "test") & (m.usable == 1)])}
# unreported episodes (uncertain) live in the train period; scored for information only
unc = m[m.uncertain == 1]

rows, scores_store = [], {}
for name, det in DETECTORS.items():
    t = time.time(); det.fit(fit_df, cal_df); fit_s = time.time() - t
    for sname, (ev_ids, d) in SETS.items():
        t = time.time(); s = det.score(d); score_s = time.time() - t
        flag = s >= det.thresholds["low"]
        alert = persistent_mask(d.timestamp, flag)
        r = classification_report_dict(d.y_detect, s, det.thresholds["low"])
        evr = event_level_report(d.timestamp, alert, [evs[i] for i in ev_ids], len(d) / 60, lookback_h=12)
        rows.append({"detector": name, "set": sname, "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
                     "minute_recall": r["recall"], "minute_precision": r["precision"], "flagged_normal_share": r["fpr"],
                     "events_detected": f"{evr['detected']}/{evr['n_events']}",
                     "lead_times_min": [None if pd.isna(e["lead_time_min"]) else round(e["lead_time_min"]) for e in evr["events"]],
                     "false_alarm_episodes": evr["false_alarm_episodes"],
                     "false_alarms_per_day": round(evr["false_alarms_per_day"], 3),
                     "fit_s": round(fit_s, 1), "score_ms_per_row": round(score_s / len(d) * 1000, 4)})
        scores_store[(name, sname)] = (d.timestamp.values, s, alert)
        print(rows[-1])
    us = det.score(unc)
    rows.append({"detector": name, "set": "unreported_episodes", "minute_recall": float((us >= det.thresholds["low"]).mean())})
res = pd.DataFrame(rows); res.to_csv(OUT / "anomaly_results.csv", index=False)

# ---- selection on VALIDATION only: most events detected, then fewest false alarms/day, then PR-AUC
v = res[res.set == "validation"].copy()
v["n_det"] = v.events_detected.str.split("/").str[0].astype(int)
v = v.sort_values(["n_det", "false_alarms_per_day", "pr_auc"], ascending=[False, True, False])
chosen = v.iloc[0].detector
print("SELECTED anomaly detector (validation):", chosen)

MOD = project_path("models/trained")
for name, det in DETECTORS.items():
    joblib.dump(det, MOD / f"anomaly_{name}.joblib")
(project_path("models/metadata") / "anomaly.json").write_text(json.dumps({
    "selected_detector": chosen, "selection": "validation: events detected, then false alarms/day, then PR-AUC",
    "features": DETECTORS[chosen].features, "thresholds": DETECTORS[chosen].thresholds,
    "fit_period": [str(fit_df.timestamp.min()), str(fit_df.timestamp.max())], "fit_rows": len(fit_df),
    "calibration_period": [str(cal_df.timestamp.min()), str(cal_df.timestamp.max())],
    "alert_rule": "persistent: >=10 flagged minutes in trailing 30 min; episodes merged within 30 min, min 5 min",
    "model_version": f"anomaly-{chosen}-v1-{datetime.now(timezone.utc):%Y%m%d}",
    "artifact": f"models/trained/anomaly_{chosen}.joblib",
    "results": res[res.detector == chosen].to_dict(orient="records")}, indent=2, default=str))

# ---- figures: score timelines around each event
for sname, (ev_ids, _) in SETS.items():
    for eid in ev_ids:
        e = evs[eid]; s0, e0 = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        fig, axes = plt.subplots(len(DETECTORS), 1, figsize=(14, 2.3 * len(DETECTORS)), sharex=True)
        for ax, (name, det) in zip(axes, DETECTORS.items()):
            ts, s, alert = scores_store[(name, sname)]
            ts = pd.to_datetime(ts); w = (ts >= s0 - pd.Timedelta(days=2)) & (ts <= e0 + pd.Timedelta(hours=12))
            ax.plot(ts[w], s[w], lw=.6)
            ax.axhline(det.thresholds["low"], c="orange", ls="--", lw=.8)
            ax.axhline(det.thresholds["high"], c="red", ls="--", lw=.8)
            ax.fill_between(ts[w], 0, 1, where=alert[w], transform=ax.get_xaxis_transform(), color="orange", alpha=.25)
            ax.axvspan(s0, e0, color="#d62728", alpha=.15)
            ax.set_ylabel(name, rotation=0, ha="right")
        axes[0].set_title(f"Anomaly scores around event {eid} ({sname}). Dashed: low/high thresholds; orange band: alerts; red: reported window")
        fig.tight_layout(); fig.savefig(FIG / f"scores_event{eid}.png", dpi=105); plt.close(fig)

# full test-period timeline of the selected detector (false alarms visible)
ts, s, alert = scores_store[(chosen, "test")]
fig, ax = plt.subplots(figsize=(15, 3.5))
ax.plot(pd.to_datetime(ts), s, lw=.4); ax.axhline(DETECTORS[chosen].thresholds["low"], c="orange", ls="--")
ax.fill_between(pd.to_datetime(ts), 0, 1, where=alert, transform=ax.get_xaxis_transform(), color="orange", alpha=.3)
ax.axvspan(pd.Timestamp(evs[4]["start"]), pd.Timestamp(evs[4]["end"]), color="#d62728", alpha=.3)
ax.set(title=f"{chosen}: anomaly score over the whole test period (Jul–Aug). Orange = alerts, red = event 4")
fig.tight_layout(); fig.savefig(FIG / "test_period_timeline.png", dpi=105); plt.close(fig)
print(f"done in {time.time()-t0:.0f}s")
