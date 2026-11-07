"""Stage 6d — sensor normal ranges + health engine evaluation on validation and test.

Normal ranges = 0.5% / 99.5% quantiles of TRAIN normal minutes (no labels used beyond 'normal').
Run: .venv/bin/python scripts/evaluate_health.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from predictops.config import project_path
from predictops.engine.health import CHECKS
from predictops.evaluation.metrics import event_level_report
from predictops.config import load_config

cfg = load_config()
m = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
normal = m[(m.split == "train") & (m.usable == 1) & (m.y_detect == 0)]
ranges = {f: {"p005": float(normal[f].quantile(0.005)), "p995": float(normal[f].quantile(0.995)),
              "median": float(normal[f].median())} for f in CHECKS}
(project_path("models/metadata/sensor_ranges.json")).write_text(json.dumps(ranges, indent=2))

from predictops.inference.predictor import PredictOpsPredictor  # needs sensor_ranges.json
P = PredictOpsPredictor()
OUT = project_path("reports/health"); OUT.mkdir(parents=True, exist_ok=True)
FIG = project_path("reports/figures/health"); FIG.mkdir(parents=True, exist_ok=True)
evs = {e["id"]: e for e in cfg["failure_events"]}
rows, timelines = [], {}
for split, ids in [("validation", [2, 3]), ("test", [4])]:
    d = m[(m.split == split) & (m.usable == 1)]
    r = P.predict_features(d)
    r["truth"] = d.y_detect.to_numpy()
    timelines[split] = r
    for truth, lab in [(0, "normal minutes"), (1, "failure minutes")]:
        s = r[r.truth == truth]
        rows.append({"split": split, "group": lab, "minutes": len(s),
                     **{f"status_{k}": round((s.health_status == k).mean(), 4) for k in ["Healthy", "Warning", "Critical"]},
                     **{f"risk_{k}": round((s.failure_risk == k).mean(), 4) for k in ["Low", "Medium", "High"]},
                     "mean_health_score": round(s.health_score.mean(), 1)})
    alarm = r.health_status.isin(["Critical"]).to_numpy()
    e = event_level_report(r.timestamp, alarm, [evs[i] for i in ids], len(r) / 60)
    rows.append({"split": split, "group": "CRITICAL as alarm (event level)",
                 "minutes": f"{e['detected']}/{e['n_events']} events; {e['false_alarms_per_day']:.2f} false alarms/day"})
    alarm = (r.health_status != "Healthy").to_numpy()
    e = event_level_report(r.timestamp, alarm, [evs[i] for i in ids], len(r) / 60)
    rows.append({"split": split, "group": "WARNING+ as alarm (event level)",
                 "minutes": f"{e['detected']}/{e['n_events']} events; {e['false_alarms_per_day']:.2f} false alarms/day"})
res = pd.DataFrame(rows); res.to_csv(OUT / "health_status_by_truth.csv", index=False)
print(res.to_string(index=False))

r = timelines["test"]; s0, e0 = pd.Timestamp(evs[4]["start"]), pd.Timestamp(evs[4]["end"])
w = r.timestamp.between(s0 - pd.Timedelta(days=2), e0 + pd.Timedelta(hours=12))
fig, ax = plt.subplots(figsize=(14, 3.8))
ax.plot(r.timestamp[w], r.health_score[w], lw=.8, c="#1f77b4")
for c, (lo, hi) in {"#2ca02c": (75, 100), "#ff7f0e": (45, 75), "#d62728": (0, 45)}.items():
    ax.axhspan(lo, hi, color=c, alpha=.07)
ax.axvspan(s0, e0, color="#d62728", alpha=.2, label="event 4 (reported)")
ax.set(title="Health score around event 4 (test). Bands: Healthy ≥75, Warning 45–75, Critical <45", ylabel="health score", ylim=(0, 102))
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "health_event4.png", dpi=110); plt.close(fig)
print(r.loc[r.timestamp.between(s0, e0)].iloc[-30][["timestamp", "health_score", "health_status", "failure_risk", "reasons", "recommendations"]].to_dict())
