"""Stage 18 — monthly drift report over the whole dataset vs the training reference.
Run: .venv/bin/python scripts/monitoring_report.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from predictops.anomaly.detectors import ANOMALY_FEATURES
from predictops.config import project_path
from predictops.inference.predictor import PredictOpsPredictor
from predictops.monitoring.drift import drift_report, psi

P = PredictOpsPredictor()
f = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
f = f[f.usable == 1]
ref = pd.read_parquet(project_path("models/metadata/reference_features.parquet"))
ref_s = pd.read_parquet(project_path("models/metadata/reference_scores.parquet")).anomaly_score.to_numpy()
thr = P.anomaly.thresholds["low"]
rows, per_feat = [], []
for month, cur in f.groupby(f.timestamp.dt.to_period("M")):
    if len(cur) < 500:
        continue
    rep = drift_report(ref, cur, ANOMALY_FEATURES)
    s = P.anomaly.score(cur)
    rows.append({"month": str(month), "minutes": len(cur), "features_major": int((rep.status == "major").sum()),
                 "features_moderate": int((rep.status == "moderate").sum()), "max_psi": rep.psi.max(),
                 "top_drifting_feature": rep.feature.iloc[0], "anomaly_flag_share": float((s >= thr).mean()),
                 "anomaly_score_psi": psi(ref_s, s), "split": cur.split.mode().iloc[0]})
    per_feat.append(rep.assign(month=str(month)))
out = pd.DataFrame(rows).round(4)
OUT = project_path("reports/monitoring"); OUT.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT / "monthly_drift.csv", index=False)
pf = pd.concat(per_feat); pf.round(4).to_csv(OUT / "monthly_feature_drift.csv", index=False)
pv = pf.pivot(index="feature", columns="month", values="psi")
fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(pv.clip(upper=1).to_numpy(), cmap="Reds", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index)
fig.colorbar(im, label="PSI (clipped at 1; >0.25 = major shift)")
ax.set_title("Feature drift per month vs training reference (Feb–May normal)")
FIG = project_path("reports/figures/monitoring"); FIG.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(FIG / "monthly_psi_heatmap.png", dpi=110)
print(out.to_string(index=False))
