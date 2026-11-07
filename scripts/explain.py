"""Stage 6c — SHAP explainability (global + local) for the selected models.

Run: .venv/bin/python scripts/explain.py   (~1–2 min)
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from predictops.config import load_config, project_path
from predictops.explainability.shap_explain import AnomalyShapExplainer, TreeShapExplainer

cfg = load_config()
FIG = project_path("reports/figures/shap"); FIG.mkdir(parents=True, exist_ok=True)
OUT = project_path("reports/shap"); OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(project_path("models/metadata/ml_detect_xgboost.json")))
FEATS = meta["features"]
pipe = joblib.load(project_path("models/trained/detect_xgboost.joblib"))
m = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
val = m[(m.split == "validation") & (m.usable == 1)]
test = m[(m.split == "test") & (m.usable == 1)]


def save(name):
    plt.gcf().set_size_inches(10, 6); plt.tight_layout(); plt.savefig(FIG / name, dpi=110, bbox_inches="tight"); plt.close("all")


# ---------------- global (validation sample: all failure minutes + 3000 normal minutes, seeded)
sample = pd.concat([val[val.y_detect == 1].sample(min(1500, int(val.y_detect.sum())), random_state=0),
                    val[val.y_detect == 0].sample(3000, random_state=0)])
tx = TreeShapExplainer(pipe)
ex = tx.explain(sample[FEATS])
shap.plots.beeswarm(ex, max_display=15, show=False); plt.title("SHAP summary — XGBoost detector (validation sample)"); save("xgb_summary_beeswarm.png")
shap.plots.bar(ex, max_display=15, show=False); plt.title("Mean |SHAP| — global feature importance"); save("xgb_global_bar.png")
glob = pd.Series(np.abs(ex.values).mean(0), index=ex.feature_names).sort_values(ascending=False)
glob.to_csv(OUT / "xgb_global_mean_abs_shap.csv", header=["mean_abs_shap"])

# ---------------- local: high-risk (event 3), normal, and event-4 (missed) minutes
cases = {
    "high_risk_event3": val[val.event_id == 3].iloc[[len(val[val.event_id == 3]) // 2]],
    "normal_minute": val[(val.y_detect == 0) & (val.timestamp.dt.hour == 10)].iloc[[100]],
    "event4_missed": test[test.event_id == 4].iloc[[-30]],
}
local = {}
for name, row in cases.items():
    e = tx.explain(row[FEATS])
    score = float(pipe.predict_proba(row[FEATS])[:, 1][0])
    shap.plots.waterfall(e[0], max_display=10, show=False)
    plt.title(f"{name} @ {row.timestamp.iloc[0]} — P(failure)={score:.3f}"); save(f"xgb_waterfall_{name}.png")
    local[name] = {"timestamp": str(row.timestamp.iloc[0]), "score": score, **tx.contributions(row[FEATS])[0]}

# ---------------- anomaly detector (OCSVM) KernelSHAP on event-4 minutes vs normal
amet = json.load(open(project_path("models/metadata/anomaly.json")))
det = joblib.load(project_path(f"models/trained/anomaly_{amet['selected_detector']}.joblib"))
normal_bg = m[(m.split == "train") & (m.usable == 1) & (m.y_detect == 0)]
ax = AnomalyShapExplainer(det, normal_bg)
ev4 = test[test.event_id == 4]
pts = pd.concat([ev4.iloc[[len(ev4) // 2, -30]], test[(test.y_detect == 0) & (test.timestamp.dt.hour == 10)].iloc[[200]]])
aex = ax.explain(pts)
for i, lbl in enumerate(["event4_mid", "event4_late", "normal"]):
    shap.plots.waterfall(aex[i], max_display=10, show=False)
    plt.title(f"Anomaly score explanation (OCSVM) — {lbl} @ {pts.timestamp.iloc[i]}"); save(f"anomaly_waterfall_{lbl}.png")
    order = np.argsort(-np.abs(aex.values[i]))[:5]
    local[f"anomaly_{lbl}"] = {"timestamp": str(pts.timestamp.iloc[i]), "score": float(det.score(pts.iloc[[i]])[0]),
                               "threshold_low": det.thresholds["low"],
                               "contributions": [{"feature": det.features[j], "value": float(aex.data[i, j]),
                                                  "shap": float(aex.values[i, j])} for j in order]}
json.dump(local, open(OUT / "local_explanations.json", "w"), indent=2)
print(glob.head(10).round(3).to_string())
print(json.dumps(local, indent=1)[:3500])
