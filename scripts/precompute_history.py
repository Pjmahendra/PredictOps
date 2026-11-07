"""Run the deployed inference pipeline over the WHOLE dataset (for historical replay,
the dashboard timelines and monitoring), and save the drift reference sample.

Note: Feb–Jun minutes were used for training/validation — the dashboard labels them
('in-sample'); only Jul–Aug (test split) are genuinely unseen.
Run: .venv/bin/python scripts/precompute_history.py   (~1–2 min)
"""
import time

import pandas as pd

from predictops.anomaly.detectors import ANOMALY_FEATURES
from predictops.config import project_path
from predictops.inference.predictor import PredictOpsPredictor

t0 = time.time()
P = PredictOpsPredictor()
m = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
res = P.predict_features(m)
keep = ["split", "usable", "is_frozen", "uncertain", "y_detect", "event_id", "segment_id"]
res = res.merge(m[["timestamp"] + keep], on="timestamp", how="left")
res["data_role"] = res.split.map({"train": "in-sample (training)", "embargo": "in-sample (embargo)",
                                  "validation": "in-sample (validation)", "test": "unseen (test)"})
res["reasons"] = res.reasons.map(lambda x: " | ".join(x))
res["recommendations"] = res.recommendations.map(lambda x: " | ".join(x))
res.drop(columns=["_row"]).to_parquet(project_path("data/processed/history_predictions.parquet"), index=False)

ref = m[(m.split == "train") & (m.usable == 1) & (m.y_detect == 0)]
cols = sorted(set(ANOMALY_FEATURES) | set(P.det_features))
ref[cols].sample(20_000, random_state=42).to_parquet(project_path("models/metadata/reference_features.parquet"), index=False)
ref_scores = P.anomaly.score(ref.sample(20_000, random_state=42))
pd.DataFrame({"anomaly_score": ref_scores}).to_parquet(project_path("models/metadata/reference_scores.parquet"), index=False)
print(f"history: {len(res):,} minutes in {time.time()-t0:.0f}s;", res.health_status.value_counts().to_dict())
