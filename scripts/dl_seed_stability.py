"""Stage 6b add-on: repeat LSTM / 1D-CNN with different random seeds (result stability).
Run: .venv/bin/python scripts/dl_seed_stability.py"""
import numpy as np, pandas as pd, torch
from predictops.config import load_config, project_path
from predictops.evaluation.metrics import best_f1_threshold, classification_report_dict, event_level_report
from predictops.models.deep import CHANNELS, CNN1D, LSTMNet, make_sequences, predict, train_model

cfg = load_config(); dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
m = pd.read_parquet(project_path("data/processed/features_1min.parquet")); tr = m[m.split == "train"]
mean = tr.loc[tr.usable == 1, CHANNELS].mean().to_numpy(np.float32); std = tr.loc[tr.usable == 1, CHANNELS].std().to_numpy(np.float32) + 1e-6
Xtr, ytr, _ = make_sequences(tr, "y_detect", 5, mean, std)
Xva, yva, _ = make_sequences(m[m.split == "validation"], "y_detect", 1, mean, std)
Xte, yte, tte = make_sequences(m[m.split == "test"], "y_detect", 1, mean, std)
ev4 = [e for e in cfg["failure_events"] if e["id"] == 4]; rows = []
for seed in [1, 2, 3]:
    for name, cls in {"lstm": LSTMNet, "cnn1d": CNN1D}.items():
        torch.manual_seed(seed); net = cls()
        train_model(net, Xtr, ytr, Xva, yva, dev, seed=seed)
        sv, st = predict(net, Xva, dev), predict(net, Xte, dev); thr = best_f1_threshold(yva, sv)
        r = classification_report_dict(yte, st, thr); e = event_level_report(pd.Series(tte), st >= thr, ev4, len(yte) / 60)
        rows.append({"seed": seed, "model": name, "val_pr_auc": classification_report_dict(yva, sv)["pr_auc"],
                     "test_pr_auc": r["pr_auc"], "test_recall": r["recall"], "event4_detected": e["events"][0]["detected"],
                     "false_alarms_per_day": e["false_alarms_per_day"]})
        print(rows[-1])
pd.DataFrame(rows).round(4).to_csv(project_path("reports/dl/dl_seed_stability.csv"), index=False)
