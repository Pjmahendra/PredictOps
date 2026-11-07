"""Stage 6b — LSTM and 1D-CNN experiments (y_detect).

Train split -> train; validation split -> early stopping + threshold; test (event 4) once.
Run: .venv/bin/python scripts/train_dl.py   (~2–6 min on Apple MPS / CPU)
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from predictops.config import load_config, project_path
from predictops.evaluation.metrics import best_f1_threshold, classification_report_dict, event_level_report
from predictops.models.deep import CHANNELS, CNN1D, LSTMNet, make_sequences, predict, train_model

cfg = load_config()
OUT = project_path("reports/dl"); OUT.mkdir(parents=True, exist_ok=True)
FIG = project_path("reports/figures/dl"); FIG.mkdir(parents=True, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
m = pd.read_parquet(project_path("data/processed/features_1min.parquet"))
tr_m = m[m.split == "train"]
mean = tr_m.loc[tr_m.usable == 1, CHANNELS].mean().to_numpy(np.float32)       # train statistics only
std = tr_m.loc[tr_m.usable == 1, CHANNELS].std().to_numpy(np.float32) + 1e-6

Xtr, ytr, _ = make_sequences(tr_m, "y_detect", stride=5, mean=mean, std=std)    # stride 5 = less redundancy
Xva, yva, tva = make_sequences(m[m.split == "validation"], "y_detect", stride=1, mean=mean, std=std)
Xte, yte, tte = make_sequences(m[m.split == "test"], "y_detect", stride=1, mean=mean, std=std)
sizes = {"train": [len(ytr), int(ytr.sum())], "validation": [len(yva), int(yva.sum())], "test": [len(yte), int(yte.sum())]}
print("sequences (n, positives):", sizes, "device:", device)

rows, hists = [], {}
ev4 = [e for e in cfg["failure_events"] if e["id"] == 4]
for name, net in {"lstm": LSTMNet(), "cnn1d": CNN1D()}.items():
    t = time.time()
    hist = train_model(net, Xtr, ytr, Xva, yva, device)
    secs = time.time() - t
    hists[name] = pd.DataFrame(hist)
    sv = predict(net, Xva, device); thr = best_f1_threshold(yva, sv)           # threshold on validation
    st = predict(net, Xte, device)
    rv, rt = classification_report_dict(yva, sv, thr), classification_report_dict(yte, st, thr)
    evr = event_level_report(pd.Series(tte), st >= thr, ev4, len(yte) / 60)
    n_params = sum(p.numel() for p in net.parameters())
    rows.append({"model": name, "params": n_params, "epochs": len(hist), "train_seconds": round(secs, 1), "threshold": thr,
                 "val_pr_auc": rv["pr_auc"], "val_f1": rv["f1"], "test_pr_auc": rt["pr_auc"], "test_roc_auc": rt["roc_auc"],
                 "test_recall": rt["recall"], "test_precision": rt["precision"], "test_f1": rt["f1"],
                 "event4_detected": evr["events"][0]["detected"], "test_false_alarms_per_day": evr["false_alarms_per_day"]})
    torch.save(net.state_dict(), project_path(f"models/trained/dl_{name}.pt"))
    print(rows[-1])
res = pd.DataFrame(rows).round(4); res.to_csv(OUT / "dl_results.csv", index=False)
json.dump({"channels": CHANNELS, "seq_len": 60, "norm_mean": mean.tolist(), "norm_std": std.tolist(),
           "sequence_counts": sizes, "device": str(device), "results": rows},
          open(project_path("models/metadata/dl.json"), "w"), indent=2, default=str)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, h) in zip(axes, hists.items()):
    ax.plot(h.epoch, h.train_loss, label="train"); ax.plot(h.epoch, h.val_loss, label="validation")
    ax.set(title=f"{name}: loss per epoch (early stopping, patience 5)", xlabel="epoch", ylabel="weighted BCE"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "loss_curves.png", dpi=110); plt.close(fig)
