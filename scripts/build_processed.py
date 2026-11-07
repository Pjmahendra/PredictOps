"""Stage 4b — preprocessing, target construction, splits.

Inputs : data/intermediate/metropt3.parquet (from build_intermediate.py)
Outputs: data/processed/metropt3_labeled.parquet   all rows + flags + targets + split
         data/{train,validation,test}/sensors.parquet
         data/sample/*.csv                          exact raw-format excerpts for demo/upload
         models/preprocessing/base_preprocessor.joblib  (fitted on TRAIN usable rows only)
         models/metadata/preprocessing.json
         reports/preprocessing_stats/*.csv          tables used in docs/03

Run:  .venv/bin/python scripts/build_processed.py
"""
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

from predictops.config import load_config, project_path
from predictops.data.ingestion import load_intermediate
from predictops.preprocessing.labels import build_labeled
from predictops.preprocessing.pipeline import make_preprocessor
from predictops.preprocessing.splits import assign_split, event_folds

t0 = time.time()
cfg = load_config()
H = cfg["targets"]["warning_horizons_hours"]
TAB = project_path("reports/preprocessing_stats"); TAB.mkdir(parents=True, exist_ok=True)

df = build_labeled(load_intermediate(), cfg)
df["split"] = assign_split(df, cfg)
out = project_path(cfg["paths"]["processed_dir"]) / "metropt3_labeled.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)

for s in ["train", "validation", "test"]:
    d = project_path(cfg["paths"][f"{s}_dir"]); d.mkdir(parents=True, exist_ok=True)
    df[df.split == s].to_parquet(d / "sensors.parquet", index=False)

# ---------------- summary tables
rows = []
for s in ["train", "embargo", "validation", "test"]:
    d = df[df.split == s]
    u = d[d.usable == 1]
    r = {"split": s, "from": d.timestamp.min(), "to": d.timestamp.max(), "rows": len(d),
         "frozen": int(d.is_frozen.sum()), "uncertain": int(d.uncertain.sum()), "usable": len(u),
         "events": sorted(set(d.event_id[d.event_id > 0].tolist())),
         "y_detect_pos": int(u.y_detect.sum()), "y_detect_rate": round(u.y_detect.mean(), 4)}
    for h in H:
        r[f"y_warn_{h}h_pos"] = int(u[f"y_warn_{h}h"].fillna(0).sum())
    rows.append(r)
split_tab = pd.DataFrame(rows); split_tab.to_csv(TAB / "split_summary.csv", index=False)

ev_rows = []
for ev in cfg["failure_events"]:
    s = pd.Timestamp(ev["start"])
    for h in H:
        m = (df[f"y_warn_{h}h"] == 1) & (df.minutes_to_onset <= h * 60) & (df.timestamp < s) & \
            (df.timestamp >= s - pd.Timedelta(hours=h))
        ev_rows.append({"event": ev["id"], "horizon_h": h, "positive_rows": int(m.sum()),
                        "usable_positive_rows": int((m & (df.usable == 1)).sum())})
ev_tab = pd.DataFrame(ev_rows).pivot(index="event", columns="horizon_h", values="usable_positive_rows")
ev_tab.columns = [f"usable y_warn_{c}h rows" for c in ev_tab.columns]
ev_tab.to_csv(TAB / "warning_positives_per_event.csv")

folds = [{"held_out_event": e, "train_rows": len(tr), "val_rows": len(va),
          "val_from": df.loc[va, "timestamp"].min(), "val_to": df.loc[va, "timestamp"].max()}
         for e, tr, va in event_folds(df, cfg)]
pd.DataFrame(folds).to_csv(TAB / "event_folds.csv", index=False)

# ---------------- base preprocessor: fit on TRAIN usable rows only
analog = [c for c in cfg["dataset"]["analog_sensors"] if c != "Reservoirs"]  # Reservoirs ≡ TP3
digital = [c for c in cfg["dataset"]["digital_sensors"] if c not in ("MPG",)]  # MPG ≈ COMP (r=0.985)
pre = make_preprocessor(analog, digital)
train_rows = df[(df.split == "train") & (df.usable == 1)]
pre.fit(train_rows[analog + digital])
mdir = project_path(cfg["paths"]["models_dir"])
(mdir / "preprocessing").mkdir(parents=True, exist_ok=True); (mdir / "metadata").mkdir(parents=True, exist_ok=True)
joblib.dump(pre, mdir / "preprocessing" / "base_preprocessor.joblib")
scaler = pre.named_transformers_["num"].named_steps["scale"]
meta = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "fitted_on": "split == 'train' & usable == 1", "fit_rows": len(train_rows),
    "fit_period": [str(train_rows.timestamp.min()), str(train_rows.timestamp.max())],
    "numeric_features": analog, "binary_features": digital,
    "dropped_as_redundant": {"Reservoirs": "r=1.000 with TP3", "MPG": "r=0.985 with COMP"},
    "scaler": "StandardScaler (mean / std); RobustScaler rejected — IQR≈0.004 for TP2, DV_pressure", "imputer": "median (numeric), most_frequent (binary)",
    "scaler_mean": dict(zip(analog, map(float, scaler.mean_))),
    "scaler_scale": dict(zip(analog, map(float, scaler.scale_))),
    "warning_horizons_hours": H, "splits": cfg["splits"],
}
(mdir / "metadata" / "preprocessing.json").write_text(json.dumps(meta, indent=2))

# ---------------- exact raw-format sample files (real rows, copied from the raw CSV text)
raw = pd.read_csv(project_path(cfg["paths"]["raw_csv"]), dtype=str)
sd = project_path(cfg["paths"]["sample_dir"]); sd.mkdir(parents=True, exist_ok=True)
samples = {
    "event4_2020-07-15.csv": ("2020-07-15 00:00:00", "2020-07-16 00:00:00"),   # test period, contains event 4
    "normal_day_2020-08-10.csv": ("2020-08-10 00:00:00", "2020-08-11 00:00:00"),  # test period, no event
    "small_2h_2020-08-10.csv": ("2020-08-10 08:00:00", "2020-08-10 10:00:00"),   # quick demo / tests
}
for name, (a, b) in samples.items():
    raw[(raw.timestamp >= a) & (raw.timestamp < b)].to_csv(sd / name, index=False)

print(split_tab.to_string(index=False))
print(ev_tab.to_string())
print(pd.DataFrame(folds).to_string(index=False))
print(f"done in {time.time()-t0:.0f}s")
