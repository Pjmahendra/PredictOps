"""Chronological, event-aware data splits.

Why not a random split?  Neighbouring rows (10 s apart) are almost identical; a random
split would put near-copies of the same minute in train and test and inflate results.
We split by TIME, and add an *embargo* before each boundary so no training row's
label or rolling window depends on data from the next split.

Splits (dates in configs/config.yaml -> splits):
    train       Feb 1  -> boundary 1   (contains event 1)
    validation  boundary 1 -> boundary 2 (contains events 2, 3)
    test        boundary 2 -> end      (contains event 4 — touched ONCE, at the end)

Event-blocked cross-validation (`event_folds`) is provided for model selection
inside the development period (train + validation).
"""
from __future__ import annotations

import pandas as pd

from predictops.config import load_config


def assign_split(df: pd.DataFrame, cfg: dict | None = None) -> pd.Series:
    cfg = cfg or load_config()
    s = cfg["splits"]
    b1, b2 = pd.Timestamp(s["validation_start"]), pd.Timestamp(s["test_start"])
    emb = pd.Timedelta(hours=s["embargo_hours"])
    split = pd.Series("train", index=df.index, dtype="object")
    split[df.timestamp >= b1] = "validation"
    split[df.timestamp >= b2] = "test"
    # embargo: drop the last `embargo_hours` before each boundary
    split[(df.timestamp >= b1 - emb) & (df.timestamp < b1)] = "embargo"
    split[(df.timestamp >= b2 - emb) & (df.timestamp < b2)] = "embargo"
    return split


def event_folds(df: pd.DataFrame, cfg: dict | None = None, event_ids=(2, 3), margin_days: int = 3):
    """Forward-chaining (expanding-window) folds over the development period.

    For each held-out event k, validation = [start_k - margin, end_k + margin] and
    training = every development row strictly BEFORE (validation start - embargo).
    The model therefore never sees data from after the period it is evaluated on.
    (Leave-one-event-out was rejected: it would train on June to evaluate May.)

    Event 1 cannot be a validation fold — nothing before it contains a failure.
    Yields (event_id, train_index, val_index)."""
    cfg = cfg or load_config()
    emb = pd.Timedelta(hours=cfg["splits"]["embargo_hours"])
    dev = df[df.split.isin(["train", "validation"])]
    for ev in cfg["failure_events"]:
        if ev["id"] not in event_ids:
            continue
        lo = pd.Timestamp(ev["start"]) - pd.Timedelta(days=margin_days)
        hi = pd.Timestamp(ev["end"]) + pd.Timedelta(days=margin_days)
        val = dev.timestamp.between(lo, hi)
        train = dev.timestamp < lo - emb
        yield ev["id"], dev.index[train], dev.index[val]
