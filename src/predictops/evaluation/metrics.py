"""Evaluation metrics shared by every experiment.

Row-level (classification) metrics + EVENT-level metrics. With only a handful of
failure events, event-level numbers (was each event caught? how early? how many
false alarms per day?) are what a maintenance engineer actually cares about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, accuracy_score)


def classification_report_dict(y_true, score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    y_pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred) if both else np.nan,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred) if both else np.nan,
        "roc_auc": roc_auc_score(y_true, score) if both else np.nan,
        "pr_auc": average_precision_score(y_true, score) if both else np.nan,
        "brier": brier_score_loss(y_true, np.clip(score, 0, 1)) if both else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "fnr": fn / (fn + tp) if (fn + tp) else np.nan,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "positives": int(y_true.sum()), "n": int(len(y_true)),
    }


def best_f1_threshold(y_true, score) -> float:
    """Threshold maximising F1 — must be chosen on VALIDATION data, never on test."""
    from sklearn.metrics import precision_recall_curve
    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.nanargmax(f1[:-1])]) if len(t) else 0.5


def alert_episodes(timestamps: pd.Series, alarm: np.ndarray, merge_gap_min: int = 30,
                   min_duration_min: int = 5) -> pd.DataFrame:
    """Group per-minute alarms into alert episodes: alarms closer than `merge_gap_min`
    are merged; episodes shorter than `min_duration_min` are dropped (debouncing).
    This is what turns thousands of flagged minutes into a handful of maintenance alerts."""
    t = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    a = np.asarray(alarm).astype(bool)
    if not a.any():
        return pd.DataFrame(columns=["start", "end", "minutes"])
    ts = t[a].reset_index(drop=True)
    new = ts.diff() > pd.Timedelta(minutes=merge_gap_min)
    gid = new.cumsum()
    ep = ts.groupby(gid).agg(start="min", end="max", minutes="size").reset_index(drop=True)
    return ep[ep.minutes >= min_duration_min].reset_index(drop=True)


def event_level_report(timestamps, alarm, events: list[dict], hours_logged: float,
                       lookback_h: float = 12, merge_gap_min: int = 30, min_duration_min: int = 5) -> dict:
    """Event-level evaluation.

    An event counts as DETECTED if an alert episode overlaps [start - lookback_h, end].
    lead time = event start - first alert in that interval (positive = warned early).
    An alert episode that overlaps no such interval is a FALSE ALARM.
    `hours_logged` = hours of data actually evaluated (for false alarms per day)."""
    ep = alert_episodes(timestamps, alarm, merge_gap_min, min_duration_min)
    rows, matched = [], np.zeros(len(ep), dtype=bool)
    for ev in events:
        s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        lo = s - pd.Timedelta(hours=lookback_h)
        hit = (ep.end >= lo) & (ep.start <= e) if len(ep) else pd.Series(dtype=bool)
        matched |= hit.to_numpy() if len(ep) else matched
        first = ep.loc[hit, "start"].min() if len(ep) and hit.any() else pd.NaT
        rows.append({"event": ev["id"], "detected": bool(pd.notna(first)),
                     "first_alert": first,
                     "lead_time_min": (s - first).total_seconds() / 60 if pd.notna(first) else np.nan})
    n_fa = int((~matched).sum())
    return {"events": rows,
            "detected": sum(r["detected"] for r in rows), "n_events": len(rows),
            "false_alarm_episodes": n_fa,
            "false_alarms_per_day": n_fa / (hours_logged / 24) if hours_logged else np.nan,
            "alert_episodes": int(len(ep))}
