"""Exploratory Data Analysis helpers.

All functions take the intermediate DataFrame (from data/intermediate/metropt3.parquet)
and either return a statistics table or save a matplotlib figure. They are used by
scripts/run_eda.py (batch generation) and notebooks/02_exploratory_analysis.ipynb.

Large-data rule: plots of raw points use either a time slice (around an event) or a
fixed random sample (seeded), never all 1.5M points at once.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from predictops.config import load_config

sns.set_theme(style="whitegrid", context="notebook")
FAIL_COLOR = "#d62728"


# ----------------------------------------------------------------- labels / derived
def add_failure_flags(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add `in_failure_window` (0/1) and `event_id` (0 = none) from the PDF windows.
    Used for EDA/evaluation only; the supervised target is defined in Phase 3."""
    cfg = cfg or load_config()
    df["in_failure_window"] = np.int8(0)
    df["event_id"] = np.int8(0)
    for ev in cfg["failure_events"]:
        m = df.timestamp.between(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]))
        df.loc[m, ["in_failure_window", "event_id"]] = [1, ev["id"]]
    return df


def operating_state(df: pd.DataFrame) -> pd.Series:
    """Physical operating state from documented Motor_current levels (PDF):
    ~0 A off, ~4 A offloaded, ~7 A loaded, ~9 A starting."""
    mc = df["Motor_current"]
    return pd.cut(mc, [-np.inf, 1.0, 5.5, 8.5, np.inf], labels=["off", "offloaded", "loaded", "starting"])


def daily_operation_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day compressor behaviour: hours logged, duty cycle (share of time loaded),
    number of load cycles (0->1 transitions of DV_eletric within a segment), mean pressures."""
    d = df[["timestamp", "segment_id", "DV_eletric", "TP3", "Oil_temperature", "Motor_current", "in_failure_window"]].copy()
    d["load_start"] = (d.groupby("segment_id")["DV_eletric"].diff() == 1).astype(int)
    g = d.groupby(d.timestamp.dt.floor("D"))
    out = pd.DataFrame({
        "rows": g.size(),
        "hours_logged": g.size() * 10 / 3600,
        "duty_cycle": g["DV_eletric"].mean(),
        "load_cycles": g["load_start"].sum(),
        "tp3_mean": g["TP3"].mean(),
        "tp3_min": g["TP3"].min(),
        "oil_temp_mean": g["Oil_temperature"].mean(),
        "failure_share": g["in_failure_window"].mean(),
    })
    out["load_cycles_per_hour"] = out.load_cycles / out.hours_logged
    return out


# ----------------------------------------------------------------- plotting utils
def _shade_failures(ax, cfg, label=True):
    for i, ev in enumerate(cfg["failure_events"]):
        ax.axvspan(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]), color=FAIL_COLOR, alpha=0.25,
                   label="reported failure window" if (label and i == 0) else None)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------- figures
def fig_data_coverage(daily: pd.DataFrame, cfg, out: Path):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.bar(daily.index, daily.hours_logged, width=1, color="#1f77b4")
    _shade_failures(ax, cfg)
    ax.axhline(24, ls="--", c="grey", lw=1)
    ax.set(title="Data coverage: hours of sensor data logged per day", ylabel="hours logged")
    ax.legend(loc="lower left")
    return _save(fig, out)


def fig_histograms(sample: pd.DataFrame, cols, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, c in zip(axes.flat, cols):
        sns.histplot(sample[c], bins=80, ax=ax, color="#1f77b4")
        ax.set_title(c); ax.set_xlabel("")
    for ax in list(axes.flat)[len(cols):]:
        ax.remove()
    fig.suptitle("Analog sensor distributions (random sample of 200k rows)", y=1.01)
    fig.tight_layout()
    return _save(fig, out)


def fig_digital_shares(df: pd.DataFrame, cols, out: Path):
    s = pd.DataFrame({"normal": df.loc[df.in_failure_window == 0, cols].mean(),
                      "failure window": df.loc[df.in_failure_window == 1, cols].mean()})
    fig, ax = plt.subplots(figsize=(10, 4))
    s.plot.bar(ax=ax, color=["#1f77b4", FAIL_COLOR])
    ax.set(title="Digital sensors: share of time active (=1), normal vs failure windows", ylabel="share active")
    ax.tick_params(axis="x", rotation=30)
    return _save(fig, out)


def fig_boxplots_by_class(sample: pd.DataFrame, cols, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    lab = sample.in_failure_window.map({0: "normal", 1: "failure"})
    for ax, c in zip(axes.flat, cols):
        sns.boxplot(x=lab, y=sample[c], ax=ax, order=["normal", "failure"],
                    palette=["#1f77b4", FAIL_COLOR], hue=lab, legend=False, fliersize=1)
        ax.set_title(c); ax.set_xlabel("")
    for ax in list(axes.flat)[len(cols):]:
        ax.remove()
    fig.suptitle("Analog sensors: normal vs reported-failure rows (outliers shown as dots)", y=1.01)
    fig.tight_layout()
    return _save(fig, out)


def fig_loaded_state_comparison(df: pd.DataFrame, out: Path):
    """Compare distributions only while the compressor is LOADED, removing the on/off
    regime effect that otherwise dominates every sensor distribution."""
    st = operating_state(df)
    d = df[st == "loaded"]
    cols = ["TP2", "TP3", "Oil_temperature", "Motor_current"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, c in zip(axes, cols):
        for lab, col in [(0, "#1f77b4"), (1, FAIL_COLOR)]:
            v = d.loc[d.in_failure_window == lab, c]
            sns.kdeplot(v.sample(min(len(v), 50_000), random_state=0), ax=ax, color=col, fill=True,
                        alpha=.3, label="failure" if lab else "normal", common_norm=False)
        ax.set_title(f"{c} (loaded state only)")
    axes[0].legend()
    fig.tight_layout()
    return _save(fig, out)


def fig_correlation(df: pd.DataFrame, cols, out: Path):
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax, square=True)
    ax.set_title("Pearson correlation between all 15 sensors (all rows)")
    return _save(fig, out)


def fig_motor_current_regimes(sample: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    sns.histplot(sample.Motor_current, bins=200, ax=ax, color="#1f77b4")
    ax.set_yscale("log")
    for x, lab in [(0, "off ~0A"), (4, "offloaded ~4A"), (7, "loaded ~7A"), (9, "start ~9A")]:
        ax.axvline(x, ls="--", c="k", lw=.8); ax.text(x + .1, ax.get_ylim()[1] * .3, lab, fontsize=8)
    ax.set(title="Motor current — operating regimes (log count; dashed = levels documented in PDF)")
    return _save(fig, out)


def fig_event_zoom(df: pd.DataFrame, ev: dict, out: Path, days_before=3, days_after=1):
    """Raw sensor traces from `days_before` before to `days_after` after a failure window."""
    s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
    d = df[df.timestamp.between(s - pd.Timedelta(days=days_before), e + pd.Timedelta(days=days_after))]
    cols = ["TP2", "TP3", "H1", "DV_pressure", "Oil_temperature", "Motor_current", "LPS"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(15, 13), sharex=True)
    for ax, c in zip(axes, cols):
        # break the line at recording gaps so gaps are visible (no fake interpolation)
        v = d.set_index("timestamp")[c].astype(float)
        v[d.set_index("timestamp")["dt_seconds"] > 600] = np.nan
        ax.plot(v.index, v.values, lw=.4, c="#1f77b4")
        ax.axvspan(s, e, color=FAIL_COLOR, alpha=.2)
        ax.set_ylabel(c, rotation=0, ha="right")
    axes[0].set_title(f"Event {ev['id']} ({ev['type']}): {days_before} days before → {days_after} day after. Red = reported failure window")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.tight_layout()
    return _save(fig, out)


def fig_daily_behaviour(daily: pd.DataFrame, cfg, out: Path):
    d = daily[daily.hours_logged >= 2]  # ignore near-empty days
    panels = [("tp3_mean", "mean TP3 (bar)"), ("tp3_min", "min TP3 (bar)"),
              ("duty_cycle", "duty cycle\n(share loaded)"), ("load_cycles_per_hour", "load cycles\nper hour"),
              ("oil_temp_mean", "mean oil temp (°C)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 12), sharex=True)
    for ax, (c, lab) in zip(axes, panels):
        ax.plot(d.index, d[c], marker=".", lw=.8)
        _shade_failures(ax, cfg, label=(c == "tp3_mean"))
        ax.set_ylabel(lab, rotation=0, ha="right")
    axes[0].legend(loc="lower left")
    axes[0].set_title("Daily compressor behaviour over the whole period (days with ≥2 h of data)")
    fig.tight_layout()
    return _save(fig, out)


def fig_class_distribution(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    vc = df.in_failure_window.value_counts().sort_index()
    axes[0].bar(["normal", "failure window"], vc.values, color=["#1f77b4", FAIL_COLOR])
    for i, v in enumerate(vc.values):
        axes[0].text(i, v, f"{v:,}\n({v/len(df):.2%})", ha="center", va="bottom")
    axes[0].set(title="Row-level class distribution", ylabel="rows"); axes[0].set_ylim(0, vc.max() * 1.2)
    m = df.groupby(df.timestamp.dt.to_period("M"))["in_failure_window"].agg(["sum", "size"])
    axes[1].bar(m.index.astype(str), m["size"], color="#1f77b4", label="all rows")
    axes[1].bar(m.index.astype(str), m["sum"], color=FAIL_COLOR, label="failure-window rows")
    axes[1].set(title="Rows per month and failure-window rows", ylabel="rows"); axes[1].legend()
    fig.tight_layout()
    return _save(fig, out)


def fig_failure_timeline(df: pd.DataFrame, cfg, out: Path):
    fig, ax = plt.subplots(figsize=(14, 2.6))
    t = df.timestamp
    ax.hlines(1, t.min(), t.max(), color="#bbbbbb", lw=8, label="observation period")
    for ev in cfg["failure_events"]:
        s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        ax.hlines(1, s, e, color=FAIL_COLOR, lw=16)
        ax.annotate(f"#{ev['id']} {s:%d %b}\n{(e-s).total_seconds()/3600:.1f} h", (s, 1.02), fontsize=9, ha="center")
    ax.set_yticks([]); ax.set_ylim(.9, 1.12)
    ax.set_title("Failure timeline: 4 reported air-leak windows (company maintenance reports)")
    return _save(fig, out)


def fig_rolling_example(df: pd.DataFrame, ev: dict, out: Path, window="30min"):
    """Rolling mean/std of TP3 (past-only, per segment) around one event."""
    s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
    d = df[df.timestamp.between(s - pd.Timedelta(days=3), e + pd.Timedelta(days=1))].copy()
    d = d.set_index("timestamp")
    r = d.groupby("segment_id")["TP3"].rolling(window, min_periods=10)
    mean = r.mean().reset_index(level=0, drop=True)
    std = r.std().reset_index(level=0, drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(d.index, d.TP3, lw=.3, c="#aaaaaa", label="raw TP3")
    axes[0].plot(mean.index, mean, lw=1, c="#1f77b4", label=f"rolling mean ({window}, past-only)")
    axes[1].plot(std.index, std, lw=.8, c="#ff7f0e")
    for ax in axes:
        ax.axvspan(s, e, color=FAIL_COLOR, alpha=.2)
    axes[0].legend(loc="lower left"); axes[0].set_ylabel("bar")
    axes[1].set_ylabel(f"rolling std\n({window})")
    axes[0].set_title(f"Rolling statistics of TP3 around event {ev['id']} (computed within segments, past data only)")
    fig.tight_layout()
    return _save(fig, out)


def fig_pairplot(sample: pd.DataFrame, out: Path):
    cols = ["TP2", "TP3", "H1", "Oil_temperature", "Motor_current"]
    s = sample.sample(4000, random_state=1).copy()
    s["class"] = s.in_failure_window.map({0: "normal", 1: "failure"})
    g = sns.pairplot(s[cols + ["class"]], hue="class", palette={"normal": "#1f77b4", "failure": FAIL_COLOR},
                     plot_kws={"s": 6, "alpha": .5}, corner=True, diag_kind="kde")
    g.figure.suptitle("Pairwise sensor relationships (class-stratified sample, 4,000 rows)", y=1.01)
    g.figure.savefig(out, dpi=90, bbox_inches="tight"); plt.close(g.figure)
    return out


# ----------------------------------------------------------------- data-quality findings
ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]


def frozen_runs(df: pd.DataFrame, min_rows: int = 360) -> pd.DataFrame:
    """Runs where ALL 7 analog sensors repeat exactly the same value for >= min_rows
    consecutive readings (360 rows ≈ 1 h). Real analog sensors always jitter slightly,
    so long identical runs indicate the logger repeating a stale value."""
    same = df[ANALOG].diff().abs().sum(axis=1) == 0
    run_id = (~same).cumsum()
    length = same.groupby(run_id).transform("sum")
    m = same & (length >= min_rows)
    g = df[m].groupby(run_id[m]).agg(start=("timestamp", "first"), end=("timestamp", "last"), rows=("timestamp", "size"))
    g["hours"] = (g.rows * 10 / 3600).round(2)
    return g.reset_index(drop=True)


def failure_signature_mask(df: pd.DataFrame) -> pd.Series:
    """Row-level pattern observed in reported failure windows (see docs/02 §5):
    H1 ≈ 0 (no filter discharge), DV_pressure > 1 bar while running, motor current > 4.5 A.
    Descriptive EDA tool only — NOT a label."""
    return (df.H1 < 0.5) & (df.DV_pressure > 1.0) & (df.Motor_current > 4.5)


def signature_episodes(df: pd.DataFrame, window_rows: int = 180, share: float = 0.8) -> pd.DataFrame:
    """Episodes where the failure signature holds for >80% of a trailing 30-min window."""
    sig = failure_signature_mask(df).astype("int8")
    sustained = sig.groupby(df.segment_id).transform(
        lambda x: x.rolling(window_rows, min_periods=window_rows).mean()) > share
    ep = (sustained != sustained.shift()).cumsum()
    g = df[sustained].groupby(ep[sustained]).agg(start=("timestamp", "first"), end=("timestamp", "last"),
                                                 share_in_reported_window=("in_failure_window", "mean"),
                                                 rows=("timestamp", "size"))
    g["hours"] = (g.rows * 10 / 3600).round(2)
    g["share_in_reported_window"] = g.share_in_reported_window.round(2)
    return g.reset_index(drop=True)
