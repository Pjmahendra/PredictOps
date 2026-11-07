"""Stage 4a — generate all EDA figures and statistics tables.

Outputs
  reports/figures/eda/*.png        charts referenced by docs/02_eda.md
  reports/eda_stats/*.csv          numeric tables behind the charts

Run:  .venv/bin/python scripts/run_eda.py   (~1–2 min)
"""
import time

import pandas as pd

from predictops import eda
from predictops.config import load_config, project_path
from predictops.data.ingestion import load_intermediate

t0 = time.time()
cfg = load_config()
analog = cfg["dataset"]["analog_sensors"]
digital = cfg["dataset"]["digital_sensors"]
FIG = project_path("reports/figures/eda")
TAB = project_path("reports/eda_stats")
TAB.mkdir(parents=True, exist_ok=True)

df = eda.add_failure_flags(load_intermediate(), cfg)
df["state"] = eda.operating_state(df)

# Samples (seeded): uniform for overall distributions; class-stratified for comparisons
uniform = df.sample(200_000, random_state=42)
strat = pd.concat([df[df.in_failure_window == 0].sample(30_000, random_state=42),
                   df[df.in_failure_window == 1].sample(20_000, random_state=42)])

daily = eda.daily_operation_stats(df)
daily.to_csv(TAB / "daily_operation_stats.csv")

# ---- statistics tables
df[analog].describe().T.to_csv(TAB / "analog_summary.csv")
pd.crosstab(df.state, df.in_failure_window, normalize="columns").to_csv(TAB / "state_by_class.csv")
df.groupby(["state", "in_failure_window"], observed=True)[analog].mean().round(3).to_csv(TAB / "means_by_state_and_class.csv")
df.groupby("in_failure_window")[analog + digital].mean().T.round(4).to_csv(TAB / "means_by_class.csv")
df[analog + digital].corr().round(3).to_csv(TAB / "correlation.csv")

# per-event: compare 7 days before onset vs. a "normal reference" (all non-failure days)
rows = []
for ev in cfg["failure_events"]:
    s = pd.Timestamp(ev["start"])
    for name, lo, hi in [("7-3 days before", s - pd.Timedelta(days=7), s - pd.Timedelta(days=3)),
                         ("72-0 h before", s - pd.Timedelta(days=3), s),
                         ("during", s, pd.Timestamp(ev["end"]))]:
        d = daily.loc[(daily.index >= lo.floor("D")) & (daily.index < hi.ceil("D")) & (daily.hours_logged >= 2)]
        rows.append({"event": ev["id"], "period": name, "days": len(d),
                     "duty_cycle": d.duty_cycle.mean(), "load_cycles_per_hour": d.load_cycles_per_hour.mean(),
                     "tp3_min": d.tp3_min.mean(), "oil_temp_mean": d.oil_temp_mean.mean()})
ref = daily[(daily.failure_share == 0) & (daily.hours_logged >= 2)]
rows.append({"event": "-", "period": "reference: all days without failure", "days": len(ref),
             "duty_cycle": ref.duty_cycle.mean(), "load_cycles_per_hour": ref.load_cycles_per_hour.mean(),
             "tp3_min": ref.tp3_min.mean(), "oil_temp_mean": ref.oil_temp_mean.mean()})
pd.DataFrame(rows).round(3).to_csv(TAB / "pre_failure_daily_behaviour.csv", index=False)

# ---- figures
eda.fig_failure_timeline(df, cfg, FIG / "01_failure_timeline.png")
eda.fig_data_coverage(daily, cfg, FIG / "02_data_coverage.png")
eda.fig_class_distribution(df, FIG / "03_class_distribution.png")
eda.fig_histograms(uniform, analog, FIG / "04_histograms.png")
eda.fig_motor_current_regimes(uniform, FIG / "05_motor_current_regimes.png")
eda.fig_digital_shares(df, digital, FIG / "06_digital_shares.png")
eda.fig_correlation(df, analog + digital, FIG / "07_correlation.png")
eda.fig_boxplots_by_class(strat, analog, FIG / "08_boxplots_by_class.png")
eda.fig_loaded_state_comparison(df, FIG / "09_loaded_state_kde.png")
eda.fig_pairplot(strat, FIG / "10_pairplot.png")
eda.fig_daily_behaviour(daily, cfg, FIG / "11_daily_behaviour.png")
for ev in cfg["failure_events"]:
    eda.fig_event_zoom(df, ev, FIG / f"12_event{ev['id']}_zoom.png")
eda.fig_rolling_example(df, cfg["failure_events"][2], FIG / "13_rolling_tp3_event3.png")

print(f"EDA done in {time.time()-t0:.0f}s -> {FIG}  and  {TAB}")

# ---- data-quality findings
fr = eda.frozen_runs(df)
fr.to_csv(TAB / "frozen_runs.csv", index=False)
ep = eda.signature_episodes(df)
ep.to_csv(TAB / "failure_signature_episodes.csv", index=False)
sig = eda.failure_signature_mask(df)
print(f"frozen runs >=1h: {len(fr)} ({fr.rows.sum():,} rows, {fr.rows.sum()/len(df):.2%})")
print(f"signature share inside reported windows: {sig[df.in_failure_window==1].mean():.1%}; "
      f"outside: {sig[df.in_failure_window==0].mean():.2%}")
print(ep[ep.hours >= 1].to_string())
