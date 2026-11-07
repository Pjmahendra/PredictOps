import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import api, ui
from lib.common import events

ui.setup("Historical Replay", "Step through recorded sensor data; each step re-runs the full PredictOps pipeline on the raw readings")
ui.note("<b>Simulated monitoring from historical data</b> — this is a replay of readings recorded in 2020, not a live equipment connection.")
fails, _ = events()
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
preset = c1.selectbox("Scenario", ["Event 4 build-up (15 Jul, unseen test)", "Event 2 (30 May)", "Normal day (10 Aug)", "Custom"])
starts = {"Event 4 build-up (15 Jul, unseen test)": "2020-07-15 10:00", "Event 2 (30 May)": "2020-05-29 21:30", "Normal day (10 Aug)": "2020-08-10 08:00"}
start = pd.Timestamp(c1.text_input("Start (YYYY-MM-DD HH:MM)", "2020-07-15 10:00") if preset == "Custom" else starts[preset])
step = c2.selectbox("Step", [5, 10, 15, 30], index=1, format_func=lambda m: f"{m} min")
n = c3.number_input("Steps", 6, 120, 36)
delay = c4.selectbox("Speed", [0.0, 0.3, 1.0], format_func=lambda s: {0.0: "fast", 0.3: "normal", 1.0: "slow"}[s])

if st.button("▶ Start replay", type="primary"):
    status_box, chart_box, reason_box = st.empty(), st.empty(), st.empty()
    rows = []
    prog = st.progress(0.0)
    for i in range(int(n)):
        t = start + pd.Timedelta(minutes=step * i)
        try:
            r = api.get("/replay/step", t=t.isoformat())
        except api.ApiError as ex:
            rows.append({"t": t, "health_score": None, "health_status": "no data", "anomaly_score": None, "detector_prob": None, "TP3": None,
                         "Oil_temperature": None}); prog.progress((i + 1) / n); continue
        a, s = r["assessment"], r["sensors"]
        rows.append({"t": pd.Timestamp(r["minute"]), **{k: a[k] for k in ["health_score", "health_status", "anomaly_score", "detector_prob"]},
                     "TP3": s["TP3"], "Oil_temperature": s["Oil_temperature"]})
        d = pd.DataFrame(rows)
        with status_box.container():
            c = st.columns(4)
            with c[0]: ui.card("Replay time", f"{pd.Timestamp(r['minute']):%d %b %H:%M}", "historical")
            with c[1]: ui.card("Health", f"{a['health_score']:.0f}", "", a["health_status"])
            with c[2]: ui.card("Failure risk", a["failure_risk"], "", a["failure_risk"])
            with c[3]: ui.card("Anomaly", a["anomaly_status"], f"severity {a['anomaly_severity']}", a["anomaly_status"])
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.06, subplot_titles=["health score", "TP3 (bar)", "oil temperature (°C)"])
        fig.add_trace(go.Scatter(x=d.t, y=d.health_score, mode="lines+markers", line=dict(color=ui.SERIES[0], width=2), marker=dict(size=8)), 1, 1)
        fig.add_trace(go.Scatter(x=d.t, y=d.TP3, line=dict(color=ui.SERIES[1], width=2)), 2, 1)
        fig.add_trace(go.Scatter(x=d.t, y=d.Oil_temperature, line=dict(color=ui.SERIES[2], width=2)), 3, 1)
        for a_, b_, _ in fails:
            for row in (1, 2, 3):
                if b_ >= d.t.min() and a_ <= d.t.max():
                    fig.add_vrect(x0=a_, x1=b_, fillcolor=ui.FAIL_FILL, line_width=0, row=row, col=1)
        fig.update_layout(height=520, showlegend=False, yaxis=dict(range=[0, 102]))
        chart_box.plotly_chart(fig, use_container_width=True, key=f"rp{i}")
        with reason_box.container():
            st.markdown("**Reasons**"); ui.bullet_list(a["reasons"]); st.markdown("**Recommended actions**"); ui.bullet_list(a["recommendations"])
        prog.progress((i + 1) / n)
        time.sleep(delay)
    st.success("Replay finished. Red shading = company-reported failure window.")
