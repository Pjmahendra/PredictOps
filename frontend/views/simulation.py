"""Live Simulation — a continuous SYNTHETIC stream that fails at random, unannounced times."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import api, ui

ui.setup("Live Simulation", "A synthetic sensor stream runs continuously and drifts from healthy to failure at random times",
         pill="Simulated stream · synthetic data")
ui.note("<b>SIMULATED DATA.</b> Readings are synthetic: pieces of real MetroPT-3 normal and failure recordings are "
        "blended on a simulated clock. This is a demo of the pipeline reacting live, not a real machine and not a result.")

PHASE_FILL = {"degrading": "rgba(250,178,25,0.14)", "failure": ui.FAIL_FILL, "maintenance": "rgba(125,138,163,0.18)"}

c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1.2])
speed = c1.select_slider("Speed (sim-min per second)", [0.5, 1, 2, 5, 10], value=2)
window = c5.selectbox("Show last", [60, 180, 360, 720], index=1, format_func=lambda m: f"{m // 60 or m} {'h' if m >= 60 else 'min'}")
sec = ui.subnav(["Live charts", "Reasons & actions", "Phase timeline"], "sim")
show_truth = st.toggle("Show injected phases (simulation ground truth)", value=True)
try:
    if c2.button("▶ Start", type="primary", use_container_width=True):
        api.post(f"/simulation/start?speed={speed}")
    if c3.button("⏸ Stop", use_container_width=True):
        api.post("/simulation/stop")
    if c4.button("↺ Reset", use_container_width=True):
        api.post(f"/simulation/start?speed={speed}&reset=true")
except api.ApiError as ex:
    ui.error_box(ex)


@st.fragment(run_every=2)
def live():
    try:
        s = api.get("/simulation/state", minutes=window)
    except api.ApiError as ex:
        ui.error_box(ex); return
    if not s["rows"]:
        st.info("Simulation not started. Press ▶ Start.")
        return
    d = pd.DataFrame(s["rows"]); d["timestamp"] = pd.to_datetime(d.timestamp)
    last = d.iloc[-1]
    c = st.columns(5)
    with c[0]: ui.card("Sim clock", f"{last.timestamp:%H:%M}", "running" if s["running"] else "stopped", animate=False)
    with c[1]: ui.card("Health", f"{last.health_score:.0f}", "", last.health_status, animate=False)
    with c[2]: ui.card("Failure risk", last.failure_risk, "", last.failure_risk, animate=False)
    with c[3]: ui.card("Anomaly severity", str(last.anomaly_severity), "", last.anomaly_status, animate=False)
    with c[4]: ui.card("Injected phase", s["current_phase"] if show_truth else "hidden", "ground truth of the simulator", animate=False)

    if sec == "Reasons & actions":
        l, r = st.columns(2)
        with l:
            st.markdown("**Reasons**"); ui.bullet_list(last.reasons or ["—"])
        with r:
            st.markdown("**Recommended actions**"); ui.bullet_list(last.recommendations or ["—"])
        return
    if sec == "Phase timeline":
        ph = pd.DataFrame(s["phases"])
        st.dataframe(ph.iloc[::-1] if show_truth else ph.assign(phase="hidden").iloc[::-1], hide_index=True, use_container_width=True)
        st.caption("Phases injected by the simulator (its ground truth). Maintenance = logger silent, so a new data segment starts.")
        return
    gaps = d.timestamp.diff() > pd.Timedelta(minutes=2)          # maintenance: logger silent -> break the lines
    if gaps.any():
        d = pd.concat([d, pd.DataFrame({"timestamp": d.timestamp[gaps] - pd.Timedelta(seconds=30)})]).sort_values("timestamp")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.05,
                        subplot_titles=["health score", "TP3 (bar)", "motor current (A)", "oil temperature (°C)"])
    for i, (col, color) in enumerate([("health_score", ui.SERIES[0]), ("TP3", ui.SERIES[1]),
                                      ("Motor_current", ui.SERIES[3]), ("Oil_temperature", ui.SERIES[2])], start=1):
        fig.add_trace(go.Scattergl(x=d.timestamp, y=d[col], mode="lines", line=dict(color=color, width=2), name=col), i, 1)
    if show_truth:
        lo = d.timestamp.min()
        for p in s["phases"]:
            if p["phase"] in PHASE_FILL and pd.Timestamp(p["end"]) >= lo:
                for r in range(1, 5):
                    fig.add_vrect(x0=max(pd.Timestamp(p["start"]), lo), x1=p["end"], fillcolor=PHASE_FILL[p["phase"]],
                                  line_width=0, layer="below", row=r, col=1)
    fig.update_layout(height=640, showlegend=False, yaxis=dict(range=[0, 102]))
    st.plotly_chart(fig, use_container_width=True)
    if show_truth:
        st.caption("Shading: amber = degrading injected · red = failure injected · grey = maintenance (logger silent)")


live()
