import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import api, ui
from lib.common import events

ui.setup("PredictOps — Intelligent Equipment Health & Failure Prediction",
         "Overview of the MetroPT-3 Air Production Unit: health, risk, anomalies and maintenance alerts")
try:
    s = api.get("/dashboard/stats")
    fails, alerts = events()
except api.ApiError as ex:
    ui.error_box(ex); st.stop()

latest = s["latest_assessment"]
c = st.columns(5)
with c[0]: ui.card("Sensor observations", f"{s['dataset']['observations_10s']:,}", f"{s['dataset']['minutes_assessed']:,} minutes assessed")
with c[1]: ui.card("Equipment health", f"{latest['health_score']:.0f}" if latest else "—",
                   "latest assessed minute" if latest else "run a prediction", latest["health_status"] if latest else None)
with c[2]: ui.card("Failure risk", latest["failure_risk"] if latest else "—", "latest run", latest["failure_risk"] if latest else None)
with c[3]: ui.card("Anomaly status", latest["anomaly_status"] if latest else "—", "latest run", latest["anomaly_status"] if latest else None)
with c[4]: ui.card("Anomaly alerts (history)", f"{len(alerts)}", f"{s['alerts_open']} open in DB · {s['dataset']['reported_failures']} reported failures")

sec = ui.subnav(["Health trend", "Status & alerts", "Latest assessment"], "overview")

if sec == "Health trend":
    ui.section("Health score over the whole dataset (daily minimum and mean)")
    h = api.history("2020-02-01", "2020-09-01 04:00", every=10, columns=["health_score", "health_status", "data_role"])
    d = h.set_index("timestamp").health_score.resample("D").agg(["min", "mean"]).dropna()
    weeks = list(range(7, len(d) + 7, 7))
    labels = [d.index[min(w, len(d)) - 1].strftime("%d %b") for w in weeks]
    def traces(n):
        return [go.Scatter(x=d.index[:n], y=d["mean"][:n], name="daily mean", line=dict(color=ui.SERIES[0], width=2.5, shape="spline", smoothing=.4)),
                go.Scatter(x=d.index[:n], y=d["min"][:n], name="daily minimum", line=dict(color=ui.SERIES[1], width=2))]
    fig = go.Figure(traces(len(d)), frames=[go.Frame(data=traces(min(w, len(d))), name=l) for w, l in zip(weeks, labels)])
    ui.shade_windows(fig, [(a, b) for a, b, _ in fails], label="reported failure")
    fig.add_vline(x=pd.Timestamp("2020-07-01"), line=dict(color=ui.MUTED, dash="dot", width=1))
    fig.add_annotation(x=pd.Timestamp("2020-07-01"), y=5, text="← in-sample | unseen test →", showarrow=False, font=dict(color=ui.MUTED, size=10))
    fig.update_layout(height=440, margin=dict(b=90), yaxis=dict(title="health score", range=[0, 102]),
                      xaxis=dict(range=[d.index.min(), d.index.max()]))
    ui.play_controls(fig, labels)
    st.caption("Press ▶ Play to watch the 7 months unfold week by week, or drag the slider.")
    st.plotly_chart(fig, use_container_width=True)

elif sec == "Status & alerts":
    c1, c2 = st.columns([1, 2])
    with c1:
        ui.section("Status share (all minutes)")
        vc = pd.Series(s["history_status_counts"]).reindex(["Healthy", "Warning", "Critical"]).fillna(0)
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", text=[f"{v/vc.sum():.1%}" for v in vc.values],
                               marker_color=[ui.STATUS[ui.STATUS_OF[k]] for k in vc.index], textposition="outside"))
        fig.update_layout(height=260, xaxis=dict(title="minutes", range=[0, vc.max() * 1.25]), hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ui.section("Recent anomaly alert episodes (historical replay of the dataset)")
        if not alerts.empty:
            t = alerts.sort_values("start", ascending=False).head(8)
            st.dataframe(t[["start", "minutes", "max_severity", "status", "data_role"]], hide_index=True, use_container_width=True)

else:
    ui.section("Latest predictions (most recent run)")
    if latest:
        st.markdown(ui.badge(latest["health_status"]) + " " + ui.badge(latest["failure_risk"]) + f"  <span style='color:{ui.MUTED}'>· minute {latest['timestamp']} · run #{latest['run_id']}</span>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.markdown("**Reasons**"); ui.bullet_list(latest["reasons"])
        with c2: st.markdown("**Recommended actions**"); ui.bullet_list(latest["recommendations"])
    else:
        ui.note("No prediction run yet. Go to <b>Failure Prediction</b> and run a sample or upload a CSV; the cards above then show the latest assessment.")

ui.note("PredictOps replays recorded data from one compressor (Feb–Aug 2020). It is not connected to live equipment.")
