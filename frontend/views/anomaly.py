import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import api, ui
from lib.common import data_role_note, date_range_picker, events, every_for

ui.setup("Anomaly Detection", "One-Class SVM trained on normal operation only — scores, severity and persistent alerts")
ui.note("An <b>unusual reading</b> is one high-score minute. <b>Abnormal behaviour</b> is a persistent anomaly (≥10 of 30 minutes) and raises an alert. "
        "A <b>confirmed failure</b> exists only for the 4 company-reported events. Alerts are never shown as confirmed failures.")
try:
    info = api.cached("/models")["anomaly_detector"]
    s, e = date_range_picker("an")
    h = api.history(s, e, every=every_for(s, e), columns=["anomaly_score", "anomaly_severity", "anomaly_alert", "data_role"])
    fails, alerts = events()
except api.ApiError as ex:
    ui.error_box(ex); st.stop()
if h.empty:
    st.info("No recorded data in this range."); st.stop()
hp, h = h, api.real(h)
data_role_note(h)
th = info["thresholds"]
LEVELS = ["normal", "low", "medium", "high"]
worst_sev = max(h.anomaly_severity, key=LEVELS.index)
c = st.columns(4)
with c[0]: ui.card("Max anomaly score", f"{h.anomaly_score.max() + 0.0:.3f}".replace("-0.000", "0.000"), f"low threshold {th['low']:.3f}")
with c[1]: ui.card("Unusual minutes", f"{(h.anomaly_severity != 'normal').mean():.1%}", "≈0.5% expected in normal data")
with c[2]: ui.card("Persistent-anomaly minutes", f"{int(h.anomaly_alert.sum())}", "minutes in alert state")
with c[3]: ui.card("Worst severity", worst_sev, "", worst_sev)

sec = ui.subnav(["Score timeline", "3D sensor space", "Severity & episodes", "Worst minute"], "anomaly")

if sec == "Score timeline":
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=hp.timestamp, y=hp.anomaly_score, name="anomaly score", line=dict(color=ui.SERIES[0], width=1.5)))
    for lvl, dash, q in [("low", "dot", "99.5"), ("medium", "dash", "99.9"), ("high", "solid", "99.99")]:
        fig.add_hline(y=th[lvl], line=dict(color=ui.STATUS[ui.STATUS_OF[lvl]], dash=dash, width=1),
                      annotation_text=f"{lvl} ({q}% of normal)", annotation_font_color=ui.TEXT_2, annotation_position="right")
    ui.shade_windows(fig, [(a, b) for a, b, _ in fails if b >= s and a <= e], label="reported failure")
    al = h.set_index("timestamp").anomaly_alert.astype(bool)
    st_, en_ = al & ~al.shift(1, fill_value=False), al & ~al.shift(-1, fill_value=False)
    ui.shade_windows(fig, list(zip(al.index[st_], al.index[en_])), color=ui.ALERT_FILL, label="alert")
    fig.update_layout(height=430, title="Anomaly score timeline (higher = less like normal operation)", yaxis_title="score")
    st.plotly_chart(fig, use_container_width=True)

elif sec == "3D sensor space":
    ui.note("Each dot is one minute placed by three sensors. Normal operation forms a tight cloud; abnormal minutes drift away from it. "
            "Drag to rotate, scroll to zoom, or press <b>▶ Rotate</b>.")
    x3 = api.real(api.history(s, e, every=max(1, every_for(s, e, 2500)),
                              columns=["TP3", "Oil_temperature", "Motor_current", "anomaly_severity"])).dropna()
    in_fail = pd.Series(False, index=x3.index)
    for a, b, _ in fails:
        in_fail |= x3.timestamp.between(a, b)
    fig = go.Figure()
    for lvl in LEVELS:
        q = x3[(x3.anomaly_severity == lvl) & ~in_fail]
        if len(q):
            fig.add_trace(go.Scatter3d(x=q.TP3, y=q.Oil_temperature, z=q.Motor_current, mode="markers", name=f"{lvl} minute",
                                       marker=dict(size=2.6 if lvl == "normal" else 4, color=ui.STATUS[ui.STATUS_OF[lvl]],
                                                   opacity=.35 if lvl == "normal" else .9),
                                       text=q.timestamp.dt.strftime("%d %b %H:%M"), hovertemplate="%{text}<br>TP3 %{x:.2f} bar<br>oil %{y:.1f} °C<br>motor %{z:.2f} A"))
    q = x3[in_fail]
    if len(q):
        fig.add_trace(go.Scatter3d(x=q.TP3, y=q.Oil_temperature, z=q.Motor_current, mode="markers", name="inside reported failure",
                                   marker=dict(size=4.5, color=ui.STATUS["critical"], symbol="diamond", line=dict(width=0.5, color="#fff")),
                                   text=q.timestamp.dt.strftime("%d %b %H:%M"), hovertemplate="%{text} (reported failure)"))
    fig.update_layout(height=560, hovermode="closest",
                      scene=dict(xaxis_title="TP3 (bar)", yaxis_title="oil temperature (°C)", zaxis_title="motor current (A)"),
                      legend=dict(y=0.98, x=0.98, xanchor="right", bgcolor="rgba(255,255,255,.7)"))
    st.plotly_chart(ui.orbit(fig), use_container_width=True)

elif sec == "Severity & episodes":
    c1, c2 = st.columns([1, 2])
    with c1:
        ui.section("Severity distribution")
        vc = h.anomaly_severity.value_counts().reindex(LEVELS).fillna(0)
        fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[ui.STATUS[ui.STATUS_OF[k]] for k in vc.index], text=vc.values.astype(int), textposition="outside"))
        fig.update_layout(height=320, hovermode="closest", yaxis_title="minutes"); st.plotly_chart(fig, use_container_width=True)
    with c2:
        ui.section("Historical anomaly events (all alert episodes in the dataset)")
        st.dataframe(alerts[["start", "end", "minutes", "max_severity", "status", "data_role"]].sort_values("start", ascending=False),
                     hide_index=True, use_container_width=True, height=320)

else:
    ui.section("Abnormal sensor measurements at the worst minute")
    worst = h.loc[h.anomaly_score.idxmax()]
    step = api.get("/replay/step", t=worst.timestamp.isoformat())
    st.markdown(f"**{worst.timestamp:%Y-%m-%d %H:%M}** — " + ui.badge(step["assessment"]["anomaly_severity"]), unsafe_allow_html=True)
    ui.bullet_list(step["assessment"]["reasons"])
