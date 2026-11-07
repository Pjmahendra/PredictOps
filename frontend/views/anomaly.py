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
c = st.columns(4)
with c[0]: ui.card("Max anomaly score", f"{h.anomaly_score.max() + 0.0:.3f}".replace("-0.000", "0.000"), f"low threshold {th['low']:.3f}")
with c[1]: ui.card("Unusual minutes", f"{(h.anomaly_severity != 'normal').mean():.1%}", "≈0.5% expected in normal data")
with c[2]: ui.card("Persistent-anomaly minutes", f"{int(h.anomaly_alert.sum())}", "minutes in alert state")
with c[3]: ui.card("Worst severity", max(h.anomaly_severity, key=lambda x: ["normal", "low", "medium", "high"].index(x)), "",
                   max(h.anomaly_severity, key=lambda x: ["normal", "low", "medium", "high"].index(x)))

fig = go.Figure()
fig.add_trace(go.Scattergl(x=hp.timestamp, y=hp.anomaly_score, name="anomaly score", line=dict(color=ui.SERIES[0], width=1.5)))
for lvl, dash in [("low", "dot"), ("medium", "dash"), ("high", "solid")]:
    fig.add_hline(y=th[lvl], line=dict(color=ui.STATUS[ui.STATUS_OF[lvl]], dash=dash, width=1),
                  annotation_text=f"{lvl} ({['99.5', '99.9', '99.99'][['low','medium','high'].index(lvl)]}% of normal)",
                  annotation_font_color=ui.TEXT_2, annotation_position="right")
ui.shade_windows(fig, [(a, b) for a, b, _ in fails if b >= s and a <= e], label="reported failure")
al = h.set_index("timestamp").anomaly_alert.astype(bool)
st_, en_ = al & ~al.shift(1, fill_value=False), al & ~al.shift(-1, fill_value=False)
ui.shade_windows(fig, list(zip(al.index[st_], al.index[en_])), color=ui.ALERT_FILL, label="alert")
fig.update_layout(height=380, title="Anomaly score timeline (higher = less like normal operation)", yaxis_title="score")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns([1, 2])
with c1:
    ui.section("Severity distribution")
    vc = h.anomaly_severity.value_counts().reindex(["normal", "low", "medium", "high"]).fillna(0)
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[ui.STATUS[ui.STATUS_OF[k]] for k in vc.index], text=vc.values.astype(int), textposition="outside"))
    fig.update_layout(height=280, hovermode="closest", yaxis_title="minutes"); st.plotly_chart(fig, use_container_width=True)
with c2:
    ui.section("Historical anomaly events (all alert episodes in the dataset)")
    st.dataframe(alerts[["start", "end", "minutes", "max_severity", "status", "data_role"]].sort_values("start", ascending=False),
                 hide_index=True, use_container_width=True, height=280)

ui.section("Abnormal sensor measurements at the worst minute")
worst = h.loc[h.anomaly_score.idxmax()]
step = api.get("/replay/step", t=worst.timestamp.isoformat())
st.markdown(f"**{worst.timestamp:%Y-%m-%d %H:%M}** — " + ui.badge(step["assessment"]["anomaly_severity"]), unsafe_allow_html=True)
ui.bullet_list(step["assessment"]["reasons"])
