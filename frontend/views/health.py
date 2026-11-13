import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import api, ui
from lib.common import data_role_note, date_range_picker, events, every_for

ui.setup("Equipment Health", "Explainable health score combining anomaly, failure-signature, sensor-range and history evidence")
ui.note("The health score (0–100) is a <b>heuristic index</b>, not a failure probability. "
        "Healthy ≥ 75 · Warning 45–75 · Critical < 45; any alert forces at least Warning, both engines agreeing forces Critical.")
try:
    s, e = date_range_picker("hl")
    h = api.history(s, e, every=every_for(s, e), columns=["health_score", "health_status", "failure_risk", "anomaly_status",
                                                          "detector_prob", "reasons", "recommendations", "data_role"])
    fails, _ = events()
except api.ApiError as ex:
    ui.error_box(ex); st.stop()
if h.empty:
    st.info("No recorded data in this range."); st.stop()
hp, h = h, api.real(h)
data_role_note(h)
cur = h.iloc[-1]
c = st.columns(4)
with c[0]: ui.card("Current health score", f"{cur.health_score:.0f}", f"at {cur.timestamp:%d %b %H:%M}", cur.health_status)
with c[1]: ui.card("Lowest in range", f"{h.health_score.min():.0f}", f"at {h.loc[h.health_score.idxmin(), 'timestamp']:%d %b %H:%M}")
with c[2]: ui.card("Time Critical", f"{(h.health_status == 'Critical').mean():.1%}", "of minutes in range")
with c[3]: ui.card("Risk now", cur.failure_risk, "", cur.failure_risk)

sec = ui.subnav(["Health trend", "3D health landscape", "Risk indicators", "Why this status?"], "health")

if sec == "Health trend":
    fig = go.Figure()
    for lo, hi, k in [(75, 100, "good"), (45, 75, "warning"), (0, 45, "critical")]:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=ui.STATUS[k], opacity=.06, line_width=0)
    fig.add_trace(go.Scattergl(x=hp.timestamp, y=hp.health_score, name="health score", line=dict(color=ui.SERIES[0], width=2)))
    ui.shade_windows(fig, [(a, b) for a, b, _ in fails if b >= s and a <= e], label="reported failure")
    fig.update_layout(height=420, yaxis=dict(range=[0, 102], title="health score"), title="Historical health trend")
    st.plotly_chart(fig, use_container_width=True)

elif sec == "3D health landscape":
    ui.note("Health score as a landscape: <b>day</b> × <b>hour of day</b> × <b>average health</b>. Valleys are periods of poor health; "
            "a flat high plateau is a healthy machine. Gaps (logger off) are left empty.")
    s3 = min(s, e - pd.Timedelta(days=14))                    # a landscape needs at least two weeks of days
    h3 = api.real(api.history(s3, e, every=10, columns=["health_score"]))
    st.caption(f"Showing {s3:%d %b} → {e:%d %b %Y} (at least 14 days, so the landscape has shape).")
    g = h3.assign(d0=h3.timestamp.dt.normalize(), hour=h3.timestamp.dt.hour)
    z = g.pivot_table(index="d0", columns="hour", values="health_score", aggfunc="mean").reindex(columns=range(24))
    fig = go.Figure(go.Surface(z=z.values, x=list(range(24)), y=[d.strftime("%d %b") for d in z.index], cmin=0, cmax=100,
                               colorscale=[[0, ui.STATUS["critical"]], [0.45, ui.STATUS["warning"]], [0.75, "#9fd3ff"], [1, ui.ACCENT]],
                               colorbar=dict(title="health", thickness=12, len=.6), lighting=dict(ambient=.65, diffuse=.7, specular=.25, roughness=.6),
                               hovertemplate="%{y} %{x}:00<br>health %{z:.0f}<extra></extra>"))
    fig.update_layout(height=560,
                      scene=dict(xaxis_title="hour of day", yaxis_title="day", zaxis=dict(title="health", range=[0, 100])))
    st.plotly_chart(ui.orbit(fig, radius=1.5, height=0.95), use_container_width=True)

elif sec == "Risk indicators":
    ui.section("Share of minutes per status and risk level in this range")
    for col, labels in [("health_status", ["Healthy", "Warning", "Critical"]), ("failure_risk", ["Low", "Medium", "High"])]:
        vc = h[col].value_counts(normalize=True).reindex(labels).fillna(0)
        st.markdown(" ".join(f"{ui.badge(k)} <span style='color:{ui.TEXT_2}'>{v:.1%}</span>" for k, v in vc.items()), unsafe_allow_html=True)

else:
    ui.section("Why this classification? (select a minute)")
    t = st.select_slider("Minute", h.timestamp.dt.strftime("%Y-%m-%d %H:%M").tolist(),
                         value=h.loc[h.health_score.idxmin(), "timestamp"].strftime("%Y-%m-%d %H:%M"))
    r = h[h.timestamp.dt.strftime("%Y-%m-%d %H:%M") == t].iloc[0]
    st.markdown(f"{ui.badge(r.health_status)} score **{r.health_score:.0f}** · risk {ui.badge(r.failure_risk)} · anomaly {ui.badge(r.anomaly_status)}",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.markdown("**Reasons**"); ui.bullet_list(r.reasons.split(" | "))
    with c2: st.markdown("**Recommended actions**"); ui.bullet_list(str(r.recommendations).split(" | "))
