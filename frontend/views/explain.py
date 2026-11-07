import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import api, ui

ui.setup("Explainable AI", "SHAP explanations of the failure-signature model and deviations behind anomaly alerts")
with st.expander("How to read SHAP (feature importance vs contribution vs correlation vs causation)", expanded=False):
    ui.bullet_list([
        "<b>Global importance</b>: how much the model relies on a feature on average (mean |SHAP|).",
        "<b>Local contribution</b>: how much a feature pushed <i>this</i> prediction up (orange) or down (blue), in log-odds.",
        "<b>Correlation</b> is a property of the data; SHAP is a property of the model.",
        "<b>SHAP is not causation</b>: it shows what the model used, not what physically caused a failure."])
try:
    g = pd.DataFrame(api.cached("/explain/global")["features"]).head(15).iloc[::-1]
except api.ApiError as ex:
    ui.error_box(ex); st.stop()
ui.section("Global feature importance — failure-signature model (XGBoost)")
fig = go.Figure(go.Bar(x=g.mean_abs_shap, y=g.feature, orientation="h", marker_color=ui.SERIES[0],
                       text=g.mean_abs_shap.round(2), textposition="outside"))
fig.update_layout(height=460, hovermode="closest", xaxis_title="mean |SHAP| (validation sample)", margin=dict(r=60))
st.plotly_chart(fig, use_container_width=True)
ui.note("The model relies mostly on <b>DV_pressure</b> (≈2 bar during events 1–3). That is why it missed test event 4, "
        "where the leak showed as falling TP3 and hot oil instead — the anomaly model caught it (see docs/07).")

ui.section("Individual prediction explanation")
runs = api.get("/runs", limit=50)
if not runs:
    st.info("Run a prediction first (Failure Prediction page)."); st.stop()
run = st.selectbox("Run", runs, format_func=lambda r: f"#{r['id']} {r['source']} {r['start_time'][:16]} → {r['end_time'][:16]}")
p = pd.DataFrame(api.get("/predictions", run_id=run["id"])); p["timestamp"] = pd.to_datetime(p.timestamp)
mode = st.radio("Pick minute", ["Highest risk", "Typical normal", "Choose"], horizontal=True)
if mode == "Highest risk":
    t = p.loc[p.health_score.idxmin(), "timestamp"]
elif mode == "Typical normal":
    t = p.loc[(p.health_score - p.health_score.max()).abs().idxmin(), "timestamp"]
else:
    t = pd.Timestamp(st.select_slider("Minute", p.timestamp.dt.strftime("%Y-%m-%d %H:%M").tolist()))
ex = api.post("/explain", json={"run_id": run["id"], "timestamp": t.isoformat(), "top_k": 10})
st.markdown(f"**{ex['timestamp'][:16]}** · status " + ui.badge(ex["health_status"] or "Healthy") +
            f" · signature score {ex['detector_probability']:.3f} (threshold {ex['detector_threshold']:.2f}) · "
            f"anomaly score {ex['anomaly_score']:.3f} (threshold {ex['anomaly_threshold']:.3f})", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    d = pd.DataFrame(ex["detector_shap"]["contributions"])
    base = ex["detector_shap"]["base_value_logit"]
    fig = go.Figure(go.Waterfall(orientation="h", y=[f"{f} = {v:.2f}" for f, v in zip(d.feature, d.value)][::-1] + ["model output (logit)"],
                                 x=list(d.shap[::-1]) + [None], measure=["relative"] * len(d) + ["total"], base=base,
                                 increasing=dict(marker_color=ui.SERIES[1]), decreasing=dict(marker_color=ui.SERIES[0]),
                                 totals=dict(marker_color=ui.MUTED)))
    fig.update_layout(height=430, hovermode="closest", title=f"SHAP waterfall (base value {base:.2f})", xaxis_title="log-odds")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Top contributions only; the remaining features make up the rest of the difference to the output.")
with c2:
    a = pd.DataFrame(ex["anomaly_top_deviations"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=a.robust_z, y=a.feature, orientation="h", marker_color=[ui.SERIES[1] if z > 0 else ui.SERIES[0] for z in a.robust_z]))
    fig.update_layout(height=430, hovermode="closest", title="Anomaly model: furthest-from-normal features", xaxis_title="robust z (vs normal median)")
    st.plotly_chart(fig, use_container_width=True)
if ex.get("reasons"):
    ui.section("Sensor contribution analysis — engine reasons"); ui.bullet_list(ex["reasons"])
