import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import api, ui

ui.setup("Failure Prediction", "Run the PredictOps pipeline on a sample or uploaded dataset — batch assessment per minute")
try:
    info = api.cached("/models")
    samples = api.cached("/datasets/samples")["samples"]
    uploads = [d for d in api.get("/datasets") if d["source"] == "upload" and d["validation"]["ok"]]
except api.ApiError as ex:
    ui.error_box(ex); st.stop()

STEPS = ["1 · Choose & run", "2 · Results", "3 · Explain a minute", "4 · Runs & downloads"]
sec = ui.subnav(STEPS, "pred")
thr = info["failure_detector"]["threshold"]
res = st.session_state.get("pred")

if sec == STEPS[0]:
    ui.note("<b>Prediction horizon:</b> the failure-signature model answers <i>“is a failure of the previously reported type in progress now?”</i>. "
            "Hours-ahead early warning was tested and is <b>not supported</b> by this dataset (docs/05 §5), so it is not offered.")
    src = st.radio("Source", ["Sample data", "Uploaded dataset", "Upload new CSV"], horizontal=True)
    req = None
    if src == "Sample data":
        name = st.selectbox("Sample (real excerpts from the unseen test period)", samples)
        req = {"sample_name": name}
    elif src == "Uploaded dataset":
        if not uploads:
            st.info("No valid uploads yet."); st.stop()
        d = st.selectbox("Dataset", uploads, format_func=lambda d: f"#{d['id']} {d['original_filename']} ({d['n_rows']:,} rows)")
        req = {"dataset_id": d["id"]}
    else:
        f = st.file_uploader("CSV in MetroPT-3 format (timestamp + 15 sensors), ≤ 50 MB", type=["csv"])
        if f is not None:
            try:
                ds = api.post("/datasets/upload", files={"file": (f.name, f.getvalue(), "text/csv")})
            except api.ApiError as ex:
                ui.error_box(ex); st.stop()
            v = ds["validation"]
            (st.success if v["ok"] else st.error)(f"Validation {'passed' if v['ok'] else 'FAILED'} — {ds['n_rows']:,} rows")
            for w in v["warnings"]: st.warning(w)
            for er in v["errors"]: st.error(er)
            if v["ok"]:
                req = {"dataset_id": ds["id"]}
    if req and st.button("Run batch prediction", type="primary"):
        with st.spinner("Cleaning → features → anomaly model → failure-signature model → health engine…"):
            try:
                st.session_state["pred"] = api.post("/predict", json=req)
                st.session_state["_goto_pred"] = STEPS[1]          # jump straight to the results
                st.rerun()
            except api.ApiError as ex:
                ui.error_box(ex)
    st.stop()

if not res:
    st.info("No prediction yet — choose data and run it in **1 · Choose & run**."); st.stop()
run, latest = res["run"], res["latest"]
p = pd.DataFrame(api.get("/predictions", run_id=run["id"])); p["timestamp"] = pd.to_datetime(p.timestamp)
st.caption(f"Run #{run['id']} · {run['n_minutes']:,} minutes · {run['start_time'][:16]} → {run['end_time'][:16]}")

if sec == STEPS[1]:
    c = st.columns(4)
    with c[0]: ui.card("Failure risk (latest minute)", latest["failure_risk"], latest["timestamp"][:16], latest["failure_risk"])
    with c[1]: ui.card("Signature-model score", f"{latest['detector_prob']:.3f}", f"threshold {thr:.2f} · not a calibrated probability")
    peak = "High" if (p.failure_risk == "High").any() else ("Medium" if (p.failure_risk == "Medium").any() else "Low")
    with c[2]: ui.card("Peak risk in run", peak, f"{(p.failure_risk == 'High').sum()} high-risk minutes", peak)
    with c[3]: ui.card("Model versions", run["model_version_detector"], run["model_version_anomaly"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.timestamp, y=p.detector_prob, name="failure-signature score", line=dict(color=ui.SERIES[0], width=2)))
    fig.add_hline(y=thr, line=dict(color=ui.STATUS["critical"], dash="dash", width=1), annotation_text=f"threshold {thr:.2f}",
                  annotation_font_color=ui.TEXT_2)
    for lvl in ["High", "Medium"]:
        q = p[p.failure_risk == lvl]
        fig.add_trace(go.Scatter(x=q.timestamp, y=[1.04 if lvl == "High" else 1.0] * len(q), mode="markers", name=f"{lvl} risk minute",
                                 marker=dict(color=ui.STATUS[ui.STATUS_OF[lvl]], size=8, symbol="square" if lvl == "High" else "triangle-up")))
    fig.update_layout(height=360, yaxis=dict(range=[0, 1.08], title="score"), title="Failure-signature score and risk classification")
    st.plotly_chart(fig, use_container_width=True)

elif sec == STEPS[2]:
    ui.section("Sensor contributions for a selected minute (SHAP)")
    opts = list(p.timestamp.dt.strftime("%Y-%m-%d %H:%M"))
    t = st.select_slider("Minute", options=opts, value=opts[int(p.health_score.idxmin())])
    try:
        ex = api.post("/explain", json={"run_id": run["id"], "timestamp": t, "top_k": 8})
    except api.ApiError as e:
        ui.error_box(e); st.stop()
    contrib = pd.DataFrame(ex["detector_shap"]["contributions"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=contrib.shap, y=[f"{f} = {v:.2f}" for f, v in zip(contrib.feature, contrib.value)], orientation="h",
                           marker_color=[ui.SERIES[1] if v > 0 else ui.SERIES[0] for v in contrib.shap]))
    fig.update_layout(height=360, hovermode="closest", xaxis_title="SHAP value (log-odds) · orange pushes toward failure, blue toward normal",
                      title=f"Why the signature model scored {ex['detector_probability']:.3f} at {ex['timestamp'][:16]}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(ex["note"])

else:
    c1, c2 = st.columns(2)
    c1.download_button("Download this run's predictions (CSV)", api.get_bytes("/predictions/download", run_id=run["id"]),
                       file_name=f"predictops_run{run['id']}.csv", mime="text/csv", use_container_width=True)
    c2.download_button("Download PDF maintenance report", api.get_bytes(f"/reports/run/{run['id']}.pdf"),
                       file_name=f"predictops_report_run{run['id']}.pdf", mime="application/pdf", use_container_width=True)
    ui.section("Historical prediction runs")
    st.dataframe(pd.DataFrame(api.get("/runs", limit=20))[["id", "source", "n_minutes", "start_time", "end_time", "model_version_detector", "created_at"]],
                 hide_index=True, use_container_width=True)
