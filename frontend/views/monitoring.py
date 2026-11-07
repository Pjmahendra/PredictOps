import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import api, ui

ui.setup("Data & Model Monitoring", "Uploaded datasets, validation status, feature drift, prediction drift and model versions")
ui.note("<b>Data drift</b> = inputs changed (e.g. hotter summer oil). <b>Concept drift</b> = the relation between inputs and failures changed. "
        "<b>Performance degradation</b> can only be measured once new failure labels exist — drift alone does not prove the model is wrong.")
try:
    monthly = pd.DataFrame(api.cached("/monitoring/monthly"))
    versions = api.get("/models/versions"); datasets = api.get("/datasets")
except api.ApiError as ex:
    ui.error_box(ex); st.stop()

t1, t2, t3, t4 = st.tabs(["Drift over time", "Check a window / upload", "Datasets & validation", "Model versions"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=monthly.month, y=monthly.anomaly_flag_share, marker_color=ui.SERIES[0], text=(monthly.anomaly_flag_share * 100).round(2).astype(str) + "%",
                               textposition="outside"))
        fig.add_hline(y=0.005, line=dict(color=ui.MUTED, dash="dot"), annotation_text="expected 0.5% on normal reference", annotation_font_color=ui.TEXT_2)
        fig.update_layout(height=320, hovermode="closest", title="Prediction drift: share of minutes flagged unusual", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure(go.Bar(x=monthly.month, y=monthly.features_major, name="major (PSI>0.25)", marker_color=ui.SERIES[1]))
        fig.add_trace(go.Bar(x=monthly.month, y=monthly.features_moderate, name="moderate (0.1–0.25)", marker_color=ui.SERIES[3]))
        fig.update_layout(height=320, barmode="stack", hovermode="closest", title="Feature drift: number of drifting features (of 18)")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(monthly, hide_index=True, use_container_width=True)
    ui.note("Even training months show 'major' PSI against the pooled Feb–May reference: monthly operating conditions differ and the "
            "textbook 0.1/0.25 PSI cut-offs are too sensitive for these multi-modal signals. Interpret drift <i>relative to training-month "
            "variation</i>; the summer months show the expected oil-temperature shift (top drifting feature).")
with t2:
    mode = st.radio("Current data", ["Historical window", "Uploaded run"], horizontal=True)
    try:
        if mode == "Historical window":
            c1, c2 = st.columns(2)
            s = c1.date_input("From", pd.Timestamp("2020-08-01")); e = c2.date_input("To", pd.Timestamp("2020-08-08"))
            d = api.get("/monitoring/drift", start=pd.Timestamp(s).isoformat(), end=(pd.Timestamp(e) + pd.Timedelta(days=1)).isoformat())
        else:
            runs = [r for r in api.get("/runs") if r["dataset_id"]]
            if not runs:
                st.info("No runs with stored data."); st.stop()
            r = st.selectbox("Run", runs, format_func=lambda r: f"#{r['id']} {r['source']} {r['start_time'][:16]}")
            d = api.get("/monitoring/drift", run_id=r["id"])
    except api.ApiError as ex:
        ui.error_box(ex); st.stop()
    c = st.columns(3)
    with c[0]: ui.card("Minutes analysed", f"{d['minutes']:,}")
    with c[1]: ui.card("Features drifting (major)", str(d["summary"]["features_major"]), f"{d['summary']['features_moderate']} moderate")
    with c[2]: ui.card("Unusual-minute share", f"{d['prediction_drift']['flagged_share_current']:.2%}",
                       f"reference {d['prediction_drift']['flagged_share_reference']:.2%}")
    f = pd.DataFrame(d["feature_drift"])
    fig = go.Figure(go.Bar(x=f.psi.clip(upper=2), y=f.feature, orientation="h", marker_color=ui.SERIES[0]))
    fig.add_vline(x=0.25, line=dict(color=ui.STATUS["serious"], dash="dash"), annotation_text="0.25", annotation_font_color=ui.TEXT_2)
    fig.update_layout(height=480, hovermode="closest", title="PSI per feature (clipped at 2)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(f.round(4), hide_index=True, use_container_width=True); st.caption(d["note"])
with t3:
    up = st.file_uploader("Upload a CSV to validate (MetroPT-3 format)", type=["csv"])
    if up is not None:
        try:
            ds = api.post("/datasets/upload", files={"file": (up.name, up.getvalue(), "text/csv")})
            (st.success if ds["validation"]["ok"] else st.error)(f"Validation {'passed' if ds['validation']['ok'] else 'failed'}: {ds['n_rows']:,} rows")
            st.json(ds["validation"])
        except api.ApiError as ex:
            ui.error_box(ex)
    if datasets:
        d = pd.DataFrame(datasets); d["valid"] = d.validation.map(lambda v: "✔ valid" if v["ok"] else "✖ " + "; ".join(v["errors"])[:80])
        st.dataframe(d[["id", "original_filename", "source", "n_rows", "start_time", "end_time", "valid", "uploaded_at"]], hide_index=True, use_container_width=True)
with t4:
    st.dataframe(pd.DataFrame(versions), hide_index=True, use_container_width=True)
    info = api.cached("/models")
    st.json({"failure_detector": {k: info["failure_detector"][k] for k in ["version", "algorithm", "threshold", "train_period", "selection"]},
             "anomaly_detector": {k: info["anomaly_detector"][k] for k in ["version", "algorithm", "fit_period", "calibration_period", "alert_rule"]},
             "deep_learning": info["deep_learning"], "limitations": info["limitations"]})
