import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import api, ui

ui.setup("Model Performance", "Validation-based selection and the single honest test on unseen event 4")
try:
    perf = api.cached("/models/performance"); curves = api.cached("/models/curves")
except api.ApiError as ex:
    ui.error_box(ex); st.stop()
ui.note("Models were <b>selected on validation</b> (events 2–3) before the test was opened. On the unseen test event 4 the supervised "
        "models <b>missed</b> the failure at their thresholds; the anomaly model detected it. Accuracy is not used — "
        "predicting 'normal' everywhere already scores ≈99.6%.")
ORDER = ["logreg", "tree", "rf", "svm", "xgboost"]
COLOR = dict(zip(ORDER, ui.SERIES))
NAMES = {"logreg": "Logistic Regression", "tree": "Decision Tree", "rf": "Random Forest", "svm": "SVM (RBF)", "xgboost": "XGBoost"}

v = pd.DataFrame(perf["ml_validation_selection"]).set_index("model").reindex(ORDER)
te = pd.DataFrame(perf["ml_test_detection"]).set_index("model").reindex(ORDER)
_sec = ui.subnav(["ML comparison", "3D model map", "Confusion matrices", "ROC / PR curves", "Anomaly & deep learning", "Calibration & warning"], "perf")
if _sec == 'ML comparison':
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Validation (out-of-fold, events 2–3)", "Test (unseen event 4)"])
    for i, (df, cols) in enumerate([(v, ["oof_pr_auc", "oof_f1", "oof_recall", "oof_precision"]), (te, ["pr_auc", "f1", "recall", "precision"])], 1):
        for j, c in enumerate(cols):
            fig.add_trace(go.Bar(x=[NAMES[m] for m in df.index], y=df[c], name=c.replace("oof_", "").upper().replace("_", "-"),
                                 marker_color=ui.SERIES[j], showlegend=(i == 1)), row=1, col=i)
    fig.update_layout(height=400, barmode="group", hovermode="closest", yaxis=dict(range=[0, 1.05]), yaxis2=dict(range=[0, 1.05]),
                      legend=dict(orientation="h", y=-0.18, x=0))
    st.plotly_chart(fig, use_container_width=True)
    show = te[["pr_auc", "roc_auc", "recall", "precision", "f1", "balanced_accuracy", "mcc", "cohen_kappa", "brier", "fpr", "fnr",
               "false_alarms_per_day", "train_seconds", "inference_ms_per_row", "selected"]].round(4)
    st.markdown("**Test metrics (all models)**"); st.dataframe(show, use_container_width=True)
if _sec == '3D model map':
    ui.note("Each ball is one model: <b>validation PR-AUC</b> (x) × <b>test PR-AUC</b> (y) × <b>test ROC-AUC</b> (z). "
            "All models sit at the far right (≈1.0 on validation) but spread out on the unseen test — the validation-vs-test gap in one picture. "
            "The dashed stems drop to the floor to show each model's position.")
    fig = go.Figure()
    for m in ORDER:
        x, y, z = v.loc[m, "oof_pr_auc"], te.loc[m, "pr_auc"], te.loc[m, "roc_auc"]
        fig.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[0, z], mode="lines", line=dict(color=COLOR[m], width=4, dash="dash"),
                                   showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers+text", name=NAMES[m], text=[NAMES[m]], textposition="top center",
                                   textfont=dict(color=ui.TEXT, size=11), marker=dict(size=11, color=COLOR[m], line=dict(width=1, color="#fff")),
                                   hovertemplate=f"{NAMES[m]}<br>validation PR-AUC %{{x:.3f}}<br>test PR-AUC %{{y:.3f}}<br>test ROC-AUC %{{z:.3f}}<extra></extra>"))
    fig.update_layout(height=560, showlegend=False,
                      scene=dict(xaxis=dict(title="validation PR-AUC", range=[0.98, 1.0], tickformat=".3f"), yaxis=dict(title="test PR-AUC", range=[0, 0.55]),
                                 zaxis=dict(title="test ROC-AUC", range=[0, 1.05])))
    st.plotly_chart(ui.orbit(fig, radius=1.55, height=0.65, start=-0.9), use_container_width=True)
if _sec == 'Confusion matrices':
    cols = st.columns(5)
    for c, m in zip(cols, ORDER):
        r = te.loc[m]
        z = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        fig = go.Figure(go.Heatmap(z=z, x=["pred normal", "pred failure"], y=["true normal", "true failure"], colorscale="Blues",
                                   showscale=False, text=z, texttemplate="%{text:,}"))
        fig.update_layout(height=260, title=NAMES[m], hovermode="closest", margin=dict(l=70, r=8, t=36, b=30))
        c.plotly_chart(fig, use_container_width=True)
if _sec == 'ROC / PR curves':
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC (test)", f"Precision–Recall (test, base rate {curves['base_rate']:.4f})"])
    for m in ORDER:
        cv = curves["curves"][m]
        fig.add_trace(go.Scatter(x=cv["roc"]["fpr"], y=cv["roc"]["tpr"], name=NAMES[m], line=dict(color=COLOR[m], width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=cv["pr"]["recall"], y=cv["pr"]["precision"], name=NAMES[m], showlegend=False, line=dict(color=COLOR[m], width=2)), 1, 2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=ui.MUTED, dash="dot"), name="random"), 1, 1)
    fig.update_layout(height=420, hovermode="closest"); fig.update_xaxes(title_text="FPR", row=1, col=1); fig.update_xaxes(title_text="recall", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)
if _sec == 'Anomaly & deep learning':
    a = pd.DataFrame(perf["anomaly"]); a = a[a.set.isin(["validation", "test"])]
    st.markdown("**Anomaly detectors** (selected on validation: One-Class SVM)")
    st.dataframe(a[["detector", "set", "events_detected", "lead_times_min", "false_alarms_per_day", "minute_recall", "minute_precision", "pr_auc", "roc_auc"]].round(3),
                 hide_index=True, use_container_width=True)
    st.markdown("**Deep learning (experimental, not deployed)** — stability across random seeds")
    st.dataframe(pd.DataFrame(perf["deep_learning_seeds"]), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame(perf["deep_learning"]).round(4), hide_index=True, use_container_width=True)
    st.markdown("**Health engine** — share of minutes per status, by ground truth")
    st.dataframe(pd.DataFrame(perf["health_engine"]), hide_index=True, use_container_width=True)
if _sec == 'Calibration & warning':
    st.markdown("**Brier score** (test) is ≈0.0035 for every model — dominated by the 99.6% normal minutes, so it is not used for selection. "
                "Calibration curve (out-of-fold validation): `reports/figures/ml/calibration_oof.png`.")
    st.markdown("**Early-warning targets** (trained on onsets 2–3, tested on onset 4) — at or below chance:")
    st.dataframe(pd.DataFrame(perf["ml_test_warning"]), hide_index=True, use_container_width=True)
    st.markdown("**Feature engineering & imbalance study** (validation PR-AUC)")
    st.dataframe(pd.DataFrame(perf["imbalance"]).round(4), hide_index=True, use_container_width=True)
