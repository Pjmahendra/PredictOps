import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib import api, ui
from lib.common import data_role_note, date_range_picker, events, every_for

ui.setup("Sensor Monitoring", "Pressure, temperature, motor current and valve signals with rolling statistics")
SENS = {"TP2": "bar", "TP3": "bar", "H1": "bar", "DV_pressure": "bar", "Oil_temperature": "°C", "Motor_current": "A", "LPS": "0/1"}
try:
    s, e = date_range_picker("sens")
    pick = st.multiselect("Sensors (one panel each — never two scales on one axis)", list(SENS),
                          default=["TP3", "Oil_temperature", "Motor_current", "DV_pressure"])
    h = api.history(s, e, every=every_for(s, e), columns=list(SENS) + ["anomaly_alert", "data_role"])
    fails, _ = events()
except api.ApiError as ex:
    ui.error_box(ex); st.stop()
if h.empty or not pick:
    st.info("No recorded data in this range (logger offline) or no sensor selected."); st.stop()
data_role_note(h)

sec = ui.subnav(["Sensor charts", "Statistics", "Raw table"], "sensors")

if sec == "Sensor charts":
    c1, c2 = st.columns(2)
    roll = c1.select_slider("Rolling mean window", ["off", "5 min", "30 min", "2 h"], value="30 min")
    hl = c2.toggle("Highlight anomaly alerts", value=True)
    fig = make_subplots(rows=len(pick), cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=[f"{p} ({SENS[p]})" for p in pick])
    x = h.set_index("timestamp")
    for i, p in enumerate(pick, start=1):
        color = ui.SERIES[i - 1 if i <= len(ui.SERIES) else 0]
        fig.add_trace(go.Scattergl(x=x.index, y=x[p], name=p, line=dict(color=color, width=1), opacity=.55 if roll != "off" else 1), row=i, col=1)
        if roll != "off":
            w = {"5 min": "5min", "30 min": "30min", "2 h": "2h"}[roll]
            fig.add_trace(go.Scatter(x=x.index, y=x[p].rolling(w).mean(), name=f"{p} rolling mean ({roll})",
                                     line=dict(color=ui.TEXT, width=2)), row=i, col=1)
        for a, b, _ in fails:
            if b >= s and a <= e:
                fig.add_vrect(x0=a, x1=b, fillcolor=ui.FAIL_FILL, line_width=0, row=i, col=1)
        if hl and x.anomaly_alert.any():
            al = x.anomaly_alert.fillna(False).astype(bool)
            starts = al & ~al.shift(1, fill_value=False); ends = al & ~al.shift(-1, fill_value=False)
            for a, b in zip(x.index[starts], x.index[ends]):
                fig.add_vrect(x0=a, x1=b, fillcolor=ui.ALERT_FILL, line_width=0, row=i, col=1)
    fig.update_layout(height=230 * len(pick) + 60, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red shading = company-reported failure window · amber shading = PredictOps anomaly alert · dark line = rolling mean (past-only).")

elif sec == "Statistics":
    ui.section("Statistics of the selected sensors in this range")
    st.dataframe(api.real(h).set_index("timestamp")[pick].describe().T.round(3), use_container_width=True)

else:
    ui.section("Raw minute values")
    st.dataframe(api.real(h).set_index("timestamp")[pick].round(3), use_container_width=True, height=520)
