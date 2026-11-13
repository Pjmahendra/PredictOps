"""PredictOps dashboard — entry point.

Run (API must be running):  .venv/bin/streamlit run frontend/app.py

Navigation: a top navigation bar (st.navigation position="top"). Groups appear as drop-down menus
(Monitor / Intelligence / Models & Data); inside each page a button bar (ui.subnav) switches between
the page's sections, so pages never need long scrolling.
"""
import streamlit as st

st.set_page_config(page_title="PredictOps", page_icon="⚙️", layout="wide", initial_sidebar_state="collapsed")

pages = {
    "": [
        st.Page("views/overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    ],
    "Monitor": [
        st.Page("views/sensors.py", title="Sensor Monitoring", icon=":material/monitoring:"),
        st.Page("views/replay.py", title="Historical Replay", icon=":material/replay:"),
        st.Page("views/simulation.py", title="Live Simulation", icon=":material/science:"),
    ],
    "Intelligence": [
        st.Page("views/prediction.py", title="Failure Prediction", icon=":material/target:"),
        st.Page("views/anomaly.py", title="Anomaly Detection", icon=":material/troubleshoot:"),
        st.Page("views/explain.py", title="Explainable AI", icon=":material/psychology:"),
        st.Page("views/health.py", title="Equipment Health", icon=":material/favorite:"),
        st.Page("views/maintenance.py", title="Maintenance", icon=":material/build:"),
    ],
    "Models & Data": [
        st.Page("views/performance.py", title="Model Performance", icon=":material/leaderboard:"),
        st.Page("views/monitoring.py", title="Data & Model Monitoring", icon=":material/database:"),
    ],
}
st.navigation(pages, position="top").run()
