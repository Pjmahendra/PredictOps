"""PredictOps dashboard — entry point.

Run (API must be running):  .venv/bin/streamlit run frontend/app.py
"""
import streamlit as st

st.set_page_config(page_title="PredictOps", page_icon="⚙️", layout="wide")

pages = {
    "Monitor": [
        st.Page("views/overview.py", title="Overview", icon="🏠", default=True),
        st.Page("views/sensors.py", title="Sensor Monitoring", icon="📈"),
        st.Page("views/replay.py", title="Historical Replay", icon="⏯️"),
    ],
    "Intelligence": [
        st.Page("views/prediction.py", title="Failure Prediction", icon="🎯"),
        st.Page("views/anomaly.py", title="Anomaly Detection", icon="🔍"),
        st.Page("views/explain.py", title="Explainable AI", icon="🧠"),
        st.Page("views/health.py", title="Equipment Health", icon="❤️"),
        st.Page("views/maintenance.py", title="Maintenance", icon="🛠️"),
    ],
    "Models & Data": [
        st.Page("views/performance.py", title="Model Performance", icon="📊"),
        st.Page("views/monitoring.py", title="Data & Model Monitoring", icon="🗂️"),
    ],
}
st.navigation(pages).run()
