import pandas as pd
import streamlit as st

from lib import api, ui
from lib.common import events

ui.setup("Maintenance Recommendations", "Detected issues, suggested actions, alert acknowledgement and history")
try:
    db_alerts = pd.DataFrame(api.get("/alerts", limit=500))
    _, hist = events()
except api.ApiError as ex:
    ui.error_box(ex); st.stop()

_sec = ui.subnav(["Alerts from prediction runs (acknowledge)", "Alert history of the replayed dataset"], "maint")
if _sec == 'Alerts from prediction runs (acknowledge)':
    if db_alerts.empty:
        st.info("No alerts yet — run a prediction (e.g. sample 'event4_2020-07-15.csv') on the Failure Prediction page.")
    else:
        show = st.radio("Show", ["Open", "Acknowledged", "All"], horizontal=True)
        d = db_alerts if show == "All" else db_alerts[db_alerts.acknowledged == (show == "Acknowledged")]
        c = st.columns(3)
        with c[0]: ui.card("Open alerts", str(int((~db_alerts.acknowledged).sum())), "", "Critical" if (~db_alerts.acknowledged).any() else "Healthy")
        with c[1]: ui.card("Acknowledged", str(int(db_alerts.acknowledged.sum())), "maintenance history")
        with c[2]: ui.card("Critical alerts", str(int((db_alerts.status == "Critical").sum())), "all time")
        for _, a in d.sort_values("start_time", ascending=False).iterrows():
            with st.container(border=True):
                st.markdown(f"{ui.badge(a.status)} {ui.badge(a.severity)} **{a.start_time[:16]} → {a.end_time[11:16]}** "
                            f"({a.minutes} min, run #{a.run_id}, min health {a.min_health_score:.0f})", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1: st.markdown("**Detected issues**"); ui.bullet_list(a.reasons)
                with c2: st.markdown("**Suggested actions**"); ui.bullet_list(a.recommendations)
                if a.acknowledged:
                    st.caption(f"✔ Acknowledged {a.acknowledged_at[:16]} — {a.note or 'no note'}")
                else:
                    note = st.text_input("Maintenance note", key=f"n{a.id}", placeholder="e.g. inspected air circuit, leak found at …")
                    if st.button("Acknowledge", key=f"b{a.id}"):
                        api.post(f"/alerts/{a.id}/acknowledge", json={"note": note}); st.rerun()
if _sec == 'Alert history of the replayed dataset':
    st.caption("Alert episodes the deployed pipeline raises when replaying the whole dataset. Compare with reported failures: "
               "18 Apr, 29–30 May, 5–7 Jun, 15 Jul.")
    st.dataframe(hist.sort_values("start", ascending=False), hide_index=True, use_container_width=True)
