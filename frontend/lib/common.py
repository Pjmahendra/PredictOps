"""Helpers shared by several pages."""
import pandas as pd
import streamlit as st

from lib import api

DATA_START, DATA_END = pd.Timestamp("2020-02-01"), pd.Timestamp("2020-09-01 03:59")


@st.cache_data(ttl=3600, show_spinner=False)
def events():
    e = api.get("/history/events")
    fails = [(pd.Timestamp(x["start"]), pd.Timestamp(x["end"]), x["id"]) for x in e["reported_failures"]]
    alerts = pd.DataFrame(e["alert_episodes"])
    if not alerts.empty:
        alerts["start"] = pd.to_datetime(alerts.start); alerts["end"] = pd.to_datetime(alerts.end)
    return fails, alerts


def date_range_picker(key: str, default_start="2020-07-14", default_end="2020-07-16", max_days=14):
    c1, c2, c3 = st.columns([1, 1, 2])
    s = c1.date_input("From", pd.Timestamp(default_start), min_value=DATA_START, max_value=DATA_END, key=f"{key}_s")
    e = c2.date_input("To", pd.Timestamp(default_end), min_value=DATA_START, max_value=DATA_END, key=f"{key}_e")
    fails, _ = events()
    quick = c3.selectbox("Jump to", ["(custom)"] + [f"Reported failure #{i} — {a:%d %b %Y}" for a, _, i in fails],
                         key=f"{key}_q")
    if quick != "(custom)":
        i = int(quick.split("#")[1].split(" ")[0])
        a, b, _ = [f for f in fails if f[2] == i][0]
        return a - pd.Timedelta(days=1), b + pd.Timedelta(hours=12)
    s, e = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(hours=23, minutes=59)
    if e <= s:
        st.warning("'To' must be after 'From'."); st.stop()
    if (e - s).days > max_days:
        st.warning(f"Range limited to {max_days} days for responsiveness; showing the first {max_days} days.")
        e = s + pd.Timedelta(days=max_days)
    return s, e


def every_for(s, e, target=4000):
    return max(1, int((e - s).total_seconds() / 60 / target))


def data_role_note(df):
    if "data_role" in df and df.data_role.notna().any():
        roles = df.data_role.dropna().unique().tolist()
        st.caption("Data role in this range: " + ", ".join(roles) +
                   " — in-sample minutes were used to train/validate the models; only 'unseen (test)' is an honest evaluation.")
