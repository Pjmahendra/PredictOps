"""Thin HTTP client for the PredictOps backend. Every dashboard number comes through here."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
TIMEOUT = httpx.Timeout(120.0, connect=5.0)


class ApiError(RuntimeError):
    pass


def _handle(r: httpx.Response):
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text[:300]
        raise ApiError(f"{r.status_code}: {detail}")
    return r


def get(path: str, **params):
    try:
        return _handle(httpx.get(f"{BASE}{path}", params=params, headers=HEADERS, timeout=TIMEOUT)).json()
    except httpx.ConnectError:
        raise ApiError(f"Cannot reach the PredictOps API at {BASE}. Start it with: "
                       "`.venv/bin/uvicorn backend.main:app --port 8000`")


def post(path: str, json=None, files=None):
    try:
        return _handle(httpx.post(f"{BASE}{path}", json=json, files=files, headers=HEADERS, timeout=TIMEOUT)).json()
    except httpx.ConnectError:
        raise ApiError(f"Cannot reach the PredictOps API at {BASE}.")


def get_bytes(path: str, **params) -> bytes:
    return _handle(httpx.get(f"{BASE}{path}", params=params, headers=HEADERS, timeout=TIMEOUT)).content


@st.cache_data(ttl=600, show_spinner=False)
def cached(path: str, **params):
    return get(path, **params)


def history(start, end, every=1, columns=None) -> pd.DataFrame:
    params = {"start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(), "every": every}
    if columns:
        params["columns"] = ",".join(columns)
    df = pd.DataFrame(cached("/history", **params))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df.timestamp)
        df = break_gaps(df, step_min=every)
    return df


def break_gaps(df: pd.DataFrame, step_min: int = 1) -> pd.DataFrame:
    """Insert an empty row after every recording gap so line charts show the gap
    instead of drawing a straight line through time with no data."""
    gap = df.timestamp.diff() > pd.Timedelta(minutes=max(15, 3 * step_min))
    if not gap.any():
        return df.assign(_gap=False)
    blanks = pd.DataFrame({"timestamp": df.timestamp[gap] - pd.Timedelta(seconds=1), "_gap": True})
    out = pd.concat([df.assign(_gap=False), blanks]).sort_values("timestamp").reset_index(drop=True)
    out["_gap"] = out["_gap"].astype(bool)
    return out


def real(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are actual minutes (gap markers removed) — use for statistics and selectors."""
    return df[~df["_gap"]].reset_index(drop=True) if "_gap" in df else df
