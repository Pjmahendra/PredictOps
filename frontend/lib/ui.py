"""PredictOps visual system: dark industrial theme, reusable cards/badges, Plotly template.

Colour rules (validated with the dataviz palette validator on the #111a2b surface):
  * categorical series use a FIXED order (SERIES) — never cycled, never by rank
  * status colours (good/warning/serious/critical) are reserved for status and always
    shown with an icon + text label, never colour alone
  * text uses text tokens, never series colours
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, SURFACE_2, BORDER = "#0b1220", "#111a2b", "#16223a", "#24324d"
TEXT, TEXT_2, MUTED = "#e8edf5", "#b4c0d3", "#7d8aa3"
ACCENT, ACCENT_2 = "#3987e5", "#22c3e6"           # blue / cyan accents
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]   # fixed categorical order
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_OF = {"Healthy": "good", "Warning": "warning", "Critical": "critical", "Low": "good", "Medium": "warning",
             "High": "critical", "Normal": "good", "Anomalous": "critical", "normal": "good", "low": "warning",
             "medium": "serious", "high": "critical"}
ICON = {"good": "●", "warning": "▲", "serious": "◆", "critical": "■"}
FAIL_FILL = "rgba(208,59,59,0.16)"                # reported failure window shading
ALERT_FILL = "rgba(250,178,25,0.16)"              # anomaly alert shading

pio.templates["predictops"] = go.layout.Template(layout=dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=TEXT_2, family="Inter, system-ui, sans-serif", size=12),
    colorway=SERIES, margin=dict(l=48, r=16, t=40, b=36), hoverlabel=dict(bgcolor=SURFACE_2, font_color=TEXT),
    xaxis=dict(gridcolor="#1c2942", zerolinecolor="#24324d", linecolor=BORDER),
    yaxis=dict(gridcolor="#1c2942", zerolinecolor="#24324d", linecolor=BORDER),
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"), hovermode="x unified",
    title=dict(font=dict(color=TEXT, size=14), x=0, xanchor="left")))
pio.templates.default = "predictops"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, [data-testid="stAppViewContainer"] {{ background:{BG}; color:{TEXT}; font-family:Inter, system-ui, sans-serif; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:#0e1729; border-right:1px solid {BORDER}; }}
.block-container {{ padding-top:1.6rem; max-width:1400px; }}
h1, h2, h3 {{ color:{TEXT}; letter-spacing:-0.01em; }}
.po-hero {{ display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:6px; flex-wrap:wrap; }}
.po-title {{ font-size:1.55rem; font-weight:700; color:{TEXT}; }}
.po-sub {{ color:{MUTED}; font-size:.9rem; }}
.po-pill {{ font-size:.75rem; color:{ACCENT_2}; border:1px solid #1d4f66; background:#0d2a38; padding:3px 10px; border-radius:999px; }}
.po-card {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:14px 16px; height:100%; }}
.po-card .lbl {{ color:{MUTED}; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }}
.po-card .val {{ color:{TEXT}; font-size:1.65rem; font-weight:700; margin-top:2px; font-family:'JetBrains Mono', monospace; }}
.po-card .hint {{ color:{TEXT_2}; font-size:.8rem; margin-top:2px; }}
.po-badge {{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:.8rem;
            font-weight:600; border:1px solid; background:rgba(255,255,255,.02); color:{TEXT}; }}
.po-section {{ color:{TEXT}; font-weight:600; font-size:1.02rem; margin:18px 0 8px; border-left:3px solid {ACCENT}; padding-left:10px; }}
.po-note {{ color:{TEXT_2}; font-size:.82rem; background:#0f1a2e; border:1px dashed {BORDER}; border-radius:8px; padding:8px 12px; }}
.po-list li {{ color:{TEXT_2}; margin-bottom:3px; }}
div[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:10px 14px; }}
.stTabs [data-baseweb="tab"] {{ color:{TEXT_2}; }}
.stTabs [aria-selected="true"] {{ color:{TEXT} !important; }}
</style>
"""


def setup(title: str, subtitle: str = ""):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"""<div class="po-hero"><div><div class="po-title">{title}</div>
    <div class="po-sub">{subtitle}</div></div>
    <span class="po-pill">Historical data · MetroPT-3 APU compressor</span></div>""", unsafe_allow_html=True)


def badge(label: str) -> str:
    s = STATUS_OF.get(label, "good")
    return f'<span class="po-badge" style="border-color:{STATUS[s]}"><span style="color:{STATUS[s]}">{ICON[s]}</span>{label}</span>'


def card(label: str, value: str, hint: str = "", status_label: str | None = None):
    b = f"<div style='margin-top:6px'>{badge(status_label)}</div>" if status_label else ""
    st.markdown(f'<div class="po-card"><div class="lbl">{label}</div><div class="val">{value}</div>'
                f'<div class="hint">{hint}</div>{b}</div>', unsafe_allow_html=True)


def section(text: str):
    st.markdown(f'<div class="po-section">{text}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="po-note">{text}</div>', unsafe_allow_html=True)


def bullet_list(items: list[str]):
    st.markdown('<ul class="po-list">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>", unsafe_allow_html=True)


def shade_windows(fig, windows, color=FAIL_FILL, label=None, row=None):
    for i, (s, e) in enumerate(windows):
        kw = dict(x0=s, x1=e, fillcolor=color, line_width=0, layer="below")
        if label and i == 0:
            kw["annotation_text"] = label; kw["annotation_position"] = "top left"
            kw["annotation_font_color"] = TEXT_2; kw["annotation_font_size"] = 10
        if row:
            fig.add_vrect(row=row, col=1, **kw)
        else:
            fig.add_vrect(**kw)
    return fig


def error_box(ex: Exception):
    st.error(f"⚠ {ex}")
