"""PredictOps visual system: light white / light-blue theme, reusable cards/badges, Plotly template.

Colour rules (validated with the dataviz palette validator on a white surface):
  * categorical series use a FIXED order (SERIES) — never cycled, never by rank
  * status colours (good/warning/serious/critical) are reserved for status and always
    shown with an icon + text label, never colour alone
  * text uses text tokens, never series colours
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, SURFACE, SURFACE_2, BORDER = "#f4f8fd", "#ffffff", "#eaf3fd", "#d5e3f3"
TEXT, TEXT_2, MUTED = "#0f2747", "#3c5674", "#6a7f9a"
ACCENT, ACCENT_2 = "#1f6fd1", "#2b9de0"           # blue / cyan accents
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]   # fixed categorical order
STATUS = {"good": "#0ca30c", "warning": "#d99a00", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_OF = {"Healthy": "good", "Warning": "warning", "Critical": "critical", "Low": "good", "Medium": "warning",
             "High": "critical", "Normal": "good", "Anomalous": "critical", "normal": "good", "low": "warning",
             "medium": "serious", "high": "critical"}
ICON = {"good": "●", "warning": "▲", "serious": "◆", "critical": "■"}
FAIL_FILL = "rgba(208,59,59,0.16)"                # reported failure window shading
ALERT_FILL = "rgba(250,178,25,0.16)"              # anomaly alert shading

pio.templates["predictops"] = go.layout.Template(layout=dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=TEXT_2, family="Inter, system-ui, sans-serif", size=12),
    colorway=SERIES, margin=dict(l=44, r=12, t=30, b=28), hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font_color=TEXT),
    xaxis=dict(gridcolor="#e7eef7", zerolinecolor="#d5e3f3", linecolor=BORDER),
    yaxis=dict(gridcolor="#e7eef7", zerolinecolor="#d5e3f3", linecolor=BORDER),
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"), hovermode="x unified",
    title=dict(font=dict(color=TEXT, size=14), x=0, xanchor="left")))
_ax3d = dict(backgroundcolor="#f3f8fe", gridcolor="#d9e6f5", zerolinecolor="#c9daf0", showbackground=True,
             title_font=dict(color=TEXT_2, size=12), tickfont=dict(color=MUTED, size=10))
pio.templates["predictops"].layout.scene = dict(xaxis=_ax3d, yaxis=_ax3d, zaxis=_ax3d, aspectmode="manual",
                                                aspectratio=dict(x=1.35, y=1.0, z=0.75))
pio.templates.default = "predictops"

FONT_BODY = "'Plus Jakarta Sans', Inter, system-ui, sans-serif"
FONT_DISPLAY = "'Bricolage Grotesque', 'Plus Jakarta Sans', system-ui, sans-serif"
FONT_MONO = "'JetBrains Mono', ui-monospace, monospace"
pio.templates["predictops"].layout.font.family = FONT_BODY.replace("'", "")
pio.templates["predictops"].layout.title.font.family = FONT_DISPLAY.replace("'", "")

# Premium-but-playful look: display font with character, soft glassy cards, springy (overshoot) easing,
# staggered entrance animations, hover lift. Everything respects prefers-reduced-motion.
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
:root {{ --spring:cubic-bezier(.34,1.56,.64,1); --smooth:cubic-bezier(.22,1,.36,1); }}
html, body, [data-testid="stAppViewContainer"] {{ color:{TEXT}; font-family:{FONT_BODY}; }}
p, label, li, input, textarea, button, [data-testid="stMarkdownContainer"], [data-baseweb="select"] {{ font-family:{FONT_BODY}; }}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(900px 480px at 8% -8%, #dcebff 0%, transparent 60%),
              radial-gradient(700px 420px at 100% 0%, #e3f6ff 0%, transparent 55%), {BG}; }}

/* ---------- top navigation bar ---------- */
[data-testid="stHeader"] {{ background:rgba(255,255,255,.78); backdrop-filter:saturate(1.6) blur(14px);
  -webkit-backdrop-filter:saturate(1.6) blur(14px); border-bottom:1px solid {BORDER}; box-shadow:0 4px 24px rgba(31,111,209,.07); }}
[data-testid="stHeader"] a, [data-testid="stHeader"] button {{ font-family:{FONT_BODY}; font-weight:600; border-radius:10px;
  transition:background .25s var(--smooth), transform .25s var(--spring); }}
[data-testid="stHeader"] a:hover, [data-testid="stHeader"] button:hover {{ background:{SURFACE_2}; transform:translateY(-1px); }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none; }}
.block-container {{ padding-top:3.9rem; padding-bottom:1.5rem; max-width:1480px; }}
[data-testid="stVerticalBlock"] {{ gap:.55rem; }}
[data-testid="stHorizontalBlock"] {{ gap:.7rem; }}
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style) {{ display:none; }}
h1, h2, h3 {{ color:{TEXT}; font-family:{FONT_DISPLAY}; letter-spacing:-0.02em; }}

/* ---------- hero ---------- */
.po-hero {{ position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; gap:16px;
  flex-wrap:wrap; margin-bottom:2px; padding:12px 18px; border-radius:18px; border:1px solid #cfe2f8;
  background:linear-gradient(120deg,#ffffff 0%,#eef6ff 40%,#e2f1ff 60%,#ffffff 100%); background-size:240% 240%;
  box-shadow:0 10px 40px -18px rgba(31,111,209,.35); animation:po-rise .7s var(--smooth) both, po-shift 14s ease-in-out infinite; }}
.po-hero::before, .po-hero::after {{ content:""; position:absolute; border-radius:50%; filter:blur(2px); opacity:.55; pointer-events:none; }}
.po-hero::before {{ width:160px; height:160px; right:18%; top:-70px; background:radial-gradient(circle,#bfe0ff,transparent 70%);
  animation:po-float 9s ease-in-out infinite; }}
.po-hero::after {{ width:110px; height:110px; right:4%; bottom:-50px; background:radial-gradient(circle,#c8f0ff,transparent 70%);
  animation:po-float 11s ease-in-out infinite reverse; }}
.po-brand {{ display:flex; align-items:center; gap:14px; position:relative; z-index:1; }}
.po-logo {{ width:40px; height:40px; border-radius:14px; display:grid; place-items:center; flex:none;
  background:linear-gradient(135deg,{ACCENT},{ACCENT_2}); box-shadow:0 8px 20px -6px rgba(31,111,209,.55); }}
.po-logo svg {{ animation:po-spin 12s linear infinite; }}
.po-hero:hover .po-logo svg {{ animation-duration:2.5s; }}
.po-title {{ font-family:{FONT_DISPLAY}; font-size:1.5rem; font-weight:800; letter-spacing:-.025em; line-height:1.15;
  background:linear-gradient(90deg,{TEXT} 0%,{ACCENT} 55%,{TEXT} 100%); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:po-shine 6s linear infinite; }}
.po-sub {{ color:{MUTED}; font-size:.86rem; margin-top:0; }}
.po-pill {{ position:relative; z-index:1; display:inline-flex; align-items:center; gap:8px; font-size:.76rem; font-weight:600;
  color:{ACCENT}; border:1px solid #b9d5f3; background:rgba(255,255,255,.85); padding:5px 12px; border-radius:999px; }}
.po-dot {{ width:8px; height:8px; border-radius:50%; background:#16b364; box-shadow:0 0 0 0 rgba(22,179,100,.6); animation:po-ping 2s infinite; }}

/* ---------- cards ---------- */
.po-card {{ position:relative; overflow:hidden; background:rgba(255,255,255,.92); border:1px solid {BORDER}; border-radius:18px;
  padding:11px 15px; height:100%; box-shadow:0 1px 2px rgba(15,39,71,.04), 0 8px 24px -16px rgba(15,39,71,.18);
  transition:transform .35s var(--spring), box-shadow .35s var(--smooth), border-color .35s; }}
.po-card.anim {{ animation:po-rise .6s var(--smooth) both; }}
.po-card::before {{ content:""; position:absolute; inset:0 0 auto 0; height:3px; background:linear-gradient(90deg,{ACCENT},{ACCENT_2});
  transform:scaleX(0); transform-origin:left; transition:transform .45s var(--smooth); }}
.po-card:hover {{ transform:translateY(-4px) scale(1.01); border-color:#b9d5f3; box-shadow:0 16px 36px -18px rgba(31,111,209,.45); }}
.po-card:hover::before {{ transform:scaleX(1); }}
[data-testid="stColumn"]:nth-child(2) .po-card.anim, [data-testid="column"]:nth-child(2) .po-card.anim {{ animation-delay:.07s; }}
[data-testid="stColumn"]:nth-child(3) .po-card.anim, [data-testid="column"]:nth-child(3) .po-card.anim {{ animation-delay:.14s; }}
[data-testid="stColumn"]:nth-child(4) .po-card.anim, [data-testid="column"]:nth-child(4) .po-card.anim {{ animation-delay:.21s; }}
[data-testid="stColumn"]:nth-child(5) .po-card.anim, [data-testid="column"]:nth-child(5) .po-card.anim {{ animation-delay:.28s; }}
.po-card .lbl {{ color:{MUTED}; font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }}
.po-card .val {{ color:{TEXT}; font-size:1.5rem; font-weight:700; margin-top:1px; font-family:{FONT_MONO}; letter-spacing:-.02em; }}
.po-card .val-row {{ display:flex; align-items:center; justify-content:space-between; gap:8px; flex-wrap:wrap; }}
.po-card .hint {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis;  color:{TEXT_2}; font-size:.8rem; margin-top:2px; }}

/* ---------- badges, sections, notes ---------- */
.po-badge {{ display:inline-flex; align-items:center; gap:6px; padding:3px 11px; border-radius:999px; font-size:.8rem; font-weight:700;
  border:1.5px solid; background:{SURFACE}; color:{TEXT}; transition:transform .3s var(--spring); }}
.po-badge:hover {{ transform:scale(1.08) rotate(-1.5deg); }}
.po-section {{ position:relative; color:{TEXT}; font-family:{FONT_DISPLAY}; font-weight:700; font-size:1.12rem; margin:6px 0 4px;
  padding-left:14px; animation:po-slide .5s var(--smooth) both; }}
.po-section::before {{ content:""; position:absolute; left:0; top:15%; bottom:15%; width:4px; border-radius:4px;
  background:linear-gradient({ACCENT},{ACCENT_2}); animation:po-grow .6s var(--spring) both; }}
.po-note {{ color:{TEXT_2}; font-size:.84rem; background:linear-gradient(90deg,{SURFACE_2},rgba(255,255,255,.7)); border:1px solid #cfe1f6;
  border-radius:14px; padding:7px 12px; margin:0; animation:po-fade .6s var(--smooth) both; }}
.po-list li {{ color:{TEXT_2}; margin-bottom:4px; }}
div[data-testid="stMetric"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:10px 14px; }}

/* ---------- in-page section navigation ---------- */
[data-testid="stButtonGroup"] {{ display:inline-flex; background:rgba(255,255,255,.9); border:1px solid {BORDER}; border-radius:16px;
  padding:4px; margin:2px 0 0; box-shadow:0 8px 24px -16px rgba(15,39,71,.25); animation:po-rise .5s var(--smooth) both; }}
[data-testid="stButtonGroup"] button {{ border-radius:12px !important; font-weight:600; padding:5px 16px; border-color:transparent;
  transition:background .3s var(--smooth), color .3s, transform .3s var(--spring), box-shadow .3s; }}
[data-testid="stButtonGroup"] button:hover {{ transform:translateY(-2px); background:{SURFACE_2}; }}
[data-testid="stButtonGroup"] button:active {{ transform:scale(.95); }}
[data-testid="stButtonGroup"] button[aria-checked="true"] {{ background:linear-gradient(135deg,{ACCENT},{ACCENT_2}) !important;
  border-color:transparent !important; box-shadow:0 8px 18px -8px rgba(31,111,209,.7); animation:po-pop .4s var(--spring); }}
[data-testid="stButtonGroup"] button[aria-checked="true"] * {{ color:#ffffff !important; }}

/* ---------- buttons, inputs, charts, tables ---------- */
.stButton button, .stDownloadButton button {{ border-radius:12px; font-weight:700; transition:transform .3s var(--spring), box-shadow .3s; }}
.stButton button:hover, .stDownloadButton button:hover {{ transform:translateY(-2px); box-shadow:0 10px 22px -12px rgba(31,111,209,.6); }}
.stButton button:active, .stDownloadButton button:active {{ transform:scale(.96); }}
.stButton button[kind="primary"] {{ background:linear-gradient(135deg,{ACCENT},{ACCENT_2}); border:none; }}
[data-baseweb="select"] > div, [data-baseweb="input"], .stDateInput input {{ border-radius:12px !important; }}
[data-testid="stPlotlyChart"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:16px; padding:4px 4px 0;
  box-shadow:0 8px 28px -20px rgba(15,39,71,.3); animation:po-rise .7s var(--smooth) both; }}
/* chart card: subtle 3D tilt on hover */
[data-testid="stPlotlyChart"] {{ transform:perspective(1400px) rotateX(0) rotateY(0); transform-style:preserve-3d;
  transition:transform .6s var(--smooth), box-shadow .6s var(--smooth); }}
[data-testid="stPlotlyChart"]:hover {{ transform:perspective(1400px) rotateX(1.6deg) rotateY(-1.2deg) translateY(-3px);
  box-shadow:0 30px 50px -28px rgba(31,111,209,.45); }}
/* chart entrance: lines draw themselves, bars and markers grow in */
.js-plotly-plot .scatterlayer .js-line {{ stroke-dasharray:24000; stroke-dashoffset:24000; animation:po-draw 2.2s var(--smooth) forwards; }}
.js-plotly-plot .barlayer .point path {{ transform-box:fill-box; transform-origin:left bottom; animation:po-grow-bar .9s var(--spring) both; }}
.js-plotly-plot .scatterlayer .points path {{ transform-box:fill-box; transform-origin:center; animation:po-pop-pt .6s var(--spring) both; }}
.js-plotly-plot .heatmaplayer image, .js-plotly-plot .shapelayer path {{ animation:po-fade 1s var(--smooth) both; }}
[data-testid="stDataFrame"] {{ border-radius:14px; overflow:hidden; animation:po-fade .6s var(--smooth) both; }}
[data-testid="stAlert"] {{ border-radius:14px; }}

/* ---------- keyframes ---------- */
@keyframes po-rise  {{ from {{ opacity:0; transform:translateY(14px) scale(.985); }} to {{ opacity:1; transform:none; }} }}
@keyframes po-draw  {{ to {{ stroke-dashoffset:0; }} }}
@keyframes po-grow-bar {{ from {{ transform:scale(.15); opacity:0; }} to {{ transform:scale(1); opacity:1; }} }}
@keyframes po-pop-pt {{ from {{ transform:scale(0); }} to {{ transform:scale(1); }} }}
@keyframes po-fade  {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
@keyframes po-slide {{ from {{ opacity:0; transform:translateX(-10px); }} to {{ opacity:1; transform:none; }} }}
@keyframes po-grow  {{ from {{ transform:scaleY(0); }} to {{ transform:scaleY(1); }} }}
@keyframes po-pop   {{ 0% {{ transform:scale(.92); }} 100% {{ transform:scale(1); }} }}
@keyframes po-shift {{ 0%,100% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} }}
@keyframes po-shine {{ to {{ background-position:200% center; }} }}
@keyframes po-spin  {{ to {{ transform:rotate(360deg); }} }}
@keyframes po-float {{ 0%,100% {{ transform:translate(0,0); }} 50% {{ transform:translate(-18px,14px); }} }}
@keyframes po-ping  {{ 0% {{ box-shadow:0 0 0 0 rgba(22,179,100,.55); }} 80%,100% {{ box-shadow:0 0 0 9px rgba(22,179,100,0); }} }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation:none !important; transition:none !important; }} }}
</style>
"""

GEAR = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" '
        'stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 '
        '2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82'
        '.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9'
        'a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09'
        'a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 '
        '1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>')


def setup(title: str, subtitle: str = "", pill: str = "Historical data · MetroPT-3 APU compressor"):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f"""<div class="po-hero"><div class="po-brand"><div class="po-logo">{GEAR}</div><div>
    <div class="po-title">{title}</div><div class="po-sub">{subtitle}</div></div></div>
    <span class="po-pill"><span class="po-dot"></span>{pill}</span></div>""", unsafe_allow_html=True)


def badge(label: str) -> str:
    s = STATUS_OF.get(label, "good")
    return f'<span class="po-badge" style="border-color:{STATUS[s]}"><span style="color:{STATUS[s]}">{ICON[s]}</span>{label}</span>'


def card(label: str, value: str, hint: str = "", status_label: str | None = None, animate: bool = True):
    """animate=False for cards that refresh every few seconds (live simulation), so they don't replay the entrance."""
    b = badge(status_label) if status_label else ""
    st.markdown(f'<div class="po-card{" anim" if animate else ""}"><div class="lbl">{label}</div>'
                f'<div class="val-row"><div class="val">{value}</div>{b}</div>'
                f'<div class="hint">{hint or "&nbsp;"}</div></div>', unsafe_allow_html=True)


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


def subnav(options: list[str], key: str) -> str:
    """In-page navigation: a button bar that shows ONE section of the page at a time (no long scrolling).

    To jump to a section from code: st.session_state[f"_goto_{key}"] = option; st.rerun().
    """
    k = f"nav_{key}"
    goto = st.session_state.pop(f"_goto_{key}", None)
    if goto in options:
        st.session_state[k] = goto
    if st.session_state.get(k) not in options:
        st.session_state[k] = options[0]
    sel = st.segmented_control("Section", options, key=k, label_visibility="collapsed")
    return sel or options[0]


def orbit(fig, frames: int = 48, radius: float = 1.55, height: float = 0.7, start: float = 0.75):
    """Add a '▶ Rotate' button that spins a 3D chart's camera around the scene (Plotly animation frames)."""
    import math
    ang = [start + 2 * math.pi * i / frames for i in range(frames)]          # start at a 3/4 view, not side-on
    fig.frames = [go.Frame(layout=dict(scene_camera=dict(eye=dict(x=radius * math.cos(a), y=radius * math.sin(a), z=height))))
                  for a in ang]
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), scene_domain=dict(x=[0, 1], y=[0, 1]),
                      scene_camera=dict(eye=dict(x=radius * math.cos(start), y=radius * math.sin(start), z=height)), updatemenus=[dict(
        type="buttons", x=0.01, y=0.99, xanchor="left", yanchor="top", direction="left", showactive=False,
        bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=ACCENT),
        buttons=[dict(label="▶ Rotate", method="animate",
                      args=[None, dict(frame=dict(duration=70, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return fig


def play_controls(fig, labels: list[str], duration: int = 120):
    """Play / pause buttons + a time slider for a figure that already has fig.frames named by `labels`."""
    fig.update_layout(updatemenus=[dict(
        type="buttons", x=0, y=-0.16, xanchor="left", yanchor="top", direction="left", showactive=False,
        bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=ACCENT),
        buttons=[dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=duration, redraw=True), transition=dict(duration=duration * .8, easing="cubic-out"),
                                       fromcurrent=True, mode="immediate")]),
                 dict(label="⏸", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=len(labels) - 1, x=0.12, len=0.88, y=-0.1, pad=dict(t=0), currentvalue=dict(visible=False),
                      bgcolor=SURFACE_2, bordercolor=BORDER, activebgcolor=ACCENT, font=dict(color=MUTED, size=9),
                      steps=[dict(label=l, method="animate", args=[[l], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                             for l in labels])])
    return fig
