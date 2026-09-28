"""Camada visual do BingX Analytics: tema, componentes HTML e template de gráficos."""
from __future__ import annotations

import html
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st

GREEN, RED, ACCENT, AMBER, BLUE, GREY = "#22c55e", "#ef4444", "#8b5cf6", "#f59e0b", "#38bdf8", "#94a3b8"
PINK, TEAL = "#ec4899", "#14b8a6"
GRID = "rgba(148,163,184,.10)"
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stApp {font-family: Inter, -apple-system, 'Segoe UI', Roboto, sans-serif;}
.stApp {background: radial-gradient(1200px 600px at 8% -10%, rgba(139,92,246,.13), transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, rgba(56,189,248,.08), transparent 55%), #0b0f19;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1560px;}
header[data-testid="stHeader"] {background: transparent;}
#MainMenu, footer, .stAppDeployButton {visibility: hidden; display: none;}

/* ── hero ── */
.hero {display:flex; align-items:center; justify-content:space-between; gap:18px; flex-wrap:wrap;
       padding: 20px 26px; margin-bottom: 14px; border-radius: 18px;
       background: linear-gradient(120deg, rgba(139,92,246,.22), rgba(56,189,248,.10) 55%, rgba(19,26,42,.6));
       border: 1px solid rgba(148,163,184,.18); box-shadow: 0 10px 40px -18px rgba(139,92,246,.55);}
.hero h1 {margin:0; font-size: 1.75rem; font-weight: 800; letter-spacing: -.02em; padding:0;}
.hero .sub {opacity:.72; font-size:.9rem; margin-top:4px;}
.hero .net {text-align:right;}
.hero .net .lbl {font-size:.7rem; letter-spacing:.09em; text-transform:uppercase; opacity:.65;}
.hero .net .val {font-family:'JetBrains Mono',monospace; font-size:2.1rem; font-weight:600; line-height:1.1;}
.hero .net .brl {font-size:.82rem; opacity:.7; font-family:'JetBrains Mono',monospace;}
.chip {display:inline-block; padding:3px 10px; margin: 6px 6px 0 0; border-radius:999px; font-size:.72rem; font-weight:600;
       background: rgba(148,163,184,.12); border:1px solid rgba(148,163,184,.22); color:#cbd5e1;}
.chip.ok {background:rgba(34,197,94,.12); border-color:rgba(34,197,94,.35); color:#86efac;}
.chip.warn {background:rgba(245,158,11,.12); border-color:rgba(245,158,11,.35); color:#fcd34d;}
.chip.demo {background:rgba(139,92,246,.18); border-color:rgba(139,92,246,.5); color:#c4b5fd;}

/* ── KPI cards ── */
.kpi {position:relative; overflow:hidden; border-radius:14px; padding: 13px 16px 12px 18px; height: 100%;
      background: linear-gradient(150deg, rgba(30,41,59,.72), rgba(19,26,42,.72)); border:1px solid rgba(148,163,184,.16);
      transition: transform .15s ease, border-color .15s ease;}
.kpi:hover {transform: translateY(-2px); border-color: rgba(139,92,246,.55);}
.kpi::before {content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:#38bdf8;}
.kpi.good::before {background:#22c55e;} .kpi.bad::before {background:#ef4444;}
.kpi.warn::before {background:#f59e0b;} .kpi.accent::before {background:#8b5cf6;}
.kpi .l {font-size:.68rem; font-weight:600; letter-spacing:.07em; text-transform:uppercase; opacity:.62; white-space:nowrap;
         overflow:hidden; text-overflow:ellipsis;}
.kpi .v {font-family:'JetBrains Mono',monospace; font-size:1.45rem; font-weight:600; margin-top:3px; letter-spacing:-.02em;}
.kpi.good .v {color:#4ade80;} .kpi.bad .v {color:#f87171;} .kpi.warn .v {color:#fbbf24;}
.kpi .s {font-size:.72rem; opacity:.6; margin-top:1px; min-height:1em;}
.kgap {height: 10px;}

/* ── insights ── */
.ins {border-left: 4px solid #38bdf8; background: linear-gradient(90deg, rgba(148,163,184,.09), rgba(148,163,184,.03));
      border-radius: 12px; padding: 12px 16px; margin: 9px 0; font-size:.93rem; line-height:1.5;}
.ins.good {border-color:#22c55e;} .ins.warn {border-color:#f59e0b;} .ins.bad {border-color:#ef4444;} .ins.info {border-color:#38bdf8;}
.ins .t {display:flex; align-items:center; gap:8px; font-weight:700; margin-bottom:2px;}
.ins .tag {font-size:.62rem; padding:1px 7px; border-radius:999px; letter-spacing:.06em; text-transform:uppercase; font-weight:700;}
.ins.good .tag {background:rgba(34,197,94,.18); color:#86efac;} .ins.warn .tag {background:rgba(245,158,11,.18); color:#fcd34d;}
.ins.bad .tag {background:rgba(239,68,68,.18); color:#fca5a5;} .ins.info .tag {background:rgba(56,189,248,.16); color:#7dd3fc;}

/* ── seções, abas, containers ── */
.sec {display:flex; align-items:baseline; gap:10px; margin: 22px 0 6px;}
.sec h3 {margin:0 !important; padding:0 !important; font-size:1.15rem !important; font-weight:700 !important; letter-spacing:-.01em;}
.sec span {font-size:.8rem; opacity:.55;}
.small {font-size:.82rem; opacity:.7;}
[role="tablist"] {gap:6px !important; border-bottom:1px solid rgba(148,163,184,.14); padding-bottom:8px; flex-wrap: wrap;}
[role="tab"] {height:40px; padding: 0 16px !important; border-radius: 10px !important; background: rgba(148,163,184,.08) !important;
              font-weight:600; border:1px solid transparent !important; transition: border-color .15s ease;}
[role="tab"]:hover {border-color: rgba(139,92,246,.45) !important;}
[role="tab"][aria-selected="true"] {background: linear-gradient(120deg, rgba(139,92,246,.42), rgba(56,189,248,.20)) !important;
                                    border-color: rgba(139,92,246,.6) !important;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display:none !important;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px; border-color: rgba(148,163,184,.14) !important;
      background: rgba(19,26,42,.42);}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f1524, #0b0f19); border-right:1px solid rgba(148,163,184,.12);}
section[data-testid="stSidebar"] h1 {font-size:1.25rem; font-weight:800; letter-spacing:-.02em;}
div[data-testid="stDataFrame"] {border-radius: 12px; overflow:hidden;}
.stDownloadButton button, .stButton button {border-radius: 10px; font-weight:600;}
@media (max-width: 900px) {.hero .net {text-align:left;} .kpi .v {font-size:1.2rem;}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ───────────────────────────── componentes ─────────────────────────────
def tone_of(x: float | None, invert: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == 0:
        return ""
    good = x > 0
    return "good" if (good != invert) else "bad"


def kpi(col, label: str, value: str, sub: str = "", tone: str = "", help: str = "") -> None:
    tip = f' title="{html.escape(help)}"' if help else ""
    size = ' style="font-size:1.08rem"' if len(re.sub(r"<[^>]+>", "", str(value))) > 13 else ""
    col.markdown(
        f'<div class="kpi {tone}"{tip}><div class="l">{html.escape(label)}</div><div class="v"{size}>{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple], gap: bool = True) -> None:
    """items: (label, value, sub, tone, help) — o que faltar vira vazio."""
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        it = list(it) + [""] * (5 - len(it))
        kpi(c, *it[:5])
    if gap:
        st.markdown('<div class="kgap"></div>', unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="sec"><h3>{title}</h3><span>{sub}</span></div>', unsafe_allow_html=True)


def insight(level: str, title: str, text: str) -> None:
    tag = {"good": "Positivo", "warn": "Atenção", "bad": "Alerta", "info": "Info"}.get(level, "Info")
    st.markdown(f'<div class="ins {level}"><div class="t">{title}<span class="tag">{tag}</span></div>{text}</div>', unsafe_allow_html=True)


def chips(items: list[tuple[str, str]]) -> str:
    return "".join(f'<span class="chip {k}">{html.escape(t)}</span>' for t, k in items)


def hero(title: str, subtitle: str, net_txt: str, net_tone: str, brl_txt: str, chip_html: str) -> None:
    color = {"good": "#4ade80", "bad": "#f87171"}.get(net_tone, "#e5e7eb")
    brl = f'<div class="brl">{brl_txt}</div>' if brl_txt else ""
    st.markdown(
        f'<div class="hero"><div><h1>{title}</h1><div class="sub">{subtitle}</div>{chip_html}</div>'
        f'<div class="net"><div class="lbl">PnL líquido do período</div><div class="val" style="color:{color}">{net_txt}</div>{brl}</div></div>',
        unsafe_allow_html=True,
    )


# ───────────────────────────── gráficos ─────────────────────────────
def colors(series) -> list[str]:
    return [GREEN if v >= 0 else RED for v in series]


def show(fig: go.Figure, height: int = 420, key: str | None = None, legend: bool = True, margin: dict | None = None) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        separators=",.",
        font=dict(family=FONT, size=12, color="#cbd5e1"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=margin or dict(l=8, r=8, t=(56 if (fig.layout.title.text or legend) else 16), b=8),
        title=dict(font=dict(size=14, color="#e2e8f0"), x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)") if legend else dict(visible=False),
        hoverlabel=dict(bgcolor="#131a2a", bordercolor="#334155", font=dict(family=FONT, size=12)),
        colorway=[ACCENT, BLUE, AMBER, GREEN, PINK, TEAL, RED],
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="rgba(148,163,184,.25)", showline=False)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(148,163,184,.25)", showline=False)
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def zero_line(fig: go.Figure) -> go.Figure:
    fig.add_hline(y=0, line_color=GREY, line_dash="dot", line_width=1)
    return fig
