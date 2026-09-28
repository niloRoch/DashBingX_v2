"""
BingX • Trading Analytics v2 — dashboard Streamlit
--------------------------------------------------
Lê os exports (CSV/JSON) de `bingx_futures_income` e `bingx_operations`
e entrega PnL, análise de trades, custos, timing, risco, comportamento e insights.

Rodar:  streamlit run app.py
Dados:  envie pela barra lateral OU coloque os arquivos na pasta ./data
        (ou aponte a variável de ambiente BINGX_DATA_DIR). Sem dados? Use o modo demonstração.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core import (DOW, add_time, br, build_events, build_roundtrips, clean_income, clean_ops, daily_frame, drawdown_episodes,
                  equity_frame, fmt_min, monte_carlo, open_after_previous, pct, prepare_roundtrips, read_table, session_of, sg,
                  spot_average_cost, stats_by, trade_stats, ulcer_index, var_cvar)
from demo import make_demo
from ui import (ACCENT, AMBER, BLUE, GREEN, GREY, PINK, RED, TEAL, chips, colors, hero, inject_css, insight, kpi_row, section, show,
                tone_of, zero_line)

st.set_page_config(page_title="BingX • Trading Analytics", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
inject_css()

FMT = {"Realizado": "{:+.2f}", "Taxas": "{:+.2f}", "Funding": "{:+.4f}", "Líquido": "{:+.2f}", "Win rate": "{:.0%}"}


# ───────────────────────────── carga de dados ─────────────────────────────
def _bytes(src) -> bytes:
    return src.getvalue() if hasattr(src, "getvalue") else Path(src).read_bytes()


@st.cache_data(show_spinner="Lendo income…")
def load_income(b: bytes, ext: str) -> pd.DataFrame:
    return clean_income(read_table(b, ext))


@st.cache_data(show_spinner="Lendo operations…")
def load_ops(b: bytes, ext: str):
    return clean_ops(read_table(b, ext))


@st.cache_data(show_spinner=False)
def load_demo(today: str):
    a, b = make_demo(end=today)
    return clean_income(a), clean_ops(b)


@st.cache_data(show_spinner="Reconstruindo posições (FIFO)…")
def cached_roundtrips(fills: pd.DataFrame, tz: str) -> pd.DataFrame:
    return prepare_roundtrips(build_roundtrips(fills), tz)


@st.cache_data(show_spinner=False)
def run_mc(net: np.ndarray, n_tr: int, n_sims: int, capital: float, ruin: float) -> dict:
    return monte_carlo(net, n_tr, n_sims, capital, ruin)


def find_local(prefix: str) -> Path | None:
    dirs = [os.environ.get("BINGX_DATA_DIR"), Path(__file__).parent / "data", Path(__file__).parent, Path.cwd() / "data", Path.cwd()]
    for d in dirs:
        if not d:
            continue
        d = Path(d)
        for ext in (".json", ".csv"):  # JSON primeiro: traz o tipo de ordem (MARKET / TAKE_PROFIT)
            p = d / f"{prefix}{ext}"
            if p.exists():
                return p
    return None


# ───────────────────────────── barra lateral: dados ─────────────────────────────
def _enable_demo():
    st.session_state["demo"] = True


with st.sidebar:
    st.markdown("# 📈 BingX Analytics")
    st.caption("v2 · PnL, trades, risco e comportamento")
    st.subheader("Dados")
    up_inc = st.file_uploader("Futures income (CSV/JSON)", type=["csv", "json"], key="up_inc")
    up_ops = st.file_uploader("Operations (CSV/JSON)", type=["csv", "json"], key="up_ops")
    demo = st.toggle("Usar dados de demonstração", key="demo", help="Dados sintéticos no mesmo formato do export da BingX — bom para explorar o painel.")

src_inc = up_inc or find_local("bingx_futures_income")
src_ops = up_ops or find_local("bingx_operations")
use_demo = bool(demo) and up_inc is None

if not use_demo and src_inc is None:
    hero("BingX • Trading Analytics", "Análise completa dos seus exports de futuros e spot", "–", "", "",
         chips([("v2", "")]))
    st.info("Envie o arquivo **bingx_futures_income** (CSV ou JSON) na barra lateral — ou coloque os arquivos na pasta `data/`.")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**`bingx_futures_income`** — fonte principal (PnL, taxas, funding)")
            st.code("symbol, incomeType, income, timestamp, info, tranId", language="text")
            st.caption("incomeType: REALIZED_PNL · TRADING_FEE · FUNDING_FEE. Timestamp em milissegundos.")
    with b:
        with st.container(border=True):
            st.markdown("**`bingx_operations`** — fills de futuros + spot (opcional)")
            st.code("market, symbol, side, positionSide, price, quantity,\nquoteQty, fee, timestamp, orderId, tradeId, role, raw.type", language="text")
            st.caption("Prefira o JSON: traz o tipo da ordem (MARKET / TAKE_PROFIT) em `raw.type`.")
    st.button("▶ Explorar com dados de demonstração", on_click=_enable_demo, type="primary")
    st.stop()

try:
    if use_demo:
        inc_raw, (fills_raw, spot_raw, spot_dups) = load_demo(pd.Timestamp.now().strftime("%Y-%m-%d"))
    else:
        inc_raw = load_income(_bytes(src_inc), Path(getattr(src_inc, "name", str(src_inc))).suffix.lower())
        if src_ops is not None:
            fills_raw, spot_raw, spot_dups = load_ops(_bytes(src_ops), Path(getattr(src_ops, "name", str(src_ops))).suffix.lower())
        else:
            fills_raw, spot_raw, spot_dups = pd.DataFrame(), pd.DataFrame(), 0
except Exception as exc:  # arquivo fora do formato esperado
    st.error(f"Não consegui ler os arquivos: {exc}")
    st.stop()

# ───────────────────────────── barra lateral: filtros ─────────────────────────────
with st.sidebar:
    st.subheader("Filtros")
    tz = st.selectbox("Fuso horário", ["America/Sao_Paulo", "UTC", "America/New_York", "Europe/London", "Asia/Singapore"], index=0)
    inc_all = add_time(inc_raw, tz)
    dmin, dmax = inc_all["day"].min().date(), inc_all["day"].max().date()
    preset = st.selectbox("Período", ["Tudo", "Últimos 7 dias", "Últimos 30 dias", "Últimos 90 dias", "Mês atual", "Ano atual", "Personalizado"], index=0)
    if preset == "Tudo":
        d0, d1 = dmin, dmax
    elif preset == "Últimos 7 dias":
        d0, d1 = max(dmin, dmax - pd.Timedelta(days=6)), dmax
    elif preset == "Últimos 30 dias":
        d0, d1 = max(dmin, dmax - pd.Timedelta(days=29)), dmax
    elif preset == "Últimos 90 dias":
        d0, d1 = max(dmin, dmax - pd.Timedelta(days=89)), dmax
    elif preset == "Mês atual":
        d0, d1 = max(dmin, dmax.replace(day=1)), dmax
    elif preset == "Ano atual":
        d0, d1 = max(dmin, dmax.replace(month=1, day=1)), dmax
    else:
        rng = st.date_input("Intervalo", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        d0, d1 = (rng[0], rng[1]) if isinstance(rng, (tuple, list)) and len(rng) == 2 else (dmin, dmax)
    syms_all = sorted(inc_all["symbol"].unique())
    syms = st.multiselect("Símbolos", syms_all, default=syms_all)
    sides = st.multiselect("Lado", ["LONG", "SHORT"], default=["LONG", "SHORT"], help="Filtra fechamentos e posições. Os totais de income (PnL, taxas, funding) não dependem do lado.")
    st.subheader("Capital e câmbio")
    capital = st.number_input("Capital inicial (USDT) — opcional", min_value=0.0, value=0.0, step=50.0,
                              help="Se informado, o app calcula retorno %, drawdown %, Sharpe e risco de ruína sobre esse capital.")
    brl_rate = st.number_input("Cotação USDT/BRL — opcional", min_value=0.0, value=0.0, step=0.05, format="%.2f",
                               help="Se informada, mostra o PnL também em reais.")
    st.caption(f"Fuso: {tz} · dados de {dmin:%d/%m/%Y} a {dmax:%d/%m/%Y}")

# ───────────────────────────── filtros ─────────────────────────────
t0, t1 = pd.Timestamp(d0), pd.Timestamp(d1)
inc = inc_all[(inc_all["day"] >= t0) & (inc_all["day"] <= t1) & inc_all["symbol"].isin(syms)].copy()
events = build_events(inc)
if not events.empty:
    events = events[events["side"].isin(sides)].reset_index(drop=True)

fills_all = add_time(fills_raw, tz) if len(fills_raw) else pd.DataFrame()
rt_all = cached_roundtrips(fills_all, tz) if not fills_all.empty else pd.DataFrame()
rt = pd.DataFrame()
if not rt_all.empty:
    rt = rt_all[(rt_all["day"] >= t0) & (rt_all["day"] <= t1) & rt_all["symbol"].isin(syms) & rt_all["side"].isin(sides)].copy()
fills = (fills_all[(fills_all["day"] >= t0) & (fills_all["day"] <= t1) & fills_all["symbol"].isin(syms)].copy()
         if not fills_all.empty else pd.DataFrame())

if inc.empty:
    st.warning("Nenhum dado no filtro selecionado.")
    st.stop()

# ───────────────────────────── agregados globais ─────────────────────────────
gross_total = inc.loc[inc["incomeType"] == "REALIZED_PNL", "income"].sum()
fees_total = inc.loc[inc["incomeType"] == "TRADING_FEE", "income"].sum()
fund_total = inc.loc[inc["incomeType"] == "FUNDING_FEE", "income"].sum()
net_total = inc["income"].sum()
st_ev = trade_stats(events["net"]) if not events.empty else {}
eq = equity_frame(inc, capital)
daily = daily_frame(inc, events)
max_dd = eq["dd"].min()
max_dd_pct = eq["dd_pct"].min() if capital > 0 else np.nan
cur_dd = eq["dd"].iloc[-1]
pos_days = (daily["net"] > 0).mean()
dr = daily["net"] / (capital + daily["net"].cumsum().shift(fill_value=0)) if capital > 0 else pd.Series(dtype=float)
recovery = net_total / abs(max_dd) if max_dd < 0 else np.nan
eq_day = capital + daily["net"].cumsum()  # equity no fechamento de cada dia
days_uw = float(((eq_day - eq_day.cummax().clip(lower=capital)) < 0).mean())  # % de dias fechando abaixo do topo anterior
var95, cvar95 = var_cvar(daily["net"])
brl_txt = f"≈ R$ {br(net_total * brl_rate)}" if brl_rate > 0 else ""

# ───────────────────────────── cabeçalho ─────────────────────────────
status = [(f"{d0:%d/%m/%Y} → {d1:%d/%m/%Y}", ""), (f"{len(inc):,} lançamentos".replace(",", "."), ""), (f"{len(events)} fechamentos", ""),
          (f"{len(syms)} símbolo(s)", "")]
if use_demo:
    status.insert(0, ("Modo demonstração", "demo"))
if not fills_all.empty and fills_all["dt"].min() > inc_all["dt"].min() + pd.Timedelta(days=1):
    status.append(("fills cobrem período menor que o income", "warn"))
elif not fills_all.empty:
    status.append(("fills e income alinhados", "ok"))
hero("BingX • Trading Analytics", "Futuros perpétuos · valores em USDT", sg(net_total) + " USDT", tone_of(net_total), brl_txt, chips(status))

tabs = st.tabs(["📊 Visão geral", "💡 Insights", "💰 PnL", "🎯 Trades", "🕐 Timing", "🔬 Posições (fills)", "🧾 Custos", "🛡️ Risco", "🪙 Spot", "🗂️ Dados"])


# ═════════════════════════ 1. VISÃO GERAL ═════════════════════════
with tabs[0]:
    wr, be = st_ev.get("win_rate", np.nan), st_ev.get("breakeven_wr", np.nan)
    pf = st_ev.get("pf", np.nan)
    kpi_row([
        ("PnL líquido", sg(net_total), brl_txt.replace("≈ ", "") if brl_txt else "realizado + taxas + funding", tone_of(net_total), "Realizado + taxas + funding"),
        ("PnL realizado (bruto)", sg(gross_total), "", tone_of(gross_total)),
        ("Taxas", sg(fees_total), (f"{pct(-fees_total / gross_total)} do bruto" if -fees_total / gross_total <= 5 else "muito acima do bruto") if gross_total > 0 else "", ""),
        ("Funding", sg(fund_total, 4), "a favor" if fund_total > 0 else "contra" if fund_total < 0 else "", tone_of(fund_total)),
        ("Win rate", pct(wr), f"{st_ev.get('wins', 0)} ganhos · {st_ev.get('losses', 0)} perdas" if st_ev else "",
         "" if np.isnan(wr) else ("good" if wr >= be else "warn"), "Fechamentos com PnL líquido (após taxa de fechamento) > 0"),
        ("Profit factor", br(pf) if st_ev else "–", "lucro bruto ÷ perda bruta", "" if not st_ev or np.isnan(pf) else ("good" if pf > 1 else "bad")),
    ])
    best_day, worst_day = daily["net"].idxmax(), daily["net"].idxmin()
    kpi_row([
        ("Fechamentos", str(st_ev.get("n", 0)), f"{len(daily)} dias com atividade"),
        ("Payoff", br(st_ev.get("payoff", np.nan)), f"equilíbrio: win rate {pct(be)}" if not np.isnan(be) else "ganho ÷ perda médios"),
        ("Expectativa", sg(st_ev.get("expectancy", np.nan), 3), "por fechamento", tone_of(st_ev.get("expectancy", np.nan))),
        ("Max drawdown", br(max_dd) + (f" ({pct(max_dd_pct)})" if capital > 0 else ""), f"agora: {br(cur_dd)}", "bad" if max_dd < 0 else ""),
        ("Melhor dia", sg(daily["net"].max()), f"{best_day:%d/%m/%Y}", "good"),
        ("Pior dia", sg(daily["net"].min()), f"{worst_day:%d/%m/%Y}", "bad" if daily["net"].min() < 0 else ""),
    ])
    if capital > 0:
        sharpe = dr.mean() / dr.std(ddof=1) * np.sqrt(365) if len(dr) > 2 and dr.std(ddof=1) > 0 else np.nan
        downside = dr[dr < 0].std(ddof=1)
        sortino = dr.mean() / downside * np.sqrt(365) if downside and downside > 0 else np.nan
        kpi_row([
            ("Retorno sobre capital", pct(net_total / capital, 2), "", tone_of(net_total)),
            ("Sharpe", br(sharpe), "diário, anualizado", tone_of(sharpe) if not np.isnan(sharpe) else ""),
            ("Sortino", br(sortino)),
            ("Calmar", br(net_total / abs(max_dd)) if max_dd < 0 else "–", "", "", "PnL total / |drawdown máximo| (não anualizado)"),
            ("Volatilidade diária", pct(dr.std(ddof=1))),
            ("Dias positivos", pct(pos_days)),
        ])
    kpi_row([
        ("SQN", br(st_ev.get("sqn", np.nan)), "qualidade do sistema (Van Tharp)", "", "Expectativa ÷ desvio × √n. Acima de 2 é bom; abaixo de 1 é ruído. Precisa de amostra."),
        ("Recovery factor", br(recovery), "PnL ÷ |max drawdown|", tone_of(recovery) if not np.isnan(recovery) else ""),
        ("Média por dia", sg(daily["net"].mean()), f"mediana {sg(daily['net'].median())}", tone_of(daily["net"].mean())),
        ("VaR 95% (dia)", sg(var95), "pior dia em 20", "", "Perda diária que só é excedida em 5% dos dias (histórico)."),
        ("CVaR 95% (dia)", sg(cvar95), "média dos piores 5%"),
        ("Sequência atual", (f"{st_ev['cur_streak']} ganhos" if st_ev.get("cur_streak", 0) > 0 else f"{-st_ev['cur_streak']} perdas" if st_ev.get("cur_streak", 0) < 0 else "neutra") if st_ev else "–",
         f"máx: {st_ev.get('streak_w', 0)} G / {st_ev.get('streak_l', 0)} P" if st_ev else ""),
    ], gap=False)

    section("Curva de PnL e drawdown", "acumulado por lançamento")
    with st.container(border=True):
        show_comp = st.toggle("Mostrar componentes (bruto, taxas, funding)", value=False)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
        fig.add_trace(go.Scatter(x=eq["dt"], y=eq["equity"] if capital > 0 else eq["cum_net"], name="PnL líquido acumulado" if capital == 0 else "Equity",
                                 line=dict(color=ACCENT, width=2.6, shape="hv"), fill="tozeroy" if capital == 0 else None,
                                 fillcolor="rgba(139,92,246,.13)"), 1, 1)
        if show_comp:
            for col, nm, cl in [("cum_gross", "Realizado bruto", GREEN), ("cum_fees", "Taxas", RED), ("cum_funding", "Funding", AMBER)]:
                fig.add_trace(go.Scatter(x=eq["dt"], y=eq[col], name=nm, line=dict(color=cl, width=1.4, shape="hv")), 1, 1)
        fig.add_trace(go.Scatter(x=eq["dt"], y=eq["dd"], name="Drawdown", line=dict(color=RED, width=1.4, shape="hv"),
                                 fill="tozeroy", fillcolor="rgba(239,68,68,.25)"), 2, 1)
        if max_dd < 0:
            i = eq["dd"].idxmin()
            fig.add_trace(go.Scatter(x=[eq.loc[i, "dt"]], y=[max_dd], mode="markers+text", text=[f"máx {br(max_dd)}"], textposition="bottom center",
                                     marker=dict(color=RED, size=9), showlegend=False), 2, 1)
        fig.update_yaxes(title_text="USDT", row=1, col=1)
        fig.update_yaxes(title_text="DD (USDT)", row=2, col=1)
        show(fig, 540)

    a, b = st.columns([1.35, 1])
    with a:
        with st.container(border=True):
            d = daily.reset_index()
            fig = go.Figure(go.Bar(x=d["day"], y=d["net"], marker_color=colors(d["net"]), name="PnL do dia",
                                   hovertemplate="%{x|%d/%m/%Y}<br>%{y:+.2f} USDT<extra></extra>"))
            fig.add_trace(go.Scatter(x=d["day"], y=d["net"].cumsum(), name="Acumulado", yaxis="y2", line=dict(color=AMBER, width=2)))
            fig.update_layout(title="PnL diário e acumulado", yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Acumulado"))
            show(fig, 380)
    with b:
        with st.container(border=True):
            fig = go.Figure(go.Waterfall(
                x=["Realizado bruto", "Taxas", "Funding", "Líquido"], measure=["relative", "relative", "relative", "total"],
                y=[gross_total, fees_total, fund_total, net_total], text=[sg(v) for v in [gross_total, fees_total, fund_total, net_total]],
                increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED), totals=dict(marker_color=ACCENT), connector=dict(line=dict(color=GREY))))
            fig.update_layout(title="De onde vem o resultado")
            show(fig, 380, legend=False)

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            ps_ = inc.groupby("symbol")["income"].sum().sort_values()
            fig = go.Figure(go.Bar(x=ps_.values, y=ps_.index, orientation="h", marker_color=colors(ps_.values),
                                   text=[sg(v) for v in ps_.values], textposition="outside", cliponaxis=False))
            fig.update_layout(title="Contribuição por símbolo (PnL líquido)")
            show(fig, max(260, 44 * len(ps_) + 90), legend=False)
    with b:
        with st.container(border=True):
            if not events.empty:
                sd = events.groupby("side")["net"].agg(["sum", "count"]).reindex(["LONG", "SHORT"]).fillna(0)
                fig = go.Figure(go.Bar(x=sd.index, y=sd["sum"], marker_color=colors(sd["sum"]), text=[f"{sg(v)}<br>{int(n)} fech." for v, n in zip(sd["sum"], sd["count"])],
                                       textposition="outside", cliponaxis=False))
                fig.update_layout(title="Long × Short (PnL líquido dos fechamentos)")
                show(fig, max(260, 44 * len(ps_) + 90), legend=False)
            else:
                st.info("Sem fechamentos no filtro.")


# ═════════════════════════ 2. INSIGHTS ═════════════════════════
def quality_scores() -> dict[str, float]:
    """Cinco dimensões 0–100 (heurística transparente; ver legenda na aba)."""
    sc: dict[str, float] = {}
    pf_ = st_ev.get("pf", np.nan)
    sc["Rentabilidade"] = 0.0 if np.isnan(pf_) else 100.0 if np.isinf(pf_) else float(np.clip((pf_ - 0.8) / 1.0, 0, 1) * 100)
    sc["Consistência"] = float(np.clip((pos_days - 0.35) / 0.30, 0, 1) * 100)
    rf = net_total / abs(max_dd) if max_dd < 0 else (3.0 if net_total > 0 else 0.0)
    sc["Controle de risco"] = float(np.clip(rf / 3, 0, 1) * 100)
    sc["Eficiência de custos"] = float(np.clip((0.6 - (-fees_total / gross_total)) / 0.5, 0, 1) * 100) if gross_total > 0 else 0.0
    sc["Amostra"] = float(np.clip((st_ev.get("n", 0) - 10) / 90, 0, 1) * 100)
    return sc


def build_insights() -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    n = st_ev.get("n", 0)
    # 1 resultado
    out.append(("good" if net_total > 0 else "bad", "Resultado do período",
                f"PnL líquido de <b>{sg(net_total)} USDT</b>: realizado bruto {sg(gross_total)}, taxas {sg(fees_total)} e funding {sg(fund_total, 4)}."
                + (f" Isso equivale a {pct(net_total / capital, 2)} do capital informado." if capital > 0 else "")))
    if n < 30:
        out.append(("info", "Tamanho da amostra", f"Há apenas {n} fechamentos no filtro. Estatísticas como win rate, payoff e Kelly oscilam muito com amostras pequenas — use como indício, não como verdade."))
    if not st_ev:
        return out
    # 2 taxas
    if gross_total > 0:
        share = -fees_total / gross_total
        lvl = "bad" if share > 0.5 else "warn" if share > 0.25 else "good"
        out.append((lvl, "Peso das taxas", f"As taxas consumiram {pct(share)} do lucro bruto realizado. "
                    + ("Isso é alto: cada operação precisa render bastante só para pagar o custo. Menos operações, alvos maiores ou ordens limit (maker) ajudam." if share > 0.25 else "Custo sob controle em relação ao lucro bruto.")))
    elif fees_total < 0:
        out.append(("warn", "Peso das taxas", f"O bruto realizado é {sg(gross_total)}, e as taxas ainda somam {sg(fees_total)}: o custo de operar está ampliando o prejuízo."))
    # 3 win rate x payoff
    wr_, be_ = st_ev["win_rate"], st_ev["breakeven_wr"]
    if not np.isnan(be_):
        ok = wr_ >= be_
        out.append(("good" if ok else "warn", "Win rate × payoff",
                    f"Win rate de {pct(wr_)} com payoff {br(st_ev['payoff'])} (ganho médio {sg(st_ev['avg_win'])} / perda média {sg(st_ev['avg_loss'])}). "
                    f"O win rate de equilíbrio para esse payoff é {pct(be_)} — você está {'acima' if ok else 'abaixo'} dele."
                    + ("" if ok else " Ou o win rate precisa subir, ou as perdas precisam ficar menores em relação aos ganhos.")))
    # 4 cauda de perda
    if st_ev["avg_win"] > 0 and st_ev["worst"] < 0:
        ratio = abs(st_ev["worst"]) / st_ev["avg_win"]
        if ratio >= 3:
            out.append(("warn", "Perda de cauda", f"A pior perda ({sg(st_ev['worst'])}) equivale a {br(ratio, 1)}× o ganho médio. Uma única operação ruim apaga vários acertos — vale ter stop e tamanho máximo por trade."))
    # 5 concentração
    ev = events.sort_values("net", ascending=False)
    if st_ev["gross_profit"] > 0 and len(ev) >= 8:
        top3 = ev["net"].head(3).clip(lower=0).sum()
        conc = top3 / st_ev["gross_profit"]
        if conc > 0.4:
            out.append(("warn", "Lucro concentrado", f"As 3 melhores operações respondem por {pct(conc)} do lucro bruto. Sem elas, o resultado dos fechamentos seria {sg(st_ev['total'] - top3)}. O edge pode depender de poucos eventos."))
    # 6 long vs short
    sd = events.groupby("side")["net"].agg(["count", "sum", "mean"])
    if set(sd.index) == {"LONG", "SHORT"} and sd["count"].min() >= 3:
        wr_side = events.groupby("side")["net"].apply(lambda s: (s > 0).mean())
        better = sd["sum"].idxmax()
        out.append(("info", "Long × Short", f"{better} rende mais: LONG {sg(sd.loc['LONG', 'sum'])} ({int(sd.loc['LONG', 'count'])} fech., win rate {pct(wr_side['LONG'])}) vs "
                    f"SHORT {sg(sd.loc['SHORT', 'sum'])} ({int(sd.loc['SHORT', 'count'])} fech., win rate {pct(wr_side['SHORT'])})."))
    # 7 horário / dia
    h = events.groupby("hour")["net"].agg(["count", "sum"])
    h = h[h["count"] >= 3]
    if len(h) >= 3:
        bh, wh = h["sum"].idxmax(), h["sum"].idxmin()
        if h.loc[wh, "sum"] < 0:
            out.append(("info", "Horários", f"Melhor faixa: {bh:02d}h ({sg(h.loc[bh, 'sum'])}, {int(h.loc[bh, 'count'])} fech.). Pior: {wh:02d}h ({sg(h.loc[wh, 'sum'])}, {int(h.loc[wh, 'count'])} fech.) — horário no fuso {tz}."))
    wdg = events.groupby("wd")["net"].agg(["count", "sum"])
    wdg = wdg[wdg["count"] >= 3]
    if len(wdg) >= 3 and wdg["sum"].min() < 0:
        out.append(("info", "Dias da semana", f"Melhor dia: {DOW[wdg['sum'].idxmax()]} ({sg(wdg['sum'].max())}). Pior: {DOW[wdg['sum'].idxmin()]} ({sg(wdg['sum'].min())})."))
    # sessões (novo)
    ev_s = events.assign(sessao=events["hour_utc"].map(session_of))
    gs = ev_s.groupby("sessao")["net"].agg(["count", "sum"])
    gs = gs[gs["count"] >= 5]
    if len(gs) >= 2 and gs["sum"].min() < 0:
        out.append(("info", "Sessões de mercado", f"Melhor sessão: {gs['sum'].idxmax()} ({sg(gs['sum'].max())}). Pior: {gs['sum'].idxmin()} ({sg(gs['sum'].min())}). Sessões em UTC: Ásia 00–08, Europa 08–13, EUA 13–21."))
    # 8 sequências
    cur = st_ev["cur_streak"]
    out.append(("warn" if st_ev["streak_l"] >= 5 else "info", "Sequências",
                f"Maior sequência de ganhos: {st_ev['streak_w']} · de perdas: {st_ev['streak_l']}. Sequência atual: "
                + (f"{cur} ganhos seguidos." if cur > 0 else f"{-cur} perdas seguidas." if cur < 0 else "neutra.")))
    # 9 overtrading
    if len(daily) >= 8 and daily["trades"].max() >= 3:
        thr = daily["trades"].quantile(0.75)
        hi, lo = daily[daily["trades"] >= max(thr, 2)], daily[daily["trades"] < max(thr, 2)]
        if len(hi) >= 2 and len(lo) >= 2:
            lvl = "warn" if hi["net"].mean() < lo["net"].mean() else "info"
            out.append((lvl, "Frequência de operação", f"Em dias mais ativos (≥ {int(max(thr, 2))} fechamentos) o PnL médio é {sg(hi['net'].mean())}, contra {sg(lo['net'].mean())} nos dias mais calmos."
                        + (" Mais operações não estão gerando mais resultado." if lvl == "warn" else "")))
    # 10 drawdown
    out.append(("warn" if cur_dd < 0 and abs(cur_dd) > 0.5 * max(abs(max_dd), 1e-9) else "info", "Drawdown",
                f"Drawdown máximo de {br(max_dd)} USDT" + (f" ({pct(max_dd_pct)} do capital)" if capital > 0 else "") + f". Agora você está {br(abs(cur_dd))} USDT abaixo do pico. {pct(pos_days)} dos dias terminaram positivos."))
    if days_uw > 0.6 and len(daily) >= 10:
        out.append(("warn", "Tempo debaixo d'água", f"Em {pct(days_uw)} dos dias você terminou abaixo do pico anterior. Recovery factor: {br(recovery)} (PnL ÷ drawdown máximo) — quanto menor, mais tempo (e risco) para cada USDT ganho."))
    # 11 funding
    if abs(fund_total) > 0:
        out.append(("info", "Funding", f"O funding somou {sg(fund_total, 4)} USDT ({'a seu favor' if fund_total > 0 else 'contra você'}) — {'irrelevante' if abs(fund_total) < 0.05 * max(abs(net_total), 1e-9) else 'relevante'} frente ao PnL líquido."))
    # risco diário (novo)
    if not np.isnan(var95):
        out.append(("info", "Risco diário (VaR)", f"Em 95% dos dias a perda ficou abaixo de {br(abs(var95))} USDT; nos 5% piores dias a média foi {sg(cvar95)} USDT. Compare com o seu limite de perda diária."))
    # SQN (novo)
    if not np.isnan(st_ev.get("sqn", np.nan)) and n >= 30:
        sq = st_ev["sqn"]
        out.append(("good" if sq >= 2 else "warn" if sq < 1 else "info", "Qualidade do sistema (SQN)", f"SQN de {br(sq)}: " + ("bom — a expectativa é alta frente à dispersão." if sq >= 2 else "abaixo de 1 — o resultado médio ainda se confunde com o ruído." if sq < 1 else "intermediário.")))
    # símbolos (novo)
    ps = inc.groupby("symbol")["income"].sum()
    if len(ps) >= 3 and ps.min() < 0 < ps.max():
        out.append(("info", "Símbolos", f"Melhor: {ps.idxmax()} ({sg(ps.max())}). Pior: {ps.idxmin()} ({sg(ps.min())}). {pct(ps.max() / ps[ps > 0].sum())} do lucro vem do melhor símbolo."))
    # 12 fills: tempo de retenção, sizing, saídas
    r = rt[~rt["orphan"]] if not rt.empty else rt
    if not r.empty and len(r) >= 8:
        win, loss = r[r["net"] > 0], r[r["net"] <= 0]
        if len(win) >= 3 and len(loss) >= 3:
            hw, hl = win["hold_min"].median(), loss["hold_min"].median()
            if hl > hw * 1.5:
                out.append(("warn", "Perdas duram mais que ganhos", f"Tempo mediano nas perdas: {fmt_min(hl)} vs {fmt_min(hw)} nos ganhos. Isso é o padrão clássico de 'segurar a perdedora / realizar cedo a ganhadora' — reveja seus stops."))
            elif hw > hl * 1.5:
                out.append(("good", "Ganhos duram mais que perdas", f"Tempo mediano nos ganhos: {fmt_min(hw)} vs {fmt_min(hl)} nas perdas — você corta perdas mais rápido do que deixa lucros correrem."))
        s = open_after_previous(r)
        if len(s) >= 10:
            a_loss, a_win = s[s["prev_net"] < 0]["notional"], s[s["prev_net"] >= 0]["notional"]
            if len(a_loss) >= 4 and len(a_win) >= 4 and a_win.mean() > 0:
                rat = a_loss.mean() / a_win.mean()
                if rat > 1.15:
                    out.append(("warn", "Aumenta a mão depois de perder", f"Após uma perda, a posição média seguinte é {br(a_loss.mean(), 0)} USDT vs {br(a_win.mean(), 0)} após um ganho (+{br((rat - 1) * 100, 0)}%). Cuidado com o 'revenge trading' / martingale."))
                elif rat < 0.85:
                    out.append(("good", "Reduz a mão depois de perder", f"Após perdas a posição média cai para {br(a_loss.mean(), 0)} USDT (vs {br(a_win.mean(), 0)} após ganhos) — comportamento disciplinado."))
        bins, labs = [0, 15, 60, 240, 1440, np.inf], ["<15min", "15–60min", "1–4h", "4–24h", ">24h"]
        gb = r.assign(bucket=pd.cut(r["hold_min"], bins=bins, labels=labs)).groupby("bucket", observed=True)["net"].agg(["count", "sum", "mean"])
        gb = gb[gb["count"] >= 3]
        if len(gb) >= 2 and gb["sum"].min() < 0:
            out.append(("info", "Duração ideal", f"Melhor faixa de duração: {gb['sum'].idxmax()} ({sg(gb['sum'].max())}). Pior: {gb['sum'].idxmin()} ({sg(gb['sum'].min())})."))
        if r["exit_type"].nunique() > 1 and (r["exit_type"] != "N/D").any():
            ge = r.groupby("exit_type")["net"].agg(["count", "sum", "mean"])
            txt = " · ".join(f"{k}: {int(v['count'])} saídas, média {sg(v['mean'], 3)}" for k, v in ge.iterrows())
            out.append(("info", "Tipo de saída", txt + ". (TAKE_PROFIT = alvo executado automaticamente; MARKET = saída manual/a mercado.)"))
        bps = (r["fees"].sum() / r["notional"].sum()) * 1e4 if r["notional"].sum() else np.nan
        if not np.isnan(bps):
            out.append(("info", "Custo por round-trip", f"Taxas de entrada + saída ≈ {br(bps, 1)} bps do nocional (~{br(bps / 100, 3)}%). O preço precisa andar pelo menos isso a seu favor só para empatar."))
    # maker × taker (novo)
    if not fills.empty and fills["role"].notna().any():
        rl = fills.assign(role=fills["role"].astype(str).str.lower()).groupby("role").agg(f=("fee", lambda s: s.abs().sum()), q=("quoteQty", "sum"))
        if {"maker", "taker"} <= set(rl.index) and rl.loc["taker", "q"] > 0 and rl.loc["maker", "q"] > 0:
            bt, bm = rl.loc["taker", "f"] / rl.loc["taker", "q"] * 1e4, rl.loc["maker", "f"] / rl.loc["maker", "q"] * 1e4
            sh = rl.loc["taker", "q"] / rl["q"].sum()
            out.append(("warn" if sh > 0.7 else "info", "Maker × Taker", f"{pct(sh)} do volume foi executado como taker ({br(bt, 2)} bps) contra {br(bm, 2)} bps no maker. Trocar parte das entradas a mercado por ordens limit reduz o custo por operação."))
    # 13 Kelly
    if not np.isnan(st_ev["kelly"]) and n >= 20:
        out.append(("info", "Kelly", f"Fração de Kelly estimada: {pct(st_ev['kelly'])} do capital por operação" + (" (negativa = sem edge estatístico na amostra). " if st_ev["kelly"] < 0 else ". Na prática usa-se 1/4 a 1/2 Kelly. ") + "Com amostra pequena, trate como teto teórico."))
    return out


with tabs[1]:
    st.markdown("### Leitura automática dos seus dados")
    st.caption("Regras estatísticas aplicadas ao filtro atual. Não são recomendações de investimento — servem para levantar hipóteses sobre o seu comportamento.")
    ins_list = build_insights()
    left, right = st.columns([1, 1.7])
    with left:
        with st.container(border=True):
            sc = quality_scores()
            overall = float(np.mean(list(sc.values())))
            sc = {{"Rentabilidade": "Rentab.", "Consistência": "Consist.", "Controle de risco": "Risco", "Eficiência de custos": "Custos", "Amostra": "Amostra"}[k]: v for k, v in sc.items()}
            cats = list(sc) + [list(sc)[0]]
            vals = list(sc.values()) + [list(sc.values())[0]]
            fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself", line=dict(color=ACCENT, width=2), fillcolor="rgba(139,92,246,.30)",
                                            hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>"))
            fig.update_layout(title=f"Radar de qualidade · índice {overall:.0f}/100",
                              polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,.2)", tickfont=dict(size=9)),
                                         angularaxis=dict(gridcolor="rgba(148,163,184,.2)")))
            show(fig, 380, legend=False, margin=dict(l=50, r=50, t=56, b=30))
            st.caption("Índice heurístico (0–100): rentabilidade (profit factor), consistência (% de dias positivos), controle de risco (recovery factor), "
                       "eficiência de custos (taxas ÷ lucro bruto) e tamanho da amostra. Serve para comparar períodos — não é previsão.")
    with right:
        lvl_filter = st.segmented_control("Mostrar", ["Todos", "Alertas", "Atenção", "Positivos", "Info"], default="Todos", key="ins_filter") or "Todos"
        wanted = {"Todos": {"good", "warn", "bad", "info"}, "Alertas": {"bad"}, "Atenção": {"warn", "bad"}, "Positivos": {"good"}, "Info": {"info"}}[lvl_filter]
        shown = [(l, t, x) for l, t, x in ins_list if l in wanted]
        if not shown:
            st.info("Nenhum insight nessa categoria para o filtro atual.")
        for lvl, title, text in shown:
            insight(lvl, title, text)
    with st.expander("🔎 Qualidade e cobertura dos dados"):
        st.write(f"**Income:** {len(inc_raw)} lançamentos de {inc_all['dt'].min():%d/%m/%Y} a {inc_all['dt'].max():%d/%m/%Y} ({tz}).")
        if not fills_all.empty:
            st.write(f"**Fills de futuros:** {len(fills_all)} execuções de {fills_all['dt'].min():%d/%m/%Y} a {fills_all['dt'].max():%d/%m/%Y}.")
            if fills_all["dt"].min() > inc_all["dt"].min() + pd.Timedelta(days=1):
                st.warning("Os fills cobrem um período **menor** que o income. Análises de posição (duração, tamanho, preço de entrada) só existem para o período dos fills; o PnL total usa o income, que é mais completo.")
            if not rt_all.empty and rt_all["orphan"].any():
                st.warning(f"{int(rt_all['orphan'].sum())} saída(s) sem abertura no período exportado (posição aberta antes do primeiro fill). Ficam de fora das métricas de posição.")
        if spot_dups:
            st.warning(f"O export de **spot** trazia {spot_dups} linhas repetidas (o mesmo trade replicado várias vezes). Deduplicado automaticamente.")
        st.caption("Fechamentos e win rate usam o PnL realizado do income menos a taxa de fechamento. A taxa de abertura entra no PnL líquido total, mas não é atribuída ao fechamento individual (veja a aba Posições para o custo completo por operação).")


# ═════════════════════════ 3. PnL ═════════════════════════
with tabs[2]:
    gran = st.segmented_control("Agrupar por", ["Dia", "Semana", "Mês"], default="Dia", key="gran") or "Dia"
    rule = {"Dia": "D", "Semana": "W-SUN", "Mês": "MS"}[gran]
    g = inc.set_index("dt")
    agg = pd.DataFrame({
        "Realizado": g[g["incomeType"] == "REALIZED_PNL"]["income"].resample(rule).sum(),
        "Taxas": g[g["incomeType"] == "TRADING_FEE"]["income"].resample(rule).sum(),
        "Funding": g[g["incomeType"] == "FUNDING_FEE"]["income"].resample(rule).sum(),
    }).fillna(0.0)
    agg["Líquido"] = agg.sum(axis=1)
    if not events.empty:
        e2 = events.set_index("dt")["net"]
        agg["Fechamentos"] = e2.resample(rule).count().reindex(agg.index).fillna(0).astype(int)
        agg["Win rate"] = (e2 > 0).astype(float).resample(rule).mean().reindex(agg.index)

    with st.container(border=True):
        fig = go.Figure()
        fig.add_bar(x=agg.index, y=agg["Realizado"], name="Realizado", marker_color=GREEN)
        fig.add_bar(x=agg.index, y=agg["Taxas"], name="Taxas", marker_color=RED)
        fig.add_bar(x=agg.index, y=agg["Funding"], name="Funding", marker_color=AMBER)
        fig.add_trace(go.Scatter(x=agg.index, y=agg["Líquido"], name="Líquido", mode="lines+markers", line=dict(color="#fff", width=2)))
        fig.update_layout(barmode="relative", title=f"Composição do PnL por {gran.lower()}")
        show(fig, 400)

    a, b = st.columns([1.5, 1])
    with a:
        section("Calendário de PnL", "dia a dia")
        with st.container(border=True):
            full = daily["net"].reindex(pd.date_range(daily.index.min(), daily.index.max()))
            start = full.index.min() - pd.Timedelta(days=full.index.min().weekday())
            weeks = pd.date_range(start, full.index.max(), freq="7D")
            z = np.full((len(weeks), 7), np.nan)
            txt = [[""] * 7 for _ in weeks]
            for d_, v in full.items():
                wi, di = (d_ - start).days // 7, d_.weekday()
                z[wi, di] = v
                txt[wi][di] = f"{d_:%d/%m}<br>{'' if np.isnan(v) else sg(v)}"
            lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
            fig = go.Figure(go.Heatmap(z=z, x=DOW, y=[f"{w:%d/%m}" for w in weeks], text=txt, texttemplate="%{text}", colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim,
                                       hoverongaps=False, showscale=False, xgap=3, ygap=3))
            fig.update_yaxes(autorange="reversed", title="Semana (início)")
            show(fig, max(300, 70 * len(weeks) + 80), legend=False)
    with b:
        section("Por símbolo")
        ps = inc.pivot_table(index="symbol", columns="incomeType", values="income", aggfunc="sum").fillna(0.0)
        for cc in ["REALIZED_PNL", "TRADING_FEE", "FUNDING_FEE"]:
            if cc not in ps:
                ps[cc] = 0.0
        ps["Líquido"] = ps.sum(axis=1)
        ps = ps.rename(columns={"REALIZED_PNL": "Realizado", "TRADING_FEE": "Taxas", "FUNDING_FEE": "Funding"})
        if not events.empty:
            ps["Fechamentos"] = events.groupby("symbol").size()
            ps["Win rate"] = events.groupby("symbol")["net"].apply(lambda s: (s > 0).mean())
        ps.index.name = "Par"
        ps = ps[[c_ for c_ in ["Realizado", "Taxas", "Funding", "Líquido", "Fechamentos", "Win rate"] if c_ in ps.columns]]
        st.dataframe(ps.sort_values("Líquido", ascending=False).style.format(FMT, na_rep="–"), width="stretch")
        with st.container(border=True):
            fig = go.Figure(go.Bar(x=ps.index, y=ps["Líquido"], marker_color=colors(ps["Líquido"])))
            fig.update_layout(title="PnL líquido por símbolo")
            show(fig, 240, legend=False)

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            sym_cum = inc.pivot_table(index="day", columns="symbol", values="income", aggfunc="sum").fillna(0.0).cumsum()
            fig = go.Figure()
            for c_ in sym_cum.columns:
                fig.add_trace(go.Scatter(x=sym_cum.index, y=sym_cum[c_], name=c_, mode="lines", line=dict(width=2, shape="hv")))
            fig.update_layout(title="PnL líquido acumulado por símbolo")
            show(fig, 360)
    with b:
        with st.container(border=True):
            mo = daily["net"].groupby([daily.index.year, daily.index.month]).sum().unstack().reindex(columns=range(1, 13))
            if len(daily.index.to_period("M").unique()) >= 2:
                lim = np.nanmax(np.abs(mo.values)) if np.isfinite(mo.values).any() else 1
                fig = go.Figure(go.Heatmap(z=mo.values, x=["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"], y=[str(i) for i in mo.index],
                                           colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim, xgap=3, ygap=3, hoverongaps=False,
                                           text=[[("" if np.isnan(v) else sg(v, 1)) for v in row] for row in mo.values], texttemplate="%{text}", showscale=False))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(title="PnL líquido por mês")
                show(fig, 360, legend=False)
            else:
                st.info("O mapa mensal aparece quando o período cobre mais de um mês.")

    section(f"Tabela por {gran.lower()}")
    tbl = agg.copy()
    tbl.index = tbl.index.strftime("%d/%m/%Y" if gran != "Mês" else "%m/%Y")
    st.dataframe(tbl.iloc[::-1].style.format(FMT, na_rep="–"), width="stretch")


# ═════════════════════════ 4. TRADES ═════════════════════════
with tabs[3]:
    if events.empty:
        st.info("Sem fechamentos no filtro.")
    else:
        kpi_row([
            ("Fechamentos", str(st_ev["n"])),
            ("Ganhos / Perdas", f"{st_ev['wins']} / {st_ev['losses']}"),
            ("Ganho médio", sg(st_ev["avg_win"]), "", "good"),
            ("Perda média", sg(st_ev["avg_loss"]), "", "bad"),
            ("Melhor fechamento", sg(st_ev["best"]), f"pior: {sg(st_ev['worst'])}", "good"),
            ("Kelly (teórico)", pct(st_ev["kelly"]), "", "", "f* = W − (1−W)/R. Negativo indica ausência de edge na amostra."),
        ])
        a, b = st.columns(2)
        with a:
            with st.container(border=True):
                fig = go.Figure(go.Histogram(x=events["net"], nbinsx=30, marker_color=ACCENT, name="Fechamentos"))
                fig.add_vline(x=0, line_color=GREY, line_dash="dot")
                fig.add_vline(x=events["net"].mean(), line_color=AMBER, annotation_text=f"média {sg(events['net'].mean(), 3)}")
                fig.update_layout(title="Distribuição do PnL líquido por fechamento", xaxis_title="USDT", yaxis_title="nº")
                show(fig, 380, legend=False)
        with b:
            with st.container(border=True):
                seq = events.reset_index(drop=True)
                fig = go.Figure(go.Bar(x=seq.index + 1, y=seq["net"], marker_color=colors(seq["net"]), name="PnL",
                                       customdata=seq[["dt", "side"]], hovertemplate="#%{x} · %{customdata[1]}<br>%{customdata[0]|%d/%m %H:%M}<br>%{y:+.3f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=seq.index + 1, y=seq["net"].cumsum(), yaxis="y2", name="Acumulado", line=dict(color=AMBER, width=2)))
                fig.update_layout(title="Sequência de fechamentos", xaxis_title="# do fechamento", yaxis2=dict(overlaying="y", side="right", showgrid=False))
                show(fig, 380)

        ls = stats_by(events, "side")
        sy = stats_by(events, "symbol")
        fmt_st = {"Win rate": "{:.0%}", "Ganho médio": "{:+.3f}", "Perda média": "{:+.3f}", "Payoff": "{:.2f}", "Profit factor": "{:.2f}", "Expectativa": "{:+.3f}", "Total": "{:+.2f}"}
        a, b = st.columns(2)
        with a:
            section("Rolling", "janela de fechamentos")
            win = st.slider("Janela", 5, max(6, min(50, len(events))), min(10, max(5, len(events) // 3)))
            with st.container(border=True):
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=events.index + 1, y=(events["net"] > 0).astype(float).rolling(win).mean(), name="Win rate", line=dict(color=BLUE, width=2)), secondary_y=False)
                fig.add_trace(go.Scatter(x=events.index + 1, y=events["net"].rolling(win).mean(), name="Expectativa", line=dict(color=GREEN, width=2)), secondary_y=True)
                fig.update_yaxes(tickformat=".0%", secondary_y=False)
                fig.update_yaxes(showgrid=False, secondary_y=True)
                show(fig, 330)
        with b:
            section("Win rate × payoff", "por símbolo")
            with st.container(border=True):
                if len(sy) >= 2:
                    fig = go.Figure(go.Scatter(x=sy["Win rate"], y=sy["Payoff"], mode="markers+text", text=sy.index, textposition="top center",
                                               marker=dict(size=np.clip(np.sqrt(sy["Fech."]) * 3 + 12, 14, 42), color=sy["Total"], colorscale="RdYlGn", cmid=0, showscale=True,
                                                           colorbar=dict(title="PnL"), line=dict(width=1, color="#fff")),
                                               hovertemplate="%{text}<br>win rate %{x:.0%}<br>payoff %{y:.2f}<extra></extra>"))
                    xs = np.linspace(0.05, 0.95, 60)
                    fig.add_trace(go.Scatter(x=xs, y=(1 - xs) / xs, mode="lines", line=dict(color=GREY, dash="dot"), name="Linha de equilíbrio", hoverinfo="skip"))
                    fig.update_xaxes(tickformat=".0%", title="win rate")
                    fig.update_yaxes(title="payoff", range=[0, max(3, float(sy["Payoff"].max() * 1.2))])
                    fig.update_layout(title="Acima da linha pontilhada = lucrativo antes das taxas de abertura")
                    show(fig, 330, legend=False)
                else:
                    st.info("O gráfico aparece quando há mais de um símbolo no filtro.")

        section("Long × Short")
        st.dataframe(ls.style.format(fmt_st), width="stretch")
        section("Por símbolo")
        st.dataframe(sy.style.format(fmt_st), width="stretch")

        a, b = st.columns(2)
        cols_ev = {"dt": "Quando", "symbol": "Par", "side": "Lado", "gross": "Bruto", "close_fee": "Taxa", "net": "Líquido"}
        fmt_ev = {"Bruto": "{:+.3f}", "Taxa": "{:+.3f}", "Líquido": "{:+.3f}"}
        with a:
            section("🏆 Top 5 melhores")
            st.dataframe(events.nlargest(5, "net")[list(cols_ev)].rename(columns=cols_ev).style.format(fmt_ev), width="stretch", hide_index=True)
        with b:
            section("💥 Top 5 piores")
            st.dataframe(events.nsmallest(5, "net")[list(cols_ev)].rename(columns=cols_ev).style.format(fmt_ev), width="stretch", hide_index=True)

        section("Todos os fechamentos")
        show_only = st.segmented_control("Filtrar tabela", ["Todos", "Só ganhos", "Só perdas"], default="Todos", key="tv_filter") or "Todos"
        tv = events[list(cols_ev)].rename(columns=cols_ev).rename(columns={"Taxa": "Taxa fech."})
        if show_only == "Só ganhos":
            tv = tv[tv["Líquido"] > 0]
        elif show_only == "Só perdas":
            tv = tv[tv["Líquido"] < 0]
        st.dataframe(tv.iloc[::-1].style.format({"Bruto": "{:+.3f}", "Taxa fech.": "{:+.3f}", "Líquido": "{:+.3f}"}).map(lambda v: f"color:{GREEN if v > 0 else RED}" if isinstance(v, float) else "", subset=["Líquido"]),
                     width="stretch", hide_index=True, height=380)


# ═════════════════════════ 5. TIMING ═════════════════════════
with tabs[4]:
    if events.empty:
        st.info("Sem fechamentos no filtro.")
    else:
        st.caption(f"Horários no fuso {tz}; baseados no momento do fechamento.")
        a, b = st.columns(2)
        with a:
            with st.container(border=True):
                h = events.groupby("hour")["net"].agg(["sum", "count", lambda s: (s > 0).mean()]).reindex(range(24))
                h.columns = ["pnl", "n", "wr"]
                fig = go.Figure(go.Bar(x=h.index, y=h["pnl"].fillna(0), marker_color=colors(h["pnl"].fillna(0)), customdata=np.c_[h["n"].fillna(0), h["wr"].fillna(0)],
                                       hovertemplate="%{x}h<br>PnL %{y:+.2f}<br>%{customdata[0]:.0f} fech. · win rate %{customdata[1]:.0%}<extra></extra>"))
                fig.update_layout(title="PnL líquido por hora do dia", xaxis=dict(dtick=1, title="hora"))
                show(fig, 360, legend=False)
        with b:
            with st.container(border=True):
                w = events.groupby("wd")["net"].agg(["sum", "count", lambda s: (s > 0).mean()]).reindex(range(7))
                w.columns = ["pnl", "n", "wr"]
                fig = go.Figure(go.Bar(x=DOW, y=w["pnl"].fillna(0), marker_color=colors(w["pnl"].fillna(0)), customdata=np.c_[w["n"].fillna(0), w["wr"].fillna(0)],
                                       hovertemplate="%{x}<br>PnL %{y:+.2f}<br>%{customdata[0]:.0f} fech. · win rate %{customdata[1]:.0%}<extra></extra>"))
                fig.update_layout(title="PnL líquido por dia da semana")
                show(fig, 360, legend=False)

        a, b = st.columns(2)
        with a:
            with st.container(border=True):
                hm = events.pivot_table(index="wd", columns="hour", values="net", aggfunc="sum").reindex(index=range(7), columns=range(24))
                lim = np.nanmax(np.abs(hm.values)) if np.isfinite(hm.values).any() else 1
                fig = go.Figure(go.Heatmap(z=hm.values, x=list(range(24)), y=DOW, colorscale="RdYlGn", zmid=0, zmin=-lim, zmax=lim, hoverongaps=False, xgap=2, ygap=2,
                                           hovertemplate="%{y} %{x}h<br>%{z:+.2f}<extra></extra>", colorbar=dict(title="USDT")))
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(title="Mapa de calor: dia da semana × hora", xaxis=dict(dtick=1))
                show(fig, 360, legend=False)
        with b:
            with st.container(border=True):
                d = daily.copy()
                d["faixa"] = pd.cut(d["trades"], bins=[-1, 0, 2, 4, 8, 1000], labels=["0", "1–2", "3–4", "5–8", "9+"])
                gb = d.groupby("faixa", observed=True)["net"].agg(["mean", "count"])
                fig = go.Figure(go.Bar(x=gb.index.astype(str), y=gb["mean"], marker_color=colors(gb["mean"]), customdata=gb["count"],
                                       hovertemplate="%{x} fech./dia<br>PnL médio %{y:+.2f}<br>%{customdata} dias<extra></extra>"))
                fig.update_layout(title="PnL médio do dia × nº de fechamentos no dia", xaxis_title="fechamentos no dia")
                show(fig, 360, legend=False)

        a, b = st.columns(2)
        with a:
            with st.container(border=True):
                ss = events.assign(sessao=events["hour_utc"].map(session_of)).groupby("sessao")["net"].agg(["sum", "count", lambda s: (s > 0).mean()])
                ss.columns = ["pnl", "n", "wr"]
                ss = ss.reindex(["Ásia", "Europa", "EUA", "Late night"]).dropna()
                fig = go.Figure(go.Bar(x=ss.index, y=ss["pnl"], marker_color=colors(ss["pnl"]), customdata=np.c_[ss["n"], ss["wr"]],
                                       hovertemplate="%{x}<br>PnL %{y:+.2f}<br>%{customdata[0]:.0f} fech. · win rate %{customdata[1]:.0%}<extra></extra>"))
                fig.update_layout(title="PnL por sessão de mercado (Ásia 00–08 · Europa 08–13 · EUA 13–21 UTC)")
                show(fig, 340, legend=False)
        with b:
            with st.container(border=True):
                gaps = events["timestamp"].diff().dropna() / 60000
                if len(gaps):
                    fig = go.Figure(go.Histogram(x=gaps.clip(upper=gaps.quantile(0.95)), nbinsx=30, marker_color=BLUE))
                    fig.update_layout(title=f"Minutos entre fechamentos (mediana {fmt_min(gaps.median())}; cortado no p95)", xaxis_title="min")
                    show(fig, 340, legend=False)


# ═════════════════════════ 6. POSIÇÕES (FILLS) ═════════════════════════
with tabs[5]:
    if fills_raw is None or len(fills_raw) == 0 or rt.empty:
        st.info("Envie também o arquivo **bingx_operations** para ver a reconstrução das posições (preço de entrada/saída, duração, tamanho).")
    else:
        st.caption("Reconstruído por FIFO a partir dos fills de futuros (só cobre o período dos fills). PnL bruto = (saída − entrada) × qtd; taxas = abertura + fechamento.")
        r = rt[~rt["orphan"]].copy()
        if r.empty:
            st.info("Nenhuma posição completa (abertura + fechamento) no filtro.")
        else:
            s_r = trade_stats(r["net"])
            kpi_row([
                ("Saídas", str(s_r["n"])),
                ("PnL líquido", sg(r["net"].sum()), "já com a taxa de abertura", tone_of(r["net"].sum())),
                ("Win rate", pct(s_r["win_rate"])),
                ("Duração mediana", fmt_min(r["hold_min"].median())),
                ("Nocional médio", br(r["notional"].mean(), 0)),
                ("Retorno médio", pct(r["ret_pct"].mean() / 100, 3), "sobre o nocional"),
            ], gap=False)

            rng0, rng1 = fills_all["timestamp"].min(), fills_all["timestamp"].max()
            inc_c = inc_all[(inc_all["incomeType"] == "REALIZED_PNL") & (inc_all["timestamp"].between(rng0, rng1)) & inc_all["symbol"].isin(syms)]["income"].sum()
            rt_c = rt_all[(~rt_all["orphan"]) & rt_all["symbol"].isin(syms)]["gross"].sum()
            st.caption(f"Conciliação (período dos fills, todos os filtros de data ignorados): realizado reportado pela BingX {sg(inc_c)} vs. calculado pelos fills {sg(rt_c)} — diferença {sg(rt_c - inc_c, 3)}.")

            section("Trades no gráfico de preço", "cada linha liga a entrada à saída")
            sym = st.selectbox("Símbolo", sorted(r["symbol"].unique()))
            rs = r[r["symbol"] == sym]
            with st.container(border=True):
                fig = go.Figure()
                for flag, cl, nm in [(True, GREEN, "Lucro"), (False, RED, "Prejuízo")]:
                    q = rs[(rs["net"] > 0) == flag]
                    if q.empty:
                        continue
                    xs_, ys_ = [], []
                    for x in q.itertuples():
                        xs_ += [x.open_dt, x.dt, None]
                        ys_ += [x.entry, x.exit, None]
                    fig.add_trace(go.Scatter(x=xs_, y=ys_, mode="lines", line=dict(color=cl, width=2), name=nm, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=rs["open_dt"], y=rs["entry"], mode="markers", name="Entrada", marker=dict(symbol="circle", size=8, color=BLUE, line=dict(width=1, color="#fff")),
                                         customdata=np.c_[rs["side"], rs["qty"], rs["net"]], hovertemplate="ENTRADA %{customdata[0]}<br>%{y:,.4f} · qtd %{customdata[1]:g}<extra></extra>"))
                fig.add_trace(go.Scatter(x=rs["dt"], y=rs["exit"], mode="markers", name="Saída", marker=dict(symbol="x", size=9, color=AMBER),
                                         customdata=np.c_[rs["side"], rs["qty"], rs["net"]], hovertemplate="SAÍDA %{customdata[0]}<br>%{y:,.4f} · qtd %{customdata[1]:g}<br>líquido %{customdata[2]:+.3f}<extra></extra>"))
                fig.update_layout(title=f"{sym} — verde = lucro, vermelho = prejuízo")
                show(fig, 480)

            a, b = st.columns(2)
            with a:
                with st.container(border=True):
                    fig = px.scatter(r, x="hold_min", y="net", color=np.where(r["net"] > 0, "Ganho", "Perda"), color_discrete_map={"Ganho": GREEN, "Perda": RED},
                                     size=r["notional"].clip(lower=1), hover_data={"side": True, "symbol": True}, log_x=True,
                                     labels={"hold_min": "duração (min, escala log)", "net": "PnL líquido"})
                    fig.add_hline(y=0, line_color=GREY, line_dash="dot")
                    fig.update_layout(title="Duração × resultado (bolha = tamanho da posição)")
                    show(fig, 400)
            with b:
                with st.container(border=True):
                    bins, labs = [0, 15, 60, 240, 1440, np.inf], ["<15min", "15–60min", "1–4h", "4–24h", ">24h"]
                    r["faixa"] = pd.cut(r["hold_min"], bins=bins, labels=labs)
                    gb = r.groupby("faixa", observed=True)["net"].agg(["sum", "count", "mean"]).reindex(labs).dropna()
                    fig = go.Figure(go.Bar(x=gb.index.astype(str), y=gb["sum"], marker_color=colors(gb["sum"]), customdata=np.c_[gb["count"], gb["mean"]],
                                           hovertemplate="%{x}<br>total %{y:+.2f}<br>%{customdata[0]:.0f} saídas · média %{customdata[1]:+.3f}<extra></extra>"))
                    fig.update_layout(title="PnL líquido por faixa de duração")
                    show(fig, 400, legend=False)

            a, b = st.columns(2)
            with a:
                with st.container(border=True):
                    fig = px.scatter(r, x="notional", y="net", color=np.where(r["net"] > 0, "Ganho", "Perda"), color_discrete_map={"Ganho": GREEN, "Perda": RED},
                                     labels={"notional": "nocional da posição (USDT)", "net": "PnL líquido"})
                    fig.add_hline(y=0, line_color=GREY, line_dash="dot")
                    fig.update_layout(title="Tamanho × resultado")
                    show(fig, 380)
            with b:
                with st.container(border=True):
                    if (r["exit_type"] != "N/D").any():
                        ge = r.groupby("exit_type")["net"].agg(["count", "sum", "mean"]).reset_index()
                        fig = go.Figure(go.Bar(x=ge["exit_type"], y=ge["sum"], marker_color=colors(ge["sum"]), customdata=np.c_[ge["count"], ge["mean"]],
                                               hovertemplate="%{x}<br>total %{y:+.2f}<br>%{customdata[0]:.0f} saídas · média %{customdata[1]:+.3f}<extra></extra>"))
                        fig.update_layout(title="Resultado por tipo de saída (MARKET vs TAKE_PROFIT)")
                        show(fig, 380, legend=False)
                    else:
                        st.info("O tipo de ordem (MARKET / TAKE_PROFIT) só vem no export **JSON** de operations.")

            a, b = st.columns(2)
            with a:
                with st.container(border=True):
                    s2 = open_after_previous(r)
                    if len(s2) >= 6:
                        s2 = s2.assign(ant=np.where(s2["prev_net"] < 0, "após perda", "após ganho"))
                        gm = s2.groupby("ant")["notional"].agg(["mean", "count"]).reindex(["após ganho", "após perda"]).dropna()
                        fig = go.Figure(go.Bar(x=gm.index, y=gm["mean"], marker_color=[GREEN if i == "após ganho" else RED for i in gm.index], customdata=gm["count"],
                                               text=[br(v, 0) for v in gm["mean"]], textposition="outside", hovertemplate="%{x}<br>nocional médio %{y:,.0f}<br>%{customdata} posições<extra></extra>"))
                        fig.update_layout(title="Tamanho da posição seguinte × resultado da anterior (revenge trading)")
                        show(fig, 340, legend=False)
                    else:
                        st.info("Poucas posições para comparar o tamanho depois de ganhos e perdas.")
            with b:
                with st.container(border=True):
                    fig = go.Figure()
                    for flag, cl, nm in [(True, GREEN, "Ganhos"), (False, RED, "Perdas")]:
                        q = r[(r["net"] > 0) == flag]
                        if len(q):
                            fig.add_trace(go.Box(y=q["hold_min"], name=nm, marker_color=cl, boxmean=True))
                    fig.update_yaxes(type="log", title="duração (min, escala log)")
                    fig.update_layout(title="Duração das posições: ganhos × perdas")
                    show(fig, 340, legend=False)

            section("Exposição ao longo do tempo")
            fs = fills[fills["symbol"] == sym].copy()
            if not fs.empty:
                fs["is_open"] = ((fs["side"] == "BUY") & (fs["positionSide"] == "LONG")) | ((fs["side"] == "SELL") & (fs["positionSide"] == "SHORT"))
                fs["delta"] = np.where(fs["is_open"], fs["quantity"], -fs["quantity"])
                with st.container(border=True):
                    fig = go.Figure()
                    for ps_, cl, sgn in [("LONG", GREEN, 1), ("SHORT", RED, -1)]:
                        q = fs[fs["positionSide"] == ps_]
                        fig.add_trace(go.Scatter(x=q["dt"], y=sgn * q["delta"].cumsum(), name=ps_, line=dict(color=cl, shape="hv", width=2), fill="tozeroy"))
                    fig.update_layout(title=f"{sym}: quantidade aberta (long acima, short abaixo de zero) — sobreposição = hedge")
                    show(fig, 340)

            section("Tabela de saídas")
            tv = rt[["dt", "open_dt", "symbol", "side", "qty", "entry", "exit", "gross", "fees", "net", "ret_pct", "hold_min", "exit_type", "orphan"]].copy()
            tv["hold_min"] = tv["hold_min"].map(fmt_min)
            tv = tv.rename(columns={"dt": "Saída", "open_dt": "Entrada (média)", "symbol": "Par", "side": "Lado", "qty": "Qtd", "entry": "Preço entrada", "exit": "Preço saída",
                                    "gross": "Bruto", "fees": "Taxas", "net": "Líquido", "ret_pct": "Ret. %", "hold_min": "Duração", "exit_type": "Tipo saída", "orphan": "Sem abertura"})
            st.dataframe(tv.iloc[::-1].style.format({"Qtd": "{:g}", "Preço entrada": "{:,.4f}", "Preço saída": "{:,.4f}", "Bruto": "{:+.3f}", "Taxas": "{:.3f}", "Líquido": "{:+.3f}", "Ret. %": "{:+.3f}"}, na_rep="–"),
                         width="stretch", hide_index=True, height=380)


# ═════════════════════════ 7. CUSTOS ═════════════════════════
with tabs[6]:
    tf = inc[inc["incomeType"] == "TRADING_FEE"].copy()
    tf["Tipo"] = np.where(tf["info"].str.contains("opening", case=False), "Abertura", "Fechamento")
    kpi_row([
        ("Taxas totais", sg(fees_total), "", "bad" if fees_total < 0 else ""),
        ("Taxas de abertura", sg(tf.loc[tf["Tipo"] == "Abertura", "income"].sum())),
        ("Taxas de fechamento", sg(tf.loc[tf["Tipo"] == "Fechamento", "income"].sum())),
        ("Funding líquido", sg(fund_total, 4), "", tone_of(fund_total)),
        ("Taxas / lucro bruto", pct(-fees_total / gross_total) if gross_total > 0 else "–"),
    ])
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            pv = tf.pivot_table(index="day", columns="Tipo", values="income", aggfunc="sum").fillna(0.0)
            fig = go.Figure()
            for col, cl in [("Abertura", AMBER), ("Fechamento", RED)]:
                if col in pv:
                    fig.add_bar(x=pv.index, y=pv[col], name=col, marker_color=cl)
            fig.update_layout(barmode="stack", title="Taxas por dia")
            show(fig, 360)
    with b:
        with st.container(border=True):
            fd = inc[inc["incomeType"] == "FUNDING_FEE"].groupby("day")["income"].sum()
            fig = go.Figure(go.Bar(x=fd.index, y=fd.values, marker_color=colors(fd.values)))
            fig.update_layout(title="Funding por dia")
            show(fig, 360, legend=False)

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            d = daily.reset_index()
            fig = go.Figure()
            fig.add_bar(x=d["day"], y=d["gross"], name="Realizado bruto", marker_color=GREEN)
            fig.add_bar(x=d["day"], y=-d["fees"].abs(), name="Taxas (negativas)", marker_color=RED)
            fig.add_trace(go.Scatter(x=d["day"], y=d["gross"] + d["fees"], name="Após taxas", mode="lines+markers", line=dict(color="#fff", width=2)))
            fig.update_layout(barmode="relative", title="Quanto as taxas comem do resultado diário")
            show(fig, 360)
    with b:
        with st.container(border=True):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq["dt"], y=eq["cum_gross"], name="Bruto acumulado", line=dict(color=GREEN, width=2, shape="hv")))
            fig.add_trace(go.Scatter(x=eq["dt"], y=eq["cum_gross"] + eq["cum_fees"], name="Após taxas", line=dict(color=AMBER, width=2, shape="hv"), fill="tonexty", fillcolor="rgba(239,68,68,.18)"))
            fig.update_layout(title="Custo acumulado: a área vermelha é o que foi para taxas")
            show(fig, 360)

    if not fills.empty:
        f2 = fills.copy()
        bps = f2["fee"].abs().sum() / f2["quoteQty"].sum() * 1e4
        kpi_row([("Taxa média por execução", f"{br(bps, 2)} bps", f"round-trip ≈ {br(2 * bps, 1)} bps (~{br(2 * bps / 100, 3)}% do nocional)", "",
                  "Round-trip ≈ 2× esse valor. É o movimento mínimo que o preço precisa dar a seu favor para empatar."),
                 ("Volume executado", br(f2["quoteQty"].sum(), 0), f"{len(f2)} execuções"),
                 ("Taxa por execução", br(f2["fee"].abs().mean(), 4), "USDT, média")], gap=False)
        a, b = st.columns(2)
        with a:
            if f2["orderType"].notna().any():
                gt = f2.groupby("orderType").agg(exec=("fee", "size"), taxa=("fee", lambda s: s.abs().sum()), nocional=("quoteQty", "sum"))
                gt["bps"] = gt["taxa"] / gt["nocional"] * 1e4
                section("Por tipo de ordem")
                st.dataframe(gt.style.format({"taxa": "{:.3f}", "nocional": "{:,.0f}", "bps": "{:.2f}"}), width="stretch")
        with b:
            if f2["role"].notna().any():
                f2["role"] = f2["role"].astype(str).str.lower()
                gr = f2.groupby("role").agg(exec=("fee", "size"), taxa=("fee", lambda s: s.abs().sum()), nocional=("quoteQty", "sum"))
                gr["bps"] = gr["taxa"] / gr["nocional"] * 1e4
                section("Maker × Taker")
                st.dataframe(gr.style.format({"taxa": "{:.3f}", "nocional": "{:,.0f}", "bps": "{:.2f}"}), width="stretch")


# ═════════════════════════ 8. RISCO (novo) ═════════════════════════
with tabs[7]:
    st.caption("Drawdown, perda em dias ruins e simulação de Monte Carlo sobre os seus fechamentos. Ferramentas para dimensionar risco — não previsões.")
    ui_ = ulcer_index(eq_day) if capital > 0 else np.nan
    v_tr, cv_tr = var_cvar(events["net"]) if not events.empty else (np.nan, np.nan)
    kpi_row([
        ("Max drawdown", br(max_dd) + (f" ({pct(max_dd_pct)})" if capital > 0 else ""), "", "bad" if max_dd < 0 else ""),
        ("Drawdown atual", br(cur_dd), "no topo" if cur_dd == 0 else "abaixo do pico", "bad" if cur_dd < 0 else "good"),
        ("Dias debaixo d'água", pct(days_uw), "% dos dias fechando abaixo do topo"),
        ("Recovery factor", br(recovery), "PnL ÷ |max DD|", tone_of(recovery) if not np.isnan(recovery) else ""),
        ("VaR 95% por fechamento", sg(v_tr, 2), f"CVaR (média dos 5% piores): {sg(cv_tr, 2)}"),
        ("Ulcer index", br(ui_) if not np.isnan(ui_) else "–", "informe o capital" if capital <= 0 else "profundidade × duração", "", "Raiz da média dos drawdowns % ao quadrado. Menor é melhor."),
    ])
    a, b = st.columns([1.4, 1])
    with a:
        with st.container(border=True):
            y = (eq["dd_pct"] * 100) if capital > 0 else eq["dd"]
            fig = go.Figure(go.Scatter(x=eq["dt"], y=y, fill="tozeroy", line=dict(color=RED, width=1.6, shape="hv"), fillcolor="rgba(239,68,68,.25)", name="Drawdown"))
            fig.update_layout(title="Curva underwater (distância até o pico)", yaxis_title="% do pico" if capital > 0 else "USDT")
            show(fig, 340, legend=False)
    with b:
        with st.container(border=True):
            fig = go.Figure(go.Histogram(x=daily["net"], nbinsx=25, marker_color=BLUE))
            if not np.isnan(var95):
                fig.add_vline(x=var95, line_color=RED, line_dash="dash", annotation_text=f"VaR95 {sg(var95)}")
            fig.add_vline(x=0, line_color=GREY, line_dash="dot")
            fig.update_layout(title="Distribuição do PnL diário", xaxis_title="USDT")
            show(fig, 340, legend=False)

    section("Maiores episódios de drawdown")
    eps = drawdown_episodes(daily["net"], capital)
    if eps.empty:
        st.info("Sem drawdowns no período — a curva só subiu.")
    else:
        tbl = eps.copy()
        for c_ in ["Topo", "Fundo", "Recuperação"]:
            tbl[c_] = tbl[c_].dt.strftime("%d/%m/%Y").fillna("–")
        if capital <= 0:
            tbl = tbl.drop(columns="Profundidade %")
        st.dataframe(tbl.style.format({"Profundidade": "{:+.2f}", "Profundidade %": "{:.1%}"}, na_rep="–"), width="stretch", hide_index=True)

    section("Simulação de Monte Carlo", "reamostra dos seus fechamentos (bootstrap)")
    if events.empty or len(events) < 10:
        st.info("São necessários pelo menos 10 fechamentos para simular.")
    else:
        c1, c2, c3 = st.columns(3)
        n_tr = c1.slider("Nº de operações futuras", 20, 500, min(100, max(20, len(events))), step=10, key="mc_n")
        n_sims = c2.select_slider("Simulações", [500, 1000, 2000, 5000], value=2000, key="mc_s")
        ruin = c3.slider("Limite de ruína (% do capital perdido)", 10, 100, 50, key="mc_r") / 100 if capital > 0 else 0.5
        mc = run_mc(events["net"].values, n_tr, n_sims, float(capital), float(ruin))
        kpis = [("Prob. de terminar no lucro", pct(mc["prob_profit"]), "", "good" if mc["prob_profit"] >= .5 else "warn"),
                ("Resultado mediano", sg(mc["p50"][-1]), f"5%: {sg(mc['p05'][-1])} · 95%: {sg(mc['p95'][-1])}", tone_of(mc["p50"][-1])),
                ("Drawdown mediano", br(float(np.median(mc["maxdd"]))), f"pior 5%: {br(float(np.percentile(mc['maxdd'], 5)))}", "bad")]
        if capital > 0:
            kpis.append((f"Prob. de perder {ruin:.0%} do capital", pct(mc["prob_ruin"]), "em algum ponto da sequência", "bad" if mc["prob_ruin"] > .05 else "good"))
        kpi_row(kpis)
        a, b = st.columns([1.5, 1])
        with a:
            with st.container(border=True):
                x = mc["x"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=mc["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=mc["p05"], fill="tonexty", fillcolor="rgba(139,92,246,.14)", line=dict(width=0), name="Faixa 5–95%", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=mc["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=mc["p25"], fill="tonexty", fillcolor="rgba(139,92,246,.30)", line=dict(width=0), name="Faixa 25–75%", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=x, y=mc["p50"], line=dict(color=ACCENT, width=2.6), name="Mediana"))
                zero_line(fig)
                fig.update_layout(title=f"Leque de resultados em {n_tr} operações ({n_sims:,} simulações)".replace(",", "."), xaxis_title="nº da operação", yaxis_title="PnL acumulado (USDT)")
                show(fig, 400)
        with b:
            with st.container(border=True):
                fig = go.Figure(go.Histogram(x=mc["final"], nbinsx=40, marker_color=ACCENT))
                fig.add_vline(x=0, line_color=GREY, line_dash="dot")
                fig.add_vline(x=float(np.median(mc["final"])), line_color=AMBER, annotation_text="mediana")
                fig.update_layout(title="Distribuição do resultado final", xaxis_title="USDT")
                show(fig, 400, legend=False)
        st.caption("⚠️ O bootstrap assume que os fechamentos futuros se parecem com os passados e são independentes entre si. Mudança de estratégia, de regime de mercado "
                   "ou sequências de perdas correlacionadas (tilt) tornam a realidade mais arriscada que o simulado. Com poucos fechamentos, o leque é otimista.")


# ═════════════════════════ 9. SPOT ═════════════════════════
with tabs[8]:
    if spot_raw is None or len(spot_raw) == 0:
        st.info("Nenhuma operação spot no arquivo de operations.")
    else:
        sp = add_time(spot_raw, tz)
        if spot_dups:
            st.warning(f"⚠️ O export trazia **{spot_dups} linhas duplicadas** de spot (mesmo trade repetido várias vezes). Foram removidas — veja abaixo apenas as execuções únicas.")
        kpi_row([
            ("Execuções únicas", str(len(sp))),
            ("Volume (USDT)", br(sp["quoteQty"].sum())),
            ("Taxas", sg(sp["fee"].sum(), 4)),
            ("Compras / Vendas", f"{(sp['side'] == 'BUY').sum()} / {(sp['side'] == 'SELL').sum()}"),
        ], gap=False)
        if (sp["side"] == "BUY").sum() == 0:
            st.info("Só há **vendas** no export — sem o preço de compra não é possível calcular PnL do spot. Exporte um período que inclua as compras para ter esse cálculo.")
        else:
            ac = spot_average_cost(sp)
            if not ac.empty:
                section("PnL realizado por custo médio", "só considera vendas com compra anterior no arquivo")
                st.dataframe(ac.rename(columns={"realizado": "PnL realizado", "qtd_em_carteira": "Qtd em carteira", "custo_medio": "Custo médio", "venda_sem_compra": "Venda sem compra"})
                             .style.format({"PnL realizado": "{:+.2f}", "Qtd em carteira": "{:.6g}", "Custo médio": "{:,.4f}", "Venda sem compra": "{:.6g}"}, na_rep="–"), width="stretch")
                if (ac["venda_sem_compra"] > 0).any():
                    st.caption("Símbolos com 'Venda sem compra' têm vendas de posições compradas antes do período exportado — o PnL desses trechos não é calculável.")
        a, b = st.columns(2)
        with a:
            with st.container(border=True):
                vol = sp.groupby(["symbol", "side"])["quoteQty"].sum().unstack().fillna(0)
                fig = go.Figure()
                for s_, cl in [("BUY", GREEN), ("SELL", RED)]:
                    if s_ in vol:
                        fig.add_bar(x=vol.index, y=vol[s_], name="Compras" if s_ == "BUY" else "Vendas", marker_color=cl)
                fig.update_layout(barmode="group", title="Volume por símbolo (USDT)")
                show(fig, 320)
        with b:
            with st.container(border=True):
                fig = go.Figure(go.Scatter(x=sp["dt"], y=sp["quoteQty"], mode="markers", marker=dict(size=9, color=[GREEN if s_ == "BUY" else RED for s_ in sp["side"]]),
                                           text=sp["symbol"], hovertemplate="%{text}<br>%{y:,.2f} USDT<extra></extra>"))
                fig.update_layout(title="Execuções ao longo do tempo (verde = compra, vermelho = venda)")
                show(fig, 320, legend=False)
        st.dataframe(sp[["dt", "symbol", "side", "role", "price", "quantity", "quoteQty", "fee"]].rename(columns={"dt": "Quando", "symbol": "Par", "side": "Lado", "role": "Papel", "price": "Preço", "quantity": "Qtd", "quoteQty": "Valor", "fee": "Taxa"}),
                     width="stretch", hide_index=True)


# ═════════════════════════ 10. DADOS ═════════════════════════
with tabs[9]:
    section("Lançamentos de income (filtrados)")
    show_cols = ["dt", "symbol", "incomeType", "info", "income"]
    st.dataframe(inc[show_cols].rename(columns={"dt": "Quando", "symbol": "Par", "incomeType": "Tipo", "info": "Info", "income": "Valor"}).iloc[::-1], width="stretch", hide_index=True, height=340)
    d1_, d2_, d3_ = st.columns(3)
    d1_.download_button("⬇️ Baixar income filtrado (CSV)", inc[show_cols].to_csv(index=False).encode("utf-8"), "income_filtrado.csv", "text/csv", width="stretch")
    if not events.empty:
        d2_.download_button("⬇️ Baixar fechamentos (CSV)", events[["dt", "symbol", "side", "gross", "close_fee", "net"]].to_csv(index=False).encode("utf-8"), "fechamentos.csv", "text/csv", width="stretch")
    if not rt.empty:
        d3_.download_button("⬇️ Baixar saídas reconstruídas (CSV)", rt.drop(columns=["dt", "day"], errors="ignore").to_csv(index=False).encode("utf-8"), "saidas_reconstruidas.csv", "text/csv", width="stretch")
    if not fills.empty:
        section("Fills de futuros (filtrados)")
        st.dataframe(fills[["dt", "symbol", "side", "positionSide", "price", "quantity", "quoteQty", "fee", "orderType"]].iloc[::-1], width="stretch", hide_index=True, height=300)
