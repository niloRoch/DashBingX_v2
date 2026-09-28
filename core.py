"""
Núcleo de cálculo do BingX Analytics (sem dependência do Streamlit).

Tudo aqui é função pura sobre DataFrames — fácil de testar e de reutilizar em notebooks.
"""
from __future__ import annotations

import io
import json
from collections import deque

import numpy as np
import pandas as pd

DOW = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
SESSIONS = [(0, 8, "Ásia"), (8, 13, "Europa"), (13, 21, "EUA"), (21, 24, "Late night")]  # horas em UTC

INCOME_REQUIRED = ["income", "timestamp", "incomeType", "symbol"]
OPS_REQUIRED = ["market", "symbol", "price", "quantity", "timestamp", "side"]


# ───────────────────────────── formatação (pt-BR) ─────────────────────────────
def _nan(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))


def br(x: float, d: int = 2) -> str:
    """Número no formato brasileiro (1.234,56)."""
    if _nan(x):
        return "–"
    if isinstance(x, (float, np.floating)) and np.isinf(x):
        return "∞"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def sg(x: float, d: int = 2) -> str:
    if _nan(x):
        return "–"
    return ("+" if x > 0 else "") + br(x, d)


def pct(x: float, d: int = 1) -> str:
    return "–" if _nan(x) else br(x * 100, d) + "%"


def fmt_min(m: float) -> str:
    if _nan(m):
        return "–"
    if m < 60:
        return f"{m:.0f}min"
    if m < 1440:
        return f"{int(m // 60)}h{int(m % 60):02d}"
    return f"{int(m // 1440)}d {int((m % 1440) // 60)}h"


# ───────────────────────────── leitura e limpeza ─────────────────────────────
def read_table(b: bytes, ext: str) -> pd.DataFrame:
    if ext == ".json":
        data = json.loads(b.decode("utf-8"))
        if isinstance(data, dict):  # alguns exports vêm embrulhados: {"data": [...]}
            for k in ("data", "rows", "items", "result"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        return pd.json_normalize(data)
    return pd.read_csv(io.BytesIO(b))


def _missing(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def clean_income(df: pd.DataFrame) -> pd.DataFrame:
    miss = _missing(df, INCOME_REQUIRED)
    if miss:
        raise ValueError(f"Faltam colunas no arquivo de income: {', '.join(miss)}")
    df = df.copy()
    df["income"] = pd.to_numeric(df["income"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if "tranId" not in df.columns:
        df["tranId"] = df["id"] if "id" in df.columns else df.index
    key = "id" if "id" in df.columns else "tranId"
    df = df.drop_duplicates(subset=[key, "incomeType"]).copy()
    df["info"] = df["info"].fillna("").astype(str) if "info" in df.columns else ""
    df["tranId"] = df["tranId"].astype(str)
    df["symbol"] = df["symbol"].fillna("—").astype(str)
    df["base"] = df["tranId"].str.replace(r"_(PNL|COMMISSION)$", "", regex=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def clean_ops(df: pd.DataFrame):
    miss = _missing(df, OPS_REQUIRED)
    if miss:
        raise ValueError(f"Faltam colunas no arquivo de operations: {', '.join(miss)}")
    df = df.copy()
    for c in ["price", "quantity", "quoteQty", "fee", "timestamp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    if "quoteQty" in df and df["quoteQty"].isna().all():
        df["quoteQty"] = df["price"] * df["quantity"]
    for c in ("orderId", "tradeId", "positionSide", "role"):
        if c not in df.columns:
            df[c] = np.nan
    df["orderType"] = df["raw.type"] if "raw.type" in df.columns else np.nan
    df = df.dropna(subset=["timestamp"])
    fut = df[df["market"] == "futures"].copy()
    spot = df[df["market"] == "spot"].copy()
    n_spot_raw = len(spot)
    spot = spot.drop_duplicates(subset=["symbol", "orderId", "tradeId", "price", "quantity", "timestamp"])
    fut = fut.drop_duplicates(subset=["orderId", "timestamp", "price", "quantity", "side", "positionSide"])
    return (
        fut.sort_values("timestamp").reset_index(drop=True),
        spot.sort_values("timestamp").reset_index(drop=True),
        n_spot_raw - len(spot),
    )


def add_time(df: pd.DataFrame, tz: str, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    utc = pd.to_datetime(df[ts_col], unit="ms", utc=True)
    df["dt"] = utc.dt.tz_convert(tz).dt.tz_localize(None)
    df["day"] = df["dt"].dt.normalize()
    df["hour"] = df["dt"].dt.hour
    df["wd"] = df["dt"].dt.weekday
    df["hour_utc"] = utc.dt.hour
    return df


def session_of(hour_utc: int) -> str:
    for a, b, name in SESSIONS:
        if a <= hour_utc < b:
            return name
    return "—"


# ───────────────────────────── estatísticas de trades ─────────────────────────────
def streaks(signs: np.ndarray) -> tuple[int, int, int]:
    """(maior sequência de ganhos, maior de perdas, sequência atual com sinal)."""
    best_w = best_l = cur = 0
    for s in signs:
        if s > 0:
            cur = cur + 1 if cur > 0 else 1
        elif s < 0:
            cur = cur - 1 if cur < 0 else -1
        else:
            cur = 0
        best_w, best_l = max(best_w, cur), max(best_l, -cur)
    return best_w, best_l, cur


def trade_stats(pnl: pd.Series) -> dict:
    pnl = pnl.dropna()
    n = len(pnl)
    if n == 0:
        return {}
    w, l = pnl[pnl > 0], pnl[pnl < 0]
    gp, gl = w.sum(), -l.sum()
    avg_w = w.mean() if len(w) else 0.0
    avg_l = l.mean() if len(l) else 0.0
    wr = len(w) / n
    payoff = avg_w / abs(avg_l) if avg_l else np.nan
    sw, sl, cur = streaks(np.sign(pnl.values))
    kelly = wr - (1 - wr) / payoff if payoff and not np.isnan(payoff) else np.nan
    std = pnl.std(ddof=1) if n > 1 else np.nan
    return dict(
        n=n, wins=len(w), losses=len(l), win_rate=wr, avg_win=avg_w, avg_loss=avg_l, payoff=payoff,
        pf=(gp / gl) if gl > 0 else (np.inf if gp > 0 else np.nan), expectancy=pnl.mean(), total=pnl.sum(),
        best=pnl.max(), worst=pnl.min(), gross_profit=gp, gross_loss=gl, streak_w=sw, streak_l=sl, cur_streak=cur,
        kelly=kelly, breakeven_wr=(1 / (1 + payoff)) if payoff and not np.isnan(payoff) else np.nan,
        std=std, median=pnl.median(),
        # SQN de Van Tharp: expectativa / desvio × √n  (>2 bom, >3 excelente; precisa de amostra razoável)
        sqn=(pnl.mean() / std * np.sqrt(min(n, 100))) if n > 1 and std and std > 0 else np.nan,
    )


def stats_by(df: pd.DataFrame, col: str, value: str = "net") -> pd.DataFrame:
    """Tabela de estatísticas de trade agrupada por uma coluna."""
    rows = []
    for k, d in df.groupby(col, observed=True):
        s = trade_stats(d[value])
        if s:
            rows.append({col: k, "Fech.": s["n"], "Win rate": s["win_rate"], "Ganho médio": s["avg_win"], "Perda média": s["avg_loss"],
                         "Payoff": s["payoff"], "Profit factor": s["pf"], "Expectativa": s["expectancy"], "Total": s["total"]})
    return pd.DataFrame(rows).set_index(col) if rows else pd.DataFrame()


# ───────────────────────────── reconstrução de eventos e posições ─────────────────────────────
def build_events(inc: pd.DataFrame) -> pd.DataFrame:
    """Cada linha REALIZED_PNL = um fechamento (parcial ou total), com a taxa de fechamento anexada."""
    ev = inc[inc["incomeType"] == "REALIZED_PNL"].copy()
    if ev.empty:
        return ev
    ev["side"] = np.where(ev["info"].str.contains("Long", case=False), "LONG", "SHORT")
    fees = inc[(inc["incomeType"] == "TRADING_FEE") & inc["info"].str.contains("closing", case=False)]
    fees = fees.groupby("base")["income"].sum().rename("close_fee").reset_index()
    ev = ev.merge(fees, on="base", how="left")
    ev["close_fee"] = ev["close_fee"].fillna(0.0)
    ev = ev.rename(columns={"income": "gross"})
    ev["net"] = ev["gross"] + ev["close_fee"]
    return ev.sort_values("timestamp").reset_index(drop=True)


def build_roundtrips(fills: pd.DataFrame) -> pd.DataFrame:
    """Reconstrói as saídas por FIFO (por símbolo + lado da posição) usando os fills de futuros."""
    if fills.empty:
        return pd.DataFrame()
    books: dict = {}
    pieces, orphans = [], []
    for r in fills.sort_values("timestamp").itertuples():
        key = (r.symbol, r.positionSide)
        is_open = (r.side == "BUY" and r.positionSide == "LONG") or (r.side == "SELL" and r.positionSide == "SHORT")
        fee_unit = abs(r.fee) / r.quantity if r.quantity else 0.0
        if np.isnan(fee_unit):
            fee_unit = 0.0
        q = books.setdefault(key, deque())
        if is_open:
            q.append([r.quantity, r.price, fee_unit, r.timestamp])
            continue
        remaining, direction = r.quantity, (1 if r.positionSide == "LONG" else -1)
        while remaining > 1e-12 and q:
            lot = q[0]
            take = min(lot[0], remaining)
            pieces.append(dict(
                close_ts=r.timestamp, open_ts=lot[3], symbol=r.symbol, side=r.positionSide, qty=take, entry=lot[1],
                exit=r.price, gross=(r.price - lot[1]) * take * direction, fees=(lot[2] + fee_unit) * take,
                exit_type=getattr(r, "orderType", np.nan),
            ))
            lot[0] -= take
            remaining -= take
            if lot[0] <= 1e-12:
                q.popleft()
        if remaining > 1e-12:
            orphans.append(dict(close_ts=r.timestamp, open_ts=np.nan, symbol=r.symbol, side=r.positionSide, qty=remaining,
                                entry=np.nan, exit=r.price, gross=np.nan, fees=fee_unit * remaining,
                                exit_type=getattr(r, "orderType", np.nan)))
    p = pd.DataFrame(pieces)
    if p.empty and not orphans:
        return pd.DataFrame()
    rows = []
    if not p.empty:
        p["notional"] = p["entry"] * p["qty"]
        p["ots_w"] = p["open_ts"] * p["qty"]
        p["exit_type"] = p["exit_type"].astype(object).where(p["exit_type"].notna(), "N/D")
        g = p.groupby(["close_ts", "symbol", "side", "exit_type", "exit"], as_index=False).agg(
            qty=("qty", "sum"), notional=("notional", "sum"), gross=("gross", "sum"), fees=("fees", "sum"), ots_w=("ots_w", "sum"))
        g["entry"] = g["notional"] / g["qty"]
        g["open_ts"] = g["ots_w"] / g["qty"]
        rows.append(g.drop(columns="ots_w"))
    if orphans:
        o = pd.DataFrame(orphans)
        o["notional"] = np.nan
        o["exit_type"] = o["exit_type"].astype(object).where(o["exit_type"].notna(), "N/D")
        rows.append(o)
    rt = pd.concat(rows, ignore_index=True)
    rt["orphan"] = rt["entry"].isna()
    rt["net"] = rt["gross"] - rt["fees"]
    rt["ret_pct"] = rt["gross"] / rt["notional"] * 100
    rt["hold_min"] = (rt["close_ts"] - rt["open_ts"]) / 60000
    return rt.sort_values("close_ts").reset_index(drop=True)


def prepare_roundtrips(rt: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Anexa colunas de tempo (fechamento em `dt`, abertura média em `open_dt`)."""
    if rt.empty:
        return rt
    rt = add_time(rt, tz, ts_col="close_ts")
    rt["open_dt"] = pd.to_datetime(rt["open_ts"], unit="ms", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    return rt


def open_after_previous(rt: pd.DataFrame) -> pd.DataFrame:
    """Para cada posição, o resultado da posição que fechou imediatamente antes de ela abrir (revenge trading)."""
    r = rt[~rt["orphan"]]
    if r.empty:
        return pd.DataFrame()
    s = r.sort_values("open_ts").copy()
    s["open_ts"] = s["open_ts"].astype(float)
    prev = r[["close_ts", "net"]].rename(columns={"close_ts": "prev_close", "net": "prev_net"}).sort_values("prev_close")
    prev["prev_close"] = prev["prev_close"].astype(float)
    s = pd.merge_asof(s, prev, left_on="open_ts", right_on="prev_close", direction="backward", allow_exact_matches=False)
    return s.dropna(subset=["prev_net"])


# ───────────────────────────── equity, drawdown e risco ─────────────────────────────
def equity_frame(inc: pd.DataFrame, capital: float) -> pd.DataFrame:
    e = inc.sort_values("timestamp")[["timestamp", "dt", "day", "incomeType", "income"]].copy()
    e["cum_net"] = e["income"].cumsum()
    for t, name in [("REALIZED_PNL", "cum_gross"), ("TRADING_FEE", "cum_fees"), ("FUNDING_FEE", "cum_funding")]:
        e[name] = e["income"].where(e["incomeType"] == t, 0.0).cumsum()
    e["equity"] = capital + e["cum_net"]
    e["peak"] = e["equity"].cummax().clip(lower=capital)
    e["dd"] = e["equity"] - e["peak"]
    e["dd_pct"] = e["dd"] / e["peak"] if capital > 0 else np.nan
    return e


def daily_frame(inc: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    daily = inc.groupby("day")["income"].sum().rename("net").to_frame()
    for t, name in [("REALIZED_PNL", "gross"), ("TRADING_FEE", "fees"), ("FUNDING_FEE", "funding")]:
        daily[name] = inc[inc["incomeType"] == t].groupby("day")["income"].sum()
    daily = daily.fillna(0.0)
    daily["trades"] = events.groupby("day").size().reindex(daily.index).fillna(0).astype(int) if not events.empty else 0
    return daily


def drawdown_episodes(daily_net: pd.Series, capital: float = 0.0, top: int = 6) -> pd.DataFrame:
    """Episódios de drawdown em resolução diária: do último topo até a recuperação."""
    if daily_net.empty:
        return pd.DataFrame()
    full = daily_net.reindex(pd.date_range(daily_net.index.min(), daily_net.index.max()), fill_value=0.0)
    s = capital + full.cumsum()
    s = pd.concat([pd.Series([float(capital)], index=[s.index[0] - pd.Timedelta(days=1)]), s])
    peak = s.cummax()
    dd = s - peak
    eps, in_dd = [], False
    peak_t = s.index[0]
    trough = 0.0
    trough_t = peak_t
    for t, v in dd.items():
        if v < 0:
            if not in_dd:
                in_dd, trough, trough_t = True, v, t
            elif v < trough:
                trough, trough_t = v, t
        else:
            if in_dd:
                eps.append((peak_t, trough_t, t, trough, peak.loc[peak_t]))
                in_dd = False
            peak_t = t
    if in_dd:
        eps.append((peak_t, trough_t, pd.NaT, trough, peak.loc[peak_t]))
    if not eps:
        return pd.DataFrame()
    out = pd.DataFrame(eps, columns=["Topo", "Fundo", "Recuperação", "Profundidade", "peak_val"])
    last = s.index[-1]
    out["Dias no fundo"] = (out["Fundo"] - out["Topo"]).dt.days
    out["Duração (dias)"] = (out["Recuperação"].fillna(last) - out["Topo"]).dt.days
    out["Status"] = np.where(out["Recuperação"].isna(), "Em aberto", "Recuperado")
    out["Profundidade %"] = np.where(out["peak_val"] > 0, out["Profundidade"] / out["peak_val"], np.nan) if capital > 0 else np.nan
    return out.sort_values("Profundidade").head(top).drop(columns="peak_val").reset_index(drop=True)


def var_cvar(x: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """VaR e CVaR históricos (valores negativos = perda) na mesma unidade de x."""
    x = x.dropna()
    if len(x) < 10:
        return np.nan, np.nan
    q = x.quantile(1 - level)
    tail = x[x <= q]
    return float(q), float(tail.mean()) if len(tail) else np.nan


def ulcer_index(equity: pd.Series) -> float:
    """Raiz da média dos drawdowns percentuais ao quadrado (mede profundidade E duração do sofrimento)."""
    e = equity.dropna()
    if len(e) < 3 or (e <= 0).any():
        return np.nan
    dd = (e / e.cummax() - 1) * 100
    return float(np.sqrt((dd ** 2).mean()))


def monte_carlo(net: np.ndarray, n_trades: int, n_sims: int = 2000, capital: float = 0.0, ruin_pct: float = 0.5, seed: int = 7) -> dict:
    """Bootstrap dos fechamentos: reamostra com reposição para estimar a dispersão de resultados futuros."""
    net = np.asarray(net, dtype=float)
    net = net[~np.isnan(net)]
    if len(net) < 10:
        return {}
    rng = np.random.default_rng(seed)
    paths = np.cumsum(net[rng.integers(0, len(net), (n_sims, n_trades))], axis=1)
    start = capital
    eq = start + paths
    peak = np.maximum(np.maximum.accumulate(eq, axis=1), start)
    maxdd = (eq - peak).min(axis=1)
    final = paths[:, -1]
    res = dict(
        x=np.arange(1, n_trades + 1), final=final, maxdd=maxdd,
        p05=np.percentile(paths, 5, axis=0), p25=np.percentile(paths, 25, axis=0), p50=np.percentile(paths, 50, axis=0),
        p75=np.percentile(paths, 75, axis=0), p95=np.percentile(paths, 95, axis=0),
        prob_profit=float((final > 0).mean()),
    )
    if capital > 0:
        res["prob_ruin"] = float((eq.min(axis=1) <= capital * (1 - ruin_pct)).mean())
    return res


# ───────────────────────────── spot ─────────────────────────────
def spot_average_cost(sp: pd.DataFrame) -> pd.DataFrame:
    """PnL realizado do spot por custo médio. Vendas sem compra anterior no arquivo ficam marcadas."""
    rows = []
    for sym, g in sp.sort_values("timestamp").groupby("symbol"):
        qty = cost = 0.0
        realized = uncovered = 0.0
        for r in g.itertuples():
            if r.side == "BUY":
                qty += r.quantity
                cost += r.quantity * r.price
            else:
                covered = min(qty, r.quantity)
                if covered > 0:
                    avg = cost / qty
                    realized += (r.price - avg) * covered
                    qty -= covered
                    cost -= avg * covered
                uncovered += r.quantity - covered
        rows.append(dict(symbol=sym, realizado=realized, qtd_em_carteira=qty, custo_medio=(cost / qty if qty > 1e-12 else np.nan),
                         venda_sem_compra=uncovered))
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
