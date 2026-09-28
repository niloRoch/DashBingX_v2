"""Dados sintéticos no mesmo schema dos exports da BingX — usados no modo demonstração e em testes."""
from __future__ import annotations

import numpy as np
import pandas as pd

SYMBOLS = {"BTC-USDT": 62000.0, "ETH-USDT": 2600.0, "SOL-USDT": 145.0, "DOGE-USDT": 0.14, "XRP-USDT": 0.55}
TAKER, MAKER = 0.0005, 0.0002


def make_demo(seed: int = 11, days: int = 75, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").normalize()
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    start_ts = end_ts - pd.Timedelta(days=days)
    inc, ops = [], []
    n_id = 0
    # trajetória horária por símbolo (random walk) — entradas e saídas seguem o "mercado"
    n_h = (days + 3) * 24
    grid = np.arange(n_h)
    paths = {s: p0 * np.exp(np.cumsum(rng.normal(0.00005, 0.007, n_h))) for s, p0 in SYMBOLS.items()}

    def price_at(sym: str, t: pd.Timestamp) -> float:
        return float(np.interp((t - start_ts).total_seconds() / 3600, grid, paths[sym]))

    def nid() -> int:
        nonlocal n_id
        n_id += 1
        return n_id

    for d in range(days):
        day0 = start_ts + pd.Timedelta(days=d)
        if day0.weekday() == 6 and rng.random() < 0.6:  # domingo mais calmo
            continue
        for _ in range(rng.poisson(3.2)):
            sym = rng.choice(list(SYMBOLS), p=[.36, .24, .2, .1, .1])
            side = rng.choice(["LONG", "SHORT"], p=[.58, .42])
            hour = int(np.clip(rng.normal(15, 5), 0, 23))
            t_open = day0 + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            hold = float(np.exp(rng.normal(3.6, 1.4)))  # minutos, log-normal
            t_close = t_open + pd.Timedelta(minutes=hold)
            if t_close > end_ts:
                continue
            px0 = price_at(sym, t_open)
            edge = 0.0027 if hold < 90 else 0.0004  # scalps rendem acima do custo, swings ~ neutros (para haver o que descobrir)
            ret = rng.normal(edge, 0.0055) * (1 if side == "LONG" else -1)
            px1 = px0 * (1 + ret)
            notional = float(np.exp(rng.normal(6.2, .7)))
            qty = notional / px0
            splits = [1.0] if rng.random() < .75 else [.5, .5]
            oid_open = f"O{nid()}"
            open_side, close_side = ("BUY", "SELL") if side == "LONG" else ("SELL", "BUY")
            open_fee = -notional * TAKER
            ops.append(dict(market="futures", symbol=sym, orderId=oid_open, tradeId=f"T{nid()}", price=px0, quantity=qty, quoteQty=notional,
                            fee=open_fee, timestamp=int(t_open.timestamp() * 1000), side=open_side, positionSide=side, role="taker",
                            **{"raw.type": "MARKET"}))
            tid = f"{nid()}"
            inc.append(dict(id=nid(), symbol=sym, incomeType="TRADING_FEE", income=open_fee, timestamp=int(t_open.timestamp() * 1000),
                            info=f"{side.title()} opening fee", tranId=f"{tid}_COMMISSION"))
            for k, frac in enumerate(splits):
                tc = t_close + pd.Timedelta(seconds=30 * k)
                q, n_ = qty * frac, notional * frac
                is_tp = rng.random() < .35
                fee_rate = MAKER if is_tp else TAKER
                cfee = -(q * px1) * fee_rate
                gross = (px1 - px0) * q * (1 if side == "LONG" else -1)
                oid = f"O{nid()}"
                ops.append(dict(market="futures", symbol=sym, orderId=oid, tradeId=f"T{nid()}", price=px1, quantity=q, quoteQty=q * px1,
                                fee=cfee, timestamp=int(tc.timestamp() * 1000), side=close_side, positionSide=side,
                                role="maker" if is_tp else "taker", **{"raw.type": "TAKE_PROFIT" if is_tp else "MARKET"}))
                base = f"{nid()}"
                inc.append(dict(id=nid(), symbol=sym, incomeType="REALIZED_PNL", income=gross, timestamp=int(tc.timestamp() * 1000),
                                info=f"{side.title()} position closed", tranId=f"{base}_PNL"))
                inc.append(dict(id=nid(), symbol=sym, incomeType="TRADING_FEE", income=cfee, timestamp=int(tc.timestamp() * 1000),
                                info=f"{side.title()} closing fee", tranId=f"{base}_COMMISSION"))
        # funding a cada 8h
        for h in (0, 8, 16):
            if rng.random() < .5:
                sym = rng.choice(list(SYMBOLS))
                tf = day0 + pd.Timedelta(hours=h)
                inc.append(dict(id=nid(), symbol=sym, incomeType="FUNDING_FEE", income=float(rng.normal(-.01, .03)),
                                timestamp=int(tf.timestamp() * 1000), info="Funding fee", tranId=f"F{nid()}"))
    # spot: só vendas + algumas compras, com linhas duplicadas (como no export real)
    for _ in range(14):
        t = start_ts + pd.Timedelta(days=int(rng.integers(0, days)), hours=int(rng.integers(0, 24)))
        sym = rng.choice(["BTC-USDT", "ETH-USDT", "SOL-USDT"])
        px = price_at(sym, t)
        q = float(rng.uniform(20, 200) / px)
        side = rng.choice(["BUY", "SELL"], p=[.55, .45])
        row = dict(market="spot", symbol=sym, orderId=f"S{nid()}", tradeId=f"ST{nid()}", price=px, quantity=q, quoteQty=q * px,
                   fee=-q * px * .001, timestamp=int(t.timestamp() * 1000), side=side, positionSide=np.nan, role=rng.choice(["maker", "taker"]),
                   **{"raw.type": "LIMIT"})
        ops.extend([row] * int(rng.choice([1, 1, 2, 3])))
    return pd.DataFrame(inc), pd.DataFrame(ops)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    out.mkdir(exist_ok=True)
    a, b = make_demo()
    a.to_csv(out / "bingx_futures_income.csv", index=False)
    b.to_csv(out / "bingx_operations.csv", index=False)
    print(f"{len(a)} lançamentos de income e {len(b)} operações gravados em {out}/")
