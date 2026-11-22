"""Performance and calibration metrics.

Two families, and you need both:

Performance  — did the strategy make money, and was it worth the variance.
Calibration  — was the *model* right. A strategy can be profitable with a badly
               calibrated model (it got lucky on a few longshots) and that will
               not survive contact with the next 200 markets. Brier score and
               the reliability curve are how you tell those apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass
class Metrics:
    n_trades: int = 0
    win_rate: float = 0.0
    roi: float = 0.0  # total PnL / total capital deployed
    total_pnl: float = 0.0
    total_deployed: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_hold_hours: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_edge: float = 0.0
    realized_edge: float = 0.0
    edge_capture: float = 0.0  # realized / predicted; <1 means costs ate it
    brier: float = float("nan")
    log_loss: float = float("nan")
    total_fees: float = 0.0
    total_slippage: float = 0.0
    final_equity: float = 0.0
    cagr: float = 0.0
    reliability: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(x: float, default: float = 0.0) -> float:
    return float(x) if np.isfinite(x) else default


def sharpe_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Sharpe on per-trade returns, annualized by trade frequency.

    Prediction-market trade returns are wildly non-normal (bimodal: you win
    (1-b)/b or lose everything), so read this as a rough ranking device, not as
    a probability statement.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    # A constant series has zero variance mathematically but ~1e-18 in floats,
    # which would report an absurd Sharpe instead of 0.
    if sd < 1e-12:
        return 0.0
    return _safe(r.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    downside = r[r < 0]
    if r.size < 2 or downside.size < 2 or downside.std(ddof=1) < 1e-12:
        return 0.0
    return _safe(r.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough decline as a fraction of the running peak."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
    return _safe(dd.max())


def brier_score(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better.

    Reference points: always predicting the base rate on a balanced book gives
    0.25; a perfect oracle gives 0. Market prices on liquid Polymarket/Kalshi
    contracts typically land around 0.10-0.16, so beating ~0.12 is the bar.
    """
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def log_loss(prob: np.ndarray, outcome: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    y = np.asarray(outcome, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    if mask.sum() == 0:
        return float("nan")
    p, y = p[mask], y[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_curve(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> list[dict]:
    """Bucketed forecast-vs-frequency table. If your 70% bucket resolves YES
    55% of the time, you are systematically overconfident and Kelly will
    over-size every bet."""
    p = np.asarray(prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size == 0:
        return []
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        out.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "n": n,
            "predicted": float(p[sel].mean()),
            "actual": float(y[sel].mean()),
            "gap": float(y[sel].mean() - p[sel].mean()),
        })
    return out


def compute(trades: pd.DataFrame, equity: pd.Series, initial_capital: float) -> Metrics:
    """Roll a trades table + equity curve into the full metric set.

    Expected trades columns: pnl, notional, ret, hold_hours, fees, slippage,
    and optionally fair_prob / outcome / edge for calibration.
    """
    m = Metrics()
    if trades.empty:
        m.final_equity = float(equity.iloc[-1]) if len(equity) else initial_capital
        return m

    pnl = trades["pnl"].to_numpy(dtype=float)
    notional = trades["notional"].to_numpy(dtype=float)
    ret = trades["ret"].to_numpy(dtype=float)

    m.n_trades = len(trades)
    m.win_rate = float((pnl > 0).mean())
    m.total_pnl = float(pnl.sum())
    m.total_deployed = float(notional.sum())
    m.roi = float(m.total_pnl / m.total_deployed) if m.total_deployed > 0 else 0.0
    m.avg_pnl_per_trade = float(pnl.mean())
    m.expectancy = float(ret.mean())

    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    m.profit_factor = _safe(wins.sum() / abs(losses.sum())) if losses.size else float("inf")
    if not np.isfinite(m.profit_factor):
        m.profit_factor = 999.0

    if "hold_hours" in trades:
        m.avg_hold_hours = float(trades["hold_hours"].mean())

    # Annualization: scale by how many trades a year this cadence implies.
    span_hours = m.avg_hold_hours if m.avg_hold_hours > 0 else 24.0
    trades_per_year = min(8760.0 / span_hours, 5000.0)
    m.sharpe = sharpe_ratio(ret, trades_per_year)
    m.sortino = sortino_ratio(ret, trades_per_year)

    eq = equity.to_numpy(dtype=float)
    m.max_drawdown = max_drawdown(eq)
    m.final_equity = float(eq[-1]) if eq.size else initial_capital

    if "ts_entry" in trades and len(trades) > 1:
        span = (trades["ts_exit"].max() - trades["ts_entry"].min()).total_seconds()
        years = span / SECONDS_PER_YEAR
        # Annualizing a two-week sample produces meaningless numbers in the
        # billions. Require a month of data before quoting a CAGR at all.
        if years >= 1 / 12 and initial_capital > 0 and m.final_equity > 0:
            m.cagr = float((m.final_equity / initial_capital) ** (1 / years) - 1)
    m.calmar = _safe(m.cagr / m.max_drawdown) if m.max_drawdown > 0 else 0.0

    m.total_fees = float(trades.get("fees", pd.Series([0.0])).sum())
    m.total_slippage = float(trades.get("slippage", pd.Series([0.0])).sum())

    if "edge" in trades:
        m.avg_edge = float(trades["edge"].abs().mean())
    if "gross_ret" in trades:
        m.realized_edge = float(trades["gross_ret"].mean())
    predicted = m.avg_edge
    if m.avg_edge > 0 and "entry_price" in trades and "side" in trades:
        # Edge is per contract; expectancy is per dollar of basis. Convert the
        # edge to the same units before dividing, as the paper portfolio does.
        px = trades["entry_price"].to_numpy(dtype=float)
        basis = np.clip(np.where(trades["side"].to_numpy() == "YES", px, 1 - px), 1e-6, None)
        predicted = float(np.mean(trades["edge"].abs().to_numpy(dtype=float) / basis))
    if predicted > 0:
        m.edge_capture = float(m.expectancy / predicted)

    if "fair_prob" in trades and "outcome" in trades:
        fp = trades["fair_prob"].to_numpy(dtype=float)
        oc = trades["outcome"].to_numpy(dtype=float)
        m.brier = brier_score(fp, oc)
        m.log_loss = log_loss(fp, oc)
        m.reliability = reliability_curve(fp, oc)

    return m
