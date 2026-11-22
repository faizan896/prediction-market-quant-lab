"""Vectorized backtest engine for binary prediction-market contracts.

Design
------
The panel is long-format: one row per (market, timestamp). A strategy returns a
`signal` in [-1, 1] and a `fair_prob` for every row. The engine never loops over
bars — it converts signals into *position segments* with groupby operations, then
prices every trade in one vectorized pass.

Everything is expressed in YES terms:
    side = +1   buy YES  at the ask
    side = -1   buy NO   (economically short YES) at the bid
    pnl        = side * contracts * (exit_price - entry_price)
which is exact for both legs, because a NO contract at 1-p is the mirror of a
YES contract at p.

Lookahead protection
--------------------
Signals are shifted forward by `signal_lag` bars (default 1) inside the engine,
*after* the strategy has run. A strategy therefore cannot accidentally trade on
the bar it used to compute the signal, no matter how it is written. Exits fill
on the bar after the exit decision, for the same reason. This is the single most
common way prediction-market backtests lie to you.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.backtest import metrics as M
from quantlab.backtest.frictions import PRESETS, FrictionModel
from quantlab.backtest.sizing import SizingConfig, position_notional

REQUIRED_COLUMNS = ("key", "ts", "price")


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    frictions: FrictionModel = field(default_factory=lambda: PRESETS["conservative"])
    sizing: SizingConfig = field(default_factory=SizingConfig)
    signal_lag: int = 1  # bars between signal computation and fill
    entry_threshold: float = 0.5  # |signal| needed to open
    exit_threshold: float = 0.2  # |signal| below which an open position closes
    max_hold_bars: int | None = None
    settle_at_resolution: bool = True
    compounding: bool = True
    max_gross_exposure: float = 1.0  # cap on simultaneous deployed capital
    min_price: float = 0.02  # skip sub-2c contracts; the tails are unhedgeable
    max_price: float = 0.98


@dataclass
class BacktestResult:
    run_id: str
    strategy: str
    params: dict
    trades: pd.DataFrame
    equity: pd.Series
    metrics: M.Metrics

    def summary_row(self) -> dict:
        m = self.metrics
        return {
            "strategy": self.strategy,
            "trades": m.n_trades,
            "win_rate": m.win_rate,
            "roi": m.roi,
            "sharpe": m.sharpe,
            "max_dd": m.max_drawdown,
            "profit_factor": m.profit_factor,
            "brier": m.brier,
            "edge_capture": m.edge_capture,
        }


# --------------------------------------------------------------------------- #
# signal -> positions
# --------------------------------------------------------------------------- #
def _apply_hysteresis(
    signal: np.ndarray,
    group_start: np.ndarray,
    entry_threshold: float,
    exit_threshold: float,
    can_open: np.ndarray | None = None,
) -> np.ndarray:
    """Turn a continuous signal into a held position in {-1, 0, +1}.

    Two thresholds, not one: open at `entry_threshold`, hold until the signal
    decays past `exit_threshold`. A single threshold makes the strategy churn
    in and out around the boundary and pay the spread every time — the fastest
    way to convert a real edge into a loss.

    `can_open` gates *entries* only. A price filter must never force an exit:
    nobody liquidates you because the contract drifted to 1.5c, and modelling
    it that way silently fragments long holds into a string of round trips,
    each paying the spread. Positions are held through untradable bars.
    """
    n = signal.size
    pos = np.zeros(n, dtype=np.int8)
    current = 0
    for i in range(n):  # single O(n) pass; the pricing work stays vectorized
        if group_start[i]:
            current = 0
        s = signal[i]
        openable = True if can_open is None else bool(can_open[i])
        if current == 0 and openable:
            if s >= entry_threshold:
                current = 1
            elif s <= -entry_threshold:
                current = -1
        elif current == 1:
            if s < exit_threshold:
                current = -1 if (s <= -entry_threshold and openable) else 0
        elif current == -1 and s > -exit_threshold:
            current = 1 if (s >= entry_threshold and openable) else 0
        pos[i] = current
    return pos


def _segments(panel: pd.DataFrame, pos: np.ndarray) -> pd.DataFrame:
    """Collapse constant-position runs into (entry_idx, exit_idx) rows."""
    key = panel["key"].to_numpy()
    new_group = np.empty(len(panel), dtype=bool)
    new_group[0] = True
    new_group[1:] = key[1:] != key[:-1]

    changed = np.empty(len(panel), dtype=bool)
    changed[0] = True
    changed[1:] = (pos[1:] != pos[:-1]) | new_group[1:]
    seg_id = changed.cumsum()

    idx = np.arange(len(panel))
    df = pd.DataFrame({"seg": seg_id, "pos": pos, "key": key, "i": idx})
    df = df[df["pos"] != 0]
    if df.empty:
        return pd.DataFrame(columns=["key", "pos", "start", "stop"])

    g = df.groupby("seg", sort=True)
    return pd.DataFrame({
        "key": g["key"].first().to_numpy(),
        "pos": g["pos"].first().to_numpy(),
        "start": g["i"].min().to_numpy(),
        "stop": g["i"].max().to_numpy(),
    })


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def run_backtest(
    panel: pd.DataFrame,
    strategy,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    for col in REQUIRED_COLUMNS:
        if col not in panel.columns:
            raise ValueError(f"panel is missing required column '{col}'")

    panel = panel.copy()
    # Panels arriving straight from a client (rather than the store) carry
    # object-dtype timestamps; coerce once so the vectorized math below works.
    if not pd.api.types.is_datetime64_any_dtype(panel["ts"]):
        panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    if "close_time" in panel.columns and not pd.api.types.is_datetime64_any_dtype(panel["close_time"]):
        panel["close_time"] = pd.to_datetime(panel["close_time"], utc=True, errors="coerce")
    panel = panel.sort_values(["key", "ts"], kind="stable").reset_index(drop=True)

    # ---- 1. strategy produces signals -------------------------------------
    out = strategy.generate(panel)
    if isinstance(out, pd.Series):
        out = out.to_frame("signal")
    signal = out["signal"].to_numpy(dtype=float)
    fair = out["fair_prob"].to_numpy(dtype=float) if "fair_prob" in out else np.full(len(panel), np.nan)
    signal = np.nan_to_num(signal, nan=0.0)

    # ---- 2. enforce the lag (anti-lookahead) ------------------------------
    if cfg.signal_lag > 0:
        s = pd.Series(signal).groupby(panel["key"].to_numpy()).shift(cfg.signal_lag)
        signal = s.to_numpy(dtype=float)
        signal = np.nan_to_num(signal, nan=0.0)
        f = pd.Series(fair).groupby(panel["key"].to_numpy()).shift(cfg.signal_lag)
        fair = f.to_numpy(dtype=float)

    # ---- 3. tradability filter (entries only) ------------------------------
    price = panel["price"].to_numpy(dtype=float)
    can_open = (price >= cfg.min_price) & (price <= cfg.max_price) & np.isfinite(price)
    signal = np.where(np.isfinite(price), signal, 0.0)

    # ---- 4. positions and segments ----------------------------------------
    key_arr = panel["key"].to_numpy()
    group_start = np.empty(len(panel), dtype=bool)
    group_start[0] = True
    group_start[1:] = key_arr[1:] != key_arr[:-1]
    pos = _apply_hysteresis(
        signal, group_start, cfg.entry_threshold, cfg.exit_threshold, can_open
    )
    segs = _segments(panel, pos)

    if segs.empty:
        equity = pd.Series([cfg.initial_capital])
        return BacktestResult(
            run_id=uuid.uuid4().hex[:12],
            strategy=getattr(strategy, "name", strategy.__class__.__name__),
            params=getattr(strategy, "params", {}),
            trades=pd.DataFrame(),
            equity=equity,
            metrics=M.compute(pd.DataFrame(), equity, cfg.initial_capital),
        )

    # optional forced exit after N bars
    if cfg.max_hold_bars:
        segs["stop"] = np.minimum(segs["stop"], segs["start"] + cfg.max_hold_bars - 1)

    trades = _price_segments(panel, segs, fair, cfg)
    trades = trades[trades["notional"] > 0].reset_index(drop=True)

    equity = _equity_curve(trades, cfg)
    mtx = M.compute(trades, equity, cfg.initial_capital)

    return BacktestResult(
        run_id=uuid.uuid4().hex[:12],
        strategy=getattr(strategy, "name", strategy.__class__.__name__),
        params=getattr(strategy, "params", {}),
        trades=trades,
        equity=equity,
        metrics=mtx,
    )


def _col(panel: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name in panel.columns:
        return panel[name].to_numpy(dtype=float)
    return np.full(len(panel), default, dtype=float)


def _price_segments(
    panel: pd.DataFrame,
    segs: pd.DataFrame,
    fair: np.ndarray,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Vectorized pricing of every position segment."""
    price = panel["price"].to_numpy(dtype=float)
    bid = _col(panel, "yes_bid")
    ask = _col(panel, "yes_ask")
    liq = _col(panel, "liquidity", 0.0)
    # tz-aware Series come back from .to_numpy() as object dtype, which breaks
    # timedelta arithmetic — go through the underlying int64 epoch instead.
    ts_index = pd.DatetimeIndex(panel["ts"])
    ts = ts_index.to_numpy(dtype="datetime64[ns]", copy=False) if ts_index.tz is None \
        else ts_index.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    key_arr = panel["key"].to_numpy()
    resolution = _col(panel, "resolution")

    start = segs["start"].to_numpy()
    stop = segs["stop"].to_numpy()
    side = segs["pos"].to_numpy().astype(float)

    # Last bar available for each market, so we know when a segment runs to expiry.
    last_idx = pd.Series(np.arange(len(panel))).groupby(key_arr).transform("max").to_numpy()
    market_last = last_idx[stop]

    # Exit fills on the bar *after* the exit decision, unless we ran out of bars.
    exit_idx = np.minimum(stop + 1, market_last)
    ran_to_expiry = stop >= market_last
    outcome = resolution[stop]
    settled = cfg.settle_at_resolution & ran_to_expiry & np.isfinite(outcome)

    fr = cfg.frictions
    entry_px = fr.entry_price(price[start], side, bid[start], ask[start])
    market_exit_px = fr.exit_price(price[exit_idx], side, bid[exit_idx], ask[exit_idx])
    exit_px = np.where(settled, outcome, market_exit_px)

    # ---- sizing ------------------------------------------------------------
    seg_fair = fair[start]
    seg_fair = np.where(np.isfinite(seg_fair), seg_fair, entry_px)
    seg_liq = liq[start]
    seg_liq = np.where(np.isfinite(seg_liq) & (seg_liq > 0), seg_liq, np.nan)

    notional = position_notional(
        cfg.sizing,
        cfg.initial_capital,
        seg_fair,
        entry_px,
        side,
        liquidity=None if np.all(np.isnan(seg_liq)) else np.nan_to_num(seg_liq, nan=1e12),
    )

    # Cost basis per contract: p for a YES leg, 1-p for a NO leg.
    basis = np.where(side > 0, entry_px, 1.0 - entry_px)
    basis = np.clip(basis, 1e-4, None)
    contracts = notional / basis

    gross_pnl = side * contracts * (exit_px - entry_px)
    # Slippage is charged on the cash that changes hands, which for the NO leg
    # is (1 - yes_price) per contract. Settlement is not a trade, so a position
    # held to resolution pays no exit costs. Both match the paper broker.
    exit_basis = np.where(side > 0, exit_px, 1.0 - exit_px)
    fees = fr.fees(entry_px, contracts) + np.where(settled, 0.0, fr.fees(exit_px, contracts))
    slip = fr.slippage(basis * contracts, seg_liq) + np.where(
        settled, 0.0, fr.slippage(exit_basis * contracts, seg_liq)
    )
    costs = fees + slip
    pnl = gross_pnl - costs

    hold = (ts[exit_idx] - ts[start]) / np.timedelta64(1, "h")

    trades = pd.DataFrame({
        "key": segs["key"].to_numpy(),
        "side": np.where(side > 0, "YES", "NO"),
        "ts_entry": ts[start],
        "ts_exit": ts[exit_idx],
        "entry_price": entry_px,
        "exit_price": exit_px,
        "fair_prob": seg_fair,
        "edge": side * (seg_fair - entry_px),
        "contracts": contracts,
        "notional": notional,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "slippage": slip,
        "pnl": pnl,
        "ret": np.where(notional > 0, pnl / np.where(notional > 0, notional, 1), 0.0),
        "gross_ret": np.where(notional > 0, gross_pnl / np.where(notional > 0, notional, 1), 0.0),
        "hold_hours": hold.astype(float),
        "settled": settled,
        "outcome": np.where(settled, outcome, np.nan),
    })
    return trades.sort_values("ts_exit", kind="stable").reset_index(drop=True)


def _equity_curve(trades: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    """Equity marked at trade exits.

    With `compounding` on, later trades are rescaled by the bankroll at the time
    they were opened. This is an approximation — proper compounding requires
    sequencing overlapping positions — but it is the honest direction of the
    error: it shrinks size after drawdowns, as a real account would.
    """
    if trades.empty:
        return pd.Series([cfg.initial_capital])

    if not cfg.compounding:
        eq = cfg.initial_capital + trades["pnl"].cumsum()
        return pd.Series(eq.to_numpy(), index=trades["ts_exit"])

    equity = cfg.initial_capital
    curve = np.empty(len(trades))
    scale_col = np.empty(len(trades))
    for i, (pnl, notional) in enumerate(zip(trades["pnl"].to_numpy(), trades["notional"].to_numpy())):
        scale = max(equity, 0.0) / cfg.initial_capital
        scale = min(scale, cfg.max_gross_exposure / max(cfg.sizing.max_fraction, 1e-9))
        scale_col[i] = scale
        equity += pnl * scale
        curve[i] = equity
    trades["scale"] = scale_col
    for col in ("pnl", "notional", "fees", "slippage", "gross_pnl"):
        trades[col] = trades[col] * scale_col
    return pd.Series(curve, index=trades["ts_exit"])


# --------------------------------------------------------------------------- #
# batch comparison — this is what fills the QUANT LAB leaderboard
# --------------------------------------------------------------------------- #
def compare(
    panel: pd.DataFrame,
    strategies: list,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    results: dict[str, BacktestResult] = {}
    rows = []
    for strat in strategies:
        res = run_backtest(panel, strat, config)
        results[res.strategy] = res
        rows.append(res.summary_row())
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return df, results


def walk_forward(
    panel: pd.DataFrame,
    strategy_factory,
    n_folds: int = 4,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Split the panel chronologically and backtest each fold separately.

    A strategy whose Sharpe is 2.1 in-sample and 0.3 out-of-sample is not a
    strategy, it is a curve fit. Always run this before believing a number.
    """
    panel = panel.sort_values("ts", kind="stable")
    bounds = pd.qcut(panel["ts"].rank(method="first"), n_folds, labels=False)
    rows = []
    for fold in range(n_folds):
        sub = panel[bounds == fold]
        if sub["key"].nunique() < 3:
            continue
        res = run_backtest(sub.reset_index(drop=True), strategy_factory(), config)
        row = res.summary_row()
        row["fold"] = fold
        row["start"] = sub["ts"].min()
        row["end"] = sub["ts"].max()
        rows.append(row)
    return pd.DataFrame(rows)
