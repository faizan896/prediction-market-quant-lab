"""Service layer: everything the routes need, with no FastAPI in it.

Keeping this framework-free means the same functions are callable from the CLI,
a notebook, or a scheduled job — and testable without spinning up a server.
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab import strategies as S
from quantlab.api import schemas as sc
from quantlab.backtest import PRESETS, BacktestConfig, run_backtest, walk_forward
from quantlab.backtest.sizing import SizingConfig
from quantlab.data import KalshiClient, PolymarketClient, Store, SyntheticClient
from quantlab.execution import PaperTrader, RiskLimits
from quantlab.execution.portfolio import mark_to_market
from quantlab.mispricing import (
    CrossVenue,
    ExternalProbability,
    LongshotCorrection,
    MarketPrior,
    MispricingEngine,
    OrderBookImbalance,
    VolumeWeightedDrift,
)
from quantlab.types import Market

SIGNAL_TYPES = {
    "market": MarketPrior,
    "cross_venue": CrossVenue,
    "book_imbalance": OrderBookImbalance,
    "longshot": LongshotCorrection,
    "drift": VolumeWeightedDrift,
}


def _clean(x) -> float | None:
    """JSON has no NaN. Anything non-finite becomes null."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------- markets
def synthetic_markets(
    n: int,
    seed: int = 11,
    steps: int = 60,
    spread: float = 0.005,
    resolve_fraction: float = 0.0,
) -> list[Market]:
    """Generated markets shaped like *live* ones, for demos and smoke tests.

    Three deliberate adjustments to the raw generator output:

    - **Open, not resolved.** The generator always stamps a resolution, because
      the backtester needs one. A scanner must never see it: a market with a
      known outcome is not tradable, and the risk screen correctly refuses every
      one of them. `resolve_fraction` re-resolves a slice so the settlement path
      is still exercisable.
    - **Real time on the clock.** The default start makes everything expire
      exactly now, which zeroes every time-to-expiry signal — they abstain, quite
      correctly, and the scanner has nothing to reason about.
    - **A 0.5c spread**, not the generator's 2c. Liquid Polymarket and Kalshi
      contracts quote far tighter, and at 2c the spread alone exceeds nearly any
      modelled edge — an honest lesson, but a demo in which nothing ever trades.
    """
    from datetime import timedelta

    start = datetime.now(timezone.utc) - timedelta(hours=steps // 2)
    markets = SyntheticClient(seed=seed).generate(
        n_markets=n, n_steps=steps, spread=spread, start=start,
        bias_fn=SyntheticClient.longshot_bias(0.05),
    )[0]

    # Which markets are resolved must vary with the seed, not with position in
    # the list. A fixed prefix would mean the scanner (which never opens a
    # resolved market) always holds names from the unresolved tail, and those
    # same names stay unresolved on every later cycle — so settlement would
    # never fire and the loop would silently only ever test entries.
    frac = max(0.0, min(1.0, resolve_fraction))
    rng = np.random.default_rng(seed + 991)
    flags = rng.random(len(markets)) < frac
    for m, is_resolved in zip(markets, flags):
        if is_resolved:
            m.status = "closed"
            m.resolved = True
        else:
            m.status = "open"
            m.resolved = False
            m.resolution = None
    return markets


def fetch_markets(
    venues: list[str],
    limit: int = 300,
    min_volume: float = 50_000,
    warnings: list[str] | None = None,
) -> list[Market]:
    """Fetch from each venue, tolerating failure of any one of them.

    A venue that is slow, rate-limiting or down must not take the whole scan
    with it. Previously one bad venue raised and the caller got a 504 with no
    explanation; now the other venue's markets still come back and the reason
    lands in `warnings` where the UI can show it.
    """
    out: list[Market] = []
    for v in venues:
        if v == "synthetic":
            out += synthetic_markets(limit)
            continue
        client = PolymarketClient() if v == "polymarket" else KalshiClient()
        try:
            out += client.list_markets(limit=limit, min_volume=min_volume)
        except Exception as exc:  # noqa: BLE001 - one venue must not sink the scan
            msg = f"{v}: unreachable ({type(exc).__name__})"
            if warnings is not None:
                warnings.append(msg)
        finally:
            client.close()
    return out


def market_out(m: Market) -> sc.MarketOut:
    return sc.MarketOut(
        key=m.key, venue=m.venue.value, market_id=m.market_id, event_id=m.event_id,
        title=m.title, slug=m.slug, yes_bid=_clean(m.yes_bid), yes_ask=_clean(m.yes_ask),
        mid=_clean(m.mid), spread=_clean(m.spread), volume=m.volume,
        volume_24h=m.volume_24h, liquidity=m.liquidity,
        close_time=m.close_time.isoformat() if m.close_time else None,
        status=m.status, resolved=m.resolved, resolution=_clean(m.resolution),
    )


def stored_markets(store: Store, venue: str | None, min_volume: float, limit: int) -> list[dict]:
    df = store.markets_df(venue=venue, min_volume=min_volume)
    if df.empty:
        return []
    df = df.nlargest(min(limit, len(df)), "volume")
    df["mid"] = df[["yes_bid", "yes_ask"]].mean(axis=1).fillna(df["last_price"])
    df["spread"] = df["yes_ask"] - df["yes_bid"]
    keep = ["key", "venue", "market_id", "event_id", "title", "slug", "yes_bid",
            "yes_ask", "mid", "spread", "volume", "volume_24h", "liquidity",
            "close_time", "status", "resolved", "resolution"]
    rows = df[keep].replace({np.nan: None}).to_dict("records")
    for r in rows:
        r["resolved"] = bool(r["resolved"])
    return rows


# --------------------------------------------------------------------- scanner
def build_engine(req: sc.ScanRequest) -> MispricingEngine:
    if req.signals is None:
        signals = [MarketPrior(), CrossVenue(), OrderBookImbalance(),
                   LongshotCorrection(), VolumeWeightedDrift()]
        weights = None
    else:
        signals, weights = [], {}
        for spec in req.signals:
            if not spec.enabled:
                continue
            sig = SIGNAL_TYPES[spec.name]()
            signals.append(sig)
            weights[spec.name] = spec.weight if spec.weight is not None else sig.default_weight

    if req.external:
        table = {k: (float(v[0]), float(v[1]) if len(v) > 1 else 0.8)
                 for k, v in req.external.items()}
        signals.append(ExternalProbability(table, source="external"))
        weights = weights or {s.name: s.default_weight for s in signals}
        weights["external"] = 2.5

    return MispricingEngine(
        signals=signals, weights=weights, frictions=PRESETS[req.frictions],
        bankroll=req.bankroll, market_anchor=req.market_anchor,
    )


def run_scan(req: sc.ScanRequest, markets: list[Market] | None = None) -> sc.ScanResponse:
    t0 = time.perf_counter()
    warnings: list[str] = []
    markets = markets if markets is not None else fetch_markets(
        req.venues, req.limit, req.min_volume, warnings
    )

    books = {}
    if req.use_books:
        for m in markets:
            client = PolymarketClient() if m.venue.value == "polymarket" else KalshiClient()
            try:
                ob = client.order_book(m.market_id)
                if ob:
                    books[m.key] = ob
            except Exception:  # noqa: BLE001, S110 - one bad book must not kill the scan
                pass
            finally:
                client.close()

    df = build_engine(req).scan(
        markets, books=books, min_edge=req.min_edge, min_net_ev=req.min_net_ev,
        min_confidence=req.min_confidence, min_liquidity=req.min_liquidity,
    )
    opps = []
    if not df.empty:
        for row in df.to_dict("records"):
            row = {k: (_clean(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in row.items()}
            row["hours_to_close"] = _clean(row.get("hours_to_close"))
            opps.append(sc.OpportunityOut(**row))

    if not markets and not warnings:
        warnings.append("No markets matched the universe filters.")

    return sc.ScanResponse(
        scanned=len(markets), returned=len(opps), opportunities=opps,
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        warnings=warnings,
    )


# -------------------------------------------------------------------- backtest
def load_panel(req: sc.BacktestRequest, store: Store) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if req.synthetic:
        gen = SyntheticClient(seed=7)
        markets, points = gen.generate(
            n_markets=req.synthetic_markets, n_steps=req.synthetic_steps,
            spread=req.synthetic_spread,
            bias_fn=SyntheticClient.longshot_bias(req.synthetic_bias),
        )
        mem = Store(path=":memory:")
        mem.upsert_markets(markets)
        mem.upsert_prices(points)
        warnings.append(
            "Synthetic data with a planted longshot bias. Useful for validating "
            "the engine, useless for validating a strategy."
        )
        return mem.panel(), warnings

    panel = store.panel(venue=req.venue)
    if panel.empty:
        warnings.append("No stored price history — run ingest and backfill first.")
    return panel, warnings


def run_backtests(req: sc.BacktestRequest, store: Store) -> sc.BacktestResponse:
    panel, warnings = load_panel(req, store)
    run_id = uuid.uuid4().hex[:12]

    if panel.empty:
        return sc.BacktestResponse(
            run_id=run_id, markets=0, bars=0, frictions=req.frictions,
            leaderboard=[], warnings=warnings,
        )

    specs = req.strategies or [sc.StrategySpec(name=n) for n in
                               ("momentum", "meanrev", "longshot", "breakout",
                                "timedecay", "buyhold")]
    cfg = BacktestConfig(
        initial_capital=req.capital, frictions=PRESETS[req.frictions],
        signal_lag=req.signal_lag,
        sizing=SizingConfig(
            kelly_multiplier=req.kelly_multiplier, max_fraction=req.max_fraction
        ),
    )

    board, equity, reliability = [], {}, {}
    for spec in specs:
        try:
            strat = S.build(spec.name, **spec.params)
        except (KeyError, TypeError) as exc:
            warnings.append(f"{spec.name}: {exc}")
            continue
        res = run_backtest(panel, strat, cfg)
        m = res.metrics
        board.append(sc.MetricsOut(
            strategy=res.strategy, trades=m.n_trades, win_rate=m.win_rate,
            roi=m.roi, sharpe=m.sharpe, sortino=m.sortino,
            max_drawdown=m.max_drawdown, profit_factor=m.profit_factor,
            brier=_clean(m.brier), edge_capture=_clean(m.edge_capture),
            total_pnl=m.total_pnl, total_fees=m.total_fees,
            total_slippage=m.total_slippage, final_equity=m.final_equity,
            avg_hold_hours=m.avg_hold_hours,
        ))
        equity[res.strategy] = _downsample(res.equity)
        if m.reliability:
            reliability[res.strategy] = m.reliability
        store.save_run(res.run_id, res.strategy, res.params, m.to_dict())

    board.sort(key=lambda r: -r.sharpe)

    folds: list[sc.FoldOut] = []
    if req.walk_forward_folds >= 2 and specs:
        head = specs[0]
        fdf = walk_forward(
            panel, lambda: S.build(head.name, **head.params),
            n_folds=req.walk_forward_folds, config=cfg,
        )
        folds = [
            sc.FoldOut(fold=int(r.fold), trades=int(r.trades),
                       win_rate=float(r.win_rate), roi=float(r.roi),
                       sharpe=float(r.sharpe))
            for r in fdf.itertuples()
        ]
        if folds:
            spread = max(f.sharpe for f in folds) - min(f.sharpe for f in folds)
            if spread >= 1.0:
                warnings.append(
                    f"{head.name}: Sharpe swings {spread:.1f} across folds — "
                    "this is a curve fit, not a strategy."
                )

    return sc.BacktestResponse(
        run_id=run_id, markets=int(panel["key"].nunique()), bars=len(panel),
        frictions=req.frictions, leaderboard=board, equity=equity,
        reliability=reliability, folds=folds, warnings=warnings,
    )


def _downsample(equity: pd.Series, max_points: int = 400) -> list[sc.EquityPoint]:
    if equity is None or len(equity) == 0:
        return []
    s = equity
    if len(s) > max_points:
        s = s.iloc[:: max(1, len(s) // max_points)]
    out = []
    for ts, val in s.items():
        label = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        v = _clean(val)
        if v is not None:
            out.append(sc.EquityPoint(ts=label, equity=v))
    return out


def list_strategies() -> list[dict]:
    """Expose every registered strategy with its tunable parameters, so the
    frontend's strategy builder is generated rather than hard-coded."""
    import inspect

    out = []
    for name, cls in sorted(S.REGISTRY.items()):
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        params = [
            {
                "name": p.name,
                "default": None if p.default is inspect.Parameter.empty else p.default,
                "type": type(p.default).__name__
                if p.default is not inspect.Parameter.empty else "float",
            }
            for p in sig.parameters.values()
            if p.name not in ("self", "params", "kwargs", "args")
        ]
        doc = (cls.__doc__ or "").strip().split("\n\n")
        out.append({
            "id": name,
            "name": getattr(cls, "name", name),
            "summary": doc[0].replace("\n", " ").strip() if doc else "",
            "description": "\n\n".join(doc[1:]).strip(),
            "params": params,
        })
    return out


# ----------------------------------------------------------------- paper trade
def _limits(cfg: sc.PaperRunCreate) -> RiskLimits:
    return RiskLimits(
        min_net_ev=cfg.min_net_ev, min_confidence=cfg.min_confidence,
        min_edge=cfg.min_edge, min_liquidity=cfg.min_liquidity,
        max_positions=cfg.max_positions, max_gross_exposure=cfg.max_gross_exposure,
        max_position=cfg.max_position,
    )


def _engine_for(cfg: dict | sc.PaperRunCreate, bankroll: float) -> MispricingEngine:
    """Build the scanner a paper run uses, from either the create request or the
    persisted `model` blob. Same code path both ways, so a resumed run is
    guaranteed to score markets identically to the run that created it."""
    if isinstance(cfg, sc.PaperRunCreate):
        cfg = {
            "frictions": cfg.frictions, "market_anchor": cfg.market_anchor,
            "signals": [s.model_dump() for s in cfg.signals] if cfg.signals else None,
        }
    req = sc.ScanRequest(
        frictions=cfg.get("frictions", "conservative"),
        market_anchor=cfg.get("market_anchor", 0.30),
        signals=[sc.SignalSpec(**s) for s in cfg["signals"]] if cfg.get("signals") else None,
        bankroll=bankroll,
    )
    return build_engine(req)


def create_paper_run(cfg: sc.PaperRunCreate, store: Store) -> str:
    trader = PaperTrader(
        name=cfg.name, limits=_limits(cfg), initial_capital=cfg.capital,
        engine=_engine_for(cfg, cfg.capital), store=store,
        model={
            "frictions": cfg.frictions, "market_anchor": cfg.market_anchor,
            "signals": [s.model_dump() for s in cfg.signals] if cfg.signals else None,
        },
    )
    return trader.run_id


def resume_trader(run_id: str, store: Store) -> PaperTrader:
    row = store.get_paper_run(run_id)
    if row is None:
        raise KeyError(run_id)
    import json as _json

    try:
        model = (_json.loads(row["params"] or "{}") or {}).get("model") or {}
    except (TypeError, ValueError):
        model = {}
    return PaperTrader.resume(
        run_id, store, engine=_engine_for(model, row["initial_capital"])
    )


def paper_step(run_id: str, req: sc.PaperStepRequest, store: Store) -> sc.StepResponse:
    trader = resume_trader(run_id, store)
    # In synthetic mode a slice of the universe arrives already resolved, so
    # each cycle exercises settlement as well as entry and exit.
    markets = (
        synthetic_markets(
            req.limit, seed=int(time.time()) % 10_000, steps=30,
            resolve_fraction=0.25,
        )
        if req.synthetic
        else fetch_markets(req.venues, req.limit, req.min_volume)
    )
    report = trader.step(markets)
    return sc.StepResponse(
        ts=report.ts.isoformat(), scanned=report.scanned,
        opened=[_fill_out(f) for f in report.opened],
        closed=[_fill_out(f) for f in report.closed],
        settled=[_fill_out(f) for f in report.settled],
        rejected=report.rejected[:50],
        summary=paper_summary(run_id, store, markets),
    )


def _fill_out(f) -> sc.FillOut:
    return sc.FillOut(
        fill_id=f.fill_id, key=f.key, title=f.title, ts=f.ts.isoformat(),
        action=f.action, side=f.side, price=f.price, contracts=f.contracts,
        notional=f.notional, fee=f.fee, realized_pnl=f.realized_pnl,
        edge=_clean(f.edge), reason=f.reason,
    )


def paper_summary(
    run_id: str, store: Store, markets: list[Market] | None = None
) -> sc.PaperSummary:
    row = store.get_paper_run(run_id)
    if row is None:
        raise KeyError(run_id)
    trader = resume_trader(run_id, store)
    by_key = {m.key: m for m in (markets or [])}
    marks = mark_to_market(trader.portfolio.positions, by_key)
    s = trader.portfolio.summary(marks)
    return sc.PaperSummary(
        run_id=run_id, name=row["name"] or "", status=row["status"],
        initial_capital=row["initial_capital"], cash=s["cash"], equity=s["equity"],
        exposure=s["exposure"], total_return=s["total_return"],
        realized_pnl=s["realized_pnl"], unrealized_pnl=s["unrealized_pnl"],
        fees_paid=s["fees_paid"], open_positions=s["open_positions"],
        closed_trades=s["closed_trades"], win_rate=_clean(s["win_rate"]),
        edge_capture=_clean(s["edge_capture"]), brier=_clean(s["brier"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


def paper_detail(run_id: str, store: Store) -> sc.PaperDetail:
    summary = paper_summary(run_id, store)
    positions = [
        sc.PositionOut(
            key=p["key"], title=p["title"] or "", side=p["side"],
            contracts=p["contracts"], avg_price=p["avg_price"],
            cost_basis=p["cost_basis"], fair_at_entry=_clean(p["fair_at_entry"]),
            edge_at_entry=_clean(p["edge_at_entry"]), opened_at=p["opened_at"],
        )
        for p in store.positions(run_id)
    ]
    fills_df = store.fills_df(run_id)
    fills = [
        sc.FillOut(
            fill_id=r["fill_id"], key=r["key"], title=r.get("title"), ts=r["ts"],
            action=r["action"], side=r["side"], price=r["price"],
            contracts=r["contracts"], notional=r["notional"], fee=r["fee"] or 0,
            realized_pnl=r["realized_pnl"] or 0, edge=_clean(r.get("edge")),
            reason=r.get("reason"),
        )
        for r in fills_df.tail(300).to_dict("records")
    ]
    eq_df = store.equity_df(run_id)
    equity = [
        sc.EquityPoint(ts=r["ts"], equity=r["equity"])
        for r in eq_df.to_dict("records")
    ]
    return sc.PaperDetail(
        summary=summary, positions=positions, fills=fills, equity=equity
    )


def list_paper_runs(store: Store) -> list[sc.PaperSummary]:
    out = []
    for row in store.list_paper_runs():
        try:
            out.append(paper_summary(row["run_id"], store))
        except Exception:  # noqa: BLE001, S112 - a corrupt run must not blank the list
            continue
    return out


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
