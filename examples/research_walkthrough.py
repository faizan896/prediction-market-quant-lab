"""A full research session, start to finish.

Run:  python examples/research_walkthrough.py
No network and no keys needed — it uses the synthetic generator, so it works
offline and produces the same numbers every time.

Swap `synthetic_panel()` for `live_panel()` at the bottom once you've run
`quantlab ingest && quantlab backfill`.
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtest import PRESETS, BacktestConfig, compare, walk_forward
from quantlab.data import Store, SyntheticClient
from quantlab.mispricing import ExternalProbability, MarketPrior, MispricingEngine
from quantlab.strategies import LongshotBias, Momentum, default_suite

pd.set_option("display.width", 160)


def synthetic_panel(n_markets: int = 400, spread: float = 0.01) -> pd.DataFrame:
    """Generate markets carrying a known, planted favourite-longshot bias."""
    gen = SyntheticClient(seed=7)
    markets, points = gen.generate(
        n_markets=n_markets, n_steps=180, spread=spread,
        bias_fn=SyntheticClient.longshot_bias(0.05),
    )
    store = Store(path=":memory:")
    store.upsert_markets(markets)
    store.upsert_prices(points)
    return store.panel()


def live_panel(venue: str = "polymarket") -> pd.DataFrame:
    return Store().panel(venue=venue)


# --------------------------------------------------------------------------- #
def step_1_verify_the_edge_exists(panel: pd.DataFrame) -> None:
    """Before backtesting anything, check the data is miscalibrated at all.

    Bucket every observed price and compare it to how often those contracts
    actually resolved YES. If `actual` tracks `price`, the market is calibrated
    and there is nothing to trade — stop here and find better data.
    """
    print("\n=== 1. calibration of the raw market ===")
    df = panel.dropna(subset=["resolution"]).copy()
    df["bucket"] = pd.cut(df["price"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    table = (
        df.groupby("bucket", observed=True)
        .agg(n=("resolution", "size"), price=("price", "mean"),
             actual=("resolution", "mean"))
        .assign(gap=lambda d: d["actual"] - d["price"])
        .round(3)
    )
    print(table.to_string())
    print("\ngap < 0 means the contract is overpriced -> sell it.")


def step_2_leaderboard(panel: pd.DataFrame) -> None:
    """Same strategies, two cost assumptions. The delta IS the finding."""
    print("\n=== 2. leaderboard: frictionless vs conservative ===")
    for preset in ("frictionless", "conservative"):
        cfg = BacktestConfig(frictions=PRESETS[preset])
        board, _ = compare(panel, default_suite(), cfg)
        print(f"\n--- {preset} ---")
        print(board[["strategy", "trades", "win_rate", "roi", "sharpe", "max_dd"]]
              .round(4).to_string(index=False))
    print("\nStrategies that are profitable frictionless and negative under")
    print("`conservative` do not have an edge. They have a cost problem.")


def step_3_inspect_one_strategy(panel: pd.DataFrame) -> None:
    """Where did the PnL actually come from? Split by whether the position was
    held to settlement — for a calibration edge, only settled trades pay."""
    print("\n=== 3. trade attribution: LongshotBias ===")
    from quantlab.backtest import run_backtest

    res = run_backtest(panel, LongshotBias(),
                       BacktestConfig(frictions=PRESETS["frictionless"]))
    t = res.trades
    if t.empty:
        print("no trades")
        return
    print(t.groupby("settled")
          .agg(n=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()),
               pnl=("pnl", "sum"), avg_hold_h=("hold_hours", "mean"))
          .round(3).to_string())
    m = res.metrics
    print(f"\nROI {m.roi:+.2%} · Sharpe {m.sharpe:.2f} · "
          f"fees ${m.total_fees:,.0f} · slippage ${m.total_slippage:,.0f}")


def step_4_walk_forward(panel: pd.DataFrame) -> None:
    """In-sample results are a hypothesis. This is the test."""
    print("\n=== 4. walk-forward (out-of-sample) ===")
    df = walk_forward(panel, Momentum, n_folds=4,
                      config=BacktestConfig(frictions=PRESETS["conservative"]))
    if df.empty:
        print("not enough data to split")
        return
    print(df[["fold", "trades", "win_rate", "roi", "sharpe"]].round(4).to_string(index=False))
    spread = df["sharpe"].max() - df["sharpe"].min()
    print(f"\nSharpe spread across folds: {spread:.2f}"
          f"{'  (stable)' if spread < 1 else '  (unstable — treat as a curve fit)'}")


def step_5_scan_for_mispricing() -> None:
    """The live terminal, driven here by a stand-in 'model'."""
    print("\n=== 5. mispricing scan ===")
    from datetime import datetime, timedelta, timezone

    from quantlab.types import Market, Venue

    markets = [
        Market(venue=Venue.POLYMARKET, market_id=f"m{i}", event_id="e", title=title,
               yes_bid=px - 0.005, yes_ask=px + 0.005, volume=2_000_000,
               volume_24h=80_000, liquidity=400_000,
               close_time=datetime.now(timezone.utc) + timedelta(days=45))
        for i, (title, px) in enumerate([
            ("Approval rating above 45% on Dec 31", 0.57),
            ("Fed cuts rates in September", 0.31),
            ("Government shutdown before November", 0.12),
        ])
    ]
    # Your model's view. In production this is polls, weather, a fitted model,
    # anything — see quantlab.mispricing.ExternalProbability.
    view = {m.key: (p, c) for m, (p, c) in zip(markets, [(0.64, 0.85), (0.29, 0.6), (0.11, 0.5)])}

    engine = MispricingEngine(signals=[MarketPrior(), ExternalProbability(view, "model")])
    # min_net_ev well below zero so the near-misses are visible too. In
    # production leave it at 0 — you only want trades that clear their costs.
    df = engine.scan(markets, min_edge=0.001, min_confidence=0.0, min_net_ev=-10)
    cols = ["title", "price", "fair_value", "edge", "side", "confidence",
            "gross_ev", "net_ev", "suggested_notional"]
    print(df[cols].round(4).to_string(index=False))
    print("""
Read this carefully, because it is the whole point of the tool.

The model claims a 7-point view on the first market (57c vs 64c). Fair value
moves barely a point, because the market price carries weight 3.0 in the pool
and gets a further pull from `market_anchor` — your model does not get to
overrule a liquid market on its own say-so. To move fair value further you
must raise the signal's weight or its confidence, deliberately.

Then costs. Kalshi's fee is 7% x contracts x p x (1-p), which at mid-book is
~3.5% of notional *per side*. Round-trip that is ~7%, before spread. So a
1-point edge on a 57c contract (gross EV ~2%) is a losing trade, and net_ev
says so.

Notice the cost gradient down the table: net_ev is -4.9% on the 57c contract
but only -1.3% on the 12c one, for the same reason — the p(1-p) fee term
collapses in the tails. The actionable read is not "no edge here", it is:
at mid-book you need ~7+ points of genuine edge to clear fees, so the tails
are where a small edge is actually harvestable.""")


if __name__ == "__main__":
    panel = synthetic_panel()
    print(f"panel: {panel['key'].nunique()} markets · {len(panel):,} bars")
    step_1_verify_the_edge_exists(panel)
    step_2_leaderboard(panel)
    step_3_inspect_one_strategy(panel)
    step_4_walk_forward(panel)
    step_5_scan_for_mispricing()
