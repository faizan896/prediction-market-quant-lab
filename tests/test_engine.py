"""Engine correctness tests.

The two that matter most:

  test_no_lookahead        — a strategy that peeks at the resolution column must
                             NOT be able to profit, because the engine lags
                             signals. If this ever fails, every backtest number
                             the platform produces is fiction.

  test_oracle_makes_money  — a strategy given genuine advance knowledge *must*
                             profit. Together these bracket the engine: it is
                             neither leaking the future nor destroying real edge.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import PRESETS, BacktestConfig, run_backtest
from quantlab.backtest.engine import _apply_hysteresis, _segments
from quantlab.data.synthetic import SyntheticClient
from quantlab.strategies.base import Strategy


def build_panel(
    n_markets: int = 40, n_steps: int = 120, seed: int = 3, spread: float = 0.02
) -> pd.DataFrame:
    """Note: the panel carries real bid/ask quotes, so PRESETS['frictionless']
    removes fees and slippage but NOT the quoted spread — you still lift the
    offer. Pass spread=0.0 when a test needs genuinely costless fills."""
    gen = SyntheticClient(seed=seed)
    markets, points = gen.generate(n_markets=n_markets, n_steps=n_steps, spread=spread)
    res = {m.market_id: m.resolution for m in markets}
    close = {m.market_id: m.close_time for m in markets}
    liq = {m.market_id: m.liquidity for m in markets}
    return pd.DataFrame({
        "key": [f"synthetic:{p.market_id}" for p in points],
        "market_id": [p.market_id for p in points],
        "ts": [p.ts for p in points],
        "price": [p.price for p in points],
        "volume": [p.volume for p in points],
        "yes_bid": [p.yes_bid for p in points],
        "yes_ask": [p.yes_ask for p in points],
        "resolution": [res[p.market_id] for p in points],
        "close_time": [close[p.market_id] for p in points],
        "liquidity": [liq[p.market_id] for p in points],
    })


class CheatingStrategy(Strategy):
    """Reads the resolution column directly — knows every answer in advance."""

    name = "Cheater"

    def generate(self, panel):
        signal = pd.Series(np.where(panel["resolution"] > 0.5, 1.0, -1.0), index=panel.index)
        return self.frame(signal, panel["resolution"].astype(float).clip(0.01, 0.99))


class NextBarOracle(Strategy):
    """Trades on the *next* bar's price move.

    With signal_lag=0 this is a perfect one-step oracle and should be
    absurdly profitable. With signal_lag=1 the engine shifts it forward by one
    bar, degrading it into ordinary momentum. The gap between those two runs is
    a direct measurement of how much lookahead the engine is leaking.
    """

    name = "Next-Bar Oracle"

    def generate(self, panel):
        nxt = panel.groupby("key", sort=False)["price"].shift(-1)
        signal = np.sign(nxt - panel["price"]).fillna(0.0)
        return self.frame(signal, panel["price"])


class ConstantLongStrategy(Strategy):
    name = "Always Long"

    def generate(self, panel):
        return self.frame(pd.Series(1.0, index=panel.index), panel["price"])


class NeverTradeStrategy(Strategy):
    name = "Flat"

    def generate(self, panel):
        return self.frame(pd.Series(0.0, index=panel.index), panel["price"])


# --------------------------------------------------------------------------- #
class TestPositionLogic:
    def test_hysteresis_holds_through_noise(self):
        sig = np.array([0.0, 0.9, 0.3, 0.25, 0.9, 0.0])
        starts = np.array([True, False, False, False, False, False])
        pos = _apply_hysteresis(sig, starts, entry_threshold=0.5, exit_threshold=0.2)
        # Opens at 0.9, holds through 0.3 and 0.25 (both > exit 0.2), closes at 0.0
        assert pos.tolist() == [0, 1, 1, 1, 1, 0]

    def test_single_threshold_would_churn_but_hysteresis_does_not(self):
        sig = np.array([0.6, 0.4, 0.6, 0.4])
        starts = np.array([True, False, False, False])
        pos = _apply_hysteresis(sig, starts, 0.5, 0.2)
        assert pos.tolist() == [1, 1, 1, 1]  # one trade, not four

    def test_positions_reset_at_market_boundaries(self):
        sig = np.array([0.9, 0.9, 0.0, 0.0])
        starts = np.array([True, False, True, False])
        pos = _apply_hysteresis(sig, starts, 0.5, 0.2)
        assert pos.tolist() == [1, 1, 0, 0]

    def test_flip_from_long_to_short(self):
        sig = np.array([0.9, -0.9])
        starts = np.array([True, False])
        pos = _apply_hysteresis(sig, starts, 0.5, 0.2)
        assert pos.tolist() == [1, -1]

    def test_segments_do_not_span_markets(self):
        panel = pd.DataFrame({"key": ["a", "a", "b", "b"]})
        pos = np.array([1, 1, 1, 1], dtype=np.int8)
        segs = _segments(panel, pos)
        assert len(segs) == 2
        assert segs["key"].tolist() == ["a", "b"]


class TestNoLookahead:
    def test_lag_destroys_next_bar_lookahead(self):
        """THE test. A perfect one-step oracle must collapse to ordinary
        performance once the engine applies its lag. If these two numbers are
        close, the engine is leaking the future and every result is fiction."""
        from quantlab.backtest.sizing import SizingConfig

        # Zero spread: otherwise the oracle's one-bar edge is smaller than the
        # cost of crossing, and both runs look mediocre for the wrong reason.
        panel = build_panel(n_markets=40, spread=0.0)
        # Flat sizing: this strategy reports fair == price, so Kelly would size
        # every bet to zero and we would be comparing two empty backtests.
        def cfg(lag: int) -> BacktestConfig:
            return BacktestConfig(
                frictions=PRESETS["frictionless"], signal_lag=lag,
                sizing=SizingConfig(method="fixed_fraction", fixed_fraction=0.02),
            )

        leaky = run_backtest(panel, NextBarOracle(), cfg(0))
        honest = run_backtest(panel, NextBarOracle(), cfg(1))
        # Without the lag the oracle is exactly that: it never loses.
        assert leaky.metrics.win_rate == pytest.approx(1.0), (
            "unlagged oracle should win every trade; if it doesn't, the "
            "segment/exit indexing is wrong"
        )
        # With the lag it degrades to a coin flip on a martingale.
        assert honest.metrics.win_rate < 0.75, (
            f"lag is not working: lagged win rate {honest.metrics.win_rate:.3f} "
            f"vs unlagged {leaky.metrics.win_rate:.3f}"
        )
        assert honest.metrics.roi < leaky.metrics.roi / 3

    def test_knowing_the_outcome_still_pays_the_market_price(self):
        """A strategy handed the resolution is profitable — real edge survives
        the engine — but it is not risk-free, because it must trade at prices
        that already reflect most of the information."""
        panel = build_panel()
        cfg = BacktestConfig(frictions=PRESETS["frictionless"], signal_lag=1)
        res = run_backtest(panel, CheatingStrategy(), cfg)
        assert res.metrics.n_trades > 0
        assert res.metrics.roi > 0.05
        assert res.metrics.roi < 5.0

    def test_signal_lag_actually_shifts(self):
        """A one-bar spike must produce a trade one bar later, never on the
        bar itself."""
        ts = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(6)]
        panel = pd.DataFrame({
            "key": ["m"] * 6,
            "ts": ts,
            "price": [0.50, 0.50, 0.90, 0.50, 0.50, 0.50],
            "yes_bid": [0.49] * 6,
            "yes_ask": [0.51] * 6,
            "resolution": [np.nan] * 6,
            "liquidity": [1e6] * 6,
        })

        class SpikeOnBarTwo(Strategy):
            name = "Spike"

            def generate(self, p):
                s = pd.Series([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], index=p.index)
                return self.frame(s, p["price"])

        cfg = BacktestConfig(frictions=PRESETS["frictionless"], signal_lag=1,
                             settle_at_resolution=False)
        res = run_backtest(panel, SpikeOnBarTwo(), cfg)
        assert len(res.trades) == 1
        # Signal computed on bar 2 (price 0.90) fills on bar 3 (price 0.50).
        assert res.trades.iloc[0]["entry_price"] == pytest.approx(0.50, abs=0.02)

    def test_flat_strategy_produces_no_trades(self):
        panel = build_panel(n_markets=10)
        res = run_backtest(panel, NeverTradeStrategy(), BacktestConfig())
        assert res.metrics.n_trades == 0
        assert res.metrics.final_equity == pytest.approx(100_000.0)


class TestPnLAccounting:
    def test_yes_leg_pnl_is_exact(self):
        ts = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(4)]
        panel = pd.DataFrame({
            "key": ["m"] * 4,
            "ts": ts,
            "price": [0.40, 0.40, 0.60, 0.60],
            # Zero spread so the arithmetic is exactly checkable.
            "yes_bid": [0.40, 0.40, 0.60, 0.60],
            "yes_ask": [0.40, 0.40, 0.60, 0.60],
            "resolution": [np.nan] * 4,
            "liquidity": [1e9] * 4,
        })

        class LongOnce(Strategy):
            name = "LongOnce"

            def generate(self, p):
                return self.frame(pd.Series([1.0, 1.0, 0.0, 0.0], index=p.index), p["price"])

        cfg = BacktestConfig(
            frictions=PRESETS["frictionless"], signal_lag=1,
            settle_at_resolution=False,
            sizing=__import__("quantlab.backtest.sizing", fromlist=["SizingConfig"]).SizingConfig(
                method="fixed_notional", fixed_notional=1_000.0, max_fraction=1.0
            ),
        )
        res = run_backtest(panel, LongOnce(), cfg)
        tr = res.trades.iloc[0]
        # Entered at 0.40 with $1000 -> 2500 contracts. Exit at 0.60.
        assert tr["contracts"] == pytest.approx(2500, rel=0.01)
        assert tr["pnl"] == pytest.approx(2500 * 0.20, rel=0.01)

    def test_no_leg_is_the_mirror_of_the_yes_leg(self):
        """Buying NO into a falling market must earn what buying YES into a
        symmetric rising market earns."""
        base_ts = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(4)]

        def make(prices):
            return pd.DataFrame({
                "key": ["m"] * 4, "ts": base_ts, "price": prices,
                "yes_bid": prices, "yes_ask": prices,
                "resolution": [np.nan] * 4, "liquidity": [1e9] * 4,
            })

        class Fixed(Strategy):
            name = "Fixed"

            def __init__(self, direction):
                super().__init__(direction=direction)

            def generate(self, p):
                s = pd.Series([self.direction, self.direction, 0.0, 0.0], index=p.index)
                return self.frame(s, p["price"])

        from quantlab.backtest.sizing import SizingConfig
        cfg = BacktestConfig(
            frictions=PRESETS["frictionless"], signal_lag=1, settle_at_resolution=False,
            sizing=SizingConfig(method="fixed_notional", fixed_notional=1_000.0, max_fraction=1.0),
        )
        up = run_backtest(make([0.40, 0.40, 0.60, 0.60]), Fixed(1.0), cfg)
        down = run_backtest(make([0.60, 0.60, 0.40, 0.40]), Fixed(-1.0), cfg)
        assert up.trades.iloc[0]["pnl"] == pytest.approx(down.trades.iloc[0]["pnl"], rel=0.01)

    def test_costs_match_the_paper_broker(self):
        """YES at 10c and NO at a 90c YES price deploy the same cash, so they
        pay the same slippage. A position held to resolution pays none on exit."""
        from quantlab.backtest.frictions import FrictionModel
        from quantlab.backtest.sizing import SizingConfig
        base_ts = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(4)]

        def make(price, resolution):
            return pd.DataFrame({
                "key": ["m"] * 4, "ts": base_ts, "price": [price] * 4,
                "yes_bid": [price] * 4, "yes_ask": [price] * 4,
                "resolution": [resolution] * 4, "liquidity": [50_000.0] * 4,
            })

        class Hold(Strategy):
            name = "Hold"

            def __init__(self, direction):
                super().__init__(direction=direction)

            def generate(self, p):
                return self.frame(pd.Series(self.direction, index=p.index), p["price"])

        def trade(price, direction, resolution, settle):
            cfg = BacktestConfig(
                frictions=FrictionModel(spread_ticks=0.0, slippage_bps=20.0, impact_coef=0.15),
                signal_lag=0, settle_at_resolution=settle, min_price=0.0, max_price=1.0,
                sizing=SizingConfig(method="fixed_notional", fixed_notional=1_000.0,
                                    max_fraction=1.0, max_liquidity_share=1.0),
            )
            return run_backtest(make(price, resolution), Hold(direction), cfg).trades.iloc[0]

        yes = trade(0.10, 1.0, np.nan, False)
        no = trade(0.90, -1.0, np.nan, False)
        assert no["slippage"] == pytest.approx(yes["slippage"])

        entry_only = yes["slippage"] / 2
        assert trade(0.10, 1.0, 1.0, True)["slippage"] == pytest.approx(entry_only)
        assert trade(0.90, -1.0, 0.0, True)["slippage"] == pytest.approx(entry_only)

    def test_edge_capture_compares_like_with_like(self):
        """A 2c edge on a 10c contract is a 20% expected return. Realizing
        exactly that is a capture of 1.0, on either leg."""
        from quantlab.backtest import metrics as M
        trades = pd.DataFrame({
            "side": ["YES", "NO"], "entry_price": [0.10, 0.90], "edge": [0.02, 0.02],
            "pnl": [200.0, 200.0], "notional": [1_000.0, 1_000.0], "ret": [0.20, 0.20],
        })
        m = M.compute(trades, pd.Series([100_200.0, 100_400.0]), 100_000.0)
        assert m.edge_capture == pytest.approx(1.0)

    def test_settlement_uses_the_resolution_value(self):
        # min/max price widened so positions aren't force-closed by the
        # tradability filter as the synthetic price converges to 0 or 1.
        panel = build_panel(n_markets=5, n_steps=30)
        cfg = BacktestConfig(
            frictions=PRESETS["frictionless"], settle_at_resolution=True,
            min_price=0.0, max_price=1.0,
        )
        res = run_backtest(panel, ConstantLongStrategy(), cfg)
        settled = res.trades[res.trades["settled"]]
        assert len(settled) > 0
        assert set(settled["exit_price"].unique()).issubset({0.0, 1.0})

    def test_frictions_reduce_pnl(self):
        panel = build_panel(n_markets=30)
        free = run_backtest(panel, ConstantLongStrategy(),
                            BacktestConfig(frictions=PRESETS["frictionless"]))
        costly = run_backtest(panel, ConstantLongStrategy(),
                              BacktestConfig(frictions=PRESETS["conservative"]))
        assert costly.metrics.total_pnl < free.metrics.total_pnl
        assert costly.metrics.total_fees + costly.metrics.total_slippage > 0


class TestSmoke:
    def test_default_suite_runs_end_to_end(self):
        from quantlab.backtest import compare
        from quantlab.strategies import default_suite

        panel = build_panel(n_markets=30, n_steps=90)
        board, _results = compare(panel, default_suite(), BacktestConfig())
        assert len(board) == len(default_suite())
        assert board["win_rate"].between(0, 1).all()
        assert not board["roi"].isna().any()

    def test_walk_forward_splits(self):
        from quantlab.backtest import walk_forward
        from quantlab.strategies import Momentum

        panel = build_panel(n_markets=30, n_steps=120)
        df = walk_forward(panel, Momentum, n_folds=3)
        assert len(df) >= 2
        assert df["fold"].nunique() >= 2

    def test_empty_panel_is_handled(self):
        panel = build_panel(n_markets=3, n_steps=25)
        res = run_backtest(panel, NeverTradeStrategy(), BacktestConfig())
        assert res.trades.empty
        assert res.metrics.n_trades == 0
