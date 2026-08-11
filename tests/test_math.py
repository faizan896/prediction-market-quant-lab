"""Unit tests for the pieces where a silent bug costs real money:
Kelly sizing, the fee formulas, and the metric definitions."""

from __future__ import annotations

import numpy as np
import pytest

from quantlab.backtest.frictions import FrictionModel
from quantlab.backtest.metrics import brier_score, max_drawdown, reliability_curve, sharpe_ratio
from quantlab.backtest.sizing import SizingConfig, kelly_fraction, position_notional


class TestKelly:
    def test_no_edge_means_no_bet(self):
        f = kelly_fraction(np.array([0.5]), np.array([0.5]), np.array([1.0]))
        assert f[0] == pytest.approx(0.0, abs=1e-9)

    def test_known_value(self):
        # p=0.6, b=0.5 -> f* = (0.6-0.5)/(1-0.5) = 0.2
        f = kelly_fraction(np.array([0.6]), np.array([0.5]), np.array([1.0]))
        assert f[0] == pytest.approx(0.2)

    def test_longshot_edge_is_sized_smaller_than_favourite_edge(self):
        """The same 2-point edge is worth far less at 5c than at 95c.
        If this ever inverts, the sizing code will blow the account up."""
        cheap = kelly_fraction(np.array([0.07]), np.array([0.05]), np.array([1.0]))[0]
        dear = kelly_fraction(np.array([0.97]), np.array([0.95]), np.array([1.0]))[0]
        assert cheap < dear
        assert cheap == pytest.approx((0.07 - 0.05) / 0.95)
        assert dear == pytest.approx((0.97 - 0.95) / 0.05)

    def test_no_leg_is_mirror_of_yes_leg(self):
        # Buying NO when fair=0.3 and price=0.4 should equal buying YES
        # when fair=0.7 and price=0.6.
        no_leg = kelly_fraction(np.array([0.3]), np.array([0.4]), np.array([-1.0]))[0]
        yes_leg = kelly_fraction(np.array([0.7]), np.array([0.6]), np.array([1.0]))[0]
        assert no_leg == pytest.approx(yes_leg)

    def test_negative_edge_never_sizes(self):
        f = kelly_fraction(np.array([0.4]), np.array([0.6]), np.array([1.0]))
        assert f[0] == 0.0

    def test_caps_are_respected(self):
        cfg = SizingConfig(kelly_multiplier=1.0, max_fraction=0.05)
        n = position_notional(
            cfg, 100_000.0, np.array([0.99]), np.array([0.50]), np.array([1.0])
        )
        assert n[0] == pytest.approx(5_000.0)  # capped, not 98% of bankroll

    def test_liquidity_cap(self):
        cfg = SizingConfig(kelly_multiplier=1.0, max_fraction=0.5, max_liquidity_share=0.1)
        n = position_notional(
            cfg, 100_000.0, np.array([0.9]), np.array([0.5]), np.array([1.0]),
            liquidity=np.array([20_000.0]),
        )
        assert n[0] == pytest.approx(2_000.0)


class TestFees:
    def test_kalshi_formula(self):
        """Kalshi: fee = ceil(0.07 * C * P * (1-P)) in cents."""
        fm = FrictionModel(fee_model="kalshi", fee_rate=0.07)
        fee = fm.fees(np.array([0.50]), np.array([100.0]))[0]
        # 0.07 * 100 * 0.25 = 1.75
        assert fee == pytest.approx(1.75, abs=0.01)

    def test_kalshi_fee_peaks_at_fifty_cents(self):
        fm = FrictionModel(fee_model="kalshi")
        c = np.array([1000.0] * 3)
        fees = fm.fees(np.array([0.05, 0.50, 0.95]), c)
        assert fees[1] > fees[0] and fees[1] > fees[2]
        assert fees[0] == pytest.approx(fees[2], rel=0.02)  # symmetric

    def test_polymarket_fee_is_symmetric_and_tail_light(self):
        fm = FrictionModel(fee_model="polymarket", fee_rate=0.05)
        fees = fm.fees(np.array([0.10, 0.90]), np.array([100.0, 100.0]))
        assert fees[0] == pytest.approx(fees[1])
        assert fees[0] == pytest.approx(0.05 * 0.10 * 100)

    def test_none_model_is_free(self):
        fm = FrictionModel(fee_model="none")
        assert fm.fees(np.array([0.5]), np.array([1000.0]))[0] == 0.0

    def test_maker_fraction_reduces_fees(self):
        taker = FrictionModel(fee_model="kalshi", maker_fraction=0.0)
        maker = FrictionModel(fee_model="kalshi", maker_fraction=0.5)
        c, p = np.array([1000.0]), np.array([0.5])
        assert maker.fees(p, c)[0] == pytest.approx(taker.fees(p, c)[0] * 0.5)


class TestFillPrices:
    def test_buy_yes_pays_the_ask_sell_hits_the_bid(self):
        fm = FrictionModel(spread_ticks=1.0, tick=0.01)
        mid = np.array([0.50, 0.50])
        side = np.array([1.0, -1.0])
        bid = np.array([0.49, 0.49])
        ask = np.array([0.51, 0.51])
        entry = fm.entry_price(mid, side, bid, ask)
        assert entry[0] == pytest.approx(0.51)  # buying YES lifts the offer
        assert entry[1] == pytest.approx(0.49)  # buying NO hits the YES bid

    def test_exit_is_the_opposite_side(self):
        fm = FrictionModel()
        mid, bid, ask = np.array([0.5]), np.array([0.48]), np.array([0.52])
        assert fm.entry_price(mid, np.array([1.0]), bid, ask)[0] == pytest.approx(0.52)
        assert fm.exit_price(mid, np.array([1.0]), bid, ask)[0] == pytest.approx(0.48)

    def test_round_trip_costs_are_positive(self):
        fm = FrictionModel(slippage_bps=20, impact_coef=0.3, fee_model="kalshi")
        cost = fm.round_trip_cost(
            np.array([0.5]), np.array([0.6]), np.array([1000.0]), np.array([50_000.0])
        )
        assert cost[0] > 0


class TestMetrics:
    def test_max_drawdown(self):
        eq = np.array([100.0, 120.0, 90.0, 110.0])
        assert max_drawdown(eq) == pytest.approx((120 - 90) / 120)

    def test_max_drawdown_monotonic_is_zero(self):
        assert max_drawdown(np.array([100.0, 101.0, 102.0])) == 0.0

    def test_sharpe_zero_variance(self):
        assert sharpe_ratio(np.array([0.01] * 10)) == 0.0

    def test_brier_bounds(self):
        perfect = brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        worst = brier_score(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
        assert perfect == pytest.approx(0.0)
        assert worst == pytest.approx(1.0)

    def test_brier_of_base_rate_on_coinflips(self):
        rng = np.random.default_rng(0)
        y = (rng.random(5000) < 0.5).astype(float)
        assert brier_score(np.full(5000, 0.5), y) == pytest.approx(0.25, abs=0.01)

    def test_reliability_curve_detects_overconfidence(self):
        # Claim 90% but only resolve YES half the time.
        prob = np.full(200, 0.9)
        outcome = np.array([1.0, 0.0] * 100)
        curve = reliability_curve(prob, outcome, bins=10)
        assert len(curve) == 1
        assert curve[0]["gap"] == pytest.approx(-0.4, abs=0.01)
