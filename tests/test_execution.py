"""Paper trading tests.

Accounting bugs here are the expensive kind: they don't crash, they just quietly
report a profit that isn't there. Every conservation law gets an explicit test.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantlab.backtest.frictions import PRESETS
from quantlab.data.store import Store
from quantlab.execution import (
    LiveBroker,
    OrderIntent,
    PaperBroker,
    PaperTrader,
    Portfolio,
    RiskLimits,
)
from quantlab.execution.broker import RejectedOrder
from quantlab.mispricing import ExternalProbability, MarketPrior, MispricingEngine
from quantlab.types import Market, Venue


def mk(key_id: str = "m1", price: float = 0.50, **kw) -> Market:
    half = kw.pop("half_spread", 0.005)
    return Market(
        venue=Venue.POLYMARKET, market_id=key_id, event_id="e1",
        title=kw.pop("title", f"Market {key_id}"),
        yes_bid=round(price - half, 6), yes_ask=round(price + half, 6),
        last_price=price, volume=1_000_000, volume_24h=100_000,
        liquidity=kw.pop("liquidity", 500_000),
        close_time=datetime.now(timezone.utc) + timedelta(days=30),
        **kw,
    )


FREE = PRESETS["frictionless"]


class TestPaperBroker:
    def test_buying_yes_lifts_the_ask(self):
        b = PaperBroker(frictions=FREE)
        fill = b.execute(OrderIntent("polymarket:m1", "YES", notional=1_000), mk())
        assert fill.price == pytest.approx(0.505)
        assert fill.contracts == pytest.approx(1_000 / 0.505, rel=1e-6)

    def test_buying_no_hits_the_bid_and_costs_one_minus_p(self):
        b = PaperBroker(frictions=FREE)
        fill = b.execute(OrderIntent("polymarket:m1", "NO", notional=1_000), mk())
        assert fill.price == pytest.approx(0.495)
        # $1000 of NO at a YES price of 0.495 costs 0.505 per contract
        assert fill.contracts == pytest.approx(1_000 / 0.505, rel=1e-6)

    def test_closing_a_yes_position_hits_the_bid(self):
        b = PaperBroker(frictions=FREE)
        fill = b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=100, action="close"), mk()
        )
        assert fill.price == pytest.approx(0.495)

    def test_size_is_capped_by_resting_liquidity(self):
        b = PaperBroker(frictions=FREE, max_liquidity_share=0.10)
        fill = b.execute(
            OrderIntent("polymarket:m1", "YES", notional=100_000),
            mk(liquidity=20_000),
        )
        assert fill.notional == pytest.approx(2_000, rel=0.01)

    def test_dust_orders_are_rejected(self):
        b = PaperBroker(frictions=FREE, min_notional=10)
        with pytest.raises(RejectedOrder):
            b.execute(OrderIntent("polymarket:m1", "YES", notional=1.0), mk())

    def test_settling_an_unresolved_market_is_rejected(self):
        b = PaperBroker(frictions=FREE)
        with pytest.raises(RejectedOrder):
            b.execute(
                OrderIntent("polymarket:m1", "YES", contracts=10, action="settle"), mk()
            )

    def test_settlement_fills_at_the_outcome_with_no_fee(self):
        b = PaperBroker(frictions=PRESETS["kalshi"])
        fill = b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=100, action="settle"),
            mk(resolution=1.0, resolved=True),
        )
        assert fill.price == 1.0
        assert fill.fee == 0.0

    def test_fees_are_charged_under_a_real_schedule(self):
        b = PaperBroker(frictions=PRESETS["kalshi"])
        fill = b.execute(OrderIntent("polymarket:m1", "YES", notional=1_000), mk())
        assert fill.fee > 0


class TestPortfolio:
    def test_cash_conservation_on_a_round_trip(self):
        """Open then close at the same price: cash back to start minus costs."""
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        market = mk(half_spread=0.0)  # zero spread so the only loss is fees

        open_fill = b.execute(OrderIntent("polymarket:m1", "YES", notional=1_000), market)
        p.apply(open_fill)
        assert p.cash == pytest.approx(9_000)

        pos = p.positions["polymarket:m1"]
        close_fill = b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=pos.contracts, action="close"),
            market,
        )
        p.apply(close_fill)
        assert p.cash == pytest.approx(10_000)
        assert p.realized_pnl == pytest.approx(0.0, abs=1e-6)
        assert not p.positions

    def test_yes_position_profits_when_price_rises(self):
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000), mk("m1", 0.40, half_spread=0)
        ))
        pos = p.positions["polymarket:m1"]
        assert pos.contracts == pytest.approx(2_500)
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=pos.contracts, action="close"),
            mk("m1", 0.60, half_spread=0),
        ))
        assert p.realized_pnl == pytest.approx(2_500 * 0.20)

    def test_no_position_is_the_exact_mirror(self):
        """A NO position into a falling market must earn what a YES position
        earns in the symmetric rising market."""
        b = PaperBroker(frictions=FREE)

        up = Portfolio(initial_capital=10_000)
        up.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000), mk("m1", 0.40, half_spread=0)
        ))
        up.apply(b.execute(
            OrderIntent("polymarket:m1", "YES",
                        contracts=up.positions["polymarket:m1"].contracts, action="close"),
            mk("m1", 0.60, half_spread=0),
        ))

        down = Portfolio(initial_capital=10_000)
        down.apply(b.execute(
            OrderIntent("polymarket:m1", "NO", notional=1_000), mk("m1", 0.60, half_spread=0)
        ))
        down.apply(b.execute(
            OrderIntent("polymarket:m1", "NO",
                        contracts=down.positions["polymarket:m1"].contracts, action="close"),
            mk("m1", 0.40, half_spread=0),
        ))
        assert up.realized_pnl == pytest.approx(down.realized_pnl)

    def test_settlement_pays_one_dollar_per_winning_contract(self):
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000), mk("m1", 0.25, half_spread=0)
        ))
        pos = p.positions["polymarket:m1"]
        assert pos.contracts == pytest.approx(4_000)
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=pos.contracts, action="settle"),
            mk("m1", 0.25, resolution=1.0, resolved=True),
        ))
        assert p.cash == pytest.approx(13_000)  # 9000 + 4000 contracts x $1
        assert p.realized_pnl == pytest.approx(3_000)

    def test_losing_settlement_costs_exactly_the_basis(self):
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000), mk("m1", 0.25, half_spread=0)
        ))
        pos = p.positions["polymarket:m1"]
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=pos.contracts, action="settle"),
            mk("m1", 0.25, resolution=0.0, resolved=True),
        ))
        assert p.cash == pytest.approx(9_000)
        assert p.realized_pnl == pytest.approx(-1_000)

    def test_cannot_spend_more_cash_than_held(self):
        p = Portfolio(initial_capital=100)
        b = PaperBroker(frictions=FREE)
        with pytest.raises(ValueError, match="insufficient cash"):
            p.apply(b.execute(OrderIntent("polymarket:m1", "YES", notional=1_000), mk()))

    def test_cannot_close_a_position_that_is_not_open(self):
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        fill = b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=10, action="close"), mk()
        )
        with pytest.raises(ValueError, match="no open position"):
            p.apply(fill)

    def test_opposite_leg_on_an_open_position_is_refused(self):
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        p.apply(b.execute(OrderIntent("polymarket:m1", "YES", notional=1_000), mk()))
        with pytest.raises(ValueError, match="close it first"):
            p.apply(b.execute(OrderIntent("polymarket:m1", "NO", notional=1_000), mk()))

    def test_adding_to_a_position_averages_the_entry(self):
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000), mk("m1", 0.40, half_spread=0)
        ))
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000), mk("m1", 0.60, half_spread=0)
        ))
        pos = p.positions["polymarket:m1"]
        assert pos.cost_basis == pytest.approx(2_000)
        # 2500 contracts at 0.40 + 1666.7 at 0.60 -> weighted 0.48
        assert pos.avg_price == pytest.approx(0.48, abs=0.005)

    def test_equity_marks_open_positions(self):
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000), mk("m1", 0.50, half_spread=0)
        ))
        assert p.equity({"polymarket:m1": 0.50}) == pytest.approx(10_000)
        assert p.equity({"polymarket:m1": 0.75}) == pytest.approx(10_500)
        assert p.equity({"polymarket:m1": 0.25}) == pytest.approx(9_500)

    def test_summary_reports_edge_capture(self):
        """Enter with a predicted 10-point edge at 0.50 and realize all of it."""
        p = Portfolio(initial_capital=10_000)
        b = PaperBroker(frictions=FREE)
        f = b.execute(
            OrderIntent("polymarket:m1", "YES", notional=1_000,
                        fair_prob=0.60, edge=0.10),
            mk("m1", 0.50, half_spread=0),
        )
        p.apply(f)
        pos = p.positions["polymarket:m1"]
        p.apply(b.execute(
            OrderIntent("polymarket:m1", "YES", contracts=pos.contracts, action="close",
                        fair_prob=0.60, edge=0.10),
            mk("m1", 0.60, half_spread=0),
        ))
        s = p.summary({})
        assert s["closed_trades"] == 1
        assert s["win_rate"] == 1.0
        # predicted return = 0.10 / 0.50 = 20%; realized = 20% -> capture 1.0
        assert s["edge_capture"] == pytest.approx(1.0, abs=0.02)


class TestPaperTrader:
    def _engine(self, markets, views):
        return MispricingEngine(
            signals=[MarketPrior(), ExternalProbability(
                {m.key: v for m, v in zip(markets, views)}, "model")],
            frictions=FREE, bankroll=100_000, market_anchor=0.1,
        )

    def test_takes_a_position_when_the_model_disagrees_enough(self):
        markets = [mk("m1", 0.30)]
        trader = PaperTrader(
            engine=self._engine(markets, [(0.60, 0.95)]),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(min_net_ev=0.02, min_confidence=0.1),
            persist=False,
        )
        rep = trader.step(markets)
        assert len(rep.opened) == 1
        assert rep.opened[0].side == "YES"
        assert trader.portfolio.positions

    def test_refuses_when_the_model_agrees_with_the_market(self):
        markets = [mk("m1", 0.30)]
        trader = PaperTrader(
            engine=self._engine(markets, [(0.30, 0.95)]),
            broker=PaperBroker(frictions=FREE), persist=False,
        )
        rep = trader.step(markets)
        assert not rep.opened
        assert rep.rejected

    def test_settles_and_books_the_pnl(self):
        markets = [mk("m1", 0.30)]
        trader = PaperTrader(
            engine=self._engine(markets, [(0.60, 0.95)]),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(min_net_ev=0.02, min_confidence=0.1),
            persist=False,
        )
        trader.step(markets)
        assert trader.portfolio.positions

        resolved = [mk("m1", 0.30, resolution=1.0, resolved=True)]
        rep = trader.step(resolved)
        assert len(rep.settled) == 1
        assert not trader.portfolio.positions
        assert trader.portfolio.realized_pnl > 0

    def test_exits_when_the_edge_disappears(self):
        markets = [mk("m1", 0.30)]
        trader = PaperTrader(
            engine=self._engine(markets, [(0.60, 0.95)]),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(min_net_ev=0.02, min_confidence=0.1),
            persist=False,
        )
        trader.step(markets)
        assert trader.portfolio.positions

        # Price catches up to the model's view: nothing left to collect.
        caught_up = [mk("m1", 0.60)]
        trader.engine = self._engine(caught_up, [(0.60, 0.95)])
        rep = trader.step(caught_up)
        assert len(rep.closed) == 1
        assert not trader.portfolio.positions

    def test_gross_exposure_limit_binds(self):
        markets = [mk(f"m{i}", 0.30) for i in range(20)]
        trader = PaperTrader(
            engine=self._engine(markets, [(0.60, 0.95)] * 20),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(
                max_gross_exposure=0.10, max_position=0.05,
                min_net_ev=0.02, min_confidence=0.1, max_new_per_step=20,
            ),
            initial_capital=100_000, persist=False,
        )
        trader.step(markets)
        assert trader.portfolio.deployed <= 100_000 * 0.10 + 1

    def test_position_count_limit_binds(self):
        markets = [mk(f"m{i}", 0.30) for i in range(10)]
        trader = PaperTrader(
            engine=self._engine(markets, [(0.60, 0.95)] * 10),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(
                max_positions=3, min_net_ev=0.02, min_confidence=0.1,
                max_new_per_step=20,
            ),
            persist=False,
        )
        trader.step(markets)
        assert len(trader.portfolio.positions) <= 3

    def test_illiquid_markets_are_screened_out(self):
        markets = [mk("m1", 0.30, liquidity=100)]
        trader = PaperTrader(
            engine=self._engine(markets, [(0.60, 0.95)]),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(min_liquidity=5_000, min_net_ev=0.02, min_confidence=0.1),
            persist=False,
        )
        rep = trader.step(markets)
        assert not rep.opened
        assert any("liquidity" in r for r in rep.rejected)

    def test_state_persists_and_resumes(self, tmp_path):
        store = Store(path=tmp_path / "p.db")
        markets = [mk("m1", 0.30)]
        trader = PaperTrader(
            run_id="run1", engine=self._engine(markets, [(0.60, 0.95)]),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(min_net_ev=0.02, min_confidence=0.1),
            store=store, persist=True,
        )
        trader.step(markets)
        cash_before = trader.portfolio.cash
        assert len(store.fills_df("run1")) == 1
        assert len(store.positions("run1")) == 1

        resumed = PaperTrader.resume("run1", store)
        assert resumed.portfolio.cash == pytest.approx(cash_before)
        assert set(resumed.portfolio.positions) == set(trader.portfolio.positions)
        store.close()

    def test_resume_restores_risk_limits(self, tmp_path):
        """A resumed run must keep the limits it was created with.

        Regression test: `resume` previously fell back to default RiskLimits,
        so every step after the first silently traded a different configuration
        from the one being evaluated — invalidating the paper record without
        any visible error.
        """
        store = Store(path=tmp_path / "p.db")
        limits = RiskLimits(
            min_net_ev=-1.0, min_confidence=0.0, max_positions=7,
            max_gross_exposure=0.11, min_liquidity=123.0,
        )
        PaperTrader(run_id="run2", limits=limits, store=store, persist=True)

        resumed = PaperTrader.resume("run2", store)
        assert resumed.limits.min_net_ev == pytest.approx(-1.0)
        assert resumed.limits.max_positions == 7
        assert resumed.limits.max_gross_exposure == pytest.approx(0.11)
        assert resumed.limits.min_liquidity == pytest.approx(123.0)
        store.close()

    def test_resume_rebuilds_the_closed_trade_blotter(self, tmp_path):
        """Regression: closed-trade metrics used to reset on every resume.

        Since the trader resumes on each cycle, win rate / Brier / edge capture
        were permanently blank — and edge capture is the whole point of keeping
        a paper record.
        """
        store = Store(path=tmp_path / "p.db")
        markets = [mk("m1", 0.30)]
        trader = PaperTrader(
            run_id="blot", engine=self._engine(markets, [(0.60, 0.95)]),
            broker=PaperBroker(frictions=FREE),
            limits=RiskLimits(min_net_ev=0.02, min_confidence=0.1),
            store=store, persist=True,
        )
        trader.step(markets)
        assert trader.portfolio.positions

        # Resolve it and settle on the next cycle.
        resumed = PaperTrader.resume("blot", store)
        assert resumed.portfolio.summary({})["closed_trades"] == 0
        resumed.step([mk("m1", 0.30, resolution=1.0, resolved=True)])

        again = PaperTrader.resume("blot", store)
        s = again.portfolio.summary({})
        assert s["closed_trades"] == 1
        assert s["win_rate"] == 1.0
        assert s["edge_capture"] == pytest.approx(
            again.portfolio.summary({})["edge_capture"]
        )
        # A settled trade carries its outcome, so Brier is computable.
        assert s["brier"] == s["brier"]  # not NaN
        store.close()

    def test_explicit_limits_override_the_saved_ones(self, tmp_path):
        store = Store(path=tmp_path / "p.db")
        PaperTrader(run_id="run3", limits=RiskLimits(max_positions=7),
                    store=store, persist=True)
        resumed = PaperTrader.resume("run3", store, limits=RiskLimits(max_positions=99))
        assert resumed.limits.max_positions == 99
        store.close()


def test_live_broker_refuses_to_pretend():
    with pytest.raises(NotImplementedError, match="not implemented"):
        LiveBroker()
