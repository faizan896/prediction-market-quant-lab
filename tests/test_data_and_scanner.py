"""Parsing, storage and scanner tests. No network calls — venue responses are
pinned as fixtures captured from the live APIs on 2026-08-10."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantlab.data.kalshi import KalshiClient, _price, _size
from quantlab.data.polymarket import PolymarketClient, _jarr
from quantlab.data.store import Store
from quantlab.mispricing import MispricingEngine
from quantlab.mispricing.signals import (
    CrossVenue,
    ExternalProbability,
    LongshotCorrection,
    MarketPrior,
    OrderBookImbalance,
)
from quantlab.types import BookLevel, Market, OrderBook, PricePoint, Venue

# --- fixtures captured live from gamma-api.polymarket.com -------------------
POLY_RAW = {
    "id": "703258",
    "question": "Will Jesus Christ return before 2027?",
    "conditionId": "0x0b4c",
    "slug": "will-jesus-christ-return-before-2027",
    "outcomes": '["Yes", "No"]',
    "outcomePrices": '["0.0195", "0.9805"]',
    "clobTokenIds": '["69324317355037271422943965141382095011871956039434394956830818206664869608517", "51797157743046504218541616681751597845468055908324407922581755135522797852101"]',
    "volumeNum": 64982786.08,
    "liquidityNum": 763974.68,
    "volume24hr": 12083.32,
    "bestBid": 0.019,
    "bestAsk": 0.02,
    "lastTradePrice": 0.019,
    "startDate": "2025-11-25T18:08:21.296Z",
    "endDate": "2026-12-31T00:00:00Z",
    "closed": False,
    "orderPriceMinTickSize": 0.001,
    "feesEnabled": True,
    "feeSchedule": {"rate": 0.05, "takerOnly": True},
    "negRisk": False,
    "spread": 0.001,
    "events": [{"id": "90178", "ticker": "will-jesus-christ-return-before-2027"}],
}

# --- fixture captured live from api.elections.kalshi.com --------------------
KALSHI_RAW = {
    "ticker": "KXHIGHNY-26AUG10-T89",
    "event_ticker": "KXHIGHNY-26AUG10",
    "title": "Will the **high temp in NYC** be <89° on Aug 10, 2026?",
    "subtitle": "88° or below",
    "yes_bid_dollars": "0.1000",
    "yes_ask_dollars": "0.1100",
    "last_price_dollars": "0.1100",
    "no_bid_dollars": "0.8900",
    "volume_fp": "100.13",
    "volume_24h_fp": "100.13",
    "open_interest_fp": "100.13",
    "liquidity_dollars": "0.0000",
    "open_time": "2026-08-09T14:00:00Z",
    "close_time": "2026-08-11T04:59:00Z",
    "status": "active",
    "result": "",
    "price_level_structure": "linear_cent",
    "strike_type": "less",
    "cap_strike": 89,
    "rules_primary": "If the highest temperature recorded in Central Park...",
    "yes_bid_size_fp": "0.40",
    "yes_ask_size_fp": "8.00",
}

KALSHI_CANDLE = {
    "end_period_ts": 1786320000,
    "volume_fp": "1524.96",
    "price": {"close_dollars": "0.2800", "open_dollars": "0.2300"},
    "yes_bid": {"close_dollars": "0.2800"},
    "yes_ask": {"close_dollars": "0.2900"},
}


class TestPolymarketParsing:
    def test_stringified_json_arrays(self):
        """The single most common Polymarket integration bug."""
        assert _jarr('["Yes", "No"]') == ["Yes", "No"]
        assert _jarr(["Yes", "No"]) == ["Yes", "No"]
        assert _jarr(None) == []
        assert _jarr("not json") == []

    def test_market_normalization(self):
        m = PolymarketClient.__new__(PolymarketClient)._to_market(POLY_RAW)
        assert m.venue == Venue.POLYMARKET
        assert m.market_id.startswith("69324317")  # YES token, index 0
        assert m.title.startswith("Will Jesus")
        assert m.yes_bid == pytest.approx(0.019)
        assert m.yes_ask == pytest.approx(0.02)
        assert m.mid == pytest.approx(0.0195)
        assert m.spread == pytest.approx(0.001)
        assert m.fee_model == "polymarket"
        assert m.min_tick == pytest.approx(0.001)
        assert m.event_id == "90178"

    def test_market_without_tokens_is_dropped(self):
        raw = dict(POLY_RAW, clobTokenIds=None)
        assert PolymarketClient.__new__(PolymarketClient)._to_market(raw) is None


class TestKalshiParsing:
    def test_dollar_string_fields(self):
        assert _price({"yes_bid_dollars": "0.1100"}, "yes_bid") == pytest.approx(0.11)

    def test_legacy_integer_cents_still_work(self):
        assert _price({"yes_bid": 11}, "yes_bid") == pytest.approx(0.11)

    def test_missing_field_is_none(self):
        assert _price({}, "yes_bid") is None

    def test_fixed_point_sizes(self):
        assert _size({"volume_fp": "100.13"}, "volume") == pytest.approx(100.13)
        assert _size({"volume": 42}, "volume") == pytest.approx(42.0)

    def test_market_normalization(self):
        m = KalshiClient.__new__(KalshiClient)._to_market(KALSHI_RAW)
        assert m.venue == Venue.KALSHI
        assert m.market_id == "KXHIGHNY-26AUG10-T89"
        assert m.yes_bid == pytest.approx(0.10)
        assert m.yes_ask == pytest.approx(0.11)
        assert m.mid == pytest.approx(0.105)
        assert m.volume == pytest.approx(100.13)
        assert m.raw["series_ticker"] == "KXHIGHNY"
        assert m.fee_model == "kalshi"
        assert not m.resolved

    def test_resolved_market_carries_outcome(self):
        m = KalshiClient.__new__(KalshiClient)._to_market(dict(KALSHI_RAW, result="yes"))
        assert m.resolved and m.resolution == 1.0
        m = KalshiClient.__new__(KalshiClient)._to_market(dict(KALSHI_RAW, result="no"))
        assert m.resolved and m.resolution == 0.0

    def test_candlestick_price_extraction(self):
        c = KalshiClient.__new__(KalshiClient)
        assert c._candle_price(KALSHI_CANDLE) == pytest.approx(0.28)

    def test_candlestick_falls_back_to_midpoint_on_no_trade_bars(self):
        c = KalshiClient.__new__(KalshiClient)
        candle = {"yes_bid": {"close_dollars": "0.2000"}, "yes_ask": {"close_dollars": "0.3000"}}
        assert c._candle_price(candle) == pytest.approx(0.25)

    def test_no_side_is_converted_to_yes_asks(self):
        """A NO bid at 0.89 is a YES ask at 0.11 — get this wrong and every
        cross-venue comparison is inverted."""
        c = KalshiClient.__new__(KalshiClient)
        assert c._level([11, 100.0]) == (0.11, 100.0)
        assert c._level([0.11, 100.0]) == (0.11, 100.0)


class TestOrderBook:
    def test_microprice_leans_toward_the_thin_side(self):
        ob = OrderBook(
            Venue.POLYMARKET, "x", datetime.now(timezone.utc),
            bids=[BookLevel(0.49, 10_000)], asks=[BookLevel(0.51, 100)],
        )
        # Heavy bids, light asks -> price should sit above the 0.50 midpoint.
        assert ob.microprice() > 0.50

    def test_depth_within(self):
        ob = OrderBook(
            Venue.KALSHI, "x", datetime.now(timezone.utc),
            bids=[BookLevel(0.50, 100), BookLevel(0.45, 500)],
            asks=[BookLevel(0.52, 200)],
        )
        bid_sz, ask_sz = ob.depth_within(0.02)
        assert bid_sz == 100  # the 0.45 level is outside 2c
        assert ask_sz == 200


class TestStore:
    def test_roundtrip(self, tmp_path):
        store = Store(path=tmp_path / "t.db")
        now = datetime.now(timezone.utc)
        m = Market(
            venue=Venue.KALSHI, market_id="T1", event_id="E1", title="Test",
            yes_bid=0.4, yes_ask=0.42, volume=1000, close_time=now + timedelta(days=1),
            resolved=True, resolution=1.0,
        )
        assert store.upsert_markets([m]) == 1
        pts = [PricePoint(Venue.KALSHI, "T1", now + timedelta(hours=i), 0.4 + i * 0.001)
               for i in range(30)]
        assert store.upsert_prices(pts) == 30

        panel = store.panel(min_points=10)
        assert len(panel) == 30
        assert panel["key"].iloc[0] == "kalshi:T1"
        assert panel["resolution"].iloc[0] == 1.0
        store.close()

    def test_upsert_is_idempotent(self, tmp_path):
        store = Store(path=tmp_path / "t.db")
        m = Market(venue=Venue.KALSHI, market_id="T1", event_id="E1", title="Test", volume=1)
        store.upsert_markets([m])
        m.volume = 999
        store.upsert_markets([m])
        df = store.markets_df()
        assert len(df) == 1 and df.iloc[0]["volume"] == 999
        store.close()

    def test_min_points_filter(self, tmp_path):
        store = Store(path=tmp_path / "t.db")
        now = datetime.now(timezone.utc)
        store.upsert_markets([Market(Venue.KALSHI, "T1", "E1", "t")])
        store.upsert_prices([PricePoint(Venue.KALSHI, "T1", now + timedelta(hours=i), 0.5)
                             for i in range(5)])
        assert store.panel(min_points=10).empty
        store.close()


def make_market(price: float = 0.5, **kw) -> Market:
    half = kw.pop("half_spread", 0.005)
    return Market(
        venue=Venue.POLYMARKET, market_id="m1", event_id="e1", title="Test market",
        yes_bid=price - half, yes_ask=price + half,
        volume=1_000_000, volume_24h=50_000, liquidity=250_000,
        close_time=datetime.now(timezone.utc) + timedelta(days=30),
        **kw,
    )


class TestSignals:
    def test_market_prior_penalizes_wide_spreads(self):
        tight = MarketPrior().estimate(make_market(0.5, half_spread=0.002))
        wide = MarketPrior().estimate(make_market(0.5, half_spread=0.05))
        assert tight.confidence > wide.confidence

    def test_cross_venue_abstains_without_a_reference(self):
        assert CrossVenue().estimate(make_market(0.5)) is None

    def test_longshot_pushes_cheap_contracts_cheaper(self):
        est = LongshotCorrection(strength=0.3).estimate(make_market(0.10))
        assert est.prob < 0.10

    def test_longshot_pushes_favourites_dearer(self):
        est = LongshotCorrection(strength=0.3).estimate(make_market(0.90))
        assert est.prob > 0.90

    def test_longshot_abstains_at_the_extremes(self):
        assert LongshotCorrection().estimate(make_market(0.001, half_spread=0.0)) is None

    def test_external_signal_plugs_in(self):
        m = make_market(0.57)
        sig = ExternalProbability({m.key: (0.64, 0.8)}, source="polls")
        est = sig.estimate(m)
        assert est.prob == 0.64 and est.source == "polls"

    def test_book_imbalance_abstains_without_a_book(self):
        assert OrderBookImbalance().estimate(make_market(0.5)) is None


class TestMispricingEngine:
    def test_logit_pooling_beats_naive_averaging_on_longshots(self):
        """Averaging 0.02 and 0.20 gives 0.11, which wildly overprices the
        longshot. Logit pooling gives roughly 0.06."""
        eng = MispricingEngine(signals=[], weights={"a": 1.0, "b": 1.0}, market_anchor=0.0)
        from quantlab.types import FairEstimate
        fair, _, _ = eng.combine(0.05, [
            FairEstimate("a", 0.02, 1.0), FairEstimate("b", 0.20, 1.0),
        ])
        assert 0.04 < fair < 0.09
        assert fair < 0.11

    def test_market_anchor_limits_how_far_fair_value_can_move(self):
        from quantlab.types import FairEstimate
        anchored = MispricingEngine(signals=[], market_anchor=0.9)
        loose = MispricingEngine(signals=[], market_anchor=0.1)
        est = [FairEstimate("x", 0.90, 1.0)]
        a, _, _ = anchored.combine(0.50, est)
        l, _, _ = loose.combine(0.50, est)
        assert abs(a - 0.50) < abs(l - 0.50)

    def test_disagreeing_signals_lower_confidence(self):
        from quantlab.types import FairEstimate
        eng = MispricingEngine(signals=[])
        _, agree_conf, _ = eng.combine(0.5, [
            FairEstimate("a", 0.60, 1.0), FairEstimate("b", 0.61, 1.0),
        ])
        _, disagree_conf, _ = eng.combine(0.5, [
            FairEstimate("a", 0.20, 1.0), FairEstimate("b", 0.80, 1.0),
        ])
        assert agree_conf > disagree_conf

    def test_no_signals_returns_the_market_price(self):
        eng = MispricingEngine(signals=[])
        fair, conf, contrib = eng.combine(0.42, [])
        assert fair == 0.42 and conf == 0.0 and contrib == {}

    def test_evaluate_produces_a_full_opportunity_row(self):
        m = make_market(0.57)
        eng = MispricingEngine(signals=[MarketPrior(), ExternalProbability({m.key: (0.70, 0.9)})])
        opp = eng.evaluate(m)
        assert opp is not None
        assert opp.side == "YES"
        assert opp.fair_value > opp.price
        assert 0 <= opp.confidence <= 1
        assert opp.net_ev < opp.gross_ev  # costs must bite
        assert opp.suggested_notional >= 0

    def test_negative_view_flags_the_no_leg(self):
        m = make_market(0.57)
        eng = MispricingEngine(signals=[MarketPrior(), ExternalProbability({m.key: (0.30, 0.9)})])
        opp = eng.evaluate(m)
        assert opp.side == "NO"
        assert opp.fair_value < opp.price

    def test_scan_filters_and_ranks(self):
        markets = [make_market(0.5 + i * 0.01) for i in range(5)]
        for i, m in enumerate(markets):
            m.market_id = f"m{i}"
        table = {m.key: (m.mid + 0.15, 0.9) for m in markets}
        eng = MispricingEngine(signals=[MarketPrior(), ExternalProbability(table)])
        # min_net_ev below zero so the ranking is exercised on near-misses too;
        # at mid-book prices Kalshi-level fees reject almost everything.
        df = eng.scan(markets, min_edge=0.01, min_confidence=0.0, min_net_ev=-10)
        assert not df.empty
        assert (df["edge"] >= 0.01).all()
        assert df["score"].is_monotonic_decreasing

    def test_scan_returns_empty_when_nothing_qualifies(self):
        markets = [make_market(0.5)]
        eng = MispricingEngine(signals=[MarketPrior()])
        assert eng.scan(markets, min_edge=0.5).empty
