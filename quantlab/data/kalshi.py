"""Kalshi client.

Market data on /trade-api/v2 is public — no key needed for markets, order books
or candlesticks. Keys (RSA-PSS request signing) are only required for trading
and portfolio endpoints, which live in quantlab.execution.

Format note: Kalshi has migrated most price fields to decimal-dollar *strings*
(`yes_bid_dollars: "0.1100"`) and sizes to fixed-point strings (`volume_fp`).
Older integer-cent fields (`yes_bid: 11`) still appear on some responses, so
`_price()` accepts either and always returns a probability in [0, 1].
"""

from __future__ import annotations

from typing import Any

from quantlab.config import settings
from quantlab.data.base import HttpMixin, parse_ts, to_float, utcnow
from quantlab.types import BookLevel, Market, OrderBook, PricePoint, Venue


def _price(raw: dict, base: str) -> float | None:
    """Read `<base>_dollars` (string dollars) or `<base>` (integer cents)."""
    v = to_float(raw.get(f"{base}_dollars"))
    if v is not None:
        return v
    cents = to_float(raw.get(base))
    return None if cents is None else cents / 100.0


def _size(raw: dict, base: str) -> float:
    v = to_float(raw.get(f"{base}_fp"))
    if v is not None:
        return v
    return to_float(raw.get(base), 0.0) or 0.0


class KalshiClient(HttpMixin):
    venue = Venue.KALSHI

    def __init__(self) -> None:
        super().__init__()
        self.base_url = settings.kalshi_api

    # ------------------------------------------------------------------ markets
    def list_markets(
        self,
        limit: int = 200,
        status: str = "open",
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        min_volume: float = 0.0,
        skip_multivariate: bool = True,
    ) -> list[Market]:
        """Paginate /markets via the opaque cursor.

        `skip_multivariate` drops the auto-generated KXMVE* parlay combos, which
        are numerous, illiquid, and pure noise for a scanner.
        """
        import time as _time

        deadline = _time.monotonic() + settings.venue_budget_seconds
        out: list[Market] = []
        cursor: str | None = None
        while len(out) < limit:
            # Same guard as Polymarket: a partial page beats a 504.
            if _time.monotonic() > deadline:
                break
            res = self.get(
                "/markets",
                limit=min(1000, max(1, limit - len(out))),
                cursor=cursor,
                status=status,
                series_ticker=series_ticker,
                event_ticker=event_ticker,
            )
            batch = res.get("markets", [])
            if not batch:
                break
            for raw in batch:
                if skip_multivariate and str(raw.get("ticker", "")).startswith("KXMVE"):
                    continue
                m = self._to_market(raw)
                if m and m.volume >= min_volume:
                    out.append(m)
            cursor = res.get("cursor")
            if not cursor:
                break
        return out[:limit]

    def get_market(self, ticker: str) -> Market | None:
        res = self.get(f"/markets/{ticker}")
        raw = res.get("market") if isinstance(res, dict) else None
        return self._to_market(raw) if raw else None

    def list_events(self, limit: int = 200, status: str = "open") -> list[dict]:
        return self.get("/events", limit=limit, status=status).get("events", [])

    def _to_market(self, raw: dict) -> Market | None:
        ticker = raw.get("ticker")
        if not ticker:
            return None
        result = (raw.get("result") or "").lower()
        resolution = {"yes": 1.0, "no": 0.0}.get(result)
        return Market(
            venue=Venue.KALSHI,
            market_id=str(ticker),
            event_id=str(raw.get("event_ticker", "")),
            title=raw.get("title", ""),
            slug=str(ticker).lower(),
            yes_bid=_price(raw, "yes_bid"),
            yes_ask=_price(raw, "yes_ask"),
            last_price=_price(raw, "last_price"),
            volume=_size(raw, "volume"),
            volume_24h=_size(raw, "volume_24h"),
            liquidity=to_float(raw.get("liquidity_dollars"), 0.0) or 0.0,
            open_interest=_size(raw, "open_interest"),
            open_time=parse_ts(raw.get("open_time")),
            close_time=parse_ts(raw.get("close_time")),
            status=raw.get("status", "unknown"),
            resolved=resolution is not None,
            resolution=resolution,
            min_tick=0.001 if raw.get("price_level_structure") == "deci_cent" else 0.01,
            fee_model="kalshi",
            raw={
                "series_ticker": str(ticker).split("-")[0],
                "subtitle": raw.get("subtitle") or raw.get("yes_sub_title", ""),
                "strike_type": raw.get("strike_type"),
                "floor_strike": raw.get("floor_strike"),
                "cap_strike": raw.get("cap_strike"),
                "rules": raw.get("rules_primary", ""),
                "yes_bid_size": _size(raw, "yes_bid_size"),
                "yes_ask_size": _size(raw, "yes_ask_size"),
            },
        )

    # ------------------------------------------------------------------ history
    def price_history(
        self,
        market_id: str,
        series_ticker: str | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
        period_interval: int = 60,
    ) -> list[PricePoint]:
        """Candlesticks. `period_interval` is minutes per bar (1, 60, or 1440)."""
        series = series_ticker or str(market_id).split("-")[0]
        import time

        end = end_ts or int(time.time())
        start = start_ts or end - 90 * 86_400
        res = self.get(
            f"/series/{series}/markets/{market_id}/candlesticks",
            start_ts=start,
            end_ts=end,
            period_interval=period_interval,
        )
        points: list[PricePoint] = []
        for c in res.get("candlesticks", []):
            price = self._candle_price(c)
            if price is None:
                continue
            points.append(
                PricePoint(
                    venue=Venue.KALSHI,
                    market_id=market_id,
                    ts=parse_ts(c.get("end_period_ts")),
                    price=price,
                    volume=_size(c, "volume"),
                    yes_bid=self._nested(c, "yes_bid", "close"),
                    yes_ask=self._nested(c, "yes_ask", "close"),
                )
            )
        return points

    @staticmethod
    def _nested(candle: dict, group: str, field: str) -> float | None:
        g = candle.get(group) or {}
        v = to_float(g.get(f"{field}_dollars"))
        if v is not None:
            return v
        cents = to_float(g.get(field))
        return None if cents is None else cents / 100.0

    def _candle_price(self, candle: dict) -> float | None:
        """Prefer traded close; fall back to the bid/ask midpoint on no-trade bars."""
        p = self._nested(candle, "price", "close")
        if p:
            return p
        bid = self._nested(candle, "yes_bid", "close")
        ask = self._nested(candle, "yes_ask", "close")
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return bid if bid is not None else ask

    # --------------------------------------------------------------------- book
    def order_book(self, market_id: str) -> OrderBook | None:
        res = self.get(f"/markets/{market_id}/orderbook", depth=20)
        ob = res.get("orderbook") or {}
        # Kalshi quotes both sides in their own terms: `yes` levels are bids for
        # YES, `no` levels are bids for NO. A NO bid at p is a YES ask at 1 - p.
        bids = [BookLevel(*self._level(l)) for l in (ob.get("yes") or [])]
        asks = [BookLevel(1.0 - p, s) for p, s in (self._level(l) for l in (ob.get("no") or []))]
        bids.sort(key=lambda x: -x.price)
        asks.sort(key=lambda x: x.price)
        return OrderBook(Venue.KALSHI, market_id, utcnow(), bids, asks)

    @staticmethod
    def _level(level: Any) -> tuple[float, float]:
        """A level is [price, size]; price may be cents (int) or dollars (str)."""
        price, size = level[0], level[1]
        p = float(price)
        if isinstance(price, (int,)) and p > 1.0:
            p /= 100.0
        elif isinstance(price, str):
            p = float(price)
            if p > 1.0:
                p /= 100.0
        return (p, float(size))
