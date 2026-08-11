"""Polymarket client.

Three public services, no auth required for reads:
  Gamma  gamma-api.polymarket.com   what exists (events, markets, metadata)
  CLOB   clob.polymarket.com        what it costs (book, midpoint, price history)
  Data   data-api.polymarket.com    who is trading it (positions, trades, holders)

Gotcha handled here: `outcomes`, `outcomePrices` and `clobTokenIds` come back as
JSON-encoded *strings*, not arrays. Every naive integration trips on this.
"""

from __future__ import annotations

import json
from typing import Any

from quantlab.config import settings
from quantlab.data.base import HttpMixin, parse_ts, to_float, utcnow
from quantlab.types import BookLevel, Market, OrderBook, PricePoint, Venue


def _jarr(value: Any) -> list:
    """Parse Polymarket's stringified JSON arrays."""
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        out = json.loads(value)
        return out if isinstance(out, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


class PolymarketClient(HttpMixin):
    venue = Venue.POLYMARKET

    def __init__(self) -> None:
        super().__init__()
        self.base_url = settings.polymarket_gamma
        self.clob = settings.polymarket_clob
        self.data = settings.polymarket_data

    # ------------------------------------------------------------------ markets
    def list_markets(
        self,
        limit: int = 200,
        active: bool = True,
        closed: bool = False,
        order: str = "volumeNum",
        min_volume: float = 0.0,
        tag: str | None = None,
    ) -> list[Market]:
        """Fetch markets, paginating Gamma's 500-per-page cap."""
        import time as _time

        deadline = _time.monotonic() + settings.venue_budget_seconds
        out: list[Market] = []
        page = 0
        while len(out) < limit:
            # Stop paginating rather than blow the caller's request budget.
            # Partial results beat a timeout with nothing in it.
            if _time.monotonic() > deadline:
                break
            batch = self.get(
                "/markets",
                limit=min(500, limit - len(out)),
                offset=page * 500,
                active=str(active).lower(),
                closed=str(closed).lower(),
                order=order,
                ascending="false",
                tag_slug=tag,
            )
            if not batch:
                break
            for raw in batch:
                m = self._to_market(raw)
                if m and m.volume >= min_volume:
                    out.append(m)
            if len(batch) < 500:
                break
            page += 1
        return out[:limit]

    def get_market(self, market_id: str) -> Market | None:
        """`market_id` here is the Gamma numeric market id."""
        raw = self.get(f"/markets/{market_id}")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        return self._to_market(raw) if raw else None

    def search(self, query: str, limit: int = 20) -> list[Market]:
        res = self.get("/public-search", q=query, limit_per_type=limit)
        markets = res.get("markets", []) if isinstance(res, dict) else []
        return [m for m in (self._to_market(r) for r in markets) if m]

    def _to_market(self, raw: dict) -> Market | None:
        tokens = _jarr(raw.get("clobTokenIds"))
        if not tokens:
            return None
        outcomes = _jarr(raw.get("outcomes")) or ["Yes", "No"]
        prices = [to_float(p, 0.0) for p in _jarr(raw.get("outcomePrices"))]

        # Index 0 is the YES leg. Multi-outcome (negRisk) events arrive as one
        # market per candidate, each already binary, so index 0 is always right.
        last = prices[0] if prices else None
        events = raw.get("events") or [{}]

        fee_schedule = raw.get("feeSchedule") or {}
        return Market(
            venue=Venue.POLYMARKET,
            market_id=str(tokens[0]),
            event_id=str(events[0].get("id", raw.get("id", ""))),
            title=raw.get("question") or raw.get("groupItemTitle") or "",
            slug=raw.get("slug", ""),
            yes_bid=to_float(raw.get("bestBid")),
            yes_ask=to_float(raw.get("bestAsk")),
            last_price=to_float(raw.get("lastTradePrice"), last),
            volume=to_float(raw.get("volumeNum"), 0.0) or 0.0,
            volume_24h=to_float(raw.get("volume24hr"), 0.0) or 0.0,
            liquidity=to_float(raw.get("liquidityNum"), 0.0) or 0.0,
            open_time=parse_ts(raw.get("startDate")),
            close_time=parse_ts(raw.get("endDate")),
            status="closed" if raw.get("closed") else "open",
            resolved=bool(raw.get("closed")),
            min_tick=to_float(raw.get("orderPriceMinTickSize"), 0.01) or 0.01,
            fee_model="polymarket" if raw.get("feesEnabled") else "none",
            raw={
                "gamma_id": raw.get("id"),
                "condition_id": raw.get("conditionId"),
                "token_ids": tokens,
                "outcomes": outcomes,
                "neg_risk": raw.get("negRisk", False),
                "enable_order_book": raw.get("enableOrderBook", False),
                "fee_rate": to_float(fee_schedule.get("rate"), 0.0) or 0.0,
                "spread": to_float(raw.get("spread")),
            },
        )

    # ------------------------------------------------------------------ history
    def price_history(
        self,
        market_id: str,
        interval: str = "max",
        fidelity: int = 60,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[PricePoint]:
        """CLOB /prices-history. `fidelity` is the bar width in minutes.

        interval: 1m | 1w | 1d | 6h | 1h | max   (ignored if start/end given)
        """
        params: dict[str, Any] = {"market": market_id, "fidelity": fidelity}
        if start_ts and end_ts:
            params["startTs"] = start_ts
            params["endTs"] = end_ts
        else:
            params["interval"] = interval
        res = self.get(f"{self.clob}/prices-history", **params)
        return [
            PricePoint(
                venue=Venue.POLYMARKET,
                market_id=market_id,
                ts=parse_ts(pt["t"]),
                price=float(pt["p"]),
            )
            for pt in (res or {}).get("history", [])
        ]

    # --------------------------------------------------------------------- book
    def order_book(self, market_id: str) -> OrderBook | None:
        res = self.get(f"{self.clob}/book", token_id=market_id)
        if not res:
            return None
        bids = [BookLevel(float(b["price"]), float(b["size"])) for b in res.get("bids", [])]
        asks = [BookLevel(float(a["price"]), float(a["size"])) for a in res.get("asks", [])]
        # Polymarket returns bids ascending and asks descending; normalize.
        bids.sort(key=lambda x: -x.price)
        asks.sort(key=lambda x: x.price)
        return OrderBook(Venue.POLYMARKET, market_id, utcnow(), bids, asks)

    def midpoint(self, market_id: str) -> float | None:
        return to_float(self.get(f"{self.clob}/midpoint", token_id=market_id).get("mid"))
