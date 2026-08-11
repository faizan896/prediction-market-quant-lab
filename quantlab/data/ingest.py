"""Ingestion orchestration: pull markets + history from both venues into the store.

Also does cross-venue market linking, which is what makes arbitrage and the
CrossVenue fair-value signal possible. Linking is the genuinely hard part:
"Will Trump win?" on Polymarket is `PRES-2028-DEM` on Kalshi. The auto-matcher
here is a token-overlap heuristic with a conservative threshold — good enough
to surface candidates, not good enough to trade blind. Confirm links manually
(`quantlab link --confirm`) before sizing anything on them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quantlab.data.kalshi import KalshiClient
from quantlab.data.polymarket import PolymarketClient
from quantlab.data.store import Store
from quantlab.types import Market

_STOP = {
    "will", "the", "a", "an", "be", "in", "on", "at", "to", "of", "for", "by",
    "is", "are", "was", "were", "and", "or", "than", "before", "after", "this",
    "that", "it", "its", "with", "as", "if", "any", "who", "what", "when",
}


def _tokens(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", title.lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def _numbers(title: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", title))


@dataclass
class LinkCandidate:
    poly_key: str
    kalshi_key: str
    poly_title: str
    kalshi_title: str
    score: float
    poly_price: float | None
    kalshi_price: float | None

    @property
    def gap(self) -> float | None:
        if self.poly_price is None or self.kalshi_price is None:
            return None
        return self.kalshi_price - self.poly_price


def ingest_markets(
    store: Store,
    venues: tuple[str, ...] = ("polymarket", "kalshi"),
    limit: int = 500,
    min_volume: float = 10_000,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if "polymarket" in venues:
        with PolymarketClient() as pc:
            ms = pc.list_markets(limit=limit, min_volume=min_volume)
            counts["polymarket"] = store.upsert_markets(ms)
    if "kalshi" in venues:
        with KalshiClient() as kc:
            ms = kc.list_markets(limit=limit, min_volume=min_volume)
            counts["kalshi"] = store.upsert_markets(ms)
    return counts


def ingest_history(
    store: Store,
    venue: str,
    market_ids: list[str],
    fidelity_minutes: int = 60,
    progress=None,
) -> int:
    """Backfill price history for the given markets. `fidelity_minutes` is the
    bar width; 60 is a sane default (1m bars on 500 markets is ~50M rows)."""
    total = 0
    client = PolymarketClient() if venue == "polymarket" else KalshiClient()
    try:
        for mid in market_ids:
            try:
                if venue == "polymarket":
                    pts = client.price_history(mid, interval="max", fidelity=fidelity_minutes)
                else:
                    pts = client.price_history(mid, period_interval=fidelity_minutes)
                total += store.upsert_prices(pts)
            except Exception as exc:  # noqa: BLE001 - one bad market shouldn't halt a backfill
                if progress:
                    progress(f"skip {mid}: {exc}")
            if progress:
                progress(f"{mid}: {total} rows")
    finally:
        client.close()
    return total


def find_links(
    store: Store,
    min_score: float = 0.55,
    max_candidates: int = 200,
) -> list[LinkCandidate]:
    """Propose Polymarket<->Kalshi pairs by title similarity.

    Score = Jaccard on content tokens, with a hard requirement that any numbers
    in the titles agree (so "above 90 degrees" never matches "above 95 degrees").
    """
    poly = store.markets_df(venue="polymarket")
    kal = store.markets_df(venue="kalshi")
    if poly.empty or kal.empty:
        return []

    kal_idx = [
        (row.key, row.title, _tokens(row.title), _numbers(row.title),
         row.last_price if row.last_price is not None else None)
        for row in kal.itertuples()
    ]

    out: list[LinkCandidate] = []
    for prow in poly.itertuples():
        ptok, pnum = _tokens(prow.title), _numbers(prow.title)
        if not ptok:
            continue
        best: LinkCandidate | None = None
        for kkey, ktitle, ktok, knum, kprice in kal_idx:
            if not ktok:
                continue
            if pnum and knum and not (pnum & knum):
                continue  # numeric strikes disagree
            inter = len(ptok & ktok)
            if inter < 2:
                continue
            score = inter / len(ptok | ktok)
            if score >= min_score and (best is None or score > best.score):
                best = LinkCandidate(
                    poly_key=prow.key, kalshi_key=kkey,
                    poly_title=prow.title, kalshi_title=ktitle,
                    score=round(score, 3),
                    poly_price=prow.last_price, kalshi_price=kprice,
                )
        if best:
            out.append(best)

    out.sort(key=lambda c: -c.score)
    return out[:max_candidates]


def save_links(store: Store, candidates: list[LinkCandidate], method: str = "auto") -> int:
    for i, c in enumerate(candidates):
        link_id = f"link-{i:05d}"
        store.save_link(link_id, c.poly_key, False, c.score, method)
        store.save_link(link_id, c.kalshi_key, False, c.score, method)
    return len(candidates)


def markets_for_backfill(store: Store, venue: str, top_n: int = 100) -> list[str]:
    df = store.markets_df(venue=venue)
    if df.empty:
        return []
    return df.nlargest(top_n, "volume")["market_id"].tolist()


def summarize(markets: list[Market]) -> str:
    if not markets:
        return "no markets"
    vol = sum(m.volume for m in markets)
    return f"{len(markets)} markets, ${vol:,.0f} cumulative volume"
