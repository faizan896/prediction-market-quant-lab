"""Shared HTTP plumbing and the client interface every venue implements."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from quantlab.config import settings
from quantlab.types import Market, OrderBook, PricePoint, Venue


class RateLimiter:
    """Trivial token-free spacing limiter. Keeps us well under venue limits."""

    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


class HttpMixin:
    base_url: str = ""

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=settings.http_timeout,
            headers={"User-Agent": "quantlab/0.1"},
            follow_redirects=True,
        )
        self._limiter = RateLimiter(settings.rate_limit_rps)

    def get(self, path: str, **params: Any) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        params = {k: v for k, v in params.items() if v is not None}
        last: Exception | None = None
        for attempt in range(settings.http_retries):
            self._limiter.wait()
            try:
                r = self._client.get(url, params=params)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last = exc
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"GET {url} failed after {settings.http_retries} attempts: {last}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class VenueClient(Protocol):
    venue: Venue

    def list_markets(self, limit: int = 200, **kw: Any) -> list[Market]: ...
    def price_history(self, market_id: str, **kw: Any) -> list[PricePoint]: ...
    def order_book(self, market_id: str) -> OrderBook | None: ...


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: Any) -> datetime | None:
    """Tolerant timestamp parser: ISO strings, epoch seconds, or None."""
    if value in (None, "", 0):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
