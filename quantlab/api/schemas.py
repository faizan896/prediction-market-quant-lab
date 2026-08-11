"""API request/response models.

These are deliberately *not* the domain dataclasses. The frontend is a separate
deployable; coupling its wire format to internal types means every refactor of
`Market` becomes a breaking API change.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ markets
class MarketOut(BaseModel):
    key: str
    venue: str
    market_id: str
    event_id: str = ""
    title: str
    slug: str = ""
    yes_bid: float | None = None
    yes_ask: float | None = None
    mid: float | None = None
    spread: float | None = None
    volume: float = 0
    volume_24h: float = 0
    liquidity: float = 0
    close_time: str | None = None
    status: str = "open"
    resolved: bool = False
    resolution: float | None = None


class OrderBookOut(BaseModel):
    key: str
    bids: list[list[float]] = []
    asks: list[list[float]] = []
    best_bid: float | None = None
    best_ask: float | None = None
    microprice: float | None = None


class PricePointOut(BaseModel):
    ts: str
    price: float
    volume: float = 0


# ------------------------------------------------------------------ scanner
class SignalSpec(BaseModel):
    """Turn individual fair-value signals on and off, and reweight them."""

    name: Literal["market", "cross_venue", "book_imbalance", "longshot", "drift"]
    enabled: bool = True
    weight: float | None = None


class ScanRequest(BaseModel):
    venues: list[str] = ["polymarket", "kalshi"]
    limit: int = Field(300, ge=1, le=2000)
    min_volume: float = 50_000
    min_edge: float = 0.02
    min_net_ev: float = 0.0
    min_confidence: float = 0.15
    min_liquidity: float = 0.0
    bankroll: float = 100_000
    market_anchor: float = Field(0.30, ge=0.0, le=0.95)
    frictions: Literal["frictionless", "polymarket", "kalshi", "conservative"] = "conservative"
    signals: list[SignalSpec] | None = None
    external: dict[str, list[float]] = Field(
        default_factory=dict,
        description="{market_key: [probability, confidence]} — bring your own model",
    )
    use_books: bool = False


class OpportunityOut(BaseModel):
    key: str
    venue: str
    title: str
    price: float
    fair_value: float
    edge: float
    side: str
    confidence: float
    gross_ev: float
    net_ev: float
    kelly_fraction: float
    suggested_notional: float
    spread: float
    liquidity: float
    volume_24h: float
    hours_to_close: float | None = None
    contributors: dict[str, float] = {}
    score: float = 0.0
    notes: str = ""


class ScanResponse(BaseModel):
    scanned: int
    returned: int
    opportunities: list[OpportunityOut]
    elapsed_ms: float
    # Non-fatal problems, chiefly a venue that could not be reached. Empty
    # results are meaningless unless you know whether the data arrived.
    warnings: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- backtest
class StrategySpec(BaseModel):
    name: str
    params: dict[str, Any] = {}


class BacktestRequest(BaseModel):
    strategies: list[StrategySpec] | None = None
    venue: str | None = None
    synthetic: bool = False
    synthetic_markets: int = Field(120, ge=5, le=1000)
    synthetic_steps: int = Field(180, ge=20, le=1000)
    synthetic_spread: float = Field(0.01, ge=0.0, le=0.2)
    synthetic_bias: float = Field(0.05, ge=0.0, le=0.3)
    frictions: Literal["frictionless", "polymarket", "kalshi", "conservative"] = "conservative"
    capital: float = 100_000
    kelly_multiplier: float = Field(0.25, gt=0, le=1.0)
    max_fraction: float = Field(0.05, gt=0, le=1.0)
    signal_lag: int = Field(1, ge=0, le=10)
    walk_forward_folds: int = Field(0, ge=0, le=10)


class MetricsOut(BaseModel):
    strategy: str
    trades: int
    win_rate: float
    roi: float
    sharpe: float
    sortino: float = 0.0
    max_drawdown: float
    profit_factor: float
    brier: float | None = None
    edge_capture: float | None = None
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    final_equity: float = 0.0
    avg_hold_hours: float = 0.0


class EquityPoint(BaseModel):
    ts: str
    equity: float


class FoldOut(BaseModel):
    fold: int
    trades: int
    win_rate: float
    roi: float
    sharpe: float


class BacktestResponse(BaseModel):
    run_id: str
    markets: int
    bars: int
    frictions: str
    leaderboard: list[MetricsOut]
    equity: dict[str, list[EquityPoint]] = {}
    reliability: dict[str, list[dict]] = {}
    folds: list[FoldOut] = []
    warnings: list[str] = []


# ------------------------------------------------------------------- paper
class PaperRunCreate(BaseModel):
    """A paper run pins both the *risk limits* and the *model* it is testing.

    Both are persisted and restored on every step. A run that quietly reverts to
    defaults between cycles is not evaluating the configuration you chose, and
    the resulting record tells you nothing.
    """

    name: str = "paper-1"
    capital: float = 100_000
    # risk limits
    min_net_ev: float = 0.02
    min_confidence: float = 0.25
    min_edge: float = 0.02
    min_liquidity: float = 5_000
    max_positions: int = 40
    max_gross_exposure: float = 0.50
    max_position: float = 0.05
    # model
    frictions: Literal["frictionless", "polymarket", "kalshi", "conservative"] = "conservative"
    market_anchor: float = Field(0.30, ge=0.0, le=0.95)
    signals: list[SignalSpec] | None = None


class PaperStepRequest(BaseModel):
    venues: list[str] = ["polymarket", "kalshi"]
    limit: int = 300
    min_volume: float = 50_000
    synthetic: bool = Field(
        False, description="Drive a step from generated markets — lets you exercise "
                           "the whole loop without waiting for real markets to move.",
    )


class PositionOut(BaseModel):
    key: str
    title: str
    side: str
    contracts: float
    avg_price: float
    cost_basis: float
    mark: float | None = None
    market_value: float | None = None
    unrealized: float | None = None
    fair_at_entry: float | None = None
    edge_at_entry: float | None = None
    opened_at: str


class FillOut(BaseModel):
    fill_id: str
    key: str
    title: str | None = None
    ts: str
    action: str
    side: str
    price: float
    contracts: float
    notional: float
    fee: float = 0
    realized_pnl: float = 0
    edge: float | None = None
    reason: str | None = None


class PaperSummary(BaseModel):
    run_id: str
    name: str
    status: str = "active"
    initial_capital: float
    cash: float
    equity: float
    exposure: float = 0
    total_return: float
    realized_pnl: float
    unrealized_pnl: float = 0
    fees_paid: float
    open_positions: int
    closed_trades: int
    win_rate: float | None = None
    edge_capture: float | None = None
    brier: float | None = None
    created_at: str | None = None
    updated_at: str | None = None


class PaperDetail(BaseModel):
    summary: PaperSummary
    positions: list[PositionOut]
    fills: list[FillOut]
    equity: list[EquityPoint]


class StepResponse(BaseModel):
    ts: str
    scanned: int
    opened: list[FillOut]
    closed: list[FillOut]
    settled: list[FillOut]
    rejected: list[str]
    summary: PaperSummary
