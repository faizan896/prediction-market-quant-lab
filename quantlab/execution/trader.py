"""The paper trading loop.

One `step()` is one decision cycle:

    settle resolved positions
      -> mark the book
      -> scan for opportunities
      -> apply risk limits
      -> submit intents to the broker
      -> persist fills, positions and an equity point

It runs against the *same* `MispricingEngine` the live scanner uses, so the
paper record measures the thing you would actually trade. Swap `PaperBroker`
for `LiveBroker` and nothing else changes.

Run it on a schedule (hourly is plenty; these markets do not move in
microseconds) and let it accumulate. The number to watch is `edge_capture` in
the portfolio summary — not the equity curve, which is mostly noise for the
first few hundred trades.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone

from quantlab.data.store import Store
from quantlab.execution.broker import Broker, Fill, OrderIntent, PaperBroker, RejectedOrder
from quantlab.execution.portfolio import Portfolio, Position, mark_to_market
from quantlab.mispricing import MispricingEngine
from quantlab.types import Market


@dataclass
class RiskLimits:
    """Hard limits applied after sizing and before submission.

    These are not tuning knobs. They are the difference between a bad week and
    a blown account, and every one of them exists because the alternative is a
    known failure mode:

    max_gross_exposure  a scanner bug that flags 400 markets at once cannot
                        deploy the whole bankroll
    max_position        one contract cannot dominate the book
    max_positions       keeps the portfolio diversifiable and the blotter legible
    min_net_ev          nothing trades unless it clears its own costs
    max_price/min_price stay out of the extreme tails, where the tick grid and
                        settlement risk dominate any modelled edge
    """

    max_gross_exposure: float = 0.50  # share of equity deployed at once
    max_position: float = 0.05  # share of equity in any one contract
    max_positions: int = 40
    min_net_ev: float = 0.02
    min_confidence: float = 0.25
    min_edge: float = 0.02
    min_liquidity: float = 5_000.0
    min_price: float = 0.03
    max_price: float = 0.97
    max_new_per_step: int = 5
    # Exit when the reason for the trade is gone, not on a price stop: these
    # are convergence trades, so a drawdown is not by itself information.
    exit_edge: float = 0.005
    max_hold_hours: float | None = 24 * 45


@dataclass
class StepReport:
    ts: datetime
    scanned: int
    settled: list[Fill] = field(default_factory=list)
    opened: list[Fill] = field(default_factory=list)
    closed: list[Fill] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    equity: float = 0.0
    summary: dict = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.ts:%Y-%m-%d %H:%M} · scanned {self.scanned} · "
            f"opened {len(self.opened)} · closed {len(self.closed)} · "
            f"settled {len(self.settled)} · equity ${self.equity:,.0f}"
        )


def _rebuild_blotter(store: Store, run_id: str) -> list[dict]:
    """Reconstruct the closed-trade blotter by replaying persisted fills.

    Without this, every metric that lives on closed trades — win rate, Brier,
    and above all **edge capture** — resets to empty on resume. Since the
    trader resumes on every single cycle, those numbers would be permanently
    blank, and edge capture is the one number a paper record exists to produce.

    Opens are tracked per market; each close or settle pairs off against them
    FIFO and emits a completed trade.
    """
    df = store.fills_df(run_id)
    if df.empty:
        return []

    open_lots: dict[str, list[dict]] = {}
    closed: list[dict] = []

    for r in df.sort_values("ts").to_dict("records"):
        key = r["key"]
        if r["action"] == "open":
            open_lots.setdefault(key, []).append(r)
            continue

        lots = open_lots.get(key) or []
        if not lots:
            continue  # opened before this run's fill history — nothing to pair
        entry = lots.pop(0)
        basis = entry["notional"] or 0.0
        ts_in = datetime.fromisoformat(entry["ts"])
        ts_out = datetime.fromisoformat(r["ts"])
        closed.append({
            "key": key,
            "title": r.get("title") or entry.get("title") or "",
            "side": r["side"],
            "entry_price": entry["price"],
            "exit_price": r["price"],
            "contracts": r["contracts"],
            "cost_basis": basis,
            "pnl": r["realized_pnl"] or 0.0,
            "ret": (r["realized_pnl"] or 0.0) / basis if basis else 0.0,
            "fair_at_entry": entry.get("fair_prob"),
            "edge_at_entry": entry.get("edge"),
            "hold_hours": (ts_out - ts_in).total_seconds() / 3600.0,
            "action": r["action"],
            "opened_at": ts_in,
            "closed_at": ts_out,
            # For a settled contract the fill price IS the outcome (0 or 1),
            # which is exactly what calibration scoring needs.
            "outcome": r["price"] if r["action"] == "settle" else None,
        })
    return closed


class PaperTrader:
    def __init__(
        self,
        run_id: str | None = None,
        name: str = "paper",
        engine: MispricingEngine | None = None,
        broker: Broker | None = None,
        limits: RiskLimits | None = None,
        initial_capital: float = 100_000.0,
        store: Store | None = None,
        persist: bool = True,
        model: dict | None = None,
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.name = name
        self.engine = engine or MispricingEngine(bankroll=initial_capital)
        self.broker = broker or PaperBroker()
        self.limits = limits or RiskLimits()
        self.portfolio = Portfolio(initial_capital=initial_capital)
        self.store = store
        self.persist = persist and store is not None
        self.history: list[StepReport] = []

        if self.persist:
            # `model` records how the scanner was configured. Storing it beside
            # the limits is what makes a run reproducible: resume rebuilds the
            # identical engine rather than silently reverting to defaults.
            self.store.create_paper_run(
                self.run_id, name, type(self.engine).__name__,
                {"limits": self.limits.__dict__, "model": model or {}},
                initial_capital,
            )

    # ------------------------------------------------------------------- load
    @classmethod
    def resume(cls, run_id: str, store: Store, **kwargs) -> PaperTrader:
        """Rebuild a trader from persisted state, risk limits included.

        Restoring the limits is not optional. They were chosen when the run was
        created, and a run that silently reverts to defaults on every resume is
        no longer testing the configuration you think you are testing — which
        makes the entire paper record worthless without announcing it.
        """
        row = store.get_paper_run(run_id)
        if row is None:
            raise KeyError(f"no paper run '{run_id}'")

        if "limits" not in kwargs:
            try:
                saved = (json.loads(row["params"] or "{}") or {}).get("limits")
            except (TypeError, ValueError):
                saved = None
            if saved:
                known = {f.name for f in fields(RiskLimits)}
                kwargs["limits"] = RiskLimits(
                    **{k: v for k, v in saved.items() if k in known}
                )

        trader = cls(
            run_id=run_id, name=row["name"] or "paper",
            initial_capital=row["initial_capital"], store=store,
            persist=False, **kwargs,
        )
        trader.persist = True
        trader.portfolio.cash = row["cash"]
        trader.portfolio.realized_pnl = row["realized_pnl"]
        trader.portfolio.fees_paid = row["fees_paid"]
        for p in store.positions(run_id):
            trader.portfolio.positions[p["key"]] = Position(
                key=p["key"], title=p["title"] or "", side=p["side"],
                contracts=p["contracts"], avg_price=p["avg_price"],
                cost_basis=p["cost_basis"],
                opened_at=datetime.fromisoformat(p["opened_at"]),
                fair_at_entry=p["fair_at_entry"], edge_at_entry=p["edge_at_entry"],
            )
        trader.portfolio.closed = _rebuild_blotter(store, run_id)
        return trader

    # ------------------------------------------------------------------- step
    def step(self, markets: list[Market], books: dict | None = None) -> StepReport:
        now = datetime.now(timezone.utc)
        by_key = {m.key: m for m in markets}
        report = StepReport(ts=now, scanned=len(markets))

        # 1. settle anything that resolved while we held it
        report.settled = self._settle(by_key)

        # 2. exit positions whose thesis is gone
        opportunities = self._scan(markets, books)
        report.closed = self._exit(by_key, opportunities, report)

        # 3. open new positions
        report.opened = self._enter(by_key, opportunities, report)

        # 4. mark and record
        marks = mark_to_market(self.portfolio.positions, by_key)
        report.equity = self.portfolio.equity(marks)
        report.summary = self.portfolio.summary(marks)

        if self.persist:
            self.store.update_paper_run(
                self.run_id, self.portfolio.cash,
                self.portfolio.realized_pnl, self.portfolio.fees_paid,
            )
            self.store.record_equity(
                self.run_id, now, report.equity, self.portfolio.cash,
                self.portfolio.exposure(marks), len(self.portfolio.positions),
            )
        self.history.append(report)
        return report

    # -------------------------------------------------------------- internals
    def _scan(self, markets: list[Market], books: dict | None) -> dict[str, dict]:
        df = self.engine.scan(
            markets, books=books,
            min_edge=0.0, min_net_ev=-10.0, min_confidence=0.0,
        )
        if df.empty:
            return {}
        return {row["key"]: row for row in df.to_dict("records")}

    def _settle(self, by_key: dict[str, Market]) -> list[Fill]:
        out: list[Fill] = []
        for key in list(self.portfolio.positions):
            market = by_key.get(key)
            if market is None or market.resolution is None:
                continue
            pos = self.portfolio.positions[key]
            intent = OrderIntent(
                key=key, side=pos.side, contracts=pos.contracts, action="settle",
                fair_prob=pos.fair_at_entry, edge=pos.edge_at_entry,
                reason="market resolved", title=pos.title,
            )
            try:
                fill = self.broker.execute(intent, market)
                fill.realized_pnl = self.portfolio.apply(fill)
            except (RejectedOrder, ValueError):
                continue
            self._record(fill, closed_key=key)
            out.append(fill)
        return out

    def _exit(
        self, by_key: dict[str, Market], opps: dict[str, dict], report: StepReport
    ) -> list[Fill]:
        out: list[Fill] = []
        now = datetime.now(timezone.utc)
        for key in list(self.portfolio.positions):
            pos = self.portfolio.positions[key]
            market = by_key.get(key)
            if market is None or market.mid is None:
                continue

            reason = None
            opp = opps.get(key)
            if opp is not None:
                # Signed edge in the direction we are actually positioned.
                signed = (opp["fair_value"] - opp["price"]) * pos.sign
                if signed < self.limits.exit_edge:
                    reason = f"edge gone ({signed:+.3f})"
            if reason is None and self.limits.max_hold_hours is not None:
                held = (now - pos.opened_at).total_seconds() / 3600
                if held > self.limits.max_hold_hours:
                    reason = f"max hold ({held:.0f}h)"
            if reason is None:
                continue

            intent = OrderIntent(
                key=key, side=pos.side, contracts=pos.contracts, action="close",
                fair_prob=pos.fair_at_entry, edge=pos.edge_at_entry,
                reason=reason, title=pos.title,
            )
            try:
                fill = self.broker.execute(intent, market)
                fill.realized_pnl = self.portfolio.apply(fill)
            except (RejectedOrder, ValueError) as exc:
                report.rejected.append(f"close {key}: {exc}")
                continue
            self._record(fill, closed_key=key)
            out.append(fill)
        return out

    def _enter(
        self, by_key: dict[str, Market], opps: dict[str, dict], report: StepReport
    ) -> list[Fill]:
        out: list[Fill] = []
        lim = self.limits
        marks = mark_to_market(self.portfolio.positions, by_key)
        equity = self.portfolio.equity(marks)

        candidates = sorted(
            (o for k, o in opps.items() if k not in self.portfolio.positions),
            key=lambda o: -o.get("score", 0.0),
        )

        for opp in candidates:
            if len(out) >= lim.max_new_per_step:
                break
            if len(self.portfolio.positions) >= lim.max_positions:
                report.rejected.append("position count limit reached")
                break

            key = opp["key"]
            market = by_key.get(key)
            if market is None:
                continue

            reject = self._screen(opp, market, lim)
            if reject:
                report.rejected.append(f"{key}: {reject}")
                continue

            notional = min(
                float(opp["suggested_notional"]),
                equity * lim.max_position,
                max(0.0, equity * lim.max_gross_exposure - self.portfolio.deployed),
                self.portfolio.cash * 0.95,
            )
            if notional < 10:
                report.rejected.append(f"{key}: size below minimum after limits")
                continue

            intent = OrderIntent(
                key=key, side=opp["side"], notional=notional, action="open",
                fair_prob=float(opp["fair_value"]), edge=float(opp["edge"]),
                reason=f"net_ev {opp['net_ev']:+.3f} conf {opp['confidence']:.2f}",
                title=opp.get("title", ""),
            )
            try:
                fill = self.broker.execute(intent, market)
                self.portfolio.apply(fill)
            except (RejectedOrder, ValueError) as exc:
                report.rejected.append(f"open {key}: {exc}")
                continue
            self._record(fill)
            out.append(fill)
        return out

    @staticmethod
    def _screen(opp: dict, market: Market, lim: RiskLimits) -> str | None:
        if opp["net_ev"] < lim.min_net_ev:
            return f"net_ev {opp['net_ev']:+.3f} below {lim.min_net_ev}"
        if opp["confidence"] < lim.min_confidence:
            return f"confidence {opp['confidence']:.2f} below {lim.min_confidence}"
        if opp["edge"] < lim.min_edge:
            return f"edge {opp['edge']:.3f} below {lim.min_edge}"
        if not (lim.min_price <= opp["price"] <= lim.max_price):
            return f"price {opp['price']:.3f} outside tradable band"
        if (market.liquidity or 0) < lim.min_liquidity:
            return f"liquidity ${market.liquidity or 0:,.0f} below minimum"
        if market.resolution is not None:
            return "already resolved"
        return None

    def _record(self, fill: Fill, closed_key: str | None = None) -> None:
        if not self.persist:
            return
        self.store.record_fill(fill.to_row(self.run_id))
        if closed_key and closed_key not in self.portfolio.positions:
            self.store.delete_position(self.run_id, closed_key)
        elif fill.key in self.portfolio.positions:
            self.store.upsert_position(
                self.run_id, self.portfolio.positions[fill.key].to_row()
            )

    # ------------------------------------------------------------------ views
    def summary(self, markets: list[Market] | None = None) -> dict:
        marks = mark_to_market(
            self.portfolio.positions, {m.key: m for m in (markets or [])}
        )
        out = self.portfolio.summary(marks)
        out["run_id"] = self.run_id
        out["name"] = self.name
        out["steps"] = len(self.history)
        return out
