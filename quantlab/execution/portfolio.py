"""Portfolio accounting for binary contracts.

Everything is held in YES terms. A NO position of N contracts entered when YES
traded at p has cost basis N x (1 - p) and is worth N x (1 - mark) later.

The one number this module exists to produce is **edge capture**: realized
return divided by the edge the model predicted at entry. A strategy can have a
real edge and still lose money if it captures 30% of it after costs. You cannot
see that in a win rate, and it is the first thing that separates a paper record
worth promoting from one that isn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quantlab.execution.broker import Fill


@dataclass
class Position:
    key: str
    title: str
    side: str  # "YES" | "NO"
    contracts: float
    avg_price: float  # YES-terms average entry
    cost_basis: float  # cash deployed
    opened_at: datetime
    fair_at_entry: float | None = None
    edge_at_entry: float | None = None

    @property
    def sign(self) -> float:
        return 1.0 if self.side == "YES" else -1.0

    def market_value(self, mark: float) -> float:
        """Current liquidation value at YES price `mark`."""
        per_contract = mark if self.sign > 0 else 1.0 - mark
        return self.contracts * per_contract

    def unrealized(self, mark: float) -> float:
        return self.market_value(mark) - self.cost_basis

    def to_row(self) -> dict:
        return {
            "key": self.key, "title": self.title, "side": self.side,
            "contracts": self.contracts, "avg_price": self.avg_price,
            "cost_basis": self.cost_basis,
            "opened_at": self.opened_at.isoformat(),
            "fair_at_entry": self.fair_at_entry,
            "edge_at_entry": self.edge_at_entry,
        }


@dataclass
class Portfolio:
    initial_capital: float = 100_000.0
    cash: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.cash == 0.0:
            self.cash = self.initial_capital

    # ------------------------------------------------------------------ state
    @property
    def deployed(self) -> float:
        return sum(p.cost_basis for p in self.positions.values())

    def exposure(self, marks: dict[str, float]) -> float:
        return sum(
            p.market_value(marks.get(k, p.avg_price)) for k, p in self.positions.items()
        )

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.exposure(marks)

    def unrealized(self, marks: dict[str, float]) -> float:
        return sum(
            p.unrealized(marks.get(k, p.avg_price)) for k, p in self.positions.items()
        )

    # ------------------------------------------------------------------ writes
    def apply(self, fill: Fill) -> float:
        """Apply a fill and return the realized PnL it produced (0 on opens)."""
        if fill.action == "open":
            return self._open(fill)
        return self._close(fill)

    def _open(self, fill: Fill) -> float:
        cost = fill.notional + fill.fee + fill.slippage
        if cost > self.cash:
            raise ValueError(
                f"insufficient cash: need ${cost:,.2f}, have ${self.cash:,.2f}"
            )
        self.cash -= cost
        self.fees_paid += fill.fee + fill.slippage

        existing = self.positions.get(fill.key)
        if existing is None:
            self.positions[fill.key] = Position(
                key=fill.key, title=fill.title, side=fill.side,
                contracts=fill.contracts, avg_price=fill.price,
                cost_basis=fill.notional, opened_at=fill.ts,
                fair_at_entry=fill.fair_prob, edge_at_entry=fill.edge,
            )
        else:
            if existing.side != fill.side:
                raise ValueError(
                    f"{fill.key}: cannot add a {fill.side} leg to a {existing.side} "
                    "position — close it first"
                )
            total = existing.contracts + fill.contracts
            existing.avg_price = (
                existing.avg_price * existing.contracts + fill.price * fill.contracts
            ) / total
            existing.contracts = total
            existing.cost_basis += fill.notional
        return 0.0

    def _close(self, fill: Fill) -> float:
        pos = self.positions.get(fill.key)
        if pos is None:
            raise ValueError(f"{fill.key}: no open position to close")

        contracts = min(fill.contracts, pos.contracts)
        frac = contracts / pos.contracts if pos.contracts else 0.0
        basis_out = pos.cost_basis * frac

        per_contract = fill.price if pos.sign > 0 else 1.0 - fill.price
        proceeds = contracts * per_contract
        costs = fill.fee + fill.slippage

        pnl = proceeds - basis_out - costs
        self.cash += proceeds - costs
        self.realized_pnl += pnl
        self.fees_paid += costs

        hold_h = (fill.ts - pos.opened_at).total_seconds() / 3600.0
        self.closed.append({
            "key": pos.key, "title": pos.title, "side": pos.side,
            "entry_price": pos.avg_price, "exit_price": fill.price,
            "contracts": contracts, "cost_basis": basis_out,
            "pnl": pnl, "ret": pnl / basis_out if basis_out else 0.0,
            "fair_at_entry": pos.fair_at_entry, "edge_at_entry": pos.edge_at_entry,
            "hold_hours": hold_h, "action": fill.action,
            "opened_at": pos.opened_at, "closed_at": fill.ts,
            # For a settled contract the outcome is the fill price itself,
            # which is exactly what calibration scoring needs.
            "outcome": fill.price if fill.action == "settle" else None,
        })

        pos.contracts -= contracts
        pos.cost_basis -= basis_out
        if pos.contracts <= 1e-9:
            del self.positions[fill.key]
        return pnl

    # ---------------------------------------------------------------- reports
    def blotter(self) -> pd.DataFrame:
        return pd.DataFrame(self.closed)

    def summary(self, marks: dict[str, float] | None = None) -> dict:
        marks = marks or {}
        eq = self.equity(marks)
        closed = self.blotter()

        out = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "exposure": self.exposure(marks),
            "equity": eq,
            "total_return": (eq - self.initial_capital) / self.initial_capital,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized(marks),
            "fees_paid": self.fees_paid,
            "open_positions": len(self.positions),
            "closed_trades": len(closed),
            "win_rate": float("nan"),
            "avg_predicted_edge": float("nan"),
            "avg_realized_return": float("nan"),
            "edge_capture": float("nan"),
            "brier": float("nan"),
        }
        if closed.empty:
            return out

        out["win_rate"] = float((closed["pnl"] > 0).mean())
        out["avg_realized_return"] = float(closed["ret"].mean())

        edges = closed["edge_at_entry"].dropna().abs()
        if len(edges):
            out["avg_predicted_edge"] = float(edges.mean())
            # Predicted edge is per contract; realized return is per dollar of
            # basis. Convert the edge to the same units before dividing, or the
            # ratio is meaningless.
            basis = closed.loc[edges.index, "entry_price"].where(
                closed.loc[edges.index, "side"] == "YES",
                1 - closed.loc[edges.index, "entry_price"],
            ).clip(lower=1e-6)
            predicted_ret = (edges / basis).mean()
            if predicted_ret > 0:
                out["edge_capture"] = float(out["avg_realized_return"] / predicted_ret)

        settled = closed[closed["action"] == "settle"].dropna(
            subset=["fair_at_entry", "outcome"]
        )
        if len(settled):
            out["brier"] = float(
                np.mean((settled["fair_at_entry"] - settled["outcome"]) ** 2)
            )
        return out


def mark_to_market(positions: dict[str, Position], markets: dict[str, object]) -> dict[str, float]:
    """Build a {key: yes_mid} mark map, falling back to entry price when a
    market has gone quote-less (delisted, settled, or simply not in the pull)."""
    marks: dict[str, float] = {}
    for key, pos in positions.items():
        m = markets.get(key)
        mid = getattr(m, "mid", None) if m is not None else None
        marks[key] = float(mid) if mid is not None else pos.avg_price
    return marks


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
