"""Strategy interface.

A strategy is a pure function from panel -> (signal, fair_prob).

    signal     in [-1, 1]. Sign picks the leg (+1 YES, -1 NO); magnitude is
               conviction, compared against the engine's entry/exit thresholds.
    fair_prob  the model's probability of YES. Drives Kelly sizing and the
               calibration metrics. Return the market price if you have no view.

Rules
-----
1. Never look forward. Use `roll()` and `shift()` helpers here, which are
   groupby-aware, rather than raw pandas on the whole panel — the panel is
   long-format and a naive .rolling() will bleed one market into the next.
2. The engine applies an extra one-bar lag on top of whatever you do, so an
   honest strategy is safe by construction. A dishonest one (e.g. reading
   `resolution`) is not, so don't.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Strategy(ABC):
    name: str = "unnamed"

    def __init__(self, **params: Any):
        self.params = params
        for k, v in params.items():
            setattr(self, k, v)

    @abstractmethod
    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with columns `signal` and (optionally) `fair_prob`,
        indexed identically to `panel`."""

    # ------------------------------------------------------------- helpers
    @staticmethod
    def by_market(panel: pd.DataFrame, col: str = "price"):
        return panel.groupby("key", sort=False)[col]

    @staticmethod
    def roll(panel: pd.DataFrame, col: str, window: int, fn: str = "mean") -> pd.Series:
        """Groupby-aware rolling reduction. Never crosses market boundaries."""
        g = panel.groupby("key", sort=False)[col]
        r = g.rolling(window, min_periods=max(2, window // 2))
        out = getattr(r, fn)()
        return out.reset_index(level=0, drop=True).sort_index()

    @staticmethod
    def lag(panel: pd.DataFrame, col: str, periods: int = 1) -> pd.Series:
        return panel.groupby("key", sort=False)[col].shift(periods)

    @staticmethod
    def bars_elapsed(panel: pd.DataFrame) -> pd.Series:
        return panel.groupby("key", sort=False).cumcount()

    @staticmethod
    def hours_to_close(panel: pd.DataFrame) -> pd.Series:
        if "close_time" not in panel.columns:
            return pd.Series(np.nan, index=panel.index)
        delta = panel["close_time"] - panel["ts"]
        return delta.dt.total_seconds() / 3600.0

    @staticmethod
    def squash(x: pd.Series, scale: float) -> pd.Series:
        """Map an unbounded score to a bounded signal in (-1, 1)."""
        return np.tanh(x.astype(float) / max(scale, 1e-9))

    @staticmethod
    def frame(signal: pd.Series, fair: pd.Series | None = None) -> pd.DataFrame:
        out = pd.DataFrame({"signal": signal.fillna(0.0).clip(-1, 1)})
        if fair is not None:
            out["fair_prob"] = fair.clip(0.001, 0.999)
        return out

    def __repr__(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({ps})"
