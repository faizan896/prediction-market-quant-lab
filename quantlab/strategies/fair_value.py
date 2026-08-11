"""Bridge: run the mispricing engine's logic as a backtestable strategy.

The scanner tells you what looks mispriced *now*. This tells you whether that
same logic would have made money over the last N months. Being able to backtest
your live scanner with the identical model is the whole point of the Research ->
Backtest -> Paper -> Live pipeline; if the scanner and the backtest use
different code, the backtest is measuring something you will never trade.

Vectorized reimplementation of the pooling in `mispricing.engine`, operating on
a panel rather than one market at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import Strategy


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FairValueStrategy(Strategy):
    """Pool a longshot correction, a drift tilt and (if present) the other
    venue's price into a fair value; trade the gap when it clears costs.

    Parameters
    ----------
    market_anchor
        Weight kept on the market price (see MispricingEngine).
    min_edge
        Gap below which nothing trades. Set this at or above your round-trip
        cost or the backtest will report a profit you cannot collect.
    longshot_strength, drift_window, cross_weight
        Per-signal knobs, matched to the live scanner's defaults.
    """

    name = "Fair Value"

    def __init__(
        self,
        market_anchor: float = 0.30,
        min_edge: float = 0.025,
        longshot_strength: float = 0.30,
        drift_window: int = 12,
        drift_weight: float = 0.7,
        cross_weight: float = 2.0,
    ):
        super().__init__(
            market_anchor=market_anchor,
            min_edge=min_edge,
            longshot_strength=longshot_strength,
            drift_window=drift_window,
            drift_weight=drift_weight,
            cross_weight=cross_weight,
        )

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = panel["price"].clip(0.005, 0.995)
        lp = _logit(p.to_numpy())

        num = np.zeros(len(panel))
        den = np.zeros(len(panel))

        # --- longshot correction -------------------------------------------
        hours = self.hours_to_close(panel)
        decay = (hours / (hours + 24.0)).fillna(0.5).clip(0, 1).to_numpy()
        ls_fair = lp * (1 + self.longshot_strength * decay)
        ls_conf = np.clip(np.abs(p.to_numpy() - 0.5) * 1.2, 0.05, 0.6) * decay
        num += ls_conf * ls_fair
        den += ls_conf

        # --- volume-weighted drift ------------------------------------------
        vol = panel["volume"] if "volume" in panel.columns else pd.Series(1.0, index=panel.index)
        pv = panel.assign(_pv=p * vol.clip(lower=1e-6), _v=vol.clip(lower=1e-6))
        vwap = (
            self.roll(pv, "_pv", self.drift_window, "sum")
            / self.roll(pv, "_v", self.drift_window, "sum")
        ).fillna(p)
        drift = (p - vwap).clip(-0.04, 0.04)
        dr_fair = _logit((p + drift).clip(0.005, 0.995).to_numpy())
        dr_conf = np.clip(np.abs(drift.to_numpy()) * 10, 0.05, 0.45) * self.drift_weight
        num += dr_conf * dr_fair
        den += dr_conf

        # --- cross-venue reference (optional) --------------------------------
        if "reference_price" in panel.columns:
            ref = panel["reference_price"]
            has_ref = ref.notna().to_numpy()
            ref_logit = _logit(ref.fillna(p).clip(0.005, 0.995).to_numpy())
            ref_conf = np.where(has_ref, self.cross_weight * 0.6, 0.0)
            num += ref_conf * ref_logit
            den += ref_conf

        pooled = np.where(den > 0, _sigmoid(num / np.where(den > 0, den, 1)), p.to_numpy())

        # --- anchor to the market -------------------------------------------
        a = self.market_anchor
        fair = _sigmoid(a * lp + (1 - a) * _logit(pooled))
        fair_s = pd.Series(fair, index=panel.index)

        edge = fair_s - p
        signal = pd.Series(0.0, index=panel.index)
        signal[edge >= self.min_edge] = 1.0
        signal[edge <= -self.min_edge] = -1.0
        signal = signal * self.squash(edge.abs(), self.min_edge).clip(0, 1)

        return self.frame(signal, fair_s)
