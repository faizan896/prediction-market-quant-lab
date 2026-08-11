"""Cross-venue and within-event arbitrage.

Two flavours, both structural rather than predictive:

CrossVenueArb
    The same question priced differently on Polymarket and Kalshi. Requires a
    verified link between the two contracts. The gap is real money only if it
    survives *both* venues' fees plus the capital lockup until resolution.

DutchBook
    Within a single multi-outcome event, the YES prices of mutually exclusive,
    collectively exhaustive outcomes should sum to 1. When they sum to 0.94 you
    can buy the whole set and collect 6c risk-free. Polymarket's negRisk events
    are the natural hunting ground.

Honest warning: published arbitrage in these markets is usually either (a)
already gone by the time your order lands, (b) a resolution-criteria difference
disguised as a price difference, or (c) sized at $50. The value of this module
is mostly in ruling out (b) systematically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import Strategy


class CrossVenueArb(Strategy):
    """Trade the spread between two linked contracts.

    The panel must carry a `link_id` column (from Store.links_df) and a
    `reference_price` column holding the *other* venue's price at the same
    timestamp. Build it with `quantlab.strategies.arbitrage.attach_reference`.
    """

    name = "Cross-Venue Arb"

    def __init__(self, min_gap: float = 0.03, trust_reference: float = 0.5):
        super().__init__(min_gap=min_gap, trust_reference=trust_reference)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = panel["price"]
        if "reference_price" not in panel.columns:
            zero = pd.Series(0.0, index=panel.index)
            return self.frame(zero, p)

        ref = panel["reference_price"]
        # Fair value pools the two venues. `trust_reference` = 0.5 treats them
        # as equally informed; lower it if one venue is thinner.
        fair = (1 - self.trust_reference) * p + self.trust_reference * ref
        gap = fair - p

        signal = pd.Series(0.0, index=panel.index)
        signal[gap >= self.min_gap] = 1.0
        signal[gap <= -self.min_gap] = -1.0
        signal = signal * self.squash(gap.abs(), self.min_gap).clip(0, 1)

        return self.frame(signal, fair.fillna(p))


class DutchBook(Strategy):
    """Buy every outcome in an event when the prices sum to less than 1.

    The panel must carry `event_id`. Signal fires on all legs of an event whose
    YES prices sum below (1 - threshold), sized so the basket is bought whole.
    """

    name = "Dutch Book"

    def __init__(self, threshold: float = 0.02, max_legs: int = 12):
        super().__init__(threshold=threshold, max_legs=max_legs)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = panel["price"]
        if "event_id" not in panel.columns:
            return self.frame(pd.Series(0.0, index=panel.index), p)

        grp = panel.groupby(["event_id", "ts"], sort=False)["price"]
        book_sum = grp.transform("sum")
        n_legs = grp.transform("size")

        valid = (n_legs >= 2) & (n_legs <= self.max_legs)
        underpriced = valid & (book_sum <= 1.0 - self.threshold)
        overpriced = valid & (book_sum >= 1.0 + self.threshold)

        signal = pd.Series(0.0, index=panel.index)
        signal[underpriced] = 1.0
        signal[overpriced] = -1.0

        # Fair value: renormalize the basket to sum to exactly 1.
        fair = (p / book_sum.replace(0, np.nan)).fillna(p).clip(0.01, 0.99)
        return self.frame(signal, fair)


DEFAULT_MATCH_TOLERANCE = pd.Timedelta("1h")


def attach_reference(
    panel: pd.DataFrame,
    links: pd.DataFrame,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Join each linked market's counterpart price onto the panel.

    Uses `merge_asof` with a backward tolerance so a Kalshi bar is matched to
    the most recent Polymarket bar at or before it, never a future one.
    """
    tolerance = tolerance or DEFAULT_MATCH_TOLERANCE
    if links.empty or panel.empty:
        return panel.assign(reference_price=np.nan, link_id=None)

    link_map = links.set_index("key")[["link_id", "invert"]]
    p = panel.merge(link_map, left_on="key", right_index=True, how="left")

    linked = p[p["link_id"].notna()].copy()
    if linked.empty:
        return p.assign(reference_price=np.nan)

    out_frames = []
    for link_id, grp in linked.groupby("link_id", sort=False):
        keys = grp["key"].unique()
        if len(keys) != 2:
            grp["reference_price"] = np.nan
            out_frames.append(grp)
            continue
        a, b = keys
        for this_key, other_key in ((a, b), (b, a)):
            left = grp[grp["key"] == this_key].sort_values("ts")
            right = grp[grp["key"] == other_key].sort_values("ts")[["ts", "price", "invert"]]
            right = right.rename(columns={"price": "reference_price"})
            right["reference_price"] = np.where(
                right["invert"].fillna(0) == 1,
                1 - right["reference_price"],
                right["reference_price"],
            )
            merged = pd.merge_asof(
                left, right[["ts", "reference_price"]],
                on="ts", direction="backward", tolerance=tolerance,
            )
            out_frames.append(merged)

    result = pd.concat(out_frames, ignore_index=True)
    unlinked = p[p["link_id"].isna()].copy()
    unlinked["reference_price"] = np.nan
    return pd.concat([result, unlinked], ignore_index=True).sort_values(["key", "ts"])
