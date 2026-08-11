"""Baseline strategies. No machine learning anywhere in this file, by design.

Start here. Get these working, understand why they do or don't make money, and
only then reach for a model. The order matters: if you can't beat the market
with a 20-line momentum rule, an XGBoost ensemble on the same features will just
overfit faster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Control group. Buys YES on every market and holds to resolution.

    Its ROI is the base rate you must beat. If a clever strategy can't beat
    buy-and-hold after costs, it is an expensive way to be average.
    """

    name = "Buy & Hold"

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        first_bar = self.bars_elapsed(panel) == 0
        signal = pd.Series(np.where(first_bar, 1.0, 1.0), index=panel.index)
        return self.frame(signal, panel["price"])


class Momentum(Strategy):
    """Trade in the direction of recent price drift.

    Rationale: prediction markets absorb news with a lag. When a market moves
    from 40c to 47c on real information, the move is usually not finished —
    retail flow arrives over hours, not seconds. The z-score normalization keeps
    the strategy from firing on ordinary noise in a quiet market.
    """

    name = "Momentum"

    def __init__(self, lookback: int = 12, vol_window: int = 48, z_entry: float = 1.0):
        super().__init__(lookback=lookback, vol_window=vol_window, z_entry=z_entry)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        price = panel["price"]
        past = self.lag(panel, "price", self.lookback)
        change = price - past

        ret = price - self.lag(panel, "price", 1)
        vol = self.roll(panel.assign(_r=ret), "_r", self.vol_window, "std")
        vol = vol.replace(0, np.nan)

        z = change / (vol * np.sqrt(self.lookback))
        signal = self.squash(z, self.z_entry)

        # Fair value = current price plus the share of the move we expect to
        # continue. Deliberately modest: half the recent drift, decayed.
        fair = (price + 0.5 * change * signal.abs()).clip(0.01, 0.99)
        return self.frame(signal, fair)


class MeanReversion(Strategy):
    """Fade moves that are not backed by volume.

    Rationale: a large price move on thin volume is usually one impatient trader
    walking the book, not information. Those moves revert. The volume filter is
    what separates this from blindly fading news, which loses money.
    """

    name = "Mean Reversion"

    def __init__(self, window: int = 24, z_entry: float = 1.5, volume_filter: bool = True):
        super().__init__(window=window, z_entry=z_entry, volume_filter=volume_filter)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        price = panel["price"]
        ma = self.roll(panel, "price", self.window, "mean")
        sd = self.roll(panel, "price", self.window, "std").replace(0, np.nan)
        z = (price - ma) / sd

        signal = -self.squash(z, self.z_entry)

        if self.volume_filter and "volume" in panel.columns:
            vol_ma = self.roll(panel, "volume", self.window, "mean")
            quiet = (panel["volume"] <= vol_ma).astype(float)
            # Only fade moves that happened on below-average volume.
            signal = signal * (0.35 + 0.65 * quiet)

        fair = ma.clip(0.01, 0.99).fillna(price)
        return self.frame(signal, fair)


class LongshotBias(Strategy):
    """Exploit the favourite-longshot bias.

    The most robust documented anomaly in betting and prediction markets:
    low-probability contracts are systematically overpriced and high-probability
    contracts systematically underpriced, because retail flow prefers lottery
    tickets. The correction is applied in logit space and shrinks toward the
    market as expiry approaches, since the bias is largest far from resolution.

    This is the closest thing to free money in the file, and it is also the one
    most likely to be arbitraged away by the time you read this. Verify on your
    own resolved-market sample before sizing it.

    Critically, this position is **held to resolution**. A calibration edge is
    a statement about what the contract is worth at settlement, not about which
    way the price moves next. An earlier version of this strategy re-evaluated
    every bar and churned in and out; it lost money on data with a large,
    verified planted bias, because it kept exiting into the same distorted price
    it had entered at. If your edge is "the market is miscalibrated", the only
    way to collect it is to still be holding when the market resolves.
    """

    name = "Longshot Bias"

    def __init__(self, strength: float = 0.35, min_hours: float = 24.0,
                 min_edge: float = 0.02, entry_window: float = 0.5):
        super().__init__(strength=strength, min_hours=min_hours,
                         min_edge=min_edge, entry_window=entry_window)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = panel["price"].clip(0.01, 0.99)
        logit = np.log(p / (1 - p))

        hours = self.hours_to_close(panel)
        # Bias is widest far from expiry; markets sharpen up near settlement.
        decay = (hours / (hours + self.min_hours)).fillna(0.5).clip(0, 1)

        fair = 1 / (1 + np.exp(-(logit * (1 + self.strength * decay))))
        edge = fair - p

        # Only consider entering in the first `entry_window` of a market's life.
        frac = self.bars_elapsed(panel) / panel.groupby("key", sort=False)["price"].transform("size")
        eligible = (edge.abs() >= self.min_edge) & (frac <= self.entry_window)

        raw = np.sign(edge).where(eligible)
        key = panel["key"]
        # Latch the FIRST qualifying signal per market and hold it unchanged to
        # the end of the series. Merely forward-filling is not enough: if the
        # price crosses 0.5 the raw sign flips, the position reverses, and you
        # close the original trade at exactly the moment it has moved against
        # you. In testing that single detail turned a +$38k edge into a -$66k
        # loss, with a 1.4% win rate on the flipped trades.
        latched = raw.groupby(key, sort=False).transform("first")
        started = raw.notna().groupby(key, sort=False).cummax()
        signal = latched.where(started).fillna(0.0)

        return self.frame(signal, fair)


class VolumeBreakout(Strategy):
    """Enter when a price move is confirmed by a volume spike.

    The complement of MeanReversion: same move, opposite conclusion, decided by
    whether volume showed up. Running both and comparing tells you which regime
    your market universe is in.
    """

    name = "Volume Breakout"

    def __init__(self, window: int = 24, vol_mult: float = 2.0, move_ticks: float = 3.0):
        super().__init__(window=window, vol_mult=vol_mult, move_ticks=move_ticks)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        price = panel["price"]
        move = price - self.lag(panel, "price", 1)

        if "volume" in panel.columns:
            vol_ma = self.roll(panel, "volume", self.window, "mean").replace(0, np.nan)
            spike = (panel["volume"] / vol_ma).fillna(1.0)
        else:
            spike = pd.Series(1.0, index=panel.index)

        confirmed = (spike >= self.vol_mult) & (move.abs() >= self.move_ticks * 0.01)
        raw = np.sign(move).where(confirmed, 0.0)
        signal = pd.Series(raw, index=panel.index).fillna(0.0)

        fair = (price + move * 0.5 * signal.abs()).clip(0.01, 0.99)
        return self.frame(signal, fair)


class TimeDecay(Strategy):
    """Sell the tails as expiry approaches.

    Contracts trading at 3c with two hours to run are usually still overpriced
    — the residual demand is people hoping, not forecasting. Small, frequent,
    high-hit-rate, and utterly dependent on fees being low enough. Included
    partly because it is a good test of whether your friction model is honest:
    it is very profitable frictionless and marginal under `conservative`.
    """

    name = "Time Decay"

    def __init__(self, max_hours: float = 12.0, tail: float = 0.10):
        super().__init__(max_hours=max_hours, tail=tail)

    def generate(self, panel: pd.DataFrame) -> pd.DataFrame:
        p = panel["price"]
        hours = self.hours_to_close(panel)
        near = (hours <= self.max_hours) & (hours > 0)

        # Short the cheap tail, buy the expensive one.
        short_tail = near & (p <= self.tail)
        long_tail = near & (p >= 1 - self.tail)

        signal = pd.Series(0.0, index=panel.index)
        signal[short_tail] = -1.0
        signal[long_tail] = 1.0

        fair = p.copy()
        fair[short_tail] = (p[short_tail] * 0.5).clip(0.005, None)
        fair[long_tail] = (1 - (1 - p[long_tail]) * 0.5).clip(None, 0.995)
        return self.frame(signal, fair)


REGISTRY: dict[str, type[Strategy]] = {
    "buyhold": BuyAndHold,
    "momentum": Momentum,
    "meanrev": MeanReversion,
    "longshot": LongshotBias,
    "breakout": VolumeBreakout,
    "timedecay": TimeDecay,
}
