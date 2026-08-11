from quantlab.strategies.arbitrage import CrossVenueArb, DutchBook, attach_reference
from quantlab.strategies.base import Strategy
from quantlab.strategies.baseline import (
    REGISTRY as BASELINE_REGISTRY,
)
from quantlab.strategies.baseline import (
    BuyAndHold,
    LongshotBias,
    MeanReversion,
    Momentum,
    TimeDecay,
    VolumeBreakout,
)
from quantlab.strategies.fair_value import FairValueStrategy

REGISTRY: dict[str, type[Strategy]] = {
    **BASELINE_REGISTRY,
    "arb": CrossVenueArb,
    "dutchbook": DutchBook,
    "fairvalue": FairValueStrategy,
}


def build(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy '{name}'. options: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


def default_suite() -> list[Strategy]:
    """The set that fills the QUANT LAB leaderboard out of the box."""
    return [
        Momentum(),
        MeanReversion(),
        LongshotBias(),
        VolumeBreakout(),
        TimeDecay(),
        BuyAndHold(),
    ]


__all__ = [
    "REGISTRY",
    "BuyAndHold",
    "CrossVenueArb",
    "DutchBook",
    "FairValueStrategy",
    "LongshotBias",
    "MeanReversion",
    "Momentum",
    "Strategy",
    "TimeDecay",
    "VolumeBreakout",
    "attach_reference",
    "build",
    "default_suite",
]
