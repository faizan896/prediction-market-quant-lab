from quantlab.data.base import VenueClient
from quantlab.data.kalshi import KalshiClient
from quantlab.data.polymarket import PolymarketClient
from quantlab.data.store import Store
from quantlab.data.synthetic import SyntheticClient

__all__ = ["KalshiClient", "PolymarketClient", "Store", "SyntheticClient", "VenueClient"]


def get_client(venue: str) -> VenueClient:
    return {
        "polymarket": PolymarketClient,
        "kalshi": KalshiClient,
        "synthetic": SyntheticClient,
    }[venue]()
