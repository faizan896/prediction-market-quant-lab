"""FastAPI app.

    uvicorn quantlab.api.main:app --reload --port 8000
    open http://localhost:8000/docs

Routes are thin: parse, delegate to `service`, return. Anything with logic in it
belongs in the service layer where it can be tested without a client.

Long scans and backtests run in a threadpool (FastAPI does this automatically
for `def` handlers, which is why they are not `async def` — the work is
CPU/IO-bound and synchronous, and marking it async would block the event loop).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from quantlab import __version__
from quantlab.api import schemas as sc
from quantlab.api import service as svc
from quantlab.config import settings
from quantlab.data import KalshiClient, PolymarketClient, Store

log = logging.getLogger("quantlab.api")

_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Deliberately does NOT open the store on startup. Creating it eagerly here
    # would run before `dependency_overrides` is applied, so tests (and any
    # embedding app) would get a connection to the default on-disk database
    # instead of the one they configured.
    if settings.storage_is_ephemeral:
        log.warning(
            "Storage looks ephemeral (%s). Scanning and backtesting are stateless "
            "and unaffected, but paper-trading fills and positions will be lost on "
            "redeploy. Mount a volume and set QUANTLAB_DATA_DIR to it.",
            settings.sqlite_path,
        )
    yield
    global _store
    if _store is not None:
        _store.close()
        _store = None


app = FastAPI(
    title="Prediction Market Quant Lab",
    version=__version__,
    description=(
        "Research -> Backtest -> Paper Trade -> Live Trade.\n\n"
        "Market data on both venues is public; no credentials are required for "
        "anything exposed here. Nothing in this API can place a real order."
    ),
    lifespan=lifespan,
)

# Origins are configurable because the deployed frontend lives on a different
# domain from the API. Set QUANTLAB_CORS_ORIGINS to your Vercel URL in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_origin_regex=r"https://.*\.vercel\.app",  # preview deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "ephemeral_storage": settings.storage_is_ephemeral,
    }


# ------------------------------------------------------------------- markets
@app.get("/markets", response_model=list[sc.MarketOut], tags=["markets"])
def markets(
    venue: str | None = Query(None, pattern="^(polymarket|kalshi)$"),
    limit: int = Query(200, ge=1, le=2000),
    min_volume: float = 0,
    live: bool = Query(False, description="Bypass the store and hit the venues"),
    store: Store = Depends(get_store),
):
    """Markets from the local store, or straight from the venues with `live=true`."""
    if live:
        venues = [venue] if venue else ["polymarket", "kalshi"]
        return [svc.market_out(m) for m in svc.fetch_markets(venues, limit, min_volume)]
    return svc.stored_markets(store, venue, min_volume, limit)


@app.get("/markets/{venue}/{market_id:path}/history",
         response_model=list[sc.PricePointOut], tags=["markets"])
def history(venue: str, market_id: str, fidelity: int = 60):
    """Price history straight from the venue."""
    if venue not in ("polymarket", "kalshi"):
        raise HTTPException(404, f"unknown venue '{venue}'")
    client = PolymarketClient() if venue == "polymarket" else KalshiClient()
    try:
        pts = (
            client.price_history(market_id, fidelity=fidelity) if venue == "polymarket"
            else client.price_history(market_id, period_interval=fidelity)
        )
    except Exception as exc:
        raise HTTPException(502, f"venue error: {exc}") from exc
    finally:
        client.close()
    return [
        sc.PricePointOut(ts=p.ts.isoformat(), price=p.price, volume=p.volume)
        for p in pts if p.ts
    ]


@app.get("/markets/{venue}/{market_id:path}/book",
         response_model=sc.OrderBookOut, tags=["markets"])
def book(venue: str, market_id: str):
    if venue not in ("polymarket", "kalshi"):
        raise HTTPException(404, f"unknown venue '{venue}'")
    client = PolymarketClient() if venue == "polymarket" else KalshiClient()
    try:
        ob = client.order_book(market_id)
    except Exception as exc:
        raise HTTPException(502, f"venue error: {exc}") from exc
    finally:
        client.close()
    if ob is None:
        raise HTTPException(404, "no order book")
    return sc.OrderBookOut(
        key=f"{venue}:{market_id}",
        bids=[[b.price, b.size] for b in ob.bids[:20]],
        asks=[[a.price, a.size] for a in ob.asks[:20]],
        best_bid=ob.best_bid, best_ask=ob.best_ask, microprice=ob.microprice(),
    )


# ------------------------------------------------------------------- scanner
@app.post("/scan", response_model=sc.ScanResponse, tags=["scanner"])
def scan(req: sc.ScanRequest):
    """The mispricing terminal.

    Rank by `net_ev`, not `edge`. An empty result is the normal outcome and
    means the filters are doing their job.
    """
    try:
        return svc.run_scan(req)
    except RuntimeError as exc:
        raise HTTPException(502, f"venue error: {exc}") from exc


# ------------------------------------------------------------------ backtest
@app.get("/strategies", tags=["backtest"])
def strategies() -> list[dict]:
    """Every registered strategy and its tunable parameters."""
    return svc.list_strategies()


@app.post("/backtest", response_model=sc.BacktestResponse, tags=["backtest"])
def backtest(req: sc.BacktestRequest, store: Store = Depends(get_store)):
    return svc.run_backtests(req, store)


@app.get("/backtest/runs", tags=["backtest"])
def backtest_runs(store: Store = Depends(get_store)) -> list[dict]:
    df = store.runs_df()
    return df.to_dict("records") if not df.empty else []


# --------------------------------------------------------------------- paper
@app.get("/paper", response_model=list[sc.PaperSummary], tags=["paper"])
def paper_runs(store: Store = Depends(get_store)):
    return svc.list_paper_runs(store)


@app.post("/paper", response_model=sc.PaperSummary, status_code=201, tags=["paper"])
def create_paper(cfg: sc.PaperRunCreate, store: Store = Depends(get_store)):
    run_id = svc.create_paper_run(cfg, store)
    return svc.paper_summary(run_id, store)


@app.get("/paper/{run_id}", response_model=sc.PaperDetail, tags=["paper"])
def paper_run(run_id: str, store: Store = Depends(get_store)):
    try:
        return svc.paper_detail(run_id, store)
    except KeyError as exc:
        raise HTTPException(404, f"no paper run '{run_id}'") from exc


@app.post("/paper/{run_id}/step", response_model=sc.StepResponse, tags=["paper"])
def step(run_id: str, req: sc.PaperStepRequest, store: Store = Depends(get_store)):
    """Run one decision cycle: settle, exit, scan, enter.

    Idempotent in spirit but not in fact — calling it twice runs two cycles.
    Schedule it hourly rather than hammering it.
    """
    try:
        return svc.paper_step(run_id, req, store)
    except KeyError as exc:
        raise HTTPException(404, f"no paper run '{run_id}'") from exc
    except RuntimeError as exc:
        raise HTTPException(502, f"venue error: {exc}") from exc


def run() -> None:  # pragma: no cover - convenience entry point
    import os

    import uvicorn

    # PaaS platforms inject $PORT and require binding 0.0.0.0; locally we stay
    # on loopback so the dev server isn't exposed to the network.
    port = int(os.environ.get("PORT", settings.port))
    on_paas = bool(os.environ.get("PORT"))
    uvicorn.run(
        "quantlab.api.main:app",
        host="0.0.0.0" if on_paas else "127.0.0.1",
        port=port,
        reload=not on_paas,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
