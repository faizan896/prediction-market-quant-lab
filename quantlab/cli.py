"""QUANT LAB terminal.

    quantlab ingest                pull markets from both venues
    quantlab backfill --venue ...  pull price history for the top markets
    quantlab link                  propose cross-venue market pairs
    quantlab scan                  the mispricing terminal
    quantlab backtest              the strategy leaderboard
    quantlab walkforward           out-of-sample check on one strategy
    quantlab demo                  full pipeline on synthetic data, no network
"""

from __future__ import annotations

import math

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from quantlab import strategies as S
from quantlab.backtest import PRESETS, BacktestConfig, compare, walk_forward
from quantlab.data import KalshiClient, PolymarketClient, Store, SyntheticClient
from quantlab.data import ingest as ing
from quantlab.mispricing import MispricingEngine

app = typer.Typer(add_completion=False, help="Prediction Market Quant Lab")
console = Console()

ACCENT = "bold cyan"
GOOD = "bold green"
BAD = "bold red"


def banner(subtitle: str = "") -> None:
    art = Text("QUANT LAB", style="bold cyan")
    sub = Text(subtitle or "prediction-market research terminal", style="dim")
    console.print(Panel(Text("\n").join([art, sub]), border_style="cyan", expand=False))


def _pct(x: float, digits: int = 1) -> str:
    return "—" if x is None or math.isnan(x) else f"{x * 100:.{digits}f}%"


def _num(x: float, fmt: str = "{:.3f}") -> str:
    return "—" if x is None or math.isnan(x) else fmt.format(x)


def _color(value: float, text: str) -> Text:
    return Text(text, style=GOOD if value > 0 else (BAD if value < 0 else "white"))


# --------------------------------------------------------------------------- #
@app.command()
def ingest(
    venues: str = typer.Option("polymarket,kalshi", help="comma-separated"),
    limit: int = typer.Option(500, help="markets per venue"),
    min_volume: float = typer.Option(10_000, help="skip markets below this volume"),
):
    """Pull the current market universe into the local store."""
    banner("ingest")
    store = Store()
    counts = ing.ingest_markets(
        store, tuple(v.strip() for v in venues.split(",")), limit, min_volume
    )
    t = Table("venue", "markets stored", box=None)
    for k, v in counts.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def backfill(
    venue: str = typer.Option("polymarket"),
    top: int = typer.Option(50, help="backfill the N highest-volume markets"),
    fidelity: int = typer.Option(60, help="bar width in minutes"),
):
    """Backfill price history. Run `ingest` first."""
    banner(f"backfill · {venue}")
    store = Store()
    ids = ing.markets_for_backfill(store, venue, top)
    if not ids:
        console.print("[yellow]no markets found — run `quantlab ingest` first[/]")
        raise typer.Exit(1)
    with console.status(f"fetching {len(ids)} markets…"):
        rows = ing.ingest_history(store, venue, ids, fidelity)
    console.print(f"[green]{rows:,}[/] price rows stored for {len(ids)} markets")


@app.command()
def link(
    min_score: float = typer.Option(0.55),
    save: bool = typer.Option(False, "--save", help="persist the proposed links"),
    top: int = typer.Option(25),
):
    """Propose Polymarket <-> Kalshi contract pairs by title similarity."""
    banner("cross-venue linking")
    store = Store()
    cands = ing.find_links(store, min_score=min_score)
    if not cands:
        console.print("[yellow]no candidates. ingest both venues first.[/]")
        raise typer.Exit()

    t = Table(title="link candidates", header_style=ACCENT)
    t.add_column("score", justify="right")
    t.add_column("polymarket", max_width=42)
    t.add_column("kalshi", max_width=42)
    t.add_column("poly", justify="right")
    t.add_column("kalshi", justify="right")
    t.add_column("gap", justify="right")
    for c in cands[:top]:
        gap = c.gap
        t.add_row(
            f"{c.score:.2f}", c.poly_title, c.kalshi_title,
            f"{c.poly_price:.3f}" if c.poly_price else "—",
            f"{c.kalshi_price:.3f}" if c.kalshi_price else "—",
            _color(gap or 0, f"{gap:+.3f}") if gap is not None else Text("—"),
        )
    console.print(t)
    console.print(
        "\n[dim]Titles matching is not the same as resolution criteria matching. "
        "Read both rule sets before trading any of these.[/]"
    )
    if save:
        n = ing.save_links(store, cands)
        console.print(f"[green]saved {n} links[/]")


# --------------------------------------------------------------------------- #
@app.command()
def scan(
    venue: str = typer.Option("polymarket,kalshi", help="comma-separated"),
    limit: int = typer.Option(300, help="markets to pull per venue"),
    min_edge: float = typer.Option(0.03),
    min_net_ev: float = typer.Option(0.0),
    min_confidence: float = typer.Option(0.15),
    min_volume: float = typer.Option(50_000),
    bankroll: float = typer.Option(100_000.0),
    top: int = typer.Option(20),
    books: bool = typer.Option(False, "--books", help="fetch order books (slower, better)"),
):
    """The mispricing terminal: Market -> Price -> Fair -> Edge -> Conf -> EV."""
    banner("mispricing scanner")
    markets = []
    with console.status("fetching markets…"):
        for v in (x.strip() for x in venue.split(",")):
            client = PolymarketClient() if v == "polymarket" else KalshiClient()
            try:
                markets += client.list_markets(limit=limit, min_volume=min_volume)
            finally:
                client.close()

    book_map = {}
    if books:
        with console.status(f"fetching {len(markets)} order books…"):
            for m in markets:
                client = PolymarketClient() if m.venue.value == "polymarket" else KalshiClient()
                try:
                    ob = client.order_book(m.market_id)
                    if ob:
                        book_map[m.key] = ob
                except Exception:  # noqa: BLE001, S110 - one bad book must not halt the scan
                    pass
                finally:
                    client.close()

    engine = MispricingEngine(bankroll=bankroll)
    df = engine.scan(
        markets, books=book_map,
        min_edge=min_edge, min_net_ev=min_net_ev, min_confidence=min_confidence,
    )

    if df.empty:
        console.print(
            f"[yellow]Nothing clears the filters across {len(markets)} markets.[/]\n"
            "[dim]That is the correct and most common answer. Loosen --min-edge "
            "to see the near-misses.[/]"
        )
        raise typer.Exit()

    t = Table(title=f"mispricing scan · {len(markets)} markets", header_style=ACCENT)
    t.add_column("market", max_width=44)
    t.add_column("venue", style="dim")
    t.add_column("price", justify="right")
    t.add_column("fair", justify="right")
    t.add_column("edge", justify="right")
    t.add_column("side", justify="center")
    t.add_column("conf", justify="right")
    t.add_column("net EV", justify="right")
    t.add_column("size", justify="right")

    for r in df.head(top).itertuples():
        t.add_row(
            r.title,
            r.venue[:4],
            f"{r.price:.3f}",
            f"{r.fair_value:.3f}",
            _color(r.edge, f"{r.edge:+.3f}"),
            Text(r.side, style=GOOD if r.side == "YES" else BAD),
            f"{r.confidence:.2f}",
            _color(r.net_ev, f"{r.net_ev * 100:+.1f}%"),
            f"${r.suggested_notional:,.0f}",
        )
    console.print(t)
    console.print(
        "\n[dim]net EV is per dollar deployed, after spread, fees and slippage. "
        "Anything under +2% is noise.[/]"
    )


# --------------------------------------------------------------------------- #
@app.command()
def backtest(
    venue: str = typer.Option("", help="filter panel by venue; empty = all stored"),
    frictions: str = typer.Option("conservative", help=f"one of {sorted(PRESETS)}"),
    capital: float = typer.Option(100_000.0),
    kelly: float = typer.Option(0.25, help="Kelly multiplier"),
    synthetic: bool = typer.Option(False, "--synthetic", help="run on generated data"),
    detail: str = typer.Option("", help="show the trade blotter for one strategy"),
):
    """Run every strategy and print the leaderboard."""
    banner("backtest")
    panel = _load_panel(venue, synthetic)
    if panel is None:
        raise typer.Exit(1)

    cfg = BacktestConfig(initial_capital=capital, frictions=PRESETS[frictions])
    cfg.sizing.kelly_multiplier = kelly

    with console.status(f"backtesting on {panel['key'].nunique()} markets…"):
        board, results = compare(panel, S.default_suite(), cfg)

    _print_board(board, panel, frictions)

    if detail:
        res = next((r for k, r in results.items() if detail.lower() in k.lower()), None)
        if res:
            _print_blotter(res)


@app.command()
def walkforward(
    strategy: str = typer.Option("momentum", help=f"one of {sorted(S.REGISTRY)}"),
    folds: int = typer.Option(4),
    frictions: str = typer.Option("conservative"),
    venue: str = typer.Option(""),
    synthetic: bool = typer.Option(False, "--synthetic"),
):
    """Chronological out-of-sample test. Believe this, not the in-sample number."""
    banner(f"walk-forward · {strategy}")
    panel = _load_panel(venue, synthetic)
    if panel is None:
        raise typer.Exit(1)

    cfg = BacktestConfig(frictions=PRESETS[frictions])
    df = walk_forward(panel, lambda: S.build(strategy), folds, cfg)
    if df.empty:
        console.print("[yellow]not enough data to split[/]")
        raise typer.Exit()

    t = Table(title=f"{strategy} · {folds} folds", header_style=ACCENT)
    for c in ("fold", "trades", "win rate", "ROI", "Sharpe", "max DD"):
        t.add_column(c, justify="right")
    for r in df.itertuples():
        t.add_row(
            str(r.fold), str(r.trades), _pct(r.win_rate),
            _color(r.roi, _pct(r.roi)), f"{r.sharpe:.2f}", _pct(r.max_dd),
        )
    console.print(t)

    sharpes = df["sharpe"]
    spread = sharpes.max() - sharpes.min()
    verdict = (
        "[green]stable across folds[/]" if spread < 1.0
        else "[yellow]unstable — Sharpe swings by "
             f"{spread:.1f} across folds, treat as a curve fit[/]"
    )
    console.print(f"\n{verdict}")


@app.command()
def paper(
    run: str = typer.Option("", help="resume an existing run id; blank starts a new one"),
    name: str = typer.Option("paper-1"),
    venue: str = typer.Option("polymarket,kalshi"),
    limit: int = typer.Option(300),
    min_volume: float = typer.Option(50_000),
    capital: float = typer.Option(100_000.0),
    min_net_ev: float = typer.Option(0.02),
    show: bool = typer.Option(False, "--show", help="report only, place no orders"),
):
    """Run one paper-trading cycle. Schedule it hourly and let it accumulate."""
    from quantlab.execution import PaperTrader, RiskLimits

    banner("paper trading")
    store = Store()

    if show and run:
        _print_paper_run(store, run)
        raise typer.Exit()

    markets = []
    with console.status("fetching markets…"):
        for v in (x.strip() for x in venue.split(",")):
            client = PolymarketClient() if v == "polymarket" else KalshiClient()
            try:
                markets += client.list_markets(limit=limit, min_volume=min_volume)
            finally:
                client.close()

    limits = RiskLimits(min_net_ev=min_net_ev)
    trader = (
        PaperTrader.resume(run, store, limits=limits) if run
        else PaperTrader(name=name, limits=limits, initial_capital=capital, store=store)
    )

    report = trader.step(markets)
    console.print(f"[bold]{report.describe()}[/]")
    console.print(f"[dim]run id: {trader.run_id}[/]")

    for label, fills, style in (
        ("opened", report.opened, GOOD), ("closed", report.closed, "yellow"),
        ("settled", report.settled, ACCENT),
    ):
        if not fills:
            continue
        t = Table(title=label, header_style=ACCENT, box=None)
        for c in ("market", "side", "price", "contracts", "notional", "pnl", "why"):
            t.add_column(c, justify="right" if c != "market" else "left")
        for f in fills:
            t.add_row(
                f.title[:38] or f.key[:38], Text(f.side, style=style),
                f"{f.price:.3f}", f"{f.contracts:,.0f}", f"${f.notional:,.0f}",
                _color(f.realized_pnl, f"${f.realized_pnl:,.0f}"), f.reason[:34],
            )
        console.print(t)

    _print_paper_run(store, trader.run_id, summary=report.summary)


def _print_paper_run(store, run_id: str, summary: dict | None = None) -> None:
    row = store.get_paper_run(run_id)
    if row is None:
        console.print(f"[red]no paper run '{run_id}'[/]")
        return
    s = summary or {}
    t = Table(title=f"portfolio · {run_id}", header_style=ACCENT, box=None)
    t.add_column("metric")
    t.add_column("value", justify="right")
    equity = s.get("equity", row["cash"])
    ret = (equity - row["initial_capital"]) / row["initial_capital"]
    rows = [
        ("equity", f"${equity:,.0f}"),
        ("total return", _pct(ret)),
        ("cash", f"${row['cash']:,.0f}"),
        ("exposure", f"${s.get('exposure', 0):,.0f}"),
        ("realized pnl", f"${row['realized_pnl']:,.0f}"),
        ("fees + slippage", f"${row['fees_paid']:,.0f}"),
        ("open positions", str(s.get("open_positions", "—"))),
        ("closed trades", str(s.get("closed_trades", "—"))),
        ("win rate", _pct(s.get("win_rate", float("nan")))),
        ("edge capture", _num(s.get("edge_capture", float("nan")), "{:.2f}")),
        ("brier (settled)", _num(s.get("brier", float("nan")))),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)
    console.print(
        "\n[dim]Edge capture is the number that matters. Below ~0.5 the model has "
        "an edge it cannot collect. Ignore the equity curve until ~200 settled trades.[/]"
    )


@app.command()
def demo(
    markets: int = typer.Option(120),
    steps: int = typer.Option(180),
    frictions: str = typer.Option("conservative"),
):
    """End-to-end run on synthetic data. No network, no API keys, no waiting."""
    banner("demo · synthetic data")
    backtest(venue="", frictions=frictions, capital=100_000.0, kelly=0.25,
             synthetic=True, detail="")


# --------------------------------------------------------------------------- #
def _load_panel(venue: str, synthetic: bool):
    if synthetic:
        gen = SyntheticClient(seed=7)
        ms, pts = gen.generate(
            n_markets=120, n_steps=180, bias_fn=SyntheticClient.longshot_bias(0.04)
        )
        store = Store(path=":memory:")
        store.upsert_markets(ms)
        store.upsert_prices(pts)
        return store.panel()

    store = Store()
    panel = store.panel(venue=venue or None)
    if panel.empty:
        console.print(
            "[yellow]No price history stored.[/]\n"
            "[dim]Run `quantlab ingest` then `quantlab backfill`, "
            "or try `quantlab backtest --synthetic`.[/]"
        )
        return None
    return panel


def _print_board(board, panel, frictions: str) -> None:
    t = Table(title="STRATEGY LEADERBOARD", header_style=ACCENT, title_style="bold cyan")
    t.add_column("Strategy")
    for c in ("Trades", "Win Rate", "ROI", "Sharpe", "Max DD", "PF", "Brier", "Edge Cap"):
        t.add_column(c, justify="right")

    for r in board.itertuples():
        t.add_row(
            r.strategy,
            f"{r.trades:,}",
            _pct(r.win_rate),
            _color(r.roi, _pct(r.roi)),
            Text(f"{r.sharpe:.2f}", style=GOOD if r.sharpe > 1 else "white"),
            _pct(r.max_dd),
            f"{r.profit_factor:.2f}",
            _num(r.brier),
            _num(r.edge_capture, "{:.2f}"),
        )
    console.print(t)
    console.print(
        f"[dim]{panel['key'].nunique()} markets · {len(panel):,} bars · "
        f"frictions={frictions} · 1-bar signal lag enforced[/]"
    )
    console.print(
        "[dim]Edge Cap = realized return / predicted edge. Below ~0.5 means "
        "costs are eating the signal.[/]"
    )


def _print_blotter(res, n: int = 15) -> None:
    tr = res.trades
    if tr.empty:
        return
    t = Table(title=f"{res.strategy} · largest trades", header_style=ACCENT)
    for c in ("market", "side", "entry", "exit", "edge", "notional", "pnl", "hrs"):
        t.add_column(c, justify="right" if c != "market" else "left")
    for r in tr.reindex(tr["pnl"].abs().sort_values(ascending=False).index).head(n).itertuples():
        t.add_row(
            str(r.key)[:34], r.side, f"{r.entry_price:.3f}", f"{r.exit_price:.3f}",
            f"{r.edge:+.3f}", f"${r.notional:,.0f}",
            _color(r.pnl, f"${r.pnl:,.0f}"), f"{r.hold_hours:.0f}",
        )
    console.print(t)
    m = res.metrics
    console.print(
        f"[dim]fees ${m.total_fees:,.0f} · slippage ${m.total_slippage:,.0f} · "
        f"final equity ${m.final_equity:,.0f}[/]"
    )
    if m.reliability:
        rt = Table(title="calibration", header_style=ACCENT, box=None)
        for c in ("bucket", "n", "predicted", "actual", "gap"):
            rt.add_column(c, justify="right")
        for row in m.reliability:
            rt.add_row(
                row["bin"], str(row["n"]), f"{row['predicted']:.2f}",
                f"{row['actual']:.2f}", _color(-abs(row["gap"]), f"{row['gap']:+.2f}"),
            )
        console.print(rt)


if __name__ == "__main__":
    app()
