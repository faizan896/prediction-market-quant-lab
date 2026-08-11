# Prediction Market Quant Lab

Research → Backtest → Paper Trade → Live Trade, for Polymarket and Kalshi.

One core, four surfaces: a **vectorized backtest engine**, a **mispricing
scanner** that ranks contracts by `Market → Price → Fair Value → Edge →
Confidence → EV`, a **paper-trading loop** that runs the identical model against
live quotes, and a **terminal UI** over all three.

Everything through paper trading is built and tested (122 tests). Live order
placement is the one deliberate gap — see [Live execution](#live-execution).

```
                              QUANT LAB
Strategy          Trades   Win Rate      ROI   Sharpe   Max DD   Brier
Time Decay           117      94.0%     3.3%    41.64     0.2%   0.001
Buy & Hold            50      54.0%     1.2%     0.37     4.6%   0.200
Longshot Bias         97      67.0%   -10.9%    -0.85    11.7%   0.207
Momentum             786      28.0%    -4.8%    -5.48    26.7%   0.012
Mean Reversion     1,439      35.0%    -9.6%   -14.46    77.2%   0.016
Volume Breakout      280      27.5%    -5.9%   -35.06    12.2%       —
```

Those are real numbers from `quantlab backtest --synthetic --frictions frictionless`,
not mockups. Under the `conservative` friction preset every one of them goes
negative. **That is the honest result and it is the most useful thing this repo
will tell you.** A platform that shows you a 2.11 Sharpe on your first afternoon
is lying to you; the value here is a harness that won't.

---

## Quick start

```bash
pip install -e ".[dev,api]"

quantlab backtest --synthetic        # full pipeline, no network, no keys
quantlab ingest                      # pull live markets from both venues
quantlab backfill --venue polymarket --top 100
quantlab backtest                    # leaderboard on real history
quantlab scan --min-edge 0.03        # the mispricing terminal
quantlab paper --name run-1          # one paper-trading cycle
```

The UI:

```bash
uvicorn quantlab.api.main:app --port 8000     # terminal 1 → /docs for OpenAPI
cd frontend && npm install && npm run dev     # terminal 2 → localhost:3000
```

Reads on both venues are public — no API key, no wallet, no auth. Keys are only
needed to place orders, and nothing in this repo places one.

**Deploying?** See [DEPLOY.md](DEPLOY.md). Frontend to Vercel, API to
Render/Railway via the included `Dockerfile`, $0 on free tiers — with one real
caveat about ephemeral storage and paper-trading history.

---

## What's actually built

| Layer | Status | Notes |
|---|---|---|
| Polymarket client | done | Gamma + CLOB + Data, incl. `prices-history`, order books |
| Kalshi client | done | markets, candlesticks, order books; handles the `_dollars`/`_fp` format migration |
| Normalization | done | one `Market`/`PricePoint` type, YES-probability convention throughout |
| Storage | done | SQLite by default, Postgres-compatible DDL |
| Cross-venue linking | done | token-overlap matcher with numeric-strike guard |
| Backtest engine | done | vectorized, anti-lookahead, exact binary PnL |
| Frictions | done | spread, slippage, √-impact, real Kalshi & Polymarket fee formulas |
| Sizing | done | exact binary Kelly, fractional + capped + liquidity-limited |
| Metrics | done | win rate, ROI, Sharpe, Sortino, max DD, PF, Brier, reliability curve |
| Strategies | done | 6 baselines + cross-venue arb + Dutch book + fair-value |
| Mispricing scanner | done | logit pooling, confidence, net EV, Kelly size |
| Terminal CLI | done | rich-rendered leaderboard, scanner, paper cycle |
| Paper trading | done | broker interface, portfolio accounting, risk limits, settlement, resumable |
| FastAPI backend | done | markets, scan, backtest, strategies, paper; OpenAPI at `/docs` |
| Next.js frontend | done | scanner, strategy builder + leaderboard, paper portfolio |
| Live execution | **deliberately not done** | see below |

122 tests pass, ruff clean, frontend typechecks and builds.

```bash
pytest -q && ruff check .
cd frontend && npm run typecheck && npm run build
```

---

## The three things that make this a real backtester

### 1. It cannot look ahead

Signals are shifted forward one bar **inside the engine**, after the strategy has
run. Exits fill on the bar *after* the exit decision. A strategy physically
cannot trade on the bar it used to decide, however it's written.

This is enforced by a test that is worth understanding:

```python
class NextBarOracle(Strategy):
    def generate(self, panel):
        nxt = panel.groupby("key")["price"].shift(-1)   # blatant lookahead
        return self.frame(np.sign(nxt - panel["price"]), panel["price"])
```

With `signal_lag=0` this strategy wins **100.0%** of its trades. With the default
`signal_lag=1` it drops to a coin flip. If that gap ever narrows, the engine is
leaking the future and every number it produces is fiction.

### 2. Frictions are modelled from the real fee schedules

- **Kalshi**: `fee = ceil(0.07 × C × P × (1−P))`, rounded up per order. Peaks at
  50¢, vanishes at the tails.
- **Polymarket**: `rate × min(P, 1−P) × C` on fee-enabled markets.
- **Spread**: you lift the offer to buy and hit the bid to sell. Always.
- **Impact**: `impact_coef × √(size / liquidity)` — prediction-market books are
  thin and this dominates above a few thousand dollars.

Presets: `frictionless`, `polymarket`, `kalshi`, `conservative`. Note that even
`frictionless` still pays the quoted bid/ask in your data; it removes fees and
slippage, not the spread.

### 3. Kelly is exact here, and that matters more than you'd think

Binary contracts make Kelly closed-form: `f* = (p − b) / (1 − b)`.

At `b = 0.05`, a 2-point edge is `f* = 0.021`. At `b = 0.95`, the *same* 2-point
edge is `f* = 0.40` — nineteen times larger. Sizing longshots as if they were
favourites is the standard way to blow up in this asset class. `SizingConfig`
defaults to quarter-Kelly, capped at 5% of bankroll, and never more than 10% of
resting depth.

---

## The mispricing engine

```python
from quantlab.data import PolymarketClient
from quantlab.mispricing import MispricingEngine, ExternalProbability, MarketPrior

markets = PolymarketClient().list_markets(limit=300, min_volume=50_000)

engine = MispricingEngine(signals=[
    MarketPrior(),
    ExternalProbability({"polymarket:0x…": (0.64, 0.85)}, source="polls"),
])
engine.scan(markets, min_edge=0.03, min_confidence=0.2)
```

**Pooling is done in logit space**, weighted by (signal weight × confidence).
This isn't stylistic. Averaging probabilities 0.02 and 0.20 gives 0.11, which
wildly overprices the longshot; logit pooling gives ≈0.063, which is how a
Bayesian actually updates. It also can't produce a fair value outside [0,1].

**`market_anchor` (default 0.30) is a handbrake.** `MarketPrior` already sits in
the pool at weight 3.0 and does most of the anchoring; this applies one further
pull afterwards. Set it high as well and you double-count — a legitimate 7-point
model view collapses to under a point. Keep it non-zero, though: overriding a
liquid market by 20 points is almost always a bug in your data, and this is the
last line of defence against a broken signal sizing a large position.

**Rank by `net_ev`, never by `edge`.** `edge` is the raw gap. `net_ev` subtracts
spread, fees and slippage per dollar deployed. Ranking by raw edge puts every
illiquid 1-cent contract at the top of your screen.

This distinction produces the single most useful output in the repo. Kalshi's
fee is `7% × C × p × (1−p)` — about **3.5% of notional per side at mid-book,
~7% round trip**. So:

> At mid-book prices you need roughly **7+ points of genuine edge** just to
> break even. In the tails the `p(1−p)` term collapses and fees nearly vanish.

Run `examples/research_walkthrough.py` and watch `net_ev` improve from −4.9% on
a 57¢ contract to −1.3% on a 12¢ one for exactly this reason. If you are hunting
small edges, hunt them in the tails.

### Adding your own model

Write one class. Nothing downstream changes — pooling, edge, Kelly, backtesting
all just work:

```python
from quantlab.mispricing import Signal
from quantlab.types import FairEstimate

class FiveThirtyEight(Signal):
    name = "polls"
    default_weight = 2.5

    def estimate(self, market, **ctx):
        p = my_poll_model(market.title)
        if p is None:
            return None          # abstaining is a first-class answer
        return FairEstimate(self.name, p, confidence=0.8)
```

A signal that always has an opinion drowns out the ones that only speak when they
have something to say — which is where the edge lives. Every signal must be able
to return `None`.

---

## Paper trading

One `step()` is one decision cycle:

```
settle resolved positions → mark the book → scan → apply risk limits
                          → submit intents → persist fills, positions, equity
```

It runs the **same `MispricingEngine` the live scanner uses**, so the paper
record measures the thing you would actually trade. `Broker` is the only
execution interface the loop knows about, so promoting to live is a one-line
swap with zero changes to decision logic — that is the entire point of having a
paper stage.

```python
from quantlab.data import Store
from quantlab.execution import PaperTrader, RiskLimits

trader = PaperTrader(name="run-1", store=Store(),
                     limits=RiskLimits(min_net_ev=0.02, max_positions=40))
report = trader.step(markets)     # schedule hourly; these markets are slow
print(report.describe())
```

**The number to watch is `edge_capture`, not P&L.** It is realized return ÷
predicted edge. A strategy can have a genuine edge and still lose money if it
only captures 30% of it after costs, and you cannot see that in a win rate.
Ignore the equity curve until roughly 200 settled trades.

Every risk limit exists because of a specific failure mode: `max_gross_exposure`
stops a scanner bug that flags 400 markets from deploying the whole bankroll;
`min_net_ev` stops anything that doesn't clear its own costs; `min_price` /
`max_price` keep you out of tails where the tick grid and settlement risk
dominate any modelled edge.

Exits are on **thesis, not price**. These are convergence trades, so a drawdown
is not by itself information — the position closes when the edge is gone, when
the market settles, or at a max-hold horizon.

### Live execution

`LiveBroker` raises `NotImplementedError` on purpose. Both integrations are
straightforward — Polymarket via `py-clob-client` (wallet-derived credentials,
USDC on Polygon, token allowances), Kalshi via RSA-PSS request signing — and
neither is hard. The reason it isn't shipped half-done is that an execution bug
loses real money, whereas every other bug in this repo only produces a wrong
number. Write it deliberately, with its own tests, once the paper record
justifies it.

---

## API and UI

```
GET  /markets                 stored or live, both venues
GET  /markets/{v}/{id}/history|book
POST /scan                    the mispricing terminal
GET  /strategies              every strategy + its tunable params, introspected
POST /backtest                leaderboard, equity curves, walk-forward, calibration
GET  /paper, POST /paper      list / create runs
GET  /paper/{id}              portfolio, positions, fills, equity
POST /paper/{id}/step         run one cycle
```

The service layer (`quantlab/api/service.py`) has no FastAPI in it, so the same
functions are callable from a notebook or a cron job and testable without a
server. `/strategies` introspects the registry, so the frontend's strategy
builder is generated rather than hard-coded — add a strategy and it appears in
the UI with its parameters.

Three screens: **Scanner** (toggle and reweight signals, per-signal attribution
on click), **Backtest** (strategy builder, leaderboard, equity curves,
walk-forward, calibration table), **Paper** (portfolio, positions, blotter, and
a rejection log that is usually the most informative panel on the page).

Every page works with zero setup — scanner and backtest default to synthetic
data, so nothing is an empty shell before you've run `quantlab ingest`. There is
no client-side finance math anywhere: if the UI computed ROI differently from
the backtester you'd have two answers and no way to tell which was right.

---

## A worked failure, because it's the most instructive thing here

`LongshotBias` exploits the best-documented anomaly in the space: cheap contracts
are systematically overpriced. The synthetic generator plants exactly this bias,
verifiably — contracts trading at 14.5¢ resolve YES only **7.4%** of the time.
A 7-point edge, planted on purpose, sitting right there.

The first implementation **lost money on it**. Breaking down the trades:

```
                n    win rate      pnl
settled       283       0.895   +$37.8k     ← held to resolution
unsettled     347       0.014   −$66.3k     ← closed early
```

A 1.4% win rate isn't a strategy being wrong, it's a bug. Two of them:

1. **The engine was force-closing positions** when price drifted outside the
   tradable band. Nobody liquidates you because a contract touched 1.5¢. The
   price filter now gates *entries* only.
2. **The signal re-signed itself.** As price crossed 0.5, the raw sign flipped
   and reversed the position — closing each trade at precisely the moment it had
   moved against it. Forward-filling wasn't enough; the first qualifying signal
   now latches for the life of the market.

The underlying lesson generalizes: **a calibration edge is a claim about
settlement value, not about the next price move. The only way to collect it is
to still be holding when the market resolves.** Both fixes are in, both are
tested, and both are commented at the site of the bug.

Three more bugs of the same family turned up while wiring the API, all found by
end-to-end runs rather than unit tests, and all now regression-tested:

- **`Store` wasn't thread-safe.** FastAPI runs sync handlers in a threadpool;
  SQLite forbids cross-thread use. Now `check_same_thread=False` behind a lock,
  with WAL enabled.
- **`PaperTrader.resume()` discarded the run's risk limits**, so every cycle
  after creation silently traded default settings instead of the configuration
  being evaluated. Limits *and* the model config are now persisted and restored.
- **The closed-trade blotter lived only in memory.** Since the trader resumes on
  every cycle, `win_rate`, `brier` and `edge_capture` were permanently blank —
  the exact metrics a paper record exists to produce. Now rebuilt by replaying
  persisted fills.

None of these would crash. All three would have quietly produced a paper record
that meant nothing.

---

## Layout

```
quantlab/
  types.py            venue-agnostic Market / PricePoint / OrderBook / FairEstimate
  config.py           env-driven settings
  data/
    polymarket.py     Gamma + CLOB + Data
    kalshi.py         trade-api/v2, handles _dollars & _fp formats
    synthetic.py      Brownian-bridge generator with plantable biases
    store.py          SQLite now, Postgres-shaped DDL
    ingest.py         orchestration + cross-venue linking
  backtest/
    engine.py         signal → positions → segments → vectorized pricing
    frictions.py      spread / slippage / impact / real fee formulas
    sizing.py         exact binary Kelly
    metrics.py        performance + calibration
  strategies/
    baseline.py       momentum, mean reversion, longshot, breakout, time decay
    arbitrage.py      cross-venue + Dutch book
    fair_value.py     the scanner's logic, backtestable
  mispricing/
    signals.py        pluggable fair-value sources
    engine.py         logit pooling, confidence, net EV, ranking
  execution/
    broker.py         Broker interface, PaperBroker, LiveBroker seam
    portfolio.py      binary-contract accounting, edge capture
    trader.py         the paper loop + risk limits
  api/
    schemas.py        wire format (deliberately not the domain types)
    service.py        all the logic, framework-free
    main.py           thin FastAPI routes
  cli.py              the terminal
frontend/
  app/                scanner · backtest · paper
  lib/api.ts          typed client mirroring schemas.py
Dockerfile            API image (Railway / Render / Fly / plain Docker)
render.yaml           Render blueprint
railway.json          Railway config
frontend/vercel.json  Vercel config (set Root Directory to `frontend`)
.github/workflows/    CI: pytest + ruff + frontend build + docker build
```

The scanner and the `FairValueStrategy` implement the *same* model — one
per-market, one vectorized over a panel. If your live scanner and your backtest
run different code, your backtest is measuring something you will never trade.

---

## Roadmap

**Next, in order.**

1. **Run the paper loop for a month.** Schedule `quantlab paper --run <id>`
   hourly against live venues and let it accumulate. Do not skip this. The whole
   architecture exists to make the paper record meaningful, and the only thing
   that makes it meaningful is time.
2. **Real external signals.** Polls, weather (NWS — Kalshi's temperature markets
   settle on the Climatological Report, so the model input is public), sportsbook
   lines, economic releases. This is where actual edge lives. Each one is a
   single `Signal` subclass; nothing downstream changes.
3. **Live execution.** Only once edge capture is stable and positive across a few
   hundred settled trades.
4. **Only then, ML.** When you have (a) a paper record, (b) a calibration curve,
   and (c) a baseline that beats buy-and-hold after costs. If a 20-line momentum
   rule can't make money, gradient boosting on the same features will just
   overfit faster.

**Infrastructure, when you outgrow the current setup:** the DDL is already
Postgres-shaped (add TimescaleDB on `price_history`); Redis for scan caching;
WebSockets instead of REST polling for live prices.

**Before you trade anything this produces, run `quantlab walkforward`.** A Sharpe
of 2.1 in-sample and 0.3 out-of-sample is not a strategy, it's a curve fit.

---

## Honest caveats

- **Cross-venue links are title-similarity guesses.** "Will Trump win?" on
  Polymarket is `PRES-2028-DEM` on Kalshi. A persistent price gap between two
  "matched" contracts is usually a resolution-criteria difference, not free
  money. Read both rule sets. `quantlab link` proposes; it does not confirm.
- **Published arbitrage is usually gone, fake, or sized at $50.** The value of
  the arb module is mostly in systematically ruling out the fake ones.
- **The longshot-bias parameters are defaults, not findings.** Fit `strength` on
  your own resolved-market sample; it varies a lot by category (sports markets
  are near-efficient, political longshots are not).
- **Compounding is approximated.** Trade sizes scale by bankroll at open;
  properly sequencing overlapping positions needs a portfolio-level accounting
  layer. The error direction is conservative (it shrinks size after drawdowns).
- **Nothing here is financial advice**, and prediction-market trading is
  restricted or prohibited in many jurisdictions. Check yours.

---

## Verified against live APIs

All endpoints and response shapes were captured from the live APIs on
2026-08-10, and the parser tests run against those exact payloads:

| Endpoint | Verified |
|---|---|
| `gamma-api.polymarket.com/markets` | ✅ incl. stringified `outcomes` / `clobTokenIds` |
| `clob.polymarket.com/prices-history` | ✅ `{t, p}` series |
| `api.elections.kalshi.com/trade-api/v2/markets` | ✅ `_dollars` / `_fp` fields |
| `.../series/{s}/markets/{m}/candlesticks` | ✅ nested `price.close_dollars` |

Kalshi has migrated prices to decimal-dollar strings and sizes to fixed-point
strings; the client accepts both that and the legacy integer-cent format, since
older responses still appear on some routes.
