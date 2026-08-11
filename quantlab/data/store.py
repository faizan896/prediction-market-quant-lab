"""Local persistence.

SQLite by default so `git clone && pytest` works with zero infrastructure.
The schema is deliberately Postgres-compatible: when you outgrow SQLite, set
QUANTLAB_DB_URL and the same DDL runs on Postgres (add TimescaleDB on
price_history and nothing else changes).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.config import settings
from quantlab.types import Market, PricePoint

SCHEMA = """
CREATE TABLE IF NOT EXISTS markets (
    key            TEXT PRIMARY KEY,
    venue          TEXT NOT NULL,
    market_id      TEXT NOT NULL,
    event_id       TEXT,
    title          TEXT,
    slug           TEXT,
    yes_bid        REAL,
    yes_ask        REAL,
    last_price     REAL,
    volume         REAL DEFAULT 0,
    volume_24h     REAL DEFAULT 0,
    liquidity      REAL DEFAULT 0,
    open_interest  REAL DEFAULT 0,
    open_time      TEXT,
    close_time     TEXT,
    status         TEXT,
    resolved       INTEGER DEFAULT 0,
    resolution     REAL,
    min_tick       REAL DEFAULT 0.01,
    fee_model      TEXT,
    raw            TEXT,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_markets_venue   ON markets(venue);
CREATE INDEX IF NOT EXISTS idx_markets_event   ON markets(event_id);
CREATE INDEX IF NOT EXISTS idx_markets_volume  ON markets(volume);

CREATE TABLE IF NOT EXISTS price_history (
    venue      TEXT NOT NULL,
    market_id  TEXT NOT NULL,
    ts         TEXT NOT NULL,
    price      REAL NOT NULL,
    volume     REAL DEFAULT 0,
    yes_bid    REAL,
    yes_ask    REAL,
    PRIMARY KEY (venue, market_id, ts)
);
CREATE INDEX IF NOT EXISTS idx_ph_market ON price_history(venue, market_id, ts);

-- Cross-venue links: the same real-world question on two exchanges.
CREATE TABLE IF NOT EXISTS market_links (
    link_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    invert      INTEGER DEFAULT 0,   -- 1 if this leg's YES == the other's NO
    score       REAL,                -- match confidence 0..1
    method      TEXT,                -- 'auto' | 'manual'
    PRIMARY KEY (link_id, key)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id      TEXT PRIMARY KEY,
    strategy    TEXT NOT NULL,
    params      TEXT,
    metrics     TEXT,
    created_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------- paper trading
CREATE TABLE IF NOT EXISTS paper_runs (
    run_id           TEXT PRIMARY KEY,
    name             TEXT,
    strategy         TEXT,
    params           TEXT,
    initial_capital  REAL NOT NULL,
    cash             REAL NOT NULL,
    realized_pnl     REAL DEFAULT 0,
    fees_paid        REAL DEFAULT 0,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    status           TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS paper_fills (
    fill_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    key         TEXT NOT NULL,
    title       TEXT,
    ts          TEXT NOT NULL,
    action      TEXT NOT NULL,      -- open | close | settle
    side        TEXT NOT NULL,      -- YES | NO
    price       REAL NOT NULL,      -- fill price in YES terms
    contracts   REAL NOT NULL,
    notional    REAL NOT NULL,      -- cash actually moved
    fee         REAL DEFAULT 0,
    slippage    REAL DEFAULT 0,
    fair_prob   REAL,               -- what the model believed at fill time
    edge        REAL,               -- predicted edge at fill time
    realized_pnl REAL DEFAULT 0,    -- non-zero on close/settle
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_fills_run ON paper_fills(run_id, ts);

CREATE TABLE IF NOT EXISTS paper_positions (
    run_id        TEXT NOT NULL,
    key           TEXT NOT NULL,
    title         TEXT,
    side          TEXT NOT NULL,
    contracts     REAL NOT NULL,
    avg_price     REAL NOT NULL,    -- average YES-terms entry price
    cost_basis    REAL NOT NULL,    -- cash deployed
    fair_at_entry REAL,
    edge_at_entry REAL,
    opened_at     TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);

CREATE TABLE IF NOT EXISTS paper_equity (
    run_id     TEXT NOT NULL,
    ts         TEXT NOT NULL,
    equity     REAL NOT NULL,
    cash       REAL NOT NULL,
    exposure   REAL NOT NULL,
    open_positions INTEGER DEFAULT 0,
    PRIMARY KEY (run_id, ts)
);
"""


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


class Store:
    """Thread-safe SQLite wrapper.

    FastAPI runs synchronous route handlers in a threadpool, so a `Store` shared
    across requests gets touched from many threads. SQLite forbids that by
    default (`SQLite objects created in a thread can only be used in that same
    thread`), so we opt out of the check and serialize access with a lock
    instead. WAL mode keeps concurrent readers from blocking on the writer.

    The lock is coarse — one lock for the whole connection. That is the right
    trade here: these are millisecond queries, and correctness under concurrency
    matters far more than throughput for a research tool.
    """

    def __init__(self, path: Path | str | None = None):
        if path is None:
            settings.resolved_data_dir.mkdir(parents=True, exist_ok=True)
            path = settings.sqlite_path
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            if str(self.path) != ":memory:":
                # WAL is unavailable in-memory and on some network filesystems;
                # it is an optimization, never a requirement.
                try:
                    self.conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.DatabaseError:  # pragma: no cover
                    pass
            self.conn.execute("PRAGMA busy_timeout=5000")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # -------------------------------------------------------------- internals
    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def _query(self, sql: str, args: list | None = None) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self.conn, params=args)

    def _fetchall(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    # ---------------------------------------------------------------- writes
    def upsert_markets(self, markets: Iterable[Market]) -> int:
        rows = [
            (
                m.key, m.venue.value, m.market_id, m.event_id, m.title, m.slug,
                m.yes_bid, m.yes_ask, m.last_price, m.volume, m.volume_24h,
                m.liquidity, m.open_interest, _iso(m.open_time), _iso(m.close_time),
                m.status, int(m.resolved), m.resolution, m.min_tick, m.fee_model,
                json.dumps(m.raw, default=str), _iso(datetime.now(timezone.utc)),
            )
            for m in markets
        ]
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT INTO markets VALUES (" + ",".join("?" * 22) + ") "
                "ON CONFLICT(key) DO UPDATE SET "
                "yes_bid=excluded.yes_bid, yes_ask=excluded.yes_ask, "
                "last_price=excluded.last_price, volume=excluded.volume, "
                "volume_24h=excluded.volume_24h, liquidity=excluded.liquidity, "
                "open_interest=excluded.open_interest, status=excluded.status, "
                "resolved=excluded.resolved, resolution=excluded.resolution, "
                "raw=excluded.raw, updated_at=excluded.updated_at",
                rows,
            )
            self.conn.commit()
        return len(rows)

    def upsert_prices(self, points: Iterable[PricePoint]) -> int:
        rows = [
            (p.venue.value, p.market_id, _iso(p.ts), p.price, p.volume, p.yes_bid, p.yes_ask)
            for p in points
            if p.ts is not None
        ]
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT INTO price_history VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(venue, market_id, ts) DO UPDATE SET "
                "price=excluded.price, volume=excluded.volume, "
                "yes_bid=excluded.yes_bid, yes_ask=excluded.yes_ask",
                rows,
            )
            self.conn.commit()
        return len(rows)

    def save_link(self, link_id: str, key: str, invert: bool, score: float, method: str) -> None:
        self._exec(
            "INSERT OR REPLACE INTO market_links VALUES (?,?,?,?,?)",
            (link_id, key, int(invert), score, method),
        )

    def save_run(self, run_id: str, strategy: str, params: dict, metrics: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO backtest_runs VALUES (?,?,?,?,?)",
            (
                run_id, strategy, json.dumps(params, default=str),
                json.dumps(metrics, default=str), _iso(datetime.now(timezone.utc)),
            ),
        )

    # ----------------------------------------------------------------- reads
    def markets_df(self, venue: str | None = None, min_volume: float = 0.0) -> pd.DataFrame:
        q = "SELECT * FROM markets WHERE volume >= ?"
        args: list = [min_volume]
        if venue:
            q += " AND venue = ?"
            args.append(venue)
        return self._query(q, args)

    def panel(
        self,
        venue: str | None = None,
        keys: list[str] | None = None,
        min_points: int = 20,
    ) -> pd.DataFrame:
        """The canonical backtest input: one row per (market, timestamp).

        Columns: key, venue, market_id, title, ts, price, volume, yes_bid,
        yes_ask, resolution, close_time, min_tick, fee_model.
        """
        q = """
        SELECT m.key, p.venue, p.market_id, m.title, p.ts, p.price, p.volume,
               p.yes_bid, p.yes_ask, m.resolution, m.close_time, m.min_tick,
               m.fee_model, m.liquidity
        FROM price_history p
        JOIN markets m ON m.venue = p.venue AND m.market_id = p.market_id
        """
        args: list = []
        where = []
        if venue:
            where.append("p.venue = ?")
            args.append(venue)
        if keys:
            where.append(f"m.key IN ({','.join('?' * len(keys))})")
            args.extend(keys)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY m.key, p.ts"

        df = self._query(q, args)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, format="mixed")
        counts = df.groupby("key")["ts"].transform("size")
        return df[counts >= min_points].reset_index(drop=True)

    def links_df(self) -> pd.DataFrame:
        return self._query("SELECT * FROM market_links")

    def runs_df(self) -> pd.DataFrame:
        return self._query("SELECT * FROM backtest_runs ORDER BY created_at DESC")

    # ------------------------------------------------------- paper trading I/O
    def create_paper_run(
        self, run_id: str, name: str, strategy: str, params: dict, initial_capital: float
    ) -> None:
        now = _iso(datetime.now(timezone.utc))
        self._exec(
            "INSERT OR REPLACE INTO paper_runs "
            "(run_id, name, strategy, params, initial_capital, cash, realized_pnl, "
            " fees_paid, created_at, updated_at, status) "
            "VALUES (?,?,?,?,?,?,0,0,?,?, 'active')",
            (run_id, name, strategy, json.dumps(params, default=str),
             initial_capital, initial_capital, now, now),
        )

    def update_paper_run(
        self, run_id: str, cash: float, realized_pnl: float, fees_paid: float
    ) -> None:
        self._exec(
            "UPDATE paper_runs SET cash=?, realized_pnl=?, fees_paid=?, updated_at=? "
            "WHERE run_id=?",
            (cash, realized_pnl, fees_paid, _iso(datetime.now(timezone.utc)), run_id),
        )

    def get_paper_run(self, run_id: str) -> dict | None:
        rows = self._fetchall("SELECT * FROM paper_runs WHERE run_id=?", (run_id,))
        return rows[0] if rows else None

    def list_paper_runs(self) -> list[dict]:
        return self._fetchall("SELECT * FROM paper_runs ORDER BY created_at DESC")

    def record_fill(self, fill: dict) -> None:
        cols = ("fill_id", "run_id", "key", "title", "ts", "action", "side", "price",
                "contracts", "notional", "fee", "slippage", "fair_prob", "edge",
                "realized_pnl", "reason")
        self._exec(
            f"INSERT OR REPLACE INTO paper_fills ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            tuple(fill.get(c) for c in cols),
        )

    def fills_df(self, run_id: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM paper_fills"
        args: list = []
        if run_id:
            q += " WHERE run_id = ?"
            args.append(run_id)
        return self._query(q + " ORDER BY ts", args)

    def upsert_position(self, run_id: str, pos: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO paper_positions "
            "(run_id, key, title, side, contracts, avg_price, cost_basis, "
            " fair_at_entry, edge_at_entry, opened_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, pos["key"], pos.get("title"), pos["side"], pos["contracts"],
             pos["avg_price"], pos["cost_basis"], pos.get("fair_at_entry"),
             pos.get("edge_at_entry"), pos["opened_at"]),
        )

    def delete_position(self, run_id: str, key: str) -> None:
        self._exec(
            "DELETE FROM paper_positions WHERE run_id=? AND key=?", (run_id, key)
        )

    def positions(self, run_id: str) -> list[dict]:
        return self._fetchall("SELECT * FROM paper_positions WHERE run_id=?", (run_id,))

    def record_equity(
        self, run_id: str, ts: datetime, equity: float, cash: float,
        exposure: float, open_positions: int,
    ) -> None:
        self._exec(
            "INSERT OR REPLACE INTO paper_equity VALUES (?,?,?,?,?,?)",
            (run_id, _iso(ts), equity, cash, exposure, open_positions),
        )

    def equity_df(self, run_id: str) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM paper_equity WHERE run_id=? ORDER BY ts", [run_id]
        )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
