"""Runtime configuration. Everything overridable via env vars or a .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="QUANTLAB_", extra="ignore")

    # --- storage -----------------------------------------------------------
    # SQLite by default so the repo runs with zero infrastructure.
    # Set QUANTLAB_DB_URL=postgresql://... once you outgrow it.
    data_dir: Path = Path("./data")
    db_url: str = ""  # empty -> sqlite at data_dir/quantlab.db

    # --- venue endpoints (verified live 2026-08-10) ------------------------
    polymarket_gamma: str = "https://gamma-api.polymarket.com"
    polymarket_clob: str = "https://clob.polymarket.com"
    polymarket_data: str = "https://data-api.polymarket.com"
    kalshi_api: str = "https://api.elections.kalshi.com/trade-api/v2"

    # Kalshi reads are public; keys are only needed for trading.
    kalshi_key_id: str = ""
    kalshi_private_key_path: str = ""

    # --- http --------------------------------------------------------------
    # These must multiply out to LESS than the deployment's request timeout.
    # The original 30s × 3 retries = 90s exceeded Vercel's 60s budget, so a
    # single slow venue turned into a hard FUNCTION_INVOCATION_TIMEOUT — the
    # whole scan died instead of degrading. 8 × 2 ≈ 24s leaves room to fetch
    # the other venue and still answer.
    http_timeout: float = 8.0
    http_retries: int = 2
    rate_limit_rps: float = 8.0

    # Hard ceiling on how long one venue may consume during a live fetch.
    # Whatever pages arrived before the deadline are kept and returned.
    venue_budget_seconds: float = 20.0

    # --- api / deployment ---------------------------------------------------
    # Comma-separated origins allowed to call the API. In production this must
    # include your Vercel domain, e.g.
    #   QUANTLAB_CORS_ORIGINS=https://quantlab.vercel.app
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    port: int = 8000

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def storage_is_ephemeral(self) -> bool:
        """True when the SQLite file lives somewhere a PaaS will wipe.

        Vercel, Render's free tier and a container without a mounted volume all
        reset the filesystem between deploys (and sometimes between requests).
        Scanning and backtesting are stateless and don't care. Paper trading
        does: losing the fill history means losing the only thing that makes a
        paper record worth keeping.
        """
        if self.db_url:
            return False
        p = str(self.sqlite_path.resolve())
        return p.startswith("/tmp") or not any(
            p.startswith(prefix) for prefix in ("/data", "/var/data", "/mnt")
        ) and bool(
            __import__("os").environ.get("VERCEL")
            or __import__("os").environ.get("RENDER")
            or __import__("os").environ.get("RAILWAY_ENVIRONMENT")
        )

    @property
    def resolved_data_dir(self) -> Path:
        """Where the SQLite file may actually be written.

        Serverless runtimes mount the deployment read-only except for /tmp, so
        the configured `./data` would raise on the very first request. Falling
        back keeps the stateless endpoints (scan, backtest) working; the
        trade-off is that /tmp is wiped between invocations, which is why
        `storage_is_ephemeral` reports true there.
        """
        import os

        if os.environ.get("VERCEL") and not str(self.data_dir).startswith("/tmp"):
            return Path("/tmp/quantlab-data")
        return self.data_dir

    @property
    def sqlite_path(self) -> Path:
        return self.resolved_data_dir / "quantlab.db"

    @property
    def resolved_db_url(self) -> str:
        return self.db_url or f"sqlite:///{self.sqlite_path}"


settings = Settings()
