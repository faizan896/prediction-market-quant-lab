"""Vercel Python serverless entrypoint.

Deploy the API as a SECOND Vercel project whose Root Directory is the repo root
(the frontend project uses `frontend`). Vercel detects `api/*.py`, sees the ASGI
`app`, and routes every request to it via the rewrite in the root `vercel.json`.

Why this exists: it removes the need for a separate host entirely. One Vercel
account runs both halves, free.

What you give up, and it is not nothing:

  cold starts     the first request after idle pays ~3-5s to import pandas
  no filesystem   only /tmp is writable, and it is wiped between invocations
  request cap     ~60s on Hobby, so very large backfills will time out

Scanning and backtesting are stateless — they read live venue data or generate
synthetic markets per request — so they are unaffected. **Paper trading is not.**
Its fills, positions and equity history live in SQLite, which resets. Treat paper
runs here as a demo of the loop, not as a record you can draw conclusions from.
For a real paper record you need durable storage: a host with a mounted volume,
or QUANTLAB_DB_URL pointed at Postgres.
"""

from quantlab.api.main import app as _fastapi_app

_MOUNT = "/api/index"


async def app(scope, receive, send):
    """Strip Vercel's rewrite prefix before handing off to FastAPI.

    The root `vercel.json` rewrites every path to `/api/index`, and Vercel
    forwards *that* rewritten path in the ASGI scope — so FastAPI sees
    `/api/index` for every request and answers 404 on all of them, including
    routes that plainly exist. The symptom is confusing because the function is
    healthy: you get FastAPI's own `{"detail":"Not Found"}` rather than a crash.

    Undoing the prefix here keeps the routes in `quantlab.api.main` written as
    the real public paths (`/health`, `/scan`), which is what the OpenAPI schema
    and the frontend client both expect.
    """
    if scope["type"] in ("http", "websocket"):
        path = scope.get("path", "")
        if path == _MOUNT:
            scope = {**scope, "path": "/", "raw_path": b"/"}
        elif path.startswith(_MOUNT + "/"):
            stripped = path[len(_MOUNT):]
            scope = {**scope, "path": stripped, "raw_path": stripped.encode()}
    await _fastapi_app(scope, receive, send)


__all__ = ["app"]
