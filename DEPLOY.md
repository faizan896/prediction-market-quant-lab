# Deploying

Two options. Both are $0.

**Option A — Vercel only (simplest).** Two Vercel projects from the same repo:
the UI from `frontend/`, the API as a Python serverless function from the root.
One account, nothing else to sign up for.

```
  Vercel project 1  ──────►  Vercel project 2  ──────►  Polymarket + Kalshi
  Next.js UI                 FastAPI (serverless)       public market data
  root: frontend/            root: ./
```

**Option B — Vercel + a real server.** UI on Vercel, API on Render or Railway
via the included `Dockerfile`. Slower to set up, but you get a long-lived
process and (with a volume) durable paper-trading history.

| | Option A: Vercel only | Option B: + Render/Railway |
|---|---|---|
| Accounts needed | 1 | 2 |
| Scanner, backtest | ✅ | ✅ |
| Paper trading | ⚠️ state resets | ✅ with a volume |
| Cold start | ~3-5s after idle | ~30s on Render free (sleeps) |
| Request timeout | 60s | none |

**Start with A.** Move to B when you actually want a paper record that
accumulates — see [Paper trading needs real storage](#paper-trading-needs-real-storage).

---

## Option A — Vercel only

### A1. The UI

1. Vercel → **Add New → Project** → import the repo.
2. **Root Directory: `frontend`.** This is the step people miss — the Next.js
   app is not at the repo root, and the build fails without it.
3. Deploy. Note the URL, e.g. `https://quantlab.vercel.app`.

### A2. The API

1. Vercel → **Add New → Project** → import the *same* repo again.
2. **Root Directory: `./`** (leave it at the root).
3. Framework Preset: **Other**. Vercel may guess "FastAPI" — that is fine too;
   what matters is that it does not try to build Next.js from the root.
4. Deploy. The root `vercel.json` routes every path to `api/index.py`, and
   `requirements.txt` pins the runtime deps.
5. Check it: `curl https://your-api.vercel.app/health`

### A3. Wire them together

On the **UI** project → Settings → Environment Variables:

```
NEXT_PUBLIC_API_URL = https://your-api.vercel.app
```

On the **API** project → Settings → Environment Variables:

```
QUANTLAB_CORS_ORIGINS = https://your-ui.vercel.app
```

Redeploy both. `NEXT_PUBLIC_*` is baked in at build time, so the UI genuinely
needs a rebuild — changing the variable alone does nothing.

---

## Option B — Vercel + Render/Railway

---

## 1. API → Render (recommended) or Railway

### Render

1. Push this repo to GitHub (see the root README).
2. Render dashboard → **New → Blueprint** → select the repo. It reads
   `render.yaml` and creates the service from the `Dockerfile`.
3. Wait for the first build (~3 min). Note the URL, e.g.
   `https://quantlab-api.onrender.com`.
4. Check it: `curl https://quantlab-api.onrender.com/health`

### Railway

1. `railway init` → **Deploy from GitHub repo**.
2. Railway reads `railway.json` and builds the `Dockerfile`. `$PORT` is injected
   automatically.
3. Settings → Networking → **Generate Domain**.

### Any Docker host

```bash
docker build -t quantlab-api .
docker run -p 8000:8000 -v quantlab-data:/data quantlab-api
```

---

## 2. Frontend → Vercel

1. Vercel → **Add New → Project** → import the repo.
2. **Set Root Directory to `frontend`.** This is the step people miss — the
   Next.js app is not at the repo root, and Vercel will fail the build without
   it.
3. Environment variable:

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://your-api.onrender.com` |

4. Deploy. Note the URL, e.g. `https://quantlab.vercel.app`.

---

## 3. Close the CORS loop

The API refuses browser requests from unknown origins, so it has to be told
about the frontend. On Render/Railway, set:

```
QUANTLAB_CORS_ORIGINS=https://quantlab.vercel.app
```

and redeploy. Vercel *preview* deployments (`*.vercel.app`) are already allowed
by a regex, so you only need to add your production domain — and any custom
domain you attach later.

Symptom if you skip this: the UI loads but every panel shows
"Cannot reach the API", and the browser console shows a CORS error. The API
itself is fine — `curl` will work.

---

## Paper trading needs real storage

**Free-tier storage is ephemeral — on both options.** Vercel functions can only
write to `/tmp`, which is wiped between invocations. Render free has no
persistent disk and recycles containers. That means:

| Feature | Survives a redeploy? |
|---|---|
| Scanner | ✅ stateless — reads live venue data every call |
| Backtest (synthetic) | ✅ stateless |
| Backtest (stored history) | ❌ needs re-ingesting |
| **Paper trading** | ❌ **fills, positions and equity history are lost** |

`/health` reports `"ephemeral_storage": true` when it detects this, and the API
logs a warning on startup. For a demo that's fine. For an actual paper record —
the thing this whole project exists to produce — you need real persistence:

- **Railway:** add a Volume mounted at `/data`. Cheapest real option.
- **Render:** paid plan, uncomment the `disk:` block in `render.yaml`.
- **Staying on Vercel:** set `QUANTLAB_DB_URL` to a Postgres URL (Neon and
  Supabase both have free tiers). The schema is already Postgres-shaped, but
  note that `Store` currently only implements the SQLite driver — porting it is
  a contained piece of work in `quantlab/data/store.py`, not a redesign.

A paper-trading run that silently loses its history is worse than no run at
all, because you will trust the number it eventually shows you.

---

## Keeping the paper loop running

The loop only advances when something calls it. One cycle per hour is plenty —
these markets do not move in microseconds.

**GitHub Actions** (free, no extra infra):

```yaml
# .github/workflows/paper.yml
name: paper cycle
on:
  schedule:
    - cron: "17 * * * *"   # hourly, off the hour to dodge the stampede
  workflow_dispatch:
jobs:
  step:
    runs-on: ubuntu-latest
    steps:
      - run: |
          curl -sS -X POST "$API/paper/$RUN_ID/step" \
            -H 'content-type: application/json' \
            -d '{"venues":["polymarket","kalshi"],"limit":300}'
        env:
          API: ${{ secrets.QUANTLAB_API_URL }}
          RUN_ID: ${{ secrets.QUANTLAB_RUN_ID }}
```

Alternatives: a Render Cron Job, Railway's cron, or `cron-job.org` pointed at
the same endpoint.

Note that free-tier Render services sleep after inactivity and take ~30s to wake
— fine for an hourly cron, annoying when you open the UI cold.

---

## Local development

```bash
pip install -e ".[dev,api]"
uvicorn quantlab.api.main:app --reload --port 8000

cd frontend
cp .env.example .env.local
npm install && npm run dev
```

---

## Environment variables

### API

| Variable | Default | Notes |
|---|---|---|
| `QUANTLAB_DATA_DIR` | `./data` | Point at your mounted volume |
| `QUANTLAB_DB_URL` | *(empty → SQLite)* | Postgres URL |
| `QUANTLAB_CORS_ORIGINS` | `localhost:3000` | Comma-separated; add your Vercel URL |
| `QUANTLAB_RATE_LIMIT_RPS` | `8` | Lower it if a venue starts 429-ing |
| `PORT` | `8000` | Injected by the platform |

### Frontend

| Variable | Notes |
|---|---|
| `NEXT_PUBLIC_API_URL` | Your deployed API base URL |

No venue credentials are needed anywhere. Polymarket and Kalshi both serve
market data publicly; keys exist only for placing orders, which this deployment
cannot do.
