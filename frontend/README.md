# Quant Lab frontend

Next.js 16 (App Router) + Tailwind 4 + Recharts. Three screens against the
FastAPI backend.

```bash
# terminal 1 — API
cd ..
uvicorn quantlab.api.main:app --reload --port 8000

# terminal 2 — UI
npm install
npm run dev          # http://localhost:3000
```

Point it somewhere else with `NEXT_PUBLIC_API_URL` in `.env.local`.

## Screens

| Route | What it does |
|---|---|
| `/` | Mispricing scanner. Toggle and reweight signals, set filters, click a row for per-signal attribution. |
| `/backtest` | Strategy builder + leaderboard, equity curves, walk-forward folds, calibration table. |
| `/paper` | Paper runs: portfolio, open positions, fill blotter, and the rejection log. |

## Notes

- Every page works with **zero setup** — the scanner and backtest default to
  synthetic data, so nothing is an empty shell before you run `quantlab ingest`.
- All numbers come from the API. There is no client-side finance math here on
  purpose: if the UI computed ROI differently from the backtester, you'd have
  two answers and no way to tell which is right.
- Types in `lib/api.ts` mirror `quantlab/api/schemas.py` by hand. To generate
  them instead: `npx openapi-typescript http://127.0.0.1:8000/openapi.json`.
- Formatting helpers tolerate `null` everywhere, because the API returns `null`
  for any non-finite metric (JSON has no `NaN`).
