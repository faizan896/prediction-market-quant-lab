/**
 * Typed client for the quantlab API.
 *
 * The types here are hand-mirrored from quantlab/api/schemas.py. If you'd
 * rather generate them, the app serves an OpenAPI document at /openapi.json —
 * `npx openapi-typescript http://127.0.0.1:8000/openapi.json` works fine. Hand
 * writing is kept for now because the surface is small and readable.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      `Cannot reach the API at ${API_URL}. Start it with: uvicorn quantlab.api.main:app --port 8000`,
      0,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

// ------------------------------------------------------------------ types
export interface Market {
  key: string;
  venue: string;
  market_id: string;
  title: string;
  yes_bid: number | null;
  yes_ask: number | null;
  mid: number | null;
  spread: number | null;
  volume: number;
  volume_24h: number;
  liquidity: number;
  close_time: string | null;
  resolved: boolean;
}

export interface Opportunity {
  key: string;
  venue: string;
  title: string;
  price: number;
  fair_value: number;
  edge: number;
  side: "YES" | "NO";
  confidence: number;
  gross_ev: number;
  net_ev: number;
  kelly_fraction: number;
  suggested_notional: number;
  spread: number;
  liquidity: number;
  volume_24h: number;
  hours_to_close: number | null;
  contributors: Record<string, number>;
  score: number;
  notes: string;
}

export interface ScanResponse {
  scanned: number;
  returned: number;
  opportunities: Opportunity[];
  elapsed_ms: number;
  /** Non-fatal problems, chiefly an unreachable venue. An empty result means
   *  nothing unless you know whether the data actually arrived. */
  warnings?: string[];
}

export type SignalName =
  | "market"
  | "cross_venue"
  | "book_imbalance"
  | "longshot"
  | "drift";

export interface ScanRequest {
  venues?: string[];
  limit?: number;
  min_volume?: number;
  min_edge?: number;
  min_net_ev?: number;
  min_confidence?: number;
  bankroll?: number;
  market_anchor?: number;
  frictions?: Frictions;
  signals?: { name: SignalName; enabled?: boolean; weight?: number }[];
  external?: Record<string, [number, number]>;
}

export type Frictions =
  | "frictionless"
  | "polymarket"
  | "kalshi"
  | "conservative";

export interface StrategyInfo {
  id: string;
  name: string;
  summary: string;
  description: string;
  params: { name: string; default: unknown; type: string }[];
}

export interface MetricsRow {
  strategy: string;
  trades: number;
  win_rate: number;
  roi: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  profit_factor: number;
  brier: number | null;
  edge_capture: number | null;
  total_pnl: number;
  total_fees: number;
  total_slippage: number;
  final_equity: number;
  avg_hold_hours: number;
}

export interface EquityPoint {
  ts: string;
  equity: number;
}

export interface BacktestResponse {
  run_id: string;
  markets: number;
  bars: number;
  frictions: string;
  leaderboard: MetricsRow[];
  equity: Record<string, EquityPoint[]>;
  reliability: Record<
    string,
    { bin: string; n: number; predicted: number; actual: number; gap: number }[]
  >;
  folds: {
    fold: number;
    trades: number;
    win_rate: number;
    roi: number;
    sharpe: number;
  }[];
  warnings: string[];
}

export interface BacktestRequest {
  strategies?: { name: string; params?: Record<string, unknown> }[];
  synthetic?: boolean;
  synthetic_markets?: number;
  synthetic_steps?: number;
  synthetic_spread?: number;
  frictions?: Frictions;
  capital?: number;
  kelly_multiplier?: number;
  signal_lag?: number;
  walk_forward_folds?: number;
}

export interface PaperSummary {
  run_id: string;
  name: string;
  status: string;
  initial_capital: number;
  cash: number;
  equity: number;
  exposure: number;
  total_return: number;
  realized_pnl: number;
  unrealized_pnl: number;
  fees_paid: number;
  open_positions: number;
  closed_trades: number;
  win_rate: number | null;
  edge_capture: number | null;
  brier: number | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface Position {
  key: string;
  title: string;
  side: string;
  contracts: number;
  avg_price: number;
  cost_basis: number;
  fair_at_entry: number | null;
  edge_at_entry: number | null;
  opened_at: string;
}

export interface Fill {
  fill_id: string;
  key: string;
  title: string | null;
  ts: string;
  action: string;
  side: string;
  price: number;
  contracts: number;
  notional: number;
  fee: number;
  realized_pnl: number;
  edge: number | null;
  reason: string | null;
}

export interface PaperDetail {
  summary: PaperSummary;
  positions: Position[];
  fills: Fill[];
  equity: EquityPoint[];
}

export interface StepResponse {
  ts: string;
  scanned: number;
  opened: Fill[];
  closed: Fill[];
  settled: Fill[];
  rejected: string[];
  summary: PaperSummary;
}

// -------------------------------------------------------------- endpoints
export const api = {
  health: () => request<{ status: string; version: string }>("/health"),

  markets: (params: {
    venue?: string;
    limit?: number;
    min_volume?: number;
    live?: boolean;
  } = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") q.set(k, String(v));
    });
    return request<Market[]>(`/markets?${q}`);
  },

  scan: (req: ScanRequest) => post<ScanResponse>("/scan", req),

  strategies: () => request<StrategyInfo[]>("/strategies"),
  backtest: (req: BacktestRequest) => post<BacktestResponse>("/backtest", req),

  paperRuns: () => request<PaperSummary[]>("/paper"),
  createPaperRun: (body: {
    name: string;
    capital: number;
    min_net_ev?: number;
    min_confidence?: number;
  }) => post<PaperSummary>("/paper", body),
  paperRun: (id: string) => request<PaperDetail>(`/paper/${id}`),
  paperStep: (
    id: string,
    body: { synthetic?: boolean; limit?: number; venues?: string[] },
  ) => post<StepResponse>(`/paper/${id}/step`, body),
};
