"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Bar } from "@/components/rig";
import {
  Button,
  Empty,
  ErrorNote,
  Field,
  Lab,
  Note,
  NumberInput,
  Panel,
  Row,
  Select,
  Stat,
  Table,
  Td,
} from "@/components/ui";
import {
  ApiError,
  api,
  type BacktestResponse,
  type Frictions,
  type StrategyInfo,
} from "@/lib/api";
import { money, num, pct, signedPct } from "@/lib/format";

const FRICTIONS: { value: Frictions; label: string }[] = [
  { value: "conservative", label: "conservative — recommended" },
  { value: "kalshi", label: "kalshi" },
  { value: "polymarket", label: "polymarket" },
  { value: "frictionless", label: "frictionless — research only" },
];

const INK = ["#12120f", "#d94f1e", "#1f6b45", "#6a675e", "#a4342a", "#9b978c"];

export default function BacktestPage() {
  const [strategies, setStrategies] = useState<StrategyInfo[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [params, setParams] = useState<Record<string, Record<string, number>>>({});
  const [nMarkets, setNMarkets] = useState(120);
  const [nSteps, setNSteps] = useState(180);
  const [spread, setSpread] = useState(0.01);
  const [frictions, setFrictions] = useState<Frictions>("conservative");
  const [capital, setCapital] = useState(100_000);
  const [kelly, setKelly] = useState(0.25);
  const [folds, setFolds] = useState(0);
  const [synthetic, setSynthetic] = useState(true);

  const [data, setData] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"board" | "equity" | "calib">("board");

  useEffect(() => {
    api
      .strategies()
      .then((s) => {
        setStrategies(s);
        setSelected(
          s
            .filter((x) =>
              ["momentum", "meanrev", "longshot", "breakout", "timedecay", "buyhold"].includes(x.id),
            )
            .map((x) => x.id),
        );
      })
      .catch((e) => setError(e instanceof ApiError ? e.message : "Cannot reach the API."));
  }, []);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(
        await api.backtest({
          strategies: selected.map((id) => ({ name: id, params: params[id] ?? {} })),
          synthetic,
          synthetic_markets: nMarkets,
          synthetic_steps: nSteps,
          synthetic_spread: spread,
          frictions,
          capital,
          kelly_multiplier: kelly,
          walk_forward_folds: folds,
        }),
      );
      setTab("board");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Unexpected error running the backtest.");
    } finally {
      setLoading(false);
    }
  }, [selected, params, synthetic, nMarkets, nSteps, spread, frictions, capital, kelly, folds]);

  /** Equity curves have different timestamps per strategy, so index by trade
   *  ordinal rather than time — the shape is what matters, not the calendar. */
  const chart = useMemo(() => {
    if (!data) return [];
    const names = Object.keys(data.equity);
    const longest = Math.max(0, ...names.map((n) => data.equity[n].length));
    return Array.from({ length: longest }, (_, i) => {
      const row: Record<string, number | string> = { i };
      names.forEach((n) => {
        const p = data.equity[n][i];
        if (p) row[n] = Math.round(p.equity);
      });
      return row;
    });
  }, [data]);

  const series = data ? Object.keys(data.equity) : [];

  return (
    <div className="grid gap-5 pt-5 lg:grid-cols-[236px_1fr]">
      <div className="space-y-4">
        <Panel label="Data">
          <div className="space-y-3.5">
            <label className="flex items-center gap-2 text-[11.5px]">
              <input
                type="checkbox"
                checked={synthetic}
                onChange={(e) => setSynthetic(e.target.checked)}
                className="accent-[var(--color-accent)]"
              />
              Synthetic data
            </label>
            {synthetic ? (
              <>
                <Field label="Markets">
                  <NumberInput value={nMarkets} onChange={setNMarkets} step={20} min={5} />
                </Field>
                <Field label="Bars / market">
                  <NumberInput value={nSteps} onChange={setNSteps} step={20} min={20} />
                </Field>
                <Field
                  label="Quoted spread"
                  hint="Try 0 vs 0.02. Most 'edges' die somewhere in between."
                >
                  <NumberInput value={spread} onChange={setSpread} step={0.005} min={0} />
                </Field>
                <Note>
                  Generated with a planted favourite-longshot bias. Good for validating the engine,
                  useless for validating a strategy.
                </Note>
              </>
            ) : (
              <Note>
                Uses your local store. Run <span className="num">quantlab ingest</span> then{" "}
                <span className="num">quantlab backfill</span> first.
              </Note>
            )}
          </div>
        </Panel>

        <Panel label="Execution">
          <div className="space-y-3.5">
            <Field label="Cost model">
              <Select value={frictions} onChange={setFrictions} options={FRICTIONS} />
            </Field>
            <Field label="Capital ($)">
              <NumberInput value={capital} onChange={setCapital} step={10_000} min={1000} />
            </Field>
            <Field
              label="Kelly multiplier"
              hint="Never run full Kelly. Quarter-Kelly gives ~87% of the growth at ~44% of the variance."
            >
              <NumberInput value={kelly} onChange={setKelly} step={0.05} min={0.01} max={1} />
            </Field>
            <Field label="Walk-forward folds" hint="0 disables. Applies to the first strategy.">
              <NumberInput value={folds} onChange={setFolds} step={1} min={0} max={10} />
            </Field>
          </div>
        </Panel>

        <Panel label="Strategies">
          <div className="space-y-2">
            {strategies.map((s) => {
              const on = selected.includes(s.id);
              return (
                <div key={s.id} className="border border-[var(--color-rule)] px-2 py-1.5">
                  <label className="flex items-start gap-2">
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={() =>
                        setSelected((c) =>
                          c.includes(s.id) ? c.filter((x) => x !== s.id) : [...c, s.id],
                        )
                      }
                      className="mt-0.5 accent-[var(--color-accent)]"
                    />
                    <span className="text-[11.5px]">{s.name}</span>
                  </label>
                  <p className="mt-1 pl-6 text-[9.5px] leading-snug text-[var(--color-ink-3)]">
                    {s.summary}
                  </p>
                  {on && s.params.length > 0 && (
                    <div className="mt-1.5 grid grid-cols-2 gap-1 pl-6">
                      {s.params
                        .filter((p) => typeof p.default === "number")
                        .map((p) => (
                          <label key={p.name} className="block">
                            <span className="lab !text-[8px]">{p.name}</span>
                            <input
                              type="number"
                              step="any"
                              value={params[s.id]?.[p.name] ?? (p.default as number)}
                              onChange={(e) =>
                                setParams((c) => ({
                                  ...c,
                                  [s.id]: { ...(c[s.id] ?? {}), [p.name]: Number(e.target.value) },
                                }))
                              }
                              className="num w-full border border-[var(--color-rule)] bg-[var(--color-paper)] px-1 py-0.5 text-[10.5px] outline-none focus:border-[var(--color-ink)]"
                            />
                          </label>
                        ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Panel>

        <Button onClick={run} disabled={loading || selected.length === 0}>
          {loading ? "Running…" : "Run backtest"}
        </Button>
      </div>

      <div className="space-y-4">
        {error && <ErrorNote message={error} />}
        {data?.warnings.map((w) => (
          <div
            key={w}
            className="border border-[var(--color-accent)] bg-[var(--color-accent-soft)] px-3.5 py-2.5 text-[11.5px] text-[var(--color-down)]"
          >
            <Lab className="!text-[var(--color-down)]">[ Warning ]</Lab> {w}
          </div>
        ))}

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Markets" value={data?.markets ?? "—"} />
          <Stat label="Bars" value={data ? data.bars.toLocaleString() : "—"} />
          <Stat label="Cost model" value={<span className="text-[15px]">{frictions}</span>} />
          <Stat label="Signal lag" value={<span className="text-[15px]">1 bar</span>} hint="anti-lookahead" />
        </div>

        <Panel
          label="Instrument_002"
          title="Strategy leaderboard"
          flush
          right={
            <div className="flex">
              {(
                [
                  ["board", "Board"],
                  ["equity", "Equity"],
                  ["calib", "Calibration"],
                ] as const
              ).map(([k, l]) => (
                <button
                  key={k}
                  onClick={() => setTab(k)}
                  className={`border border-[var(--color-rule)] px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.14em] ${
                    tab === k
                      ? "bg-[var(--color-ink)] text-[var(--color-paper)]"
                      : "text-[var(--color-ink-3)]"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>
          }
        >
          {!data?.leaderboard.length ? (
            <Empty title="No run yet.">Configure on the left, then run a backtest.</Empty>
          ) : tab === "board" ? (
            <Table
              cols={[
                { key: "s", label: "Strategy", align: "left" },
                { key: "t", label: "Trades", align: "right" },
                { key: "w", label: "Win rate", align: "right" },
                { key: "r", label: "ROI", align: "right" },
                { key: "sh", label: "Sharpe", align: "right" },
                { key: "dd", label: "Max DD", align: "right" },
                { key: "pf", label: "PF", align: "right" },
                { key: "b", label: "Brier", align: "right" },
                { key: "ec", label: "Edge cap", align: "right" },
                { key: "f", label: "Costs", align: "right" },
              ]}
            >
              {data.leaderboard.map((r) => (
                <Row key={r.strategy}>
                  <Td align="left" mono={false}>
                    {r.strategy}
                  </Td>
                  <Td tone="muted">{r.trades.toLocaleString()}</Td>
                  <Td>{pct(r.win_rate)}</Td>
                  <Td tone={r.roi > 0 ? "up" : "down"}>{signedPct(r.roi)}</Td>
                  <Td tone={r.sharpe > 1 ? "up" : undefined}>{num(r.sharpe)}</Td>
                  <Td tone="muted">{pct(r.max_drawdown)}</Td>
                  <Td tone="muted">{num(r.profit_factor)}</Td>
                  <Td tone="muted">{num(r.brier, 3)}</Td>
                  <Td tone={(r.edge_capture ?? 0) > 0.5 ? "up" : "accent"}>{num(r.edge_capture)}</Td>
                  <Td tone="muted">{money(r.total_fees + r.total_slippage)}</Td>
                </Row>
              ))}
            </Table>
          ) : tab === "equity" ? (
            <div className="h-80 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chart}>
                  <CartesianGrid stroke="#d5d1c6" vertical={false} />
                  <XAxis
                    dataKey="i"
                    tick={{ fill: "#9b978c", fontSize: 10 }}
                    stroke="#d5d1c6"
                    label={{ value: "trade #", position: "insideBottom", offset: -2, fill: "#9b978c", fontSize: 10 }}
                  />
                  <YAxis
                    tick={{ fill: "#9b978c", fontSize: 10 }}
                    stroke="#d5d1c6"
                    tickFormatter={(v: number) => `${Math.round(v / 1000)}k`}
                    domain={["auto", "auto"]}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#f6f5f1",
                      border: "1px solid #d5d1c6",
                      borderRadius: 0,
                      fontSize: 11,
                    }}
                    formatter={(v) => money(Number(v))}
                  />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  {series.map((n, i) => (
                    <Line
                      key={n}
                      type="monotone"
                      dataKey={n}
                      stroke={INK[i % INK.length]}
                      dot={false}
                      strokeWidth={1.4}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="p-4">
              {Object.keys(data.reliability).length === 0 ? (
                <Empty title="No calibration data.">
                  Calibration needs strategies that report a fair probability and trade to
                  settlement.
                </Empty>
              ) : (
                Object.entries(data.reliability).map(([strategy, rows]) => (
                  <div key={strategy} className="mb-5 last:mb-0">
                    <Lab>{strategy}</Lab>
                    <div className="mt-2 space-y-1.5">
                      {rows.map((r) => (
                        <div key={r.bin} className="flex items-center gap-3">
                          <span className="num w-16 text-[10.5px] text-[var(--color-ink-3)]">
                            {r.bin}
                          </span>
                          <span className="flex-1">
                            <Bar value={r.actual} />
                          </span>
                          <span className="num w-14 text-right text-[10.5px]">
                            {pct(r.predicted, 0)}
                          </span>
                          <span className="num w-14 text-right text-[10.5px]">
                            {pct(r.actual, 0)}
                          </span>
                          <span
                            className={`num w-14 text-right text-[10.5px] ${
                              Math.abs(r.gap) > 0.1
                                ? "text-[var(--color-accent)]"
                                : "text-[var(--color-ink-3)]"
                            }`}
                          >
                            {signedPct(r.gap, 0)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          <div className="border-t border-[var(--color-rule)] px-4 py-3">
            <Note label="Note">
              Edge capture is realized return ÷ predicted edge. Below ~0.5 the signal is real but
              the costs are eating it. A strategy that is positive frictionless and negative under{" "}
              <span className="num">conservative</span> does not have an edge — it has a cost
              problem.
            </Note>
          </div>
        </Panel>

        {data && data.folds.length > 0 && (
          <Panel label="Out of sample" title="Walk-forward" flush>
            <Table
              cols={[
                { key: "f", label: "Fold", align: "left" },
                { key: "t", label: "Trades", align: "right" },
                { key: "w", label: "Win rate", align: "right" },
                { key: "r", label: "ROI", align: "right" },
                { key: "s", label: "Sharpe", align: "right" },
              ]}
            >
              {data.folds.map((f) => (
                <Row key={f.fold}>
                  <Td align="left">{f.fold}</Td>
                  <Td tone="muted">{f.trades}</Td>
                  <Td>{pct(f.win_rate)}</Td>
                  <Td tone={f.roi > 0 ? "up" : "down"}>{signedPct(f.roi)}</Td>
                  <Td tone={f.sharpe > 0 ? "up" : "down"}>{num(f.sharpe)}</Td>
                </Row>
              ))}
            </Table>
          </Panel>
        )}
      </div>
    </div>
  );
}
