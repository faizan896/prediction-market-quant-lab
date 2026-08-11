"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SideTag } from "@/components/rig";
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
  Stat,
  Table,
  Td,
} from "@/components/ui";
import { ApiError, api, type PaperDetail, type PaperSummary, type StepResponse } from "@/lib/api";
import { ago, money, num, pct, signedPct } from "@/lib/format";

export default function PaperPage() {
  const [runs, setRuns] = useState<PaperSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [detail, setDetail] = useState<PaperDetail | null>(null);
  const [lastStep, setLastStep] = useState<StepResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [name, setName] = useState("run-1");
  const [capital, setCapital] = useState(100_000);
  const [minNetEv, setMinNetEv] = useState(0.02);
  const [useSynthetic, setUseSynthetic] = useState(true);

  const loadRuns = useCallback(async () => {
    try {
      const r = await api.paperRuns();
      setRuns(r);
      if (!activeId && r.length) setActiveId(r[0].run_id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Cannot reach the API.");
    }
  }, [activeId]);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (!activeId) return;
    api
      .paperRun(activeId)
      .then(setDetail)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Failed to load run."));
  }, [activeId]);

  const create = async () => {
    setBusy(true);
    setError(null);
    try {
      const run = await api.createPaperRun({ name, capital, min_net_ev: minNetEv });
      setActiveId(run.run_id);
      await loadRuns();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to create run.");
    } finally {
      setBusy(false);
    }
  };

  const step = async () => {
    if (!activeId) return;
    setBusy(true);
    setError(null);
    try {
      setLastStep(await api.paperStep(activeId, { synthetic: useSynthetic, limit: 200 }));
      setDetail(await api.paperRun(activeId));
      await loadRuns();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Step failed.");
    } finally {
      setBusy(false);
    }
  };

  const s = detail?.summary;
  const equity = (detail?.equity ?? []).map((p, i) => ({ i, equity: Math.round(p.equity) }));

  return (
    <div className="grid gap-5 pt-5 lg:grid-cols-[236px_1fr]">
      <div className="space-y-4">
        <Panel label="New run">
          <div className="space-y-3.5">
            <Field label="Name">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="num w-full border border-[var(--color-rule)] bg-[var(--color-paper)] px-2 py-1.5 text-[12.5px] outline-none focus:border-[var(--color-ink)]"
              />
            </Field>
            <Field label="Capital ($)">
              <NumberInput value={capital} onChange={setCapital} step={10_000} min={1000} />
            </Field>
            <Field label="Min net EV" hint="Nothing trades unless it clears its own costs.">
              <NumberInput value={minNetEv} onChange={setMinNetEv} step={0.005} />
            </Field>
            <Button onClick={create} disabled={busy}>
              Create run
            </Button>
          </div>
        </Panel>

        <Panel label="Runs" flush>
          {runs.length === 0 ? (
            <div className="px-4 py-5">
              <Lab>No runs yet</Lab>
            </div>
          ) : (
            <div>
              {runs.map((r) => (
                <button
                  key={r.run_id}
                  onClick={() => setActiveId(r.run_id)}
                  className={`flex w-full items-center justify-between border-b border-[var(--color-rule-soft)] px-4 py-2.5 text-left last:border-0 ${
                    r.run_id === activeId ? "bg-[var(--color-paper-sunk)]" : ""
                  }`}
                >
                  <span>
                    <span className="block text-[11.5px]">{r.name}</span>
                    <span className="lab mt-0.5 block normal-case tracking-[0.05em]">
                      {r.closed_trades} closed · {ago(r.updated_at)}
                    </span>
                  </span>
                  <span
                    className={`num text-[11.5px] ${
                      r.total_return > 0
                        ? "text-[var(--color-up)]"
                        : r.total_return < 0
                          ? "text-[var(--color-down)]"
                          : ""
                    }`}
                  >
                    {signedPct(r.total_return)}
                  </span>
                </button>
              ))}
            </div>
          )}
        </Panel>

        <Panel label="Run a cycle">
          <div className="space-y-3">
            <label className="flex items-center gap-2 text-[11.5px]">
              <input
                type="checkbox"
                checked={useSynthetic}
                onChange={(e) => setUseSynthetic(e.target.checked)}
                className="accent-[var(--color-accent)]"
              />
              Synthetic markets
            </label>
            <Note>
              One cycle: settle resolved → exit dead theses → scan → enter. Against live venues,
              schedule this hourly; these markets do not move in microseconds.
            </Note>
            <Button onClick={step} disabled={busy || !activeId}>
              {busy ? "Running…" : "Step"}
            </Button>
          </div>
        </Panel>
      </div>

      <div className="space-y-4">
        {error && <ErrorNote message={error} />}

        {!s ? (
          <Panel label="Instrument_003" title="Paper trading" flush>
            <Empty title="Create or select a run to begin.">
              The loop runs the same MispricingEngine the scanner uses, so the record measures what
              you would actually trade.
            </Empty>
          </Panel>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat
                label="Equity"
                value={money(s.equity)}
                tone={s.total_return > 0 ? "up" : s.total_return < 0 ? "down" : "ink"}
                hint={signedPct(s.total_return)}
              />
              <Stat label="Cash" value={money(s.cash)} />
              <Stat label="Exposure" value={money(s.exposure)} />
              <Stat
                label="Edge capture"
                value={num(s.edge_capture)}
                tone={(s.edge_capture ?? 0) > 0.5 ? "up" : "accent"}
                hint="realized ÷ predicted"
              />
            </div>

            <Panel label="Read this before the equity curve">
              <Note>
                Edge capture is the number that matters, not the P&amp;L. A strategy can have a
                genuine edge and still lose money if it only captures 30% of it after costs — and
                you cannot see that in a win rate. Ignore the curve entirely until roughly 200
                settled trades; before that it is almost all noise.
              </Note>
            </Panel>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Open positions" value={s.open_positions} />
              <Stat label="Closed trades" value={s.closed_trades} />
              <Stat label="Win rate" value={pct(s.win_rate)} />
              <Stat label="Fees + slippage" value={money(s.fees_paid)} tone="accent" />
            </div>

            {equity.length > 1 && (
              <Panel label="Instrument_003" title="Equity" flush>
                <div className="h-56 p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={equity}>
                      <defs>
                        <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#d94f1e" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#d94f1e" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#d5d1c6" vertical={false} />
                      <XAxis dataKey="i" tick={{ fill: "#9b978c", fontSize: 10 }} stroke="#d5d1c6" />
                      <YAxis
                        tick={{ fill: "#9b978c", fontSize: 10 }}
                        stroke="#d5d1c6"
                        domain={["auto", "auto"]}
                        tickFormatter={(v: number) => `${Math.round(v / 1000)}k`}
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
                      <Area
                        type="monotone"
                        dataKey="equity"
                        stroke="#d94f1e"
                        fill="url(#eq)"
                        strokeWidth={1.4}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            )}

            <Panel label="Held" title="Open positions" flush>
              {!detail?.positions.length ? (
                <Empty title="Flat." />
              ) : (
                <Table
                  cols={[
                    { key: "m", label: "Market", align: "left" },
                    { key: "s", label: "Side", align: "right", w: "48px" },
                    { key: "c", label: "Contracts", align: "right" },
                    { key: "p", label: "Avg price", align: "right" },
                    { key: "b", label: "Basis", align: "right" },
                    { key: "f", label: "Fair @ entry", align: "right" },
                    { key: "e", label: "Edge @ entry", align: "right" },
                    { key: "o", label: "Opened", align: "right" },
                  ]}
                >
                  {detail.positions.map((p) => (
                    <Row key={p.key}>
                      <Td align="left" mono={false} className="max-w-[280px]">
                        <span className="block truncate" title={p.title}>
                          {p.title || p.key}
                        </span>
                      </Td>
                      <Td>
                        <SideTag side={p.side} />
                      </Td>
                      <Td tone="muted">{Math.round(p.contracts).toLocaleString()}</Td>
                      <Td>{(p.avg_price * 100).toFixed(1)}</Td>
                      <Td tone="muted">{money(p.cost_basis)}</Td>
                      <Td tone="accent">
                        {p.fair_at_entry == null ? "—" : (p.fair_at_entry * 100).toFixed(1)}
                      </Td>
                      <Td tone={(p.edge_at_entry ?? 0) > 0 ? "up" : "down"}>
                        {signedPct(p.edge_at_entry)}
                      </Td>
                      <Td tone="muted">{ago(p.opened_at)}</Td>
                    </Row>
                  ))}
                </Table>
              )}
            </Panel>

            <Panel
              label="Blotter"
              title="Fills"
              flush
              right={
                lastStep ? (
                  <Lab>
                    last cycle · scanned {lastStep.scanned} · opened {lastStep.opened.length} ·
                    closed {lastStep.closed.length} · settled {lastStep.settled.length}
                  </Lab>
                ) : undefined
              }
            >
              {!detail?.fills.length ? (
                <Empty title="No fills yet." >Run a cycle.</Empty>
              ) : (
                <Table
                  cols={[
                    { key: "m", label: "Market", align: "left" },
                    { key: "a", label: "Action", align: "right", w: "62px" },
                    { key: "s", label: "Side", align: "right", w: "48px" },
                    { key: "p", label: "Price", align: "right" },
                    { key: "c", label: "Contracts", align: "right" },
                    { key: "n", label: "Notional", align: "right" },
                    { key: "f", label: "Fee", align: "right" },
                    { key: "l", label: "P&L", align: "right" },
                  ]}
                >
                  {[...detail.fills]
                    .reverse()
                    .slice(0, 40)
                    .map((f) => (
                      <Row key={f.fill_id}>
                        <Td align="left" mono={false} className="max-w-[260px]">
                          <span className="block truncate" title={f.title ?? f.key}>
                            {f.title || f.key}
                          </span>
                          {f.reason && (
                            <span className="lab mt-0.5 block normal-case tracking-[0.04em]">
                              {f.reason}
                            </span>
                          )}
                        </Td>
                        <Td tone={f.action === "settle" ? "accent" : "muted"} className="!text-[9px] uppercase tracking-[0.12em]">
                          {f.action}
                        </Td>
                        <Td>
                          <SideTag side={f.side} />
                        </Td>
                        <Td>{(f.price * 100).toFixed(1)}</Td>
                        <Td tone="muted">{Math.round(f.contracts).toLocaleString()}</Td>
                        <Td tone="muted">{money(f.notional)}</Td>
                        <Td tone="muted">{money(f.fee, 2)}</Td>
                        <Td tone={f.realized_pnl > 0 ? "up" : f.realized_pnl < 0 ? "down" : "muted"}>
                          {f.realized_pnl ? money(f.realized_pnl) : "—"}
                        </Td>
                      </Row>
                    ))}
                </Table>
              )}
            </Panel>

            {lastStep && lastStep.rejected.length > 0 && (
              <Panel label="Rejected" title="Why the loop declined to trade" flush>
                <ul className="divide-y divide-[var(--color-rule-soft)]">
                  {lastStep.rejected.slice(0, 20).map((r, i) => (
                    <li key={i} className="num px-4 py-1.5 text-[10.5px] text-[var(--color-ink-2)]">
                      {r}
                    </li>
                  ))}
                </ul>
              </Panel>
            )}
          </>
        )}
      </div>
    </div>
  );
}
