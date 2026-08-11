"use client";

import { useCallback, useEffect, useState } from "react";

import { Bar, Delta, Rig, SideTag } from "@/components/rig";
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
  Toggle,
} from "@/components/ui";
import {
  ApiError,
  api,
  type Frictions,
  type Opportunity,
  type ScanResponse,
  type SignalName,
} from "@/lib/api";
import { hours, money, num, pct, signedPct } from "@/lib/format";

const SIGNALS: { name: SignalName; label: string; blurb: string }[] = [
  { name: "market", label: "Market prior", blurb: "The price itself. Should usually dominate." },
  { name: "longshot", label: "Longshot bias", blurb: "Cheap contracts are overpriced." },
  { name: "cross_venue", label: "Cross-venue", blurb: "The other exchange on a linked contract." },
  { name: "drift", label: "Volume drift", blurb: "Recent volume-weighted price drift." },
  { name: "book_imbalance", label: "Book imbalance", blurb: "Microprice from resting depth." },
];

const FRICTIONS: { value: Frictions; label: string }[] = [
  { value: "conservative", label: "conservative — recommended" },
  { value: "kalshi", label: "kalshi" },
  { value: "polymarket", label: "polymarket" },
  { value: "frictionless", label: "frictionless — research only" },
];

export default function ResearchPage() {
  const [venues, setVenues] = useState<string[]>(["polymarket", "kalshi"]);
  const [limit, setLimit] = useState(300);
  const [minVolume, setMinVolume] = useState(50_000);
  const [minEdge, setMinEdge] = useState(0.02);
  const [minNetEv, setMinNetEv] = useState(0.0);
  const [minConfidence, setMinConfidence] = useState(0.15);
  const [bankroll, setBankroll] = useState(100_000);
  const [anchor, setAnchor] = useState(0.3);
  const [frictions, setFrictions] = useState<Frictions>("conservative");
  const [weights, setWeights] = useState<Partial<Record<SignalName, number>>>({});
  const [enabled, setEnabled] = useState<Record<SignalName, boolean>>({
    market: true,
    longshot: true,
    cross_venue: true,
    drift: true,
    book_imbalance: false,
  });

  const [data, setData] = useState<ScanResponse | null>(null);
  const [nearMiss, setNearMiss] = useState<ScanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<Opportunity | null>(null);

  const body = useCallback(
    (overrides: Record<string, unknown> = {}) => ({
      venues,
      limit,
      min_volume: minVolume,
      min_edge: minEdge,
      min_net_ev: minNetEv,
      min_confidence: minConfidence,
      bankroll,
      market_anchor: anchor,
      frictions,
      signals: SIGNALS.filter((s) => enabled[s.name]).map((s) => ({
        name: s.name,
        weight: weights[s.name],
      })),
      ...overrides,
    }),
    [venues, limit, minVolume, minEdge, minNetEv, minConfidence, bankroll, anchor, frictions, enabled, weights],
  );

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    setNearMiss(null);
    try {
      const res = await api.scan(body());
      setData(res);
      setPicked(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Unexpected error running the scan.");
    } finally {
      setLoading(false);
    }
  }, [body]);

  /** The rejected set is the most informative thing on the page — it says how
   *  much edge existed and exactly how much of it the costs took. */
  const loadNearMisses = useCallback(async () => {
    try {
      setNearMiss(await api.scan(body({ min_net_ev: -10, min_confidence: 0, min_edge: 0.005 })));
    } catch {
      /* non-fatal: the primary result is already on screen */
    }
  }, [body]);

  useEffect(() => {
    // First paint uses a smaller universe deliberately: a cold serverless
    // function fetching 600 markets is the difference between a page that
    // renders in two seconds and one that trips the request timeout. Press
    // "Run scan" for the full universe.
    api
      .scan({
        venues: ["polymarket", "kalshi"],
        limit: 120,
        min_volume: 50_000,
        min_edge: 0.02,
        min_net_ev: 0,
        min_confidence: 0.15,
        frictions: "conservative",
      })
      .then(setData)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Cannot reach the API."));
  }, []);

  const opps = data?.opportunities ?? [];
  const best = opps.length ? Math.max(...opps.map((o) => o.net_ev)) : null;
  const rejected = nearMiss
    ? nearMiss.opportunities.filter((o) => o.net_ev < minNetEv)
    : [];

  return (
    <div className="grid gap-5 pt-5 lg:grid-cols-[236px_1fr]">
      {/* ------------------------------------------------------------ controls */}
      <div className="space-y-4">
        <Panel label="Universe">
          <div className="space-y-3.5">
            <Field label="Venues">
              <div className="flex flex-wrap gap-1.5">
                {["polymarket", "kalshi", "synthetic"].map((v) => (
                  <Toggle
                    key={v}
                    on={venues.includes(v)}
                    onClick={() =>
                      setVenues((c) => (c.includes(v) ? c.filter((x) => x !== v) : [...c, v]))
                    }
                  >
                    {v.slice(0, 4)}
                  </Toggle>
                ))}
              </div>
            </Field>
            <Field label="Markets / venue">
              <NumberInput value={limit} onChange={setLimit} step={50} min={1} />
            </Field>
            <Field label="Min volume ($)">
              <NumberInput value={minVolume} onChange={setMinVolume} step={10_000} min={0} />
            </Field>
          </div>
        </Panel>

        <Panel label="Filters">
          <div className="space-y-3.5">
            <Field label="Min edge" hint="Raw gap between fair value and price.">
              <NumberInput value={minEdge} onChange={setMinEdge} step={0.005} />
            </Field>
            <Field
              label="Min net EV"
              hint="Per dollar deployed, after spread, fees and slippage. This is the one that matters."
            >
              <NumberInput value={minNetEv} onChange={setMinNetEv} step={0.01} />
            </Field>
            <Field label="Min confidence">
              <NumberInput
                value={minConfidence}
                onChange={setMinConfidence}
                step={0.05}
                min={0}
                max={1}
              />
            </Field>
            <Field label="Cost model">
              <Select value={frictions} onChange={setFrictions} options={FRICTIONS} />
            </Field>
          </div>
        </Panel>

        <Panel label="Signals" title={undefined}>
          <div className="space-y-2">
            {SIGNALS.map((s) => (
              <div key={s.name} className="border border-[var(--color-rule)] px-2 py-1.5">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={enabled[s.name]}
                    onChange={(e) => setEnabled((c) => ({ ...c, [s.name]: e.target.checked }))}
                    className="accent-[var(--color-accent)]"
                  />
                  <span className="text-[11.5px]">{s.label}</span>
                  {enabled[s.name] && (
                    <input
                      type="number"
                      placeholder="wt"
                      step={0.5}
                      value={weights[s.name] ?? ""}
                      onChange={(e) =>
                        setWeights((c) => ({
                          ...c,
                          [s.name]: e.target.value === "" ? undefined : Number(e.target.value),
                        }))
                      }
                      className="num ml-auto w-12 border border-[var(--color-rule)] bg-[var(--color-paper)] px-1 py-0.5 text-right text-[10.5px] outline-none focus:border-[var(--color-ink)]"
                    />
                  )}
                </label>
                <p className="mt-1 pl-6 text-[9.5px] leading-snug text-[var(--color-ink-3)]">
                  {s.blurb}
                </p>
              </div>
            ))}

            <Field
              label={`Market anchor · ${anchor.toFixed(2)}`}
              hint="Extra pull toward the market price after pooling. At 0 your model can move fair value freely — almost always a data bug, not an edge."
            >
              <input
                type="range"
                min={0}
                max={0.9}
                step={0.05}
                value={anchor}
                onChange={(e) => setAnchor(Number(e.target.value))}
                className="w-full"
              />
            </Field>
            <Field label="Bankroll ($)">
              <NumberInput value={bankroll} onChange={setBankroll} step={10_000} min={0} />
            </Field>
          </div>
        </Panel>

        <Button onClick={run} disabled={loading || venues.length === 0}>
          {loading ? "Scanning…" : "Run scan"}
        </Button>
      </div>

      {/* --------------------------------------------------------------- rig */}
      <div className="space-y-4">
        {error && <ErrorNote message={error} />}

        {data?.warnings?.map((w) => (
          <div
            key={w}
            className="border border-[var(--color-accent)] bg-[var(--color-accent-soft)] px-3.5 py-2.5 text-[11.5px] text-[var(--color-down)]"
          >
            <Lab className="!text-[var(--color-down)]">[ Warning ]</Lab> {w}
          </div>
        ))}

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat
            label="Tradable after costs"
            value={data ? data.returned : "—"}
            hint={data ? `of ${data.scanned} scanned` : undefined}
          />
          <Stat
            label="Best net EV"
            value={best === null ? "—" : signedPct(best)}
            tone={best && best > 0 ? "up" : "muted"}
            hint={opps[0]?.title?.slice(0, 26)}
          />
          <Stat label="Cost model" value={<span className="text-[15px]">{frictions}</span>} />
          <Stat
            label="Elapsed"
            value={data ? `${Math.round(data.elapsed_ms)}ms` : "—"}
            tone="muted"
          />
        </div>

        <Panel label="Instrument_001" title="Mispricing scan" flush>
          {opps.length > 0 && <Rig items={opps} onSelect={setPicked} />}

          {opps.length === 0 ? (
            <Empty title="Nothing clears its costs right now.">
              This is the normal answer, not a failure. Markets showed edges; spread and fees took
              them.{" "}
              <button
                onClick={loadNearMisses}
                className="underline decoration-dotted underline-offset-2 hover:text-[var(--color-ink)]"
              >
                Show what was rejected
              </button>
            </Empty>
          ) : (
            <Table
              cols={[
                { key: "id", label: "ID", w: "62px" },
                { key: "mkt", label: "Market", align: "left" },
                { key: "d", label: "Price → Fair", align: "right", w: "150px" },
                { key: "ev", label: "Net EV", align: "right", w: "78px" },
                { key: "conf", label: "Conf", align: "right", w: "56px" },
                { key: "size", label: "Size", align: "right", w: "78px" },
                { key: "exp", label: "Exp", align: "right", w: "52px" },
                { key: "side", label: "Side", align: "right", w: "48px" },
              ]}
            >
              {opps.map((o) => (
                <Row key={o.key} onClick={() => setPicked(o)}>
                  <Td tone="muted" className="!text-[10px]">
                    {o.key.slice(-4)}
                  </Td>
                  <Td align="left" mono={false} className="max-w-[300px]">
                    <span className="block truncate" title={o.title}>
                      {o.title || o.key}
                    </span>
                    <span className="lab mt-0.5 block normal-case tracking-[0.05em]">
                      {o.venue} · {money(o.liquidity)} liq
                    </span>
                  </Td>
                  <Td>
                    <Delta price={o.price} fair={o.fair_value} />
                  </Td>
                  <Td tone={o.net_ev > 0 ? "up" : "down"}>{signedPct(o.net_ev)}</Td>
                  <Td tone="muted">{num(o.confidence)}</Td>
                  <Td tone="muted">{money(o.suggested_notional)}</Td>
                  <Td tone="muted">{hours(o.hours_to_close)}</Td>
                  <Td>
                    <SideTag side={o.side} />
                  </Td>
                </Row>
              ))}
            </Table>
          )}

          <div className="border-t border-[var(--color-rule)] px-4 py-3">
            <Note label="Note">
              Ranked by net EV × confidence, not raw edge. Kalshi&apos;s fee is 7% × contracts × p ×
              (1−p) — about 7% round-trip at mid-book — so a 50¢ contract needs roughly 7 points of
              real edge to break even, while the tails need far less.
            </Note>
          </div>
        </Panel>

        {/* ------------------------------------------------- rejected markets */}
        {rejected.length > 0 && (
          <Panel
            label="Rejected"
            title={`${rejected.length} lost their edge to costs`}
            flush
          >
            <Table
              cols={[
                { key: "mkt", label: "Market", align: "left" },
                { key: "g", label: "Gross EV", align: "right", w: "84px" },
                { key: "c", label: "Cost", align: "right", w: "84px" },
                { key: "n", label: "Net EV", align: "right", w: "84px" },
              ]}
            >
              {rejected.slice(0, 12).map((o) => (
                <Row key={o.key}>
                  <Td align="left" mono={false} className="max-w-[380px]">
                    <span className="block truncate" title={o.title}>
                      {o.title}
                    </span>
                  </Td>
                  <Td tone="muted">{signedPct(o.gross_ev)}</Td>
                  <Td tone="accent">−{pct(Math.max(0, o.gross_ev - o.net_ev))}</Td>
                  <Td tone="down">{signedPct(o.net_ev)}</Td>
                </Row>
              ))}
            </Table>
          </Panel>
        )}

        {/* ------------------------------------------------------ attribution */}
        {picked && (
          <Panel
            label="Signal attribution"
            title={picked.title}
            right={
              <Button variant="ghost" onClick={() => setPicked(null)}>
                Close
              </Button>
            }
          >
            <div className="grid gap-3 sm:grid-cols-4">
              <Stat label="Price" value={`${(picked.price * 100).toFixed(1)}%`} />
              <Stat label="Fair value" value={`${(picked.fair_value * 100).toFixed(1)}%`} tone="accent" />
              <Stat label="Gross EV" value={signedPct(picked.gross_ev)} hint="before costs" />
              <Stat
                label="Net EV"
                value={signedPct(picked.net_ev)}
                tone={picked.net_ev > 0 ? "up" : "down"}
                hint="after costs"
              />
            </div>

            <div className="mt-4">
              <Lab>Pooled weights</Lab>
              <div className="mt-2 space-y-1.5">
                {Object.entries(picked.contributors).map(([name, w]) => {
                  const total = Object.values(picked.contributors).reduce((a, b) => a + b, 0);
                  const share = total > 0 ? w / total : 0;
                  return (
                    <div key={name} className="flex items-center gap-3">
                      <span className="w-28 text-[11px] text-[var(--color-ink-2)]">{name}</span>
                      <span className="flex-1">
                        <Bar value={share} />
                      </span>
                      <span className="num w-11 text-right text-[10.5px] text-[var(--color-ink-3)]">
                        {pct(share, 0)}
                      </span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3">
                <Note>
                  Kelly fraction {num(picked.kelly_fraction, 3)} · spread{" "}
                  {(picked.spread * 100).toFixed(1)}¢ · 24h volume {money(picked.volume_24h)}
                </Note>
              </div>
            </div>
          </Panel>
        )}
      </div>
    </div>
  );
}
