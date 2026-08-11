"use client";

import type { Opportunity } from "@/lib/api";
import { money, pct, signedPct } from "@/lib/format";
import { Lab } from "@/components/ui";

/**
 * The instrument rig.
 *
 * Cards hang from a rail at a height proportional to net EV, so the composition
 * IS the ranking — you read the order before you read a single number. In the
 * design this was borrowed from, card position was purely aesthetic; binding it
 * to a real quantity is what makes the layout worth its space.
 *
 * It shows the top few only. Beyond about six the stems turn into spaghetti and
 * a table does the job better, which is exactly what sits underneath it.
 */
export function Rig({
  items,
  max = 5,
  onSelect,
}: {
  items: Opportunity[];
  max?: number;
  onSelect?: (o: Opportunity) => void;
}) {
  const top = items.slice(0, max);
  if (top.length === 0) return null;

  // Scale to the best EV on screen, with a floor so a single weak result
  // doesn't get dramatically tall and imply more than it should.
  const ceiling = Math.max(0.1, ...top.map((o) => o.net_ev));
  const H = 250;
  const PAD = 26;

  return (
    <div className="rig-grid relative border-b border-[var(--color-rule)]" style={{ height: H }}>
      {[1, 0.5, 0].map((f) => (
        <div
          key={f}
          className="pointer-events-none absolute left-0 right-0 flex items-center gap-2"
          style={{ top: PAD + (1 - f) * (H - PAD * 2 - 40) }}
        >
          <div className="h-px flex-1 bg-[var(--color-rule)] opacity-60" />
          <span className="lab pr-3">{pct(ceiling * f, 0)}</span>
        </div>
      ))}

      <div className="absolute inset-0 flex items-stretch gap-3 overflow-x-auto px-4">
        {top.map((o, i) => {
          const frac = Math.max(0, Math.min(1, o.net_ev / ceiling));
          const top_ = PAD + (1 - frac) * (H - PAD * 2 - 118);
          const hot = i === 0;
          return (
            <div key={o.key} className="relative w-[118px] shrink-0">
              <div
                className="absolute left-1/2 w-px bg-[var(--color-ink-3)] opacity-40"
                style={{ top: top_ + 112, bottom: 0 }}
              />
              <button
                onClick={() => onSelect?.(o)}
                title={o.title}
                className={`absolute left-0 w-[118px] border px-2.5 py-2 text-left transition ${
                  hot
                    ? "border-[var(--color-ink)] bg-[var(--color-ink)]"
                    : "border-[var(--color-rule)] bg-[var(--color-paper-raised)] hover:border-[var(--color-ink-2)]"
                }`}
                style={{ top: top_ }}
              >
                <span
                  className="num block text-[8px] tracking-[0.12em]"
                  style={{ color: hot ? "#8d8a82" : "var(--color-ink-3)" }}
                >
                  {o.venue.slice(0, 4).toUpperCase()}-{o.key.slice(-4)}
                </span>

                <span className="lab mt-1.5 block" style={{ color: hot ? "#8d8a82" : undefined }}>
                  Market
                </span>
                <span
                  className="num display block text-[19px]"
                  style={{ color: hot ? "var(--color-paper)" : "var(--color-ink)" }}
                >
                  {(o.price * 100).toFixed(1)}
                  <span className="text-[11px] font-medium">%</span>
                </span>

                <span
                  className="mt-1.5 block border-t"
                  style={{ borderColor: hot ? "#3a3833" : "var(--color-rule)" }}
                />
                <span className="lab mt-1.5 block" style={{ color: hot ? "#8d8a82" : undefined }}>
                  Net EV
                </span>
                <span className="num display block text-[15px] text-[var(--color-accent)]">
                  {signedPct(o.net_ev)}
                </span>
              </button>
            </div>
          );
        })}
      </div>

      <div className="absolute bottom-2 left-4">
        <Lab>[ Height = net EV after costs ]</Lab>
      </div>
    </div>
  );
}

/** Compact price → fair readout used inside table rows. */
export function Delta({ price, fair }: { price: number; fair: number }) {
  const d = fair - price;
  return (
    <span className="num text-[12.5px]">
      <span className="text-[var(--color-ink-3)]">{(price * 100).toFixed(1)}</span>
      <span className="mx-1 text-[var(--color-ink-3)]">→</span>
      <span className="text-[var(--color-ink)]">{(fair * 100).toFixed(1)}</span>
      <span className="ml-1.5 text-[var(--color-accent)]">
        {d >= 0 ? "+" : ""}
        {(d * 100).toFixed(1)}
      </span>
    </span>
  );
}

export function SideTag({ side }: { side: string }) {
  const yes = side === "YES";
  return (
    <span
      className={`num text-[9px] uppercase tracking-[0.14em] ${
        yes ? "text-[var(--color-up)]" : "text-[var(--color-accent)]"
      }`}
    >
      {side}
    </span>
  );
}

export function Bar({ value, max = 1 }: { value: number; max?: number }) {
  const w = Math.max(0, Math.min(1, value / max));
  return (
    <span className="block h-[5px] w-full bg-[var(--color-paper-sunk)]">
      <span
        className="block h-full bg-[var(--color-ink)] opacity-80"
        style={{ width: `${w * 100}%` }}
      />
    </span>
  );
}

export function Money({ v }: { v: number }) {
  return <span className="num">{money(v)}</span>;
}
