"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { API_URL, api } from "@/lib/api";

const TABS = [
  { href: "/", label: "Research" },
  { href: "/backtest", label: "Backtest" },
  { href: "/paper", label: "Paper Trade" },
];

export function Nav() {
  const path = usePathname();
  const [status, setStatus] = useState<"…" | "calibrated" | "offline">("…");
  const [clock, setClock] = useState("--:--:--");

  useEffect(() => {
    api
      .health()
      .then(() => setStatus("calibrated"))
      .catch(() => setStatus("offline"));
  }, []);

  useEffect(() => {
    const tick = () => setClock(new Date().toISOString().slice(11, 19));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="pt-5">
      <div className="flex items-start justify-between">
        <Link href="/" className="block">
          <svg width="34" height="17" viewBox="0 0 34 17" aria-hidden className="mb-1.5">
            <path
              d="M1 16 L11 2 L21 16 M13 16 L23 2 L33 16"
              fill="none"
              stroke="var(--color-ink)"
              strokeWidth="1.6"
            />
          </svg>
          <span className="text-[12px] font-extrabold tracking-[0.08em]">QUANTLAB</span>
        </Link>

        <div className="text-right">
          <div className="lab">
            System status:{" "}
            <span
              className={
                status === "offline"
                  ? "text-[var(--color-down)]"
                  : status === "calibrated"
                    ? "text-[var(--color-up)]"
                    : ""
              }
            >
              {status}
            </span>
          </div>
          <div className="lab mt-1 tracking-[0.1em]">UTC {clock}</div>
          {status === "offline" && (
            <div className="lab mt-1 normal-case tracking-[0.04em] text-[var(--color-down)]">
              cannot reach {API_URL}
            </div>
          )}
        </div>
      </div>

      <nav className="mt-5 flex border-y border-[var(--color-rule)]">
        {TABS.map((t) => {
          const on = path === t.href;
          return (
            <Link
              key={t.href}
              href={t.href}
              className={`border-r border-[var(--color-rule)] px-5 py-2.5 font-mono text-[9px] uppercase tracking-[0.15em] transition ${
                on
                  ? "bg-[var(--color-paper-raised)] text-[var(--color-ink)]"
                  : "text-[var(--color-ink-3)] hover:text-[var(--color-ink-2)]"
              }`}
            >
              {t.label}
            </Link>
          );
        })}
        {/* Shown but inert: LiveBroker raises NotImplementedError on purpose.
            Displaying the full pipeline is honest; enabling it would not be. */}
        <span
          title="Live execution is deliberately not implemented — see LiveBroker"
          className="cursor-not-allowed border-r border-[var(--color-rule)] px-5 py-2.5 font-mono text-[9px] uppercase tracking-[0.15em] text-[var(--color-rule)]"
        >
          Live Trade
        </span>
        <a
          href={`${API_URL}/docs`}
          target="_blank"
          rel="noreferrer"
          className="ml-auto px-5 py-2.5 font-mono text-[9px] uppercase tracking-[0.15em] text-[var(--color-ink-3)] transition hover:text-[var(--color-ink)]"
        >
          API ↗
        </a>
      </nav>
    </header>
  );
}
