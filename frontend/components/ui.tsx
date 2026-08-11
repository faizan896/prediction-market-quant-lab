"use client";

import type { ReactNode } from "react";

/* ---------------------------------------------------------------- primitives */

export function Lab({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`lab ${className}`}>{children}</span>;
}

export function Panel({
  label,
  title,
  right,
  children,
  flush = false,
}: {
  label?: string;
  title?: string;
  right?: ReactNode;
  children: ReactNode;
  flush?: boolean;
}) {
  return (
    <section className="border border-[var(--color-rule)] bg-[var(--color-paper-raised)]">
      {(label || title || right) && (
        <header className="flex items-end justify-between gap-4 border-b border-[var(--color-rule)] px-4 py-3">
          <div>
            {label && <Lab>{label}</Lab>}
            {title && (
              <h2 className="mt-1 text-[15px] font-bold uppercase tracking-[-0.02em]">{title}</h2>
            )}
          </div>
          {right}
        </header>
      )}
      <div className={flush ? "" : "p-4"}>{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  hint,
  tone = "ink",
  size = "md",
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "ink" | "accent" | "up" | "down" | "muted";
  size?: "md" | "lg";
}) {
  const color = {
    ink: "text-[var(--color-ink)]",
    accent: "text-[var(--color-accent)]",
    up: "text-[var(--color-up)]",
    down: "text-[var(--color-down)]",
    muted: "text-[var(--color-ink-3)]",
  }[tone];
  return (
    <div className="border border-[var(--color-rule)] bg-[var(--color-paper-raised)] px-3 py-2.5">
      <Lab>{label}</Lab>
      <div
        className={`num display mt-1.5 ${color} ${size === "lg" ? "text-[30px]" : "text-[21px]"}`}
      >
        {value}
      </div>
      {hint && <div className="lab mt-2 normal-case tracking-[0.06em]">{hint}</div>}
    </div>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "solid",
  type = "button",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "solid" | "ghost";
  type?: "button" | "submit";
  title?: string;
}) {
  const base =
    "font-mono text-[9.5px] uppercase tracking-[0.15em] px-3.5 py-2 transition disabled:opacity-35 disabled:cursor-not-allowed";
  const style =
    variant === "solid"
      ? "bg-[var(--color-ink)] text-[var(--color-paper)] hover:bg-[var(--color-accent)]"
      : "border border-[var(--color-rule)] text-[var(--color-ink-2)] hover:border-[var(--color-ink)] hover:text-[var(--color-ink)]";
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title} className={`${base} ${style}`}>
      {children}
    </button>
  );
}

export function Toggle({
  children,
  on,
  onClick,
}: {
  children: ReactNode;
  on: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`font-mono text-[9px] uppercase tracking-[0.14em] px-2.5 py-1.5 border transition ${
        on
          ? "border-[var(--color-ink)] bg-[var(--color-ink)] text-[var(--color-paper)]"
          : "border-[var(--color-rule)] text-[var(--color-ink-3)] hover:border-[var(--color-ink-2)] hover:text-[var(--color-ink-2)]"
      }`}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <Lab>{label}</Lab>
      <div className="mt-1.5">{children}</div>
      {hint && (
        <p className="mt-1.5 text-[10px] leading-snug text-[var(--color-ink-3)]">{hint}</p>
      )}
    </label>
  );
}

export function NumberInput({
  value,
  onChange,
  step = 0.01,
  min,
  max,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  return (
    <input
      type="number"
      value={value}
      step={step}
      min={min}
      max={max}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (Number.isFinite(v)) onChange(v);
      }}
      className="num w-full border border-[var(--color-rule)] bg-[var(--color-paper)] px-2 py-1.5 text-[12.5px] outline-none focus:border-[var(--color-ink)]"
    />
  );
}

export function Select<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly { value: T; label: string }[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
      className="num w-full border border-[var(--color-rule)] bg-[var(--color-paper)] px-2 py-1.5 text-[12px] outline-none focus:border-[var(--color-ink)]"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

/* -------------------------------------------------------------------- table */

export type Col = { key: string; label: string; align?: "left" | "right"; w?: string };

export function Table({ cols, children }: { cols: Col[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-[var(--color-rule)]">
            {cols.map((c) => (
              <th
                key={c.key}
                style={c.w ? { width: c.w } : undefined}
                className={`px-4 py-2.5 ${c.align === "right" ? "text-right" : "text-left"}`}
              >
                <Lab>{c.label}</Lab>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Row({ children, onClick }: { children: ReactNode; onClick?: () => void }) {
  return (
    <tr
      onClick={onClick}
      className={`border-b border-[var(--color-rule-soft)] last:border-0 ${
        onClick ? "cursor-pointer hover:bg-[var(--color-paper-sunk)]" : ""
      }`}
    >
      {children}
    </tr>
  );
}

export function Td({
  children,
  align = "right",
  mono = true,
  tone,
  className = "",
  title,
}: {
  children: ReactNode;
  align?: "left" | "right";
  mono?: boolean;
  tone?: "ink" | "muted" | "accent" | "up" | "down";
  className?: string;
  title?: string;
}) {
  const color = tone
    ? {
        ink: "text-[var(--color-ink)]",
        muted: "text-[var(--color-ink-3)]",
        accent: "text-[var(--color-accent)]",
        up: "text-[var(--color-up)]",
        down: "text-[var(--color-down)]",
      }[tone]
    : "";
  return (
    <td
      title={title}
      className={`px-4 py-2.5 text-[12.5px] ${mono ? "num" : ""} ${
        align === "right" ? "text-right" : "text-left"
      } ${color} ${className}`}
    >
      {children}
    </td>
  );
}

/* ------------------------------------------------------------------- states */

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="px-6 py-10 text-center">
      <div className="text-[14px]">{title}</div>
      {children && (
        <div className="mx-auto mt-2.5 max-w-[470px] text-[11.5px] leading-relaxed text-[var(--color-ink-2)]">
          {children}
        </div>
      )}
    </div>
  );
}

export function Note({ label, children }: { label?: string; children: ReactNode }) {
  return (
    <p className="text-[11px] leading-relaxed text-[var(--color-ink-2)]">
      {label && (
        <>
          <Lab>[ {label} ]</Lab>{" "}
        </>
      )}
      {children}
    </p>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="border border-[var(--color-down)] bg-[var(--color-accent-soft)] px-3.5 py-2.5 text-[11.5px] text-[var(--color-down)]">
      <Lab className="!text-[var(--color-down)]">[ Error ]</Lab> {message}
    </div>
  );
}
