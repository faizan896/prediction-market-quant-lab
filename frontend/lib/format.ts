/** Formatting helpers. Every one of them tolerates null/NaN, because the API
 *  returns null for any metric that isn't finite (JSON has no NaN). */

export const DASH = "—";

export function pct(x: number | null | undefined, digits = 1): string {
  if (x == null || !Number.isFinite(x)) return DASH;
  return `${(x * 100).toFixed(digits)}%`;
}

export function signedPct(x: number | null | undefined, digits = 1): string {
  if (x == null || !Number.isFinite(x)) return DASH;
  // Unicode minus, not hyphen: it shares a width with "+" so signed columns
  // stay aligned under the eye.
  const s = (x * 100).toFixed(digits);
  return x >= 0 ? `+${s}%` : `−${s.replace("-", "")}%`;
}

export function money(x: number | null | undefined, digits = 0): string {
  if (x == null || !Number.isFinite(x)) return DASH;
  const sign = x < 0 ? "-" : "";
  return `${sign}$${Math.abs(x).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
}

export function num(x: number | null | undefined, digits = 2): string {
  if (x == null || !Number.isFinite(x)) return DASH;
  return x.toFixed(digits);
}

export function compact(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return DASH;
  return Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(x);
}

export function price(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return DASH;
  return `${(x * 100).toFixed(1)}¢`;
}

export function hours(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return DASH;
  if (x < 48) return `${Math.round(x)}h`;
  return `${Math.round(x / 24)}d`;
}

export function ago(iso: string | null | undefined): string {
  if (!iso) return DASH;
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return DASH;
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  if (mins < 1440) return `${Math.round(mins / 60)}h ago`;
  return `${Math.round(mins / 1440)}d ago`;
}

/** Green above zero, red below, neutral at exactly zero or null. */
export function tone(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x) || x === 0) return "text-slate-300";
  return x > 0 ? "text-emerald-400" : "text-red-400";
}
