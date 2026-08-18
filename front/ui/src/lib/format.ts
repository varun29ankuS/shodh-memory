/** Shared display formatting. Numbers only — no invented data, ever. */

/** 1234 → "1.2k", 1234567 → "1.2M". Token counts, not bytes. */
export function formatTokens(count: number): string {
  if (!Number.isFinite(count)) return "0";
  if (count < 1000) return String(Math.round(count));
  if (count < 1_000_000) return `${(count / 1000).toFixed(count < 10_000 ? 1 : 0)}k`;
  return `${(count / 1_000_000).toFixed(1)}M`;
}

/**
 * USD cost from pi's per-token pricing. Sub-cent costs are the norm for a
 * single message, so two significant decimals below a cent instead of
 * rounding everything to "$0.00". Zero (local models) renders as null so
 * callers can omit it — a $0.00 label on a local model implies metering that
 * is not happening.
 */
export function formatCost(usd: number): string | null {
  if (!Number.isFinite(usd) || usd <= 0) return null;
  if (usd >= 0.1) return `$${usd.toFixed(2)}`;
  if (usd >= 0.01) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(4)}`;
}

/** Relative day, mirroring ResultList's scale ("today" … "2y ago"). */
export function relativeDay(iso: string): string {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const minutes = Math.floor((Date.now() - then.getTime()) / 60_000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days === 1) return "yesterday";
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

/** Context window: 200000 → "200k ctx". */
export function formatContext(tokens: number): string {
  return `${formatTokens(tokens)} ctx`;
}

/**
 * A count with thousands separators.
 *
 * The figures here reach five digits — 19,553 memories in one profile — and an
 * unseparated `19553` is read as a token rather than as a quantity, which is
 * the one thing a status strip needs its numbers to be.
 *
 * THE GROUPING IS HAND-ROLLED, NOT `toLocaleString()`, and that is the whole
 * point of this function existing. `toLocaleString()` with no locale argument
 * asks the runtime what locale it is in, and answers in that locale's
 * numbering system — a machine resolving to a non-latn default renders
 * `formatCount(230)` as something other than "230". That is not hypothetical:
 * the covering test failed once in roughly six runs and passed in isolation,
 * which is what an ICU default resolving differently under parallel workers
 * looks like. A displayed figure must not depend on which thread formatted it.
 *
 * The separator is therefore a decision this product makes rather than one it
 * inherits: a comma every three digits, identically on every machine, in every
 * test run, in the screenshot and in the deck.
 */
export function formatCount(value: number): string {
  if (!Number.isFinite(value)) return "0";
  const rounded = Math.round(value);
  const digits = Math.abs(rounded).toString();
  const grouped = digits.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return rounded < 0 ? `-${grouped}` : grouped;
}

/**
 * Whether a listing has seen everything it is about to make a claim about.
 *
 * Every listing surface in this product reads ONE capped page — 500 rows — and
 * then draws conclusions on screen. On a profile of 19,553 that page is 2.6% of
 * the corpus, and a conclusion drawn from it is a conclusion about the page.
 * The distinction this predicate draws is the difference between "none exist"
 * and "we did not look", which is the one claim this product is not allowed to
 * blur.
 *
 * `read >= total` is a complete read and the strong claim is then earned — on a
 * profile smaller than the page cap, the page IS the corpus and no denominator
 * needs to be printed at all. That is why this returns a boolean rather than
 * always-on hedging copy: an honest screen gets SHORTER when it has the
 * evidence, not longer.
 *
 * A missing figure reports as NOT partial, and the comparison alone is what
 * does that — any `<` against NaN is false. An explicit `Number.isFinite`
 * guard was written here first and then removed: no mutation of it could be
 * made to fail a test, which is the definition of dead code claiming to be a
 * safeguard. The behaviour it was meant to provide is covered directly.
 */
export function isPartialRead(read: number, total: number): boolean {
  return read < total;
}

/**
 * The denominator, as a strip token: "500 of 19,553 read".
 *
 * Null when the read was complete, so a call site can drop the token entirely
 * rather than print a fraction whose two halves are the same number. Callers
 * render it beside the figures it qualifies; the verb is "read" rather than
 * "shown" because it names what the request fetched, which is what the
 * conclusions above it were actually computed over.
 */
export function sampleNote(read: number, total: number): string | null {
  if (!isPartialRead(read, total)) return null;
  return `${formatCount(read)} of ${formatCount(total)} read`;
}
