import { parseTime } from "@/features/briefing/derive";
import type { MifAdapter, SessionHistoryEntry } from "./api";

/**
 * Everything the Sources screen states, derived from what the server returned.
 *
 * Kept out of the component for the reason the briefing's twin gives: each of
 * these figures is a claim that a particular thing wrote into this profile at a
 * particular time, and a claim nobody can read without a router and a query
 * client around it is a claim nobody checks.
 *
 * ONE RULE RUNS THROUGH ALL OF IT. Where a figure has no source, the function
 * returns null and the view omits the element. It never returns a zero, because
 * on this screen a confident zero would read as "this source has delivered
 * nothing" when the truth is "nothing recorded whether it did", and those are
 * different facts about a product.
 *
 * WHY EVERY SESSION FIGURE IS A FLOOR. A session summary is written when a
 * session ENDS — the hook's Stop handler posts it (hooks/memory-hook.ts). Its
 * per-event writes land as they happen. So a session that is still running, or
 * that was killed before Stop, has already put memories in the store and left
 * no row here. Measured on the live `claude-code` profile: newest memory
 * 10:49, newest session summary 07:37, on the same day. Nothing in this module
 * is named "last write" or "live" for that reason — the wording throughout is
 * "last completed session", which is what the data supports.
 */

/* ------------------------------------------------------------------ *
 * WHO WROTE THE SESSION SUMMARY
 * ------------------------------------------------------------------ */

/**
 * Two different things write a `session-summary` memory, and only one of them
 * is a source.
 *
 *   - The Claude Code Stop hook tags its summary `source:hook`
 *     (hooks/memory-hook.ts builds `["session-summary", "source:hook", ...]`).
 *     That is an outside writer: a session happened, and it delivered.
 *   - The server's own context compression writes one tagged `session-digest`
 *     (src/handlers/sessions.rs). That is this program summarising memories it
 *     already held. It brought nothing in.
 *
 * A summary carrying neither mark is `unmarked` and is NOT folded into either.
 * 7 of 230 entries on the live profile are in that state — early summaries
 * written before the hook stamped itself — and deciding after the fact which
 * writer produced them would be inventing provenance on the one screen whose
 * subject is provenance.
 */
export type SessionWriter = "hook" | "consolidation" | "unmarked";

export function classifySession(entry: SessionHistoryEntry): SessionWriter {
  if (entry.entities.includes("source:hook")) return "hook";
  if (entry.entities.includes("session-digest")) return "consolidation";
  return "unmarked";
}

/* ------------------------------------------------------------------ *
 * WHAT THE SESSION RECORD SAYS
 * ------------------------------------------------------------------ */

export interface SessionRecord {
  /** Every recorded session in this profile, from the handler's own `total` —
   *  not the length of the page. */
  total: number;
  /** How many entries are in hand. The "across the last N" figures below are
   *  over exactly these, and the surface prints N beside them. */
  page: number;
  /** The page covers every recorded session, so the figures below are the
   *  whole record rather than a slice of it. Gates the span statement: an
   *  earliest date taken from a page is the earliest of a page. */
  complete: boolean;
  /** Sessions in the page each writer is responsible for. */
  byWriter: Record<SessionWriter, number>;
  /** Newest `created_at` in the page, or null on an empty record. */
  latest: string | null;
  /** Oldest `created_at`, only when the page is the whole record. */
  earliest: string | null;
  /** Memories the hook reported creating, summed over the entries that
   *  reported a count. */
  memoriesCreated: number;
  /** How many entries reported one. Printed beside the sum, because 95 of 230
   *  reporting is the difference between a total and a sample. */
  reported: number;
  /** Session time, summed over the entries that reported a duration. */
  recordedSecs: number;
}

/**
 * Read a page of session history.
 *
 * `latest` and `earliest` are COMPUTED, not read off the ends. The handler does
 * sort newest-first before paging, and relying on that would work today — but
 * this is the one figure a reader will check against their own memory of when
 * they last worked, and a max is three lines and cannot be wrong.
 *
 * `memoriesCreated` sums only entries that reported a count, and `reported`
 * carries how many did. Treating a missing count as zero would silently drag
 * the total down by whatever the 135 unreported sessions did; treating the sum
 * as complete would state a total that is a sample. Both numbers travel
 * together so the view cannot print one without the other.
 */
export function readSessions(
  entries: readonly SessionHistoryEntry[],
  total: number,
): SessionRecord {
  const byWriter: Record<SessionWriter, number> = { hook: 0, consolidation: 0, unmarked: 0 };
  let latest: string | null = null;
  let latestAt = -Infinity;
  let earliest: string | null = null;
  let earliestAt = Infinity;
  let memoriesCreated = 0;
  let reported = 0;
  let recordedSecs = 0;

  for (const entry of entries) {
    byWriter[classifySession(entry)] += 1;

    // A timestamp the platform cannot parse contributes to NEITHER bound and is
    // not carried through as a candidate. `new Date(x)` yields NaN for anything
    // malformed, and NaN loses every comparison silently — which would leave the
    // bound sitting on whichever row happened to come first.
    const at = parseTime(entry.created_at);
    if (at !== null) {
      if (at > latestAt) {
        latestAt = at;
        latest = entry.created_at;
      }
      if (at < earliestAt) {
        earliestAt = at;
        earliest = entry.created_at;
      }
    }

    if (entry.memories_created !== null) {
      memoriesCreated += entry.memories_created;
      reported += 1;
    }
    if (entry.duration_secs !== null) recordedSecs += entry.duration_secs;
  }

  const complete = entries.length >= total;

  return {
    total,
    page: entries.length,
    complete,
    byWriter,
    latest,
    earliest: complete ? earliest : null,
    memoriesCreated,
    reported,
    recordedSecs,
  };
}

/* ------------------------------------------------------------------ *
 * WORDS AND FIGURES
 * ------------------------------------------------------------------ */

const HOUR_SECS = 3600;

/**
 * A span of recorded session time, or null when nothing reported one.
 *
 * Null rather than "0m", for the rule at the top: no session on this profile
 * reporting its duration and every session lasting no time are different
 * things, and "0m" says the second.
 *
 * Rounded to whole minutes below an hour and one decimal hour above it. This
 * is a total across dozens of sessions read for scale — nobody acts on the
 * seconds, and printing them implies a precision the sum does not have, since
 * it omits every session that reported nothing.
 */
export function formatRecorded(seconds: number): string | null {
  if (!Number.isFinite(seconds) || seconds <= 0) return null;
  if (seconds < HOUR_SECS) return `${Math.max(1, Math.round(seconds / 60))}m`;
  return `${(seconds / HOUR_SECS).toFixed(1)}h`;
}

/**
 * A session's own duration, for a row in the recent list, or null when it
 * reported none.
 *
 * Seconds are kept below a minute here and are not in `formatRecorded`,
 * because these are single sessions and a 26-second one is a real thing that
 * happened — the live profile's newest session is exactly that. Rounding it to
 * "0m" would make the shortest sessions look like recording failures.
 */
export function formatSessionLength(seconds: number | null): string | null {
  if (seconds === null || !Number.isFinite(seconds) || seconds <= 0) return null;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < HOUR_SECS) return `${Math.round(seconds / 60)}m`;
  const hours = Math.floor(seconds / HOUR_SECS);
  const minutes = Math.round((seconds % HOUR_SECS) / 60);
  return minutes === 0 ? `${hours}h` : `${hours}h ${minutes}m`;
}

/**
 * The import formats, ordered for a list that does not move between reads.
 *
 * Sorted by the human name rather than left in registry order: the registry's
 * order is construction order in `AdapterRegistry::new()` and carries no
 * meaning a reader could use, while a stable alphabetical list can be compared
 * against the same list on another install.
 *
 * An adapter with a blank name is DROPPED rather than rendered as an empty
 * chip. The name is the whole content of the item; a nameless one is a row
 * that says nothing and looks like a rendering fault.
 *
 * `filter` FIRST IS WHAT MAKES THE SORT SAFE, and it is load-bearing rather
 * than incidental: the registry arrives as a react-query cache entry shared
 * with anything else reading it, and `Array.prototype.sort` mutates in place.
 * A defensive `.slice()` stood here until a mutation sweep showed it could be
 * deleted with every test still green — `filter` had already copied, so the
 * slice was dead code claiming to be a safeguard. The copy is named where it
 * actually happens instead.
 */
export function readableFormats(adapters: readonly MifAdapter[]): MifAdapter[] {
  return adapters
    .filter((adapter) => adapter.name.trim().length > 0)
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/**
 * A count with thousands separators.
 *
 * The figures here reach five digits — 10,758 memories across the recorded
 * sessions of one profile — and an unseparated `10758` is read as a token
 * rather than as a quantity, which is the one thing this screen needs its
 * numbers to be.
 */
export function formatCount(value: number): string {
  return value.toLocaleString();
}
