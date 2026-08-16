import type { CorpusMemory } from "@/lib/api/corpus";
import type { UniverseStar } from "@/lib/api/graph";
import { countryAt, isInIndia } from "@/lib/atlas";
import type { DotMapPoint } from "./DotMap";

/**
 * Everything the briefing states, derived from what the store returned.
 *
 * Pure and separate from the view for one reason: every figure on that screen
 * is a claim about a real corpus, and a claim that cannot be read on its own —
 * without a canvas, a router and a query client around it — is a claim nobody
 * checks. Nothing here invents a value. Where a figure has no source, the
 * function returns `null` and the view omits the element rather than printing
 * a zero, because a confident zero and an unanswered question look identical
 * and only one of them is true.
 */

// =============================================================================
// TIME
// =============================================================================

/** Milliseconds, or `null` for a timestamp the server sent in a shape this
 *  cannot read. Never `NaN` escaping into arithmetic. */
export function parseTime(iso: string | undefined | null): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : null;
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/**
 * How long ago, in the shortest form that is still unambiguous.
 *
 * These sit in a right-hand gutter beside a line of text and are read at a
 * glance, so they are terse — "2h", "yest", "5d". "yest" earns its own word
 * rather than collapsing into "1d" because yesterday is a place a person
 * remembers being, and a day count is not.
 */
export function shortAgo(iso: string, now: number): string | null {
  const t = parseTime(iso);
  if (t === null) return null;
  const delta = now - t;
  // A clock skew between this machine and the server is not worth a wrong
  // word: anything in the future reads as just-written.
  if (delta < MINUTE) return "now";
  if (delta < HOUR) return `${Math.floor(delta / MINUTE)}m`;
  if (delta < DAY) return `${Math.floor(delta / HOUR)}h`;
  if (delta < 2 * DAY) return "yest";
  if (delta < 14 * DAY) return `${Math.floor(delta / DAY)}d`;
  if (delta < 60 * DAY) return `${Math.floor(delta / (7 * DAY))}w`;
  return `${Math.floor(delta / (30 * DAY))}mo`;
}

/** The same quantity spelled out, for the footer's one prose statement. */
export function longAgo(iso: string, now: number): string | null {
  const t = parseTime(iso);
  if (t === null) return null;
  const delta = Math.max(0, now - t);
  if (delta < MINUTE) return "just now";
  if (delta < HOUR) {
    const m = Math.floor(delta / MINUTE);
    return `${m} min ago`;
  }
  if (delta < DAY) {
    const h = Math.floor(delta / HOUR);
    return h === 1 ? "an hour ago" : `${h} hours ago`;
  }
  const d = Math.floor(delta / DAY);
  return d === 1 ? "yesterday" : `${d} days ago`;
}

// =============================================================================
// THE CORPUS, AS A SENTENCE
// =============================================================================

/**
 * The window the corpus covers — "from March 2024" — or `null`.
 *
 * ONLY WHEN THE WHOLE CORPUS IS IN HAND. `GET /api/list` caps at
 * `CORPUS_LIMIT`, so on a larger store the oldest row present is the oldest of
 * a PAGE and not the oldest of the corpus. Printing it would state a start
 * date that is simply wrong, and it would be wrong in the direction that
 * flatters the product — a corpus looks younger and denser than it is. So the
 * span is claimed only when `memories.length === total`.
 *
 * The listing is also not reliably ordered — `defence-live` returns its newest
 * row first and a LATER row after it — so the bound is computed rather than
 * read off either end.
 */
export function corpusSpan(
  memories: CorpusMemory[],
  total: number,
  now: number,
): { from: string } | null {
  if (memories.length === 0 || memories.length !== total) return null;
  let oldest = Infinity;
  for (const m of memories) {
    const t = parseTime(m.created_at);
    if (t !== null && t < oldest) oldest = t;
  }
  if (!Number.isFinite(oldest)) return null;
  // A corpus written entirely this month has no span worth stating; "from
  // August 2026 to today" is a sentence that says nothing.
  const start = new Date(oldest);
  const nowDate = new Date(now);
  if (
    start.getFullYear() === nowDate.getFullYear() &&
    start.getMonth() === nowDate.getMonth()
  ) {
    return null;
  }
  return {
    from: start.toLocaleDateString(undefined, { month: "long", year: "numeric" }),
  };
}

/** The newest write in the corpus, for the footer. */
export function lastWrite(memories: CorpusMemory[]): string | null {
  let newest = -Infinity;
  let iso: string | null = null;
  for (const m of memories) {
    const t = parseTime(m.created_at);
    if (t !== null && t > newest) {
      newest = t;
      iso = m.created_at;
    }
  }
  return iso;
}

// =============================================================================
// SINCE YOU LEFT
// =============================================================================

/**
 * What arrived while you were away.
 *
 * ONE CLAUSE, NOT THREE. The mockup's line reads "14 new memories, 2 new
 * links, 1 thing worth a look", and only the first of those has a source: a
 * `GravitationalConnection` carries a strength and a tier but no timestamp, so
 * "new links" cannot be computed from anything the server sends and is
 * dropped rather than approximated from a total that also moves when an edge
 * decays out.
 *
 * `null` on a first visit. There is no "since" without a previous one, and
 * treating an absent mark as the beginning of time would greet a new person
 * with their entire corpus reported as new.
 */
export function sinceYouLeft(
  memories: CorpusMemory[],
  lastVisit: number | null,
  now: number,
): { when: string; added: number } | null {
  if (lastVisit === null || lastVisit > now) return null;
  const when = new Date(lastVisit);
  const elapsed = now - lastVisit;
  // Inside a week a weekday is the more human handle — "on Tuesday" is a
  // memory, "on 11 August" is a lookup. Past that the weekday is ambiguous and
  // the date is the only honest form.
  const label =
    elapsed < 6 * DAY
      ? when.toLocaleDateString(undefined, { weekday: "long" })
      : when.toLocaleDateString(undefined, { day: "numeric", month: "long" });
  let added = 0;
  for (const m of memories) {
    const t = parseTime(m.created_at);
    if (t !== null && t > lastVisit) added += 1;
  }
  return { when: label, added };
}

// =============================================================================
// THE ONTOLOGY
// =============================================================================

export interface OntologyBand {
  label: string;
  n: number;
  /** True for the one type holding most of the corpus. */
  dominant: boolean;
  /** Its share, as a whole percent. Present only on the dominant band. */
  share?: number;
}

/** One type holding this much of a corpus is not a fact about the world, it is
 *  a fact about the typer — and it is the single most useful thing this screen
 *  can tell someone about the quality of what they are about to search. */
const DOMINANCE = 0.6;

/** How many types are named before the rest are rolled up. Four plus a
 *  remainder fits one line at the width this sits in, and a fifth rare type
 *  named individually says less than the count of everything else. */
const NAMED_BANDS = 4;

/**
 * The distribution, stated rather than drawn.
 *
 * THE ROLL-UP HAS TO SUM. A truncated top-five leaves a reader adding five
 * numbers that do not reach the total and quietly wondering what is missing;
 * an explicit "other" makes the line closed, so the numbers on screen ARE the
 * corpus. Entities with no label are counted as untyped rather than dropped —
 * an unlabelled entity is a typing outcome, and hiding it would flatter
 * exactly the measure this line exists to expose.
 */
export function ontology(stars: UniverseStar[]): OntologyBand[] {
  if (stars.length === 0) return [];
  const tally = new Map<string, number>();
  for (const s of stars) {
    const t = (s.entity_type ?? "untyped").toLowerCase();
    tally.set(t, (tally.get(t) ?? 0) + 1);
  }
  const total = stars.length;
  const sorted = [...tally.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));

  const named = sorted.slice(0, NAMED_BANDS);
  const rest = sorted.slice(NAMED_BANDS);
  const bands: { label: string; n: number }[] = named.map(([label, n]) => ({ label, n }));
  if (rest.length > 0) {
    bands.push({ label: "other", n: rest.reduce((a, [, n]) => a + n, 0) });
  }

  const topCount = sorted[0][1];
  const topLabel = sorted[0][0];
  const lopsided = topCount / total >= DOMINANCE;

  return bands
    .sort((a, b) => b.n - a.n)
    .map((b) => {
      const dominant = lopsided && b.label === topLabel;
      return dominant
        ? { ...b, dominant, share: Math.round((topCount / total) * 100) }
        : { ...b, dominant: false };
    });
}

// =============================================================================
// QUESTIONS THIS CORPUS CAN ANSWER
// =============================================================================

/**
 * Three real questions, built from this corpus's own most-mentioned entities.
 *
 * A person who has never seen this product learns what it answers by reading
 * examples of an answerable question. Interface copy does not achieve that —
 * "search your memory" states the mechanism, not the shape of a question that
 * will work. Built from the most-mentioned entities rather than the first
 * three: an entity the corpus barely knows makes a question that returns
 * nothing, which teaches the opposite of what this is for.
 */
export function suggestedQuestions(stars: UniverseStar[]): string[] {
  const names = [...stars]
    .sort((a, b) => (b.mention_count ?? 0) - (a.mention_count ?? 0))
    .map((s) => s.name.trim())
    .filter((n) => n.length > 0)
    .slice(0, 3);
  const out: string[] = [];
  if (names[0]) out.push(`What do we know about ${names[0]}?`);
  if (names[0] && names[1]) out.push(`How is ${names[0]} connected to ${names[1]}?`);
  if (names[2]) out.push(`What changed about ${names[2]}?`);
  return out;
}

// =============================================================================
// PLACES
// =============================================================================

/**
 * Coordinates are quantised to this many degrees before being counted as one
 * site — roughly 11 km at the equator.
 *
 * A robot patrolling a warehouse writes a memory every few metres, and drawn
 * raw that is thirty overlapping discs where the corpus has one place. The
 * grid is coarse enough to collapse a site and fine enough to keep two cities
 * apart, and the marks are sized by the count it recovers, so nothing is lost
 * by merging — the place gets bigger instead of denser.
 */
const SITE_DEGREES = 0.1;

export interface Places {
  /** One mark per site, ready for the map. */
  sites: DotMapPoint[];
  /** Memories carrying a usable coordinate. */
  located: number;
  /** Of those, the ones inside India's official boundary. */
  inIndia: number;
  /** Sites inside India, for the India map. */
  indiaSites: DotMapPoint[];
  /** The countries these coordinates fall in, most memories first. */
  countries: { name: string; n: number }[];
}

/**
 * Every located memory, aggregated into sites and named by country.
 *
 * COORDINATE ORDER IS A REAL HAZARD. The wire carries `[lat, lon, alt]`
 * (src/validation.rs) and every d3-geo entry point takes `[lon, lat]`. The
 * swap happens exactly once, here, because transposed coordinates do not
 * throw — they silently plot Baltimore in China.
 *
 * A coordinate outside its own domain is corrupt, not interesting: plotting it
 * would put a mark on a meaningless part of the map and imply the data is
 * fine, so it is dropped from every figure including the located count.
 */
export function places(memories: CorpusMemory[]): Places {
  /* The quantised position is for GROUPING DOTS. It must never decide which
     country a memory is in.

     Rounding to a site grid moves a coastal point by up to half a cell, which
     is enough to push San Francisco into the Pacific — so the lookup returned
     open water and the memory vanished from the country tally. It survived
     only because the 1:110m basemap's coastline was blocky enough to still
     contain the moved point; the 1:50m coastline is accurate and correctly
     says that spot is sea. A latent bug that a better basemap exposed.

     So each bucket keeps a REPRESENTATIVE TRUE COORDINATE — the first real one
     that landed in it — and every geometric question is asked of that. */
  const bucket = new Map<
    string,
    { lon: number; lat: number; count: number; trueLon: number; trueLat: number }
  >();
  let located = 0;

  for (const m of memories) {
    const geo = m.geo_location;
    if (!geo) continue;
    const [lat, lon] = geo;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) continue;
    located += 1;
    const qlat = Math.round(lat / SITE_DEGREES) * SITE_DEGREES;
    const qlon = Math.round(lon / SITE_DEGREES) * SITE_DEGREES;
    const key = `${qlat.toFixed(2)},${qlon.toFixed(2)}`;
    const at = bucket.get(key);
    if (at) at.count += 1;
    else bucket.set(key, { lon: qlon, lat: qlat, count: 1, trueLon: lon, trueLat: lat });
  }

  const sites = [...bucket.values()].sort((a, b) => b.count - a.count);

  const byCountry = new Map<string, number>();
  const indiaSites: DotMapPoint[] = [];
  let inIndia = 0;
  for (const s of sites) {
    if (isInIndia(s.trueLon, s.trueLat)) {
      indiaSites.push(s);
      inIndia += s.count;
    }
    const name = countryAt(s.trueLon, s.trueLat);
    if (name) byCountry.set(name, (byCountry.get(name) ?? 0) + s.count);
  }

  return {
    sites,
    located,
    inIndia,
    indiaSites,
    countries: [...byCountry.entries()]
      .map(([name, n]) => ({ name, n }))
      .sort((a, b) => b.n - a.n || a.name.localeCompare(b.name)),
  };
}
