import { describe, expect, it } from "vitest";
import type { CorpusMemory } from "@/lib/api/corpus";
import type { UniverseStar } from "@/lib/api/graph";
import { corpusSpan, lastWrite, ontology, places, shortAgo, sinceYouLeft } from "./derive";

/**
 * The briefing states figures about a real corpus. Every one of them is a
 * claim, and the claims worth pinning are the ones where being wrong is
 * INVISIBLE — a span computed off a truncated page, a roll-up that does not
 * sum, a "since you left" that counts the whole corpus. Those all render as a
 * perfectly plausible number.
 */

const NOW = Date.parse("2026-08-14T18:20:00Z");
const at = (iso: string, extra: Partial<CorpusMemory> = {}): CorpusMemory => ({
  id: iso,
  content: "x",
  content_truncated: false,
  content_length: 1,
  memory_type: "Observation",
  importance: 0.5,
  tags: [],
  created_at: iso,
  tier: "Working",
  ...extra,
});

const star = (entity_type: string | null, i = 0): UniverseStar => ({
  id: `${entity_type}-${i}`,
  name: `n${i}`,
  entity_type,
  salience: 0,
  mention_count: 0,
  is_proper_noun: false,
  position: { x: 0, y: 0, z: 0 },
  color: "",
  size: 0,
});

describe("shortAgo", () => {
  it("gives yesterday its own word", () => {
    expect(shortAgo("2026-08-13T18:00:00Z", NOW)).toBe("yest");
    expect(shortAgo("2026-08-12T18:00:00Z", NOW)).toBe("2d");
  });

  it("treats a future timestamp as just written rather than negative", () => {
    // Clock skew between this machine and the server is not worth "-3m".
    expect(shortAgo("2026-08-14T19:00:00Z", NOW)).toBe("now");
  });

  it("returns null rather than NaN for an unreadable timestamp", () => {
    expect(shortAgo("not a date", NOW)).toBeNull();
  });
});

describe("corpusSpan", () => {
  const window = [at("2024-03-01T00:00:00Z"), at("2026-08-01T00:00:00Z")];

  it("claims a span only when the whole corpus is in hand", () => {
    // 2 rows of a 900-row corpus: the oldest present is the oldest of a PAGE.
    expect(corpusSpan(window, 900, NOW)).toBeNull();
    expect(corpusSpan(window, 2, NOW)?.from).toContain("2024");
  });

  it("does not compute the bound from the array order", () => {
    // /api/list is NOT reliably newest-first — defence-live returns a later
    // row after its first one.
    const shuffled = [window[1], window[0]];
    expect(corpusSpan(shuffled, 2, NOW)?.from).toEqual(corpusSpan(window, 2, NOW)?.from);
  });

  it("says nothing when the corpus was all written this month", () => {
    expect(corpusSpan([at("2026-08-02T00:00:00Z")], 1, NOW)).toBeNull();
  });
});

describe("lastWrite", () => {
  it("finds the newest write regardless of position", () => {
    const m = [at("2024-01-01T00:00:00Z"), at("2026-05-05T00:00:00Z"), at("2025-01-01T00:00:00Z")];
    expect(lastWrite(m)).toBe("2026-05-05T00:00:00Z");
  });

  it("is null on an empty corpus, so the footer omits the clause", () => {
    expect(lastWrite([])).toBeNull();
  });
});

describe("sinceYouLeft", () => {
  const memories = [at("2026-08-14T10:00:00Z"), at("2026-08-01T00:00:00Z")];

  it("says nothing on a first visit rather than reporting the whole corpus", () => {
    expect(sinceYouLeft(memories, null, NOW)).toBeNull();
  });

  it("counts only what arrived after the mark", () => {
    expect(sinceYouLeft(memories, Date.parse("2026-08-13T00:00:00Z"), NOW)?.added).toBe(1);
  });

  it("distinguishes 'nothing new' from 'no previous visit'", () => {
    const r = sinceYouLeft(memories, Date.parse("2026-08-14T12:00:00Z"), NOW);
    expect(r).not.toBeNull();
    expect(r!.added).toBe(0);
  });
});

describe("ontology", () => {
  const corpus = [
    ...Array.from({ length: 831 }, (_, i) => star("Technology", i)),
    ...Array.from({ length: 90 }, (_, i) => star("Organization", i)),
    ...Array.from({ length: 46 }, (_, i) => star("Task", i)),
    ...Array.from({ length: 31 }, (_, i) => star("Location", i)),
    ...Array.from({ length: 9 }, (_, i) => star("Person", i)),
    star(null),
  ];

  it("closes the line: the bands sum to the corpus", () => {
    const bands = ontology(corpus);
    expect(bands.reduce((a, b) => a + b.n, 0)).toBe(corpus.length);
  });

  it("rolls the tail into one 'other' rather than truncating it", () => {
    const bands = ontology(corpus);
    expect(bands.map((b) => b.label)).toContain("other");
    // Person (9) + untyped (1) — the two beyond the four named bands.
    expect(bands.find((b) => b.label === "other")!.n).toBe(10);
  });

  it("marks the type that owns the corpus, with its share", () => {
    const bands = ontology(corpus);
    const top = bands.find((b) => b.dominant)!;
    expect(top.label).toBe("technology");
    expect(top.share).toBe(82);
    expect(bands.filter((b) => b.dominant)).toHaveLength(1);
  });

  it("marks nothing when the distribution is not lopsided", () => {
    const even = [star("Technology"), star("Organization", 1), star("Person", 2)];
    expect(ontology(even).some((b) => b.dominant)).toBe(false);
  });

  it("counts unlabelled entities rather than hiding a typing failure", () => {
    expect(ontology([star(null), star(null, 1)])[0]).toMatchObject({ label: "untyped", n: 2 });
  });
});

describe("places", () => {
  const sf = (lat: number, lon: number) => at(`${lat},${lon}`, { geo_location: [lat, lon, 0] });

  it("reads [lat, lon] from the wire and never transposes it", () => {
    // Transposed, San Francisco lands in Xinjiang and would report China.
    const p = places([sf(37.7785, -122.4179)]);
    expect(p.countries[0].name).toContain("United States");
    expect(p.located).toBe(1);
  });

  it("collapses a patrol into one site and keeps the count", () => {
    const patrol = [sf(37.7785, -122.4179), sf(37.7755, -122.4187), sf(37.777, -122.4157)];
    const p = places(patrol);
    expect(p.located).toBe(3);
    expect(p.sites).toHaveLength(1);
    expect(p.sites[0].count).toBe(3);
  });

  it("drops a corrupt coordinate from every figure, not just from the map", () => {
    const p = places([sf(999, 0), at("no geo")]);
    expect(p.located).toBe(0);
    expect(p.sites).toHaveLength(0);
  });

  it("answers India from the official boundary, including Aksai Chin", () => {
    // 35.1N 79.5E — inside Aksai Chin, which Natural Earth places outside
    // India. The LGD boundary is the whole reason that asset is vendored.
    expect(places([sf(35.1, 79.5)]).inIndia).toBe(1);
    // 27.5N 93.5E — central Arunachal Pradesh.
    expect(places([sf(27.5, 93.5)]).inIndia).toBe(1);
    // 34.0N 74.8E — Srinagar.
    expect(places([sf(34.0, 74.8)]).inIndia).toBe(1);
  });

  it("leaves a memory in open water unnamed rather than guessing the nearest country", () => {
    expect(places([sf(0, -140)]).countries).toHaveLength(0);
  });
});
