import { describe, expect, it } from "vitest";
import type { MifAdapter, SessionHistoryEntry } from "./api";
import {
  classifySession,
  formatCount,
  formatRecorded,
  formatSessionLength,
  readSessions,
  readableFormats,
} from "./derive";

/**
 * Every test here is written to fail if the line it covers is removed, and each
 * was checked that way rather than assumed. The alternative — an assertion that
 * holds however the module is mutated — is how this codebase previously carried
 * two whole modules, 489 and 443 lines, that were alive only via their own
 * self-tests.
 *
 * The fixtures are shaped like the live `claude-code` profile because that is
 * where every awkward case comes from: 135 of its 230 recorded sessions carry
 * no duration and no memory count at all, and 7 carry neither writer mark.
 */

function entry(over: Partial<SessionHistoryEntry> = {}): SessionHistoryEntry {
  return {
    session_id: "9d3a44fb-910a-418c-9805-adfaa3a3e922",
    content: "Session in shodh-memory (0min):",
    entities: ["session-summary", "source:hook"],
    started_at: "2026-08-16T07:36:37.491601200Z",
    duration_secs: 26,
    memories_created: 4,
    created_at: "2026-08-16T07:37:28.092967400+00:00",
    ...over,
  };
}

describe("classifySession", () => {
  it("reads the hook's own mark", () => {
    expect(classifySession(entry({ entities: ["session-summary", "source:hook"] }))).toBe("hook");
  });

  it("reads the server's consolidation digest as a different writer", () => {
    expect(classifySession(entry({ entities: ["session-summary", "session-digest"] }))).toBe(
      "consolidation",
    );
  });

  it("leaves a summary with neither mark unattributed", () => {
    // The 7 legacy entries. Folding these into "hook" would be inventing
    // provenance on the one screen whose subject is provenance.
    expect(classifySession(entry({ entities: ["session-summary", "shodh-memory"] }))).toBe(
      "unmarked",
    );
    expect(classifySession(entry({ entities: [] }))).toBe("unmarked");
  });

  it("prefers the hook's mark when a summary carries both", () => {
    // `session_history` dedupes by session_id keeping the newest, and the hook
    // stop fires after compression — so a session that produced both is
    // represented by the hook's row. The order of the two checks in
    // `classifySession` is what encodes that.
    expect(
      classifySession(entry({ entities: ["session-digest", "source:hook"] })),
    ).toBe("hook");
  });
});

describe("readSessions", () => {
  it("reports the server's total, not the size of the page", () => {
    const record = readSessions([entry(), entry()], 230);
    expect(record.total).toBe(230);
    expect(record.page).toBe(2);
  });

  it("withholds the earliest date while the page is a slice", () => {
    // An earliest taken from 50 of 230 sessions is the earliest of a page, and
    // printing it would date the profile's first session months late.
    const record = readSessions([entry({ created_at: "2026-08-01T00:00:00Z" })], 230);
    expect(record.complete).toBe(false);
    expect(record.earliest).toBeNull();
    // The newest is still exact: the handler returns `total` and sorts
    // newest-first regardless of page size.
    expect(record.latest).toBe("2026-08-01T00:00:00Z");
  });

  it("states the earliest date once the page is the whole record", () => {
    const record = readSessions(
      [entry({ created_at: "2026-08-16T07:37:28Z" }), entry({ created_at: "2026-04-01T06:53:19Z" })],
      2,
    );
    expect(record.complete).toBe(true);
    expect(record.earliest).toBe("2026-04-01T06:53:19Z");
    expect(record.latest).toBe("2026-08-16T07:37:28Z");
  });

  it("finds the newest by time rather than by position", () => {
    // Guards against trusting the handler's sort. Position 0 is the OLDEST here.
    const record = readSessions(
      [
        entry({ created_at: "2026-04-01T06:53:19Z" }),
        entry({ created_at: "2026-08-16T07:37:28Z" }),
        entry({ created_at: "2026-06-29T07:56:08Z" }),
      ],
      3,
    );
    expect(record.latest).toBe("2026-08-16T07:37:28Z");
    expect(record.earliest).toBe("2026-04-01T06:53:19Z");
  });

  it("lets an unreadable timestamp contribute to neither bound", () => {
    const record = readSessions(
      [entry({ created_at: "not a date" }), entry({ created_at: "2026-08-16T07:37:28Z" })],
      2,
    );
    expect(record.latest).toBe("2026-08-16T07:37:28Z");
    expect(record.earliest).toBe("2026-08-16T07:37:28Z");
  });

  it("has no dates at all when nothing carries a readable one", () => {
    const record = readSessions([entry({ created_at: "not a date" })], 1);
    expect(record.latest).toBeNull();
    expect(record.earliest).toBeNull();
  });

  it("counts a session against the writer that recorded it", () => {
    const record = readSessions(
      [
        entry({ entities: ["source:hook"] }),
        entry({ entities: ["source:hook"] }),
        entry({ entities: ["session-digest"] }),
        entry({ entities: ["session-summary"] }),
      ],
      4,
    );
    expect(record.byWriter).toEqual({ hook: 2, consolidation: 1, unmarked: 1 });
  });

  it("sums only the sessions that reported a count, and says how many did", () => {
    // The failure this exists to prevent: treating a missing count as zero
    // states a total that silently omits 135 of 230 sessions, and reporting the
    // sum without `reported` presents a sample as a total.
    const record = readSessions(
      [
        entry({ memories_created: 46 }),
        entry({ memories_created: null }),
        entry({ memories_created: 4 }),
        entry({ memories_created: null }),
      ],
      4,
    );
    expect(record.memoriesCreated).toBe(50);
    expect(record.reported).toBe(2);
  });

  it("sums only the durations that were recorded", () => {
    const record = readSessions(
      [entry({ duration_secs: 4404 }), entry({ duration_secs: null }), entry({ duration_secs: 26 })],
      3,
    );
    expect(record.recordedSecs).toBe(4430);
  });

  it("reads an empty record without inventing anything", () => {
    // What `defence-live` and `gdelt-bridge` actually return: 245 memories,
    // zero recorded sessions.
    const record = readSessions([], 0);
    expect(record).toEqual({
      total: 0,
      page: 0,
      complete: true,
      byWriter: { hook: 0, consolidation: 0, unmarked: 0 },
      latest: null,
      earliest: null,
      memoriesCreated: 0,
      reported: 0,
      recordedSecs: 0,
    });
  });
});

describe("formatRecorded", () => {
  it("returns null rather than a zero when nothing reported a duration", () => {
    // "0m" would say every session took no time; null says nothing measured one.
    expect(formatRecorded(0)).toBeNull();
    expect(formatRecorded(-1)).toBeNull();
    expect(formatRecorded(Number.NaN)).toBeNull();
  });

  it("gives whole minutes below an hour", () => {
    expect(formatRecorded(1800)).toBe("30m");
    expect(formatRecorded(3599)).toBe("60m");
  });

  it("never rounds a real total down to zero minutes", () => {
    expect(formatRecorded(20)).toBe("1m");
  });

  it("gives tenths of an hour above one", () => {
    expect(formatRecorded(3600)).toBe("1.0h");
    expect(formatRecorded(19_800)).toBe("5.5h");
  });
});

describe("formatSessionLength", () => {
  it("returns null for a session that recorded no duration", () => {
    expect(formatSessionLength(null)).toBeNull();
    expect(formatSessionLength(0)).toBeNull();
    expect(formatSessionLength(Number.NaN)).toBeNull();
  });

  it("keeps seconds for a short session", () => {
    // The live profile's newest session is 26 seconds. Rounding it to "0m"
    // would make the shortest real sessions look like recording failures.
    expect(formatSessionLength(26)).toBe("26s");
  });

  it("gives minutes up to an hour", () => {
    expect(formatSessionLength(60)).toBe("1m");
    expect(formatSessionLength(4404 % 3600)).toBe("13m");
  });

  it("gives hours and minutes beyond one hour", () => {
    expect(formatSessionLength(4404)).toBe("1h 13m");
  });

  it("drops the minutes when there are none", () => {
    expect(formatSessionLength(7200)).toBe("2h");
  });
});

describe("readableFormats", () => {
  const registry: MifAdapter[] = [
    { name: "Shodh Memory (MIF v2/v1)", format: "shodh" },
    { name: "mem0", format: "mem0" },
    { name: "Generic JSON", format: "generic" },
    { name: "Markdown (YAML frontmatter)", format: "markdown" },
  ];

  it("orders by name so two reads of the same server agree", () => {
    expect(readableFormats(registry).map((a) => a.format)).toEqual([
      "generic",
      "markdown",
      "shodh",
      "mem0",
    ]);
  });

  it("drops an adapter with no name rather than rendering an empty chip", () => {
    const withBlank = [...registry, { name: "   ", format: "future" }];
    expect(readableFormats(withBlank).map((a) => a.format)).not.toContain("future");
    expect(readableFormats(withBlank)).toHaveLength(4);
  });

  it("returns a new array rather than sorting the caller's", () => {
    // The registry is a react-query cache entry shared with anything else that
    // reads it, and `Array.prototype.sort` mutates in place — sorting it
    // directly would reorder that cache as a side effect.
    //
    // ASSERTED ON IDENTITY, NOT ON CONTENT. `expect(registry).toEqual(before)`
    // stood here first and a mutation sweep proved it could not fail: the input
    // is already alphabetical apart from `mem0`, and no rewrite of this function
    // reorders it visibly. Identity fails the moment the copy goes away.
    const unsorted: MifAdapter[] = [
      { name: "mem0", format: "mem0" },
      { name: "Generic JSON", format: "generic" },
    ];
    expect(readableFormats(unsorted)).not.toBe(unsorted);
    expect(unsorted.map((a) => a.format)).toEqual(["mem0", "generic"]);
  });
});

describe("formatCount", () => {
  it("leaves a three-digit count bare", () => {
    expect(formatCount(230)).toBe("230");
  });

  it("groups a large count so it reads as a quantity", () => {
    // Asserted as "contains a non-digit" rather than as "10,758": the separator
    // is the reader's own locale, and pinning it to a comma would make this test
    // a statement about the machine running it. A plain `String(value)` fails
    // this, which is the mutation it is here to catch.
    const grouped = formatCount(10_758);
    expect(grouped).not.toBe("10758");
    expect(grouped).toMatch(/\D/);
    expect(grouped.replace(/\D/g, "")).toBe("10758");
  });
});
