import { describe, expect, it } from "vitest";
import type { ProvenanceRecord, RelationshipEdge } from "@/lib/api/graph";
import {
  UNTYPED,
  coOccurrenceOnly,
  observedWindow,
  typingCensus,
  typingLabel,
} from "./provenance";

function record(typedBy: string | null, mentions = 1, first = "2026-08-16T13:18:34Z", last = first): ProvenanceRecord {
  return {
    source_episode_id: `ep-${typedBy}-${mentions}-${first}`,
    mention_count: mentions,
    first_observed: first,
    last_observed: last,
    confidence: null,
    evidence_span: [0, 150],
    typed_by: typedBy,
  };
}

function edge(from: string, to: string, provenance: ProvenanceRecord[]): RelationshipEdge {
  return {
    uuid: `${from}-${to}`,
    from_entity: from,
    to_entity: to,
    strength: 1,
    source_episode_id: null,
    context: "ctx",
    provenance,
  };
}

describe("typingCensus", () => {
  it("counts provenance records, not edges", () => {
    // One edge, three attesting sources. An edge-level count would say 1.
    const census = typingCensus(
      [edge("A", "B", [record("CoOccurrence"), record("CoOccurrence", 2), record("Catena")])],
      "A",
    );
    expect(census).toEqual([
      { method: "Catena", count: 1 },
      { method: "CoOccurrence", count: 2 },
    ]);
  });

  it("ignores edges that do not touch the entity", () => {
    // traverse returns the whole subgraph, including neighbour-to-neighbour
    // edges. Counting B–C would describe the neighbourhood, not A.
    const census = typingCensus(
      [edge("A", "B", [record("Semantic")]), edge("B", "C", [record("Catena")])],
      "A",
    );
    expect(census).toEqual([{ method: "Semantic", count: 1 }]);
  });

  it("counts an edge that ends at the entity as well as one that starts there", () => {
    const census = typingCensus([edge("B", "A", [record("Cue")])], "A");
    expect(census).toEqual([{ method: "Cue", count: 1 }]);
  });

  it("buckets a null typed_by as untyped rather than dropping it", () => {
    const census = typingCensus([edge("A", "B", [record(null), record(null)])], "A");
    expect(census).toEqual([{ method: UNTYPED, count: 2 }]);
  });

  it("orders read-the-relation methods before co-occurrence, not by count", () => {
    // CoOccurrence is the largest bucket and must still sort last: the point of
    // the ordering is epistemic rank, not magnitude.
    const census = typingCensus(
      [
        edge("A", "B", [
          record("CoOccurrence"),
          record("CoOccurrence", 2),
          record("CoOccurrence", 3),
          record("Catena"),
          record("OpenIe"),
        ]),
      ],
      "A",
    );
    expect(census.map((c) => c.method)).toEqual(["OpenIe", "Catena", "CoOccurrence"]);
  });

  it("sorts an unknown server-side method after every known one but keeps it", () => {
    const census = typingCensus(
      [edge("A", "B", [record("SomeNewTyper"), record("Semantic")])],
      "A",
    );
    expect(census).toEqual([
      { method: "Semantic", count: 1 },
      { method: "SomeNewTyper", count: 1 },
    ]);
  });

  it("is empty when an edge carries no provenance at all", () => {
    expect(typingCensus([edge("A", "B", [])], "A")).toEqual([]);
  });
});

describe("typingLabel", () => {
  it("names the methods seen live", () => {
    expect(typingLabel("CoOccurrence")).toBe("co-occurrence");
    expect(typingLabel("Catena")).toBe("temporal/causal");
    expect(typingLabel("Semantic")).toBe("sentence meaning");
    expect(typingLabel("LabelPair")).toBe("entity types");
    expect(typingLabel("Cue")).toBe("connective cue");
  });

  it("names the declared variants that live data has not yet produced", () => {
    expect(typingLabel("Glirel")).toBe("relation model");
    expect(typingLabel("OpenIe")).toBe("open extraction");
    expect(typingLabel("Learned")).toBe("learned typer");
  });

  it("returns an unrecognised method verbatim instead of hiding it in 'other'", () => {
    expect(typingLabel("SomeNewTyper")).toBe("SomeNewTyper");
  });
});

describe("coOccurrenceOnly", () => {
  it("is false for an empty census — nothing measured is not a finding", () => {
    expect(coOccurrenceOnly([])).toBe(false);
  });

  it("is true when every record is co-occurrence", () => {
    expect(coOccurrenceOnly([{ method: "CoOccurrence", count: 12 }])).toBe(true);
  });

  it("counts untyped records as unread, so co-occurrence plus untyped is still only", () => {
    expect(
      coOccurrenceOnly([
        { method: "CoOccurrence", count: 12 },
        { method: UNTYPED, count: 3 },
      ]),
    ).toBe(true);
  });

  it("is false as soon as one record read the relation", () => {
    expect(
      coOccurrenceOnly([
        { method: "Catena", count: 1 },
        { method: "CoOccurrence", count: 99 },
      ]),
    ).toBe(false);
  });
});

describe("observedWindow", () => {
  it("marks a single observation as same-day rather than a range", () => {
    // 48 of 79 live records have first === last. A UI that always printed a
    // range would show "16 Aug - 16 Aug" for most sources.
    const w = observedWindow(record("Cue", 1, "2026-08-16T13:18:34Z"));
    expect(w).not.toBeNull();
    expect(w!.sameDay).toBe(true);
    expect(w!.mentions).toBe(1);
  });

  it("reports a genuine multi-day span as not same-day", () => {
    const w = observedWindow(record("Cue", 6, "2026-08-14T13:18:34Z", "2026-08-17T09:00:00Z"));
    expect(w!.sameDay).toBe(false);
    expect(w!.first).toBe("2026-08-14T13:18:34Z");
    expect(w!.last).toBe("2026-08-17T09:00:00Z");
    expect(w!.mentions).toBe(6);
  });

  it("compares calendar days in the LOCAL zone the dates are rendered in", () => {
    // The surface prints these with `toLocaleDateString`, so the same-day test
    // has to agree with it or a record would be labelled "seen once" above two
    // different printed dates. Both instants below are built from a local
    // midnight, which makes this hold in every timezone — an earlier version of
    // this test hard-coded two UTC times on one UTC day and failed at UTC+5:30,
    // where they straddle the local boundary. That failure was the code being
    // right.
    const morning = new Date(2026, 7, 16, 1, 0, 0);
    const evening = new Date(2026, 7, 16, 23, 0, 0);
    const w = observedWindow(record("Cue", 3, morning.toISOString(), evening.toISOString()));
    expect(w!.sameDay).toBe(true);

    // And one second past local midnight is a different day.
    const nextDay = new Date(2026, 7, 17, 0, 0, 1);
    const across = observedWindow(record("Cue", 3, morning.toISOString(), nextDay.toISOString()));
    expect(across!.sameDay).toBe(false);
  });

  it("is null for an unparseable timestamp rather than rendering Invalid Date", () => {
    expect(observedWindow(record("Cue", 1, "not-a-date"))).toBeNull();
    expect(observedWindow(record("Cue", 1, "2026-08-16T13:18:34Z", "nonsense"))).toBeNull();
  });
});
