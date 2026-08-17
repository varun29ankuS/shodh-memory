import type { ProvenanceRecord, RelationshipEdge } from "@/lib/api/graph";

/**
 * Reading the provenance the traverse response already carries.
 *
 * `ProvenanceRecord` arrives on every edge and the Inspector read exactly one
 * field of it — `source_episode_id`, to hydrate an excerpt. The rest was
 * received and dropped. Two of the dropped fields are structurally empty and
 * are refused (the reasoning is on the interface in lib/api/graph.ts); the
 * three that are real are what this module reads.
 *
 * WHY `typed_by` IS THE ONE THAT MATTERS. A graph that draws a line asserts
 * that two things are connected. What makes that assertion checkable is not how
 * heavy the line is — strength is a Hebbian counter, and co-activation makes a
 * wrong edge heavy just as readily as a right one — but HOW THE EDGE WAS
 * DECIDED. `CoOccurrence` means two names landed inside the same 150 characters
 * and nothing read the relation between them. `Semantic`, `Glirel`, `OpenIe`
 * and `Catena` mean an extractor did. Those are different epistemic claims
 * wearing the same visual weight, and until now the difference reached the
 * browser on every traverse and was thrown away.
 */

/** How an entity's edges were typed, commonest first. */
export interface TypingCount {
  method: string;
  count: number;
}

/**
 * `TypingMethod` variants, in the order a reader should meet them: the ones
 * where something read the relation first, co-occurrence last, because that is
 * the ranking by how much the edge actually knows.
 *
 * The full set is src/graph_memory.rs:898-908. It is enumerated here ONLY to
 * order the census — an unknown method is never dropped, it sorts after the
 * known ones, so a variant added server-side degrades to "shown but unranked"
 * rather than to "silently missing".
 */
const METHOD_ORDER = [
  "OpenIe",
  "Glirel",
  "Catena",
  "Semantic",
  "LabelPair",
  "Learned",
  "Cue",
  "CoOccurrence",
];

/** The unranked bucket — records whose `typed_by` the server did not set. Edges
 *  written before the field existed carry `#[serde(default)]` null. */
export const UNTYPED = "untyped";

/**
 * How the edges incident to one entity were typed.
 *
 * Counts PROVENANCE RECORDS, not edges: one edge can be attested by up to eight
 * sources (`PROVENANCE_MAX_SOURCES_DEFAULT`) and each carries its own
 * `typed_by`, so an edge first drawn by co-occurrence and later re-attested by
 * an extractor is two different claims and is counted as two.
 *
 * Only edges TOUCHING the entity are counted. `traverse_from_entity` returns
 * the whole traversed subgraph including edges between the entity's neighbours
 * (GraphTraversal, src/graph_memory.rs:7334-7338), and counting those would
 * describe the neighbourhood while appearing to describe the entity.
 */
export function typingCensus(
  edges: readonly RelationshipEdge[],
  entityId: string,
): TypingCount[] {
  const counts = new Map<string, number>();
  for (const edge of edges) {
    if (edge.from_entity !== entityId && edge.to_entity !== entityId) continue;
    for (const record of edge.provenance ?? []) {
      const method = record.typed_by ?? UNTYPED;
      counts.set(method, (counts.get(method) ?? 0) + 1);
    }
  }

  return [...counts.entries()]
    .map(([method, count]) => ({ method, count }))
    .sort((a, b) => {
      const ai = METHOD_ORDER.indexOf(a.method);
      const bi = METHOD_ORDER.indexOf(b.method);
      // Unknown methods and the untyped bucket sort after every known one,
      // then by count so the larger unknown leads.
      const ar = ai === -1 ? METHOD_ORDER.length : ai;
      const br = bi === -1 ? METHOD_ORDER.length : bi;
      return ar - br || b.count - a.count || a.method.localeCompare(b.method);
    });
}

/**
 * A `TypingMethod` in words a reader who has not read the Rust can use.
 *
 * Each label states WHAT WAS READ, because that is the whole distinction the
 * census exists to draw. An unrecognised value is returned verbatim rather than
 * mapped to "other": a server-side addition should show up as its own name on
 * screen, not disappear into a bucket.
 */
export function typingLabel(method: string): string {
  switch (method) {
    case "CoOccurrence":
      return "co-occurrence";
    case "Semantic":
      return "sentence meaning";
    case "LabelPair":
      return "entity types";
    case "Catena":
      return "temporal/causal";
    case "Cue":
      return "connective cue";
    case "Glirel":
      return "relation model";
    case "OpenIe":
      return "open extraction";
    case "Learned":
      return "learned typer";
    case UNTYPED:
      return "untyped";
    default:
      return method;
  }
}

/**
 * True when every attestation of this entity came from names sharing a window
 * rather than from anything reading the relation.
 *
 * This is the caveat the section states out loud, and it is a real distinction
 * on real data: an entity held together entirely by co-occurrence has a graph
 * neighbourhood built from adjacency alone. Untyped records count as
 * unread — an edge that never recorded HOW it was typed cannot be evidence
 * that something read it.
 */
export function coOccurrenceOnly(census: readonly TypingCount[]): boolean {
  if (census.length === 0) return false;
  return census.every((c) => c.method === "CoOccurrence" || c.method === UNTYPED);
}

/** When a source was observed, and whether that was once or over a span. */
export interface ObservedWindow {
  /** ISO of the first observation. */
  first: string;
  /** ISO of the last, equal to `first` for a single observation. */
  last: string;
  /** True when first and last fall on the same calendar day. */
  sameDay: boolean;
  mentions: number;
}

/**
 * The observation window for one provenance record.
 *
 * `first_observed === last_observed` on every record with `mention_count === 1`
 * — which is 48 of 79 on live data — so a UI that always printed a range would
 * render "16 Aug – 16 Aug" for most sources: a span that is not a span, which
 * reads as corroboration over time where there is none. `sameDay` is what lets
 * the surface say "seen once" for those and reserve a range for records that
 * genuinely have one.
 *
 * Returns `null` for an unparseable timestamp rather than a window built from
 * `NaN`, so a bad date is absent rather than rendered as "Invalid Date".
 */
export function observedWindow(record: ProvenanceRecord): ObservedWindow | null {
  const first = new Date(record.first_observed);
  const last = new Date(record.last_observed);
  if (Number.isNaN(first.getTime()) || Number.isNaN(last.getTime())) return null;

  return {
    first: record.first_observed,
    last: record.last_observed,
    sameDay: first.toDateString() === last.toDateString(),
    mentions: record.mention_count,
  };
}
