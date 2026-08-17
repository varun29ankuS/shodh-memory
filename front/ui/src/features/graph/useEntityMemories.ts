import { useQuery } from "@tanstack/react-query";
import {
  traverseEntity,
  fetchEpisode,
  type EpisodicNode,
  type ProvenanceRecord,
} from "@/lib/api/graph";
import { typingCensus, type TypingCount } from "./provenance";

/**
 * Entity → the memories it was extracted from.
 *
 * Two chained calls, because no single endpoint does this (see the long note in
 * lib/api/graph.ts):
 *
 *  1. `POST /api/graph/traverse` at depth 1 for the entity's name, then keep
 *     only the edges that actually touch this entity — traverse returns the
 *     whole subgraph, including edges between the entity's neighbours.
 *  2. Collect every `source_episode_id` off those edges and their provenance
 *     records. Each is a memory id, because an episode's uuid IS the memory id
 *     (src/handlers/state.rs:3349-3350).
 *  3. Hydrate a bounded number of them through `POST /api/graph/episode/get`.
 *
 * The hydration is capped. Each id is its own request, and an entity in a dense
 * corpus can attest hundreds of edges; firing all of them to fill a panel that
 * shows a handful would be a burst of traffic nobody asked for. The cap is
 * stated in the UI alongside the true count.
 */

const HYDRATE_LIMIT = 8;

/** One hydrated source, with the attestation that pointed at it. */
export interface SourceEpisode {
  episode: EpisodicNode;
  /**
   * The provenance record naming this episode, when one did.
   *
   * Absent for an id that came from an edge's own `source_episode_id` rather
   * than from its provenance list — the two are collected together because
   * both are memory ids, but only the provenance records carry `typed_by`,
   * `mention_count` and the observed window.
   */
  provenance?: ProvenanceRecord;
}

export interface EntityMemories {
  sources: SourceEpisode[];
  /** Distinct source memory ids found, before the hydration cap. */
  totalSources: number;
  /**
   * How the edges incident to this entity were typed, commonest-first over the
   * WHOLE traversal rather than over the eight hydrated sources — the hydration
   * cap exists to bound requests, and a census that inherited it would report
   * the shape of the cap instead of the shape of the evidence.
   */
  census: TypingCount[];
}

export function entityMemoriesKey(profile: string | null, entityName: string | null) {
  return ["entity-memories", profile, entityName] as const;
}

export function useEntityMemories(profile: string | null, entityName: string | null, entityId: string | null) {
  return useQuery<EntityMemories>({
    queryKey: entityMemoriesKey(profile, entityName),
    enabled: profile !== null && entityName !== null && entityId !== null,
    staleTime: 5 * 60_000,
    queryFn: async ({ signal }) => {
      const traversal = await traverseEntity(
        { user_id: profile!, entity_name: entityName!, max_depth: 1 },
        signal,
      );

      const relationships = traversal.relationships ?? [];

      const ids: string[] = [];
      const seen = new Set<string>();
      // The attestation for each id, when a provenance record named it.
      //
      // DEDUPING THE ID AND CAPTURING THE RECORD ARE SEPARATE STEPS, and
      // collapsing them silently drops most of the metadata. An edge's own
      // `source_episode_id` and its first provenance record's are THE SAME ID —
      // both are `memory_id.0` at src/handlers/state.rs:3893,3901 — so the
      // untagged `add(edge.source_episode_id)` below always runs first and, if
      // it returned early on the second call, the record would never attach.
      // Observed in the browser before this was fixed: the leading sources on
      // `gdelt-bridge` rendered an excerpt with no typing and no mention count
      // while every later one had both.
      //
      // FIRST RECORD WINS thereafter, so a memory attesting several of this
      // entity's edges reports the first one consistently rather than whichever
      // happened to be iterated last.
      const attestation = new Map<string, ProvenanceRecord>();
      const add = (id: string | null | undefined, record?: ProvenanceRecord) => {
        if (!id) return;
        if (!seen.has(id)) {
          seen.add(id);
          ids.push(id);
        }
        if (record && !attestation.has(id)) attestation.set(id, record);
      };

      for (const edge of relationships) {
        // Traverse returns the whole traversed subgraph; only edges incident to
        // THIS entity are evidence about it.
        if (edge.from_entity !== entityId && edge.to_entity !== entityId) continue;
        add(edge.source_episode_id);
        for (const p of edge.provenance ?? []) add(p.source_episode_id, p);
      }

      const hydrated = await Promise.all(
        ids.slice(0, HYDRATE_LIMIT).map((id) =>
          // One unknown episode must not empty the panel.
          fetchEpisode(profile!, id, signal).catch(() => null),
        ),
      );

      return {
        sources: hydrated
          .filter((e): e is EpisodicNode => e !== null && !!e.content)
          .map((episode) => ({ episode, provenance: attestation.get(episode.uuid) })),
        totalSources: ids.length,
        census: typingCensus(relationships, entityId!),
      };
    },
  });
}
