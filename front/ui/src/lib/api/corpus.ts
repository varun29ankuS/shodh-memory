import { useQuery } from "@tanstack/react-query";
import { api } from "./client";
import type { Reachability } from "./health";
import type { RecallMemory } from "./types";
import { useSession } from "@/stores/session";

/**
 * The corpus listing — the newest `CORPUS_LIMIT` memories in the active
 * profile, newest first.
 *
 * Two surfaces want a view of the store BEFORE any query exists: Geo plots
 * every located memory as ambient context, and Recall shows the newest
 * memories so the destination opens onto the corpus rather than onto an
 * instruction. Both read this one cache entry.
 *
 * `GET /api/list/{user_id}` returns previews (content capped at 500 chars),
 * which is exactly right here: these rows are orientation, and selecting one
 * routes through the Inspector, which fetches the full record.
 *
 * # Order and coverage
 *
 * The endpoint sorts by `created_at` descending before it paginates
 * (`list_memories_inner`, src/handlers/crud.rs), so this really is the newest
 * page and not merely a page that happens to arrive sorted. That distinction
 * is the whole point: the endpoint used to return tier-then-storage order,
 * which made this an arbitrary sample — on the 18k-memory `claude-code`
 * profile the "newest 500" spanned four and a half months and omitted most of
 * the recent ones. A client-side sort cannot repair that, because the rows it
 * needs were never in the page.
 *
 * This is still a page, not the corpus. When `total > CORPUS_LIMIT` — which is
 * the normal case on a real profile — Geo is plotting the newest
 * `CORPUS_LIMIT` located memories, not every located memory, and any surface
 * that presents these rows as complete is overstating them.
 */

/** `ListMemoryItem` — src/handlers/crud.rs */
export interface CorpusMemory {
  id: string;
  content: string;
  content_truncated: boolean;
  content_length: number;
  memory_type: string;
  importance: number;
  tags: string[];
  created_at: string;
  tier: string;
  /** `[lat, lon, alt]` when the memory carries coordinates. */
  geo_location?: [number, number, number];
}

/** Exported because the Inspector reads this cache entry directly, the same way
 *  it already reads the recall one: a memory selected before any query has run
 *  exists only here, and refetching it to render a detail view would pay again
 *  for data the app is already holding. */
export interface CorpusListResponse {
  memories: CorpusMemory[];
  /** Every memory in the profile, which is NOT `memories.length` — the request
   *  caps at `CORPUS_LIMIT`. */
  total: number;
}

const CORPUS_LIMIT = 500;

export function corpusKey(profile: string | null) {
  return ["corpus", profile] as const;
}

/**
 * Adapt a corpus row to the `RecallMemory` shape the result/geo surfaces
 * render. `score: 0` is a statement, not a placeholder — an unqueried corpus
 * row has no retrieval evidence, and every consumer that sizes or ranks by
 * score treats zero as the quiet baseline.
 */
export function corpusToRecallMemory(m: CorpusMemory): RecallMemory {
  return {
    id: m.id,
    experience: {
      content: m.content,
      memory_type: m.memory_type,
      tags: m.tags,
      geo_location: m.geo_location,
    },
    importance: m.importance,
    created_at: m.created_at,
    score: 0,
    tier: m.tier,
  };
}

export function useCorpus(reach: Reachability) {
  const profile = useSession((s) => s.profile);
  const enabled = reach.state === "online" && profile !== null;

  const { data, error, isFetching } = useQuery({
    queryKey: corpusKey(profile),
    queryFn: ({ signal }) =>
      api.get<CorpusListResponse>(
        `/api/list/${encodeURIComponent(profile!)}?limit=${CORPUS_LIMIT}`,
        signal,
      ),
    enabled,
  });

  return { data, error, isFetching, enabled, profile };
}
