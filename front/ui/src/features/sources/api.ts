import { api } from "@/lib/api";

/**
 * The three reads behind the Sources screen, and the reason each is the one
 * that can be trusted.
 *
 * THE QUESTION was "I don't see a connectors tab — are we improving the
 * connectors?" There is no connector subsystem: nothing in the Rust tree is
 * named connector, ingest, watcher, scheduler or source registry, and no route
 * in `src/handlers/router.rs` polls anything. Two things nevertheless write
 * into a profile, and neither had ever appeared in the product. This module
 * fetches what can be honestly said about them, and nothing else.
 *
 * WHAT IS DELIBERATELY NOT FETCHED, because each would put a false figure on a
 * screen whose entire purpose is to be checkable:
 *
 *   - `POST /api/sessions` and `GET /api/sessions/stats`. The session store is
 *     `RwLock<HashMap<..>>` held in process memory with a 50-entry ring buffer
 *     per user and a one-hour idle timeout (src/memory/sessions.rs
 *     `SessionStore`). It is never persisted. Measured against a live server
 *     immediately after a restart, `POST /api/sessions` answered
 *     `{"sessions":[],"count":0}` for a profile holding 18,032 memories and 230
 *     recorded sessions. A surface built on it would report "nothing has ever
 *     arrived" every time the server is restarted — the exact false negative
 *     this screen exists to avoid.
 *
 *   - `POST /api/recall/tags` with the hook's own tags. It would give the true
 *     newest hook-written memory, except for two things. Its result set is
 *     truncated on HashSet iteration order rather than by recency — the same
 *     defect `session_history` works around by over-fetching and sorting
 *     itself (src/handlers/sessions.rs) — so a 20-row read of `source:hook`
 *     against the live server returned nothing newer than June for a profile
 *     written to that morning. And it serialises the full internal `Memory`,
 *     embeddings included: those 20 rows were 273KB, which puts an honest
 *     answer at roughly 250MB over the wire.
 *
 *   - `GET /api/list/{user}?query=...`. `query` is a case-insensitive substring
 *     over content AND tags (src/handlers/crud.rs), and a session summary's
 *     content literally contains the string "source:hook", so counting tagged
 *     memories this way over-counts by construction.
 */

/**
 * `SessionHistoryEntry` — src/handlers/sessions.rs.
 *
 * Four of these are nullable in the Rust struct and are nullable in practice,
 * not theoretically: of 230 entries read from the live `claude-code` profile,
 * 95 carried `session_id`, `started_at`, `duration_secs` and `memories_created`
 * and 135 carried none of them. They come from `experience.metadata`, which
 * older writers did not populate. Every consumer here treats absence as
 * absence.
 */
export interface SessionHistoryEntry {
  session_id: string | null;
  content: string;
  /** `experience.entities` — the merged tag list. `source:hook` here is the
   *  hook's own mark; the server's compression digest writes `session-digest`
   *  instead (src/handlers/sessions.rs). */
  entities: string[];
  started_at: string | null;
  duration_secs: number | null;
  memories_created: number | null;
  created_at: string;
}

/**
 * `SessionHistoryResponse` — src/handlers/sessions.rs.
 *
 * `project_threads` is not declared: the handler only computes it when
 * `group_by_project` is set, and this screen never sets it, so the field is
 * always `[]` on our responses. Declaring a shape we never populate would
 * invite a reader to use it.
 */
export interface SessionHistoryResponse {
  success: boolean;
  sessions: SessionHistoryEntry[];
  /** Every recorded session, not just the returned page. The handler dedupes
   *  by `session_id` before counting, so this is sessions and not summaries. */
  total: number;
}

/** `mif::list_adapters` — src/handlers/mif.rs. `format` is the identifier
 *  `POST /api/import/mif` accepts; `name` is the human one. */
export interface MifAdapter {
  name: string;
  format: string;
}

export interface MifAdaptersResponse {
  adapters: MifAdapter[];
  default_export: string;
  default_import: string;
}

/**
 * `MemoryStats` — src/memory/types.rs, served by `GET /api/stats?user_id=`.
 *
 * Only the one field this screen states is declared. The struct carries eleven
 * more (tier counts, retrievals, graph size); every one of them belongs to a
 * question the briefing already answers, and none of them says anything about
 * where a memory came from.
 */
export interface ProfileStats {
  total_memories: number;
}

/**
 * How many recorded sessions to read.
 *
 * The handler returns `total` and sorts newest-first regardless of page size,
 * so the two figures that must be exact — how many sessions exist, and when the
 * newest one was — cost the same at any limit. The page is what backs the
 * "across the last N" figures, which are labelled with N on the surface. Fifty
 * entries measured 36KB against the live server; the whole 230 measured 164KB,
 * and paying that for a figure that would still be a floor is not a trade worth
 * making.
 */
export const SESSION_PAGE = 50;

export function fetchSessionHistory(profile: string, signal?: AbortSignal) {
  return api.post<SessionHistoryResponse>(
    "/api/sessions/history",
    // `group_by_project` is left at its default. The clustering it switches on
    // is a union-find over shared entities (src/handlers/sessions.rs
    // `compute_project_threads`) — an inference, and this screen shows records.
    { user_id: profile, limit: SESSION_PAGE },
    signal,
  );
}

/** No profile: the adapter registry is a property of the build, identical for
 *  every profile on the server. */
export function fetchMifAdapters(signal?: AbortSignal) {
  return api.get<MifAdaptersResponse>("/api/mif/adapters", signal);
}

export function fetchProfileStats(profile: string, signal?: AbortSignal) {
  return api.get<ProfileStats>(`/api/stats?user_id=${encodeURIComponent(profile)}`, signal);
}
