// =============================================================================
// Memory content rendering — honest truncation (issue #396)
//
// Every place a MEMORY BODY is sliced for tool output MUST route through
// `renderContent` so truncation is ALWAYS explicitly marked with real lengths
// and (when an id is available) the read_memory follow-up affordance. A bare
// "..." that hides incompleteness is the failure class this module removes:
// agents were treating short previews as complete memories.
// =============================================================================

/**
 * Default preview cap (in characters) for memory bodies rendered by
 * recall / recall_by_tags / proactive_context. Raised from the historical
 * 150 to 500 per issue #396 so previews carry meaningfully more content
 * before truncation kicks in.
 */
export const MEMORY_PREVIEW_MAX = 500;

/**
 * Render a memory body for tool output with honest truncation.
 *
 * - `full === true`  → return the complete body verbatim, no marker.
 * - body fits in `max` → return it verbatim, no marker (never a bare "...").
 * - body exceeds `max` → return the first `max` chars followed by an explicit
 *   marker showing shown/total lengths. When `id` is present the marker also
 *   carries the `read_memory("<id>")` follow-up affordance; when it is absent
 *   the marker omits the hint but still reports the real lengths.
 *
 * @param content Full memory body (the engine always returns the whole thing).
 * @param id      Memory id for the read_memory follow-up, or undefined when the
 *                render site has no addressable id (e.g. todos, surfaced memories).
 * @param max     Preview cap in characters.
 * @param full    When true, bypass truncation entirely and return the full body.
 */
export function renderContent(
  content: string,
  id: string | undefined,
  max: number,
  full: boolean,
): string {
  if (full || content.length <= max) {
    return content;
  }

  const shown = content.slice(0, max);
  const hint = id ? ` — read_memory("${id}") for full` : "";
  return `${shown}…[truncated ${shown.length}/${content.length} chars${hint}]`;
}
