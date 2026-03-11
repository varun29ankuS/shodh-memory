/**
 * Strip system scaffolding from context before sending to the memory backend.
 * AI clients often pass the full conversation context including XML tags like
 * <task-notification>, <system-reminder>, etc. These overwhelm BM25/embedding
 * and provide zero semantic signal for memory retrieval.
 */
export function stripSystemNoise(text: string): string {
  let result = text;
  const tagPatterns = [
    /<task-notification>[\s\S]*?<\/task-notification>/g,
    /<system-reminder>[\s\S]*?<\/system-reminder>/g,
    /<shodh-context[\s\S]*?<\/shodh-context>/g,
    /<shodh-memory[\s\S]*?<\/shodh-memory>/g,
    /<command-name>[\s\S]*?<\/command-name>/g,
  ];
  for (const pattern of tagPatterns) {
    result = result.replace(pattern, "");
  }
  result = result.replace(/\s{3,}/g, " ").trim();
  return result;
}

/** Helper: Get content from memory (handles nested and flat structure) */
export function getContent(m: { content?: string; experience?: { content?: string } }): string {
  return m.content || m.experience?.content || "";
}

/** Helper: Get memory type from memory (handles both formats) */
export function getType(m: {
  memory_type?: string;
  experience?: { memory_type?: string; experience_type?: string };
}): string {
  return m.memory_type || m.experience?.memory_type || m.experience?.experience_type || "Observation";
}

/** Format surfaced memories for inclusion in tool response */
export function formatSurfacedMemories(
  memories: Array<{ content: string; relevance_score?: number }> | null | undefined
): string {
  if (!memories || memories.length === 0) return "";

  const formatted = memories
    .map(
      (m, i) =>
        `  ${i + 1}. [${((m.relevance_score ?? 0) * 100).toFixed(0)}%] ${m.content.slice(0, 80)}...`
    )
    .join("\n");

  return `\n\n[Relevant memories surfaced]\n${formatted}`;
}

/** Stream tool call content formatter (pure formatting, no I/O) */
export function formatToolCallContent(
  toolName: string,
  args: Record<string, unknown>,
  resultText: string
): string | null {
  // Skip ingesting memory management tools to avoid noise
  if (["remember", "recall", "forget", "list_memories"].includes(toolName)) return null;

  const argsStr = JSON.stringify(args, null, 2);
  return `Tool: ${toolName}\nInput: ${argsStr}\nResult: ${resultText.slice(0, 1000)}${resultText.length > 1000 ? "..." : ""}`;
}
