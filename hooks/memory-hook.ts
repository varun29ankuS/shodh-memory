#!/usr/bin/env bun
/**
 * Shodh Memory Hook — Native Claude Code Integration
 *
 * Aggressive proactive context surfacing at every opportunity.
 * Memory is woven into every interaction — the AI thinks with memory.
 *
 * Events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, SubagentStop, Stop
 *
 * Architecture: Each hook event spawns a new Bun process. All state is persisted
 * to a temp file keyed by session_id, loaded at process start, saved on exit.
 */

// ---------------------------------------------------------------------------
// 1. Config Constants
// ---------------------------------------------------------------------------

const SHODH_API_URL = process.env.SHODH_API_URL || "http://127.0.0.1:3030";
const SHODH_API_KEY = process.env.SHODH_API_KEY || "sk-shodh-dev-local-testing-key";
const SHODH_USER_ID = process.env.SHODH_USER_ID || "claude-code";
const HOOK_TIMEOUT_MS = 5000;
const CIRCUIT_BREAKER_THRESHOLD = 3;

// ---------------------------------------------------------------------------
// 2. Type Interfaces
// ---------------------------------------------------------------------------

interface HookInput {
  hook_event_name: string;
  session_id?: string;
  transcript_path?: string;
  cwd?: string;
  // UserPromptSubmit
  prompt?: string;
  // PreToolUse / PostToolUse
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_output?: string;
  tool_response?: unknown;
  // Stop
  stop_reason?: string;
  // SubagentStop
  agent_id?: string;
  agent_type?: string;
  agent_transcript_path?: string;
  // Legacy field names (backward compat)
  subagent_type?: string;
  subagent_result?: string;
}

interface SurfacedMemory {
  id: string;
  content: string;
  memory_type: string;
  score: number;
  importance: number;
  created_at: string;
  tags: string[];
  relevance_reason: string;
  matched_entities: string[];
}

interface ProactiveContextResponse {
  memories: SurfacedMemory[];
  due_reminders: unknown[];
  context_reminders: unknown[];
  memory_count: number;
  reminder_count: number;
  ingested_memory_id: string | null;
  feedback_processed: { memories_evaluated: number; reinforced: string[]; weakened: string[] } | null;
  relevant_todos: { id: string; short_id: string; content: string; status: string; priority: string; project: string | null; due_date: string | null; relevance_reason: string }[];
  todo_count: number;
  relevant_facts: { id: string; fact: string; confidence: number; support_count: number; related_entities: string[] }[];
  latency_ms: number;
  detected_entities: { name: string; entity_type: string }[];
}

interface ToolAction {
  tool_name: string;
  inputs: Record<string, string>;
  success: boolean;
  output_snippet?: string;
}

interface RememberResponse {
  id?: string;
  memory_id?: string;
}

interface SurfaceMetadata {
  count: number;
  bestScore: number;
  bestAge: string;
  factsCount: number;
  todosCount: number;
}

interface SurfaceResult {
  text: string;
  meta: SurfaceMetadata;
}

// ---------------------------------------------------------------------------
// 3. State Persistence — survives across hook invocations
// ---------------------------------------------------------------------------

interface HookState {
  sessionId: string;
  startedAt: string;
  episodeSequenceNumber: number;
  lastStoredMemoryId: string | null;
  pendingToolActions: ToolAction[];
  stats: {
    memoriesStored: number;
    memoriesSurfaced: number;
    errorsTracked: number;
    editsTracked: number;
    commandsTracked: number;
    factsReturned: number;
    todosReturned: number;
    apiFailures: number;
  };
  consecutiveFailures: number;
  surfacedMemoryIds: string[];
  // Conversation extraction
  turnCount: number;
  lastTranscriptOffset: number;
  // TaskCreate/TaskUpdate bridging
  taskIdMap: Record<string, string>;
}

function defaultState(sessionId: string): HookState {
  return {
    sessionId,
    startedAt: new Date().toISOString(),
    episodeSequenceNumber: 0,
    lastStoredMemoryId: null,
    pendingToolActions: [],
    stats: {
      memoriesStored: 0,
      memoriesSurfaced: 0,
      errorsTracked: 0,
      editsTracked: 0,
      commandsTracked: 0,
      factsReturned: 0,
      todosReturned: 0,
      apiFailures: 0,
    },
    consecutiveFailures: 0,
    surfacedMemoryIds: [],
    turnCount: 0,
    lastTranscriptOffset: 0,
    taskIdMap: {},
  };
}

function stateFilePath(sessionId: string): string {
  const tmpDir = process.env.TMPDIR || process.env.TEMP || process.env.TMP || "/tmp";
  return `${tmpDir}/shodh-hook-${sessionId}.json`;
}

function loadState(sessionId: string): HookState {
  try {
    const raw = require("fs").readFileSync(stateFilePath(sessionId), "utf-8");
    const parsed = JSON.parse(raw);
    // Validate structure minimally
    if (parsed && typeof parsed.sessionId === "string" && parsed.stats) {
      return parsed as HookState;
    }
  } catch {
    // File doesn't exist or is corrupted — start fresh
  }
  return defaultState(sessionId);
}

function saveState(state: HookState): void {
  try {
    require("fs").writeFileSync(stateFilePath(state.sessionId), JSON.stringify(state), "utf-8");
  } catch {
    // Best effort — don't crash the hook
  }
}

function deleteState(sessionId: string): void {
  try {
    require("fs").unlinkSync(stateFilePath(sessionId));
  } catch {
    // Already gone or never created
  }
}

/** Mutable hook state — loaded in main(), saved in finally block */
let hookState: HookState;

// ---------------------------------------------------------------------------
// 4. API Helpers with Circuit Breaker
// ---------------------------------------------------------------------------

async function callBrain(endpoint: string, body: Record<string, unknown>): Promise<unknown> {
  // Circuit breaker: skip API calls after consecutive failures
  if (hookState.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
    hookState.stats.apiFailures++;
    return null;
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
    const response = await fetch(`${SHODH_API_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": SHODH_API_KEY,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      hookState.consecutiveFailures++;
      hookState.stats.apiFailures++;
      if (hookState.consecutiveFailures === 1) {
        console.error(`[shodh] API error: ${endpoint} → ${response.status}`);
      }
      if (hookState.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
        console.error(`[shodh] ${CIRCUIT_BREAKER_THRESHOLD} consecutive failures — pausing API calls`);
      }
      return null;
    }
    hookState.consecutiveFailures = 0;
    return await response.json();
  } catch (e) {
    hookState.consecutiveFailures++;
    hookState.stats.apiFailures++;
    if (hookState.consecutiveFailures === 1) {
      console.error(`[shodh] API unreachable: ${endpoint} — ${e instanceof Error ? e.message : "unknown"}`);
    }
    if (hookState.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
      console.error(`[shodh] ${CIRCUIT_BREAKER_THRESHOLD} consecutive failures — pausing API calls`);
    }
    return null;
  }
}

async function callBrainGet(endpoint: string): Promise<unknown> {
  if (hookState.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
    hookState.stats.apiFailures++;
    return null;
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
    const response = await fetch(`${SHODH_API_URL}${endpoint}`, {
      headers: { "X-API-Key": SHODH_API_KEY },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!response.ok) {
      hookState.consecutiveFailures++;
      hookState.stats.apiFailures++;
      if (hookState.consecutiveFailures === 1) {
        console.error(`[shodh] API error: GET ${endpoint} → ${response.status}`);
      }
      if (hookState.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
        console.error(`[shodh] ${CIRCUIT_BREAKER_THRESHOLD} consecutive failures — pausing API calls`);
      }
      return null;
    }
    hookState.consecutiveFailures = 0;
    return await response.json();
  } catch (e) {
    hookState.consecutiveFailures++;
    hookState.stats.apiFailures++;
    if (hookState.consecutiveFailures === 1) {
      console.error(`[shodh] API unreachable: GET ${endpoint} — ${e instanceof Error ? e.message : "unknown"}`);
    }
    if (hookState.consecutiveFailures >= CIRCUIT_BREAKER_THRESHOLD) {
      console.error(`[shodh] ${CIRCUIT_BREAKER_THRESHOLD} consecutive failures — pausing API calls`);
    }
    return null;
  }
}

// ---------------------------------------------------------------------------
// 5. Enrichment: Emotional classification, source typing, episode threading
// ---------------------------------------------------------------------------

interface EmotionalSignature {
  valence: number;   // -1.0 (negative) to 1.0 (positive)
  arousal: number;   // 0.0 (calm) to 1.0 (highly aroused)
  emotion: string | null;
}

interface SourceSignature {
  source_type: string;
  credibility: number; // 0.0 to 1.0
}

/**
 * Classify emotional signature by event type.
 *
 * Errors get high arousal (0.7) to cross PREFETCH_AROUSAL_THRESHOLD (0.6)
 * in the Rust retrieval pipeline, ensuring they surface in future recalls.
 * Based on the emotional salience effect — LaBar & Cabeza (2006).
 */
function classifyEmotion(eventType: "edit" | "write" | "error" | "bash_ok" | "subagent" | "task"): EmotionalSignature {
  switch (eventType) {
    case "edit":
      return { valence: 0.3, arousal: 0.3, emotion: "satisfaction" };
    case "write":
      return { valence: 0.3, arousal: 0.3, emotion: "satisfaction" };
    case "error":
      return { valence: -0.5, arousal: 0.7, emotion: "frustration" };
    case "bash_ok":
      return { valence: 0.1, arousal: 0.2, emotion: null };
    case "subagent":
      return { valence: 0.2, arousal: 0.3, emotion: "satisfaction" };
    case "task":
      return { valence: 0.4, arousal: 0.4, emotion: "satisfaction" };
  }
}

/**
 * Determine source type and credibility per event.
 *
 * Tool outputs are deterministic system events (high credibility).
 * AI-generated summaries from subagents are less reliable.
 */
function classifySource(eventType: "file" | "error" | "command" | "subagent" | "task"): SourceSignature {
  switch (eventType) {
    case "file":
      return { source_type: "system", credibility: 0.9 };
    case "error":
      return { source_type: "system", credibility: 0.95 };
    case "command":
      return { source_type: "system", credibility: 0.8 };
    case "subagent":
      return { source_type: "ai_generated", credibility: 0.6 };
    case "task":
      return { source_type: "ai_generated", credibility: 0.6 };
  }
}

/**
 * Default importance by memory type. Bypasses server-side auto-calculation
 * which lacks hook context (e.g., whether this was a user decision vs routine log).
 * Scale: 0.0 (ephemeral) to 1.0 (critical). Values chosen to match
 * Tulving's (1972) encoding specificity — decisions/errors encode deeper.
 */
function importanceForType(memoryType: string): number {
  switch (memoryType) {
    case "Decision":    return 0.8;
    case "Error":       return 0.75;
    case "Learning":    return 0.7;
    case "Discovery":   return 0.65;
    case "Pattern":     return 0.6;
    case "Task":        return 0.55;
    case "CodeEdit":    return 0.4;
    case "Command":     return 0.35;
    case "FileAccess":  return 0.3;
    case "Search":      return 0.3;
    case "Context":     return 0.3;
    case "Conversation": return 0.25;
    case "Observation": return 0.25;
    default:            return 0.3;
  }
}

/**
 * Build the enrichment fields for a callBrain("/api/remember") payload.
 * Returns a flat object to spread into the request body.
 */
function buildEnrichmentFields(
  emotionType: Parameters<typeof classifyEmotion>[0],
  sourceType: Parameters<typeof classifySource>[0],
): Record<string, unknown> {
  const emo = classifyEmotion(emotionType);
  const src = classifySource(sourceType);
  hookState.episodeSequenceNumber++;

  const fields: Record<string, unknown> = {
    emotional_valence: emo.valence,
    emotional_arousal: emo.arousal,
    source_type: src.source_type,
    credibility: src.credibility,
    sequence_number: hookState.episodeSequenceNumber,
  };

  if (emo.emotion) {
    fields.emotion = emo.emotion;
  }
  if (hookState.sessionId) {
    fields.episode_id = hookState.sessionId;
  }
  if (hookState.lastStoredMemoryId) {
    fields.preceding_memory_id = hookState.lastStoredMemoryId;
  }

  return fields;
}

/**
 * Store a memory with full enrichment. Wraps callBrain("/api/remember") and
 * tracks the returned memory ID for episode chaining.
 */
async function rememberEnriched(
  content: string,
  memoryType: string,
  tags: string[],
  emotionType: Parameters<typeof classifyEmotion>[0],
  sourceType: Parameters<typeof classifySource>[0],
  extra?: Record<string, unknown>,
): Promise<void> {
  const enrichment = buildEnrichmentFields(emotionType, sourceType);
  const resp = (await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content,
    memory_type: memoryType,
    tags,
    importance: importanceForType(memoryType),
    ...enrichment,
    ...(extra || {}),
  })) as RememberResponse | null;

  // Chain: capture returned memory ID for preceding_memory_id
  const memId = resp?.id || resp?.memory_id;
  if (memId) {
    hookState.lastStoredMemoryId = memId;
    hookState.stats.memoriesStored++;
    if (memoryType === "Error") hookState.stats.errorsTracked++;
    if (memoryType === "CodeEdit") hookState.stats.editsTracked++;
  }
}

// ---------------------------------------------------------------------------
// 6. Format Helpers
// ---------------------------------------------------------------------------

export function formatRelativeTime(isoDate: string): string {
  const d = new Date(isoDate);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

/**
 * Format memories into a human-readable context block.
 * Returns null when all scores are noise-level (<0.02).
 * Returns { text, meta } for attribution headers.
 */
export function formatMemoriesForContext(memories: SurfacedMemory[]): SurfaceResult | null {
  if (!memories.length) return null;

  // Filter noise — if best score is below noise floor, return nothing
  const raw = memories.map(m => m.score || 0);
  const maxScore = Math.max(...raw, 0);
  if (maxScore < 0.02) return null;

  const minScore = Math.min(...raw);
  const range = maxScore - minScore;

  const lines: string[] = [];
  let bestAge = "today";

  for (let i = 0; i < memories.length; i++) {
    const m = memories[i];
    const time = formatRelativeTime(m.created_at);
    if (i === 0) bestAge = time;
    const displayScore = range < 0.001
      ? 100
      : Math.round(((raw[i] - minScore) / range) * 100);
    const snippet = m.content.slice(0, 120) + (m.content.length > 120 ? "..." : "");
    lines.push(`\u2022 [${displayScore}% match, ${time}] ${snippet}`);
  }

  return {
    text: lines.join("\n"),
    meta: {
      count: memories.length,
      bestScore: Math.round(maxScore * 100),
      bestAge,
      factsCount: 0,
      todosCount: 0,
    },
  };
}

// ---------------------------------------------------------------------------
// 7. Error Detection (improved — word-boundary regex, false-positive exclusions)
// ---------------------------------------------------------------------------

const ERROR_PATTERNS = [
  /\berror\[E\d+\]/i,         // Rust compiler errors
  /\bError:/,                   // Generic "Error:" at word boundary
  /\bFAILED\b/,                // Test/build failures
  /\bexit code [1-9]\d*/,      // Non-zero exit codes
  /\bpanic(?:ked)?\b/i,        // Rust panics
  /\bcommand not found\b/i,    // Shell errors
  /\bpermission denied\b/i,    // Filesystem errors
  /\bsyntax error\b/i,         // Parse errors
  /\bSegmentation fault\b/,    // Segfault
  /\bfatal:/i,                  // Git fatal, linker fatal
  /\bAborted\b/,               // Process aborted
  /\bERROR\b/,                 // All-caps ERROR
];

const ERROR_FALSE_POSITIVES = [
  /\b0 errors?\b/i,
  /\bno errors?\b/i,
  /\bsucceeded\b/i,
  /\berror-free\b/i,
  /\bwithout errors?\b/i,
  /\berrors?: 0\b/i,
];

export function isErrorOutput(toolOutput: string): boolean {
  // Check false positives first (short-circuit on known-good patterns)
  for (const fp of ERROR_FALSE_POSITIVES) {
    if (fp.test(toolOutput)) return false;
  }
  // Check real error patterns
  for (const pattern of ERROR_PATTERNS) {
    if (pattern.test(toolOutput)) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// 8. Capture Helpers
// ---------------------------------------------------------------------------

/** Summarize an Edit tool's diff from tool_input */
function summarizeEditDiff(toolInput: Record<string, unknown>): string {
  const filePath = toolInput.file_path as string;
  const oldStr = toolInput.old_string as string | undefined;
  const newStr = toolInput.new_string as string | undefined;

  let summary = `Modified file: ${filePath}`;
  if (oldStr && newStr) {
    const oldLines = oldStr.split("\n").length;
    const newLines = newStr.split("\n").length;
    summary += ` (${oldLines}\u2192${newLines} lines)`;
    // Include a compact diff snippet for semantic embedding
    const oldSnippet = oldStr.slice(0, 80).replace(/\n/g, "\u21B5");
    const newSnippet = newStr.slice(0, 80).replace(/\n/g, "\u21B5");
    summary += `\n- ${oldSnippet}\n+ ${newSnippet}`;
  }
  return summary;
}

/** Summarize a Write tool's output from tool_input */
function summarizeWrite(toolInput: Record<string, unknown>): string {
  const filePath = toolInput.file_path as string;
  const content = toolInput.content as string | undefined;
  let summary = `Wrote file: ${filePath}`;
  if (content) {
    const lines = content.split("\n").length;
    const bytes = new TextEncoder().encode(content).length;
    summary += ` (${lines} lines, ${bytes} bytes)`;
  }
  return summary;
}

export function buildPreToolContext(toolName: string, toolInput: Record<string, unknown>): string {
  if (toolName === "Edit" || toolName === "Write") {
    const filePath = toolInput.file_path as string;
    if (filePath) {
      return `Editing file: ${filePath}`;
    }
  }
  return `About to use ${toolName}`;
}

// ---------------------------------------------------------------------------
// 9. Surfacing (with dedup)
// ---------------------------------------------------------------------------

/**
 * Surface proactive context from the memory server.
 * Deduplicates memories already surfaced this turn (cleared on UserPromptSubmit).
 * Returns { text, meta } or null if nothing relevant.
 */
async function surfaceProactiveContext(context: string, maxResults = 3, autoIngest = false): Promise<SurfaceResult | null> {
  // Drain pending tool actions for feedback attribution
  const toolActions = hookState.pendingToolActions.splice(0, hookState.pendingToolActions.length);

  const response = (await callBrain("/api/proactive_context", {
    user_id: SHODH_USER_ID,
    context,
    max_results: maxResults,
    semantic_threshold: 0.6,
    entity_match_weight: 0.3,
    recency_weight: 0.2,
    auto_ingest: autoIngest,
    ...(toolActions.length > 0 ? { tool_actions: toolActions } : {}),
  })) as ProactiveContextResponse | null;

  if (!response) return null;

  // Dedup: filter out memories already surfaced this turn
  const freshMemories = (response.memories || []).filter(
    m => !hookState.surfacedMemoryIds.includes(m.id)
  );

  // Track newly surfaced IDs
  for (const m of freshMemories) {
    hookState.surfacedMemoryIds.push(m.id);
  }

  const hasMemories = freshMemories.length > 0;
  const hasFacts = response.relevant_facts?.length > 0;
  const hasTodos = response.relevant_todos?.length > 0;
  const hasReminders = (response.due_reminders?.length || 0) + (response.context_reminders?.length || 0) > 0;

  // Track activity
  hookState.stats.memoriesSurfaced += freshMemories.length;
  hookState.stats.factsReturned += response.relevant_facts?.length || 0;
  hookState.stats.todosReturned += response.relevant_todos?.length || 0;

  if (!hasMemories && !hasFacts && !hasTodos && !hasReminders) return null;

  // Format memories
  const memResult = hasMemories ? formatMemoriesForContext(freshMemories) : null;

  const meta: SurfaceMetadata = {
    count: freshMemories.length,
    bestScore: memResult?.meta.bestScore || 0,
    bestAge: memResult?.meta.bestAge || "today",
    factsCount: response.relevant_facts?.length || 0,
    todosCount: response.relevant_todos?.length || 0,
  };

  let output = "";

  if (memResult) {
    output += memResult.text;
  }

  if (hasFacts) {
    if (output) output += "\n";
    output += "Facts:";
    for (const f of response.relevant_facts.slice(0, 3)) {
      output += `\n\u2022 (${Math.round(f.confidence * 100)}%) ${f.fact}`;
    }
  }
  if (hasTodos) {
    if (output) output += "\n";
    output += "Todos:";
    for (const t of response.relevant_todos.slice(0, 3)) {
      const icon = t.status === "in_progress" ? "[active]" : "[ ]";
      output += `\n${icon} ${t.content.slice(0, 80)}`;
    }
  }
  if (hasReminders) {
    const allReminders = [...(response.due_reminders || []), ...(response.context_reminders || [])];
    if (output) output += "\n";
    output += `${allReminders.length} reminder(s) active`;
  }

  return { text: output, meta };
}

// ---------------------------------------------------------------------------
// 10. Conversation Extraction — captures assistant responses from transcript
// ---------------------------------------------------------------------------

const DECISION_PATTERNS = [
  /\bshould\b/i, /\bdecided\b/i, /\bthe approach is\b/i,
  /\brecommendation\b/i, /\bwon't work because\b/i,
  /\bthe fix is\b/i, /\broot cause\b/i, /\bchose\b/i,
  /\binstead of\b/i, /\btrade-?off\b/i,
];

const DIRECTIVE_PATTERNS = [
  /\balways\b/i, /\bnever\b/i, /\bmust\b/i, /\bdon'?t\b/i,
  /\bcritical\b/i, /\bimportant\b/i,
];

const INSIGHT_PATTERNS = [
  /\bthe problem was\b/i, /\blearned that\b/i, /\bturns out\b/i,
  /\bkey insight\b/i, /\bpattern\b/i, /\barchitecture\b/i,
];

/**
 * Score an assistant response for decision/insight content.
 * Returns 0 for short acknowledgments, higher for substantive responses.
 */
function scoreResponse(text: string): number {
  if (text.length < 200) return 0;

  let score = 0;

  // Length bonus (logarithmic, caps at 3)
  score += Math.min(Math.log2(text.length / 200), 3);

  for (const p of DECISION_PATTERNS) {
    if (p.test(text)) score += 1;
  }
  for (const p of DIRECTIVE_PATTERNS) {
    if (p.test(text)) score += 1.5;
  }
  for (const p of INSIGHT_PATTERNS) {
    if (p.test(text)) score += 1;
  }

  return score;
}

/**
 * Read transcript tail, extract assistant text responses, score them,
 * and store top 5 as Conversation memories.
 *
 * Uses byte offset tracking for incremental reads — never reads the full file.
 * Capped at 512KB per extraction to stay within 5s hook timeout.
 */
async function extractConversationMemories(transcriptPath: string): Promise<void> {
  const fs = require("fs");
  const MAX_READ_BYTES = 512 * 1024;

  let fileSize: number;
  try {
    const stat = fs.statSync(transcriptPath);
    fileSize = stat.size;
  } catch {
    return;
  }

  // File shrank (truncated/rotated) — reset offset
  if (hookState.lastTranscriptOffset > fileSize) {
    hookState.lastTranscriptOffset = 0;
    return;
  }

  // Nothing new since last extraction
  if (hookState.lastTranscriptOffset >= fileSize) return;

  const bytesToRead = Math.min(MAX_READ_BYTES, fileSize - hookState.lastTranscriptOffset);
  const readStart = Math.max(hookState.lastTranscriptOffset, fileSize - MAX_READ_BYTES);

  let fd: number;
  try {
    fd = fs.openSync(transcriptPath, "r");
  } catch {
    return;
  }

  try {
    const buffer = Buffer.alloc(bytesToRead);
    fs.readSync(fd, buffer, 0, bytesToRead, readStart);

    let text = buffer.toString("utf-8");

    // If we seeked past the stored offset (file too large), skip first partial line
    if (readStart > hookState.lastTranscriptOffset) {
      const firstNewline = text.indexOf("\n");
      if (firstNewline >= 0) {
        text = text.substring(firstNewline + 1);
      }
    }

    // Update offset for next extraction
    hookState.lastTranscriptOffset = fileSize;

    // Parse JSONL, extract assistant text blocks
    const assistantTexts: string[] = [];
    const lines = text.split("\n");

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      let entry: { type?: string; message?: { content?: Array<{ type: string; text?: string }> } };
      try {
        entry = JSON.parse(trimmed);
      } catch {
        continue;
      }

      if (entry.type !== "assistant" || !entry.message?.content) continue;

      // Extract only text blocks, skip tool_use/thinking blocks
      const textBlocks = entry.message.content
        .filter((b: { type: string; text?: string }) => b.type === "text" && b.text)
        .map((b: { type: string; text?: string }) => b.text!)
        .join("\n");

      if (textBlocks.length > 0) {
        assistantTexts.push(textBlocks);
      }
    }

    // Score and pick top 5
    const scored = assistantTexts
      .map((t) => ({ text: t, score: scoreResponse(t) }))
      .filter((t) => t.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);

    if (scored.length === 0) return;

    // Store each as Conversation memory (parallel)
    await Promise.all(
      scored.map((item) =>
        rememberEnriched(
          item.text.slice(0, 2000),
          "Conversation",
          ["source:transcript", "auto-extract"],
          "task",
          "task",
        )
      )
    );
  } finally {
    fs.closeSync(fd);
  }
}

// ---------------------------------------------------------------------------
// 11. TaskCreate/TaskUpdate Bridging — mirrors Claude Code tasks to shodh todos
// ---------------------------------------------------------------------------

async function handleTaskCreate(input: HookInput): Promise<void> {
  const toolInput = input.tool_input;
  const toolOutput = input.tool_output;
  if (!toolInput) return;

  const subject = toolInput.subject as string;
  const description = toolInput.description as string | undefined;

  if (!subject) return;

  // Parse output to get Claude Code's taskId
  let claudeTaskId: string | null = null;
  if (toolOutput) {
    try {
      const parsed = typeof toolOutput === "string" ? JSON.parse(toolOutput) : toolOutput;
      claudeTaskId = parsed.taskId || null;
    } catch {
      // Output wasn't JSON
    }
  }

  // Create shodh todo
  const resp = (await callBrain("/api/todos/add", {
    user_id: SHODH_USER_ID,
    content: subject,
    notes: description || undefined,
    project: "session-tasks",
    priority: "high",
    external_id: claudeTaskId ? `claude-task:${claudeTaskId}` : undefined,
    tags: ["source:hook", "claude-task"],
  })) as { success?: boolean; todo?: { id: string; seq_num?: number; project_prefix?: string } } | null;

  // Map Claude taskId to shodh short_id for future updates
  if (claudeTaskId && resp?.todo) {
    const shortId = resp.todo.project_prefix && resp.todo.seq_num != null
      ? `${resp.todo.project_prefix}-${resp.todo.seq_num}`
      : resp.todo.id;
    hookState.taskIdMap[claudeTaskId] = shortId;
  }
}

async function handleTaskUpdate(input: HookInput): Promise<void> {
  const toolInput = input.tool_input;
  if (!toolInput) return;

  const claudeTaskId = toolInput.taskId as string;
  if (!claudeTaskId) return;

  const shodhShortId = hookState.taskIdMap[claudeTaskId];
  if (!shodhShortId) return;

  const status = toolInput.status as string | undefined;

  if (status === "completed") {
    await callBrain(`/api/todos/${shodhShortId}/complete`, {
      user_id: SHODH_USER_ID,
    });
  } else if (status === "in_progress") {
    await callBrain(`/api/todos/${shodhShortId}/update`, {
      user_id: SHODH_USER_ID,
      status: "in_progress",
    });
  } else if (status === "deleted") {
    await callBrain(`/api/todos/${shodhShortId}/update`, {
      user_id: SHODH_USER_ID,
      status: "cancelled",
    });
  }

  // If subject changed, update content
  const newSubject = toolInput.subject as string | undefined;
  if (newSubject) {
    await callBrain(`/api/todos/${shodhShortId}/update`, {
      user_id: SHODH_USER_ID,
      content: newSubject,
    });
  }
}

// ---------------------------------------------------------------------------
// 12. Event Handlers
// ---------------------------------------------------------------------------

async function handleSessionStart(input: HookInput): Promise<void> {
  const projectDir = process.env.CLAUDE_PROJECT_DIR || input.cwd || process.cwd();
  const projectName = projectDir.split(/[/\\]/).pop() || "unknown";
  const claudeDir = `${projectDir}/.claude`;

  try {
    require("fs").mkdirSync(claudeDir, { recursive: true });
  } catch {
    // Best effort
  }

  // ── Phase 1: Parallel data fetch ──────────────────────────────────────
  // Four independent calls — stats, session history, todos, semantic context

  interface SessionEntry {
    content: string;
    entities: string[];
    session_id?: string;
    started_at?: string;
    duration_secs?: number;
    memories_created?: number;
    created_at: string;
  }
  interface ProjectThread {
    name: string;
    sessions: number[];
    shared_entities: string[];
    session_count: number;
  }
  interface SessionHistoryResponse {
    success: boolean;
    sessions: SessionEntry[];
    project_threads: ProjectThread[];
    total: number;
  }

  interface SummaryItem {
    id: string;
    content: string;
    importance: number;
    created_at: string;
  }
  interface ContextSummaryResponse {
    total_memories: number;
    decisions: SummaryItem[];
    learnings: SummaryItem[];
    context: SummaryItem[];
    patterns: SummaryItem[];
    errors: SummaryItem[];
  }

  const [statsResp, historyResp, activeTodosResult, memoryResult, summaryResp] = await Promise.all([
    callBrainGet(
      `/api/stats?user_id=${encodeURIComponent(SHODH_USER_ID)}`
    ) as Promise<{ total_memories?: number; graph_nodes?: number; graph_edges?: number; total_facts?: number } | null>,

    callBrain("/api/sessions/history", {
      user_id: SHODH_USER_ID,
      limit: 5,
      group_by_project: true,
    }) as Promise<SessionHistoryResponse | null>,

    callBrain("/api/todos/list", {
      user_id: SHODH_USER_ID,
      status: ["in_progress", "todo"],
      limit: 8,
    }) as Promise<{ todos?: Array<{ id: string; seq_num?: number; project_prefix?: string; content: string; status: string; priority: string }> } | null>,

    surfaceProactiveContext(`Starting session in project: ${projectName}`, 5),

    callBrain("/api/context_summary", {
      user_id: SHODH_USER_ID,
      max_items: 5,
      include_decisions: true,
      include_learnings: true,
      include_context: false,
    }) as Promise<ContextSummaryResponse | null>,
  ]);

  // ── Phase 2: Compute state ────────────────────────────────────────────

  const totalMemories = statsResp?.total_memories || 0;
  const graphEdges = statsResp?.graph_edges || 0;
  const totalFacts = statsResp?.total_facts || 0;
  const sessions = historyResp?.success ? historyResp.sessions : [];
  const projectThreads = historyResp?.success ? historyResp.project_threads : [];
  const todos = activeTodosResult?.todos || [];
  const isFirstSession = totalMemories < 5 && sessions.length === 0;

  // ── Phase 3: stderr status line (always visible to user) ──────────────

  if (isFirstSession) {
    console.error(`[shodh] First session — memory capture active`);
  } else {
    const statParts: string[] = [];
    if (totalMemories > 0) statParts.push(`${totalMemories} memories`);
    if (graphEdges > 0) statParts.push(`${graphEdges} edges`);
    if (totalFacts > 0) statParts.push(`${totalFacts} facts`);
    console.error(`[shodh] ${statParts.join(" | ")}`);

    // Show recent sessions in stderr so user sees continuity
    if (sessions.length > 0) {
      const sessionLines = sessions.slice(0, 3).map((s, i) => {
        const age = timeAgo(s.created_at);
        const dur = s.duration_secs ? ` (${formatDuration(s.duration_secs)})` : "";
        const preview = s.content.length > 80 ? s.content.slice(0, 77) + "..." : s.content;
        return `  ${i + 1}. ${age}${dur}: ${preview}`;
      });
      console.error(`[shodh] Recent sessions:\n${sessionLines.join("\n")}`);
    }

    if (todos.length > 0) {
      const inProgress = todos.filter(t => t.status === "in_progress").length;
      const pending = todos.filter(t => t.status === "todo").length;
      const parts: string[] = [];
      if (inProgress > 0) parts.push(`${inProgress} in progress`);
      if (pending > 0) parts.push(`${pending} pending`);
      console.error(`[shodh] Active work: ${parts.join(", ")}`);
    }
  }

  // ── Phase 4: Build additionalContext for Claude ───────────────────────

  if (isFirstSession) {
    console.log(JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: `\n<shodh-memory>\nFirst session detected. Shodh will learn automatically — edits, errors, and commands are captured as memories. A session report will be shown when you end the session.\n</shodh-memory>`,
      },
    }));
    return;
  }

  const contextParts: string[] = [];

  // Session history — structured timeline
  if (sessions.length > 0) {
    const sessionBlock: string[] = ["Recent sessions:"];
    for (const s of sessions) {
      const age = timeAgo(s.created_at);
      const dur = s.duration_secs ? ` (${formatDuration(s.duration_secs)})` : "";
      const memCount = s.memories_created != null ? ` | ${s.memories_created} memories` : "";
      sessionBlock.push(`• ${age}${dur}${memCount}`);
      sessionBlock.push(`  ${s.content.slice(0, 200)}`);
      if (s.entities.length > 0) {
        const filtered = s.entities.filter(e =>
          !["session-summary", "session-digest", "source:hook"].includes(e)
        );
        if (filtered.length > 0) {
          sessionBlock.push(`  Topics: ${filtered.slice(0, 8).join(", ")}`);
        }
      }
    }
    contextParts.push(sessionBlock.join("\n"));
  }

  // Project threads — cross-session continuity
  if (projectThreads.length > 0) {
    const threadBlock: string[] = ["Active projects:"];
    for (const t of projectThreads) {
      threadBlock.push(`• ${t.name} — ${t.session_count} sessions`);
      if (t.shared_entities.length > 0) {
        threadBlock.push(`  Shared context: ${t.shared_entities.slice(0, 6).join(", ")}`);
      }
    }
    contextParts.push(threadBlock.join("\n"));
  }

  // Active todos — what's in flight
  if (todos.length > 0) {
    const todoBlock: string[] = ["Active work:"];
    for (const t of todos) {
      const shortId = t.project_prefix && t.seq_num != null
        ? `${t.project_prefix}-${t.seq_num}`
        : "?";
      const icon = t.status === "in_progress" ? "\u25D0" : "\u25CB";
      const truncContent = t.content.length > 60
        ? t.content.slice(0, 57) + "..."
        : t.content;
      todoBlock.push(`${icon} ${shortId.padEnd(8)} ${truncContent}`);
    }
    contextParts.push(todoBlock.join("\n"));
  }

  // Consolidated knowledge — decisions, learnings, patterns from past sessions
  if (summaryResp) {
    const knowledgeBlock: string[] = [];

    const decisions = summaryResp.decisions || [];
    const learnings = summaryResp.learnings || [];
    const patterns = summaryResp.patterns || [];
    const errors = summaryResp.errors || [];

    if (decisions.length > 0) {
      knowledgeBlock.push("Key decisions:");
      for (const d of decisions.slice(0, 3)) {
        const age = timeAgo(d.created_at);
        const preview = d.content.length > 120 ? d.content.slice(0, 117) + "..." : d.content;
        knowledgeBlock.push(`  • [${age}] ${preview}`);
      }
    }

    if (learnings.length > 0) {
      knowledgeBlock.push("Learnings:");
      for (const l of learnings.slice(0, 3)) {
        const age = timeAgo(l.created_at);
        const preview = l.content.length > 120 ? l.content.slice(0, 117) + "..." : l.content;
        knowledgeBlock.push(`  • [${age}] ${preview}`);
      }
    }

    if (patterns.length > 0) {
      knowledgeBlock.push("Patterns observed:");
      for (const p of patterns.slice(0, 2)) {
        const preview = p.content.length > 120 ? p.content.slice(0, 117) + "..." : p.content;
        knowledgeBlock.push(`  • ${preview}`);
      }
    }

    if (errors.length > 0) {
      knowledgeBlock.push("Recent errors:");
      for (const e of errors.slice(0, 2)) {
        const age = timeAgo(e.created_at);
        const preview = e.content.length > 120 ? e.content.slice(0, 117) + "..." : e.content;
        knowledgeBlock.push(`  • [${age}] ${preview}`);
      }
    }

    if (knowledgeBlock.length > 0) {
      contextParts.push(knowledgeBlock.join("\n"));
    }
  }

  // Semantic memories — relevant to this project
  if (memoryResult) {
    contextParts.push(memoryResult.text);
  }

  if (contextParts.length === 0) {
    return;
  }

  const briefing = contextParts.join("\n\n");
  const surfacedCount = memoryResult?.meta.count || 0;
  console.error(`[shodh] Briefing loaded (${sessions.length} sessions, ${todos.length} todos, ${surfacedCount} memories)`);

  // Inject into Claude's context
  console.log(JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: `\n<shodh-memory>\n${briefing}\n</shodh-memory>`,
    },
  }));

  // Persist to project memory file for reference
  const memoryFile = `${claudeDir}/memory-context.md`;
  try {
    await Bun.write(memoryFile, `# Session Briefing\n\n${briefing}\n`);
  } catch {
    // Best effort
  }
}

/** Format seconds into human-readable duration */
function formatDuration(secs: number): string {
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  const h = Math.floor(secs / 3600);
  const m = Math.round((secs % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

/** Format ISO timestamp into relative age */
function timeAgo(isoDate: string): string {
  const now = Date.now();
  const then = new Date(isoDate).getTime();
  if (isNaN(then)) return "unknown";

  const diffMs = now - then;
  const diffMin = Math.floor(diffMs / 60_000);
  const diffHr = Math.floor(diffMs / 3_600_000);
  const diffDay = Math.floor(diffMs / 86_400_000);

  if (diffMin < 1) return "just now";
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDay === 1) return "yesterday";
  if (diffDay < 7) return `${diffDay} days ago`;
  if (diffDay < 30) return `${Math.floor(diffDay / 7)} weeks ago`;
  return `${Math.floor(diffDay / 30)} months ago`;
}

async function handleUserPrompt(input: HookInput): Promise<void> {
  const prompt = input.prompt;
  if (!prompt || prompt.length < 10) return;

  // Clear surfaced memory IDs — new user turn starts fresh dedup
  hookState.surfacedMemoryIds = [];

  // Single call: surface memories AND ingest the prompt in one pipeline pass
  const result = await surfaceProactiveContext(prompt.slice(0, 1000), 3, true);

  if (result) {
    // Visibility Moment 1: Attribution header
    const header = result.meta.count > 0
      ? `[${result.meta.count} memories surfaced, strongest: ${result.meta.bestScore}% from ${result.meta.bestAge}]`
      : "";

    const attribution = `If you use any of these memories in your response, briefly mention the source (e.g. "from a previous session..." or "based on past experience...").`;

    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "UserPromptSubmit",
          additionalContext: `\n<shodh-memory>\n${header ? header + "\n" : ""}${result.text}\n${attribution}\n</shodh-memory>`,
        },
      })
    );
  }

  // Conversation extraction every 10 turns
  hookState.turnCount++;
  if (hookState.turnCount % 10 === 0 && input.transcript_path) {
    await extractConversationMemories(input.transcript_path);
  }
}

async function handlePreToolUse(input: HookInput): Promise<void> {
  const toolName = input.tool_name;
  const toolInput = input.tool_input;
  if (!toolName || !toolInput) return;

  // Only surface context for Edit/Write — UserPromptSubmit context is sufficient for Bash/Read
  if (toolName !== "Edit" && toolName !== "Write") return;

  const context = buildPreToolContext(toolName, toolInput);

  const result = await surfaceProactiveContext(context, 2);

  if (result) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          additionalContext: `\n<shodh-memory context="pre-${toolName.toLowerCase()}">\n${result.text}\n</shodh-memory>`,
        },
      })
    );
  }
}

async function handlePostToolUse(input: HookInput): Promise<void> {
  const toolName = input.tool_name;
  const toolInput = input.tool_input;
  const toolOutput = input.tool_output;

  if (!toolName) return;

  // Record tool action for feedback attribution (before any early returns)
  if (toolName !== "Task") {
    const actionRecord: ToolAction = {
      tool_name: toolName,
      inputs: {},
      success: true,
    };
    if (toolInput) {
      for (const [k, v] of Object.entries(toolInput)) {
        if (typeof v === "string") {
          actionRecord.inputs[k] = v.slice(0, 500);
        }
      }
    }
    if (toolOutput) {
      actionRecord.success = !isErrorOutput(toolOutput);
      actionRecord.output_snippet = toolOutput.slice(0, 200);
    }
    hookState.pendingToolActions.push(actionRecord);
    if (toolName === "Bash") hookState.stats.commandsTracked++;
    if (hookState.pendingToolActions.length > 50) {
      hookState.pendingToolActions.splice(0, hookState.pendingToolActions.length - 50);
    }
  }

  // TaskCreate/TaskUpdate bridging — mirror to shodh todos
  if (toolName === "TaskCreate") {
    await handleTaskCreate(input);
    return;
  }
  if (toolName === "TaskUpdate") {
    await handleTaskUpdate(input);
    return;
  }

  // Orchestration: handle Task tool completions
  if (toolName === "Task") {
    await handlePostToolUseTask(input);
    return;
  }

  // Store significant tool uses with full enrichment
  if (toolName === "Edit") {
    if (toolInput) {
      const filePath = toolInput.file_path as string;
      if (filePath) {
        const content = summarizeEditDiff(toolInput);
        const fileName = filePath.split(/[/\\]/).pop() || filePath;
        await rememberEnriched(
          content,
          "CodeEdit",
          [`tool:Edit`, `file:${fileName}`],
          "edit",
          "file",
        );
      }
    }
  } else if (toolName === "Write") {
    if (toolInput) {
      const filePath = toolInput.file_path as string;
      if (filePath) {
        const content = summarizeWrite(toolInput);
        const fileName = filePath.split(/[/\\]/).pop() || filePath;
        await rememberEnriched(
          content,
          "FileAccess",
          [`tool:Write`, `file:${fileName}`],
          "write",
          "file",
        );
      }
    }
  } else if (toolName === "Bash" && toolOutput) {
    const command = toolInput?.command as string;

    if (isErrorOutput(toolOutput)) {
      // Store errors with high arousal for future surfacing
      await rememberEnriched(
        `Command failed: ${command?.slice(0, 100)} \u2192 ${toolOutput.slice(0, 200)}`,
        "Error",
        ["tool:Bash", "error"],
        "error",
        "error",
      );

      // Surface past errors for this type of command
      const result = await surfaceProactiveContext(
        `Error with command: ${command?.slice(0, 100)}`,
        2
      );
      if (result) {
        console.log(
          JSON.stringify({
            hookSpecificOutput: {
              hookEventName: "PostToolUse",
              additionalContext: `\n<shodh-memory context="similar-errors">\n${result.text}\n</shodh-memory>`,
            },
          })
        );
      }
    }
  }
}

// --- Orchestration: PostToolUse(Task) handler ---

const ORCH_TAG_RE = /\[ORCH-TODO:([A-Z]+-\d+)\]/;

async function unblockDependents(completedShortId: string): Promise<void> {
  const dashIdx = completedShortId.lastIndexOf("-");
  if (dashIdx < 0) return;

  // List all projects to find the one matching this prefix
  const projectsResp = (await callBrain("/api/projects/list", {
    user_id: SHODH_USER_ID,
  })) as { projects?: Array<[{ id: string; name: string; prefix?: string }, unknown]> } | null;

  if (!projectsResp?.projects?.length) return;

  const prefix = completedShortId.substring(0, dashIdx).toUpperCase();
  const projectEntry = projectsResp.projects.find(
    (entry) => {
      const p = Array.isArray(entry) ? entry[0] : entry;
      return (p.prefix || "").toUpperCase() === prefix;
    }
  );
  if (!projectEntry) return;
  const project = Array.isArray(projectEntry) ? projectEntry[0] : projectEntry;

  // List blocked todos in this project
  const todosResp = (await callBrain("/api/todos/list", {
    user_id: SHODH_USER_ID,
    project: project.name,
    status: ["blocked"],
  })) as { todos?: Array<{ id: string; seq_num?: number; project_prefix?: string; blocked_on?: string }> } | null;

  if (!todosResp?.todos?.length) return;

  for (const todo of todosResp.todos) {
    if (!todo.blocked_on) continue;

    const blockers = todo.blocked_on.split(",").map((s) => s.trim());
    const remaining = blockers.filter((b) => b !== completedShortId);
    const todoId = todo.project_prefix && todo.seq_num != null
      ? `${todo.project_prefix}-${todo.seq_num}`
      : todo.id;

    if (remaining.length === 0) {
      await callBrain(`/api/todos/${todoId}/update`, {
        user_id: SHODH_USER_ID,
        status: "todo",
        blocked_on: "",
      });
      await callBrain(`/api/todos/${todoId}/comments`, {
        user_id: SHODH_USER_ID,
        content: `Unblocked: dependency ${completedShortId} completed`,
        comment_type: "activity",
      });
    } else {
      await callBrain(`/api/todos/${todoId}/update`, {
        user_id: SHODH_USER_ID,
        blocked_on: remaining.join(","),
      });
    }
  }
}

async function handlePostToolUseTask(input: HookInput): Promise<void> {
  const toolInput = input.tool_input;
  const toolResult = input.tool_output ?? input.tool_response;

  if (!toolInput) return;

  const prompt = (toolInput.prompt as string) || "";
  const resultText = typeof toolResult === "string"
    ? toolResult
    : toolResult != null
      ? JSON.stringify(toolResult)
      : "";

  // Check for orchestration tag
  const tagMatch = prompt.match(ORCH_TAG_RE);

  if (!tagMatch) {
    // Not an orchestration task — store as generic memory
    if (resultText) {
      await rememberEnriched(
        `Task agent completed: ${resultText.slice(0, 300)}`,
        "Task",
        ["subagent:task", "source:hook"],
        "task",
        "task",
      );
    }
    return;
  }

  const todoShortId = tagMatch[1];

  // 1. Add Resolution comment with agent result
  if (resultText) {
    await callBrain(`/api/todos/${todoShortId}/comments`, {
      user_id: SHODH_USER_ID,
      content: resultText.slice(0, 4000),
      comment_type: "resolution",
    });
  }

  // 2. Complete the todo
  await callBrain(`/api/todos/${todoShortId}/complete`, {
    user_id: SHODH_USER_ID,
  });

  // 3. Unblock dependents — retry once on failure
  try {
    await unblockDependents(todoShortId);
  } catch (e) {
    console.error(`[shodh] unblockDependents failed for ${todoShortId}, retrying: ${e}`);
    try {
      await unblockDependents(todoShortId);
    } catch (e2) {
      console.error(`[shodh] unblockDependents retry failed for ${todoShortId}: ${e2}`);
    }
  }

  // 4. Store memory of orchestration completion
  await rememberEnriched(
    `Orchestration task ${todoShortId} completed: ${resultText.slice(0, 200)}`,
    "Task",
    ["orchestration", `todo:${todoShortId}`, "source:hook"],
    "task",
    "task",
  );

  // 5. Surface orchestration status
  const result = await surfaceProactiveContext(
    `Orchestration: task ${todoShortId} completed, checking for unblocked work`,
    2
  );
  if (result) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          additionalContext: `\n<shodh-memory context="orchestration">\n${result.text}\n</shodh-memory>`,
        },
      })
    );
  }
}

async function handleSubagentStop(input: HookInput): Promise<void> {
  const agentType = input.agent_type || input.subagent_type;
  const agentId = input.agent_id;
  const result = input.subagent_result;

  if (!agentType) return;

  const content = result
    ? `${agentType} agent completed: ${result.slice(0, 300)}`
    : `${agentType} agent (${agentId || "unknown"}) completed`;

  await rememberEnriched(
    content,
    "Task",
    [`subagent:${agentType}`, "source:hook"],
    "subagent",
    "subagent",
  );
}

async function handleStop(input: HookInput): Promise<void> {
  const sessionId = hookState.sessionId;
  const stopReason = input.stop_reason || "unknown";
  const cwd = input.cwd || process.cwd();
  const projectDir = cwd.split(/[/\\]/).pop() || cwd;

  // Calculate session duration
  const startedAt = new Date(hookState.startedAt);
  const now = new Date();
  const durationMs = now.getTime() - startedAt.getTime();
  const durationMin = Math.round(durationMs / 60000);

  // Build human-readable session report
  const lines: string[] = [];
  lines.push(`[shodh] Session report (${durationMin}min):`);

  const s = hookState.stats;

  if (s.memoriesStored > 0) {
    const parts: string[] = [];
    if (s.editsTracked > 0) parts.push(`${s.editsTracked} edits`);
    if (s.errorsTracked > 0) parts.push(`${s.errorsTracked} errors`);
    if (s.commandsTracked > 0) parts.push(`${s.commandsTracked} commands`);
    const breakdown = parts.length > 0 ? ` (${parts.join(", ")})` : "";
    lines.push(`  Captured: ${s.memoriesStored} memories${breakdown}`);
  }

  if (s.memoriesSurfaced > 0) {
    lines.push(`  Surfaced: ${s.memoriesSurfaced} relevant memories`);
  }

  if (s.factsReturned > 0) {
    lines.push(`  Facts used: ${s.factsReturned}`);
  }

  if (s.todosReturned > 0) {
    lines.push(`  Todos matched: ${s.todosReturned}`);
  }

  if (s.apiFailures > 0) {
    lines.push(`  API failures: ${s.apiFailures}`);
  }

  if (s.memoriesStored === 0 && s.memoriesSurfaced === 0) {
    lines.push(`  No memory activity this session.`);
  }

  // Print to stderr (visible in Claude Code terminal)
  console.error(lines.join("\n"));

  // Visibility Moment 2: Persistent session log
  const claudeDir = `${cwd}/.claude`;
  try {
    require("fs").mkdirSync(claudeDir, { recursive: true });
    const logFile = `${claudeDir}/shodh-sessions.md`;
    const startTime = startedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const endTime = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const dateStr = now.toLocaleDateString([], { month: "short", day: "numeric" });

    let entry = `\n## ${dateStr}, ${startTime}\u2013${endTime} (${durationMin}min)\n`;
    if (s.memoriesStored > 0) {
      const parts: string[] = [];
      if (s.editsTracked > 0) parts.push(`${s.editsTracked} edits`);
      if (s.errorsTracked > 0) parts.push(`${s.errorsTracked} errors`);
      if (s.commandsTracked > 0) parts.push(`${s.commandsTracked} commands`);
      const breakdown = parts.length > 0 ? ` (${parts.join(", ")})` : "";
      entry += `Captured: ${s.memoriesStored} memories${breakdown}\n`;
    }
    if (s.memoriesSurfaced > 0 || s.factsReturned > 0 || s.todosReturned > 0) {
      const surfParts: string[] = [];
      if (s.memoriesSurfaced > 0) surfParts.push(`${s.memoriesSurfaced} relevant memories`);
      if (s.factsReturned > 0) surfParts.push(`Facts: ${s.factsReturned}`);
      if (s.todosReturned > 0) surfParts.push(`Todos: ${s.todosReturned}`);
      entry += `Surfaced: ${surfParts.join(" | ")}\n`;
    }

    // Append to session log
    require("fs").appendFileSync(logFile, entry, "utf-8");
  } catch {
    // Best effort — don't crash the hook
  }

  // Final conversation extraction — capture any remaining assistant responses
  if (input.transcript_path) {
    await extractConversationMemories(input.transcript_path);
  }

  // Fetch structured digest for rich session summary
  const digestResp = (await callBrain("/api/sessions/digest", {
    user_id: SHODH_USER_ID,
  })) as {
    digest?: {
      session_id: string;
      started_at: string;
      duration_secs: number;
      memories_created: number;
      memories_surfaced: number;
      memories_used: number;
      memory_hit_rate: number;
      todos_created: number;
      todos_completed: number;
      entities_extracted: string[];
      entity_count: number;
      tools_used: Record<string, number>;
      topic_changes: number;
      compressions: number;
    };
  } | null;

  let summaryContent: string;
  let summaryEntities: string[] = [];
  let summaryMetadata: Record<string, string> = {};

  if (digestResp?.digest) {
    const d = digestResp.digest;
    const dm = Math.round(d.duration_secs / 60);
    const topTools = Object.entries(d.tools_used || {})
      .sort(([, a], [, b]) => (b as number) - (a as number))
      .slice(0, 5)
      .map(([t, c]) => `${t}(${c})`)
      .join(", ");

    summaryContent = [
      `Session in ${projectDir} (${dm}min):`,
      `  Memories: ${d.memories_created} created, ${d.memories_surfaced} surfaced (${Math.round(d.memory_hit_rate * 100)}% hit rate)`,
      `  Todos: ${d.todos_created} created, ${d.todos_completed} completed`,
      d.entities_extracted.length > 0
        ? `  Entities: ${d.entities_extracted.slice(0, 15).join(", ")}`
        : null,
      topTools ? `  Tools: ${topTools}` : null,
      `  Topics changed: ${d.topic_changes} | Compressions: ${d.compressions}`,
      `  Ended: ${stopReason}`,
    ]
      .filter(Boolean)
      .join("\n");

    summaryEntities = d.entities_extracted.slice(0, 20);
    summaryMetadata = {
      session_digest: "true",
      session_id: String(d.session_id),
      started_at: d.started_at,
      duration_secs: String(d.duration_secs),
      memories_created: String(d.memories_created),
      entity_count: String(d.entity_count),
    };
  } else {
    // Fallback: digest unavailable (server down, no active session)
    summaryContent = `Session in ${projectDir} (${durationMin}min): ${s.memoriesStored} memories captured (${s.editsTracked} edits, ${s.errorsTracked} errors), ${s.memoriesSurfaced} memories surfaced, ${s.factsReturned} facts used. Ended: ${stopReason}.`;
  }

  // Merge entities into tags — RememberRequest only has `tags`, not `entities`.
  // Deduplicate to avoid indexing the same entity twice.
  const allTags = ["session-summary", "source:hook", ...summaryEntities];
  const uniqueTags = [...new Set(allTags)];

  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content: summaryContent,
    memory_type: "Context",
    tags: uniqueTags,
    source_type: "system",
    importance: 0.4,
    episode_id: sessionId,
    preceding_memory_id: hookState.lastStoredMemoryId || undefined,
    credibility: 1.0,
    metadata: summaryMetadata,
  });
}

// ---------------------------------------------------------------------------
// 12. Main — load state, dispatch, save/delete
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const inputText = await Bun.stdin.text();

  let input: HookInput;
  try {
    input = JSON.parse(inputText);
  } catch {
    const eventType = process.argv[2];
    input = { hook_event_name: eventType || "SessionStart" };
  }

  const sessionId = input.session_id || "unknown";
  const eventName = input.hook_event_name;

  // Load persisted state (or create fresh for new session)
  hookState = loadState(sessionId);

  // Update session ID if this is the first event
  if (hookState.sessionId !== sessionId && sessionId !== "unknown") {
    hookState = defaultState(sessionId);
  }

  try {
    switch (eventName) {
      case "SessionStart":
        await handleSessionStart(input);
        break;
      case "UserPromptSubmit":
        await handleUserPrompt(input);
        break;
      case "PreToolUse":
        await handlePreToolUse(input);
        break;
      case "PostToolUse":
        await handlePostToolUse(input);
        break;
      case "SubagentStop":
        await handleSubagentStop(input);
        break;
      case "Stop":
        await handleStop(input);
        break;
    }
  } finally {
    if (eventName === "Stop") {
      // Session ended — clean up temp file
      deleteState(sessionId);
    } else {
      // Persist state for next invocation
      saveState(hookState);
    }
  }
}

if (import.meta.main) {
  try {
    await main();
  } catch {
    // Silent — hooks must not crash Claude Code
  }
}
