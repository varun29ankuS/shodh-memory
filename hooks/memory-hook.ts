#!/usr/bin/env bun
/**
 * Shodh Memory Hook - Native Claude Code Integration
 *
 * Aggressive proactive context surfacing at every opportunity.
 * Memory should be woven into every interaction - the AI thinks with memory.
 *
 * Events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, SubagentStop, Stop
 */

const SHODH_API_URL = process.env.SHODH_API_URL || "http://127.0.0.1:3030";
const SHODH_API_KEY = process.env.SHODH_API_KEY || "sk-shodh-dev-local-testing-key";
const SHODH_USER_ID = process.env.SHODH_USER_ID || "claude-code";

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

const HOOK_TIMEOUT_MS = 5000;

// ---------------------------------------------------------------------------
// Enrichment: Emotional classification, episode threading, source typing
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

// ---------------------------------------------------------------------------
// Episode threading — chains memories within a session
// ---------------------------------------------------------------------------

/** Current session_id from Claude Code, used as episode_id */
let currentEpisodeId: string | null = null;

/** Monotonically increasing sequence within the session */
let episodeSequenceNumber = 0;

/** ID of the last memory stored in this session, for preceding_memory_id chaining */
let lastStoredMemoryId: string | null = null;

interface RememberResponse {
  id?: string;
  memory_id?: string;
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
  episodeSequenceNumber++;

  const fields: Record<string, unknown> = {
    emotional_valence: emo.valence,
    emotional_arousal: emo.arousal,
    source_type: src.source_type,
    credibility: src.credibility,
    sequence_number: episodeSequenceNumber,
  };

  if (emo.emotion) {
    fields.emotion = emo.emotion;
  }
  if (currentEpisodeId) {
    fields.episode_id = currentEpisodeId;
  }
  if (lastStoredMemoryId) {
    fields.preceding_memory_id = lastStoredMemoryId;
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
    ...enrichment,
    ...(extra || {}),
  })) as RememberResponse | null;

  // Chain: capture returned memory ID for preceding_memory_id
  const memId = resp?.id || resp?.memory_id;
  if (memId) {
    lastStoredMemoryId = memId;
  }
}

// ---------------------------------------------------------------------------
// Structured capture helpers
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
    summary += ` (${oldLines}→${newLines} lines)`;
    // Include a compact diff snippet for semantic embedding
    const oldSnippet = oldStr.slice(0, 80).replace(/\n/g, "↵");
    const newSnippet = newStr.slice(0, 80).replace(/\n/g, "↵");
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

/** Tool actions collected since last proactive_context call for feedback attribution */
const pendingToolActions: { tool_name: string; inputs: Record<string, string>; success: boolean; output_snippet?: string }[] = [];

async function callBrain(endpoint: string, body: Record<string, unknown>): Promise<unknown> {
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

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

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

export function formatMemoriesForContext(memories: SurfacedMemory[]): string {
  if (!memories.length) return "";

  return memories
    .map((m) => {
      const time = formatRelativeTime(m.created_at);
      const score = Math.round(m.score * 100);
      return `• [${score}%] (${time}) ${m.content.slice(0, 120)}${m.content.length > 120 ? "..." : ""}`;
    })
    .join("\n");
}

export function isErrorOutput(toolOutput: string): boolean {
  return (
    toolOutput.includes("error") ||
    toolOutput.includes("Error") ||
    toolOutput.includes("failed") ||
    toolOutput.includes("FAILED")
  );
}

export function buildPreToolContext(toolName: string, toolInput: Record<string, unknown>): string {
  if (toolName === "Edit" || toolName === "Write") {
    const filePath = toolInput.file_path as string;
    if (filePath) {
      return `Editing file: ${filePath}`;
    }
  } else if (toolName === "Bash") {
    const command = toolInput.command as string;
    if (command) {
      return `Running command: ${command.slice(0, 100)}`;
    }
  }

  return `About to use ${toolName}`;
}

async function surfaceProactiveContext(context: string, maxResults = 3, autoIngest = false): Promise<string | null> {
  // Drain pending tool actions for feedback attribution
  const toolActions = pendingToolActions.splice(0, pendingToolActions.length);

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

  const hasMemories = response.memories?.length > 0;
  const hasFacts = response.relevant_facts?.length > 0;
  const hasTodos = response.relevant_todos?.length > 0;
  const hasReminders = (response.due_reminders?.length || 0) + (response.context_reminders?.length || 0) > 0;

  if (!hasMemories && !hasFacts && !hasTodos && !hasReminders) return null;

  const now = new Date();
  const header = `📅 ${now.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })} ${now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;

  let output = header;
  if (hasMemories) {
    output += `\n${formatMemoriesForContext(response.memories)}`;
  }
  if (hasFacts) {
    output += `\n🧠 Facts:`;
    for (const f of response.relevant_facts.slice(0, 3)) {
      output += `\n• (${Math.round(f.confidence * 100)}%) ${f.fact}`;
    }
  }
  if (hasTodos) {
    output += `\n📋 Todos:`;
    for (const t of response.relevant_todos.slice(0, 3)) {
      const icon = t.status === "in_progress" ? "🔄" : "☐";
      output += `\n${icon} ${t.content.slice(0, 80)}`;
    }
  }
  if (hasReminders) {
    const allReminders = [...(response.due_reminders || []), ...(response.context_reminders || [])];
    output += `\n⏰ ${allReminders.length} reminder(s) active`;
  }
  return output;
}

async function handleSessionStart(): Promise<void> {
  const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const projectName = projectDir.split(/[/\\]/).pop() || "unknown";

  const context = `Starting session in project: ${projectName}`;
  const memoryContext = await surfaceProactiveContext(context, 5);

  if (memoryContext) {
    console.error(`[shodh] Session context loaded`);

    // Write to project memory file
    const memoryFile = `${projectDir}/.claude/memory-context.md`;
    try {
      await Bun.write(memoryFile, `# Shodh Memory Context\n\n${memoryContext}\n`);
    } catch {
      // Directory might not exist
    }
  }

  // Session start is tracked implicitly — the proactive_context call above
  // surfaces relevant memories without creating noise in the activity log.
}

async function handleUserPrompt(input: HookInput): Promise<void> {
  const prompt = input.prompt;
  if (!prompt || prompt.length < 10) return;

  // Single call: surface memories AND ingest the prompt in one pipeline pass
  const memoryContext = await surfaceProactiveContext(prompt.slice(0, 1000), 3, true);

  if (memoryContext) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "UserPromptSubmit",
          additionalContext: `\n<shodh-memory>\n${memoryContext}\n</shodh-memory>`,
        },
      })
    );
  }
}

async function handlePreToolUse(input: HookInput): Promise<void> {
  const toolName = input.tool_name;
  const toolInput = input.tool_input;
  if (!toolName || !toolInput) return;

  const context = buildPreToolContext(toolName, toolInput);

  // Surface relevant context BEFORE the tool runs
  const memoryContext = await surfaceProactiveContext(context, 2);

  if (memoryContext) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PreToolUse",
          additionalContext: `\n<shodh-memory context="pre-${toolName.toLowerCase()}">\n${memoryContext}\n</shodh-memory>`,
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
    const actionRecord: (typeof pendingToolActions)[number] = {
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
    pendingToolActions.push(actionRecord);
    if (pendingToolActions.length > 50) {
      pendingToolActions.splice(0, pendingToolActions.length - 50);
    }
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
        `Command failed: ${command?.slice(0, 100)} → ${toolOutput.slice(0, 200)}`,
        "Error",
        ["tool:Bash", "error"],
        "error",
        "error",
      );

      // Surface past errors for this type of command
      const memoryContext = await surfaceProactiveContext(
        `Error with command: ${command?.slice(0, 100)}`,
        2
      );
      if (memoryContext) {
        console.log(
          JSON.stringify({
            hookSpecificOutput: {
              hookEventName: "PostToolUse",
              additionalContext: `\n<shodh-memory context="similar-errors">\n${memoryContext}\n</shodh-memory>`,
            },
          })
        );
      }
    }
  } else if (toolName === "Read") {
    const filePath = toolInput?.file_path as string;
    if (filePath) {
      // Surface what we know about this file
      const memoryContext = await surfaceProactiveContext(
        `Reading file: ${filePath}`,
        2
      );
      if (memoryContext) {
        console.log(
          JSON.stringify({
            hookSpecificOutput: {
              hookEventName: "PostToolUse",
              additionalContext: `\n<shodh-memory context="file-context">\n${memoryContext}\n</shodh-memory>`,
            },
          })
        );
      }
    }
  }
}

// --- Orchestration: PostToolUse(Task) handler ---

const ORCH_TAG_RE = /\[ORCH-TODO:([A-Z]+-\d+)\]/;

async function callBrainGet(endpoint: string): Promise<unknown> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), HOOK_TIMEOUT_MS);
    const response = await fetch(`${SHODH_API_URL}${endpoint}`, {
      headers: { "X-API-Key": SHODH_API_KEY },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

async function unblockDependents(completedShortId: string): Promise<void> {
  const dashIdx = completedShortId.lastIndexOf("-");
  if (dashIdx < 0) return;

  // List all projects to find the one matching this prefix
  // API returns [project, stats] tuples in the projects array
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
    // Construct short_id from project_prefix + seq_num, fall back to UUID
    const todoId = todo.project_prefix && todo.seq_num != null
      ? `${todo.project_prefix}-${todo.seq_num}`
      : todo.id;

    if (remaining.length === 0) {
      // All blockers resolved — unblock
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
      // Some blockers remain — update the list
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

  // 2. Complete the todo (path-based endpoint)
  await callBrain(`/api/todos/${todoShortId}/complete`, {
    user_id: SHODH_USER_ID,
  });

  // 3. Unblock dependents — retry once on failure to prevent orchestration deadlocks
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
  const memoryContext = await surfaceProactiveContext(
    `Orchestration: task ${todoShortId} completed, checking for unblocked work`,
    2
  );
  if (memoryContext) {
    console.log(
      JSON.stringify({
        hookSpecificOutput: {
          hookEventName: "PostToolUse",
          additionalContext: `\n<shodh-memory context="orchestration">\n${memoryContext}\n</shodh-memory>`,
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

async function handleStop(_input: HookInput): Promise<void> {
  // Session end is tracked implicitly by memory timestamps and decay.
  // Storing explicit "Session ended" memories creates noise in the activity log
  // and gets re-ingested by proactive_context auto-ingest, causing duplicate events.
}

async function main(): Promise<void> {
  const inputText = await Bun.stdin.text();

  let input: HookInput;
  try {
    input = JSON.parse(inputText);
  } catch {
    const eventType = process.argv[2];
    input = { hook_event_name: eventType || "SessionStart" };
  }

  // Initialize episode threading from Claude Code's session_id
  if (input.session_id && !currentEpisodeId) {
    currentEpisodeId = input.session_id;
  }

  const eventName = input.hook_event_name;

  switch (eventName) {
    case "SessionStart":
      await handleSessionStart();
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
}

if (import.meta.main) {
  try {
    await main();
  } catch {
    // Silent — hooks must not crash Claude Code
  }
}
