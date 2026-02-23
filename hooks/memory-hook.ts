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

async function callBrain(endpoint: string, body: Record<string, unknown>): Promise<unknown> {
  try {
    const response = await fetch(`${SHODH_API_URL}${endpoint}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": SHODH_API_KEY,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function formatRelativeTime(isoDate: string): string {
  const d = new Date(isoDate);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "today";
  if (diffDays === 1) return "yesterday";
  if (diffDays < 7) return `${diffDays}d ago`;
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

function formatMemoriesForContext(memories: SurfacedMemory[]): string {
  if (!memories.length) return "";

  return memories
    .map((m) => {
      const time = formatRelativeTime(m.created_at);
      const score = Math.round(m.score * 100);
      return `• [${score}%] (${time}) ${m.content.slice(0, 120)}${m.content.length > 120 ? "..." : ""}`;
    })
    .join("\n");
}

async function surfaceProactiveContext(context: string, maxResults = 3, autoIngest = false): Promise<string | null> {
  const response = (await callBrain("/api/proactive_context", {
    user_id: SHODH_USER_ID,
    context,
    max_results: maxResults,
    semantic_threshold: 0.6,
    entity_match_weight: 0.3,
    recency_weight: 0.2,
    auto_ingest: autoIngest,
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

  // Build context from tool input
  let context = `About to use ${toolName}`;

  if (toolName === "Edit" || toolName === "Write") {
    const filePath = toolInput.file_path as string;
    if (filePath) {
      context = `Editing file: ${filePath}`;
    }
  } else if (toolName === "Bash") {
    const command = toolInput.command as string;
    if (command) {
      context = `Running command: ${command.slice(0, 100)}`;
    }
  }

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

  // Orchestration: handle Task tool completions
  if (toolName === "Task") {
    await handlePostToolUseTask(input);
    return;
  }

  // Store significant tool uses
  if (toolName === "Edit" || toolName === "Write") {
    const filePath = toolInput?.file_path as string;
    if (filePath) {
      await callBrain("/api/remember", {
        user_id: SHODH_USER_ID,
        content: `Modified file: ${filePath}`,
        memory_type: "FileAccess",
        tags: [`tool:${toolName}`, `file:${filePath.split(/[/\\]/).pop()}`],
      });
    }
  } else if (toolName === "Bash" && toolOutput) {
    const command = toolInput?.command as string;

    // Store errors/failures for learning
    if (
      toolOutput.includes("error") ||
      toolOutput.includes("Error") ||
      toolOutput.includes("failed") ||
      toolOutput.includes("FAILED")
    ) {
      await callBrain("/api/remember", {
        user_id: SHODH_USER_ID,
        content: `Command failed: ${command?.slice(0, 100)} → ${toolOutput.slice(0, 200)}`,
        memory_type: "Error",
        tags: ["tool:Bash", "error"],
      });

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
    const response = await fetch(`${SHODH_API_URL}${endpoint}`, {
      headers: { "X-API-Key": SHODH_API_KEY },
    });
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
      await callBrain("/api/remember", {
        user_id: SHODH_USER_ID,
        content: `Task agent completed: ${resultText.slice(0, 300)}`,
        memory_type: "Task",
        tags: ["subagent:task", "source:hook"],
      });
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

  // 3. Unblock dependents
  await unblockDependents(todoShortId);

  // 4. Store memory of orchestration completion
  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content: `Orchestration task ${todoShortId} completed: ${resultText.slice(0, 200)}`,
    memory_type: "Task",
    tags: ["orchestration", `todo:${todoShortId}`, "source:hook"],
  });

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

  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content,
    memory_type: "Task",
    tags: [`subagent:${agentType}`, "source:hook"],
  });
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

try {
  await main();
} catch {
  // Silent — hooks must not crash Claude Code
}
