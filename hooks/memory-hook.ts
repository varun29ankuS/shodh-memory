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
  prompt?: string;
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_output?: string;
  stop_reason?: string;
  session_id?: string;
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

  // Also ingest session start
  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content: `Session started in ${projectName}`,
    memory_type: "Context",
    tags: ["session:start", `project:${projectName}`],
  });
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

async function handleSubagentStop(input: HookInput): Promise<void> {
  const subagentType = input.subagent_type;
  const result = input.subagent_result;

  if (!subagentType || !result) return;

  // Store significant subagent results
  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content: `${subagentType} agent completed: ${result.slice(0, 300)}`,
    memory_type: "Task",
    tags: [`subagent:${subagentType}`, "source:hook"],
  });
}

async function handleStop(input: HookInput): Promise<void> {
  await callBrain("/api/remember", {
    user_id: SHODH_USER_ID,
    content: `Session ended: ${input.stop_reason || "user_stop"}`,
    memory_type: "Context",
    tags: ["session:end", `reason:${input.stop_reason || "unknown"}`],
  });
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

main().catch(console.error);
