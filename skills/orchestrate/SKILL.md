---
name: orchestrate
description: This skill should be used when the user asks to 'orchestrate a task', 'break down work into parallel agents', 'coordinate subtasks', 'run agents in parallel', or mentions 'multi-agent'. Decomposes complex tasks into tracked subtasks, dispatches parallel subagents, and coordinates until completion.
version: 1.0.0
author: Shodh AI
tags:
  - orchestration
  - multi-agent
  - parallel
  - task-decomposition
  - coordination
---

# Agent Orchestration — Todo-Driven Parallel Execution

You are orchestrating a complex task by decomposing it into tracked subtasks, dispatching parallel agents, and coordinating dependencies until completion. Shodh-memory todos are your task graph. Claude Code's Task tool is your agent spawner. Hooks handle the automation.

## Phase 1: Decompose

Break the user's request into 3-10 concrete, independently executable subtasks.

### Create the project

```
add_project(name="orch-{kebab-case-summary}")
```

The project auto-generates a prefix (e.g., `ORCH`). All todos in this project use that prefix for short IDs like `ORCH-1`, `ORCH-2`.

### Create todos with dependencies

For each subtask, create a todo in the project:

**Independent tasks** (can run immediately):
```
add_todo(
  content="Clear, specific description of what this subtask produces",
  project="orch-{name}",
  priority="high",
  tags=["orchestration", "batch:1"]
)
```

**Dependent tasks** (must wait for others):
```
add_todo(
  content="Description of dependent work",
  project="orch-{name}",
  status="blocked",
  blocked_on="ORCH-1,ORCH-3",
  tags=["orchestration", "batch:2"]
)
```

The `blocked_on` field is comma-separated short IDs. The `batch:N` tag groups tasks by execution wave.

### Dependency rules
- A task blocked on `"ORCH-1,ORCH-3"` cannot start until BOTH are done
- Keep dependency chains shallow (max 3-4 levels deep)
- Maximize parallelism — identify tasks that are truly independent
- Never create circular dependencies

### Present the plan

Show the user the task graph before executing:

```
Project: orch-refactor-auth (ORCH)

Batch 1 (parallel):
  ORCH-1: [todo] Extract JWT utilities into auth/tokens.ts
  ORCH-2: [todo] Create password hashing module

Batch 2 (after batch 1):
  ORCH-3: [blocked on ORCH-1] Update login endpoint
  ORCH-4: [blocked on ORCH-1] Update token refresh endpoint
  ORCH-5: [blocked on ORCH-2] Update registration endpoint

Batch 3 (after batch 2):
  ORCH-6: [blocked on ORCH-3,ORCH-4,ORCH-5] Integration tests
```

Wait for user approval before dispatching.

## Phase 2: Dispatch

### Find unblocked work

```
list_todos(project="orch-{name}", status=["todo"])
```

### For each unblocked todo:

1. Mark it in-progress:
```
update_todo(todo_id="ORCH-N", status="in_progress")
```

2. Spawn a Task agent with the todo tag in the prompt:

**CRITICAL:** Every Task prompt MUST start with `[ORCH-TODO:ORCH-N]` where N is the todo's sequence number. The PostToolUse hook extracts this tag to automatically complete the todo and unblock dependents.

```
Task(
  description="ORCH-N: brief summary",
  prompt="[ORCH-TODO:ORCH-N] Full detailed instructions for the agent...",
  subagent_type="general-purpose"
)
```

3. Spawn independent tasks in parallel — make multiple Task calls in a single response.

### Choose the right agent type

| Agent Type | Best For |
|---|---|
| `Explore` | Research, codebase exploration, finding patterns |
| `Plan` | Architecture design, trade-off analysis |
| `Bash` | Running commands, builds, deployments |
| `general-purpose` | Code changes, implementation, multi-step work |

### Include sufficient context in each prompt

Each agent runs in isolation. Include in every Task prompt:
- What files to look at or modify
- What the expected output/deliverable is
- Any constraints or patterns to follow
- Context from previously completed tasks (copy relevant resolution comments)

## Phase 3: Monitor & Continue

After agents return, the PostToolUse hook automatically:
- Adds the agent's result as a Resolution comment on the matching todo
- Completes the todo
- Unblocks dependent todos (removes from `blocked_on`, changes status to `todo`)

### Check project state

```
list_todos(project="orch-{name}")
```

Review the status:
- `done` — completed by agents
- `todo` — newly unblocked, ready for next batch
- `blocked` — still waiting on dependencies
- `in_progress` — agents still running
- `cancelled` — failed permanently

### Dispatch next batch

If there are `todo` status items, repeat Phase 2 for the next batch. Continue until all todos are `done` or `cancelled`.

### Summarize results

When all todos are complete:
1. List all resolution comments to gather agent outputs
2. Synthesize a summary for the user
3. Note any cancelled tasks and why

## Handling Failures

When a Task agent returns an error or incomplete result:

1. Add a Progress comment documenting the failure:
```
add_todo_comment(
  todo_id="ORCH-N",
  content="Agent failed: {error description}",
  comment_type="progress"
)
```

2. Retry (max 2 attempts) with additional context:
```
Task(
  prompt="[ORCH-TODO:ORCH-N] RETRY: Previous attempt failed because {reason}. {updated instructions}...",
  subagent_type="general-purpose"
)
```

3. If retry fails, cancel the todo:
```
update_todo(todo_id="ORCH-N", status="cancelled", notes="Failed after 2 retries: {reason}")
```

4. Check if cancelled todo blocks other work — inform the user and ask how to proceed.

## Cross-Session Continuity

If a session ends mid-orchestration, the todo state persists. On the next session:

1. Check for in-progress orchestration projects:
```
list_projects()
list_todos(project="orch-{name}")
```

2. Resume from where you left off — dispatch any `todo` status items.

## Example

User: `/orchestrate Add comprehensive error handling to the API layer`

**Planning:**
```
Project: orch-api-error-handling (ORCH)

ORCH-1: [todo] Audit current error handling patterns across all handlers
ORCH-2: [todo] Design error response format and error codes enum
ORCH-3: [blocked on ORCH-1,ORCH-2] Implement centralized error middleware
ORCH-4: [blocked on ORCH-3] Update all handler functions to use new error types
ORCH-5: [blocked on ORCH-4] Add error handling integration tests
```

**Batch 1 dispatch (parallel):**
```
Task("[ORCH-TODO:ORCH-1] Explore the codebase and audit...", subagent_type="Explore")
Task("[ORCH-TODO:ORCH-2] Design an error response format...", subagent_type="Plan")
```

**After batch 1 completes:**
- Hook auto-completes ORCH-1 and ORCH-2
- Hook unblocks ORCH-3 (both blockers resolved)
- Claude dispatches ORCH-3

**Continue until ORCH-5 is done.**
