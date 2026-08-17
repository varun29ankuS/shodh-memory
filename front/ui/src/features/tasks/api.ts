import { api, type Todo } from "@/lib/api";

/**
 * The endpoints this screen needs beyond `listTodos`, kept in-fence.
 *
 * `lib/api/todos.ts` exposes only the list call, and `lib/api/types.ts` carries
 * a deliberately reduced `Todo` — "only the fields this UI renders". Triage
 * needs three fields that reduction dropped and four routes it never had, so
 * they are declared here rather than by widening the shared module.
 *
 * EVERY SHAPE BELOW WAS READ OFF THE HANDLER, NOT GUESSED. Route table:
 * src/handlers/router.rs:298-332. The flat aliases registered there
 * (`/api/todos/update`, `/complete`, `/delete`, `/reorder`, router.rs:301-304)
 * are NOT used: each points at a handler whose first extractor is
 * `Path<String>` on a route with no `{todo_id}` capture, which under axum 0.8
 * fails extraction rather than matching. The `{todo_id}` path forms are what
 * the MCP server calls (mcp-server/index.ts:4533-4597) and are what is used
 * here.
 */

/** `TodoCommentType` — src/memory/types.rs:4050-4063, `rename_all = "snake_case"`. */
export type TodoCommentType = "comment" | "activity" | "progress" | "resolution";

/** `TodoComment` — src/memory/types.rs:3997-4020. Ships inline on every listed
 *  todo, so a dismissal reason is readable without a second request. */
export interface TodoComment {
  id: string;
  todo_id: string;
  author: string;
  content: string;
  comment_type: TodoCommentType;
  created_at: string;
  /** Null until the comment is edited — verified against a live response, not
   *  assumed from the Rust `Option<DateTime>`. */
  updated_at: string | null;
}

/**
 * The listed todo, with the three fields `lib/api`'s `Todo` leaves out.
 *
 * `list_todos` serialises `Vec<Todo>` with no reducing DTO
 * (src/handlers/todos.rs:1442-1448), so all three are already on the wire —
 * this widens the type to match what the server actually sends, it does not
 * ask for anything new.
 *
 * `external_id` carries `skip_serializing_if = "Option::is_none"`
 * (src/memory/types.rs:4114) and so is genuinely absent, not null, on todos
 * that have none.
 */
export interface TriageTodo extends Todo {
  /** src/memory/types.rs:4154. Ids only — resolving one costs a request. */
  related_memory_ids?: string[];
  /** src/memory/types.rs:4143. */
  comments?: TodoComment[];
  /** src/memory/types.rs:4115. Set by the session hook as `claude-task:{id}`
   *  (hooks/memory-hook.ts:1004); otherwise an external-sync key. */
  external_id?: string;
  /**
   * src/memory/types.rs:4156-4161 — todos that must complete before this one.
   *
   * Bare UUID strings: `TodoId` is `#[serde(transparent)]` over a `Uuid`
   * (types.rs:3781), so these are NOT the short "SHOD-3" form the create and
   * update requests accept as input (todos.rs:191-194). Resolving one to
   * something a person can read means matching it against the todos in hand.
   *
   * Distinct from the free-text `blocked_on`, and the struct's own comment says
   * why: these "reference real todos so blocked chains can be walked and cycles
   * rejected" (types.rs:4157-4158), which `blocked_on` cannot be.
   */
  blocked_by?: string[];
}

/**
 * `Project` — src/memory/types.rs:4340-4406, with the one field `lib/api`'s
 * reduced `Project` leaves out.
 *
 * `status` is `ProjectStatus` (types.rs:4329-4336) under `rename_all =
 * "snake_case"`, so the wire values are these four and not the PascalCase
 * variant names. It is needed because an archived project and an active one
 * must not read the same: on the live `claude-code` profile six of nine
 * projects are archived and hold 72 of 93 tasks, so ranking them equally would
 * bury the three that are still running.
 *
 * `todo_counts` IS ON THE WIRE AND IS DELIBERATELY NOT DECLARED HERE. Nothing
 * populates it: `ProjectTodoCounts` has exactly three construction sites in the
 * whole Rust tree (types.rs:4440, types.rs:4464, mif/import.rs:194) and every
 * one is `::default()`, so the server sends `{total: 0, backlog: 0, todo: 0,
 * in_progress: 0, blocked: 0, done: 0}` for every project regardless of how
 * many todos it holds — verified live against all nine. A meter sourced from it
 * would read "0 of 0" beside thirteen finished tasks. Every count on this
 * screen is therefore derived from the todos actually in hand.
 */
export interface TriageProject {
  id: string;
  user_id: string;
  name: string;
  prefix: string | null;
  status: "active" | "on_hold" | "completed" | "archived";
}

/** `TodoListResponse` — src/handlers/todos.rs:211-218. `count` is the total
 *  BEFORE truncation (todos.rs:1421), which is the only way this screen learns
 *  it is showing a slice. `projects` is the profile's whole project list, not
 *  only those represented in `todos` (todos.rs:1435-1438). */
export interface TriageListResponse {
  success: boolean;
  count: number;
  todos: TriageTodo[];
  projects: TriageProject[];
}

/**
 * `POST /api/todos` — the same route `lib/api/todos.ts` calls, typed to the
 * fields triage needs.
 *
 * `status` and `include_completed` are both real request fields
 * (`ListTodosRequest`, src/handlers/todos.rs:233-255). Asking for
 * `["done", "cancelled"]` with `include_completed: true` returns settled work
 * only — verified live against the running backend, because
 * `include_completed` alone returns open work too and a client-side filter
 * would have made `count` meaningless for the truncation notice.
 */
export function listTriageTodos(
  req: {
    user_id: string;
    limit: number;
    status?: string[];
    include_completed?: boolean;
  },
  signal?: AbortSignal,
): Promise<TriageListResponse> {
  return api.post<TriageListResponse>("/api/todos", req, signal);
}

/**
 * `Memory` — the source a `related_memory_ids` entry resolves to.
 *
 * `GET /api/memory/{id}?user_id=` → `MemoryWithHierarchy`
 * (src/handlers/crud.rs:29-37), which is `#[serde(flatten)]` over `Memory`, so
 * these sit at the top level. Only the fields this screen reads are declared:
 * the real response also carries a 384-float embedding and the full robotics
 * block, which is exactly why it is fetched one row at a time.
 */
export interface LinkedMemory {
  id: string;
  experience: {
    content: string;
    experience_type: string;
    tags: string[];
  };
  created_at: string;
}

/** `GET /api/memory/{memory_id}?user_id=` — src/handlers/router.rs:129,
 *  src/handlers/crud.rs:242-276. `user_id` is a required query param. */
export function getMemory(
  memoryId: string,
  userId: string,
  signal?: AbortSignal,
): Promise<LinkedMemory> {
  return api.get<LinkedMemory>(
    `/api/memory/${encodeURIComponent(memoryId)}?user_id=${encodeURIComponent(userId)}`,
    signal,
  );
}

/** `TodoResponse` — src/handlers/todos.rs:202-208. */
interface TodoResponse {
  success: boolean;
  todo: TriageTodo | null;
  formatted: string;
}

/**
 * `POST /api/todos/{todo_id}/update` — src/handlers/router.rs:308,
 * `UpdateTodoRequest` src/handlers/todos.rs:259-292.
 *
 * Status strings are the six the handler accepts (todos.rs:1259-1265). Note
 * that this request REPLACES `related_memory_ids` when the field is sent, so
 * it is never sent from here — a status change must not silently drop a
 * task's memory links.
 */
export function updateTodoStatus(
  todoId: string,
  userId: string,
  status: "backlog" | "todo" | "in_progress" | "blocked" | "done" | "cancelled",
): Promise<TodoResponse> {
  return api.post<TodoResponse>(`/api/todos/${encodeURIComponent(todoId)}/update`, {
    user_id: userId,
    status,
  });
}

/** `CommentResponse` — src/handlers/todos.rs:355-361. */
interface CommentResponse {
  success: boolean;
  comment: TodoComment | null;
  formatted: string;
}

/**
 * The name this surface signs its own writes with.
 *
 * `AddCommentRequest.author` is optional and the handler defaults it to
 * `user_id` (src/handlers/todos.rs:2233), so before this every comment written
 * from this dashboard was indistinguishable from one written by the MCP tools,
 * a hook, or a curl command — they all read as the profile name. Microsoft's
 * HAX guideline G9-C states attribution and reversal as two halves of ONE
 * requirement ("communicating that the system adjusted the user's input" and
 * "providing a mechanism for the user to reverse the decision"), and this
 * surface had only the second half.
 *
 * IT IS SELF-DECLARED AND THE SCREEN SAYS SO. The server does not authenticate
 * this string; it stores what it is handed. What it buys is a three-way split
 * that is honest about exactly how much each part is worth:
 *
 *   - `system` — set by the server itself (`TodoComment::system_activity`,
 *     src/memory/types.rs:4037-4047, which hardcodes it). Not caller-supplied,
 *     so it is the only author value on the wire that is evidence.
 *   - this marker — set by nothing but this surface.
 *   - anything else — a caller that named itself, unverified. Every comment
 *     written before this convention is in this bucket and reads as such
 *     rather than being retroactively claimed.
 */
export const DASHBOARD_AUTHOR = "shodh-dashboard";

/**
 * `POST /api/todos/{todo_id}/comments` — src/handlers/router.rs:316-319,
 * `AddCommentRequest` src/handlers/todos.rs:339-346. `comment_type` is parsed
 * at todos.rs:2211-2231 and accepts `comment | progress | resolution |
 * activity`.
 *
 * A dismissal reason is written as `resolution` — the enum's own
 * "Resolution/fix description" (src/memory/types.rs:4061-4062) — so that the
 * reason is recoverable as data rather than as a convention this screen
 * invented and only this screen understands.
 */
export function addTodoComment(
  todoId: string,
  userId: string,
  content: string,
  commentType: TodoCommentType,
): Promise<CommentResponse> {
  return api.post<CommentResponse>(`/api/todos/${encodeURIComponent(todoId)}/comments`, {
    user_id: userId,
    content,
    comment_type: commentType,
    author: DASHBOARD_AUTHOR,
  });
}
