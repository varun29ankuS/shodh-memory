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
}

/** `TodoListResponse` — src/handlers/todos.rs:211-218. `count` is the total
 *  BEFORE truncation (todos.rs:1421), which is the only way this screen learns
 *  it is showing a slice. */
export interface TriageListResponse {
  success: boolean;
  count: number;
  todos: TriageTodo[];
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
  });
}
