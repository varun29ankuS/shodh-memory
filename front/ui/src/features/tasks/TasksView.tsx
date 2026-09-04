import { useQuery } from "@tanstack/react-query";
import {
  Circle,
  CircleCheck,
  CircleDashed,
  CircleDot,
  CircleSlash,
  CircleX,
  type LucideIcon,
} from "lucide-react";
import { listTodos, ApiError, NetworkError, type Reachability, type Todo, type TodoStatus } from "@/lib/api";
import { useSession } from "@/stores/session";
import { EmptyState } from "@/components/ui/empty-state";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { completedOrder, isOverdue, nextUp } from "./ordering";

/**
 * Tasks — GTD todos captured from sessions, against the real
 * `POST /api/todos` (src/handlers/todos.rs:1160).
 *
 * WORKFLOWS.md's Chain 4 reads "Tasks → task → the session that captured it
 * → Chain 2", handing off into the Inspector like every other list. That
 * hand-off is not built here, deliberately: `Todo` (src/memory/types.rs:3646)
 * carries no session reference — only `related_memory_ids`, a link to
 * memories, not to the session that created them — and the Inspector reads
 * its data out of the `["recall", profile, query]` react-query cache, which
 * a todo was never part of. Wiring a `select()` call here would open the
 * Inspector on a stale or absent cache entry. This is one list that does not
 * feed the one Inspector; it is a real, standalone list against real data
 * instead of a half-wired chain.
 *
 * TWO QUERIES, AND WHY NOT ONE. Open work and finished work are fetched
 * separately. `include_completed: true` would return both from a single call,
 * but the handler paginates AFTER filtering and applies no ordering of its own
 * (todos.rs:1419-1432), so on this profile 82 finished todos share a limit with
 * 11 open ones and the open ones are what would be lost. Passing
 * `status: ["done","cancelled"]` instead routes to a dedicated store lookup
 * (todos.rs:1303-1307), which is a narrower question and a truthful one.
 *
 * It also leaves `["todos", profile]` alone. The briefing's work panel reads
 * that exact entry with that exact body; widening it here would have changed
 * what a different screen renders, through a cache, with nothing in either file
 * to say so.
 *
 * NOTHING ON THIS SCREEN EDITS. Todos are captured from sessions rather than
 * entered (see above), so there is no checkbox, no drag, no inline edit. A
 * control that looks actionable and is not costs more trust than the
 * convenience it fakes would earn.
 */

const STATUS_ORDER: TodoStatus[] = ["in_progress", "blocked", "todo", "backlog"];

const STATUS_META: Record<TodoStatus, { label: string; icon: LucideIcon; iconClass: string }> = {
  in_progress: { label: "In progress", icon: CircleDot, iconClass: "text-foreground" },
  // Blocked is the one status that is itself "worth a look" rather than
  // ordinary workflow state — `--warn`, not the chrome accent.
  blocked: { label: "Blocked", icon: CircleSlash, iconClass: "text-warn" },
  todo: { label: "Todo", icon: Circle, iconClass: "text-muted-foreground" },
  backlog: { label: "Backlog", icon: CircleDashed, iconClass: "text-muted-foreground/60" },
  done: { label: "Done", icon: CircleCheck, iconClass: "text-[var(--live)]" },
  cancelled: { label: "Cancelled", icon: CircleX, iconClass: "text-muted-foreground/60" },
};

/** Mirrors `Todo::short_id()` (src/memory/types.rs:3776-3784) exactly,
 *  including its "SHO" fallback prefix and its first-4-hex-chars fallback
 *  for legacy todos with no `seq_num` — neither is invented here. */
function shortId(t: Todo): string {
  if (t.seq_num > 0) return `${t.project_prefix ?? "SHO"}-${t.seq_num}`;
  return `SHO-${t.id.slice(0, 4)}`;
}

/** `Todo::is_overdue()` (src/memory/types.rs:3787-3795), re-derived: the
 *  server does not send a boolean, only `due_date`. */
function dueMeta(t: Todo): { label: string; tone: "destructive" | "warn" | "muted" } | null {
  if (!t.due_date) return null;
  const due = new Date(t.due_date);
  if (Number.isNaN(due.getTime())) return null;
  const days = Math.ceil((due.getTime() - Date.now()) / 86_400_000);
  if (days < 0) return { label: `Overdue ${Math.abs(days)}d`, tone: "destructive" };
  if (days === 0) return { label: "Due today", tone: "warn" };
  if (days <= 3) return { label: `Due in ${days}d`, tone: "warn" };
  return { label: `Due ${due.toLocaleDateString()}`, tone: "muted" };
}

function TaskRow({ todo }: { todo: Todo }) {
  const meta = STATUS_META[todo.status];
  const Icon = meta.icon;
  const due = dueMeta(todo);
  // Cancelled is not a quieter kind of done. Struck through and dimmed, so 23
  // abandoned items among 59 finished ones cannot read as 82 completed.
  const abandoned = todo.status === "cancelled";

  return (
    <div className="border-border flex items-start gap-3 border-b px-4 py-3">
      <Icon aria-hidden="true" className={cn("mt-0.5 size-3.5 shrink-0", meta.iconClass)} strokeWidth={1.8} />
      <div className="min-w-0 flex-1">
        <p
          className={cn(
            "line-clamp-2 text-[13px] leading-relaxed",
            abandoned && "text-muted-foreground line-through decoration-1",
          )}
        >
          {todo.content}
        </p>
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
          <span className="text-muted-foreground/70 mono text-[10px]">{shortId(todo)}</span>
          {todo.project_prefix ? <Badge>{todo.project_prefix}</Badge> : null}
          {todo.priority === "urgent" ? <Badge variant="destructive">Urgent</Badge> : null}
          {todo.priority === "high" ? <Badge variant="warn">High</Badge> : null}
          {due ? (
            due.tone === "muted" ? (
              <span className="text-muted-foreground mono text-[10px]">{due.label}</span>
            ) : (
              <Badge variant={due.tone} className="mono">
                {due.label}
              </Badge>
            )
          ) : null}
        </div>
      </div>
    </div>
  );
}

function StatusGroup({ status, todos }: { status: TodoStatus; todos: Todo[] }) {
  if (todos.length === 0) return null;
  const meta = STATUS_META[status];
  const Icon = meta.icon;
  return (
    <section>
      <div className="border-border bg-muted/50 sticky top-0 z-10 flex items-center gap-2 border-b px-4 py-1.5 backdrop-blur-sm">
        <Icon aria-hidden="true" className={cn("size-3", meta.iconClass)} strokeWidth={1.8} />
        <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          {meta.label}
        </span>
        <span className="text-muted-foreground/60 mono text-[10px]">{todos.length}</span>
      </div>
      {todos.map((t) => (
        <TaskRow key={t.id} todo={t} />
      ))}
    </section>
  );
}

function TaskRowSkeleton() {
  return (
    <div className="border-border flex items-start gap-3 border-b px-4 py-3">
      <Skeleton className="mt-0.5 size-3.5 shrink-0 rounded-full" />
      <div className="min-w-0 flex-1">
        <Skeleton className="h-[13px] w-[75%]" />
        <div className="mt-1.5 flex items-center gap-1.5">
          <Skeleton className="h-2.5 w-12" />
          <Skeleton className="h-2.5 w-10" />
        </div>
      </div>
    </div>
  );
}

/** The response's `count` is the total before truncation, so the board can say
 *  "50 of 82" rather than implying 82 is all there is. */
const COMPLETED_LIMIT = 50;

function describe(error: unknown): string {
  if (error instanceof ApiError) {
    return error.isAuthFailure
      ? "the server rejected this key."
      : `the server answered ${error.status}.`;
  }
  if (error instanceof NetworkError) return "the server stopped responding mid-request.";
  return "something went wrong.";
}

function Stat({ label, value, tone }: { label: string; value: number; tone?: "warn" | "destructive" }) {
  // A zero is dimmed rather than dropped: "0 blocked" is worth reading, and a
  // strip whose columns move between profiles is harder to scan than one that
  // always says the same six things.
  return (
    <div className="flex items-baseline gap-1.5">
      <dd
        className={cn(
          "text-base font-semibold tabular-nums",
          value === 0
            ? "text-muted-foreground/50"
            : tone === "destructive"
              ? "text-destructive"
              : tone === "warn"
                ? "text-warn"
                : "text-foreground",
        )}
      >
        {value}
      </dd>
      <dt className="text-muted-foreground mono text-[10px] tracking-wide uppercase">{label}</dt>
    </div>
  );
}

function ColumnHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-muted-foreground mono mb-2 text-[11px] tracking-widest uppercase">
      {children}
    </h2>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return (
    <p className="border-border text-muted-foreground rounded border border-dashed px-4 py-3 text-xs">
      {children}
    </p>
  );
}

export function TasksView({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  const enabled = reach.state === "online" && profile !== null;

  // Unchanged, and deliberately so: the briefing's work panel reads this exact
  // key with this exact body, and they must stay one cache entry.
  const open = useQuery({
    queryKey: ["todos", profile],
    queryFn: ({ signal }) => listTodos({ user_id: profile!, limit: 200 }, signal),
    enabled,
  });

  const done = useQuery({
    queryKey: ["todos", profile, "completed"],
    queryFn: ({ signal }) =>
      listTodos(
        { user_id: profile!, status: ["done", "cancelled"], limit: COMPLETED_LIMIT },
        signal,
      ),
    enabled,
  });

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="Open work appears here once the memory server is running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile to browse"
        body="This instance holds no memory yet."
      />
    );
  }

  // Only the OPEN query can empty the whole board. The two fetch separately and
  // fail separately, so each column reports its own state below — letting
  // either speak for the view would hide one column's data behind the other's
  // words.
  if (open.error) {
    return <EmptyState size="page" title="Could not load tasks" body={describe(open.error)} />;
  }

  if (open.isFetching && !open.data) {
    return (
      <div className="mx-auto h-full w-full max-w-2xl">
        {Array.from({ length: 6 }, (_, i) => (
          <TaskRowSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (!open.data) return null;

  const todos = open.data.todos;
  const completed = done.data ? completedOrder(done.data.todos) : [];
  const upcoming = nextUp(todos);
  const overdue = todos.filter((t) => isOverdue(t));

  // A profile whose work is all finished has nothing open and plenty of
  // history. The previous full-page "Nothing outstanding" fired on open work
  // alone, and would now hide exactly the column this board was asked for.
  if (todos.length === 0 && completed.length === 0) {
    return (
      <EmptyState
        size="page"
        title="Nothing outstanding"
        body="No todos in this profile."
        more="Tasks are picked up from what was recorded in a session — yours or an agent's — rather than entered here, so they appear as work is captured."
      />
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-6xl px-4 py-4">
        <dl className="border-border mb-4 flex flex-wrap items-baseline gap-x-7 gap-y-2 border-b pb-4">
          <Stat label="open" value={todos.length} />
          <Stat label="in progress" value={todos.filter((t) => t.status === "in_progress").length} />
          <Stat label="blocked" value={todos.filter((t) => t.status === "blocked").length} tone="warn" />
          <Stat label="overdue" value={overdue.length} tone="destructive" />
          {/* One figure, because only one is knowable. Splitting this into done
              and cancelled read "68 done, 14 cancelled" against a true 59 and
              23: `count` is the total before truncation, `cancelled` was
              counted from the 50 rows that came back, and subtracting a sample
              from a total produces a number that is neither. The split is real
              per row -- each carries its own icon, and cancelled work is struck
              through -- it just cannot be totalled without fetching all of it. */}
          <Stat label="completed" value={done.data?.count ?? 0} />
        </dl>

        {/* One column below lg. This is read on a laptop, and two 300px columns
            are worse than one readable one. */}
        <div className="grid gap-x-8 gap-y-6 lg:grid-cols-[1.45fr_1fr]">
          <div className="min-w-0">
            <ColumnHeading>Open</ColumnHeading>
            {todos.length === 0 ? (
              <Note>Nothing open in this profile.</Note>
            ) : (
              <div className="border-border overflow-hidden rounded border">
                {STATUS_ORDER.map((status) => (
                  <StatusGroup
                    key={status}
                    status={status}
                    todos={todos.filter((t) => t.status === status)}
                  />
                ))}
              </div>
            )}
          </div>

          <div className="min-w-0 space-y-6">
            <section>
              <ColumnHeading>Next up</ColumnHeading>
              {upcoming.length === 0 ? (
                /* Stated rather than hidden: an absent section is
                   indistinguishable from one that was never built. */
                <Note>Nothing overdue, due within three days, or marked urgent.</Note>
              ) : (
                <div className="border-border overflow-hidden rounded border">
                  {upcoming.map((t) => (
                    <TaskRow key={t.id} todo={t} />
                  ))}
                </div>
              )}
            </section>

            <section>
              <ColumnHeading>
                Completed
                {done.data && done.data.count > completed.length ? (
                  <span className="text-muted-foreground/60 mono ml-2 text-[10px] normal-case">
                    {completed.length} of {done.data.count}
                  </span>
                ) : null}
              </ColumnHeading>
              {done.error ? (
                <Note>Finished work could not be read — {describe(done.error)}</Note>
              ) : done.isFetching && !done.data ? (
                <div className="border-border overflow-hidden rounded border">
                  {Array.from({ length: 3 }, (_, i) => (
                    <TaskRowSkeleton key={i} />
                  ))}
                </div>
              ) : completed.length === 0 ? (
                <Note>Nothing finished in this profile yet.</Note>
              ) : (
                <div className="border-border overflow-hidden rounded border">
                  {completed.map((t) => (
                    <TaskRow key={t.id} todo={t} />
                  ))}
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </ScrollArea>
  );
}
