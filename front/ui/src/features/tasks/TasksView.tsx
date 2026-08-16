import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Check,
  Circle,
  CircleDashed,
  CircleDot,
  CircleSlash,
  Inbox,
  RotateCcw,
  X,
  type LucideIcon,
} from "lucide-react";
import { ApiError, NetworkError, type Reachability, type TodoStatus } from "@/lib/api";
import { useSession } from "@/stores/session";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import {
  addTodoComment,
  getMemory,
  listTriageTodos,
  updateTodoStatus,
  type TriageTodo,
} from "./api";
import {
  TASK_LIMIT,
  dueMeta,
  originOf,
  priorityLabel,
  provenanceOf,
  settledReason,
  shortId,
  summarise,
} from "./derive";

/**
 * Tasks — triage for work that arrived from somewhere else.
 *
 * WHAT THIS SCREEN IS. Nothing here is typed on this screen. Every item was
 * written by an API call — a person through the MCP tools, or an agent's
 * session hook mirroring a Claude Code task event
 * (hooks/memory-hook.ts:976-1006). So this is not a place to author work, and
 * a to-do list is the wrong shape for it. The reference class is triage: work
 * arrives, and a person decides whether it is real, whether it is theirs, and
 * when. The three actions on every row are that decision.
 *
 * THE CLAIM ON THE CAPTION IS NOT ONE THIS BACKEND CAN MAKE, AND IS NOT MADE
 * HERE. "Open work found in memory" implies extraction. There is none:
 * `store_todo` has exactly three callers — the create handler, MIF import, and
 * recurrence rollover (src/handlers/todos.rs:987, src/handlers/mif.rs:252,
 * src/memory/types.rs:4256-4272) — and no NLP or regex anywhere turns memory
 * text into a todo. Nothing was "found". The screen says "recorded", which is
 * what actually happened, and states in the open that the product does not
 * record who or what recorded it.
 *
 * WHY THERE IS NO "NEEDS A DECISION" BUCKET. A triage queue wants an
 * unconfirmed state to accept out of. `TodoStatus` (src/memory/types.rs:3827)
 * has six variants and none of them is "proposed" or "unconfirmed", and the
 * struct carries no origin, confidence or extraction flag of any kind
 * (types.rs:4067-4162). Dividing these rows into "unconfirmed" and "committed"
 * would have meant inventing a distinction the server cannot store and cannot
 * tell the TUI or the MCP tools about — a filter that looks like state. Two of
 * the three triage verbs DO map onto real, shared, persisted status: defer is
 * `backlog` (the enum's own "someday/maybe"), dismiss is `cancelled`. Accept
 * has nothing to map to, so it is not offered and its absence is explained on
 * screen rather than papered over.
 *
 * ROW ORDER IS THE SERVER'S AND IS STILL NOT TOUCHED — see derive.ts. The
 * server sorts by a MANUAL `sort_order` first; re-sorting here would discard
 * the one ordering a person actually set.
 */

const STATUS_ORDER: TodoStatus[] = ["in_progress", "blocked", "todo", "backlog"];

const STATUS_META: Record<TodoStatus, { label: string; icon: LucideIcon; iconClass: string }> = {
  in_progress: { label: "In progress", icon: CircleDot, iconClass: "text-foreground" },
  // Blocked is the one status that is itself "worth a look" rather than
  // ordinary workflow state — `--warn`, not the chrome accent.
  blocked: { label: "Blocked", icon: CircleSlash, iconClass: "text-warn" },
  todo: { label: "Todo", icon: Circle, iconClass: "text-muted-foreground" },
  backlog: { label: "Backlog", icon: CircleDashed, iconClass: "text-muted-foreground/60" },
  done: { label: "Done", icon: Check, iconClass: "text-muted-foreground" },
  cancelled: { label: "Dismissed", icon: X, iconClass: "text-muted-foreground" },
};

/** A field of the record that only some todos carry. Rendered as a labelled
 *  line rather than another chip: these are read, not scanned. */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 text-[12px] leading-relaxed">
      <span className="text-muted-foreground/70 w-16 shrink-0 text-[11px]">{label}</span>
      <span className="min-w-0 flex-1">{children}</span>
    </div>
  );
}

/**
 * Where a task came from — the one question this surface can answer and no
 * other screen can.
 *
 * FETCHED ON OPEN, NOT WITH THE LIST. `list_todos` sends `related_memory_ids`
 * but not the memories themselves, and resolving one costs a
 * `GET /api/memory/{id}` whose response carries a 384-float embedding and the
 * entire robotics block. Fifty rows' worth on arrival would be megabytes for
 * text almost nobody opens. That is a reduction, so the closed row says only
 * that links exist — never how many are sources, which is not knowable until
 * they are fetched and classified.
 *
 * THE ECHOES ARE THE POINT. Most links are the task's own lifecycle restating
 * itself (see derive.ts). They are counted, named, and kept out of the source
 * list, because a row that answers "where did this come from" with "[SHO-1]
 * Todo created: <this row's title>" is worse than a row that admits it has no
 * source.
 */
function SourceTrail({ todo, profile }: { todo: TriageTodo; profile: string }) {
  const ids = todo.related_memory_ids ?? [];

  const { data, error, isFetching } = useQuery({
    queryKey: ["todo-sources", profile, todo.id, ids.join(",")],
    queryFn: async ({ signal }) =>
      Promise.all(ids.map((id) => getMemory(id, profile, signal))),
    enabled: ids.length > 0,
    staleTime: 5 * 60 * 1000,
  });

  if (ids.length === 0) {
    return (
      <p className="text-muted-foreground/80 text-[11px] leading-relaxed">
        No memory is linked to this task, so there is nothing to trace it back to. A link is only
        recorded when whoever created the task passed one.
      </p>
    );
  }

  if (isFetching && !data) {
    return <Skeleton className="h-3 w-48" />;
  }

  if (error) {
    return (
      <p className="text-muted-foreground/80 text-[11px] leading-relaxed">
        {ids.length === 1 ? "The linked memory" : `All ${ids.length} linked memories`} could not be
        read
        {error instanceof ApiError ? ` — the server answered ${error.status}.` : "."}
      </p>
    );
  }

  if (!data) return null;

  const { sources, echoes } = provenanceOf(data);

  return (
    <div className="space-y-1.5">
      {sources.map((memory) => (
        <figure key={memory.id} className="border-primary/40 border-l pl-2.5">
          <blockquote className="text-[12px] leading-relaxed">
            {memory.experience.content}
          </blockquote>
          <figcaption className="mt-0.5">
            <Meta>
              <span>{memory.experience.experience_type}</span>
              <span>{new Date(memory.created_at).toLocaleDateString()}</span>
              <span className="mono">{memory.id.slice(0, 8)}</span>
            </Meta>
          </figcaption>
        </figure>
      ))}

      {sources.length === 0 ? (
        <p className="text-muted-foreground/80 text-[11px] leading-relaxed">
          No source memory. {echoes === 1 ? "The one link" : `All ${echoes} links`} on this task
          {echoes === 1 ? " is" : " are"} its own history — records this server wrote when the task
          was created or changed, which restate the task rather than explain it.
        </p>
      ) : echoes > 0 ? (
        <p className="text-muted-foreground/70 text-[11px]">
          {echoes} further {echoes === 1 ? "link is" : "links are"} this task's own history, not a
          source.
        </p>
      ) : null}
    </div>
  );
}

/**
 * The three triage decisions, on real persisted status.
 *
 * Dismissal takes two steps AND a reason. The reason is not ceremony: a
 * dismissed task is the only evidence anyone has that something was recorded
 * which should not have been, and a `cancelled` row with a blank beside it
 * carries none of that. It is written as a `resolution` comment
 * (src/memory/types.rs:4061) so the TUI and the MCP tools read it as data
 * rather than as a convention private to this screen.
 *
 * Nothing here deletes. `DELETE /api/todos/{id}` exists and is not called:
 * dismissing is a judgement that can be wrong, and Settled offers every one of
 * them a way back.
 */
function TriageActions({ todo, profile }: { todo: TriageTodo; profile: string }) {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [reason, setReason] = useState("");

  const refresh = () => {
    void queryClient.invalidateQueries({ queryKey: ["todos", profile] });
    void queryClient.invalidateQueries({ queryKey: ["todos-settled", profile] });
  };

  const move = useMutation({
    mutationFn: (status: "backlog" | "done") => updateTodoStatus(todo.id, profile, status),
    onSuccess: refresh,
  });

  const dismiss = useMutation({
    mutationFn: async (why: string) => {
      // Reason first. If the comment fails the task stays open and visible,
      // which is recoverable; cancelling first and failing to record why
      // would produce exactly the unexplained dismissal this is meant to
      // prevent.
      await addTodoComment(todo.id, profile, why, "resolution");
      return updateTodoStatus(todo.id, profile, "cancelled");
    },
    onSuccess: () => {
      setConfirming(false);
      setReason("");
      refresh();
    },
  });

  const busy = move.isPending || dismiss.isPending;
  const failure = move.error ?? dismiss.error;

  if (confirming) {
    return (
      <div className="border-destructive/40 space-y-1.5 border-l pl-2.5">
        <label htmlFor={`reason-${todo.id}`} className="block text-[12px]">
          Why is this not real work?
        </label>
        <input
          id={`reason-${todo.id}`}
          value={reason}
          autoFocus
          onChange={(e) => setReason(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") setConfirming(false);
            if (e.key === "Enter" && reason.trim()) dismiss.mutate(reason.trim());
          }}
          placeholder="Already done elsewhere / not mine / never was a task"
          aria-label={`Reason for dismissing ${shortId(todo)}`}
          className={cn(
            "border-input bg-background h-7 w-full rounded-md border px-2 text-[12px]",
            "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
          )}
        />
        <div className="flex items-center gap-1.5">
          <Button
            size="sm"
            variant="destructive"
            disabled={reason.trim().length === 0 || busy}
            onClick={() => dismiss.mutate(reason.trim())}
            aria-label={`Confirm dismissal of ${shortId(todo)}`}
          >
            {dismiss.isPending ? "Dismissing…" : "Dismiss"}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={busy}
            onClick={() => setConfirming(false)}
            aria-label="Keep this task open"
          >
            Keep
          </Button>
          <span className="text-muted-foreground/70 text-[11px]">
            Moves to Settled with this reason. Reversible from there.
          </span>
        </div>
        {dismiss.error ? (
          <p className="text-destructive text-[11px]">
            Not dismissed — {dismiss.error instanceof ApiError
              ? `the server answered ${dismiss.error.status}.`
              : "the server did not respond."}{" "}
            The task is unchanged.
          </p>
        ) : null}
      </div>
    );
  }

  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-1.5">
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => move.mutate("done")}
          aria-label={`Mark ${shortId(todo)} done`}
        >
          <Check aria-hidden="true" />
          Done
        </Button>
        {todo.status !== "backlog" ? (
          <Button
            size="sm"
            variant="ghost"
            disabled={busy}
            onClick={() => move.mutate("backlog")}
            aria-label={`Defer ${shortId(todo)} to backlog`}
          >
            Defer
          </Button>
        ) : null}
        <Button
          size="sm"
          variant="ghost"
          disabled={busy}
          onClick={() => setConfirming(true)}
          aria-label={`Dismiss ${shortId(todo)}`}
        >
          Dismiss…
        </Button>
      </div>
      {failure ? (
        <p className="text-destructive text-[11px]">
          Not changed —{" "}
          {failure instanceof ApiError
            ? `the server answered ${failure.status}.`
            : "the server did not respond."}
        </p>
      ) : null}
    </div>
  );
}

/**
 * One task: a single ~28px line, with its record, its origin and its decision
 * one click below.
 *
 * The title truncates rather than wrapping, and the metadata is right-aligned
 * where the eye finishes the row. Both only work because the click target
 * exists: a truncated line with nothing behind it loses information for good.
 */
function TaskRow({
  todo,
  profile,
  showProject,
}: {
  todo: TriageTodo;
  profile: string;
  showProject: boolean;
}) {
  const [open, setOpen] = useState(false);
  const meta = STATUS_META[todo.status];
  const Icon = meta.icon;
  const due = dueMeta(todo, Date.now());
  const priority = priorityLabel(todo.priority);
  const origin = originOf(todo);
  const linked = (todo.related_memory_ids ?? []).length;
  const captured = new Date(todo.created_at).toLocaleDateString();

  return (
    <div className="border-border border-b">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`${shortId(todo)} — ${todo.content}`}
        className={cn(
          "hover:bg-accent/60 flex w-full items-center gap-2.5 px-4 py-1.5 text-left",
          "transition-colors duration-100 motion-reduce:transition-none",
          "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        )}
      >
        <Icon aria-hidden="true" className={cn("size-3.5 shrink-0", meta.iconClass)} strokeWidth={1.8} />
        <span className={cn("min-w-0 flex-1 text-[13px]", open ? "whitespace-normal" : "truncate")}>
          {todo.content}
        </span>
        <Meta className="shrink-0 flex-nowrap">
          <span className="text-muted-foreground/70 mono text-[10px]">{shortId(todo)}</span>
          {showProject && todo.project_prefix ? (
            <span className="mono text-[10px]">{todo.project_prefix}</span>
          ) : null}
          {/* Links exist, count unqualified: how many are SOURCES is not
              knowable until they are fetched and classified on open. */}
          {linked > 0 ? (
            <span className="text-muted-foreground/70 text-[11px]">
              {linked} linked
            </span>
          ) : null}
          {todo.priority === "urgent" ? <Badge variant="destructive">urgent</Badge> : null}
          {todo.priority === "high" ? <Badge variant="warn">high</Badge> : null}
          {priority && todo.priority !== "urgent" && todo.priority !== "high" ? (
            <span className="text-muted-foreground/70 text-[11px]">{priority}</span>
          ) : null}
          {due ? (
            due.tone === "muted" ? (
              <span className="text-muted-foreground mono text-[10px]">{due.label}</span>
            ) : (
              <Badge variant={due.tone} className="mono">
                {due.label}
              </Badge>
            )
          ) : null}
        </Meta>
      </button>

      {open ? (
        <div className="space-y-2.5 px-4 pt-1 pb-3 pl-[26px]">
          {todo.notes ? <Field label="Notes">{todo.notes}</Field> : null}
          {todo.blocked_on ? (
            <Field label="Blocked on">
              <span className="text-warn">{todo.blocked_on}</span>
            </Field>
          ) : null}
          {todo.tags.length > 0 ? (
            <Field label="Tags">
              <span className="flex flex-wrap gap-1">
                {todo.tags.map((tag) => (
                  <Badge key={tag}>{tag}</Badge>
                ))}
              </span>
            </Field>
          ) : null}
          {todo.contexts.length > 0 ? (
            <Field label="Contexts">
              <span className="flex flex-wrap gap-1">
                {todo.contexts.map((context) => (
                  <Badge key={context}>{context}</Badge>
                ))}
              </span>
            </Field>
          ) : null}

          <div className="space-y-1">
            <p className="text-muted-foreground/70 text-[11px] font-medium tracking-wide uppercase">
              Where this came from
            </p>
            <SourceTrail todo={todo} profile={profile} />
            <Meta>
              <span>recorded {captured}</span>
              {/* Only stated when the backend actually stamped it. Silence
                  here means unrecorded, and the header says so. */}
              {origin.kind !== "unrecorded" ? <span>{origin.label}</span> : null}
            </Meta>
          </div>

          <TriageActions todo={todo} profile={profile} />
        </div>
      ) : null}
    </div>
  );
}

function StatusGroup({
  status,
  todos,
  profile,
  showProject,
}: {
  status: TodoStatus;
  todos: TriageTodo[];
  profile: string;
  showProject: boolean;
}) {
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
        <TaskRow key={t.id} todo={t} profile={profile} showProject={showProject} />
      ))}
    </section>
  );
}

/**
 * Settled — done and dismissed, with the reason.
 *
 * A separate request (`status: ["done","cancelled"]`, `include_completed:
 * true`), fired only when this is opened, because the point of the main list
 * is what is still outstanding and this would otherwise grow without bound.
 *
 * The reason is the whole reason this section exists. A dismissal is a
 * judgement that something was recorded which should not have been, and that
 * is the only feedback anyone gets about whatever is writing these tasks.
 * Anything settled outside this screen has no reason recorded, and that says
 * so explicitly rather than leaving a blank that reads like one.
 */
function SettledSection({ profile }: { profile: string }) {
  const [open, setOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data, error, isFetching } = useQuery({
    queryKey: ["todos-settled", profile],
    queryFn: ({ signal }) =>
      listTriageTodos(
        {
          user_id: profile,
          status: ["done", "cancelled"],
          include_completed: true,
          limit: TASK_LIMIT,
        },
        signal,
      ),
    enabled: open,
  });

  const reopen = useMutation({
    mutationFn: (id: string) => updateTodoStatus(id, profile, "todo"),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["todos", profile] });
      void queryClient.invalidateQueries({ queryKey: ["todos-settled", profile] });
    },
  });

  return (
    <section>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label="Settled work — done and dismissed tasks"
        className={cn(
          "hover:bg-accent/60 flex w-full items-center gap-2 border-b px-4 py-1.5 text-left",
          "border-border transition-colors duration-100 motion-reduce:transition-none",
          "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        )}
      >
        <Inbox aria-hidden="true" className="text-muted-foreground/60 size-3" strokeWidth={1.8} />
        <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          Settled
        </span>
        <span className="text-muted-foreground/60 text-[11px]">
          {open ? (data ? `${data.todos.length}` : "…") : "done and dismissed, with the reason"}
        </span>
      </button>

      {open ? (
        <div className="px-4 py-2">
          {isFetching && !data ? <Skeleton className="h-3 w-40" /> : null}
          {error ? (
            <p className="text-muted-foreground/80 text-[11px]">
              Settled work could not be loaded
              {error instanceof ApiError ? ` — the server answered ${error.status}.` : "."}
            </p>
          ) : null}
          {data && data.todos.length === 0 ? (
            <p className="text-muted-foreground/80 text-[11px] leading-relaxed">
              Nothing has been settled in this profile yet. Tasks land here once they are marked
              done or dismissed, and a dismissal keeps the reason it was dismissed for.
            </p>
          ) : null}
          {data?.todos.map((todo) => {
            const reason = settledReason(todo);
            const dismissed = todo.status === "cancelled";
            return (
              <div key={todo.id} className="border-border/60 flex items-start gap-2.5 border-b py-1.5 last:border-b-0">
                {dismissed ? (
                  <X aria-hidden="true" className="text-muted-foreground/60 mt-0.5 size-3 shrink-0" strokeWidth={1.8} />
                ) : (
                  <Check aria-hidden="true" className="text-muted-foreground/60 mt-0.5 size-3 shrink-0" strokeWidth={1.8} />
                )}
                <div className="min-w-0 flex-1">
                  <p className="text-muted-foreground text-[12px] leading-snug">{todo.content}</p>
                  {dismissed ? (
                    reason ? (
                      <p className="text-[11px] leading-relaxed">
                        <span className="text-muted-foreground/70">dismissed — </span>
                        {reason}
                      </p>
                    ) : (
                      <p className="text-muted-foreground/60 text-[11px]">
                        dismissed, no reason recorded
                      </p>
                    )
                  ) : reason ? (
                    <p className="text-[11px] leading-relaxed">
                      <span className="text-muted-foreground/70">done — </span>
                      {reason}
                    </p>
                  ) : null}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={reopen.isPending}
                  onClick={() => reopen.mutate(todo.id)}
                  aria-label={`Reopen ${shortId(todo)}`}
                >
                  <RotateCcw aria-hidden="true" />
                  Reopen
                </Button>
              </div>
            );
          })}
          {reopen.error ? (
            <p className="text-destructive text-[11px]">
              Not reopened —{" "}
              {reopen.error instanceof ApiError
                ? `the server answered ${reopen.error.status}.`
                : "the server did not respond."}
            </p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function TaskRowSkeleton({ width }: { width: string }) {
  return (
    <div className="border-border flex items-center gap-2.5 border-b px-4 py-1.5">
      <Skeleton className="size-3.5 shrink-0 rounded-full" />
      <Skeleton className="h-[13px] flex-1" style={{ maxWidth: width }} />
      <Skeleton className="h-2.5 w-10 shrink-0" />
    </div>
  );
}

export function TasksView({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  const enabled = reach.state === "online" && profile !== null;

  const { data, error, isFetching } = useQuery({
    queryKey: ["todos", profile],
    queryFn: ({ signal }) =>
      listTriageTodos({ user_id: profile!, limit: TASK_LIMIT }, signal),
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

  if (error) {
    const detail =
      error instanceof ApiError
        ? error.isAuthFailure
          ? "The server rejected this key."
          : `The server answered ${error.status}.`
        : error instanceof NetworkError
          ? "The server stopped responding mid-request."
          : "Something went wrong loading tasks.";
    return <EmptyState size="page" title="Could not load tasks" body={detail} />;
  }

  if (isFetching && !data) {
    return (
      <div className="mx-auto h-full w-full max-w-4xl">
        <div className="border-border border-b px-4 py-2.5">
          <Skeleton className="h-3 w-40" />
        </div>
        {["72%", "58%", "81%", "45%", "66%", "77%", "52%", "69%"].map((width, i) => (
          <TaskRowSkeleton key={i} width={width} />
        ))}
      </div>
    );
  }

  if (!data) return null;

  const summary = summarise(data.todos, data.count, Date.now());

  if (data.todos.length === 0) {
    return (
      <div className="mx-auto h-full w-full max-w-4xl">
        <EmptyState
          size="page"
          title="Tasks is where work recorded elsewhere gets decided on"
          body="Nothing is open in this profile. A task appears the moment a session records one — through the MCP tools or an agent's session hook — and this is where it is marked done, deferred or dismissed."
          more="Nothing is authored on this screen; every task arrived from somewhere else. Completed and dismissed work is not in this list — it is under Settled, which keeps the reason each one was dismissed for."
        />
        <SettledSection profile={profile} />
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-4xl pb-16">
        <header className="border-border flex flex-wrap items-center gap-x-3 gap-y-1.5 border-b px-4 py-2.5">
          <Meta className="text-[12px]">
            <Stat value={summary.shown} label="open" />
            {summary.overdue > 0 ? (
              <span className="text-destructive">{summary.overdue} overdue</span>
            ) : null}
            {summary.urgent > 0 ? <Stat value={summary.urgent} label="urgent" /> : null}
            {summary.high > 0 ? <Stat value={summary.high} label="high" /> : null}
            {summary.projects > 1 ? <Stat value={summary.projects} label="projects" /> : null}
          </Meta>
          <span className="text-muted-foreground/70 ml-auto flex items-center gap-1.5 text-[11px]">
            arrived from elsewhere
            <InfoHint label="what this list contains" align="right">
              Nothing here was typed on this screen. Every task was written by an API call — a
              person through the memory tools, or an agent's session hook mirroring a task event.
              Each row asks the same three questions: is it real, is it yours, and when.
              <br />
              <br />
              The server records no origin flag, so this screen cannot tell you which of those
              wrote a given task unless the writer stamped it, and it does not guess. There is also
              no "unconfirmed" state to accept work out of — the six statuses are backlog, todo, in
              progress, blocked, done and cancelled — so deferring and dismissing are offered and
              accepting is not.
              <br />
              <br />
              This list asks for open work only; done and dismissed tasks are under Settled. Rows
              keep the server's order: a manual position first, then priority, then due date.
            </InfoHint>
          </span>
          {summary.truncated ? (
            <p className="border-warn/40 text-muted-foreground/80 w-full border-l pl-2.5 text-[11px] leading-relaxed">
              Showing the first {summary.shown} of {summary.total}. The rest are past this screen's
              limit of {TASK_LIMIT} and are not counted in the figures above.
            </p>
          ) : null}
        </header>

        {STATUS_ORDER.map((status) => (
          <StatusGroup
            key={status}
            status={status}
            profile={profile}
            showProject={summary.projects > 1}
            todos={data.todos.filter((t) => t.status === status)}
          />
        ))}

        <SettledSection profile={profile} />
      </div>
    </ScrollArea>
  );
}
