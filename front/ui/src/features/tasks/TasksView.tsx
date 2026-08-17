import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  Check,
  Circle,
  CircleDashed,
  CircleDot,
  CircleSlash,
  Cog,
  Inbox,
  Play,
  RotateCcw,
  User,
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
  DASHBOARD_AUTHOR,
  addTodoComment,
  getMemory,
  listTriageTodos,
  updateTodoStatus,
  type TriageTodo,
} from "./api";
import {
  TASK_LIMIT,
  axisOf,
  blockerIsSatisfied,
  blockersOf,
  boardOf,
  dueMeta,
  elapsedLabel,
  lanesOf,
  lifelineOf,
  originOf,
  priorityLabel,
  provenanceOf,
  settledReason,
  shortId,
  subtaskProgress,
  type Blocker,
  type Lane,
} from "./derive";
import { LaneStrip } from "./LaneStrip";

/**
 * Tasks — what needs doing, what is underway, how much has landed, and what is
 * waiting on something that has not arrived.
 *
 * WHAT CHANGED AND WHY. This screen used to render open work as a flat triage
 * queue, on the reading that the model had little to show. That was wrong. The
 * `Todo` struct carries projects, subtasks, structured dependencies with
 * server-side cycle rejection, a free-text "waiting on", and — the part nothing
 * was reading — a per-task activity log the server writes on every change. On
 * the live `claude-code` profile that log holds 204 entries across 93 tasks and
 * is the ONLY record of when work started, and for 43 of 82 settled tasks the
 * only record that they ever finished. The screen was showing a fraction of
 * what the profile knew.
 *
 * IT IS A LIST, NOT A BOARD, AND THAT IS A DECISION. A column per status is the
 * default shape for this kind of surface and it answers the wrong question
 * here: columns are for moving work by hand between states, and nothing on this
 * screen is authored or dragged — every task arrives from an API call. What a
 * reader needs is what changed and what is stuck, which is a vertical scan.
 * Linear's My Issues groups the same way and puts completed work last
 * (linear.app/docs/my-issues); the grouping order below follows that principle,
 * without its SLA, cycle and triage buckets, which have nothing behind them in
 * this model.
 *
 * THE ONE THING MAXIMISED IS MOVEMENT. Whether work is progressing is the
 * question a person actually arrives with, and it is the question a status
 * count cannot answer — "31 of 40 settled" reads identically for a project that
 * shipped steadily and one that sat still for three months. So the top of the
 * screen is a per-project curve of scope against settled work on a shared time
 * axis, and everything else recedes beneath it. What recedes: settled rows
 * (82 of 93 on the live profile — at equal weight they bury the open work),
 * tags, contexts, and the memory provenance, all of which move behind a click.
 *
 * NOTHING HERE IS EXTRACTED FROM MEMORY. `store_todo` has exactly three callers
 * — the create handler, MIF import and recurrence rollover — and no NLP
 * anywhere turns memory text into a task. The caption says "recorded", which is
 * what happened.
 *
 * THERE IS STILL NO ORIGIN FIELD, so this screen cannot say whether a person or
 * an agent wrote a given task, and does not guess. Linear's answer to the same
 * problem is to make the agent a `delegate` while a human stays the assignee,
 * so the two are always visible as different roles
 * (linear.app/developers/agents) — this model has neither field. What it does
 * have is `TodoComment.author`, which the server sets itself for its own
 * entries, and that is rendered for exactly what it is worth.
 */

/* -------------------------------------------------------------------------- *
 * TOKENS
 *
 * COLOUR MEANS TWO THINGS ON THIS SCREEN AND NO MORE. `--warn` is work waiting
 * on something; `--destructive` is late. The accent belongs to the curves,
 * where it carries the one quantity the screen exists to show. Status is
 * otherwise ink and shape, because six statuses would need six hues and the
 * product's rule reserves saturated colour for data.
 * -------------------------------------------------------------------------- */

const STATUS_META: Record<TodoStatus, { label: string; icon: LucideIcon; iconClass: string }> = {
  in_progress: { label: "Underway", icon: CircleDot, iconClass: "text-foreground" },
  blocked: { label: "Blocked", icon: CircleSlash, iconClass: "text-warn" },
  todo: { label: "To do", icon: Circle, iconClass: "text-muted-foreground" },
  backlog: { label: "Backlog", icon: CircleDashed, iconClass: "text-muted-foreground/60" },
  done: { label: "Done", icon: Check, iconClass: "text-muted-foreground" },
  cancelled: { label: "Dismissed", icon: X, iconClass: "text-muted-foreground" },
};

/** Underway first, then anything stuck, then work not yet picked up. Settled
 *  work is not in this list at all — it has its own section at the bottom. */
const OPEN_ORDER: TodoStatus[] = ["in_progress", "blocked", "todo", "backlog"];

const dateOf = (ms: number) => new Date(ms).toLocaleDateString();

/** A labelled line inside an opened row. Read, not scanned, so a line rather
 *  than another chip. */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 text-[12px] leading-relaxed">
      <span className="text-muted-foreground/70 w-[74px] shrink-0 text-[11px]">{label}</span>
      <span className="min-w-0 flex-1">{children}</span>
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * WAITING ON
 * -------------------------------------------------------------------------- */

/**
 * What a task is waiting for, with the two kinds kept apart.
 *
 * These are genuinely different objects and the screen would mislead if it drew
 * them alike. `blocked_by` names another task, which has a status of its own —
 * so the screen can say whether the thing being waited on is itself moving, and
 * whether it has already been settled and is simply stale. `blocked_on` is free
 * text about a person or a thing, which nothing can chase and which the product
 * cannot resolve to anything.
 *
 * Linear draws its equivalent as flags in the issue sidebar and does not make
 * blocking a status at all (linear.app/docs/issue-relations); this model has
 * the status AND both kinds of object, so the status carries the alarm and
 * these carry the answer to "on what".
 */
function Blockers({ blockers }: { blockers: readonly Blocker[] }) {
  if (blockers.length === 0) return null;
  return (
    <Field label="Waiting on">
      <span className="flex flex-wrap items-center gap-1.5">
        {blockers.map((blocker) => {
          if (blocker.kind === "waiting") {
            // Quoted, because it is somebody's words and not a value this
            // product understands.
            return (
              <span key={`w-${blocker.text}`} className="text-warn text-[12px]">
                “{blocker.text}”
              </span>
            );
          }
          if (blocker.kind === "task-missing") {
            return (
              <Badge key={blocker.id} variant="outline" className="text-muted-foreground/70">
                a task past this screen&apos;s limit
              </Badge>
            );
          }
          const satisfied = blockerIsSatisfied(blocker);
          return (
            <Badge
              key={blocker.id}
              variant={satisfied ? "outline" : "warn"}
              // Line-through, not just a colour change: a settled blocker is no
              // longer holding anything up and the server agrees
              // (`unblocked_by_completion` treats Done and Cancelled as
              // satisfied). A stale chain must not read as a live one.
              className={cn(satisfied && "line-through opacity-70")}
            >
              <span className="mono">{shortId(blocker.todo)}</span>
              <span className="max-w-[220px] truncate">{blocker.todo.content}</span>
            </Badge>
          );
        })}
      </span>
    </Field>
  );
}

/* -------------------------------------------------------------------------- *
 * WHAT HAPPENED TO THIS TASK
 * -------------------------------------------------------------------------- */

/**
 * Who did what to this task, and when — the audit trail for this domain.
 *
 * IT IS NOT THE SEAT'S TRAIL, AND /history CANNOT COVER THIS. The conversation
 * seat exposes three tools (`recall_memory`, `remember_memory`,
 * `record_seat_learning`, seat/src/memory-tools.ts) and its ledger has no todo
 * event kind at all (seat/src/ledger.ts), so no task change can ever appear in
 * the audit export /history renders. Todos are written through the MCP server
 * and the session hook, which do not pass through the seat. The record that
 * DOES exist is the todo's own `comments` array, and it ships inline with every
 * listed todo at no extra request.
 *
 * WHAT EACH AUTHOR IS WORTH, stated on the surface rather than flattened:
 * `system` is set by the server itself and is the only author value on the wire
 * that is evidence; this dashboard's own marker is set by nothing else; any
 * other name was chosen by whatever called the API and is unverified. Microsoft
 * HAX G9-C treats attribution and reversal as two halves of one requirement,
 * and both halves are on this row — the trail says what happened, and every
 * action below has an inverse.
 */
function TaskHistory({ todo }: { todo: TriageTodo }) {
  const comments = useMemo(
    () =>
      [...(todo.comments ?? [])].sort(
        (a, b) => Date.parse(a.created_at) - Date.parse(b.created_at),
      ),
    [todo.comments],
  );

  if (comments.length === 0) {
    return (
      <p className="text-muted-foreground/80 text-[11px] leading-relaxed">
        Nothing was recorded against this task beyond its creation.
      </p>
    );
  }

  return (
    <ol className="space-y-1">
      {comments.map((comment) => {
        const server = comment.author === "system";
        const here = comment.author === DASHBOARD_AUTHOR;
        return (
          <li key={comment.id} className="flex items-baseline gap-2 text-[11px] leading-relaxed">
            <time
              dateTime={comment.created_at}
              className="text-muted-foreground/70 mono w-[78px] shrink-0 text-[10px]"
            >
              {new Date(comment.created_at).toLocaleDateString()}
            </time>
            {server ? (
              <Cog aria-hidden="true" className="text-muted-foreground/60 size-3 shrink-0" strokeWidth={1.8} />
            ) : here ? (
              <User aria-hidden="true" className="text-muted-foreground size-3 shrink-0" strokeWidth={1.8} />
            ) : (
              <Bot aria-hidden="true" className="text-muted-foreground/60 size-3 shrink-0" strokeWidth={1.8} />
            )}
            <span className={cn("min-w-0 flex-1", server && "text-muted-foreground")}>
              {comment.content}
            </span>
            {!server ? (
              <span
                className="text-muted-foreground/60 shrink-0 text-[10px]"
                title={
                  here
                    ? "Written from this dashboard."
                    : "This name was supplied by whatever wrote the comment. The server does not verify it."
                }
              >
                {here ? "from this dashboard" : `${comment.author} — self-declared`}
              </span>
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

/* -------------------------------------------------------------------------- *
 * PROVENANCE
 * -------------------------------------------------------------------------- */

/**
 * Where a task came from — the one question this surface can answer and no
 * other screen can.
 *
 * FETCHED ON OPEN, NOT WITH THE LIST. Resolving one link costs a
 * `GET /api/memory/{id}` whose response carries a 384-float embedding and the
 * entire robotics block; fifty rows' worth on arrival would be megabytes for
 * text almost nobody opens.
 *
 * THE ECHOES ARE THE POINT. Most links are the task's own lifecycle restating
 * itself, so they are counted, named and kept out of the source list — a row
 * that answers "where did this come from" with "[SHO-1] Todo created: <this
 * row's title>" is worse than one that admits it has no source. On this
 * instance no todo in any profile carries a link at all, so what this renders
 * today is the honest empty case.
 */
function SourceTrail({ todo, profile }: { todo: TriageTodo; profile: string }) {
  const ids = todo.related_memory_ids ?? [];

  const { data, error, isFetching } = useQuery({
    queryKey: ["todo-sources", profile, todo.id, ids.join(",")],
    queryFn: async ({ signal }) => Promise.all(ids.map((id) => getMemory(id, profile, signal))),
    enabled: ids.length > 0,
    staleTime: 5 * 60 * 1000,
  });

  if (ids.length === 0) {
    return (
      <p className="text-muted-foreground/80 text-[11px] leading-relaxed">
        No memory is linked to this task. A link is only recorded when whoever created it passed
        one.
      </p>
    );
  }

  if (isFetching && !data) return <Skeleton className="h-3 w-48" />;

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
          <blockquote className="text-[12px] leading-relaxed">{memory.experience.content}</blockquote>
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
          {echoes} further {echoes === 1 ? "link is" : "links are"} this task&apos;s own history, not
          a source.
        </p>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * DECISIONS
 * -------------------------------------------------------------------------- */

/**
 * The moves, on real persisted status, each with its inverse on screen.
 *
 * EVERY ACTION IS REVERSIBLE FROM WHERE IT LANDS. Start is undone by Pause,
 * Done and Dismiss by Reopen in the section they fall into. Nothing here
 * deletes: `DELETE /api/todos/{id}` exists and is not called, because a
 * judgement that can be wrong should not be the one irreversible act on the
 * screen.
 *
 * The reversibility flavour is post-hoc undo rather than a staged proposal, and
 * that follows the blast radius: a status change is cheap, granular and
 * individually visible, which is the case where the products that ship both
 * patterns choose undo. It is also the only flavour available — there is no
 * proposed or unconfirmed status in the enum to stage into, and inventing one
 * would mean a state the server cannot store and cannot tell the TUI or the MCP
 * tools about.
 *
 * DISMISSAL TAKES A REASON. A dismissed task is the only evidence anyone has
 * that something was recorded which should not have been, and a `cancelled` row
 * with a blank beside it carries none of that. It is written as a `resolution`
 * comment so the TUI and MCP tools read it as data, and signed with this
 * surface's own author so the trail says where the decision was taken.
 */
function TaskActions({ todo, profile }: { todo: TriageTodo; profile: string }) {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const [reason, setReason] = useState("");

  const refresh = () => queryClient.invalidateQueries({ queryKey: ["tasks", profile] });

  const move = useMutation({
    mutationFn: (status: "todo" | "backlog" | "in_progress" | "done") =>
      updateTodoStatus(todo.id, profile, status),
    onSuccess: refresh,
  });

  const dismiss = useMutation({
    mutationFn: async (why: string) => {
      // Reason first. If the comment fails the task stays open and visible,
      // which is recoverable; cancelling first and failing to record why would
      // produce exactly the unexplained dismissal this exists to prevent.
      await addTodoComment(todo.id, profile, why, "resolution");
      return updateTodoStatus(todo.id, profile, "cancelled");
    },
    onSuccess: () => {
      setConfirming(false);
      setReason("");
      void refresh();
    },
  });

  const busy = move.isPending || dismiss.isPending;
  const failure = move.error ?? dismiss.error;
  const settled = todo.status === "done" || todo.status === "cancelled";

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
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => setConfirming(false)}>
            Keep
          </Button>
          <span className="text-muted-foreground/70 text-[11px]">
            Moves to Settled with this reason. Reversible from there.
          </span>
        </div>
        {dismiss.error ? (
          <p className="text-destructive text-[11px]">
            Not dismissed —{" "}
            {dismiss.error instanceof ApiError
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
        {settled ? (
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => move.mutate("todo")}
            aria-label={`Reopen ${shortId(todo)}`}
          >
            <RotateCcw aria-hidden="true" />
            Reopen
          </Button>
        ) : (
          <>
            {todo.status === "in_progress" ? (
              <Button
                size="sm"
                variant="ghost"
                disabled={busy}
                onClick={() => move.mutate("todo")}
                aria-label={`Stop work on ${shortId(todo)}`}
              >
                Pause
              </Button>
            ) : (
              <Button
                size="sm"
                variant="outline"
                disabled={busy}
                onClick={() => move.mutate("in_progress")}
                aria-label={`Start ${shortId(todo)}`}
              >
                <Play aria-hidden="true" />
                Start
              </Button>
            )}
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
          </>
        )}
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

/* -------------------------------------------------------------------------- *
 * ONE TASK
 * -------------------------------------------------------------------------- */

function TaskRow({
  todo,
  profile,
  all,
  byId,
  showProject,
}: {
  todo: TriageTodo;
  profile: string;
  all: readonly TriageTodo[];
  byId: ReadonlyMap<string, TriageTodo>;
  showProject: boolean;
}) {
  const [open, setOpen] = useState(false);
  const meta = STATUS_META[todo.status];
  const Icon = meta.icon;
  const due = dueMeta(todo, Date.now());
  const priority = priorityLabel(todo.priority);
  const blockers = blockersOf(todo, byId);
  const subtasks = subtaskProgress(todo, all);
  const line = lifelineOf(todo);
  const origin = originOf(todo);

  // Cycle time from the task's own record: recorded → settled where it landed,
  // recorded → now where it has not. The server's own "Marked complete after
  // 0.0 days" is NOT used — it is `{:.1}` rounded, so it reads 0.0 for anything
  // under about 72 minutes, and quoting it would restate a rounding as a
  // measurement.
  const took = line?.settled !== null && line !== null ? elapsedLabel(line.recorded, line.settled) : null;

  return (
    <div className="border-border border-b">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`${shortId(todo)} — ${todo.content}`}
        className={cn(
          "hover:bg-accent/60 flex w-full items-center gap-2.5 px-4 py-1.5 text-left",
          "transition-colors duration-100",
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
          {/* Only where there are real children to count. A ratio on a task
              with no subtasks would be invented. */}
          {subtasks ? (
            <span className="text-muted-foreground/80 mono text-[10px]">
              {subtasks.done}/{subtasks.total}
            </span>
          ) : null}
          {/* The count, not the detail: what it is waiting ON is a line in the
              body, because it is prose or a task title and neither fits here. */}
          {blockers.length > 0 ? (
            <Badge variant="warn">
              waiting on {blockers.length === 1 ? "1" : blockers.length}
            </Badge>
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
          <Blockers blockers={blockers} />
          {subtasks ? (
            <Field label="Subtasks">
              {subtasks.done} of {subtasks.total} settled
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
              What happened
            </p>
            <TaskHistory todo={todo} />
            <Meta>
              {line ? <span>recorded {dateOf(line.recorded)}</span> : null}
              {line?.started ? <span>started {dateOf(line.started)}</span> : null}
              {took ? <span>took {took}</span> : null}
              {/* Stated only where it is true, and it is true for more than half
                  the settled work on the live profile. */}
              {line?.settledFromLog ? (
                <span title="No completed_at was written; this came from the activity log.">
                  completion time from the log
                </span>
              ) : null}
              {origin.kind !== "unrecorded" ? <span>{origin.label}</span> : null}
            </Meta>
          </div>

          <div className="space-y-1">
            <p className="text-muted-foreground/70 text-[11px] font-medium tracking-wide uppercase">
              Where this came from
            </p>
            <SourceTrail todo={todo} profile={profile} />
          </div>

          <TaskActions todo={todo} profile={profile} />
        </div>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * HOW THE WORK HAS MOVED — the hero
 * -------------------------------------------------------------------------- */

/**
 * One row per project: the ratio, and the shape of how it got there.
 *
 * A LANE IS SELECTABLE AND FILTERS THE LIST BELOW. That is the whole
 * interaction: the curves answer "which project is moving", and the list
 * answers "what exactly is in it", and clicking joins the two. Selection is a
 * real pressed state with a surface change, never colour alone.
 */
function LaneRow({
  lane,
  axis,
  selected,
  suppressRatio,
  onSelect,
}: {
  lane: Lane;
  axis: ReturnType<typeof axisOf>;
  selected: boolean;
  suppressRatio: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onSelect}
      className={cn(
        "grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 px-4 py-1.5 text-left",
        "sm:grid-cols-[minmax(0,150px)_minmax(0,1fr)_auto]",
        "transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        selected ? "bg-accent" : "hover:bg-accent/60",
      )}
    >
      <span className="flex min-w-0 items-baseline gap-1.5">
        <span className="truncate text-[12px]">{lane.name}</span>
        {/* Archived is stated, not implied by position: a lane that is finished
            for good reads differently from one that has merely gone quiet. */}
        {lane.archived ? (
          <span className="text-muted-foreground/60 shrink-0 text-[10px]">archived</span>
        ) : null}
      </span>

      <span className="col-span-2 sm:col-span-1 sm:min-w-0">
        {axis ? (
          <LaneStrip lane={lane} axis={axis} />
        ) : (
          <span className="text-muted-foreground/50 block text-[10px]">not yet moved</span>
        )}
      </span>

      <Meta className="shrink-0 flex-nowrap justify-self-end">
        {suppressRatio ? (
          <span className="text-muted-foreground/70">{lane.total} shown</span>
        ) : (
          <Stat value={`${lane.settled}/${lane.total}`} label="settled" />
        )}
        {lane.underway > 0 ? <Stat value={lane.underway} label="underway" /> : null}
        {lane.blocked > 0 ? <span className="text-warn">{lane.blocked} blocked</span> : null}
      </Meta>
    </button>
  );
}

/* -------------------------------------------------------------------------- *
 * THE SCREEN
 * -------------------------------------------------------------------------- */

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
  const [lane, setLane] = useState<string | null>(null);
  const [showSettled, setShowSettled] = useState(false);
  const enabled = reach.state === "online" && profile !== null;

  /**
   * ONE REQUEST FOR EVERYTHING, WHICH IS WHAT MAKES A RATIO POSSIBLE. The
   * previous shape asked for open work and fetched settled work separately and
   * lazily, so the screen never held a project's whole population at once and
   * could not have computed a denominator if it had wanted one. On the largest
   * live profile this is 93 rows in a single call.
   */
  const { data, error, isFetching } = useQuery({
    queryKey: ["tasks", profile],
    queryFn: ({ signal }) =>
      listTriageTodos({ user_id: profile!, limit: TASK_LIMIT, include_completed: true }, signal),
    enabled,
  });

  const todos = useMemo(() => data?.todos ?? [], [data]);
  const byId = useMemo(() => new Map(todos.map((t) => [t.id, t])), [todos]);
  const lanes = useMemo(() => lanesOf(todos, data?.projects ?? []), [todos, data?.projects]);
  const board = useMemo(() => boardOf(todos, data?.count ?? 0, lanes), [todos, data?.count, lanes]);
  const axis = useMemo(() => axisOf(lanes, board), [lanes, board]);

  const visible = useMemo(
    () => (lane === null ? todos : todos.filter((t) => (t.project_id ?? "") === lane)),
    [todos, lane],
  );
  const settled = useMemo(
    () => visible.filter((t) => t.status === "done" || t.status === "cancelled"),
    [visible],
  );

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="Recorded work appears here once the memory server is running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState size="page" title="No profile to browse" body="This instance holds no memory yet." />
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

  /**
   * THE HONEST EMPTY STATE. `defence-live` and `gdelt-bridge` hold zero todos,
   * and that is not a fault — nothing has ever recorded work against them.
   * Saying "no tasks" and stopping would leave a reader unable to tell an
   * empty profile from a broken screen, so it says what would put one here.
   */
  if (todos.length === 0) {
    return (
      <ScrollArea className="h-full">
        <div className="mx-auto max-w-4xl pb-16">
          <div className="py-20">
            <EmptyState
              size="page"
              title="No work has been recorded against this profile"
              body="A task appears the moment something records one — the memory tools, an import, or an agent's session hook — and this is where it is started, finished, deferred or dismissed."
              more="Nothing is authored on this screen and nothing is extracted from memory: creating a task is an explicit API call, and no part of the server turns memory text into one. A profile can hold a great deal of memory and no tasks at all, which is what this is."
            />
          </div>
        </div>
      </ScrollArea>
    );
  }

  const laneName = lane === null ? null : lanes.find((l) => l.key === lane)?.name;

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-4xl pb-16">
        {/* WHAT IS IN HERE. Figures first, then the two limits that change how
            they should be read — on the surface, not behind the info
            affordance. HAX G11 warns that an explanation increases trust by its
            mere presence, so the caveats sit where the numbers are. */}
        <header className="border-border border-b px-4 py-2.5">
          <Meta className="text-[12px]">
            <Stat value={board.open} label="open" />
            {board.underway > 0 ? <Stat value={board.underway} label="underway" /> : null}
            {board.blocked > 0 ? (
              <span className="text-warn">{board.blocked} blocked</span>
            ) : null}
            <Stat value={board.settled} label="settled" />
            {board.projects > 1 ? <Stat value={board.projects} label="projects" /> : null}
          </Meta>

          {/* "ARE WE BLOCKED" IS ANSWERED EVERY TIME, INCLUDING WHEN THE ANSWER
              IS NO — it is one of the questions this screen exists for, and a
              silence would be indistinguishable from a screen that cannot tell.
              It is TWO claims when the answer is no, because the second is what
              makes the first weak evidence. */}
          <p className="text-muted-foreground mt-1.5 text-[12px] leading-relaxed">
            {board.blocked > 0 || board.waiting > 0 ? (
              <>
                <span className="text-warn">
                  {board.blocked > 0
                    ? `${board.blocked} ${board.blocked === 1 ? "task is" : "tasks are"} blocked`
                    : `${board.waiting} ${board.waiting === 1 ? "task names" : "tasks name"} something it is waiting on`}
                </span>
                {board.dependencies > 0
                  ? ` — ${board.dependencies} ${board.dependencies === 1 ? "names another task" : "name other tasks"} as the blocker.`
                  : " — none of it names another task as the blocker, so the chain cannot be walked."}
              </>
            ) : (
              <>
                Nothing is marked blocked. No task in this profile has ever recorded a dependency on
                another either, so that is the absence of a signal rather than evidence the work is
                clear.
              </>
            )}
          </p>

          <dl className="border-border mt-2 grid grid-cols-[86px_1fr] gap-x-2 gap-y-1 border-l-2 pl-2.5 text-[11px] leading-relaxed">
            <dt className="text-muted-foreground/70">Proves</dt>
            <dd>
              what was recorded, what state each task is in now, and — from the server&apos;s own
              activity log — when it started and when it settled.
            </dd>
            <dt className="text-muted-foreground/70">Cannot prove</dt>
            <dd className="text-muted-foreground">
              who recorded any of it. There is no origin field on a task, so this screen cannot tell
              a person from an agent and does not guess.
            </dd>
            <dt className="text-muted-foreground/70">Does not cover</dt>
            <dd className="text-muted-foreground">
              anything not written down as a task. Nothing is extracted from memory — a task exists
              only because something called the API to create one.
              <InfoHint label="what this screen counts and how" className="ml-1.5 translate-y-[2px]">
                Progress is a count of tasks, because the model has no estimate or size field to
                weight by. A project&apos;s figure is settled tasks over the tasks in this profile,
                counted here rather than read from the server&apos;s own per-project counters, which
                are never populated and ship as zeros.
                <br />
                <br />
                Done and dismissed both count as settled: a dropped task is not outstanding work.
                They are reported apart in the Settled section, because finishing something and
                abandoning it are different outcomes.
                <br />
                <br />
                No individual task gets a percentage. A number appears only where there is a real
                population behind it — a project, or a task&apos;s own subtasks.
                <br />
                <br />
                The curves stop at today. Nothing is projected forward: there is no estimate, cycle
                or velocity here to extrapolate from.
              </InfoHint>
            </dd>
          </dl>

          {board.truncated ? (
            <p className="border-warn/40 text-muted-foreground/80 mt-2 border-l pl-2.5 text-[11px] leading-relaxed">
              Showing {board.shown} of {board.total} tasks — the rest are past this screen&apos;s
              limit of {TASK_LIMIT}. Every ratio is withheld while that is true: the server
              paginates by priority and due date rather than by project, so this is an arbitrary
              slice across all of them and no project&apos;s figure would be its own.
            </p>
          ) : null}
        </header>

        {/* THE HERO. Suppressed entirely when there is only one project and it
            has never moved — a single flat curve is a one-bar bar chart. */}
        {lanes.length > 0 ? (
          <section className="border-border border-b py-1.5">
            <div className="flex items-baseline justify-between gap-2 px-4 pb-1">
              <h2 className="text-[12px] font-medium tracking-tight">
                {axis ? "How the work has moved" : "What has been recorded"}
              </h2>
              {axis ? (
                <Meta className="mono text-[10px]">
                  <span>{dateOf(axis.from)}</span>
                  <span>{dateOf(axis.to)}</span>
                </Meta>
              ) : null}
            </div>

            {/* THE FINDING THAT REPLACES THE DRAWING. On the live `claude`
                profile all 50 tasks were recorded inside 33 minutes on one day
                and not one has ever changed state. A timeline there would plot
                the instants rows were written and read as a picture of
                progress, so it says the thing instead. */}
            {!axis && board.from !== null && board.to !== null ? (
              <p className="text-muted-foreground border-warn/40 mx-4 mb-1.5 border-l pl-2.5 text-[11px] leading-relaxed">
                Nothing in this profile has ever changed state.{" "}
                {board.shown === 1 ? "The one task was" : `All ${board.shown} tasks were`} recorded
                {board.to - board.from < 3_600_000
                  ? ` within ${elapsedLabel(board.from, board.to) ?? "moments"} on ${dateOf(board.from)}`
                  : ` between ${dateOf(board.from)} and ${dateOf(board.to)}`}
                , and none has been started, finished or dismissed since.
              </p>
            ) : null}

            {lanes.map((l) => (
              <LaneRow
                key={l.key}
                lane={l}
                axis={axis}
                selected={lane === l.key}
                suppressRatio={board.truncated}
                onSelect={() => setLane(lane === l.key ? null : l.key)}
              />
            ))}
          </section>
        ) : null}

        {laneName ? (
          <div className="border-border flex items-center gap-2 border-b px-4 py-1.5">
            <span className="text-muted-foreground text-[11px]">
              Showing {laneName} only — {visible.length} of {board.shown} tasks.
            </span>
            <Button size="sm" variant="ghost" onClick={() => setLane(null)}>
              Show all
            </Button>
          </div>
        ) : null}

        {OPEN_ORDER.map((status) => {
          const rows = visible.filter((t) => t.status === status);
          if (rows.length === 0) return null;
          const meta = STATUS_META[status];
          const Icon = meta.icon;
          return (
            <section key={status}>
              <div className="border-border bg-muted/50 sticky top-0 z-10 flex items-center gap-2 border-b px-4 py-1.5 backdrop-blur-sm">
                <Icon aria-hidden="true" className={cn("size-3", meta.iconClass)} strokeWidth={1.8} />
                <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
                  {meta.label}
                </span>
                <span className="text-muted-foreground/60 mono text-[10px]">{rows.length}</span>
              </div>
              {rows.map((t) => (
                <TaskRow
                  key={t.id}
                  todo={t}
                  profile={profile}
                  all={todos}
                  byId={byId}
                  showProject={lane === null && lanes.length > 1}
                />
              ))}
            </section>
          );
        })}

        {visible.length === settled.length && settled.length > 0 ? (
          <p className="text-muted-foreground/80 px-4 py-3 text-[11px] leading-relaxed">
            {laneName ? `Nothing is open in ${laneName}.` : "Nothing is open in this profile."} All{" "}
            {settled.length} recorded {settled.length === 1 ? "task has" : "tasks have"} been
            settled.
          </p>
        ) : null}

        {/* SETTLED RECEDES. It is 82 of 93 rows on the live profile; at equal
            weight it buries the work that still needs a decision. It is also
            the only place a dismissal reason is readable, so it is one click
            away rather than gone. */}
        {settled.length > 0 ? (
          <section>
            <button
              type="button"
              onClick={() => setShowSettled((v) => !v)}
              aria-expanded={showSettled}
              className={cn(
                "hover:bg-accent/60 flex w-full items-center gap-2 border-b px-4 py-1.5 text-left",
                "border-border transition-colors duration-100",
                "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
              )}
            >
              <Inbox aria-hidden="true" className="text-muted-foreground/60 size-3" strokeWidth={1.8} />
              <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
                Settled
              </span>
              <span className="text-muted-foreground/60 mono text-[10px]">{settled.length}</span>
              <span className="text-muted-foreground/60 text-[11px]">
                {showSettled ? "" : "done and dismissed, with the reason"}
              </span>
            </button>

            {showSettled ? (
              <div>
                {settled.map((todo) => {
                  const reason = settledReason(todo);
                  const dismissed = todo.status === "cancelled";
                  const line = lifelineOf(todo);
                  const took = line && line.settled !== null ? elapsedLabel(line.recorded, line.settled) : null;
                  return (
                    <div
                      key={todo.id}
                      className="border-border/60 flex items-start gap-2.5 border-b px-4 py-1.5"
                    >
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
                        <Meta>
                          {line?.settled ? <span>{dateOf(line.settled)}</span> : null}
                          {took ? <span>took {took}</span> : null}
                        </Meta>
                      </div>
                      <TaskActions todo={todo} profile={profile} />
                    </div>
                  );
                })}
              </div>
            ) : null}
          </section>
        ) : null}
      </div>
    </ScrollArea>
  );
}
