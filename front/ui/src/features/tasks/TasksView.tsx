import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Circle,
  CircleDashed,
  CircleDot,
  CircleSlash,
  type LucideIcon,
} from "lucide-react";
import { listTodos, ApiError, NetworkError, type Reachability, type Todo, type TodoStatus } from "@/lib/api";
import { useSession } from "@/stores/session";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { TASK_LIMIT, dueMeta, priorityLabel, shortId, summarise } from "./derive";

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
 * Inspector on a stale or absent cache entry.
 *
 * WHAT A ROW LEADS TO INSTEAD. It was leading nowhere at all: fifty inert
 * two-line blocks, each clamped at `line-clamp-2` with no way to see the rest
 * of a title that ran past it, and no sight of the notes, tags, contexts or
 * blocking relation the server had already sent down the same wire. A row is
 * now a disclosure: one dense line, and its whole record underneath on click.
 * That is the same one-level disclosure the rest of the product uses, and it
 * makes truncation recoverable rather than final — which is the only reason a
 * single-line row is allowed here.
 *
 * ORDER IS THE SERVER'S AND IS NOT TOUCHED. `list_todos`
 * (src/memory/todos.rs:1008-1026) sorts by `sort_order` — a MANUAL position
 * someone set — then priority, then due date. Re-sorting here by priority
 * would look tidier and would silently discard the one ordering a human
 * actually chose. Priority is made visible on every row instead, which was the
 * real complaint: it was rendered for `high` and `urgent` only, so a corpus of
 * medium and low todos showed a column of blanks and no way to tell them apart.
 *
 * WHAT IS NOT ON THIS SCREEN IS STATED ON IT. `include_completed` defaults to
 * false server-side (todos.rs:1224), and the request is capped at
 * `TASK_LIMIT`. Both are reductions, and an unmarked reduction is how "nothing
 * outstanding" comes to mean "nothing in the first two hundred".
 */

const STATUS_ORDER: TodoStatus[] = ["in_progress", "blocked", "todo", "backlog"];

const STATUS_META: Record<TodoStatus, { label: string; icon: LucideIcon; iconClass: string }> = {
  in_progress: { label: "In progress", icon: CircleDot, iconClass: "text-foreground" },
  // Blocked is the one status that is itself "worth a look" rather than
  // ordinary workflow state — `--warn`, not the chrome accent.
  blocked: { label: "Blocked", icon: CircleSlash, iconClass: "text-warn" },
  todo: { label: "Todo", icon: Circle, iconClass: "text-muted-foreground" },
  backlog: { label: "Backlog", icon: CircleDashed, iconClass: "text-muted-foreground/60" },
  done: { label: "Done", icon: Circle, iconClass: "text-muted-foreground" },
  cancelled: { label: "Cancelled", icon: Circle, iconClass: "text-muted-foreground" },
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
 * One todo: a single ~28px line, with its full record one click below.
 *
 * The title truncates rather than wrapping, and the metadata is right-aligned
 * where the eye finishes the row — the same arrangement that makes a dense
 * table scannable without reading every cell. Both only work because the click
 * target exists: a truncated line with nothing behind it loses information for
 * good, which is what the two-line clamp this replaces was doing.
 */
function TaskRow({ todo, showProject }: { todo: Todo; showProject: boolean }) {
  const [open, setOpen] = useState(false);
  const meta = STATUS_META[todo.status];
  const Icon = meta.icon;
  const due = dueMeta(todo, Date.now());
  const priority = priorityLabel(todo.priority);
  const detail =
    todo.notes || todo.blocked_on || todo.tags.length > 0 || todo.contexts.length > 0;
  const captured = new Date(todo.created_at).toLocaleDateString();
  const updated = new Date(todo.updated_at).toLocaleDateString();

  return (
    <div className="border-border border-b">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`${shortId(todo)} — ${todo.content}`}
        className={cn(
          "hover:bg-accent/60 flex w-full items-center gap-2.5 px-4 py-1.5 text-left transition-colors duration-100",
          "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        )}
      >
        <Icon aria-hidden="true" className={cn("size-3.5 shrink-0", meta.iconClass)} strokeWidth={1.8} />
        <span
          className={cn("min-w-0 flex-1 text-[13px]", open ? "whitespace-normal" : "truncate")}
        >
          {todo.content}
        </span>
        <Meta className="shrink-0 flex-nowrap">
          <span className="text-muted-foreground/70 mono text-[10px]">{shortId(todo)}</span>
          {showProject && todo.project_prefix ? (
            <span className="mono text-[10px]">{todo.project_prefix}</span>
          ) : null}
          {/* Priority on EVERY row that has one. Urgent and high earn a chip
              because they are a call to act; medium and low are muted text,
              which distinguishes them from each other and from a blank
              without adding two more colours to the screen. */}
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
        <div className="space-y-1.5 px-4 pt-0.5 pb-2.5 pl-[26px]">
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
          <Meta>
            <Stat value={captured} label="captured" />
            {/* Compared as RENDERED dates, not as timestamps. The server writes
                `updated_at` milliseconds after `created_at` on every capture,
                so a raw string comparison prints the same date twice on every
                todo that has never actually been touched. */}
            {updated !== captured ? <Stat value={updated} label="updated" /> : null}
            {/* An expanded row with nothing under it is a dead end unless it
                says why. The record genuinely holds no more than the title. */}
            {!detail ? <span>no notes, tags or contexts recorded</span> : null}
          </Meta>
        </div>
      ) : null}
    </div>
  );
}

function StatusGroup({
  status,
  todos,
  showProject,
}: {
  status: TodoStatus;
  todos: Todo[];
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
        <TaskRow key={t.id} todo={t} showProject={showProject} />
      ))}
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
    queryFn: ({ signal }) => listTodos({ user_id: profile!, limit: TASK_LIMIT }, signal),
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
    // Skeleton rows carry the shape they are standing in for — one dense line
    // with a right-hand token — so the arrival of data does not restructure
    // the list under someone already reading it.
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

  if (data && data.todos.length === 0) {
    return (
      // What the destination is FOR, then what fills it. Not an apology, and
      // not "nothing outstanding" — the server was never asked about completed
      // or cancelled work, so this screen cannot claim there is none.
      <EmptyState
        size="page"
        title="Tasks lists the open work recorded in this profile's memory"
        body="Nothing is open here yet. One appears the moment a session records a todo against this profile."
        more="Tasks are picked up from what was written during a session — yours or an agent's — rather than entered on this screen. Completed and cancelled work is not listed: this screen asks the server for open items only."
      />
    );
  }

  if (!data) return null;

  const summary = summarise(data.todos, data.count, Date.now());

  return (
    <ScrollArea className="h-full">
      {/* A measured column, widened from the previous `max-w-2xl`: a row is now
          one line whose title runs to the metadata, so the extra 224px is
          fewer truncated titles rather than more whitespace. */}
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
          {/* Both reductions, on screen rather than behind the affordance: what
              is excluded changes how the count above should be read, and the
              published finding on info tips is that most people never open
              them. */}
          <span className="text-muted-foreground/70 ml-auto flex items-center gap-1.5 text-[11px]">
            completed not listed
            <InfoHint label="what this list contains" align="right">
              This screen asks the server for open work only — done and cancelled todos are never
              requested, so a count here is a count of what is outstanding and not of everything
              ever captured.
              <br />
              <br />
              Rows keep the server's order: a manual position first, then priority, then due date.
              Sorting them by priority on arrival would have discarded the one ordering a person
              actually set.
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
            showProject={summary.projects > 1}
            todos={data.todos.filter((t) => t.status === status)}
          />
        ))}
      </div>
    </ScrollArea>
  );
}
