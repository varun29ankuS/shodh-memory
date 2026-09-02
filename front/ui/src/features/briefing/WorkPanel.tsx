import { useQuery } from "@tanstack/react-query";

import { listTodos, type Reachability, type Todo } from "@/lib/api";
import { useSession } from "@/stores/session";

/**
 * "What you're working on" — open work, blocked first.
 *
 * A briefing answers "what should I look at" before anything is asked, and the
 * honest answer is rarely a map. Blocked work leads because that is the only
 * category where knowing about it changes what you do next: an in-progress task
 * is already being handled by the person reading this, and a backlog item is by
 * definition not urgent. Something waiting on another thing is the item that
 * goes stale silently.
 *
 * THIS PANEL FETCHES. The query is deliberately byte-identical to the one in
 * TasksView -- same key, same body, same limit -- so the two share a single
 * cache entry rather than two entries that disagree. A same-key-different-body
 * pair would be worse than a second fetch: whichever view rendered second would
 * silently show the other one's data. Reading the briefing first now makes the
 * tasks destination instant, which is the same bargain the maps strike with geo.
 */

const OPEN: Todo["status"][] = ["backlog", "todo", "in_progress", "blocked"];

/** Blocked first, then work already underway, then by priority. Ties break on
 *  seq_num so the order does not depend on what the server happened to send. */
const RANK: Record<string, number> = { blocked: 0, in_progress: 1, todo: 2, backlog: 3 };
const PRIORITY: Record<string, number> = { urgent: 0, high: 1, medium: 2, low: 3, none: 4 };

/** Enough to orient, few enough that the panel stays a summary. */
const SHOWN = 5;

export function isBlocked(t: Todo): boolean {
  // Both, because they are different facts. `status` is what someone set; a
  // non-null `blocked_on` is a dependency the store itself is tracking, and a
  // todo can carry one while its status still reads "todo".
  return t.status === "blocked" || t.blocked_on !== null;
}

export function isOpen(t: Todo): boolean {
  return OPEN.includes(t.status);
}

/**
 * The briefing's order: blocked, then underway, then by priority.
 *
 * Exported and pure because this ordering IS the panel's claim -- "blocked
 * first" is the only reason a summary of five rows beats opening /tasks -- and
 * a claim rendered inline is a claim nothing checks. Ties break on seq_num so
 * two runs over the same data cannot disagree, the same discipline the recall
 * sorts follow.
 */
export function briefingOrder(todos: Todo[]): Todo[] {
  return [...todos].sort(
    (a, b) =>
      (isBlocked(a) ? 0 : 1) - (isBlocked(b) ? 0 : 1) ||
      (RANK[a.status] ?? 9) - (RANK[b.status] ?? 9) ||
      (PRIORITY[a.priority] ?? 9) - (PRIORITY[b.priority] ?? 9) ||
      a.seq_num - b.seq_num,
  );
}

function ref(t: Todo): string {
  return t.project_prefix ? `${t.project_prefix}-${t.seq_num}` : `#${t.seq_num}`;
}

export function WorkPanel({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  const enabled = reach.state === "online" && profile !== null;

  const { data, error, isFetching } = useQuery({
    queryKey: ["todos", profile],
    queryFn: ({ signal }) => listTodos({ user_id: profile!, limit: 200 }, signal),
    enabled,
  });

  // No panel at all until there is an answer. A heading over a spinner claims
  // there is work here before anyone has looked.
  if (!data && isFetching) return null;

  // A FAILED FETCH IS NOT AN EMPTY BACKLOG. Returning null here would render
  // exactly like "nothing open" -- the quiet form of the same bug the corpus
  // branch in BriefingView fixes loudly, and the more dangerous form: a false
  // claim can be argued with, an absence cannot even be noticed. So the panel
  // stays, and says which of the two it is.
  if (error && !data) {
    return (
      <section>
        <h2 className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
          What you're working on
        </h2>
        <p className="mt-3 text-sm text-muted-foreground">
          Open work could not be read, so none is being reported — this is not the same as having
          none.
        </p>
      </section>
    );
  }

  const all = data?.todos ?? [];
  const open = all.filter(isOpen);
  const blocked = open.filter(isBlocked);

  if (open.length === 0) return null;

  const shown = briefingOrder(open).slice(0, SHOWN);

  return (
    <section>
      <h2 className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
        What you're working on
      </h2>

      <p className="mt-3 text-sm">
        <span className="font-semibold text-foreground">{open.length}</span> open
        {blocked.length > 0 && (
          <>
            , <span className="font-semibold text-[var(--node-anomalous)]">{blocked.length}</span>{" "}
            blocked
          </>
        )}
      </p>

      <ul className="mt-3 space-y-1.5">
        {shown.map((t) => (
          <li key={t.id} className="flex gap-2 text-[13px] leading-snug">
            <span
              className="mt-[7px] size-1.5 shrink-0 rounded-full"
              style={{
                background: isBlocked(t)
                  ? "var(--node-anomalous)"
                  : t.status === "in_progress"
                    ? "var(--live)"
                    : "var(--muted-foreground)",
              }}
              // The dot is the status; a reader hovering asks it directly
              // rather than decoding three hues from a legend that is not here.
              title={isBlocked(t) ? `blocked${t.blocked_on ? ` on ${t.blocked_on}` : ""}` : t.status}
            />
            <span className="mono shrink-0 text-xs text-muted-foreground">{ref(t)}</span>
            <span className="truncate">{t.content}</span>
          </li>
        ))}
      </ul>

      {open.length > shown.length && (
        <a href="#/tasks" className="mt-3 inline-block font-mono text-xs text-primary hover:underline">
          all {open.length} open →
        </a>
      )}
    </section>
  );
}
