import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { listReminders } from "./api";
import { standingReminders } from "./prospective";

/**
 * What you asked to be told about.
 *
 * WHY IT IS HERE AND NOT ON TASKS. A reminder in this product is not a todo
 * with a bell on it. `ProspectiveTask` (src/memory/types.rs:3679) is built from
 * a memory and carries no link to a `Todo` of any kind — no `todo_id`, no
 * `parent_id`, nothing the two objects share — so a reminders panel on the
 * Tasks screen would be a second object type wearing the same due dates as the
 * first, and the one honest thing that screen says about itself ("recorded work
 * only") would stop being true. The subsystem's own purpose settles it: a
 * reminder is something you asked to be told UNPROMPTED, and the briefing is
 * the only surface in the product a person arrives at without asking for
 * anything. It is the same claim as "since you left", one line further in.
 *
 * IT FOLLOWS `Learning.tsx`'s GRAMMAR EXACTLY, and reuses its Row measure so
 * the two label columns line up rather than nearly lining up: a fixed mono
 * gutter, a figure-or-phrase per item, no heading of its own, no empty state.
 * Two standing reminders read as two rows.
 *
 * NOTHING HERE IS A CONTROL. See `api.ts` — three of the eight reminder routes
 * mutate what they return, and dismissal has no inverse anywhere in the crate.
 * This reads.
 *
 * AN ABSENT SECTION MUST NOT BE A CONFIDENT ZERO. Nothing standing renders
 * nothing at all, which is right — but a read that FAILED would render the
 * same nothing and quietly report that you are owed no reminders. So the
 * failure gets one line, in the same grammar, saying only that it could not be
 * read. It is mounted only while the store is reachable, so an offline session
 * does not repeat what the standfirst already said.
 */

/** `Learning.tsx`'s Row measure, deliberately identical: these two sections sit
 *  one above the other and a label column that is nearly aligned reads as a
 *  mistake rather than as a different section. */
const HEADING =
  "mono text-muted-foreground w-[8.5rem] shrink-0 text-[10px] tracking-[0.14em] uppercase opacity-80";

export function Standing({ profile, now }: { profile: string; now: number }) {
  const { data, error } = useQuery({
    queryKey: ["reminders", profile],
    queryFn: ({ signal }) => listReminders(profile, signal),
    staleTime: 60_000,
  });

  // THE PAGE'S CLOCK, NOT ONE OF ITS OWN. `now` is the same minute-resolution
  // state the dateline is printed from, passed in rather than sampled here for
  // the reason Tasks passes its clock down to every row: a gutter reading "3
  // days late" and a marker deciding whether the row is late are ONE judgement,
  // and a second `Date.now()` in this component could straddle a due instant
  // and disagree with the dateline above it. It also means these rows re-read
  // themselves once a minute, so a reminder that falls due while the page is
  // open changes without a refetch.
  const board = data ? standingReminders(data.reminders, now) : null;

  if (error) {
    return (
      <section className="border-border flex flex-wrap items-baseline gap-x-[1.1rem] gap-y-1.5 border-b py-4">
        <span className={HEADING}>You asked for</span>
        <span role="status" className="text-muted-foreground text-[13px] leading-[1.45]">
          Your reminders could not be read — this says nothing about whether any are standing.
        </span>
      </section>
    );
  }

  if (!board) return null;

  return (
    <section className="border-border flex flex-wrap items-baseline gap-x-[1.1rem] gap-y-1.5 border-b py-4">
      <span className={HEADING}>You asked for</span>
      <ul className="m-0 min-w-0 flex-1 list-none p-0">
        {board.rows.map((row) => (
          <li
            key={row.id}
            className="border-border flex items-baseline gap-2 border-t py-1.5 text-[13px] leading-[1.45] first:border-t-0 first:pt-0"
          >
            {/* The page's own alarm mark, on the only rows that can earn it: a
                timed reminder whose instant has passed. A context reminder has
                no due instant and never takes this token, however long it has
                been standing. */}
            {row.late ? (
              <span className="text-destructive mono shrink-0 text-[10px]" aria-hidden="true">
                ▲
              </span>
            ) : null}
            <span className="min-w-0 break-words">{row.text}</span>
            <span
              className={cn(
                "mono ml-auto shrink-0 pl-3 text-[10px] tracking-[0.06em] opacity-80",
                row.late ? "text-destructive" : "text-muted-foreground",
              )}
            >
              {row.meta.join(" · ")}
            </span>
          </li>
        ))}
        {board.hidden > 0 ? (
          <li className="border-border text-muted-foreground mono border-t py-1.5 text-[10px] tracking-[0.06em] opacity-80">
            {board.hidden} more standing
          </li>
        ) : null}
      </ul>
    </section>
  );
}
