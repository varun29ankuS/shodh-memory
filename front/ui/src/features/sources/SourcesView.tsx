import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { FileInput, Terminal, type LucideIcon } from "lucide-react";
import { ApiError, NetworkError, type Reachability } from "@/lib/api";
import { useSession } from "@/stores/session";
import { longAgo, shortAgo } from "@/features/briefing/derive";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  fetchMifAdapters,
  fetchProfileStats,
  fetchSessionHistory,
  type SessionHistoryEntry,
} from "./api";
import {
  classifySession,
  formatCount,
  formatRecorded,
  formatSessionLength,
  readSessions,
  readableFormats,
} from "./derive";

/**
 * Sources — what has written into this profile, and what left a record saying so.
 *
 * THE QUESTION was "I don't see a connectors tab on the left — are we improving
 * the connectors?" The honest answer, found by reading the tree rather than
 * assuming: there is no connector subsystem at all. Nothing in `src/` is named
 * connector, ingest, watcher, scheduler or source registry, and no route polls
 * anything on a timer. But two things do write into a profile, and until this
 * screen neither was visible anywhere in the product — including the one that
 * has been running continuously for months.
 *
 * WHY THIS IS NOT A CONNECTOR LIST. A connector list implies a registry of
 * things that can be turned on, and there is nothing to turn on: no watched
 * folder, no schedule, no credential to store. What exists instead is a
 * RECORD — two writers, one of which keeps one and one of which does not. So
 * the screen is organised around what each writer left behind, and the second
 * card's whole content is the absence, stated rather than hidden. There is no
 * Connect button on this screen because there is nothing on the other side of
 * it, and a button that cannot connect anything is worse than no button.
 *
 * WHAT IT CLAIMS, AND WHAT IT REFUSES TO. Following History, the limits are
 * printed on the surface and not filed behind the info affordance, because each
 * changes how the figures beneath it must be read:
 *
 *   - Nothing here attributes an INDIVIDUAL memory. A stored memory has no
 *     origin field — `Memory` and `Experience` (src/memory/types.rs) carry no
 *     `source`, and the entity graph's `attributes.source` has no counterpart
 *     on memories. Every figure on this page is a record some writer kept about
 *     itself, never a property of the store.
 *
 *   - Every session figure is a FLOOR. A summary is written when a session
 *     ends, so one still running has already delivered memories and left no row
 *     here. Measured on the live `claude-code` profile: newest memory 10:49,
 *     newest session summary 07:37, same day.
 *
 *   - THERE IS NO "LIVE" INDICATOR, and its absence is the deliberate part. The
 *     server cannot attest that a hook is installed anywhere — a hook removed a
 *     minute ago is indistinguishable from one about to fire. Recency is the
 *     only thing the data supports, so recency is the only thing shown.
 *
 * TWO OTHER READS WERE AVAILABLE AND ARE NOT USED; `api.ts` carries the
 * measurements that ruled each out. The short version: the session store is
 * process-local and answered `{"sessions":[],"count":0}` for a profile with
 * 18,032 memories right after a restart, and `POST /api/recall/tags` truncates
 * on hash order rather than recency while serialising embeddings at ~13KB a
 * row.
 */

/* -------------------------------------------------------------------------- *
 * THE SHELL OF A SOURCE
 *
 * Both cards take the same skeleton — what it is, what its record says, what it
 * cannot do — and the point of the screen is the contrast between what fills
 * that skeleton. File import's record line is an absence, and it sits in the
 * same place a date sits on the card above it, so the gap is legible as a gap
 * rather than as an omission.
 * -------------------------------------------------------------------------- */

function Source({
  icon: Icon,
  name,
  blurb,
  record,
  cannot,
  children,
}: {
  icon: LucideIcon;
  name: string;
  /** What this is, in the words a person would use for it. Never the
   *  subsystem's name: nobody recognises "PostToolUse hook". */
  blurb: string;
  /** What its own record says — a date and figures, or the statement that
   *  there is none. */
  record: React.ReactNode;
  /** The limit that changes how the record above should be read. */
  cannot: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <section className="border-border border-b px-4 py-3.5">
      <div className="flex items-baseline gap-2">
        <Icon
          aria-hidden="true"
          className="text-muted-foreground size-3.5 shrink-0 translate-y-[2px]"
          strokeWidth={1.8}
        />
        <h2 className="text-[13px] font-medium tracking-tight">{name}</h2>
      </div>
      <p className="text-muted-foreground mt-1 max-w-prose pl-[22px] text-[12px] leading-relaxed">
        {blurb}
      </p>

      <div className="mt-2.5 pl-[22px]">{record}</div>

      {children ? <div className="mt-3 pl-[22px]">{children}</div> : null}

      {/* The limit is drawn at the weight of a caveat and given the border-left
          the rest of the product uses for one, so it reads as attached to the
          figures above rather than as more content. */}
      <p className="border-border text-muted-foreground/80 mt-3 ml-[22px] max-w-prose border-l pl-2.5 text-[11px] leading-relaxed">
        {cannot}
      </p>
    </section>
  );
}

/* -------------------------------------------------------------------------- *
 * THE RECENT SESSIONS
 * -------------------------------------------------------------------------- */

/**
 * The last few completed sessions, as rows.
 *
 * SIX, and the number is doing work. This list is texture — evidence that the
 * figures above came from real events at real times, which a bare count cannot
 * give — and a reader who wants the whole trail has History. Rendering all
 * fifty would turn the card into a log and bury the second source under it.
 *
 * A session that reported no duration and no count still gets its row, showing
 * only its date. Dropping it would make the record look denser than it is: 135
 * of 230 entries on the live profile are exactly that shape.
 */
function RecentSessions({ entries, now }: { entries: readonly SessionHistoryEntry[]; now: number }) {
  const rows = entries.slice(0, 6);
  if (rows.length === 0) return null;

  return (
    <ul className="space-y-1">
      {rows.map((entry, index) => {
        const writer = classifySession(entry);
        const length = formatSessionLength(entry.duration_secs);
        return (
          <li
            // `created_at` is not unique — two summaries can share an instant —
            // and `session_id` is null on the majority of entries. Position in
            // a list that is never re-sorted is unique by construction.
            key={index}
            className="flex items-baseline gap-2.5"
          >
            {/* `shortAgo`, not `longAgo`, and the column width is why: this is
                the right-hand-gutter position that form was written for
                (features/briefing/derive.ts). A profile last worked on in April
                gives "137 days ago", which wraps a fixed gutter on every row of
                a stale profile — the one case where these rows matter most. The
                exact instant stays on `title`, and the sentence above the list,
                which has a whole line, keeps the spelled-out form. */}
            <time
              dateTime={entry.created_at}
              title={entry.created_at}
              className="text-muted-foreground mono w-[42px] shrink-0 text-[11px]"
            >
              {shortAgo(entry.created_at, now) ?? "—"}
            </time>
            <Meta className="min-w-0 flex-1">
              {length ? <Stat value={length} label="session" /> : null}
              {entry.memories_created !== null ? (
                <Stat value={formatCount(entry.memories_created)} label="memories" />
              ) : null}
              {/* Said only when it is not the ordinary case. A badge on every
                  row would be a column that never varies. */}
              {writer === "consolidation" ? (
                <Badge variant="secondary">this server, consolidating</Badge>
              ) : writer === "unmarked" ? (
                <Badge>writer not recorded</Badge>
              ) : null}
              {length === null && entry.memories_created === null ? (
                <span className="text-muted-foreground/70">recorded, nothing measured</span>
              ) : null}
            </Meta>
          </li>
        );
      })}
    </ul>
  );
}

/* -------------------------------------------------------------------------- *
 * THE SCREEN
 * -------------------------------------------------------------------------- */

/** One phrase for a failed read, in the two words the shell distinguishes
 *  everywhere: a status means the key or the route, a network failure means the
 *  server. Collapsing them tells people to check their network when the backend
 *  is simply not running. */
function readFailure(error: unknown): string {
  if (error instanceof ApiError) return `the server answered ${error.status}`;
  if (error instanceof NetworkError) return "the server did not respond";
  return "the response could not be read";
}

export function SourcesView({ reach }: { reach: Reachability }) {
  const profile = useSession((s) => s.profile);
  // Frozen at mount for the reason History freezes its own: every "how long
  // ago" on this screen is measured from one instant, so two rows cannot be
  // dated against clocks that drifted apart while the page was open.
  const [now] = useState(() => Date.now());
  const online = reach.state === "online";
  const enabled = online && profile !== null;

  const sessions = useQuery({
    queryKey: ["sources", "sessions", profile],
    queryFn: ({ signal }) => fetchSessionHistory(profile!, signal),
    enabled,
  });

  const stats = useQuery({
    queryKey: ["sources", "stats", profile],
    queryFn: ({ signal }) => fetchProfileStats(profile!, signal),
    enabled,
  });

  // Not keyed on the profile: the adapter registry is a property of the build
  // (`AdapterRegistry::new()`), identical for every profile on the server, so
  // switching profiles must not refetch it or invalidate it.
  const adapters = useQuery({
    queryKey: ["sources", "mif-adapters"],
    queryFn: ({ signal }) => fetchMifAdapters(signal),
    enabled: online,
  });

  const record = useMemo(
    () =>
      sessions.data === undefined
        ? null
        : readSessions(sessions.data.sessions, sessions.data.total),
    [sessions.data],
  );

  const formats = useMemo(
    () => (adapters.data === undefined ? [] : readableFormats(adapters.data.adapters)),
    [adapters.data],
  );

  if (!online) {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="What has written into a profile is read from the memory server."
        more="This screen reads the server's own session record and its import adapter registry. Neither is cached in the browser, so when the server is not running there is nothing to show from a previous visit."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile selected"
        body="Every record on this screen belongs to one profile."
        more="Profiles are separate stores. A session recorded against one says nothing about any other, so this screen cannot be answered until one is chosen."
      />
    );
  }

  if (sessions.error) {
    return (
      <EmptyState
        size="page"
        title="Could not read the session record"
        body={`Reading this profile's completed sessions failed — ${readFailure(sessions.error)}.`}
      />
    );
  }

  if (record === null) {
    return (
      <div className="mx-auto w-full max-w-3xl">
        <div className="border-border border-b px-4 py-2.5">
          <Skeleton className="h-3 w-64" />
        </div>
        {[0, 1].map((i) => (
          <div key={i} className="border-border space-y-2 border-b px-4 py-3.5">
            <Skeleton className="h-3.5 w-40" />
            <Skeleton className="h-3 w-full max-w-md" />
            <Skeleton className="h-3 w-56" />
          </div>
        ))}
      </div>
    );
  }

  const recordedTime = formatRecorded(record.recordedSecs);
  // A failed stats read leaves this null and every element that depends on it
  // is omitted, rather than the screen reporting a profile of zero memories.
  // The session record is the substance here and stands without it; an error
  // banner for a supporting figure would put the failure above the content.
  const held = stats.data?.total_memories ?? null;

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-3xl pb-16">
        {/* WHAT IS IN HERE. The profile's own size leads, because it is the
            quantity every figure below has to be read against — and because on
            a profile filled by something that left no record, it is the only
            true number on the screen. */}
        <header className="border-border border-b px-4 py-2.5">
          <Meta className="text-[12px]">
            {held !== null ? <Stat value={formatCount(held)} label="memories held" /> : null}
            <Stat
              value={formatCount(record.total)}
              label={record.total === 1 ? "recorded session" : "recorded sessions"}
            />
            {/* Dropped entirely rather than shown as a dash when the registry
                could not be read. A "—" beside a label is a fact-shaped hole,
                and this strip is read as a row of facts. */}
            {formats.length > 0 ? <Stat value={formats.length} label="import formats" /> : null}
          </Meta>

          <dl className="border-border mt-2 grid grid-cols-[86px_1fr] gap-x-2 gap-y-1 border-l-2 pl-2.5 text-[11px] leading-relaxed">
            <dt className="text-muted-foreground/70">Proves</dt>
            <dd>
              that sessions ran against this profile, when the last one finished, and how much each
              reported writing.
            </dd>
            <dt className="text-muted-foreground/70">Cannot prove</dt>
            <dd className="text-muted-foreground">
              where any individual memory came from. A stored memory has no field recording how it
              arrived, so nothing here accounts for one.
            </dd>
            <dt className="text-muted-foreground/70">Does not cover</dt>
            <dd className="text-muted-foreground">
              anything written straight through the API, or read in from a file — neither leaves a
              trace that can be read back afterwards.
              <InfoHint label="what a source can and cannot leave behind" className="ml-1.5 translate-y-[2px]">
                Entities in the knowledge graph carry an <code>attributes.source</code>. Memories
                carry no equivalent: the stored record is content, tags, a type and a time, and none
                of those is written by the thing that delivered it.
                <br />
                <br />
                So a source becomes visible here only by keeping its own record. The Claude Code
                hook does, by writing a summary memory when a session ends. File import does not —
                it writes imported memories under their original timestamps and marks them with
                nothing at all.
                <br />
                <br />
                The server does keep an internal audit trail, and an import writes one line to it.
                No HTTP route serves it, and it is trimmed on a retention timer, so nothing on this
                screen reflects it.
              </InfoHint>
            </dd>
          </dl>
        </header>

        <Source
          icon={Terminal}
          name="Claude Code sessions"
          blurb="Claude Code writes into this profile while you work. When a session ends, it records what that session did."
          record={
            record.total === 0 ? (
              <p className="text-muted-foreground text-[12px] leading-relaxed">
                No completed session has been recorded against this profile.
              </p>
            ) : (
              <div className="space-y-1">
                <p className="text-[12px]">
                  Last completed session{" "}
                  <span className="font-medium">
                    {record.latest === null ? "at an unreadable time" : longAgo(record.latest, now)}
                  </span>
                  {record.earliest !== null ? (
                    <span className="text-muted-foreground">
                      , first {longAgo(record.earliest, now)}
                    </span>
                  ) : null}
                  .
                </p>
                <Meta>
                  {/* Sums are over the page, and the page size is printed with
                      them. Saying "10,758 memories" without "across the last
                      50" would state a total this read cannot see. */}
                  {record.reported > 0 ? (
                    <Stat
                      value={formatCount(record.memoriesCreated)}
                      label={
                        record.complete
                          ? `memories, across the ${record.reported} of ${record.total} sessions that reported a count`
                          : `memories, across the ${record.reported} of the last ${record.page} sessions that reported a count`
                      }
                    />
                  ) : null}
                  {recordedTime ? <Stat value={recordedTime} label="of session time" /> : null}
                  {record.byWriter.consolidation > 0 ? (
                    <Stat
                      value={record.byWriter.consolidation}
                      label="written by this server, not a session"
                    />
                  ) : null}
                </Meta>
              </div>
            )
          }
          cannot={
            <>
              A session is recorded when it <em>ends</em>. One still running, or killed before it
              finished, has already written its memories and left no row here — so the count and the
              time above are floors, not totals. Nothing on this page can say whether the hook is
              still installed; only when it last delivered.
            </>
          }
        >
          {sessions.data ? <RecentSessions entries={sessions.data.sessions} now={now} /> : null}
        </Source>

        <Source
          icon={FileInput}
          name="File import"
          blurb="Memories, tasks and a knowledge graph read in bulk from a file — the format is detected rather than declared."
          record={
            adapters.error ? (
              <p className="text-muted-foreground text-[12px]">
                The formats could not be listed — {readFailure(adapters.error)}.
              </p>
            ) : (
              <div className="space-y-2">
                <p className="text-muted-foreground text-[12px] leading-relaxed">
                  No import has a record. Not "none has run" — nothing anywhere would say either
                  way.
                </p>
                {formats.length > 0 ? (
                  <ul className="flex flex-wrap gap-1">
                    {formats.map((adapter) => (
                      <li key={adapter.format}>
                        <Badge variant="secondary" title={`format id: ${adapter.format}`}>
                          {adapter.name}
                        </Badge>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            )
          }
          cannot={
            <>
              An import preserves each memory's <em>original</em> timestamp and marks it with
              nothing, so a past import cannot be found afterwards — not by a tag, and not even as a
              cluster in time. This is a capability the server has, listed with the formats it
              genuinely reads; it is not a feed, and this screen would look identical the day after
              a million rows arrived through it.
            </>
          }
        />

        {/* THE CLOSING RECONCILIATION, and it is deliberately not arithmetic.
            Subtracting what the sessions reported from what the profile holds
            would set a flow against a stock — memories are deleted, compressed
            and consolidated, and the summaries are themselves memories — so the
            two numbers are stated side by side and never differenced. */}
        {held !== null ? (
          <p className="text-muted-foreground mx-4 mt-3 max-w-prose text-[11px] leading-relaxed">
            This profile holds {formatCount(held)} {held === 1 ? "memory" : "memories"}. The figures
            above are records that writers kept about themselves, not a property of those memories —
            so they neither add up to that number nor are meant to.
          </p>
        ) : null}
      </div>
    </ScrollArea>
  );
}
