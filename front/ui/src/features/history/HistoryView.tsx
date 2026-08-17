import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Bot, Cog, CircleHelp, Download, User, type LucideIcon } from "lucide-react";
import { ApiError, NetworkError } from "@/lib/api";
import { fetchAuditFile, fetchAuditTrail } from "@/lib/seat/client";
import type {
  AuditRow,
  AuditSource,
  LedgerActorView,
  SeatReachability,
} from "@/lib/seat/types";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import {
  AUDIT_WINDOWS,
  type AuditFormat,
  type AuditWindowId,
  actorLabel,
  auditExportPath,
  clock,
  conversationsIn,
  dayLabel,
  exportFilename,
  formatDuration,
  groupByDay,
  kindLabel,
  matchesView,
  outcomeOf,
  parseAuditJsonl,
  summarise,
  toggle,
  toolCallDetail,
  viewDimensionLabel,
  viewOutcomeDetail,
  toolCensus,
} from "./derive";

/**
 * History — who did what, with which tool, when, across every conversation.
 *
 * THE QUESTION THIS SCREEN EXISTS FOR was asked in one line: "auditability and
 * reproducibility is king, what about which tool was used when?" The seat has
 * recorded the answer for as long as it has recorded anything — tool calls with
 * durations, every change to memory with an actor, every retrieval with its
 * scores — and until this screen there was nowhere to read it. What existed was
 * a per-conversation transcript, which answers "what happened in this chat" and
 * cannot answer "when was this tool last used" at all.
 *
 * IT RENDERS THE EXPORT FILE ITSELF, and that is the load-bearing decision.
 * `GET /v1/audit/export` is the only read that returns the whole merged trail;
 * this screen parses that body and the download saves the same bytes untouched.
 * So what a reviewer receives cannot differ from what the person who sent it
 * was looking at — the class of failure where a screen quietly shows more, or
 * less, than the artefact is structurally impossible here rather than tested
 * for. features/history/derive.ts carries the full reasoning.
 *
 * WHAT IT CLAIMS, AND WHAT IT REFUSES TO. Four limits are printed on the
 * surface rather than filed behind the info affordance, because each one
 * changes how the rows above it should be read and the info panel is documented
 * to be a place most people never open (components/ui/info-hint.tsx):
 *
 *   - Tool RESULTS are never persisted seat-side and there is no pre-query
 *     state snapshot, so this proves what happened and cannot re-run it. A
 *     surface that implied replay would be lying about the strongest claim on
 *     it.
 *   - The Rust backend keeps its OWN audit trail — 23 `log_event` sites into
 *     RocksDB's CF_AUDIT — which is not served over HTTP and rotates on a
 *     retention timer. This is the seat's trail, not the system's.
 *   - Events are persisted at TURN granularity (seat/src/store.ts), so a turn
 *     that crashed mid-flight left no rows at all. Absence here is not evidence
 *     that nothing happened.
 *   - Ledger entries written before the seat recorded an actor read "unknown"
 *     and are deliberately not backfilled (seat/src/ledger.ts `entryActor`).
 *     They render as unknown. Guessing one would be fabricating audit data,
 *     which is the exact thing this screen exists to make impossible.
 *
 * The one gap that IS visible in the data gets the loudest treatment on the
 * screen: a tool call with no end event keeps null duration and null outcome,
 * and is drawn in `--warn` reading "never returned". A reviewer scanning this
 * list is looking for precisely that row.
 *
 * QUERY FILTERS AND VIEW FILTERS ARE SEPARATED IN THE LAYOUT, not merely in the
 * code. The window and the conversation are sent to the seat and are therefore
 * carried by the downloaded file; the actor and type toggles narrow rows
 * already in hand and are NOT. Someone who believed the download matched the
 * narrowed list would hand over a wider file than they meant to, so the two
 * rows are labelled for what they are and the export states its own coverage.
 */

/* -------------------------------------------------------------------------- *
 * TOKENS
 *
 * ACTOR IS CARRIED BY ICON AND WORD, NEVER BY COLOUR. Four actors would need
 * four categorical hues on a surface whose design rule reserves saturated
 * colour for data and allows exactly one accent (index.css) — and colouring
 * `unknown` at all would give a recorded GAP the same standing as a recorded
 * fact. Colour on this screen means one thing only: something went wrong.
 * `--destructive` is a tool that returned an error, `--warn` is a tool that
 * never returned. Everything else is ink.
 * -------------------------------------------------------------------------- */

const ACTOR_META: Record<LedgerActorView, { icon: LucideIcon; hint: string }> = {
  user: { icon: User, hint: "A person, through the seat's HTTP surface" },
  agent: { icon: Bot, hint: "The model — it emitted this tool call or write" },
  system: { icon: Cog, hint: "An automatic loop, with no decision by either party" },
  unknown: { icon: CircleHelp, hint: "Written before the seat recorded actors; not guessed at" },
};

const ACTOR_ORDER: LedgerActorView[] = ["agent", "system", "user", "unknown"];

const SOURCE_META: Record<AuditSource, { label: string; hint: string }> = {
  tool_call: { label: "Tool calls", hint: "A tool the model invoked, with how long it took" },
  ledger: { label: "Memory changes", hint: "Something written to or adjusted in memory" },
  retrieval: { label: "Retrievals", hint: "What was searched for, and what came back" },
  view: { label: "View changes", hint: "Where the model asked to take you and why — and what your workbench did about it" },
};

const SOURCE_ORDER: AuditSource[] = ["tool_call", "ledger", "retrieval", "view"];

/** A labelled line inside an opened row. Same shape as the Tasks detail: these
 *  are read, not scanned, so they are lines rather than more chips. */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 text-[12px] leading-relaxed">
      <span className="text-muted-foreground/70 w-[88px] shrink-0 text-[11px]">{label}</span>
      <span className="mono min-w-0 flex-1 text-[11px] break-all">{children}</span>
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * THE CENSUS
 * -------------------------------------------------------------------------- */

/**
 * Which tool was used, how often, and how long it took.
 *
 * A TABLE WITH A PROPORTIONAL TRACK BEHIND THE COUNT, not a chart with a
 * tooltip. Every value it encodes — calls, median, slowest, failures — is
 * printed on its own row, so the bar adds ranking at a glance and gates nothing;
 * there is no figure here reachable only by pointing at something. One neutral
 * fill for every bar, because the length already carries the magnitude and
 * darkening the bigger ones would spend the only free channel restating it.
 *
 * The track is drawn only when there are at least two tools. A single full-width
 * bar encodes nothing at all — it is a one-bar bar chart, and the number beside
 * it is the whole content.
 */
function ToolCensus({ rows }: { rows: readonly AuditRow[] }) {
  const stats = useMemo(() => toolCensus(rows), [rows]);
  if (stats.length === 0) return null;

  const busiest = stats[0].calls;
  const withTrack = stats.length > 1;

  return (
    <section className="border-border border-b px-4 py-3">
      <div className="mb-2 flex items-baseline gap-2">
        <h2 className="text-[12px] font-medium tracking-tight">Tools used in this window</h2>
        <Meta>
          <Stat value={stats.length} label={stats.length === 1 ? "tool" : "tools"} />
        </Meta>
      </div>

      <ul className="space-y-1">
        {stats.map((tool) => (
          <li key={tool.name} className="flex items-center gap-3">
            <span className="mono w-44 shrink-0 truncate text-[12px]" title={tool.name}>
              {tool.name}
            </span>
            {withTrack ? (
              <span aria-hidden="true" className="bg-muted h-1.5 w-28 shrink-0 overflow-hidden rounded-full">
                <span
                  className="bg-foreground/25 block h-full rounded-full"
                  style={{ width: `${(tool.calls / busiest) * 100}%` }}
                />
              </span>
            ) : null}
            <Meta className="min-w-0 flex-1 flex-nowrap">
              <Stat value={tool.calls} label={tool.calls === 1 ? "call" : "calls"} />
              {tool.p50 !== null ? <Stat value={formatDuration(tool.p50)} label="median" /> : null}
              {tool.max !== null && tool.max !== tool.p50 ? (
                <Stat value={formatDuration(tool.max)} label="slowest" />
              ) : null}
              {tool.failed > 0 ? (
                <span className="text-destructive">{tool.failed} failed</span>
              ) : null}
              {tool.unterminated > 0 ? (
                <span className="text-warn">
                  {tool.unterminated} never returned
                </span>
              ) : null}
            </Meta>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* -------------------------------------------------------------------------- *
 * THE TRAIL
 * -------------------------------------------------------------------------- */

function TrailRow({ row, showConversation }: { row: AuditRow; showConversation: boolean }) {
  const [open, setOpen] = useState(false);
  const actor = ACTOR_META[row.actor];
  const ActorIcon = actor.icon;
  const outcome = outcomeOf(row);
  const detail = toolCallDetail(row);
  const viewOutcome = viewOutcomeDetail(row);
  const duration = formatDuration(detail?.duration_ms ?? null);

  // Pretty-printed for reading; the raw string is kept verbatim if it cannot be
  // parsed, because a `detail` this build cannot read is itself worth seeing.
  const payload = useMemo(() => {
    try {
      return JSON.stringify(JSON.parse(row.detail), null, 2);
    } catch {
      return row.detail;
    }
  }, [row.detail]);

  return (
    <div className="border-border/60 border-b last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`${clock(row.ts)} — ${actorLabel(row.actor)} — ${kindLabel(row)}`}
        className={cn(
          "hover:bg-accent/60 flex w-full items-center gap-2.5 px-4 py-1.5 text-left",
          "transition-colors duration-100",
          "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        )}
      >
        <time
          dateTime={row.ts}
          className="text-muted-foreground/70 mono w-[62px] shrink-0 text-[10px]"
        >
          {clock(row.ts)}
        </time>
        <ActorIcon
          aria-hidden="true"
          className={cn(
            "size-3.5 shrink-0",
            // The gap is drawn at the weight of a gap. It is legible, and it
            // does not compete with a row that records a real actor.
            row.actor === "unknown" ? "text-muted-foreground/40" : "text-muted-foreground",
          )}
          strokeWidth={1.8}
        />
        <span className="text-muted-foreground w-[68px] shrink-0 text-[11px]">
          {actorLabel(row.actor)}
        </span>
        <span
          className={cn(
            "min-w-0 flex-1 truncate text-[13px]",
            // The tool's own name is the answer to "which tool"; ledger and
            // retrieval labels are prose about the seat's own housekeeping.
            row.source === "tool_call" ? "mono text-[12px]" : "text-muted-foreground",
          )}
        >
          {kindLabel(row)}
          {/* WHICH AXIS, beside what happened to it. One direct_view produces
              several outcome rows in the same second and they differ only in
              this — a list showing four consecutive "Held it as an offer" with
              nothing to tell them apart reads as a repeated row rather than as
              four facts about four parts of the view. */}
          {viewOutcome ? (
            <span className="text-muted-foreground/60"> · {viewDimensionLabel(viewOutcome.dimension)}</span>
          ) : null}
        </span>
        <Meta className="shrink-0 flex-nowrap">
          {outcome === "unterminated" ? (
            <Badge variant="warn">never returned</Badge>
          ) : outcome === "error" ? (
            <Badge variant="destructive">error</Badge>
          ) : null}
          {duration ? <span className="mono text-[10px]">{duration}</span> : null}
          {showConversation ? (
            <span className="text-muted-foreground/70 mono text-[10px]" title={row.conversation_id}>
              {row.conversation_id.slice(0, 8)}
            </span>
          ) : null}
          <span className="text-muted-foreground/70 text-[11px]">turn {row.turn}</span>
        </Meta>
      </button>

      {open ? (
        <div className="space-y-2 px-4 pt-1 pb-3 pl-[76px]">
          <Field label="Recorded">{row.ts}</Field>
          <Field label="Reference">{row.ref}</Field>
          <Field label="Conversation">{row.conversation_id}</Field>
          {/* The namespace the operation RAN AGAINST, which for harness-scope
              ledger entries is the derived `<user>.seat-harness` rather than the
              person. It is shown rather than normalised — the two are genuinely
              different stores. */}
          <Field label="Profile">{row.user_id}</Field>
          {detail && detail.ended_at !== null ? (
            <Field label="Returned">{detail.ended_at}</Field>
          ) : null}
          {outcome === "unterminated" ? (
            <p className="border-warn/40 text-muted-foreground border-l pl-2.5 text-[11px] leading-relaxed">
              No end was ever recorded for this call. The turn it belonged to stopped before the
              tool returned — an abort, a crash, or a killed process. Its duration and its outcome
              are not known and are not guessed at.
            </p>
          ) : null}
          <pre className="bg-muted border-border mono max-h-64 overflow-auto rounded-md border p-2 text-[11px] leading-relaxed">
            {payload}
          </pre>
        </div>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * CONTROLS
 * -------------------------------------------------------------------------- */

/** A filter toggle. Pressed state is `aria-pressed` and a real surface change,
 *  never colour alone — a chip that reads as "on" only by hue is unusable to a
 *  reader who cannot separate the two. */
function Toggle({
  on,
  count,
  label,
  hint,
  onClick,
}: {
  on: boolean;
  count: number;
  label: string;
  hint: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      title={hint}
      className={cn(
        "flex h-6 items-center gap-1.5 rounded-md border px-2 text-[11px]",
        "transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
        on
          ? "border-primary/40 bg-primary/10 text-primary"
          : "border-border text-muted-foreground hover:bg-accent hover:text-foreground",
        // A zero-count option is kept, disabled: removing it would leave a
        // reader unable to tell "none by this actor" from "never offered".
        count === 0 && "opacity-45",
      )}
      disabled={count === 0}
    >
      <span>{label}</span>
      <span className="mono text-[10px] opacity-70">{count}</span>
    </button>
  );
}

/**
 * The download.
 *
 * Fetched and saved as a Blob rather than opened as a link, because in the
 * shipped product a plain navigation would not produce a file: the seat sets
 * `Content-Disposition: attachment; filename=…`, and the shodh-front proxy
 * forwards only `content-type` and `cache-control` (front/src/main.rs
 * `forward`), so the attachment header — and the name on it — never reach the
 * browser. The body is saved byte-for-byte; only the filename is rebuilt.
 */
function ExportControls({
  path,
  filenameFor,
  disabled,
}: {
  path: (format: AuditFormat) => string;
  filenameFor: (format: AuditFormat) => string;
  disabled: boolean;
}) {
  const [busy, setBusy] = useState<AuditFormat | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const save = async (format: AuditFormat) => {
    setBusy(format);
    setFailure(null);
    try {
      const { blob, filename } = await fetchAuditFile(path(format), filenameFor(format));
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      // Released on the next task rather than inline: the click queues the
      // download asynchronously, and revoking in the same tick has been
      // observed to cancel it before the browser has read the blob.
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (error) {
      setFailure(
        error instanceof ApiError
          ? `the seat answered ${error.status}.`
          : error instanceof NetworkError
            ? "the seat did not respond."
            : "the file could not be written.",
      );
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col items-end gap-0.5">
      <div className="flex items-center gap-1.5">
        {(["jsonl", "csv"] as const).map((format) => (
          <Button
            key={format}
            size="sm"
            variant={format === "jsonl" ? "default" : "outline"}
            disabled={disabled || busy !== null}
            onClick={() => void save(format)}
            aria-label={`Download this window as ${format.toUpperCase()}`}
          >
            <Download aria-hidden="true" />
            {busy === format ? "Saving…" : format.toUpperCase()}
          </Button>
        ))}
      </div>
      {failure ? (
        <p className="text-destructive text-[11px]">Not saved — {failure}</p>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- *
 * THE SCREEN
 * -------------------------------------------------------------------------- */

const NO_ROWS: readonly AuditRow[] = [];

export function HistoryView({ seat }: { seat: SeatReachability }) {
  // FROZEN AT MOUNT, AND DELIBERATELY. Every window's lower bound is derived
  // from this instant, and it is the same instant the export URL is built from,
  // so the file cannot cover a window that moved out from under the rows the
  // reader was looking at when they clicked. A `Date.now()` inside the query
  // key would also refetch the whole trail on every render.
  const [now] = useState(() => Date.now());
  const [windowId, setWindowId] = useState<AuditWindowId>("all");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [actors, setActors] = useState<ReadonlySet<LedgerActorView>>(new Set());
  const [sources, setSources] = useState<ReadonlySet<AuditSource>>(new Set());

  const query = { window: windowId, conversationId };
  const path = auditExportPath(query, "jsonl", now);

  const { data, error, isFetching } = useQuery({
    queryKey: ["audit-trail", path],
    queryFn: ({ signal }) => fetchAuditTrail(path, signal),
    enabled: seat.state === "online",
  });

  const parsed = useMemo(() => (data === undefined ? null : parseAuditJsonl(data)), [data]);
  // One shared empty array, not a fresh literal: every memo below takes `rows`
  // as a dependency, and a new `[]` per render would re-derive the whole
  // summary on every keystroke elsewhere in the shell while the trail loads.
  const rows = parsed?.rows ?? NO_ROWS;
  const summary = useMemo(() => summarise(rows), [rows]);
  const conversations = useMemo(() => conversationsIn(rows), [rows]);
  const shown = useMemo(
    () => rows.filter((row) => matchesView(row, { actors, sources })),
    [rows, actors, sources],
  );
  const days = useMemo(() => groupByDay(shown), [shown]);
  const narrowed = shown.length !== rows.length;
  /**
   * A row's position in the whole trail, used as its React key.
   *
   * NOT `(ts, source, ref)`. That triple is the export's SORT key and is total
   * as an ordering, which is a weaker property than being unique — two rows can
   * compare equal on all three (a tool call id is only documented unique within
   * a conversation, and two proactive passes in one turn share a ref). Duplicate
   * React keys make two rows share the open/closed state of one. Position is
   * unique by construction, and taken from the unfiltered trail so a row does
   * not change identity when the Who/What toggles move.
   */
  const position = useMemo(() => new Map(rows.map((row, index) => [row, index])), [rows]);

  if (seat.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="The record of what ran, and when, lives with the conversation seat."
        more="This screen reads the seat's audit export. The seat is a local process started by hand; when it is not running there is nothing to read, and nothing is cached from a previous session."
      />
    );
  }

  if (error) {
    const detail =
      error instanceof ApiError
        ? `The seat answered ${error.status}.`
        : error instanceof NetworkError
          ? "The seat stopped responding mid-request."
          : "Something went wrong reading the trail.";
    return <EmptyState size="page" title="Could not read the history" body={detail} />;
  }

  if (isFetching && parsed === null) {
    return (
      <div className="mx-auto h-full w-full max-w-5xl">
        <div className="border-border border-b px-4 py-2.5">
          <Skeleton className="h-3 w-56" />
        </div>
        {["64%", "48%", "71%", "39%", "56%", "67%", "44%", "59%"].map((width, i) => (
          <div key={i} className="border-border/60 flex items-center gap-2.5 border-b px-4 py-1.5">
            <Skeleton className="h-2.5 w-[62px] shrink-0" />
            <Skeleton className="size-3.5 shrink-0 rounded-full" />
            <Skeleton className="h-[13px] flex-1" style={{ maxWidth: width }} />
          </div>
        ))}
      </div>
    );
  }

  if (parsed === null) return null;

  const windowLabel = AUDIT_WINDOWS.find((w) => w.id === windowId)?.label ?? "";

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-5xl pb-16">
        {/* WHAT IS IN HERE — the figures first, because they are what tells a
            reader at a glance whether this window is worth reading. */}
        <header className="border-border border-b px-4 py-2.5">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1.5">
            <Meta className="text-[12px]">
              <Stat value={summary.rows} label={summary.rows === 1 ? "event" : "events"} />
              <Stat value={summary.toolCalls} label="tool calls" />
              <Stat
                value={summary.conversations}
                label={summary.conversations === 1 ? "conversation" : "conversations"}
              />
              {summary.durationP50 !== null ? (
                <Stat value={formatDuration(summary.durationP50)} label="median tool call" />
              ) : null}
              {summary.failed > 0 ? (
                <span className="text-destructive">{summary.failed} failed</span>
              ) : null}
              {summary.unterminated > 0 ? (
                <span className="text-warn">{summary.unterminated} never returned</span>
              ) : null}
            </Meta>
          </div>

          {/* THE FOUR LIMITS, ON SCREEN. Each changes how the rows below are
              read, so none of them is behind the info affordance — only the
              mechanism behind them is. */}
          <dl className="border-border mt-2 grid grid-cols-[86px_1fr] gap-x-2 gap-y-1 border-l-2 pl-2.5 text-[11px] leading-relaxed">
            <dt className="text-muted-foreground/70">Proves</dt>
            <dd>
              which tool ran, when, for how long and in which conversation — and every change to
              memory the seat recorded, with who caused it.
            </dd>
            <dt className="text-muted-foreground/70">Cannot prove</dt>
            <dd className="text-muted-foreground">
              that any of it can be re-run. Tool results are not kept, and nothing snapshots what
              memory held beforehand.
            </dd>
            <dt className="text-muted-foreground/70">Does not cover</dt>
            <dd className="text-muted-foreground">
              the memory server's own trail, which is kept elsewhere and rotates; or a turn that
              crashed before it was written down — a quiet stretch here is not evidence that
              nothing ran.
              <InfoHint label="what this trail is and is not" className="ml-1.5 translate-y-[2px]">
                This is the conversation seat's own record: its learning ledger, its tool calls
                joined back together from their start and end events, and its retrievals, merged
                into one stream and sorted so that two exports of the same window can be compared
                line for line.
                <br />
                <br />
                The memory server keeps a separate audit trail inside its own database. That one
                is not served over HTTP at all and is trimmed on a retention timer, so nothing on
                this screen reflects it.
                <br />
                <br />
                Rows are written when a turn finishes, not as it runs. A tool that was invoked and
                never returned still appears, marked as such; a turn that died before the write
                leaves nothing behind.
              </InfoHint>
            </dd>
          </dl>
        </header>

        {/* QUERY — sent to the seat, and carried by the file. */}
        <div className="border-border flex flex-wrap items-start gap-x-3 gap-y-2 border-b px-4 py-2.5">
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-muted-foreground/70 text-[11px] font-medium tracking-wide uppercase">
                Window
              </span>
              {AUDIT_WINDOWS.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  aria-pressed={windowId === option.id}
                  onClick={() => setWindowId(option.id)}
                  className={cn(
                    "flex h-6 items-center rounded-md border px-2 text-[11px]",
                    "transition-colors duration-100",
                    "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
                    windowId === option.id
                      ? "border-primary/40 bg-primary/10 text-primary"
                      : "border-border text-muted-foreground hover:bg-accent hover:text-foreground",
                  )}
                >
                  {option.label}
                </button>
              ))}

              <label className="ml-1 flex items-center gap-1.5">
                <span className="text-muted-foreground/70 text-[11px] font-medium tracking-wide uppercase">
                  Conversation
                </span>
                {/* Options come from the ids present in the trail, not from
                    GET /v1/conversations: that route is keyed on the profile,
                    and it lists conversations that produced no audit rows. */}
                <select
                  value={conversationId ?? ""}
                  onChange={(e) => setConversationId(e.target.value || null)}
                  className={cn(
                    "border-border bg-background text-muted-foreground hover:text-foreground",
                    "mono h-6 max-w-[180px] rounded-md border px-1.5 text-[11px]",
                    "focus-visible:ring-ring focus-visible:ring-2 focus-visible:outline-none",
                  )}
                >
                  <option value="" className="bg-popover text-popover-foreground">
                    all ({conversations.length})
                  </option>
                  {conversations.map((id) => (
                    <option key={id} value={id} className="bg-popover text-popover-foreground">
                      {id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {/* VIEW — narrows the list only. Counts are of the whole window and
                do not move as toggles are pressed, so they stay readable as
                "what is in here" rather than "what is left". */}
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-muted-foreground/70 text-[11px] font-medium tracking-wide uppercase">
                Who
              </span>
              {ACTOR_ORDER.map((actor) => (
                <Toggle
                  key={actor}
                  on={actors.has(actor)}
                  count={summary.actors[actor]}
                  label={actorLabel(actor)}
                  hint={ACTOR_META[actor].hint}
                  onClick={() => setActors(toggle(actors, actor))}
                />
              ))}
              <span className="text-muted-foreground/70 ml-1 text-[11px] font-medium tracking-wide uppercase">
                What
              </span>
              {SOURCE_ORDER.map((source) => (
                <Toggle
                  key={source}
                  on={sources.has(source)}
                  count={summary.sources[source]}
                  label={SOURCE_META[source].label}
                  hint={SOURCE_META[source].hint}
                  onClick={() => setSources(toggle(sources, source))}
                />
              ))}
            </div>
          </div>

          <div className="flex flex-col items-end gap-1">
            <ExportControls
              // The PATH is built from the frozen mount instant, so the file
              // covers the window the rows on screen came from. The NAME is
              // stamped when the button is pressed, which is what the seat
              // itself does — two exports of the same window taken an hour
              // apart are different artefacts and must not arrive under one
              // filename for the browser to disambiguate with "(1)".
              path={(format) => auditExportPath(query, format, now)}
              filenameFor={(format) => exportFilename(format, Date.now())}
              disabled={summary.rows === 0}
            />
            {/* THE ONE SENTENCE THAT KEEPS THE DOWNLOAD HONEST. */}
            <p className="text-muted-foreground/70 max-w-[268px] text-right text-[11px] leading-relaxed">
              The file covers {windowLabel.toLowerCase()},{" "}
              {conversationId ? "one conversation" : "every conversation"} —{" "}
              {summary.rows} {summary.rows === 1 ? "row" : "rows"}. Who and What narrow this list
              only.
            </p>
          </div>
        </div>

        {parsed.unreadable > 0 ? (
          <p className="border-warn/40 text-muted-foreground/80 mx-4 my-2 border-l pl-2.5 text-[11px] leading-relaxed">
            {parsed.unreadable} {parsed.unreadable === 1 ? "line" : "lines"} of the export could not
            be read on this screen — a torn file, or a seat newer than this build. They are in the
            download and are not counted above.
          </p>
        ) : null}

        {summary.actors.unknown > 0 ? (
          <p className="text-muted-foreground/70 mx-4 my-2 text-[11px] leading-relaxed">
            {summary.actors.unknown} {summary.actors.unknown === 1 ? "entry was" : "entries were"}{" "}
            written before the seat recorded who caused a change. They read <em>Unknown</em> and
            are left that way — an actor inferred after the fact is not evidence.
          </p>
        ) : null}

        {summary.rows === 0 ? (
          <div className="py-20">
            <EmptyState
              size="page"
              title="Nothing was recorded in this window"
              body="Widen it, or start a conversation — every tool call and memory change lands here."
              more="A row appears the moment a turn finishes: each tool the model invoked with its duration, each write or reinforcement of memory with who caused it, and each retrieval with what came back. Nothing is written mid-turn, so a turn still running is not here yet."
            />
          </div>
        ) : (
          <>
            <ToolCensus rows={rows} />

            {shown.length === 0 ? (
              <div className="py-16">
                <EmptyState
                  title="Nothing matches this narrowing"
                  body="Every row in the window is filtered out by Who and What."
                />
              </div>
            ) : (
              <>
                {narrowed ? (
                  <p className="text-muted-foreground/70 px-4 pt-2 text-[11px]">
                    Showing {shown.length} of {summary.rows}.
                  </p>
                ) : null}
                {days.map((group) => (
                  <section key={group.day}>
                    <div className="border-border bg-muted/50 sticky top-0 z-10 flex items-center gap-2 border-b px-4 py-1.5 backdrop-blur-sm">
                      <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
                        {dayLabel(group.day)}
                      </span>
                      <span className="text-muted-foreground/60 mono text-[10px]">
                        {group.rows.length}
                      </span>
                    </div>
                    {group.rows.map((row) => (
                      <TrailRow
                        key={position.get(row)}
                        row={row}
                        // Constant down every row once one is selected, so the
                        // column is dropped rather than repeated.
                        showConversation={conversationId === null}
                      />
                    ))}
                  </section>
                ))}
              </>
            )}
          </>
        )}
      </div>
    </ScrollArea>
  );
}
