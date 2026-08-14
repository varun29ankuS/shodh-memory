import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ApiError, NetworkError, type Reachability } from "@/lib/api";
import { fetchUniverse } from "@/lib/api/graph";
import { useCorpus } from "@/lib/api/corpus";
import { useSession } from "@/stores/session";
import { useGround } from "@/lib/ground";
import { LAND, INDIA } from "@/lib/atlas";
import {
  isolatedMemories,
  offPatternLocations,
  quantityOutliers,
  type Finding,
  type LensResult,
} from "@/features/anomalies/measures";
import { cn } from "@/lib/utils";
import mark from "@/assets/shodh-mark.png";
import { DotMap, type DotMapExtent } from "./DotMap";
import {
  corpusSpan,
  lastWrite,
  longAgo,
  ontology,
  places,
  shortAgo,
  sinceYouLeft,
  suggestedQuestions,
} from "./derive";

/**
 * The briefing — what you land on.
 *
 * NOT A DASHBOARD. A dashboard is pre-composed, glanceable and monitored: a
 * grid of tiles each with a fixed job. An analyst does not monitor a memory,
 * they interrogate it. So this is a front page — a masthead, one sentence
 * saying what is in here, the shape of what it knows, four DOORS and a way to
 * ask. Each door takes you somewhere; none of them is a widget you watch.
 *
 * PRINT SEPARATES WITH RULES AND COLUMNS, NOT CARDS. There is not a single box
 * on this screen. The doors are columns divided by hairlines, the sections are
 * divided by rules, and the two that matter most are simply WIDER — which is
 * how the page says which one matters without a badge saying so.
 *
 * WEIGHT DECREASES AS SIZE GROWS. The large figures are set at weight 380 and
 * the small labels at 600. That is optical-size compensation, and it is the
 * reason a big number on paper reads as set rather than as shouted.
 *
 * EVERYTHING HERE IS RETRIEVED. Nothing is generated, and nothing is invented
 * when it is missing: an element with no data source is omitted rather than
 * filled with a plausible figure. The one rule that governs every branch below
 * is that AN EMPTY BRIEFING AND A BROKEN ONE MUST NEVER LOOK THE SAME. A
 * corpus with nothing in it says so; a corpus whose entity read failed says
 * THAT, and never reports zero entities on its behalf.
 */

// =============================================================================
// FAILURE, IN WORDS
// =============================================================================

/**
 * What went wrong, in the words that tell someone what to do about it.
 *
 * `client.ts` distinguishes a server that answered from a server that was
 * never reached, because a 401 means "fix the key" and a network failure means
 * "start the backend" — collapsing them produces the advice to check your
 * network when nothing is running.
 */
function describeFailure(err: unknown): string {
  if (err instanceof ApiError) {
    return err.isAuthFailure
      ? `the store rejected this key (HTTP ${err.status})`
      : `the store answered HTTP ${err.status}`;
  }
  if (err instanceof NetworkError) return `the store could not be reached — ${err.message}`;
  return err instanceof Error ? err.message : "an unrecognised failure";
}

/**
 * A read that failed, stated where the figures it would have produced belong.
 *
 * TWO DIFFERENT FAILURES, AND THEY MUST NOT SHARE A SENTENCE. A read that
 * failed with nothing cached leaves the page with no figure at all; a read
 * that failed while a previous answer is still in the cache leaves the page
 * showing NUMBERS THAT ARE NO LONGER CURRENT. Telling someone "these are
 * absent rather than zero" while a full ontology sits above it is a worse lie
 * than the zero would have been.
 *
 * Bordered on the alarm hue rather than tinted: this is a fact about the
 * session, not an alert to be dismissed.
 */
function ReadFailed({ what, err, stale }: { what: string; err: unknown; stale: boolean }) {
  return (
    <p
      role="status"
      className="border-border border-l-destructive text-muted-foreground mt-4 max-w-[78ch] border border-l-[3px] px-4 py-3 text-[13px] leading-normal"
    >
      <span className="text-destructive mono mb-1 block text-[10px] tracking-[0.14em] uppercase">
        {what} could not be read
      </span>
      {stale ? (
        <>
          The figures on this page from that read are the last ones that arrived, not the
          current state — {describeFailure(err)}.
        </>
      ) : (
        <>
          Nothing below is standing in for it — {describeFailure(err)}. The figures this read
          would have produced are absent rather than zero.
        </>
      )}
    </p>
  );
}

// =============================================================================
// MASTHEAD
// =============================================================================

function GroundSwitch() {
  const { ground, setGround } = useGround();
  return (
    <div
      role="group"
      aria-label="Ground"
      className="border-border ml-2 inline-flex self-center border"
    >
      {(
        [
          ["light", "Paper"],
          ["dark", "Night"],
        ] as const
      ).map(([value, label], i) => (
        <button
          key={value}
          type="button"
          onClick={() => setGround(value)}
          aria-pressed={ground === value}
          aria-label={`Read on ${label.toLowerCase()}`}
          className={cn(
            "mono focus-visible:ring-ring cursor-pointer px-2 py-1 text-[10px] tracking-[0.12em] uppercase focus-visible:ring-2 focus-visible:outline-none",
            i > 0 && "border-border border-l",
            ground === value ? "bg-foreground text-background" : "text-muted-foreground",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function Masthead({ profile, now }: { profile: string | null; now: number }) {
  const dateline = useMemo(() => {
    const d = new Date(now);
    const day = d.toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
    const time = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
    return `${day} · ${time}`;
  }, [now]);

  return (
    <header className="border-foreground flex flex-wrap items-baseline gap-4 border-b-2 pb-3">
      {/* The mark is a low-poly elephant in reds and oranges, and it keeps its
          colour on both grounds — it is the one warm object allowed on either,
          and dimming it would make the only piece of brand on the page the
          least present thing on it. Decorative here: the wordmark beside it
          already carries the name. */}
      <img src={mark} alt="" className="size-[30px] self-center object-contain" />
      <h1 className="mono m-0 text-[22px] font-bold tracking-[-0.02em]">shodh</h1>
      {profile ? (
        <span className="mono border-border text-muted-foreground border px-1.5 py-0.5 text-[11px] tracking-[0.14em] uppercase">
          {profile}
        </span>
      ) : null}
      <span className="mono text-muted-foreground ml-auto text-[11px] tracking-[0.06em]">
        {dateline}
      </span>
      <GroundSwitch />
    </header>
  );
}

// =============================================================================
// DOORS
// =============================================================================

interface DoorRow {
  key: string;
  text: string;
  /** The right-hand gutter — a time, a count, a magnitude. */
  meta?: string;
  flagged?: boolean;
}

/**
 * One door.
 *
 * A `<button>`, not a card with a link in it: the whole column is the target,
 * so there is no small hit area to find and the keyboard reaches it in one
 * tab. `aria-label` states the destination and the figure, because a screen
 * reader arriving at a button called "The world" learns nothing about what is
 * behind it.
 */
function Door({
  index,
  title,
  label,
  figure,
  unit,
  note,
  caption,
  rows,
  wide,
  last,
  children,
  onOpen,
}: {
  index: string;
  title: string;
  label: string;
  figure?: string;
  unit?: string;
  /** Stands in for the figure when there is no number to state. */
  note?: string;
  caption?: string;
  rows?: DoorRow[];
  wide?: boolean;
  last?: boolean;
  children?: React.ReactNode;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={label}
      className={cn(
        "group border-border hover:bg-card focus-visible:bg-card col-span-12 flex min-w-0 cursor-pointer flex-col gap-3 border-b px-4 py-5 text-left transition-colors duration-150 focus-visible:outline-none",
        "lg:border-b-0 lg:border-r",
        wide ? "lg:col-span-4" : "lg:col-span-2",
        last && "lg:border-r-0",
        "focus-visible:shadow-[inset_0_0_0_2px_var(--ring)]",
      )}
    >
      <div className="flex items-baseline gap-2">
        <span className="mono text-muted-foreground text-[10px] tracking-[0.16em] uppercase">
          {index}
        </span>
        <h2 className="m-0 text-[1.02rem] leading-tight font-semibold tracking-[-0.015em]">
          {title}
        </h2>
      </div>

      {figure !== undefined ? (
        <div className="flex items-baseline gap-1.5 tabular-nums">
          <span className="text-[44px] leading-none font-[380] tracking-[-0.03em]">{figure}</span>
          {unit ? (
            <span className="mono text-muted-foreground text-[10px] tracking-[0.12em] uppercase">
              {unit}
            </span>
          ) : null}
        </div>
      ) : null}

      {note ? <p className="text-muted-foreground m-0 text-[12.5px] leading-snug">{note}</p> : null}

      {children}

      {caption ? (
        <p className="mono text-muted-foreground m-0 text-[10px] tracking-[0.06em] opacity-80">
          {caption}
        </p>
      ) : null}

      {rows && rows.length > 0 ? (
        <ul className="m-0 flex list-none flex-col p-0">
          {rows.map((r) => (
            <li
              key={r.key}
              className="border-border text-muted-foreground flex items-baseline gap-2 border-t py-1.5 text-[12.5px] leading-snug"
            >
              {r.flagged ? (
                <span className="text-destructive mono shrink-0 text-[10px]" aria-hidden="true">
                  ▲
                </span>
              ) : null}
              {/* Clamped to two lines rather than cut to one. A memory preview
                  is a sentence, and one line of it is a fragment nobody can
                  place; two is enough to recognise what it is about, which is
                  the whole job of a row in a door. */}
              <span className="line-clamp-2 min-w-0 break-words">{r.text}</span>
              {r.meta ? (
                <span className="mono text-muted-foreground ml-auto shrink-0 text-[10px] opacity-80">
                  {r.meta}
                </span>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}

      <span className="text-primary mono mt-auto flex items-center gap-1.5 pt-1 text-[10px] tracking-[0.14em] uppercase">
        Open
        <span
          aria-hidden="true"
          className="transition-transform duration-150 ease-out group-hover:translate-x-[3px] group-focus-visible:translate-x-[3px]"
        >
          →
        </span>
      </span>
    </button>
  );
}

// =============================================================================
// MAP GEOMETRY
// =============================================================================

/**
 * The world, cropped below the ice.
 *
 * 83°N to −58°S rather than the full −90..90: Antarctica is a third of the
 * frame's height and carries no memory in any corpus this product serves, so
 * including it would spend a third of a small map on a landmass nobody is
 * asking about and shrink every other continent to pay for it.
 */
const WORLD_EXTENT: DotMapExtent = [
  [-180, -58],
  [180, 83],
];

/** India's official extent with a small margin — 68.178..97.413 E,
 *  6.753..37.088 N per the LGD geometry's own bounds. The northern edge is the
 *  whole reason that asset is vendored, so the frame must not clip it. */
const INDIA_EXTENT: DotMapExtent = [
  [67.2, 5.9],
  [98.6, 37.9],
];

/** Natural Earth draws India on de-facto lines, so India's own boundary is
 *  filled into the SAME stencil — the world map inherits the correction rather
 *  than needing a second basemap. */
const WORLD_SHAPES = [LAND, INDIA];
const INDIA_SHAPES = [INDIA];

// =============================================================================
// THE SCREEN
// =============================================================================

/** Where the previous visit was marked, per profile. Per profile because
 *  "since you left" is a claim about one corpus, and carrying one mark across
 *  a profile switch would report another store's writes as yours. */
const VISIT_KEY = (profile: string) => `shodh.briefing.visited.${profile}`;

function readVisit(profile: string): number | null {
  try {
    const raw = localStorage.getItem(VISIT_KEY(profile));
    if (!raw) return null;
    const t = Number(raw);
    return Number.isFinite(t) ? t : null;
  } catch {
    return null;
  }
}

function writeVisit(profile: string, at: number) {
  try {
    localStorage.setItem(VISIT_KEY(profile), String(at));
  } catch {
    /* A visit that cannot be recorded costs one line on the next visit. */
  }
}

/** One finding per lens, then a second from each, and so on.
 *
 *  NOT SORTED ACROSS LENSES. `Finding.deviation` is comparable only WITHIN a
 *  lens — a geo modified z-score and a quantity ratio are different quantities
 *  — so ranking them against each other would be arithmetic on incompatible
 *  units. Round-robin gives each lens its strongest finding first, which is
 *  the only ordering the measures actually support. */
function interleaveFindings(lenses: LensResult[], limit: number): Finding[] {
  const queues = lenses.map((l) => (l.state === "findings" ? l.findings : []));
  const out: Finding[] = [];
  for (let depth = 0; out.length < limit; depth += 1) {
    let drew = false;
    for (const q of queues) {
      if (depth < q.length) {
        out.push(q[depth]);
        drew = true;
        if (out.length === limit) break;
      }
    }
    if (!drew) break;
  }
  return out;
}

export function BriefingView({ reach }: { reach: Reachability }) {
  const navigate = useNavigate();
  const profile = useSession((s) => s.profile);
  const setActiveQuery = useSession((s) => s.setActiveQuery);
  const askRef = useRef<HTMLInputElement | null>(null);
  const [draft, setDraft] = useState("");

  // The dateline is a clock, so it is one. Sixty seconds is the resolution it
  // prints at; anything faster re-renders the page for a value that has not
  // changed.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(id);
  }, []);

  const { data: corpus, error: corpusError, isFetching: corpusFetching } = useCorpus(reach);

  const {
    data: universe,
    error: universeError,
    isFetching: universeFetching,
  } = useQuery({
    queryKey: ["universe", profile],
    queryFn: ({ signal }) => fetchUniverse(profile!, signal),
    enabled: reach.state === "online" && profile !== null,
  });

  const memories = useMemo(() => corpus?.memories ?? [], [corpus]);
  const stars = useMemo(() => universe?.stars ?? [], [universe]);

  // ---------------------------------------------------------------- since you left
  //
  // The previous mark is captured ONCE, before the new one is written — read
  // it on every render and the line would report zero the moment the visit is
  // recorded. The new mark is written only after the corpus has actually
  // loaded: stamping a visit for a read that failed would silently swallow
  // everything written between the two sessions.
  const [priorVisit, setPriorVisit] = useState<number | null>(null);
  const stampedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!profile) return;
    if (stampedFor.current === profile) return;
    if (!corpus) return;
    setPriorVisit(readVisit(profile));
    writeVisit(profile, Date.now());
    stampedFor.current = profile;
  }, [profile, corpus]);
  useEffect(() => {
    // A profile change invalidates the mark that was captured for the old one.
    if (stampedFor.current !== profile) setPriorVisit(null);
  }, [profile]);

  // ---------------------------------------------------------------- derived
  const bands = useMemo(() => ontology(stars), [stars]);
  const questions = useMemo(() => suggestedQuestions(stars), [stars]);
  const geo = useMemo(() => places(memories), [memories]);
  const span = useMemo(
    () => corpusSpan(memories, corpus?.total ?? 0, now),
    [memories, corpus?.total, now],
  );
  const since = useMemo(
    () => sinceYouLeft(memories, priorVisit, now),
    [memories, priorVisit, now],
  );
  const written = useMemo(() => lastWrite(memories), [memories]);

  const lenses = useMemo<LensResult[]>(
    () => [offPatternLocations(memories), quantityOutliers(memories), isolatedMemories(memories)],
    [memories],
  );
  const findingCount = lenses.reduce(
    (a, l) => a + (l.state === "findings" ? l.findings.length : 0),
    0,
  );
  const noLensRan = lenses.every((l) => l.state === "insufficient");
  const topFindings = useMemo(() => interleaveFindings(lenses, 3), [lenses]);

  const recent = useMemo(
    () =>
      [...memories]
        .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
        .slice(0, 4),
    [memories],
  );

  const totalMemories = corpus?.total ?? 0;
  const totalEntities = universe?.total_entities ?? 0;
  // Two reads, tracked separately. `/universe` is uncapped and takes seconds on
  // a large graph; every door below is built from the corpus alone, so tying
  // them to the slower read would blank a page whose data is already in hand.
  // Only the standfirst waits for both, because its sentence names both.
  const corpusLoading = corpusFetching && !corpus && !corpusError;
  const universeLoading = universeFetching && !universe && !universeError;
  const loading = corpusLoading || universeLoading;
  // A failed read with a cached answer behind it is a different state from a
  // failed read with nothing behind it: the first can still say something true
  // (marked as no longer current), the second must say nothing at all.
  const corpusLost = Boolean(corpusError) && !corpus;
  const universeLost = Boolean(universeError) && !universe;

  // ---------------------------------------------------------------- the ask
  const ask = (q: string) => {
    const query = q.trim();
    if (!query) return;
    setActiveQuery(query);
    navigate("/recall");
  };

  // "/" is printed on the field, so it has to work. Ignored while the caret is
  // already in something typeable, or the shortcut would eat the character.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || el?.isContentEditable) return;
      e.preventDefault();
      askRef.current?.focus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ---------------------------------------------------------------- standfirst
  //
  // Four outcomes, kept apart on purpose. "Not yet read", "read and empty",
  // "read and partial" and "the read failed" are four different facts, and the
  // stub this replaces collapsed all four into one sentence about an empty
  // profile — a corpus with fifty memories and no extracted entities reported
  // that nothing had ever been learned in it, directly above a door saying 50.
  let standfirst: React.ReactNode;
  if (reach.state === "unauthorized") {
    standfirst = (
      <>
        The memory store rejected this key (HTTP {reach.status}). Nothing on this page is
        standing in for the corpus.
      </>
    );
  } else if (reach.state === "offline") {
    standfirst = <>The memory store is not answering — {reach.detail}.</>;
  } else if (profile === null) {
    standfirst = <>No profile is open. The store listed none this session.</>;
  } else if (loading) {
    standfirst = <>Reading this profile…</>;
  } else if (corpusLost) {
    standfirst = <>This profile could not be read.</>;
  } else if (universeLost) {
    standfirst = (
      <>
        <b className="font-[620]">{totalMemories.toLocaleString()} memories</b> in this profile.
      </>
    );
  } else if (totalMemories === 0 && totalEntities === 0) {
    standfirst = <>Nothing has been written to this profile yet.</>;
  } else if (totalEntities === 0) {
    standfirst = (
      <>
        <b className="font-[620]">{totalMemories.toLocaleString()} memories</b>, and no entity
        has been extracted from them yet.
      </>
    );
  } else {
    standfirst = (
      <>
        <b className="font-[620]">{totalEntities.toLocaleString()} things</b> learned from{" "}
        <b className="font-[620]">{totalMemories.toLocaleString()} memories</b>
        {span ? <>, from {span.from} to today</> : null}.
      </>
    );
  }

  const online = reach.state === "online" && profile !== null;
  const showDoors = online && !corpusLost && !corpusLoading;

  return (
    <div className="relative h-full">
      {/* Paper tooth. A 3px dither cell, barely there — enough that the ground
          reads as stock rather than as a flat fill with the brightness turned
          down. Outside the scroller so it does not travel with the content. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-0 opacity-50"
        style={{
          backgroundImage: "radial-gradient(var(--border) 0.5px, transparent 0.5px)",
          backgroundSize: "3px 3px",
        }}
      />

      <div className="relative z-[1] h-full overflow-auto">
        <div className="mx-auto max-w-[1180px] px-4 pt-6 pb-20 sm:px-9 sm:pt-10">
          <Masthead profile={profile} now={now} />

          {/* ------------------------------------------------ standfirst
              Two sentences across the top and the ontology beneath them. The
              sentences are set at reading size rather than at label size: this
              is the front page's lede, and shrinking the second one to a
              caption would make what changed while you were away look like
              metadata about the first. */}
          <section className="border-border grid items-end gap-6 border-b pt-4 pb-5 md:grid-cols-[minmax(0,1fr)_auto]">
            <p className="m-0 max-w-[34ch] text-[clamp(1.15rem,2.1vw,1.45rem)] leading-[1.35] tracking-[-0.015em] text-balance">
              {standfirst}
            </p>

            {since ? (
              <p className="text-muted-foreground m-0 max-w-[46ch] text-[clamp(1.15rem,2.1vw,1.45rem)] leading-[1.35] tracking-[-0.015em]">
                Since you left on {since.when}:{" "}
                {since.added > 0 ? (
                  <>
                    <b className="text-foreground font-semibold">{since.added}</b> new{" "}
                    {since.added === 1 ? "memory" : "memories"}.
                  </>
                ) : (
                  <>nothing new.</>
                )}
              </p>
            ) : (
              <span />
            )}

            {/* The ontology, stated rather than drawn. One line does what a
                hairball of a thousand nodes could not: it makes the shape of
                what was extracted — including a typer that collapsed five
                sixths of a corpus into one label — the first thing you see. */}
            {bands.length > 0 ? (
              <div
                className="flex flex-wrap gap-x-[1.1rem] gap-y-1.5"
                aria-label="What kinds of things are in here"
              >
                {bands.map((b) => (
                  <span
                    key={b.label}
                    className="mono text-muted-foreground flex items-baseline gap-1.5 text-[11px] tracking-[0.06em]"
                  >
                    <span
                      className={cn(
                        "text-[15px] tabular-nums",
                        b.dominant ? "text-destructive" : "text-foreground",
                      )}
                    >
                      {b.n.toLocaleString()}
                    </span>
                    {b.label}
                    {b.dominant ? (
                      <span className="border-destructive text-destructive border px-1 text-[9px] tracking-[0.04em]">
                        {b.share}%
                      </span>
                    ) : null}
                  </span>
                ))}
              </div>
            ) : null}

            {online && !loading && !corpusLost && !universeLost && totalMemories === 0 && totalEntities === 0 ? (
              <p className="text-muted-foreground m-0 max-w-[46ch] text-[13px]">
                Memories, entities and links appear here once something is written. Start a
                conversation, or write to this profile through the API.
              </p>
            ) : null}
          </section>

          {corpusError ? (
            <ReadFailed what="This profile's memories" err={corpusError} stale={!corpusLost} />
          ) : null}
          {universeError ? (
            <ReadFailed what="This profile's entities" err={universeError} stale={!universeLost} />
          ) : null}

          {/* ------------------------------------------------ start with */}
          {questions.length > 0 ? (
            <section className="border-border flex flex-wrap items-baseline gap-2.5 border-b py-3.5">
              <span className="mono text-muted-foreground text-[10px] tracking-[0.14em] uppercase opacity-80">
                Start with
              </span>
              {questions.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => ask(q)}
                  aria-label={`Ask: ${q}`}
                  className="border-border text-muted-foreground hover:border-primary hover:text-primary focus-visible:border-primary focus-visible:text-primary cursor-pointer rounded-full border px-3 py-1 text-[13px] transition-colors duration-150 focus-visible:outline-none"
                >
                  {q}
                </button>
              ))}
              <span className="mono text-muted-foreground text-[10px] tracking-[0.14em] uppercase opacity-80">
                or ask anything below
              </span>
            </section>
          ) : null}

          {/* ------------------------------------------------ the doors */}
          {showDoors ? (
            <div className="grid grid-cols-12">
              <Door
                index="01"
                title="What you're working on"
                label={`What you're working on — ${totalMemories.toLocaleString()} memories in this profile. Open the corpus.`}
                figure={totalMemories.toLocaleString()}
                unit={totalMemories === 1 ? "memory" : "memories"}
                rows={recent.map((m) => ({
                  key: m.id,
                  text: m.content.replace(/\s+/g, " ").trim(),
                  meta: shortAgo(m.created_at, now) ?? undefined,
                }))}
                onOpen={() => navigate("/recall")}
              />

              <Door
                index="02"
                title="What's interesting"
                label={
                  noLensRan
                    ? "What's interesting — not enough of this corpus to have an opinion yet. Open anomalies."
                    : `What's interesting — ${findingCount} findings worth a look. Open anomalies.`
                }
                figure={noLensRan ? undefined : String(findingCount)}
                unit={noLensRan ? undefined : "worth a look"}
                note={
                  noLensRan
                    ? "Not enough of this corpus carries a place, a quantity or a link for these measures to have an opinion yet."
                    : undefined
                }
                rows={topFindings.map((f) => ({
                  key: f.memoryId,
                  text: f.content.replace(/\s+/g, " ").trim(),
                  meta: f.value,
                  flagged: true,
                }))}
                onOpen={() => navigate("/anomalies")}
              />

              <Door
                index="03"
                wide
                title="The world"
                label={
                  geo.located > 0
                    ? `The world — a dot-matrix world map with ${geo.sites.length} sites marked, from ${geo.located} of ${totalMemories} memories that carry a place. Open the map.`
                    : `The world — a dot-matrix world map with nothing marked: none of this profile's ${totalMemories} memories carries a coordinate. Open the map.`
                }
                caption={
                  geo.located > 0
                    ? `${geo.located} of ${totalMemories.toLocaleString()} memories carry a place`
                    : `no memory in this profile carries a place`
                }
                rows={geo.countries.slice(0, 3).map((c) => ({
                  key: c.name,
                  text: c.name,
                  meta: String(c.n),
                }))}
                onOpen={() => navigate("/geo")}
              >
                <DotMap shapes={WORLD_SHAPES} extent={WORLD_EXTENT} points={geo.sites} cell={3.2} />
              </Door>

              <Door
                index="04"
                wide
                last
                title="India"
                label={
                  geo.inIndia > 0
                    ? `India — a dot-matrix map of India drawn from the official LGD boundary, with ${geo.indiaSites.length} sites marked from ${geo.inIndia} memories. Open the map.`
                    : "India — a dot-matrix map of India drawn from the official LGD boundary. No memory in this profile falls inside it. Open the map."
                }
                caption={
                  geo.inIndia > 0
                    ? `${geo.inIndia} ${geo.inIndia === 1 ? "memory" : "memories"}, ${geo.indiaSites.length} ${geo.indiaSites.length === 1 ? "site" : "sites"} · official boundary (LGD)`
                    : "no memory here falls inside it · official boundary (LGD)"
                }
                onOpen={() => navigate("/geo")}
              >
                <DotMap
                  shapes={INDIA_SHAPES}
                  extent={INDIA_EXTENT}
                  points={geo.indiaSites}
                  cell={2.8}
                />
              </Door>
            </div>
          ) : null}

          {/* ------------------------------------------------ the ask, ambient */}
          <form
            className="border-foreground mt-8 flex flex-wrap items-center gap-3 border-t-2 pt-4"
            onSubmit={(e) => {
              e.preventDefault();
              ask(draft);
            }}
          >
            <label
              htmlFor="briefing-ask"
              className="mono text-muted-foreground text-[10px] tracking-[0.16em] uppercase"
            >
              Ask
            </label>
            <input
              id="briefing-ask"
              ref={askRef}
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={questions[0] ?? "Ask this memory a question"}
              aria-label="Ask this memory a question"
              className="border-foreground text-foreground placeholder:text-muted-foreground focus:border-primary min-w-[240px] flex-1 border-0 border-b bg-transparent py-1.5 text-[1.05rem] transition-colors duration-150 focus:outline-none"
            />
            <kbd className="mono border-border text-muted-foreground border px-1.5 py-0.5 text-[10px]">
              /
            </kbd>
          </form>

          {/* ------------------------------------------------ foot */}
          <div className="border-border text-muted-foreground mono mt-10 flex flex-wrap gap-x-5 gap-y-1 border-t pt-3 text-[10px] tracking-[0.06em] opacity-80">
            <span>Nothing here is generated — every figure is retrieved and traceable</span>
            {written ? (
              <>
                <span aria-hidden="true">·</span>
                <span>Last write {longAgo(written, now)}</span>
              </>
            ) : null}
            <span aria-hidden="true">·</span>
            <span>Local · no network</span>
            <span aria-hidden="true">·</span>
            <span>Boundary: LGD via bharatlas · CC0-1.0 / CC-BY-4.0</span>
          </div>
        </div>
      </div>
    </div>
  );
}
