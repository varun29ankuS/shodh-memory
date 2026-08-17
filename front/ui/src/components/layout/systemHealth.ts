import { isHumanProfile, type Reachability } from "@/lib/api";
import type { SeatReachability } from "@/lib/seat/types";

/**
 * What the two services are, said in one place, as sentences.
 *
 * WHY THIS IS NOT IN THE COMPONENT. Every string below is a claim about what a
 * person can and cannot do right now, and the previous version of this — a
 * nested ternary inside `StatusStrip` — could not be tested, so two of its
 * claims were wrong and nobody found out. It told a reader to "start the shodh
 * backend" over a server that had answered 500 (it was running), and it did not
 * mention the seat at all, so "Connected" was displayed while the assistant was
 * dead. Both are pinned by tests here.
 *
 * There are TWO services and they fail independently:
 *
 *  - the memory server, which every screen reads from, and
 *  - the seat, the local process the assistant runs in.
 *
 * A person is not served by knowing that "the system" is up. They are served by
 * knowing which of the two is down, what they have therefore lost, and the one
 * thing that would fix it — so a reading carries all three.
 */

/**
 * How a state is coloured, and what each tone is allowed to mean.
 *
 * `live` / `warn` / `alarm` map onto `--live` / `--warn` / `--destructive`,
 * whose meanings this codebase is strict about: `--warn` is waiting-on,
 * `--destructive` is late or wrong. So a service nobody has started yet is
 * `warn` — the app is waiting on a person — while a server that answered with
 * an error, or two components that disagree about a server that is up, is
 * `alarm`, because something is wrong rather than merely absent.
 *
 * `unknown` deliberately takes NEITHER. It is the tone of a reading this page
 * cannot vouch for, and the honest colour for "I do not know" is the colour of
 * ordinary label text (`--muted-foreground`), not a borrowed alarm. Reaching
 * for `--warn` here would have made "not checked recently" indistinguishable
 * from "waiting on you", which is the collapse the tone vocabulary exists to
 * prevent.
 */
export type Tone = "live" | "warn" | "alarm" | "unknown";

export interface ServiceReading {
  /** Stable identity, for React keys and for tests that must not match on prose. */
  id: "memory" | "assistant";
  /** The service, in the person's terms. */
  service: string;
  /** The state, in one or two words. Paired with the tone so colour is never
   *  the only carrier. */
  state: string;
  tone: Tone;
  /**
   * What this state means for what can be done RIGHT NOW, or null when nothing
   * is lost. "Not running" is a state; "recall and graph have nothing to read"
   * is the consequence, and it is the only half of the pair a person can act
   * on without already knowing the architecture.
   */
  consequence: string | null;
  /** The one action that changes the state, or null when there is nothing to do. */
  remedy: string | null;
  /** What was actually observed. Always present: a liveness reading with no
   *  evidence behind it cannot be told from a guess. Deliberately holds no
   *  clock reading — see `checked`. */
  evidence: string;
  /**
   * How long ago this was checked, as a phrase.
   *
   * SEPARATE FROM `evidence` BECAUSE IT TICKS. The banner is a `role="status"`
   * region, and a live region whose text changes once a second is announced
   * once a second — a screen reader would read the whole outage aloud on every
   * tick. Keeping the only moving string apart lets the view hide it from the
   * accessibility tree while still showing it.
   */
  checked: string;
}

/** `useReachability` and `useSeatHealth` both poll on this interval. */
export const POLL_INTERVAL_MS = 10_000;

/**
 * When a reading stops being evidence about the present.
 *
 * Two and a half poll intervals: long enough that one slow or dropped probe
 * does not make a healthy ribbon flicker, short enough that a person cannot
 * read a stopped ribbon as a live one.
 *
 * THIS IS NOT THEORETICAL IN THIS APP. `app/providers.tsx` sets
 * `refetchOnWindowFocus: false`, and react-query's `refetchIntervalInBackground`
 * defaults to false — so polling stops entirely while the tab is in the
 * background and does NOT resume on focus, only on the next interval tick after
 * it. A tab left in the background for an hour therefore comes back showing an
 * hour-old reading, and if the server died in that hour the reading is green
 * and wrong. A liveness indicator that cannot be told from a frozen one is
 * worse than none, which is why staleness revokes the green below.
 */
export const STALE_AFTER_MS = 25_000;

export type Freshness =
  | { kind: "unprobed" }
  | { kind: "fresh"; ageMs: number }
  | { kind: "stale"; ageMs: number };

/**
 * How much this page can vouch for a reading.
 *
 * `checkedAt` is react-query's `dataUpdatedAt`, which is the moment the probe
 * RESOLVED — and both probes resolve for every state they model, including the
 * failures, so it is genuinely "when this was last checked" and not "when it
 * was last healthy". Zero means no probe has ever resolved.
 *
 * A `now` in the past of `checkedAt` is treated as age zero rather than as a
 * negative age: the system clock moving backwards is not evidence of staleness,
 * and a negative age would print as "-3s ago".
 */
export function freshnessOf(checkedAt: number, now: number): Freshness {
  if (checkedAt <= 0) return { kind: "unprobed" };
  const ageMs = Math.max(0, now - checkedAt);
  return ageMs > STALE_AFTER_MS ? { kind: "stale", ageMs } : { kind: "fresh", ageMs };
}

/**
 * An age, at the scale this ribbon works at.
 *
 * `lib/format.ts::relativeDay` is deliberately not reused: it reads
 * `Date.now()` itself (so it cannot be tested against a fixed clock) and its
 * finest step is one minute, which is six poll intervals — every age this
 * component cares about would render as "now".
 */
export function describeAge(ageMs: number): string {
  const seconds = Math.floor(ageMs / 1000);
  if (seconds < 3) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/** `checked 4s ago`, or the honest absence of a check. */
export function checkedPhrase(freshness: Freshness): string {
  return freshness.kind === "unprobed"
    ? "not checked yet"
    : `checked ${describeAge(freshness.ageMs)}`;
}

/**
 * Staleness applied to a reading, asymmetrically and on purpose.
 *
 * A GREEN CLAIM IS A CLAIM ABOUT NOW, so it is the one that must be withdrawn:
 * "Live" asserts the server is answering, and a reading nobody has taken for
 * two minutes asserts nothing of the kind. It becomes `Unconfirmed` in the
 * unknown tone.
 *
 * A FAILURE CLAIM IS NOT WITHDRAWN. A server that was not running two minutes
 * ago is still, in all likelihood, not running, and downgrading that to "we are
 * not sure" would hide a real outage behind a technicality — the same trade the
 * green case is being protected from, pointing the other way. Every reading
 * carries its own age in `checked` regardless, so a reader can weigh it.
 */
function withFreshness(reading: ServiceReading, freshness: Freshness): ServiceReading {
  if (freshness.kind !== "stale" || reading.tone !== "live") return reading;
  return {
    ...reading,
    state: "Unconfirmed",
    tone: "unknown",
    consequence: `${reading.service} answered when it was last checked, but this page has stopped hearing from it — nothing on screen is known to be current.`,
    remedy: "Reload to check again.",
  };
}

const MEMORY_LOST = "Recall, graph, geo and tasks have nothing to read";

/**
 * The memory server: the thing every screen in this product reads from.
 *
 * THE TWO OFFLINE CASES ARE SPLIT HERE AND WERE NOT BEFORE. `Reachability`
 * models "nothing answered" and "answered with an error" as one `offline`
 * state, and `lib/api/health.ts` is careful about the difference — its
 * `outageOf` refuses to head a 500 with "Not running" precisely because that
 * would send a reader to restart a healthy server. The strip made exactly that
 * mistake: it printed `Not running — start the shodh backend` over a backend
 * that had answered 500 and was therefore already running. Two surfaces, one
 * screen, contradicting each other about whether to start a process.
 */
export function readMemory(reach: Reachability, freshness: Freshness): ServiceReading {
  const service = "Memory";
  const checked = checkedPhrase(freshness);

  if (freshness.kind === "unprobed") {
    return {
      id: "memory",
      service,
      state: "Checking…",
      tone: "unknown",
      consequence: null,
      remedy: null,
      evidence: "the first check has not answered yet",
      checked,
    };
  }

  if (reach.state === "unauthorized") {
    return withFreshness(
      {
        id: "memory",
        service,
        state: "Key rejected",
        tone: "alarm",
        // Saying the server is running is the whole correction: it stops a
        // reader from restarting a healthy backend to fix an authentication
        // problem. Same sentence the stage-level `outageOf` makes.
        consequence: `The memory server is running and refused this key, so ${MEMORY_LOST.toLowerCase()} — nothing has been read at all.`,
        remedy: "Set SHODH_API_KEY to the key the backend was started with, then reload.",
        evidence: `answered ${reach.status}`,
        checked,
      },
      freshness,
    );
  }

  if (reach.state === "offline") {
    return reach.answered === undefined
      ? withFreshness(
          {
            id: "memory",
            service,
            state: "Not running",
            tone: "warn",
            consequence: `${MEMORY_LOST}.`,
            remedy: "Start the shodh backend.",
            evidence: reach.detail,
            checked,
          },
          freshness,
        )
      : withFreshness(
          {
            id: "memory",
            service,
            state: "Erroring",
            tone: "alarm",
            consequence: `The memory server is running but could not serve the request, so ${MEMORY_LOST.toLowerCase()}.`,
            remedy: "Read the shodh backend's log — it is already up, so starting it again is not the fix.",
            evidence: `answered ${reach.answered}`,
            checked,
          },
          freshness,
        );
  }

  const profiles = reach.profiles.filter(isHumanProfile).length;
  if (profiles === 0) {
    return withFreshness(
      {
        id: "memory",
        service,
        state: "No profiles",
        tone: "warn",
        // Not an error — a fresh install. But not usable either, and a screen
        // that is empty for this reason looks identical to one that is broken.
        consequence: `Connected, but nothing is stored yet, so ${MEMORY_LOST.toLowerCase()}.`,
        remedy: "Store a memory through the assistant or the MCP server to create a profile.",
        evidence: "answering, 0 profiles",
        checked,
      },
      freshness,
    );
  }

  return withFreshness(
    {
      id: "memory",
      service,
      state: "Live",
      tone: "live",
      consequence: null,
      remedy: null,
      evidence: `${profiles} profile${profiles === 1 ? "" : "s"}`,
      checked,
    },
    freshness,
  );
}

/**
 * The seat, named for what it does rather than for what it is.
 *
 * "Seat" is the process; "Assistant" is what a person loses when it stops. The
 * jargon is kept in the remedy, where it is the thing you actually type, and
 * out of the name, where it explains nothing. (`features/chat/ChatView.tsx` and
 * `features/providers/ProvidersView.tsx` both head this state "Seat not
 * running"; those are another owner's files, and aligning them is a follow-up,
 * not a silent edit.)
 *
 * `memoryReadable` is whether THIS PAGE can read the memory server. It exists
 * for one state and it is the most valuable one here — see below.
 */
export function readAssistant(
  seat: SeatReachability,
  memoryReadable: boolean,
  freshness: Freshness,
): ServiceReading {
  const service = "Assistant";
  const checked = checkedPhrase(freshness);

  if (freshness.kind === "unprobed") {
    return {
      id: "assistant",
      service,
      state: "Checking…",
      tone: "unknown",
      consequence: null,
      remedy: null,
      evidence: "the first check has not answered yet",
      checked,
    };
  }

  if (seat.state === "offline") {
    return withFreshness(
      {
        id: "assistant",
        service,
        state: "Not running",
        tone: "warn",
        // The consequence is the point. Someone reading "Memory: Live" has no
        // way to know the assistant cannot move this view or touch tasks, and
        // will find out by asking it to and getting nothing.
        consequence:
          "The assistant cannot answer, move this view, or touch tasks. Conversations and History have nothing to show.",
        remedy: "Start the seat on port 3141.",
        evidence: seat.detail,
        checked,
      },
      freshness,
    );
  }

  /**
   * The seat is up and says IT cannot reach the memory server.
   *
   * When this page cannot reach it either, that is one outage seen from two
   * vantage points, and reporting it twice would put a second alarm on screen
   * for a fact the memory row already states with a better remedy. The seat
   * process is genuinely fine, so it reports as such and the memory row carries
   * the outage.
   *
   * When this page CAN read memory, the two components disagree about a server
   * that is demonstrably up — the seat is pointed somewhere else, or its key is
   * not this one. That is invisible everywhere else in the product and it is
   * silently fatal: the assistant answers, and remembers nothing.
   */
  if (!seat.backendOk) {
    return memoryReadable
      ? {
          id: "assistant",
          service,
          state: "No memory",
          tone: "alarm",
          consequence:
            "The assistant is running but cannot reach the memory server, which this page can read — so it will answer, and recall and remember nothing.",
          remedy: "Check the seat's SHODH_API_URL and SHODH_API_KEY against the ones this UI proxies to.",
          evidence: `seat reports: ${seat.backendDetail}`,
          checked,
        }
      : withFreshness(
          {
            id: "assistant",
            service,
            state: "Ready",
            tone: "live",
            consequence: null,
            remedy: null,
            evidence: "running, waiting on the memory server like this page",
            checked,
          },
          freshness,
        );
  }

  return withFreshness(
    {
      id: "assistant",
      service,
      state: "Ready",
      tone: "live",
      consequence: null,
      remedy: null,
      evidence: "seat and its backend both answering",
      checked,
    },
    freshness,
  );
}

/** Severity order. `unknown` outranks `live` because "I cannot vouch for this"
 *  must not be painted as health, and sits below the two real failures because
 *  a known failure is more urgent than an unconfirmed reading. */
const SEVERITY: Record<Tone, number> = { live: 0, unknown: 1, warn: 2, alarm: 3 };

/** The tone of the whole system: its worst service. One green service does not
 *  offset a dead one. */
export function ribbonToneOf(readings: readonly ServiceReading[]): Tone {
  return readings.reduce<Tone>(
    (worst, r) => (SEVERITY[r.tone] > SEVERITY[worst] ? r.tone : worst),
    "live",
  );
}

/**
 * The services that have something to say, worst first.
 *
 * `unknown` is excluded, and that is the whole restraint of this design: the
 * first frame after load is `Checking…` for both services, and a banner that
 * flashed an explanation there would announce a problem that does not exist on
 * every single page load. An unconfirmed reading has already withdrawn its
 * green in the line and the chip, which is proportionate to a state that means
 * "ask again", not "something is wrong".
 */
export function alertsOf(readings: readonly ServiceReading[]): ServiceReading[] {
  return readings
    .filter((r) => r.tone === "warn" || r.tone === "alarm")
    .sort((a, b) => SEVERITY[b.tone] - SEVERITY[a.tone]);
}
