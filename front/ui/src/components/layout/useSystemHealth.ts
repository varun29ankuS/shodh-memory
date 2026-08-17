import { useEffect, useState } from "react";
import type { Reachability } from "@/lib/api";
import { useReachabilityProbe } from "@/app/useReachability";
import { useSeatProbe } from "@/app/useSeatHealth";
import {
  alertsOf,
  freshnessOf,
  readAssistant,
  readMemory,
  ribbonToneOf,
  type ServiceReading,
  type Tone,
} from "./systemHealth";

/**
 * How often the ribbon advances its own clock.
 *
 * A SECOND, AND NOT THE POLL INTERVAL. The probes resolve every ten seconds,
 * and if the only thing that re-rendered this were a probe result, the moment a
 * reading crosses from fresh into stale would be reported up to ten seconds
 * late — and in the case that matters most, when polling has STOPPED, it would
 * never be reported at all, because the thing that stopped is the only thing
 * that would have re-rendered. The ticker is what makes staleness detectable at
 * all rather than a state the app can only fall into silently.
 */
const TICK_MS = 1_000;

/** Wall-clock, re-read on an interval, so a component can age something. */
function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

export interface SystemHealth {
  /** Both services, memory first, always both, whatever state they are in. */
  readings: ServiceReading[];
  /** The worst tone among them — what the ribbon line is coloured. */
  tone: Tone;
  /** The ones with something to say, worst first. Empty when all is well. */
  alerts: ServiceReading[];
}

/**
 * The whole system's health, as the ribbon shows it.
 *
 * `reach` IS TAKEN AS AN ARGUMENT RATHER THAN READ. It is the instance the
 * shell already threads into every view, so passing it in is what guarantees
 * the header and the stage can never state two different diagnoses of the same
 * server — the exact contradiction `lib/api/health.ts` was written to end, with
 * a strip reading "Key rejected" over a body telling the reader to start a
 * running server. Only the CLOCK is read from the query cache, because a prop
 * cannot carry when it was fetched.
 *
 * IMPORTS FROM `@/app`, WHICH NOTHING ELSE IN `components/` DOES. The direction
 * is deliberate rather than careless: the alternative was re-declaring the
 * query keys, poll interval and retry policy here — two copies of a polling
 * contract that must not drift, so that a ribbon reporting "checked 4s ago"
 * could not quietly be reading a 30s poll.
 *
 * Both probe hooks are already mounted once in the shell. Mounting them again
 * here adds a second react-query observer on the same key, not a second poll:
 * the cache entry is shared and an in-flight fetch is deduped.
 */
export function useSystemHealth(reach: Reachability): SystemHealth {
  const { checkedAt: reachAt } = useReachabilityProbe();
  const { seat, checkedAt: seatAt } = useSeatProbe();
  const now = useNow(TICK_MS);

  const memory = readMemory(reach, freshnessOf(reachAt, now));
  // `reach.state === "online"` and not `memory.tone === "live"`: the question is
  // whether THIS PAGE can read the memory server at all, which is what makes a
  // seat that cannot reach it a disagreement worth an alarm. A profile-less but
  // answering server is still a server this page reached.
  const assistant = readAssistant(seat, reach.state === "online", freshnessOf(seatAt, now));

  const readings = [memory, assistant];
  return { readings, tone: ribbonToneOf(readings), alerts: alertsOf(readings) };
}
