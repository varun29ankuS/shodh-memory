import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { isHumanProfile, probeBackend, type Reachability } from "@/lib/api";
import { useSession } from "@/stores/session";

export interface ReachabilityProbe {
  reach: Reachability;
  /**
   * When the probe last RESOLVED, in epoch ms. Zero before the first one does.
   *
   * This is react-query's `dataUpdatedAt`, and it is the right clock because
   * `probeBackend` resolves for every state it models — offline and
   * unauthorized included — rather than throwing. So it means "last checked",
   * not "last healthy", which is the distinction a staleness reading depends
   * on: a probe that keeps succeeding at reporting an outage is not stale.
   */
  checkedAt: number;
}

/**
 * Connection state, polled, with the moment it was taken.
 *
 * Polling exists so the app recovers on its own when the backend comes up
 * instead of needing a reload — the common case is someone starting the server
 * after opening the page.
 *
 * SPLIT FROM `useReachability` so the status ribbon can date the reading it
 * shows. `useReachability` returns a bare `Reachability`, which is what every
 * screen in the app takes as a prop and must keep being; widening that return
 * type would have rippled through App.tsx and a dozen views to serve one
 * component. Two observers on one query key is not two polls — react-query
 * shares the cache entry and dedupes an in-flight fetch.
 */
export function useReachabilityProbe(): ReachabilityProbe {
  const { data, dataUpdatedAt } = useQuery({
    queryKey: ["reachability"],
    queryFn: ({ signal }) => probeBackend(signal),
    refetchInterval: 10_000,
    // This query resolves rather than throws for every expected state, so a
    // rejection here is a real bug and should not be retried into silence.
    retry: false,
    staleTime: 0,
  });

  return {
    reach: data ?? { state: "offline", detail: "not probed yet" },
    // `dataUpdatedAt` is 0 until the first resolution, which is precisely the
    // "never checked" the ribbon needs to tell apart from "checked, and down".
    checkedAt: data === undefined ? 0 : dataUpdatedAt,
  };
}

/**
 * Connection state and the profile list it carries.
 *
 * Keeps the profile reconciliation, which must happen exactly once no matter
 * how many things read the probe — hence its living here and not in
 * `useReachabilityProbe`.
 */
export function useReachability(): Reachability {
  const reconcileProfiles = useSession((s) => s.reconcileProfiles);
  const { reach } = useReachabilityProbe();

  // The array's identity changes on every poll even when its contents do not,
  // so the effect depends on a serialisation of the contents rather than on the
  // array itself. Without this it re-runs ten times a minute forever.
  const profileKey = JSON.stringify(
    reach.state === "online" ? reach.profiles.filter(isHumanProfile) : [],
  );

  useEffect(() => {
    reconcileProfiles(JSON.parse(profileKey) as string[]);
  }, [profileKey, reconcileProfiles]);

  return reach;
}
