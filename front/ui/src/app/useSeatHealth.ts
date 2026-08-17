import { useQuery } from "@tanstack/react-query";
import { probeSeat } from "@/lib/seat/client";
import type { SeatReachability } from "@/lib/seat/types";

export interface SeatProbe {
  seat: SeatReachability;
  /** When the probe last resolved, in epoch ms; zero before the first one does.
   *  Same clock, and the same reasoning, as `ReachabilityProbe.checkedAt`. */
  checkedAt: number;
}

/**
 * Seat reachability, polled like the backend's (useReachability) and for the
 * same reason: the seat is a local process someone starts by hand, so "not
 * running" is ordinary and the app must recover on its own when it comes up.
 * Kept separate from the backend probe because the two fail independently and
 * need different remedies in front of a person.
 */
export function useSeatProbe(): SeatProbe {
  const { data, dataUpdatedAt } = useQuery({
    queryKey: ["seat-health"],
    queryFn: ({ signal }) => probeSeat(signal),
    refetchInterval: 10_000,
    retry: false,
    staleTime: 0,
  });
  return {
    seat: data ?? { state: "offline", detail: "not probed yet" },
    checkedAt: data === undefined ? 0 : dataUpdatedAt,
  };
}

/** The seat's state alone, for the screens that gate on it. */
export function useSeatHealth(): SeatReachability {
  return useSeatProbe().seat;
}
