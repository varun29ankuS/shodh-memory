import { useQuery } from "@tanstack/react-query";

import { listMcpServers } from "@/lib/seat/client";
import type { McpServerInfo } from "@/lib/seat/types";

/**
 * Connected tool servers, for the egress badge.
 *
 * Shares the cache entry the Providers screen already populates
 * (`["seat-mcp-servers"]`), so the badge costs nothing extra when that screen
 * has been open and one cheap local request when it has not.
 *
 * The badge is a safety claim, so it must not go stale silently: a connector
 * that comes up between turns is a new way out of the machine, and an operator
 * reading "Local" while it is live is worse than a slightly chatty poll.
 */
export function useEgressServers(): McpServerInfo[] {
  const query = useQuery({
    queryKey: ["seat-mcp-servers"],
    queryFn: ({ signal }) => listMcpServers(signal),
    staleTime: 10_000,
    refetchInterval: 15_000,
  });
  return query.data?.servers ?? [];
}
