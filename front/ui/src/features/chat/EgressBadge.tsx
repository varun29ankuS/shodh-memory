import { cn } from "@/lib/utils";
import type { McpServerInfo, SeatModelInfo } from "@/lib/seat/types";
import { computeEgress } from "./egress";

/**
 * Where this conversation's content goes — stated wherever the conversation
 * is happening, not in a settings pane someone visited once.
 *
 * This badge decides whether the sovereignty claim is true on screen, so it has
 * to account for every exit, not just the obvious one. It used to read the
 * model alone: local model, "Local", nothing leaves. That was correct until the
 * seat became an MCP host. A connector reached over http or sse is a second
 * way out — an agent that emails a summary has sent the conversation off the
 * machine as surely as a hosted model has — and a badge still reading "Local"
 * at that moment is the claim, wrong, in front of the buyer who cares most.
 *
 * So: local is now local model AND no remote connectors. Otherwise the badge
 * counts the exits, because "is anything leaving" is a yes/no question but
 * "how many ways out are there" is what an operator actually has to weigh.
 *
 * Factual, without alarm. Hosted models and remote connectors are normal
 * choices, so the non-local state is neutral text rather than a warning colour.
 * Only "Local" gets a hue (--live, the same green the connection strip uses),
 * because it is the property worth confirming from across a room in a demo.
 *
 * Renders nothing for an unresolved model: no badge is honest, a guessed one
 * is not.
 */
export function EgressBadge({
  info,
  servers,
}: {
  info: SeatModelInfo | null;
  /** Connected tool servers. Omitted where the caller has none to hand — the
   *  badge then reports model-only egress, which is what it did before. */
  servers?: readonly McpServerInfo[];
}) {
  const egress = computeEgress(info, servers ?? []);
  if (!egress) return null;

  return (
    <span
      title={egress.title}
      className={cn(
        "mono flex h-[22px] shrink-0 items-center gap-1.5 rounded border px-1.5 text-[10px]",
        egress.local
          ? "border-[var(--live)]/30 text-[var(--live)]"
          : "border-border text-muted-foreground",
      )}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {egress.label}
    </span>
  );
}
