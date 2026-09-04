import type { McpServerInfo, SeatModelInfo } from "@/lib/seat/types";

/**
 * Everything that can carry this conversation off the machine.
 *
 * The badge used to compute this from the model alone, and that was true until
 * the seat became an MCP host. A connector reached over http or sse is a second
 * exit — an agent that emails a summary has sent the conversation out just as
 * surely as a hosted model has, and a badge still reading "Local" at that
 * moment is not a cosmetic bug. It is the sovereignty claim, wrong, on screen,
 * in front of the buyer who cares about it most.
 *
 * The distinction is already on the wire and needs no new endpoint: a `stdio`
 * server is a process on this machine, `http` and `sse` are somewhere else.
 *
 * Deliberately counts CONNECTED servers only. A configured-but-failed connector
 * has not carried anything anywhere, and inflating the count would train the
 * operator to ignore it — the same reason the badge stays neutral for hosted
 * models instead of shouting.
 */

export type Egress = {
  /** True when nothing can leave: local model, no remote connectors. */
  local: boolean;
  /** Short label for the badge. */
  label: string;
  /** Full sentence for the tooltip. */
  title: string;
};

/** Remote transports. `stdio` runs a command here; the rest are elsewhere. */
function isRemote(server: McpServerInfo): boolean {
  return server.transport === "http" || server.transport === "sse";
}

export function computeEgress(
  model: SeatModelInfo | null,
  servers: readonly McpServerInfo[] = [],
): Egress | null {
  // No badge is honest for an unresolved model; a guessed one is not.
  if (!model) return null;

  const modelLocal = model.billing === "none";
  const remote = servers.filter((s) => s.status === "ready" && isRemote(s));
  const local = modelLocal && remote.length === 0;

  if (local) {
    return {
      local: true,
      label: "Local",
      title:
        "Runs on this machine, and every connected tool server is a local process. " +
        "Conversation content does not leave it.",
    };
  }

  const exits: string[] = [];
  if (!modelLocal) exits.push(`the model (${model.provider})`);
  for (const s of remote) exits.push(`${s.name} (${s.transport})`);

  // The count is the honest headline: "Local" or not is a yes/no question, but
  // how MANY ways out there are is what an operator actually needs to weigh.
  const label = exits.length === 1 ? "1 exit" : `${exits.length} exits`;

  return {
    local: false,
    label,
    title:
      `Content can leave this machine by ${exits.length === 1 ? "one route" : `${exits.length} routes`}: ` +
      `${exits.join(", ")}. ` +
      (modelLocal
        ? "The model itself is local — these are tool connectors."
        : "Each turn is sent to the model provider; connectors may send more."),
  };
}
