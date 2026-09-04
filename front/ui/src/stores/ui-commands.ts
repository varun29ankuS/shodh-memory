import type { UiCommand } from "@/lib/seat/types";

/**
 * Agent-issued screen changes, on their way from the event stream to the shell.
 *
 * These do not belong in the chat store. A `ui_command` is not conversation
 * state — replaying it while rebuilding a turn would re-navigate the app — and
 * applying it needs the router, which is a hook and therefore unreachable from
 * a zustand action. So the stream dispatches here and the shell subscribes.
 *
 * Deliberately not a queue. If two commands arrive in one turn the operator
 * should end on the last one, exactly as they would clicking twice; buffering
 * them would replay a navigation they have already watched happen.
 */

export type UiCommandHandler = (command: UiCommand, reason: string) => void;

const handlers = new Set<UiCommandHandler>();

/** Subscribe. Returns the unsubscribe, for use as a useEffect cleanup. */
export function onUiCommand(handler: UiCommandHandler): () => void {
  handlers.add(handler);
  return () => {
    handlers.delete(handler);
  };
}

export function dispatchUiCommand(command: UiCommand, reason: string): void {
  // A throwing subscriber must not take the event stream down with it: this is
  // called from inside the SSE handler, and an unhandled error there ends the
  // turn the operator is reading.
  for (const handler of handlers) {
    try {
      handler(command, reason);
    } catch (error) {
      console.error("ui_command handler failed", error);
    }
  }
}
