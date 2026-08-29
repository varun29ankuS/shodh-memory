import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import type { UiCommand } from "@/lib/seat/types";
import { useSession } from "@/stores/session";
import { onUiCommand } from "@/stores/ui-commands";

/**
 * Applies agent-issued screen changes.
 *
 * The whole point of the design is here: a `ui_command` carries an intent, and
 * this hook satisfies it by calling the same router and the same store setters
 * the rail and the profile switcher call. There is no separate agent path
 * through the app, so an agent-driven navigation cannot reach a state a person
 * could not reach by clicking — and does not need its own tests to prove it.
 *
 * The announcement is not optional. A screen that changes by itself with no
 * explanation reads as a fault rather than as help, and the operator has to be
 * able to tell "the agent did this" from "something broke". It is returned
 * rather than rendered here so the shell decides where it belongs.
 */

export type UiAnnouncement = { text: string; at: number } | null;

/** How long the announcement stays up. Long enough to read one line without
 *  becoming furniture the operator learns to ignore. */
const ANNOUNCE_MS = 6000;

export function useUiCommands(): UiAnnouncement {
  const navigate = useNavigate();
  const setProfile = useSession((s) => s.setProfile);
  const select = useSession((s) => s.select);
  const [announcement, setAnnouncement] = useState<UiAnnouncement>(null);

  useEffect(() => {
    const apply = (command: UiCommand, reason: string) => {
      switch (command.kind) {
        case "open":
          navigate(`/${command.view}`);
          break;
        case "select_profile":
          // Through the same setter the switcher uses, so the profile is
          // pinned exactly as a deliberate human choice would be — an agent
          // switching store is a deliberate act, not an adoption from the
          // server list.
          setProfile(command.profile);
          break;
        case "select_memory":
          select(command.memory_id);
          break;
      }
      setAnnouncement({ text: reason, at: Date.now() });
    };

    return onUiCommand(apply);
  }, [navigate, setProfile, select]);

  useEffect(() => {
    if (!announcement) return;
    const timer = window.setTimeout(() => setAnnouncement(null), ANNOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [announcement]);

  return announcement;
}
