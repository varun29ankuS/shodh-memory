import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useLocation, useNavigate } from "react-router-dom";
import type { UniverseModel } from "@/features/graph/universe";
import { universeKey } from "@/features/graph/useUniverse";
import { reportView } from "@/lib/seat/client";
import type { ViewReportWire, ViewSnapshotWire } from "@/lib/seat/types";
import { commandsFromOp, isAlreadyThere } from "@/lib/view/commands";
import { EMPTY_CURSOR, advance, type TurnCursor } from "@/lib/view/cursor";
import type { ViewVerdict } from "@/lib/view/outcome";
import { useChat } from "@/stores/chat";
import { useSession } from "@/stores/session";
import { useView } from "@/stores/view";

/**
 * The conversation's producer on the view bus — mounted ONCE, in the shell.
 *
 * `EvidencePanel` and `MessageList` already read the same `memory_recall` ops
 * and they stay exactly as they are: they render what happened. This is the one
 * thing that turns what happened into what is on screen. Keeping it single is
 * not tidiness — two translators would double-dispatch every op, and each would
 * mount and unmount with its own surface, so which of them was listening would
 * depend on where you happened to be standing.
 *
 * It also carries the other direction: the human's hands do not know the bus
 * exists, so their touches are DETECTED rather than reported. Three of the four
 * dimensions are observable as state — the cue and the selected entity are
 * fields in the session store, the destination is the URL — so they are watched
 * here instead of being instrumented at a dozen call sites that would each have
 * to remember. The fourth, the camera, is a gesture with no state to watch, so
 * `EntityCanvas` reports it directly at the one place it already detects a real
 * wheel or drag.
 *
 * AND IT CARRIES THE RETURN LEG. The authority ledger's verdicts used to end
 * here: the browser decided, the view moved or did not, and nobody upstream ever
 * found out — so the model described a view it could not see and History could
 * only say what was asked. This hook is where those verdicts leave, because it
 * is the one place that knows both the bus and which conversation is live. It
 * also answers the seat's `view_probe`, which is how `inspect_view` gets a
 * reading rather than a guess.
 */

export function useAgentView(): void {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  /** The path as of this render, for listeners that must not re-run on a move. */
  const pathRef = useRef(pathname);
  pathRef.current = pathname;

  /**
   * What is on screen, assembled at the moment it is asked for.
   *
   * READ THROUGH `getState`, NEVER FROM A RENDER, because every caller of this
   * is an event handler or a store transition — a snapshot closed over a render
   * would describe the view as it was when this component last painted, which on
   * a fast navigation is a different view from the one the person is looking at.
   *
   * The entity's NAME comes from the loaded graph when there is one and is null
   * when there is not. Null is the honest answer: the browser holds a uuid, and
   * a name it has not loaded is not a name it can supply.
   */
  /**
   * The graph's name for an entity uuid, or null.
   *
   * Read from the universe query's cache rather than fetched: this runs inside a
   * store transition and cannot wait on the network, and the universe is
   * uncapped, so a fetch here would be a multi-megabyte read to label one line
   * of a tool result. Null when the person has not opened the graph yet, which
   * is a state the seat reports as "the workbench has not loaded the graph that
   * would name it" — true, and better than a guess.
   */
  const nameOf = (id: string, profile: string | null): string | null => {
    const model = queryClient.getQueryData<UniverseModel>(universeKey(profile));
    const index = model?.index.get(id);
    return index === undefined ? null : (model?.nodes[index]?.name ?? null);
  };

  const snapshot = useRef<() => ViewSnapshotWire>(() => {
    const session = useSession.getState();
    const view = useView.getState();
    const selected = session.selectedEntityId;

    let focus: ViewSnapshotWire["focus"] = null;
    if (selected !== null) {
      focus = { id: selected, name: view.focus?.id === selected ? view.focus.name : nameOf(selected, session.profile) };
    }

    // The cue is the model's only while the bus's record still matches what is
    // in the field. The instant the person types, `touch("cue")` drops that
    // record — so comparing them is what keeps the attribution honest rather
    // than sticky.
    const cue =
      session.cueDraft.length > 0 || session.cueEntities.length > 0
        ? {
            text: session.cueDraft,
            entities: session.cueEntities,
            author: (view.cue?.text === session.cueDraft ? "agent" : "user") as "user" | "agent",
          }
        : null;

    return {
      destination: pathRef.current,
      profile: session.profile,
      cue,
      focus,
      claimed: [...view.claimed],
      offers: Object.entries(view.offers).map(([dimension, command]) => ({
        dimension: dimension as ViewSnapshotWire["claimed"][number],
        reason: command.reason ?? "",
      })),
    };
  });

  /**
   * Send one report to the seat, or drop it when there is nowhere to send it.
   *
   * `activeId` is read at call time rather than captured: a verdict can be
   * produced by a person accepting an offer long after the turn that raised it,
   * and by then the live conversation may have changed. There is exactly one
   * conversation whose seat is waiting for this — the one that is active — and
   * with none, the seat's own answer is already "not known".
   */
  const send = useRef<(report: Omit<ViewReportWire, "view">) => void>((report) => {
    const conversationId = useChat.getState().activeId;
    if (!conversationId) return;
    if (report.probe_id === null && report.outcomes.length === 0) return;
    reportView(conversationId, { ...report, view: snapshot.current() });
  });

  /**
   * Verdicts, batched across one synchronous burst.
   *
   * BATCHING IS NOT AN OPTIMISATION HERE, IT IS THE CONTRACT. One `direct_view`
   * arrives as up to four commands and dispatches four times, each producing its
   * own verdict; the seat resolves the waiting tool call on the FIRST report it
   * receives (there is no way for it to know how many are coming). Sent one at a
   * time, the model would be told what happened to the cue and never learn that
   * the destination it actually cared about was held as an offer.
   *
   * A microtask is the right grain because it is exactly the burst: every
   * dispatch caused by one event, and every dispatch `follow()` makes, runs
   * synchronously before the queue drains. Anything later is a different act by
   * a person and deserves its own report.
   */
  const queued = useRef<ViewVerdict[]>([]);
  const scheduled = useRef(false);
  const report = useRef<(verdicts: readonly ViewVerdict[]) => void>((verdicts) => {
    if (verdicts.length === 0) return;
    queued.current.push(...verdicts);
    if (scheduled.current) return;
    scheduled.current = true;
    queueMicrotask(() => {
      scheduled.current = false;
      const batch = queued.current;
      queued.current = [];
      if (batch.length === 0) return;
      send.current({
        probe_id: null,
        outcomes: batch.map((verdict) => ({
          tool_call_id: verdict.origin,
          dimension: verdict.dimension,
          state: verdict.state,
        })),
      });
    });
  });

  /**
   * The bus's verdicts, on their way out.
   *
   * Registered once, and torn down on unmount so a stale hook cannot keep
   * reporting for a shell that is gone.
   */
  useEffect(() => {
    const sink = report.current;
    useView.getState().setReporter(sink);
    return () => {
      if (useView.getState().report === sink) useView.getState().setReporter(null);
    };
  }, []);

  /**
   * The conversation, watched THROUGH THE STORE rather than through a render.
   *
   * This was a `useEffect` on the live turn, and the ordering was wrong in a way
   * that only shows up under a real hand. `send` appends the pending turn inside
   * a `set`, but an effect does not run until React has committed — so anything
   * the person did in that window (typing into the cue field as they hit enter)
   * was seen FIRST, and `beginTurn` then wiped the claim they had just made.
   * Observed live: `touch(cue)` at t, `beginTurn` at t+119ms, and the model went
   * on to take a dimension the user was holding.
   *
   * A store subscriber runs synchronously inside that same `set`, so the turn
   * boundary lands before any consequence of the send can be acted on. The
   * cursor arithmetic is in `lib/view/cursor.ts`, where it is testable.
   */
  const cursor = useRef<TurnCursor>(EMPTY_CURSOR);
  useEffect(
    () =>
      useChat.subscribe((state) => {
        const id = state.activeId;
        if (!id) return;
        const convo = state.conversations[id];
        if (!convo) return;
        // KEYED ON POSITION, NOT ON THE TURN NUMBER. `send` numbers a turn by
        // array length and the seat's `turn_start` then overwrites it with its
        // own count (stores/chat.ts). Keying on the number makes those two
        // values look like two different turns, which would reopen the
        // authority window mid-answer and discard a touch made in between.
        const index = convo.turns.length - 1;
        const live = convo.turns[index];
        if (!live) return;

        const step = advance(cursor.current, {
          key: `${id}#${index}`,
          pending: live.pending,
          ops: live.ops,
        });
        cursor.current = step.cursor;

        if (step.beginTurn) useView.getState().beginTurn(pathRef.current);
        for (const op of step.fresh) {
          // A probe is a question, not a command. Answered on the spot and
          // separately from the verdict batch, because it carries no verdict —
          // what it carries is the current reading, which `send` attaches to
          // every report.
          if (op.type === "view_probe") {
            send.current({ probe_id: op.probe_id, outcomes: [] });
            continue;
          }

          // A destination the person is ALREADY ON produces no command, so
          // nothing downstream would ever answer for it — and a model told
          // nothing at all would conclude its request evaporated. It did not:
          // the person is looking at exactly what was asked for. Reported
          // through the same batch as the commands beside it, so one call gets
          // one answer.
          if (op.type === "view_command" && isAlreadyThere(op, pathRef.current)) {
            // TOLD TO BOTH PARTIES, AND THIS IS THE HALF THAT USED TO BE
            // MISSING. The seat learns the fate of its ask; the person learns
            // WHY the model brought it up, which for a request whose only
            // content was this destination is otherwise never shown anywhere —
            // no command is produced, so no notice records the reason and the
            // block that renders it has nothing to render. Called before the
            // dispatch loop, so commands from the same request join this notice
            // on their shared reason rather than replacing it.
            useView.getState().alreadyThere("destination", op.reason);
            report.current([{ origin: op.tool_call_id, dimension: "destination", state: "already" }]);
          }

          for (const command of commandsFromOp(op, pathRef.current)) {
            useView.getState().dispatch(command, "agent");
          }
        }
      }),
    [],
  );

  /**
   * The cue, and who wrote it.
   *
   * Watched rather than instrumented because there are several ways to set one
   * — the header field per keystroke, a cue chip on the empty recall state, a
   * committed search — and a producer that has to be remembered at each of them
   * is a producer that will be forgotten at the next one. A change that does
   * not match what the bus just applied came from a person.
   */
  useEffect(
    () =>
      useSession.subscribe((state, previous) => {
        if (state.profile !== previous.profile) {
          // The model's terms name entities in the corpus it was reading. In
          // another profile they match nothing, so the graph would sit dimmed
          // under a chip crediting a conversation that was about somewhere else.
          useView.getState().release();
          return;
        }
        if (state.cueDraft !== previous.cueDraft && state.cueDraft !== useView.getState().cue?.text) {
          useView.getState().touch("cue");
        }
        // The selected entity, watched for the same reason and by the same
        // test. It is set from the canvas, from a neighbour link in the
        // inspector and from a keyboard step through the nodes — three call
        // sites today and no reason to think that is the last of them — so a
        // producer that had to be remembered at each would be forgotten at the
        // next. A selection that is not the one the bus just applied is a hand.
        if (
          state.selectedEntityId !== previous.selectedEntityId &&
          state.selectedEntityId !== useView.getState().focus?.id
        ) {
          useView.getState().touch("focus");
        }
      }),
    [],
  );

  /**
   * The destination, and who moved it.
   *
   * Every navigation in the product is a bare `navigate(path)` from a door, a
   * rail row, a spine or the evidence panel — none of which has heard of the
   * bus. So the URL is the signal: any change that is not the one the bus just
   * applied was a person's.
   */
  const firstPath = useRef(true);
  const applied = useRef<string | null>(null);
  useEffect(() => {
    if (firstPath.current) {
      firstPath.current = false;
      return;
    }
    if (applied.current === pathname) {
      applied.current = null;
      return;
    }
    useView.getState().touch("destination");
  }, [pathname]);

  const destination = useView((s) => s.destination);
  useEffect(() => {
    if (!destination) return;
    if (destination.path === pathRef.current) return;
    applied.current = destination.path;
    navigate(destination.path);
  }, [destination, navigate]);
}
