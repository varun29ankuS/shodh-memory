import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ViewCommand } from "@/lib/view/commands";
import type { ViewVerdict } from "@/lib/view/outcome";
import { useSession } from "./session";
import { useView } from "./view";

/**
 * The bus's side of the return path: does every way an offer can end actually
 * say so, and does it say the right thing.
 *
 * These are store tests rather than pure-function tests because the thing under
 * test is the WIRING — `lib/view/outcome.ts` already pins what each transition
 * should report, and the failure this file exists to catch is a transition that
 * computes the right verdict and never hands it to anyone. That failure is
 * invisible from either side alone: the rule passes its tests, the store moves
 * the view correctly, and the seat is simply never told.
 */

const REASON = "these 12 memories cluster on the Malabar coast";

function agentCommand(over: Partial<Extract<ViewCommand, { dimension: "destination" }>> = {}): ViewCommand {
  return { dimension: "destination", path: "/geo", from: "/chat", reason: REASON, origin: "call-1", ...over };
}

let reported: ViewVerdict[];

beforeEach(() => {
  reported = [];
  useView.setState({
    claimed: [],
    offers: {},
    cue: null,
    frame: null,
    destination: null,
    focus: null,
    notice: null,
    seq: 0,
    report: (verdicts) => reported.push(...verdicts),
  });
  useSession.setState({ selectedEntityId: null, selectedMemoryId: null, cueDraft: "", cueEntities: [] });
});

describe("the verdict reaches the reporter", () => {
  it("reports an applied command", () => {
    useView.getState().dispatch(agentCommand(), "agent");
    expect(reported).toEqual([{ origin: "call-1", dimension: "destination", state: "applied" }]);
  });

  it("reports a held command as offered and leaves the view where it was", () => {
    useView.setState({ claimed: ["destination"] });
    expect(useView.getState().dispatch(agentCommand(), "agent")).toBe("offer");
    expect(useView.getState().destination).toBeNull();
    expect(reported).toEqual([{ origin: "call-1", dimension: "destination", state: "offered" }]);
  });

  it("reports an accepted offer as followed", () => {
    useView.setState({ claimed: ["destination"] });
    useView.getState().dispatch(agentCommand(), "agent");
    reported = [];
    useView.getState().follow();
    expect(reported).toEqual([{ origin: "call-1", dimension: "destination", state: "followed" }]);
    expect(useView.getState().destination?.path).toBe("/geo");
  });

  it("reports a dismissed offer as declined", () => {
    useView.setState({ claimed: ["destination"] });
    useView.getState().dispatch(agentCommand(), "agent");
    reported = [];
    useView.getState().dismiss();
    expect(reported).toEqual([{ origin: "call-1", dimension: "destination", state: "declined" }]);
  });

  it("reports an offer the person answered with their own hand as declined", () => {
    useView.setState({ claimed: ["destination"] });
    useView.getState().dispatch(agentCommand(), "agent");
    reported = [];
    useView.getState().touch("destination");
    expect(reported).toEqual([{ origin: "call-1", dimension: "destination", state: "declined" }]);
  });

  it("reports an offer the turn ended over as expired, NOT as declined", () => {
    // Nobody said no. A trail that could not tell these apart would let "they
    // refused" be counted for every offer that merely scrolled past.
    useView.setState({ claimed: ["destination"] });
    useView.getState().dispatch(agentCommand(), "agent");
    reported = [];
    useView.getState().beginTurn("/");
    expect(reported).toEqual([{ origin: "call-1", dimension: "destination", state: "expired" }]);
    expect(useView.getState().offers).toEqual({});
  });

  it("reports nothing for a command nobody asked for", () => {
    useView.getState().dispatch({ dimension: "cue", text: "port", entities: ["port"] }, "agent");
    expect(reported).toEqual([]);
  });

  it("moves the view even with no reporter wired", () => {
    // An unwired bus is the honest "not known" the seat already handles; it must
    // never be a reason for the view to stop working.
    useView.setState({ report: null });
    expect(() => useView.getState().dispatch(agentCommand(), "agent")).not.toThrow();
    expect(useView.getState().destination?.path).toBe("/geo");
  });
});

describe("focus", () => {
  const focusCommand: ViewCommand = {
    dimension: "focus",
    id: "uuid-9",
    name: "Dali",
    reason: REASON,
    origin: "call-1",
  };

  it("opens the entity in the inspector and records who opened it", () => {
    useView.getState().dispatch(focusCommand, "agent");
    expect(useSession.getState().selectedEntityId).toBe("uuid-9");
    expect(useView.getState().focus).toMatchObject({ id: "uuid-9", name: "Dali" });
    expect(reported).toEqual([{ origin: "call-1", dimension: "focus", state: "applied" }]);
  });

  it("writes its record BEFORE the selection, or the hand-detector misreads it", () => {
    // `useAgentView` tells a hand-made selection from this one by comparing the
    // session's id against this record. Selecting first would let that watcher
    // see a selection with no record behind it, conclude a person clicked the
    // node, and claim the axis against the model that had just set it.
    const order: string[] = [];
    const unsubscribe = useSession.subscribe(() => {
      order.push(useView.getState().focus?.id === "uuid-9" ? "record-first" : "selection-first");
    });
    useView.getState().dispatch(focusCommand, "agent");
    unsubscribe();
    expect(order).toContain("record-first");
    expect(order).not.toContain("selection-first");
  });

  it("waits as an offer when the person is holding the selection", () => {
    useView.setState({ claimed: ["focus"] });
    expect(useView.getState().dispatch(focusCommand, "agent")).toBe("offer");
    expect(useSession.getState().selectedEntityId).toBeNull();
  });

  it("drops the model's record when the person selects something themselves", () => {
    useView.getState().dispatch(focusCommand, "agent");
    useView.getState().touch("focus");
    expect(useView.getState().focus).toBeNull();
    expect(useView.getState().claimed).toContain("focus");
  });

  it("is cleared by release, because an entity uuid names nothing in another corpus", () => {
    // Release runs on a profile switch as well as by hand. Left standing, the
    // inspector would sit on an id from a graph that is no longer loaded, under
    // a record crediting a conversation about somewhere else.
    useView.getState().dispatch(focusCommand, "agent");
    useView.getState().release();
    expect(useView.getState().focus).toBeNull();
    expect(useSession.getState().selectedEntityId).toBeNull();
  });
});

describe("the person still outranks the model after the loop is closed", () => {
  it("does not apply a command on a claimed axis, whatever the verdict is worth", () => {
    const spy = vi.fn();
    useView.setState({ claimed: ["destination"], report: spy });
    useView.getState().dispatch(agentCommand(), "agent");
    useView.getState().dispatch(agentCommand({ origin: "call-2", path: "/graph" }), "agent");
    // Two asks, neither applied. Knowing the first was declined gave the second
    // no advantage — the ledger is consulted identically both times.
    expect(useView.getState().destination).toBeNull();
    expect(spy.mock.calls.flat(2).filter((v: ViewVerdict) => v.state === "applied")).toHaveLength(0);
  });
});

/**
 * The inverse of a destination change.
 *
 * THE ONE AXIS THAT HAD NO WAY BACK. A cue can be cleared, a camera re-panned, a
 * selection re-clicked; a reader whose stage was swapped mid-sentence had to
 * remember where they had been. These pin that going back is treated as the
 * person's own act — so the model cannot re-take the axis this turn, and the
 * account of a move no longer describes the view.
 */
describe("back", () => {
  beforeEach(() => {
    useView.setState({ claimed: [], offers: {}, destination: null, notice: null, seq: 0 });
  });

  it("returns to the stage the move took the person off", () => {
    useView.getState().dispatch(agentCommand({ path: "/geo", from: "/tasks" }), "agent");
    useView.getState().back();
    expect(useView.getState().destination?.path).toBe("/tasks");
  });

  it("claims the destination, so the model cannot take it again this turn", () => {
    useView.getState().dispatch(agentCommand({ path: "/geo", from: "/tasks" }), "agent");
    useView.getState().back();
    expect(useView.getState().claimed).toContain("destination");
    expect(useView.getState().dispatch(agentCommand({ path: "/graph" }), "agent")).toBe("offer");
  });

  it("drops the account of a move it has just undone", () => {
    useView.getState().dispatch(agentCommand({ path: "/geo", from: "/tasks" }), "agent");
    expect(useView.getState().notice?.reason).toBe(REASON);
    useView.getState().back();
    expect(useView.getState().notice).toBeNull();
  });

  it("offers no return trip from the return trip, so it cannot become a toggle", () => {
    useView.getState().dispatch(agentCommand({ path: "/geo", from: "/tasks" }), "agent");
    useView.getState().back();
    expect(useView.getState().destination?.from).toBeNull();
  });

  it("does nothing when there is nowhere to go back to", () => {
    useView.getState().dispatch(agentCommand({ path: "/geo", from: null }), "agent");
    const seq = useView.getState().seq;
    useView.getState().back();
    expect(useView.getState().destination?.path).toBe("/geo");
    expect(useView.getState().seq).toBe(seq);
  });

  it("does nothing when no destination command has ever landed", () => {
    useView.getState().back();
    expect(useView.getState().destination).toBeNull();
  });

  it("reports no verdict, because nobody asked for the return trip", () => {
    useView.getState().dispatch(agentCommand({ path: "/geo", from: "/tasks" }), "agent");
    reported = [];
    useView.getState().back();
    expect(reported).toEqual([]);
  });
});

/**
 * The axis the model asked for and did not have to move.
 *
 * A destination equal to the current path produces NO command, so before this
 * the request's reason — the whole payload of the feature — reached the seat and
 * never reached the person. These pin that it lands on the same record the rest
 * of the request lands on, and that it is never mistaken for a move.
 */
describe("alreadyThere", () => {
  beforeEach(() => {
    useView.setState({ claimed: [], offers: {}, destination: null, notice: null, seq: 0 });
  });

  it("records the reason for a request that moved nothing", () => {
    useView.getState().alreadyThere("destination", REASON);
    expect(useView.getState().notice).toEqual({
      reason: REASON,
      dimensions: [],
      already: ["destination"],
      seq: 0,
    });
  });

  it("never counts it as a move", () => {
    useView.getState().alreadyThere("destination", REASON);
    expect(useView.getState().notice?.dimensions).toEqual([]);
  });

  it("does not advance the sequence, because nothing moved", () => {
    useView.getState().alreadyThere("destination", REASON);
    expect(useView.getState().seq).toBe(0);
  });

  it("joins the commands of the same request on their shared reason", () => {
    useView.getState().alreadyThere("destination", REASON);
    useView.getState().dispatch(
      { dimension: "frame", entities: ["Dali"], reason: REASON, origin: "call-1" },
      "agent",
    );
    expect(useView.getState().notice).toMatchObject({
      reason: REASON,
      dimensions: ["frame"],
      already: ["destination"],
    });
  });

  it("is replaced by a later request rather than merged into it", () => {
    useView.getState().alreadyThere("destination", REASON);
    useView.getState().alreadyThere("destination", "a different account entirely");
    expect(useView.getState().notice).toMatchObject({
      reason: "a different account entirely",
      already: ["destination"],
    });
  });

  it("does not list the same axis twice when a request repeats it", () => {
    useView.getState().alreadyThere("destination", REASON);
    useView.getState().alreadyThere("destination", REASON);
    expect(useView.getState().notice?.already).toEqual(["destination"]);
  });

  it("records nothing for a reason that is only whitespace", () => {
    useView.getState().alreadyThere("destination", "   ");
    expect(useView.getState().notice).toBeNull();
  });

  it("dies when the person takes the axis it accounted for", () => {
    useView.getState().alreadyThere("destination", REASON);
    useView.getState().touch("destination");
    expect(useView.getState().notice).toBeNull();
  });

  it("survives a hand on an axis it never claimed", () => {
    useView.getState().alreadyThere("destination", REASON);
    useView.getState().touch("frame");
    expect(useView.getState().notice?.reason).toBe(REASON);
  });
});

/**
 * More than one axis can be already-right under one account, and the store is
 * the wrong place to assume otherwise.
 *
 * Only the destination reaches `alreadyThere` today — it is the one axis whose
 * command is suppressed when it names where the person already is. But this is a
 * public action on the bus, and a store that quietly dropped the second axis
 * handed to it would fail silently the day a second producer appears, in the
 * one part of this feature whose whole job is not to under-report.
 */
describe("alreadyThere across axes", () => {
  beforeEach(() => {
    useView.setState({ claimed: [], offers: {}, destination: null, notice: null, seq: 0 });
  });

  it("records more than one axis under one reason", () => {
    useView.getState().alreadyThere("destination", REASON);
    useView.getState().alreadyThere("frame", REASON);
    expect(useView.getState().notice?.already).toEqual(["destination", "frame"]);
  });
});
