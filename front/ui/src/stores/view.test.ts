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
  return { dimension: "destination", path: "/geo", reason: REASON, origin: "call-1", ...over };
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
