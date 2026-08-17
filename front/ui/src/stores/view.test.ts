import { beforeEach, describe, expect, it } from "vitest";
import { useView } from "./view";
import { useSession } from "./session";
import type { ViewCommand } from "@/lib/view/commands";

/**
 * The bus, end to end: a command in, a verdict out, and the view state that
 * results. The pure rule is tested in lib/view/authority.test.ts; what is
 * checked here is that the store carries the verdict out faithfully — including
 * the part that is easy to get wrong, which is that a declined command survives
 * as something the user can accept.
 */

const CUE: ViewCommand = { dimension: "cue", text: "baltimore port", entities: ["Maersk"] };
const FRAME: ViewCommand = { dimension: "frame", entities: ["Maersk"] };
const GEO: ViewCommand = { dimension: "destination", path: "/geo" };

beforeEach(() => {
  // `notice` is listed explicitly: setState MERGES, so a record left by the
  // previous test would survive into the next one and make an assertion about a
  // reason pass for the wrong reason.
  useView.setState({
    claimed: [],
    offers: {},
    cue: null,
    frame: null,
    destination: null,
    notice: null,
    seq: 0,
  });
  useSession.setState({ cueDraft: "", cueEntities: [], activeQuery: "" });
});

describe("dispatch", () => {
  it("applies an agent cue by writing through to the session store", () => {
    expect(useView.getState().dispatch(CUE, "agent")).toBe("apply");

    // The cue lives in one place — every surface already reads it from there.
    expect(useSession.getState().cueDraft).toBe("baltimore port");
    expect(useSession.getState().cueEntities).toEqual(["Maersk"]);
    // …and the record here is provenance, so the chip can say who set it.
    expect(useView.getState().cue).toMatchObject({ text: "baltimore port" });
  });

  it("records frame and destination with a rising seq, so consumers can key an effect on it", () => {
    useView.getState().dispatch(FRAME, "agent");
    useView.getState().dispatch(GEO, "agent");

    const state = useView.getState();
    expect(state.frame).toMatchObject({ entities: ["Maersk"] });
    expect(state.destination).toMatchObject({ path: "/geo" });
    expect(state.destination!.seq).toBeGreaterThan(state.frame!.seq);
  });

  it("declines a touched dimension and holds the command as an offer", () => {
    useView.getState().touch("cue");

    expect(useView.getState().dispatch(CUE, "agent")).toBe("offer");
    // NOT DISCARDED. Silently dropping it leaves a model claiming to have done
    // something the user cannot see.
    expect(useView.getState().offers.cue).toEqual(CUE);
    expect(useSession.getState().cueDraft).toBe("");
  });

  it("declines only the touched dimension", () => {
    useView.getState().touch("cue");

    expect(useView.getState().dispatch(CUE, "agent")).toBe("offer");
    expect(useView.getState().dispatch(GEO, "agent")).toBe("apply");
    expect(useView.getState().destination).toMatchObject({ path: "/geo" });
  });

  it("keeps only the newest offer per dimension", () => {
    useView.getState().touch("destination");
    useView.getState().dispatch(GEO, "agent");
    useView.getState().dispatch({ dimension: "destination", path: "/graph" }, "agent");

    expect(useView.getState().offers.destination).toEqual({
      dimension: "destination",
      path: "/graph",
    });
  });

  it("lets a user command through whatever has been touched", () => {
    useView.getState().touch("cue");
    expect(useView.getState().dispatch(CUE, "user")).toBe("apply");
    expect(useSession.getState().cueDraft).toBe("baltimore port");
  });
});

describe("touch", () => {
  it("drops the model's attribution when the person takes the cue back", () => {
    useView.getState().dispatch(CUE, "agent");
    expect(useView.getState().cue).not.toBeNull();

    useView.getState().touch("cue");
    expect(useView.getState().cue).toBeNull();
  });

  it("clears a stale offer for the dimension it touches", () => {
    useView.getState().touch("destination");
    useView.getState().dispatch(GEO, "agent");
    expect(useView.getState().offers.destination).toBeDefined();

    useView.getState().touch("destination");
    expect(useView.getState().offers.destination).toBeUndefined();
  });
});

describe("beginTurn", () => {
  it("hands the wheel back: what was touched last turn does not mute this one", () => {
    useView.getState().touch("cue");
    useView.getState().touch("frame");

    useView.getState().beginTurn("/graph");

    expect(useView.getState().claimed).toEqual([]);
    expect(useView.getState().dispatch(CUE, "agent")).toBe("apply");
  });

  it("seeds the hold for the surface the turn was sent from", () => {
    useView.getState().beginTurn("/chat");

    expect(useView.getState().dispatch(GEO, "agent")).toBe("offer");
    expect(useView.getState().dispatch(CUE, "agent")).toBe("apply");
  });

  it("clears offers, so no chip outlives the question that produced it", () => {
    useView.getState().touch("cue");
    useView.getState().dispatch(CUE, "agent");
    expect(useView.getState().offers.cue).toBeDefined();

    useView.getState().beginTurn("/graph");
    expect(useView.getState().offers).toEqual({});
  });

  it("leaves an applied narrowing standing — a new question does not undo the last answer", () => {
    useView.getState().dispatch(CUE, "agent");
    useView.getState().beginTurn("/graph");

    expect(useView.getState().cue).not.toBeNull();
    expect(useSession.getState().cueDraft).toBe("baltimore port");
  });
});

describe("follow", () => {
  it("applies every waiting offer and empties the chip", () => {
    useView.getState().touch("cue");
    useView.getState().touch("destination");
    useView.getState().dispatch(CUE, "agent");
    useView.getState().dispatch(GEO, "agent");

    useView.getState().follow();

    expect(useSession.getState().cueDraft).toBe("baltimore port");
    expect(useView.getState().destination).toMatchObject({ path: "/geo" });
    expect(useView.getState().offers).toEqual({});
  });
});

describe("dismiss", () => {
  it("refuses visibly and keeps the wheel: a second recall cannot re-ask", () => {
    useView.getState().touch("cue");
    useView.getState().dispatch(CUE, "agent");
    useView.getState().dismiss();

    expect(useView.getState().offers).toEqual({});
    expect(useView.getState().dispatch(CUE, "agent")).toBe("offer");
  });
});

describe("release", () => {
  it("hands the whole corpus back and stops claiming the model narrowed it", () => {
    useView.getState().dispatch(CUE, "agent");
    useView.getState().dispatch(FRAME, "agent");

    useView.getState().release();

    expect(useSession.getState().cueDraft).toBe("");
    expect(useSession.getState().cueEntities).toEqual([]);
    expect(useView.getState().cue).toBeNull();
    expect(useView.getState().frame).toBeNull();
  });
});

/* -------------------------------------------------------------------------- *
 * WHY THE VIEW IS WHERE IT IS
 * -------------------------------------------------------------------------- */

const REASON = "these 12 memories cluster on the Malabar coast";
const WHY_CUE: ViewCommand = {
  dimension: "cue",
  text: "Malabar Coast",
  entities: ["Malabar Coast"],
  reason: REASON,
};
const WHY_FRAME: ViewCommand = { dimension: "frame", entities: ["Malabar Coast"], reason: REASON };
const WHY_GEO: ViewCommand = { dimension: "destination", path: "/geo", reason: REASON };

describe("the notice", () => {
  it("gathers one request's three commands onto a single record", () => {
    // They arrive as three dispatches and the store never sees the call that
    // produced them. If each landed on its own record, the reason would expire
    // when the first of its axes was touched and survive on the other two.
    useView.getState().dispatch(WHY_CUE, "agent");
    useView.getState().dispatch(WHY_FRAME, "agent");
    useView.getState().dispatch(WHY_GEO, "agent");

    expect(useView.getState().notice).toMatchObject({
      reason: REASON,
      dimensions: ["cue", "frame", "destination"],
    });
  });

  it("is replaced outright by a different reason, never appended to", () => {
    useView.getState().dispatch(WHY_CUE, "agent");
    useView.getState().dispatch({ dimension: "destination", path: "/sources", reason: "an import ran in March" }, "agent");

    expect(useView.getState().notice).toMatchObject({
      reason: "an import ran in March",
      dimensions: ["destination"],
    });
  });

  it("stays null for a command that gave no reason", () => {
    // A narrowing inferred from a recall was nobody's stated decision. A record
    // here would let the chip quote words that were never said.
    useView.getState().dispatch(CUE, "agent");
    expect(useView.getState().notice).toBeNull();
  });

  it("is NOT written by a declined command", () => {
    // The command did not apply. A line saying the model narrowed this view,
    // beside a view it never touched, is the precise claim the offer exists to
    // avoid making.
    useView.getState().touch("cue");
    expect(useView.getState().dispatch(WHY_CUE, "agent")).toBe("offer");
    expect(useView.getState().notice).toBeNull();
  });

  it("survives Follow, because accepting does not make the reason the user's", () => {
    useView.getState().touch("destination");
    useView.getState().dispatch(WHY_GEO, "agent");
    expect(useView.getState().notice).toBeNull();

    useView.getState().follow();

    // Follow re-dispatches as the USER — the decision is theirs, the words are
    // still the model's, and the chip must still be able to quote them.
    expect(useView.getState().notice).toMatchObject({ reason: REASON });
  });

  it("dies when the person takes an axis it accounted for", () => {
    useView.getState().dispatch(WHY_CUE, "agent");
    useView.getState().touch("cue");
    expect(useView.getState().notice).toBeNull();
  });

  it("survives a touch of an axis it never claimed", () => {
    // "I opened the map because these cluster on the coast" is not falsified by
    // panning the graph. Clearing on any touch would make the reason vanish for
    // the first person who moved their hand.
    useView.getState().dispatch(WHY_GEO, "agent");
    useView.getState().touch("frame");
    expect(useView.getState().notice).toMatchObject({ reason: REASON });
  });

  it("survives the next turn, because the view is still where the model put it", () => {
    // beginTurn hands the WHEEL back; it does not undo the last answer's
    // narrowing. Dropping the words here would leave a moved view with no
    // account of itself — the state this record exists to abolish.
    useView.getState().dispatch(WHY_GEO, "agent");
    useView.getState().beginTurn("/geo");
    expect(useView.getState().notice).toMatchObject({ reason: REASON });
  });

  it("is cleared by Release even when it only ever covered the destination", () => {
    // Release goes through `touch("cue")`, which tests the dimension list — and
    // a destination-only notice is not in it. Handing the whole corpus back
    // while a reason for a narrowing that no longer exists stays on screen is
    // the failure this clears unconditionally.
    useView.getState().dispatch(WHY_GEO, "agent");
    useView.getState().release();
    expect(useView.getState().notice).toBeNull();
  });
});
