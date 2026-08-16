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
  useView.setState({ claimed: [], offers: {}, cue: null, frame: null, destination: null, seq: 0 });
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
