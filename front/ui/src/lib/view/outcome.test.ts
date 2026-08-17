import { describe, expect, it } from "vitest";
import type { ViewCommand } from "./commands";
import { verdictsForDispatch, verdictsForEndedOffers } from "./outcome";

/**
 * The verdict rules are the honesty contract of the return path, so every test
 * here asserts a specific verdict list rather than that something was reported.
 *
 * The distinctions being defended are all the same shape: two things that look
 * alike in the store and mean different things to a person. Accepting an offer
 * against applying a fresh command. Refusing an offer against never seeing it.
 * A request replaced by the model's own next request against one the person
 * turned down.
 */

const ORIGIN = "call-1";

// No default parameter: `f(undefined)` would silently take one, and the cases
// that matter most here are precisely the ones passing an absent origin.
function agentCue(origin?: string): ViewCommand {
  return { dimension: "cue", text: "Dali", entities: ["Dali"], reason: "why", origin };
}

function agentDestination(origin?: string): ViewCommand {
  return { dimension: "destination", path: "/geo", from: "/chat", reason: "why", origin };
}

describe("verdictsForDispatch", () => {
  it("reports an agent command that applied as applied", () => {
    expect(
      verdictsForDispatch({ previous: undefined, command: agentCue(ORIGIN), author: "agent", verdict: "apply" }),
    ).toEqual([{ origin: ORIGIN, dimension: "cue", state: "applied" }]);
  });

  it("reports a held command as offered, and never as applied", () => {
    // The one thing the model must not be allowed to believe. `offered` is what
    // stops it writing "I've pulled that up on the map" over a view that never
    // moved.
    expect(
      verdictsForDispatch({
        previous: undefined,
        command: agentDestination(ORIGIN),
        author: "agent",
        verdict: "offer",
      }),
    ).toEqual([{ origin: ORIGIN, dimension: "destination", state: "offered" }]);
  });

  it("reports the person accepting an offer as followed, not as a fresh apply", () => {
    // `follow()` re-dispatches the model's command AS THE USER, because the
    // decision to apply it is theirs. Reported as `applied` it would read as the
    // model having got its way, erasing the person's act entirely.
    const offer = agentDestination(ORIGIN);
    expect(
      verdictsForDispatch({ previous: offer, command: offer, author: "user", verdict: "apply" }),
    ).toEqual([{ origin: ORIGIN, dimension: "destination", state: "followed" }]);
  });

  it("does not report the accepted offer twice — followed replaces applied, it does not precede it", () => {
    const offer = agentDestination(ORIGIN);
    const verdicts = verdictsForDispatch({
      previous: offer,
      command: offer,
      author: "user",
      verdict: "apply",
    });
    expect(verdicts).toHaveLength(1);
    expect(verdicts.map((v) => v.state)).not.toContain("superseded");
  });

  it("closes an older offer as superseded when the model asks again on the same axis", () => {
    // Without this the first ask hangs open on the seat's side forever, and the
    // trail shows a question that was never answered when in fact its answer was
    // "the model changed its mind".
    expect(
      verdictsForDispatch({
        previous: agentDestination("call-0"),
        command: agentDestination("call-1"),
        author: "agent",
        verdict: "offer",
      }),
    ).toEqual([
      { origin: "call-0", dimension: "destination", state: "superseded" },
      { origin: "call-1", dimension: "destination", state: "offered" },
    ]);
  });

  it("supersedes an older offer that a NEW command applies over", () => {
    expect(
      verdictsForDispatch({
        previous: agentCue("call-0"),
        command: agentCue("call-1"),
        author: "agent",
        verdict: "apply",
      }),
    ).toEqual([
      { origin: "call-0", dimension: "cue", state: "superseded" },
      { origin: "call-1", dimension: "cue", state: "applied" },
    ]);
  });

  it("says nothing at all about a command nobody asked for", () => {
    // Recall-derived commands carry no origin: the model never requested them,
    // so there is no ask to answer and inventing a recipient would put a verdict
    // in the trail for a request that was never made.
    expect(
      verdictsForDispatch({
        previous: undefined,
        command: agentCue(undefined),
        author: "agent",
        verdict: "apply",
      }),
    ).toEqual([]);
  });

  it("still closes a real offer that an origin-less command displaces", () => {
    // The displaced ask is real even though the thing displacing it is not.
    expect(
      verdictsForDispatch({
        previous: agentCue("call-0"),
        command: agentCue(undefined),
        author: "agent",
        verdict: "apply",
      }),
    ).toEqual([{ origin: "call-0", dimension: "cue", state: "superseded" }]);
  });
});

describe("verdictsForEndedOffers", () => {
  it("answers every waiting offer, one verdict each", () => {
    expect(
      verdictsForEndedOffers({ cue: agentCue("a"), destination: agentDestination("b") }, "declined"),
    ).toEqual([
      { origin: "a", dimension: "cue", state: "declined" },
      { origin: "b", dimension: "destination", state: "declined" },
    ]);
  });

  it("carries the caller's state, because only the caller knows why it ended", () => {
    // A turn boundary and a Dismiss both empty the same map. "They said no" and
    // "they never saw it" are different facts about a person and this is the
    // only place the difference still exists.
    expect(verdictsForEndedOffers({ cue: agentCue("a") }, "expired")[0].state).toBe("expired");
    expect(verdictsForEndedOffers({ cue: agentCue("a") }, "declined")[0].state).toBe("declined");
  });

  it("skips offers with no origin and reports nothing for an empty set", () => {
    expect(verdictsForEndedOffers({ cue: agentCue(undefined) }, "expired")).toEqual([]);
    expect(verdictsForEndedOffers({}, "expired")).toEqual([]);
  });
});
