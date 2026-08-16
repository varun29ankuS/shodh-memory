import { describe, expect, it } from "vitest";
import { MIN_TERM_CHARS, cueMatches } from "./cue";

describe("cueMatches", () => {
  it("keeps the typed cue a substring test, so the ring follows the keystroke", () => {
    expect(cueMatches("Port of Baltimore", "ba", [])).toBe(true);
    expect(cueMatches("Maersk", "ba", [])).toBe(false);
  });

  it("is case-insensitive on both sides", () => {
    expect(cueMatches("MAERSK", "maersk", [])).toBe(true);
    expect(cueMatches("Maersk", "", ["MAERSK"])).toBe(true);
  });

  it("matches a term inside a name — the extractor emits fragments, not ids", () => {
    expect(cueMatches("Port of Baltimore", "", ["Baltimore"])).toBe(true);
  });

  it("matches a name inside a term, which exact matching would miss entirely", () => {
    expect(cueMatches("Maersk", "", ["Maersk Line"])).toBe(true);
  });

  it("ignores terms too short to narrow anything", () => {
    const short = "x".repeat(MIN_TERM_CHARS - 1);
    expect(cueMatches(`${short}enon`, "", [short])).toBe(false);
  });

  it("ignores short NAMES too, so a two-letter entity is not matched by everything", () => {
    // The containment test runs both ways; without a floor on the name side,
    // "US" is inside "USS Dali", "customs", "surplus" and most of a corpus.
    expect(cueMatches("US", "", ["customs"])).toBe(false);
  });

  it("matches nothing when the cue is empty, so a cleared field shows the corpus", () => {
    expect(cueMatches("Port of Baltimore", "", [])).toBe(false);
  });

  it("lights a node the model recalled even when the query text does not name it", () => {
    // The whole point of the second channel: the model asked "what happened at
    // the bridge", and the facts it got back name the ship.
    expect(cueMatches("Dali", "what happened at the bridge", ["Dali", "Patapsco"])).toBe(true);
  });
});
