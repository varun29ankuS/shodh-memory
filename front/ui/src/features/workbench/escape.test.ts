import { describe, expect, it } from "vitest";
import { escapeVerdict, trailActs, type EscapeContext } from "./escape";

/** Nothing open, nothing selected, nowhere to go. */
const quiet: EscapeContext = {
  editable: false,
  transientOpen: false,
  conversationExpanded: false,
  hasSelection: false,
  canGoBack: false,
};

const ctx = (over: Partial<EscapeContext>): EscapeContext => ({ ...quiet, ...over });

describe("escapeVerdict", () => {
  it("goes back one level when nothing more local wants the key", () => {
    expect(escapeVerdict(ctx({ canGoBack: true }))).toBe("back");
  });

  it("does nothing at the base of the trail", () => {
    expect(escapeVerdict(quiet)).toBe("none");
  });

  it("closes the detail before it pops a pane", () => {
    // A selection IS a level. Popping the pane it was selected in would skip
    // one, and would take away the thing the person was reading.
    expect(escapeVerdict(ctx({ hasSelection: true, canGoBack: true }))).toBe(
      "clear-selection",
    );
  });

  it("leaves a text field's Escape to the text field", () => {
    // The cue field abandons what was typed; a rename abandons the rename.
    // Neither survives the trail also popping a pane on the same keypress.
    expect(
      escapeVerdict(ctx({ editable: true, hasSelection: true, canGoBack: true })),
    ).toBe("editable");
  });

  it("yields to an open transient surface", () => {
    expect(
      escapeVerdict(ctx({ transientOpen: true, hasSelection: true, canGoBack: true })),
    ).toBe("transient");
    expect(
      escapeVerdict(
        ctx({ conversationExpanded: true, hasSelection: true, canGoBack: true }),
      ),
    ).toBe("transient");
  });

  it("orders the whole ladder: field, then overlay, then detail, then trail", () => {
    const table: [Partial<EscapeContext>, string][] = [
      [{ editable: true, transientOpen: true, conversationExpanded: true, hasSelection: true, canGoBack: true }, "editable"],
      [{ transientOpen: true, conversationExpanded: true, hasSelection: true, canGoBack: true }, "transient"],
      [{ conversationExpanded: true, hasSelection: true, canGoBack: true }, "transient"],
      [{ hasSelection: true, canGoBack: true }, "clear-selection"],
      [{ canGoBack: true }, "back"],
      [{}, "none"],
    ];
    for (const [over, expected] of table) {
      expect(escapeVerdict(ctx(over))).toBe(expected);
    }
  });

  it("still clears a selection when the trail has nowhere to go", () => {
    // Escape on the briefing with a memory open closes the memory. The trail
    // having no level left does not mean nothing does.
    expect(escapeVerdict(ctx({ hasSelection: true }))).toBe("clear-selection");
  });
});

describe("trailActs", () => {
  it("is true exactly for the verdicts the workbench carries out", () => {
    expect(trailActs("clear-selection")).toBe(true);
    expect(trailActs("back")).toBe(true);
    expect(trailActs("editable")).toBe(false);
    expect(trailActs("transient")).toBe(false);
    expect(trailActs("none")).toBe(false);
  });
});
