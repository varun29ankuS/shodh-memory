import { ChevronLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import { spineText } from "./trail";

/**
 * A pane compressed to its spine.
 *
 * 40px, and that number is not a taste. Andy Matuschak's stacked notes and
 * Obsidian's stacked tabs arrived at 40px independently, which is the width
 * where a vertical line of body text is legible without becoming a second
 * column competing for attention.
 *
 * `writing-mode: vertical-lr`, NOT a rotate transform. A rotated span is a
 * picture of text: it does not wrap, it does not select, it does not search,
 * and it does not participate in layout, so it has to be positioned by hand
 * and re-positioned whenever anything around it changes. Vertical writing mode
 * is real text laid out vertically — selectable, findable, and sized by the
 * same rules as everything else.
 *
 * IT IS A BUTTON. Matuschak's spines are not clickable, which is the one real
 * defect in that implementation: the compressed thing is the most obvious
 * target on the screen for "go back", and it does nothing. Here it is the
 * primary way back, so it is a real control with a real accessible name and a
 * focus ring, reachable by tab.
 *
 * FULL OR SPINE, NEVER PARTIALLY OCCLUDED. The other defect worth avoiding is
 * a pane sliced mid-glyph by the one on top of it, which reads as a rendering
 * fault rather than as a stack. A compressed pane renders THIS and nothing
 * else — the view it held is unmounted, not clipped — and the title ellipses
 * along the vertical axis if it is ever taller than the screen, so a word is
 * never cut through.
 *
 * NOTHING HERE MOVES ON HOVER. The chevron and the title occupy their final
 * positions at rest; hover and focus change colour and the backing surface
 * only. A spine that grew or slid on approach would move the target while it
 * was being aimed at.
 *
 * IT SAYS WHAT IT DOES, IN WORDS, BECAUSE THE GLYPH ALONE DID NOT.
 *
 * This read `1`, a chevron, and `Briefing`, as three items spaced 10px apart
 * down a 40px column, and two people — including the person whose product it
 * is — could not work out what it was for. That is the same defect as a lane
 * row whose name, state and figure float apart: fragments near each other are
 * not an object, and a reader given a bare number, an arrow and a noun will
 * assemble a meaning out of them or give up, and giving up is the common case.
 *
 * The remedies, in the order they matter:
 *
 *   1. THE TEXT IS THE SENTENCE. `Back to Briefing`, not `Briefing`. A
 *      destination on its own is a label; a label plus a verb is a control.
 *      This is now the same string the accessible name has always carried, so
 *      what is seen and what is spoken agree instead of the sighted reader
 *      getting the poorer half.
 *   2. THE CHEVRON JOINS THE TEXT. It sits directly against the words at a
 *      1.5-unit gap rather than floating 10px clear of them, so the pair reads
 *      as one arrow-and-phrase and not as two marks that happen to be stacked.
 *   3. THE ORDINAL EARNS ITS PLACE OR IS NOT DRAWN. With one spine on screen
 *      it is a lone `1` above an unrelated word, which is the single most
 *      confusing thing in the column and says nothing a reader did not know.
 *      It is drawn only from the second spine onward, where a stack genuinely
 *      has an order to state.
 *
 * WHAT WAS NOT DONE, AND WHY. The alternative was to hide the spine until the
 * trail has depth. There is no such state: the trail is `[briefing, …primary]`,
 * so every screen in the product that is not the briefing has exactly one
 * spine or more, and hiding it at one would delete the way back from every
 * screen that has one. The 40px is not rent the spine has to justify against
 * an empty alternative — it is the only thing on screen that says where you
 * came from.
 */

/** The spine's width, in px. Read by the workbench for its flex basis, so the
 *  layout and the control cannot disagree. */
export const SPINE_WIDTH_PX = 40;

export function Spine({
  title,
  caption,
  ordinal,
  onOpen,
}: {
  title: string;
  caption: string;
  /** 1-based position in the trail, or null where there is only one spine and
   *  a position is not a fact worth stating. Drawn, and `aria-hidden`: the
   *  accessible name states the destination in words, which is a better answer
   *  to "which of these is which" than a digit read aloud. */
  ordinal: number | null;
  onOpen: () => void;
}) {
  const { visible, accessible } = spineText(title, caption);

  return (
    <button
      type="button"
      onClick={onOpen}
      // States the destination and what is there. A control called "Briefing"
      // tells a screen reader which button it is, not what pressing it does.
      aria-label={accessible}
      title={visible}
      style={{ width: `${SPINE_WIDTH_PX}px` }}
      className={cn(
        // Pinned to the LEFT EDGE of its pane rather than filling it. While a
        // pane is compressing, its box shrinks from full width to 40px and its
        // left edge does not move — so anchoring the spine there means the
        // title is stationary for the whole transition instead of drifting in
        // from the middle. The eye's anchor is kept by not moving it.
        // gap-1.5, not gap-2.5: the chevron and the phrase are one control and
        // are spaced as one. The ordinal, when there is one, buys its own
        // separation back with a margin rather than pushing the pair apart.
        "group absolute inset-y-0 left-0 z-10 flex cursor-pointer flex-col items-center gap-1.5 py-3",
        "border-border bg-background border-r",
        "hover:bg-accent focus-visible:bg-accent transition-colors duration-100",
        // An INSET focus ring, and that is a correctness fix rather than a
        // preference: a spine fills its pane exactly, the pane clips its
        // contents so a compressing view can never spill, and a `ring` is a
        // spread box-shadow drawn OUTSIDE the border box — so an outset ring
        // is clipped away entirely and the control looks unfocusable. Verified
        // in the browser, where the outset version drew nothing at all.
        "focus-visible:shadow-[inset_0_0_0_2px_var(--ring)] focus-visible:outline-none",
      )}
    >
      {ordinal === null ? null : (
        <span
          aria-hidden="true"
          className="mono text-muted-foreground/70 mb-1 text-[10px] leading-none tabular-nums"
        >
          {ordinal}
        </span>
      )}
      <ChevronLeft
        aria-hidden="true"
        className="text-muted-foreground group-hover:text-primary group-focus-visible:text-primary size-3.5 shrink-0 transition-colors duration-100"
        strokeWidth={1.7}
      />
      <span
        className={cn(
          "text-muted-foreground group-hover:text-foreground group-focus-visible:text-foreground",
          "min-h-0 text-[13px] font-[450] tracking-normal transition-colors duration-100",
        )}
        style={{
          writingMode: "vertical-lr",
          // Ellipsis rather than a clipped glyph if a title ever outruns the
          // height. In vertical writing mode the inline axis is the vertical
          // one, so both of these act along the column.
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {visible}
      </span>
    </button>
  );
}
