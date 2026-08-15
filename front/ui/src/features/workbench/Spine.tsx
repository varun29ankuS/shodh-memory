import { ChevronLeft } from "lucide-react";
import { cn } from "@/lib/utils";

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
  /** 1-based position in the trail, spoken to screen readers so a stack of
   *  spines has an order rather than being four buttons named "back to". */
  ordinal: number;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      // States the destination and what is there. A control called "Briefing"
      // tells a screen reader which button it is, not what pressing it does.
      aria-label={`Back to ${title} — ${caption}`}
      title={`Back to ${title}`}
      style={{ width: `${SPINE_WIDTH_PX}px` }}
      className={cn(
        // Pinned to the LEFT EDGE of its pane rather than filling it. While a
        // pane is compressing, its box shrinks from full width to 40px and its
        // left edge does not move — so anchoring the spine there means the
        // title is stationary for the whole transition instead of drifting in
        // from the middle. The eye's anchor is kept by not moving it.
        "group absolute inset-y-0 left-0 z-10 flex cursor-pointer flex-col items-center gap-2.5 py-3",
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
      <span
        aria-hidden="true"
        className="mono text-muted-foreground text-[10px] leading-none tabular-nums"
      >
        {ordinal}
      </span>
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
        {title}
      </span>
    </button>
  );
}
