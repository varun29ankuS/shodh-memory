import { useMemo } from "react";
import { cn } from "@/lib/utils";
import type { Reachability } from "@/lib/api";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { RecallDiagram } from "./RecallDiagram";
import { GraphCanvas, useMemoryTypes } from "./GraphCanvas";
import { useRecall } from "./useRecall";
import {
  memoryTier,
  MEMORY_TIER_LABEL,
  MEMORY_TIER_ORDER,
  memoryTierSwatch,
  type MemoryTier,
} from "./tier";

/**
 * The graph stage.
 *
 * Two states, and which one shows is decided by whether there is anything to
 * draw rather than by whether the backend is up. With no results the stage
 * carries RecallDiagram — an explanation of how recall works, which is a claim
 * about the product in general. Once a recall returns memories the canvas
 * mounts and the label above it ("How this connects") becomes true of the data
 * on screen, which is a different and much stronger claim. A label describing
 * absent content is worse than no label, so both it and the legend render with
 * the canvas and not before.
 *
 * Connection state is not repeated here. It is stated once, in the status strip
 * in the top bar, which is visible from every destination and carries the
 * remedy with it.
 */

export function GraphStage({ reach, explain = false }: { reach: Reachability; explain?: boolean }) {
  const { data } = useRecall(reach);
  const memories = data?.memories ?? [];
  const lineage = data?.lineage ?? [];
  const types = useMemoryTypes(memories);
  const hasGraph = memories.length > 0;

  const edgeCount = lineage.length;

  /** Tier populations over the memories actually drawn. Counted here from the
   *  same `memoryTier` normaliser the canvas uses, so the key and the picture
   *  cannot disagree about which step a node is on. */
  const tierCounts = useMemo(() => {
    const counts: Record<MemoryTier, number> = { Working: 0, Session: 0, LongTerm: 0 };
    for (const m of memories) counts[memoryTier(m.tier)] += 1;
    return counts;
  }, [memories]);

  return (
    // `min-w-0`: a flex item's default min-width is its content's intrinsic
    // size, which can silently refuse to shrink below that even when the
    // sibling result column has already claimed the rest of `main`'s box at
    // a narrow viewport. `overflow-hidden` alone doesn't fix that — it only
    // hides what min-width still won't let the box shrink past.
    <section className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
      {/* The floor. Same ruling the canvas is plotted against, so the
          empty state and the populated one are the same surface. */}
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      {hasGraph ? (
        <>
          <GraphCanvas memories={memories} lineage={lineage} />
          <div className="text-muted-foreground pointer-events-none absolute top-3 left-4 z-10 text-[12px]">
            How this connects
          </div>
        </>
      ) : explain ? (
        /* The explainer is now ASKED FOR, not assumed.

           It used to fill this stage whenever a recall had not run — which is
           the resting state of the screen, so on a profile holding 245 real
           memories roughly 60% of the surface was a cartoon of a search box
           and three fake result bars, while the memories themselves were in a
           340px column truncating mid-word. An explainer is read once; it had
           permanent tenancy on the largest surface in the product.

           It is good work and it states the product's claim more clearly than
           anything else here, so it is one control away rather than deleted.

           `hidden md:flex`, not always-mounted: below ~768px viewport width
           the Inspector and result column's reserved widths (see
           RecallView.tsx / Inspector.tsx) leave this stage under ~135px wide.
           The diagram's own "Docked" explanation text has no minimum-width
           floor, so at that width it wraps character-by-character and its
           vertical overflow escapes past the 92px box it's docked to (the
           ancestor `overflow-hidden` bounds the whole section, not that box).
           RecallDiagram.tsx is explicitly not to be restructured, so the fix
           is here: don't hand it a width it cannot render into. */
        <div className="absolute inset-0 hidden flex-col items-center justify-center px-8 md:flex">
          <RecallDiagram />
        </div>
      ) : null}

      {/* The legend decodes the canvas, so it renders only when there is a
          canvas to decode — a colour key over an empty stage explains nothing.

          Legend and hints share one row with a gap that cannot collide. An
          earlier layout overlapped them below ~1500px, truncating the hint to
          "ICK A CLUSTER TO DRILL IN" on a 1440 laptop. */}
      {hasGraph ? (
        <div
          className={cn(
            "pointer-events-none absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center",
            "justify-between gap-x-6 gap-y-2",
          )}
          style={{ paddingRight: "var(--overlay-dock-inset, 0px)" }}
        >
          {/* Two encodings, two rows. Node HUE is memory type; node PRESENCE —
              fill and ring weight — is consolidation tier. They are kept on
              separate palettes on purpose: categorical colour belongs to the
              type, and the tier is a progression, so it climbs in weight rather
              than changing hue. The same split governs the entity graph
              (features/graph/GraphView.tsx:215-219). */}
          <div className="flex flex-col gap-1.5">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
              {/* The legend is the memory types actually present, in the order
                  the canvas assigns hues in. The server's set is closed — 14
                  Debug-rendered enum variants — but a fixed legend would name
                  eleven categories this result set does not contain, and 14
                  categories cannot map onto 5 chart hues without arbitrary
                  collisions. Listing what is present keeps the key honest. */}
              {types.map((t, i) => (
                <span
                  key={t}
                  className="text-muted-foreground flex items-center gap-1.5 text-[11px]"
                >
                  <span
                    className="size-2 rounded-full"
                    style={{ background: `var(--chart-${(i % 5) + 1})` }}
                  />
                  {t}
                </span>
              ))}
            </div>

            {/* All three tiers, always, counts included — unlike the memory-type
                row above. There are only three and they are a fixed ladder, so
                a zero beside "Long-term" is a finding about this result set
                (nothing here has consolidated yet) rather than a category the
                legend had no business naming. */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
              {/* This row is named and the one above is not, because the one
                  above needs no name: coloured dots beside a graph read as
                  categories on sight. Three rings labelled Working / Session /
                  Long-term do not say what dimension they are three steps OF
                  until someone is told, and a legend that has to be puzzled out
                  has already failed. */}
              <span className="text-muted-foreground/50 text-[11px]">Consolidation</span>
              {MEMORY_TIER_ORDER.map((t) => (
                <span
                  key={t}
                  className="text-muted-foreground flex items-center gap-1.5 text-[11px]"
                >
                  {/* A ring, not a bar: these describe nodes, and matching the
                      mark to what it describes is what makes the key readable
                      beside the type dots above. Neutral, because tier is not a
                      hue on this canvas — it is how filled-in a node is. 12px
                      rather than the type dots' 8px, because this swatch encodes
                      ring WEIGHT and a 2.2px ring on an 8px circle leaves too
                      little middle for the fill step to read. */}
                  <span className="size-3 rounded-full" style={memoryTierSwatch(t)} />
                  {MEMORY_TIER_LABEL[t]}
                  <span className="text-muted-foreground/60">{tierCounts[t]}</span>
                </span>
              ))}
            </div>
          </div>
          {/* `shrink-0`: this row wraps, and without it the hint is squeezed
              into a narrow column instead of dropping to a line of its own —
              at the stage width /recall actually has (viewport minus rail,
              result column and Inspector) it was rendering as four stacked
              fragments. Refusing to shrink makes the wrap happen at the row
              level, which is what `flex-wrap` is here for. */}
          {/* WHAT IS DRAWN, AND NOTHING ABOUT THE MOUSE. Half of this strip was
              "scroll to zoom · drag to pan · click a node to inspect" — three
              gestures, identical on all three canvases in this product, printed
              under every session of every one of them. They are learned once,
              so they moved behind the icon. The count of what is on screen did
              not, because it is a finding.

              Say what the edges ARE, not just how to move the camera: zero
              causal edges across a result set is a fact about the corpus, not a
              broken canvas, and staying silent about it invites the opposite
              reading.

              "lineage edges", not "causal edges". Every edge here comes from
              the causal lineage graph, but only some classify as the bright
              causal class (Caused, TriggeredBy) — InformedBy and ResolvedBy
              draw cool. Calling all twelve "causal" while two thirds render in
              the typed colour reads as a bug in the legend. */}
          <Meta className="shrink-0">
            {edgeCount > 0 ? (
              <Stat value={edgeCount} label={`lineage ${edgeCount === 1 ? "edge" : "edges"}`} />
            ) : (
              <span>No lineage edges connect these results</span>
            )}
            <InfoHint label="canvas controls" align="right" side="up">
              Scroll to zoom, drag to pan, click a node to inspect it. Nodes are the memories this
              search returned; edges are the causal links the same response carried.
            </InfoHint>
          </Meta>
        </div>
      ) : null}
    </section>
  );
}
