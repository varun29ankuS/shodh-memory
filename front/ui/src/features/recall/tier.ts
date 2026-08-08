import type { CSSProperties } from "react";

/**
 * Memory consolidation tiers, and how a memory node wears one.
 *
 * WHAT THE WIRE CARRIES. `RecallMemory.tier` is a plain `String`
 * (src/handlers/types.rs:249, no `skip_serializing_if`), populated as
 * `format!("{:?}", m.tier)` at every construction site of that struct —
 * src/handlers/recall.rs:830, :3172 and :3581. So the values are the *Debug*
 * renderings of `MemoryTier` (src/memory/types.rs:1047-1076): `Working`,
 * `Session`, `LongTerm`, and the retired `Archive`.
 *
 * This is a genuine progression, not a set of kinds. A memory starts in
 * `Working` — immediate context, dense and volatile — is promoted to `Session`
 * once it earns it, and reaches `LongTerm`, the terminal tier, only after the
 * age and importance thresholds are met (the ladder is documented at
 * src/memory/mod.rs:6226-6233). "How settled is this?" is therefore an ordinal
 * question, and the encoding below answers it as one.
 *
 * `Archive` is deliberately NOT given a step. src/memory/types.rs:1056-1075 is
 * explicit that nothing assigns it and nothing should — the variant survives
 * only because `MemoryTier` is positionally encoded inside `MemoryFlat` and
 * deleting discriminant 3 would bet live user data on a negative-existence
 * proof. Drawing a fourth step for a tier the server cannot emit would put a
 * key on screen that no memory can ever match.
 */

/** The three tiers the ladder actually produces. */
export type MemoryTier = "Working" | "Session" | "LongTerm";

/** Least- to most-consolidated. The order the legend reads in, and the order
 *  the visual ramp climbs. */
export const MEMORY_TIER_ORDER: MemoryTier[] = ["Working", "Session", "LongTerm"];

/**
 * Normalise the wire string.
 *
 * Anything unrecognised — a future variant, or the inert `Archive` — falls back
 * to `Working`, the LEAST consolidated step. That direction is the honest one:
 * the encoding is a claim about how durable a memory is, and understating
 * durability for a value this client does not understand is safe, whereas
 * drawing an unknown tier as `LongTerm` would assert consolidation the client
 * cannot vouch for. It mirrors what the canvas already does with an absent
 * `memory_type`, which falls back to the muted hue rather than guessing a
 * category (GraphCanvas.tsx).
 */
export function memoryTier(raw: string | null | undefined): MemoryTier {
  return raw === "LongTerm" || raw === "Session" ? raw : "Working";
}

/**
 * Legend and tooltip copy.
 *
 * Plain English, not the enum. `LongTerm` is how Rust spells an identifier;
 * "Long-term" is how a person reads one, and an analyst surface should not make
 * someone learn a variant name to read a chart. Mirrors `TIER_LABEL` in
 * features/graph/universe.ts, which does the same job for edge tiers.
 */
export const MEMORY_TIER_LABEL: Record<MemoryTier, string> = {
  Working: "Working",
  Session: "Session",
  LongTerm: "Long-term",
};

/**
 * One-line explanation of what a tier means.
 *
 * For the Inspector, not the hover tooltip. The canvas shows one memory's
 * tooltip per pointer position and a repeat user reads that dozens of times a
 * session, so a sentence they learned on day one becomes noise there; the
 * Inspector shows a single selected memory deliberately, and has the room.
 */
export const MEMORY_TIER_MEANING: Record<MemoryTier, string> = {
  Working: "immediate context, not yet consolidated",
  Session: "promoted out of working memory",
  LongTerm: "consolidated and durable",
};

/**
 * The visual step for each tier.
 *
 * WHY NOT COLOUR. Edge tiers get their own hue ramp (`--edge-l1/l2/l3`,
 * index.css:153-171) because nothing else on the canvas claims edge colour.
 * Memory nodes have no such freedom: node hue is already `memory_type`
 * (GraphCanvas.tsx `readTokens`), and the rule the graph legend states outright
 * — categorical colour belongs to nodes, the tier ramp is a progression and the
 * two never share a palette (features/graph/GraphView.tsx:215-219) — forbids
 * spending hue twice on one mark. Tier therefore takes the one channel still
 * free on a node: how PRESENT it is.
 *
 * The ramp direction follows the edge ramp's, for the same stated reason: the
 * ground is near-black (#08090a), so on this canvas "settled and solid" means
 * more present, not darker. A working memory is a faint outline; a long-term
 * one is filled and firmly ringed. That reads at a glance, before anyone looks
 * at a legend, which is the test this encoding has to pass.
 *
 * `Session` is deliberately today's appearance (fill 0.28, ring 1.3px): the
 * middle of a new ramp should be the look the canvas already had, so this adds
 * a distinction rather than restyling every node.
 *
 * Channels NOT used, because they are already spoken for on this canvas:
 * radius (retrieval score), hue (memory_type), a dashed outer ring (a memory
 * nothing in the result set connects to), stroke colour and globalAlpha
 * (selection and focus dimming).
 */
export interface MemoryTierMark {
  /** Alpha for the node's fill, over its `memory_type` hue. */
  fill: number;
  /** Ring width in CSS pixels, before the zoom transform divides it. */
  ring: number;
  /** Alpha for the ring, over the same hue. */
  ringAlpha: number;
}

export const MEMORY_TIER_MARK: Record<MemoryTier, MemoryTierMark> = {
  Working: { fill: 0.11, ring: 1.0, ringAlpha: 0.55 },
  Session: { fill: 0.28, ring: 1.3, ringAlpha: 0.8 },
  LongTerm: { fill: 0.5, ring: 2.2, ringAlpha: 1.0 },
};

/** Selection adds to the tier's own step rather than replacing it, so a
 *  selected node still says which tier it is. The accent stroke colour is what
 *  makes selection unmistakable; these only stop the selected node from being
 *  the faintest thing on screen when it happens to be a working memory. */
export const MEMORY_TIER_SELECTED_FILL = 0.2;
export const MEMORY_TIER_SELECTED_RING = 1.2;

/**
 * The legend swatch, as CSS.
 *
 * Drawn in `--muted-foreground` rather than any data hue, because that is
 * exactly what the encoding claims: tier is not a colour here, it is weight and
 * presence, and a coloured key would imply a hue mapping that the canvas does
 * not have. A ring, not a bar — the graph legend uses a bar for edge tiers
 * because edges are lines (features/graph/GraphView.tsx:240-246); these describe
 * nodes, so they are round.
 */
export function memoryTierSwatch(tier: MemoryTier): CSSProperties {
  const mark = MEMORY_TIER_MARK[tier];
  return {
    borderWidth: `${mark.ring}px`,
    borderStyle: "solid",
    borderColor: `color-mix(in srgb, var(--muted-foreground) ${Math.round(
      mark.ringAlpha * 100,
    )}%, transparent)`,
    background: `color-mix(in srgb, var(--muted-foreground) ${Math.round(
      mark.fill * 100,
    )}%, transparent)`,
  };
}
