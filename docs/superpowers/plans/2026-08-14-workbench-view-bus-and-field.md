# Workbench View Bus and Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the workbench a single view bus that both the user and the agent drive, so that asking a question visibly narrows the field to what the model actually retrieved, and a conversation started on one destination stays visible on every other.

**Architecture:** All view state moves into one zustand store with a single `dispatch(command, author)` entry point. Commands are named and serializable. Two producers write to it — human interaction, and one adapter that translates the `SeatEvent`s already streaming from the seat. The reducer, the event translation and the viewport fit are pure modules so they are testable with the tooling already on `main`; React components stay thin wrappers over them.

**Tech Stack:** React 19, TypeScript, zustand, d3 (`d3-zoom`, `d3-force`), vitest 4, Tailwind v4.

## Global Constraints

- **Ships as exactly one self-contained `dist/index.html`.** `vite-plugin-singlefile` inlines every chunk; the shodh-front Rust binary embeds it with `include_str!`. No sibling `.js`/`.css`, no hashed assets.
- **Fully offline. No network at runtime.** No CDN fonts, no tile servers, no remote images. Runtime dependencies added in this plan: **none**. Dev-only devDependencies are permitted (they do not enter the bundle).
- **Dark-only.** No theme switch — the product has one visual world (`front/ui/DIRECTION.md`).
- **One accent, `#f4622e`.** It marks focus, the primary action, and active nav. Never anomaly — anomaly is `--destructive` (`#f2555a`) and `--warn` (`#e5b567`). Chrome never uses a data hue and data never uses the accent, except `--node-active`.
- **Every colour comes from a token in `src/index.css`.** No raw hex in `.tsx`/`.ts` outside the five canvas files that already carry them.
- **`prefers-reduced-motion` collapses every transition to an instant state change.** Both canvases already branch on it (`features/graph/EntityCanvas.tsx:260`, `features/recall/GraphCanvas.tsx:258`).
- **`aria-label` on every control**, labels in the DOM at all times — never bare icons for a screen reader.
- **No `TODO`, no placeholder, no mock, no stub.** Production-grade only.
- **Every test must be shown to fail before the code that makes it pass.** A test that cannot fail is invisible to a failing-test sweep.
- **Commit messages carry no attribution footers.**

## Scope

This plan implements spec §3 (the field's computed rest state), §4 (the view bus), §5.2 (implicit agent sync) and §7 (the conversation dock).

**Deliberately not in this plan**, because each is independently useful and would double the length: spec §8 (the history surface), §8.4 (the ledger `actor` field), §5.3 (explicit seat view tools). Those get a second plan once this one is working — the view bus is their prerequisite, and the trail this plan builds is the data the history surface reads.

## File Structure

**Create:**
- `src/lib/view/commands.ts` — the command vocabulary, the dimension mapping, and validation. Pure. No React, no zustand.
- `src/lib/view/commands.test.ts`
- `src/lib/view/fit.ts` — extent of a point set, and the transform that frames it. Pure maths, no d3 import.
- `src/lib/view/fit.test.ts`
- `src/lib/view/fromSeatEvents.ts` — `ChatOp` → `ViewCommand[]`. Pure.
- `src/lib/view/fromSeatEvents.test.ts`
- `src/stores/view.ts` — the store, the authority rule, the trail.
- `src/stores/view.test.ts`
- `src/app/useViewSync.ts` — the single adapter, mounted once.
- `src/test/setup.ts` — testing-library cleanup (Task 6 only).
- `src/features/chat/ConversationDock.test.tsx` (Task 6 only).

**Modify:**
- `src/app/App.tsx` — mount `useViewSync()` in `Shell`; keep `ConversationOverlay` mounted on every route.
- `src/features/recall/GraphCanvas.tsx` — fit on load; read `frame` from the store.
- `src/features/chat/ConversationOverlay.tsx:154-158` — always present, collapse instead of dismiss.
- `vite.config.ts` — add the `test` block (Task 6 only).
- `package.json` — add `jsdom`, `@testing-library/react`, `@testing-library/dom` as devDependencies (Task 6 only).

Why the split: `commands.ts`, `fit.ts` and `fromSeatEvents.ts` are pure and each holds one responsibility, so they are unit-testable today with vitest's default node environment. Only Task 6 needs a DOM, and it installs what it needs.

---

### Task 1: The command vocabulary

**Files:**
- Create: `front/ui/src/lib/view/commands.ts`
- Test: `front/ui/src/lib/view/commands.test.ts`

**Interfaces:**
- Consumes: `DestinationId` from `@/components/layout/Sidebar` (exported at `Sidebar.tsx:137`).
- Produces: `ViewCommand`, `Author`, `Dimension`, `dimensionOf(command): Dimension`, `validateCommand(value: unknown): ViewCommand | { error: string }`.

- [ ] **Step 1: Write the failing test**

Create `front/ui/src/lib/view/commands.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { dimensionOf, validateCommand } from "./commands";

describe("dimensionOf", () => {
  it("maps each command kind to the dimension it occupies", () => {
    expect(dimensionOf({ kind: "open", view: "/graph" })).toBe("destination");
    expect(dimensionOf({ kind: "cue", text: "maersk" })).toBe("cue");
    expect(dimensionOf({ kind: "frame", ids: ["a"] })).toBe("frame");
    expect(dimensionOf({ kind: "focus", id: "a", of: "memory" })).toBe("focus");
    expect(dimensionOf({ kind: "filter", patch: {} })).toBe("filters");
  });
});

describe("validateCommand", () => {
  it("accepts a well-formed open command", () => {
    expect(validateCommand({ kind: "open", view: "/graph" })).toEqual({
      kind: "open",
      view: "/graph",
    });
  });

  it("accepts frame with the literal all", () => {
    expect(validateCommand({ kind: "frame", ids: "all" })).toEqual({
      kind: "frame",
      ids: "all",
    });
  });

  // The model produces these. Every one of these inputs is a real thing a
  // language model emits, and each must fail closed rather than reach the store.
  it.each([
    [null, "not an object"],
    ["open", "not an object"],
    [{}, "missing kind"],
    [{ kind: "teleport" }, "unknown kind"],
    [{ kind: "open" }, "open needs a view"],
    [{ kind: "open", view: "/nowhere" }, "unknown destination"],
    [{ kind: "cue" }, "cue needs text"],
    [{ kind: "cue", text: 42 }, "cue text must be a string"],
    [{ kind: "frame" }, "frame needs ids"],
    [{ kind: "frame", ids: [1, 2] }, "frame ids must be strings"],
    [{ kind: "focus", id: "a" }, "focus needs of"],
    [{ kind: "focus", id: "a", of: "planet" }, "focus of must be memory or entity"],
  ])("rejects %j", (input) => {
    const result = validateCommand(input);
    expect(result).toHaveProperty("error");
    expect(typeof (result as { error: string }).error).toBe("string");
  });

  it("does not mutate or pass through extra fields from the model", () => {
    const result = validateCommand({ kind: "cue", text: "x", danger: "drop table" });
    expect(result).toEqual({ kind: "cue", text: "x" });
  });
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `cd front/ui && npx vitest run src/lib/view/commands.test.ts`
Expected: FAIL — `Failed to resolve import "./commands"`.

- [ ] **Step 3: Write the implementation**

Create `front/ui/src/lib/view/commands.ts`:

```ts
import { DESTINATIONS } from "@/components/layout/Sidebar";

/**
 * The view command vocabulary.
 *
 * Every change to what is on screen is one of these, whoever caused it. That
 * is what lets a single store serve two producers — the user's hands and the
 * agent's events — without a second code path that can drift.
 *
 * Commands are serializable by requirement, not by accident: the trail they
 * produce is a record rather than a rendering, which is what the history
 * surface reads.
 */

/** A destination path, as listed in `DESTINATIONS`. */
export type ViewPath = (typeof DESTINATIONS)[number]["path"];

export interface Filters {
  /** Inclusive epoch-ms bounds, or null for unbounded. */
  since: number | null;
  until: number | null;
  /** Coarse entity classes to keep; empty means no type filter. */
  coarse: string[];
}

export type ViewCommand =
  | { kind: "open"; view: ViewPath }
  | { kind: "cue"; text: string }
  | { kind: "frame"; ids: string[] | "all" }
  | { kind: "focus"; id: string; of: "memory" | "entity" }
  | { kind: "filter"; patch: Partial<Filters> };

/** Who caused a command. The trail is worthless without this. */
export type Author = "human" | "model";

/**
 * The slice of view state a command occupies.
 *
 * The authority rule (stores/view.ts) is per-dimension: framing the field by
 * hand must not stop the agent opening a different destination, because those
 * are not the same act and blocking both would make the agent feel broken
 * rather than deferential.
 */
export type Dimension = "destination" | "cue" | "frame" | "focus" | "filters";

export function dimensionOf(command: ViewCommand): Dimension {
  switch (command.kind) {
    case "open":
      return "destination";
    case "cue":
      return "cue";
    case "frame":
      return "frame";
    case "focus":
      return "focus";
    case "filter":
      return "filters";
  }
}

const PATHS: ReadonlySet<string> = new Set(DESTINATIONS.map((d) => d.path));

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Parse an untrusted value into a command, failing closed.
 *
 * Explicit view-tool arguments arrive as `unknown` from a language model
 * (`lib/seat/types.ts:98` types `args` as `unknown`). Nothing reaches the store
 * without passing through here, and the returned object is rebuilt field by
 * field rather than spread, so a model cannot smuggle extra properties into
 * view state.
 */
export function validateCommand(value: unknown): ViewCommand | { error: string } {
  if (!isRecord(value)) return { error: "command must be an object" };
  const kind = value.kind;
  if (typeof kind !== "string") return { error: "command is missing a string kind" };

  switch (kind) {
    case "open": {
      const view = value.view;
      if (typeof view !== "string") return { error: "open requires a string view" };
      if (!PATHS.has(view)) return { error: `unknown destination ${JSON.stringify(view)}` };
      return { kind: "open", view: view as ViewPath };
    }
    case "cue": {
      const text = value.text;
      if (typeof text !== "string") return { error: "cue requires string text" };
      return { kind: "cue", text };
    }
    case "frame": {
      const ids = value.ids;
      if (ids === "all") return { kind: "frame", ids: "all" };
      if (!Array.isArray(ids)) return { error: 'frame requires an id array or "all"' };
      if (!ids.every((id): id is string => typeof id === "string")) {
        return { error: "frame ids must all be strings" };
      }
      return { kind: "frame", ids: [...ids] };
    }
    case "focus": {
      const id = value.id;
      const of = value.of;
      if (typeof id !== "string") return { error: "focus requires a string id" };
      if (of !== "memory" && of !== "entity") {
        return { error: 'focus "of" must be "memory" or "entity"' };
      }
      return { kind: "focus", id, of };
    }
    case "filter": {
      const patch = value.patch;
      if (!isRecord(patch)) return { error: "filter requires a patch object" };
      const next: Partial<Filters> = {};
      if ("since" in patch) {
        if (patch.since !== null && typeof patch.since !== "number") {
          return { error: "filter since must be a number or null" };
        }
        next.since = patch.since as number | null;
      }
      if ("until" in patch) {
        if (patch.until !== null && typeof patch.until !== "number") {
          return { error: "filter until must be a number or null" };
        }
        next.until = patch.until as number | null;
      }
      if ("coarse" in patch) {
        const coarse = patch.coarse;
        if (!Array.isArray(coarse) || !coarse.every((c) => typeof c === "string")) {
          return { error: "filter coarse must be an array of strings" };
        }
        next.coarse = [...coarse];
      }
      return { kind: "filter", patch: next };
    }
    default:
      return { error: `unknown command kind ${JSON.stringify(kind)}` };
  }
}

/** Narrow a validate result without repeating the shape test at call sites. */
export function isCommand(result: ViewCommand | { error: string }): result is ViewCommand {
  return !("error" in result);
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `cd front/ui && npx vitest run src/lib/view/commands.test.ts`
Expected: PASS, 4 test blocks (the `it.each` counts as 12).

- [ ] **Step 5: Typecheck**

Run: `cd front/ui && npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add front/ui/src/lib/view/commands.ts front/ui/src/lib/view/commands.test.ts
git commit -m "feat(ui): view command vocabulary with fail-closed validation

Every change to what is on screen becomes one named, serializable command,
so one store can serve both the user's hands and the agent's events without
a second code path. Untrusted values are rebuilt field by field rather than
spread, so a model cannot smuggle extra properties into view state."
```

---

### Task 2: Framing the field

**Files:**
- Create: `front/ui/src/lib/view/fit.ts`
- Test: `front/ui/src/lib/view/fit.test.ts`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Extent`, `extentOf(points): Extent | null`, `fitTransform(extent, viewport, opts): { k: number; x: number; y: number }`.

This is the spec's §3 claim that framing is the computed rest state. Both canvases
currently start at `zoomIdentity` (`features/recall/GraphCanvas.tsx:169`,
`features/graph/EntityCanvas.tsx:143`) and never compute a fit, which is the
long-standing "graph doesn't fit the viewport" defect.

- [ ] **Step 1: Write the failing test**

Create `front/ui/src/lib/view/fit.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { extentOf, fitTransform } from "./fit";

describe("extentOf", () => {
  it("returns null when no point has coordinates", () => {
    expect(extentOf([])).toBeNull();
    expect(extentOf([{}, { x: undefined, y: 3 }])).toBeNull();
  });

  it("ignores points missing either coordinate", () => {
    expect(extentOf([{ x: 1, y: 2 }, { x: 5 }, { y: 9 }])).toEqual({
      minX: 1,
      minY: 2,
      maxX: 1,
      maxY: 2,
    });
  });

  it("spans every complete point", () => {
    expect(extentOf([{ x: -4, y: 10 }, { x: 6, y: 2 }])).toEqual({
      minX: -4,
      minY: 2,
      maxX: 6,
      maxY: 10,
    });
  });
});

describe("fitTransform", () => {
  const apply = (t: { k: number; x: number; y: number }, p: { x: number; y: number }) => ({
    x: p.x * t.k + t.x,
    y: p.y * t.k + t.y,
  });

  it("places every node inside the viewport, at several aspect ratios", () => {
    const points = [
      { x: 0, y: 0 },
      { x: 400, y: 120 },
      { x: 90, y: 300 },
      { x: -50, y: -20 },
    ];
    const extent = extentOf(points)!;
    for (const viewport of [
      { width: 1600, height: 900 },
      { width: 600, height: 1000 },
      { width: 900, height: 900 },
    ]) {
      const t = fitTransform(extent, viewport, { padding: 24 });
      for (const p of points) {
        const s = apply(t, p);
        expect(s.x).toBeGreaterThanOrEqual(0);
        expect(s.x).toBeLessThanOrEqual(viewport.width);
        expect(s.y).toBeGreaterThanOrEqual(0);
        expect(s.y).toBeLessThanOrEqual(viewport.height);
      }
    }
  });

  it("centres the extent in the viewport", () => {
    const extent = { minX: 0, minY: 0, maxX: 100, maxY: 100 };
    const t = fitTransform(extent, { width: 800, height: 400 }, { padding: 0 });
    const centre = apply(t, { x: 50, y: 50 });
    expect(centre.x).toBeCloseTo(400, 6);
    expect(centre.y).toBeCloseTo(200, 6);
  });

  it("does not divide by zero on a single point", () => {
    const t = fitTransform({ minX: 7, minY: 7, maxX: 7, maxY: 7 }, { width: 500, height: 500 }, {});
    expect(Number.isFinite(t.k)).toBe(true);
    expect(Number.isFinite(t.x)).toBe(true);
    expect(Number.isFinite(t.y)).toBe(true);
    const centre = apply(t, { x: 7, y: 7 });
    expect(centre.x).toBeCloseTo(250, 6);
    expect(centre.y).toBeCloseTo(250, 6);
  });

  it("clamps into the canvas scaleExtent so d3 cannot reject the transform", () => {
    // GraphCanvas.tsx:422 uses scaleExtent([0.2, 6]). A tiny extent would
    // otherwise want a scale far above 6 and the first pan gesture would jump.
    const tight = fitTransform(
      { minX: 0, minY: 0, maxX: 1, maxY: 1 },
      { width: 1600, height: 900 },
      { padding: 24, scaleExtent: [0.2, 6] },
    );
    expect(tight.k).toBeLessThanOrEqual(6);

    const huge = fitTransform(
      { minX: 0, minY: 0, maxX: 500000, maxY: 500000 },
      { width: 800, height: 600 },
      { padding: 24, scaleExtent: [0.2, 6] },
    );
    expect(huge.k).toBeGreaterThanOrEqual(0.2);
  });

  it("degrades to a centred identity-scale view on a zero-sized viewport", () => {
    const t = fitTransform({ minX: 0, minY: 0, maxX: 10, maxY: 10 }, { width: 0, height: 0 }, {});
    expect(Number.isFinite(t.k)).toBe(true);
    expect(t.k).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `cd front/ui && npx vitest run src/lib/view/fit.test.ts`
Expected: FAIL — `Failed to resolve import "./fit"`.

- [ ] **Step 3: Write the implementation**

Create `front/ui/src/lib/view/fit.ts`:

```ts
/**
 * Framing a point set into a viewport.
 *
 * Kept free of d3 so it is testable in vitest's default node environment and
 * so both canvases can share it. The caller turns the result into a
 * `ZoomTransform` — see the note in GraphCanvas about applying it through the
 * zoom behaviour rather than assigning the ref.
 */

export interface Extent {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface Viewport {
  width: number;
  height: number;
}

export interface FitOptions {
  /** Breathing room in screen pixels on every side. */
  padding?: number;
  /** The canvas's own d3 scaleExtent, so the fit cannot produce a scale d3 will clamp. */
  scaleExtent?: [number, number];
}

/**
 * The bounding box of every point that has both coordinates.
 *
 * d3-force nodes are `SimulationNodeDatum`, whose `x` and `y` are optional and
 * genuinely absent before the first tick. A partial point is skipped rather
 * than coerced, because treating a missing coordinate as 0 drags the extent to
 * the origin and frames a box the nodes are not in.
 */
export function extentOf(points: readonly { x?: number; y?: number }[]): Extent | null {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let seen = false;

  for (const p of points) {
    if (typeof p.x !== "number" || typeof p.y !== "number") continue;
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    seen = true;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  return seen ? { minX, minY, maxX, maxY } : null;
}

/**
 * The transform that frames `extent` inside `viewport`.
 *
 * Screen position is `p * k + offset`, matching d3's `ZoomTransform.apply`.
 */
export function fitTransform(
  extent: Extent,
  viewport: Viewport,
  { padding = 0, scaleExtent }: FitOptions = {},
): { k: number; x: number; y: number } {
  const spanX = extent.maxX - extent.minX;
  const spanY = extent.maxY - extent.minY;

  // A viewport smaller than its own padding would ask for a negative or zero
  // available box and produce a non-finite or inverted scale. Clamp to a
  // positive floor: an unmeasured container (width 0 before layout) must
  // still yield a usable transform rather than NaN painted onto the canvas.
  const availableWidth = Math.max(viewport.width - padding * 2, 1);
  const availableHeight = Math.max(viewport.height - padding * 2, 1);

  const scaleX = spanX > 0 ? availableWidth / spanX : Number.POSITIVE_INFINITY;
  const scaleY = spanY > 0 ? availableHeight / spanY : Number.POSITIVE_INFINITY;

  // Both infinite means a single point (or a perfectly degenerate line in both
  // axes): there is no span to fit, so scale is meaningless and 1 is honest.
  let k = Math.min(scaleX, scaleY);
  if (!Number.isFinite(k) || k <= 0) k = 1;

  if (scaleExtent) {
    const [lo, hi] = scaleExtent;
    k = Math.min(Math.max(k, lo), hi);
  }

  const centreX = (extent.minX + extent.maxX) / 2;
  const centreY = (extent.minY + extent.maxY) / 2;

  return {
    k,
    x: viewport.width / 2 - k * centreX,
    y: viewport.height / 2 - k * centreY,
  };
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `cd front/ui && npx vitest run src/lib/view/fit.test.ts`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add front/ui/src/lib/view/fit.ts front/ui/src/lib/view/fit.test.ts
git commit -m "feat(ui): compute the transform that frames a point set

Framing is the field's rest state, so it is computed rather than defaulted.
Kept free of d3 so both canvases share it and it tests in the default node
environment. Guards the three ways this returns NaN in practice: a node with
no coordinates before the first force tick, a single-point extent, and a
container measured at zero width before layout."
```

---

### Task 3: The view store and the authority rule

**Files:**
- Create: `front/ui/src/stores/view.ts`
- Test: `front/ui/src/stores/view.test.ts`

**Interfaces:**
- Consumes: `ViewCommand`, `Author`, `Dimension`, `dimensionOf` from `@/lib/view/commands` (Task 1).
- Produces: `useView` store with state `{ destination, cue, frame, focus, filters, trail, offered, touched }` and actions `dispatch(command, author, at?)`, `beginTurn()`, `acceptOffer(id)`, `dismissOffer(id)`, plus the exported pure reducer `applyCommand(state, command)` and type `TrailEntry`.

- [ ] **Step 1: Write the failing test**

Create `front/ui/src/stores/view.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { useView } from "./view";

const reset = () =>
  useView.setState({
    destination: "/chat",
    cue: "",
    frame: "all",
    focus: null,
    filters: { since: null, until: null, coarse: [] },
    trail: [],
    offered: [],
    touched: [],
  });

describe("view store", () => {
  beforeEach(reset);

  it("applies a human command and records it in the trail", () => {
    useView.getState().dispatch({ kind: "open", view: "/graph" }, "human", 1000);
    const s = useView.getState();
    expect(s.destination).toBe("/graph");
    expect(s.trail).toHaveLength(1);
    expect(s.trail[0]).toMatchObject({ author: "human", status: "applied", at: 1000 });
  });

  it("applies a model command when the user has not touched that dimension", () => {
    useView.getState().dispatch({ kind: "frame", ids: ["a", "b"] }, "model", 2000);
    const s = useView.getState();
    expect(s.frame).toEqual(["a", "b"]);
    expect(s.trail[0]).toMatchObject({ author: "model", status: "applied" });
    expect(s.offered).toHaveLength(0);
  });

  it("offers rather than applies when the user has touched that dimension", () => {
    useView.getState().dispatch({ kind: "frame", ids: ["mine"] }, "human", 1000);
    useView.getState().dispatch({ kind: "frame", ids: ["theirs"] }, "model", 2000);
    const s = useView.getState();
    expect(s.frame).toEqual(["mine"]);
    expect(s.offered).toHaveLength(1);
    expect(s.offered[0].command).toEqual({ kind: "frame", ids: ["theirs"] });
    expect(s.trail[1]).toMatchObject({ author: "model", status: "offered" });
  });

  it("blocks per dimension, not globally", () => {
    // Framing by hand must not stop the agent opening a destination. Blocking
    // both would make the agent read as broken rather than deferential.
    useView.getState().dispatch({ kind: "frame", ids: ["mine"] }, "human", 1000);
    useView.getState().dispatch({ kind: "open", view: "/geo" }, "model", 2000);
    const s = useView.getState();
    expect(s.destination).toBe("/geo");
    expect(s.offered).toHaveLength(0);
  });

  it("clears the touch record when a new turn begins", () => {
    useView.getState().dispatch({ kind: "frame", ids: ["mine"] }, "human", 1000);
    useView.getState().beginTurn();
    useView.getState().dispatch({ kind: "frame", ids: ["theirs"] }, "model", 2000);
    expect(useView.getState().frame).toEqual(["theirs"]);
  });

  it("does not clear the trail when a new turn begins", () => {
    useView.getState().dispatch({ kind: "open", view: "/graph" }, "human", 1000);
    useView.getState().beginTurn();
    expect(useView.getState().trail).toHaveLength(1);
  });

  it("applies an accepted offer and removes it", () => {
    useView.getState().dispatch({ kind: "frame", ids: ["mine"] }, "human", 1000);
    useView.getState().dispatch({ kind: "frame", ids: ["theirs"] }, "model", 2000);
    const id = useView.getState().offered[0].id;
    useView.getState().acceptOffer(id, 3000);
    const s = useView.getState();
    expect(s.frame).toEqual(["theirs"]);
    expect(s.offered).toHaveLength(0);
    expect(s.trail.at(-1)).toMatchObject({ status: "applied", author: "model" });
  });

  it("drops a dismissed offer without applying it", () => {
    useView.getState().dispatch({ kind: "frame", ids: ["mine"] }, "human", 1000);
    useView.getState().dispatch({ kind: "frame", ids: ["theirs"] }, "model", 2000);
    const id = useView.getState().offered[0].id;
    useView.getState().dismissOffer(id);
    const s = useView.getState();
    expect(s.frame).toEqual(["mine"]);
    expect(s.offered).toHaveLength(0);
  });

  it("keeps only the newest offer per dimension", () => {
    useView.getState().dispatch({ kind: "cue", text: "mine" }, "human", 1000);
    useView.getState().dispatch({ kind: "cue", text: "first" }, "model", 2000);
    useView.getState().dispatch({ kind: "cue", text: "second" }, "model", 3000);
    const s = useView.getState();
    expect(s.offered).toHaveLength(1);
    expect(s.offered[0].command).toEqual({ kind: "cue", text: "second" });
  });

  it("merges a filter patch rather than replacing the filter set", () => {
    useView.getState().dispatch({ kind: "filter", patch: { since: 5 } }, "human", 1000);
    useView.getState().dispatch({ kind: "filter", patch: { coarse: ["person"] } }, "human", 2000);
    expect(useView.getState().filters).toEqual({ since: 5, until: null, coarse: ["person"] });
  });

  it("records a rejected command in the trail without changing state", () => {
    useView.getState().reject("focus", "focus of must be memory or entity", "model", 4000);
    const s = useView.getState();
    expect(s.focus).toBeNull();
    expect(s.trail.at(-1)).toMatchObject({
      status: "rejected",
      author: "model",
      reason: "focus of must be memory or entity",
    });
  });
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `cd front/ui && npx vitest run src/stores/view.test.ts`
Expected: FAIL — `Failed to resolve import "./view"`.

- [ ] **Step 3: Write the implementation**

Create `front/ui/src/stores/view.ts`:

```ts
import { create } from "zustand";
import {
  dimensionOf,
  type Author,
  type Dimension,
  type Filters,
  type ViewCommand,
  type ViewPath,
} from "@/lib/view/commands";

/**
 * Everything about what is on screen.
 *
 * One store, one entry point, two producers: the user's hands and the single
 * adapter that translates the agent's events (app/useViewSync.ts). Two paths
 * into the same screen state would drift, and the drift would be invisible
 * until a demo.
 *
 * `selection` is deliberately NOT duplicated here. It already lives in
 * stores/session.ts under the "one selected object at a time" rule from
 * WORKFLOWS.md, and a second copy is a second source of truth.
 */

export type TrailStatus = "applied" | "offered" | "rejected";

export interface TrailEntry {
  /** Monotonic within a session; the history surface merges on `at`. */
  id: string;
  command: ViewCommand | null;
  dimension: Dimension;
  author: Author;
  status: TrailStatus;
  reason?: string;
  at: number;
}

export interface Offer {
  id: string;
  command: ViewCommand;
  dimension: Dimension;
  at: number;
}

interface ViewState {
  destination: ViewPath;
  cue: string;
  /** Which ids the field frames; "all" is the rest state — the whole corpus. */
  frame: string[] | "all";
  focus: { id: string; of: "memory" | "entity" } | null;
  filters: Filters;

  /** Every command this session, applied or not. Session-scoped by design:
   *  where the camera pointed is not a fact about the corpus, and mixing UI
   *  navigation into the durable ledger weakens exactly the property the
   *  product is sold on. */
  trail: TrailEntry[];
  /** Model commands declined by the authority rule, awaiting a Follow. */
  offered: Offer[];
  /** Dimensions the user has touched since the current turn began. */
  touched: Dimension[];

  dispatch: (command: ViewCommand, author: Author, at?: number) => void;
  reject: (dimension: Dimension, reason: string, author: Author, at?: number) => void;
  beginTurn: () => void;
  acceptOffer: (id: string, at?: number) => void;
  dismissOffer: (id: string) => void;
}

let counter = 0;
/** Session-unique id. Not crypto — these never leave the tab. */
function nextId(): string {
  counter += 1;
  return `v${counter}`;
}

/** Fold one command into the view slice. Pure; exported for direct testing. */
export function applyCommand(
  state: Pick<ViewState, "destination" | "cue" | "frame" | "focus" | "filters">,
  command: ViewCommand,
): Pick<ViewState, "destination" | "cue" | "frame" | "focus" | "filters"> {
  switch (command.kind) {
    case "open":
      return { ...state, destination: command.view };
    case "cue":
      return { ...state, cue: command.text };
    case "frame":
      return { ...state, frame: command.ids === "all" ? "all" : [...command.ids] };
    case "focus":
      return { ...state, focus: { id: command.id, of: command.of } };
    case "filter":
      // Merge, not replace: a patch that names only `since` must not silently
      // clear an entity-type filter the user set a moment ago.
      return { ...state, filters: { ...state.filters, ...command.patch } };
  }
}

export const useView = create<ViewState>((set, get) => ({
  destination: "/chat",
  cue: "",
  frame: "all",
  focus: null,
  filters: { since: null, until: null, coarse: [] },
  trail: [],
  offered: [],
  touched: [],

  dispatch: (command, author, at = Date.now()) => {
    const state = get();
    const dimension = dimensionOf(command);

    // The human always has the wheel. A model command on a dimension the user
    // has already moved this turn is not applied and not discarded — silently
    // dropping it leaves the user with a model that claims to have done
    // something invisible.
    if (author === "model" && state.touched.includes(dimension)) {
      const offer: Offer = { id: nextId(), command, dimension, at };
      set({
        offered: [...state.offered.filter((o) => o.dimension !== dimension), offer],
        trail: [
          ...state.trail,
          { id: nextId(), command, dimension, author, status: "offered", at },
        ],
      });
      return;
    }

    set({
      ...applyCommand(state, command),
      touched:
        author === "human" && !state.touched.includes(dimension)
          ? [...state.touched, dimension]
          : state.touched,
      trail: [...state.trail, { id: nextId(), command, dimension, author, status: "applied", at }],
    });
  },

  reject: (dimension, reason, author, at = Date.now()) =>
    set((s) => ({
      trail: [
        ...s.trail,
        { id: nextId(), command: null, dimension, author, status: "rejected", reason, at },
      ],
    })),

  // A new turn is a new negotiation. The trail is history and survives; the
  // touch record is about who is steering right now and does not.
  beginTurn: () => set({ touched: [] }),

  acceptOffer: (id, at = Date.now()) => {
    const state = get();
    const offer = state.offered.find((o) => o.id === id);
    if (!offer) return;
    set({
      ...applyCommand(state, offer.command),
      offered: state.offered.filter((o) => o.id !== id),
      trail: [
        ...state.trail,
        {
          id: nextId(),
          command: offer.command,
          dimension: offer.dimension,
          author: "model",
          status: "applied",
          at,
        },
      ],
    });
  },

  dismissOffer: (id) => set((s) => ({ offered: s.offered.filter((o) => o.id !== id) })),
}));
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `cd front/ui && npx vitest run src/stores/view.test.ts`
Expected: PASS, 11 tests.

- [ ] **Step 5: Typecheck**

Run: `cd front/ui && npm run typecheck`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add front/ui/src/stores/view.ts front/ui/src/stores/view.test.ts
git commit -m "feat(ui): view store with a per-dimension authority rule

One store, one dispatch, two producers. The human always has the wheel: a
model command on a dimension the user moved this turn becomes a Follow offer
rather than being applied or silently dropped. Blocking is per dimension, so
framing the field by hand does not stop the agent opening a destination.

The trail is session-scoped by design. Where the camera pointed is not a fact
about the corpus, and mixing UI navigation into the durable ledger weakens
the property the product is sold on."
```

---

### Task 4: Translating the agent's events into commands

**Files:**
- Create: `front/ui/src/lib/view/fromSeatEvents.ts`
- Test: `front/ui/src/lib/view/fromSeatEvents.test.ts`

**Interfaces:**
- Consumes: `ViewCommand` from `@/lib/view/commands` (Task 1); `ChatOp` from `@/stores/chat`.
- Produces: `commandsFromOp(op: ChatOp): ViewCommand[]`.

This is spec §5.2 — implicit sync, which needs no seat change, no new tool and
no prompt change. It is the sync contract, and it is the test that must not be
weakened.

- [ ] **Step 1: Write the failing test**

Create `front/ui/src/lib/view/fromSeatEvents.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import type { ChatOp } from "@/stores/chat";
import { commandsFromOp } from "./fromSeatEvents";

/** A memory_recall as the seat emits it (lib/seat/types.ts:101). */
const recall = (over: Partial<Extract<ChatOp, { type: "memory_recall" }>> = {}) =>
  ({
    type: "memory_recall",
    scope: "user",
    query: "key bridge",
    mode: "hybrid",
    memories: [{ id: "m1" }, { id: "m2" }],
    facts: [],
    todos: [],
    lineage: [],
    took_ms: 12,
    ...over,
  }) as ChatOp;

describe("commandsFromOp", () => {
  it("turns a user-scope recall into a cue and a frame", () => {
    expect(commandsFromOp(recall())).toEqual([
      { kind: "cue", text: "key bridge" },
      { kind: "frame", ids: ["m1", "m2"] },
    ]);
  });

  it("ignores harness-scope recalls", () => {
    // Harness recalls are the seat's own bookkeeping, not the user's question.
    // Framing the field on them would move the view for something the user
    // never asked and cannot see the reason for.
    expect(commandsFromOp(recall({ scope: "harness" }))).toEqual([]);
  });

  it("emits no frame when a recall returned nothing", () => {
    // An empty frame would blank the field and read as data loss. The corpus
    // stays framed; only the cue changes.
    expect(commandsFromOp(recall({ memories: [] }))).toEqual([
      { kind: "cue", text: "key bridge" },
    ]);
  });

  it("frames a proactive_context on its surfaced memories", () => {
    const op = {
      type: "proactive_context",
      scope: "user",
      query: "what did we decide",
      memories: [{ id: "p1" }, { id: "p2" }],
      injected_memory_ids: ["p1"],
      feedback: null,
      temporal_credits_applied: null,
      took_ms: 4,
    } as ChatOp;
    expect(commandsFromOp(op)).toEqual([
      { kind: "cue", text: "what did we decide" },
      { kind: "frame", ids: ["p1", "p2"] },
    ]);
  });

  it("produces nothing for ops that say nothing about what is on screen", () => {
    for (const op of [
      { type: "memory_write", scope: "user", memory_id: "x" },
      { type: "model_changed", model: { provider: "anthropic", model: "opus" } },
      { type: "tool_call_end", tool_call_id: "t", tool_name: "recall_memory", is_error: false },
      { type: "error", message: "boom" },
    ] as ChatOp[]) {
      expect(commandsFromOp(op)).toEqual([]);
    }
  });

  it("drops entries with no usable id rather than framing on undefined", () => {
    const op = recall({ memories: [{ id: "m1" }, {}, { id: "" }] as never });
    expect(commandsFromOp(op)).toEqual([
      { kind: "cue", text: "key bridge" },
      { kind: "frame", ids: ["m1"] },
    ]);
  });
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `cd front/ui && npx vitest run src/lib/view/fromSeatEvents.test.ts`
Expected: FAIL — `Failed to resolve import "./fromSeatEvents"`.

- [ ] **Step 3: Write the implementation**

Create `front/ui/src/lib/view/fromSeatEvents.ts`:

```ts
import type { ChatOp } from "@/stores/chat";
import type { ViewCommand } from "./commands";

/**
 * What the agent's events say about what should be on screen.
 *
 * The seat already narrates itself to the browser — `memory_recall` carries
 * the memories actually retrieved (lib/seat/types.ts:101). Nothing new is sent
 * over the wire to make the field follow the agent; the browser simply stops
 * only printing what it already receives.
 *
 * This is the sync contract. Keep it a pure function of one op so it can be
 * tested against recorded event sequences.
 */

function usableIds(memories: readonly { id?: unknown }[]): string[] {
  const ids: string[] = [];
  for (const m of memories) {
    if (typeof m.id === "string" && m.id.length > 0) ids.push(m.id);
  }
  return ids;
}

export function commandsFromOp(op: ChatOp): ViewCommand[] {
  switch (op.type) {
    case "memory_recall":
    case "proactive_context": {
      // Harness-scope recalls are the seat's own bookkeeping. Moving the view
      // for a question the user never asked, and cannot see the reason for,
      // reads as the app twitching.
      if (op.scope !== "user") return [];

      const commands: ViewCommand[] = [];
      if (op.query) commands.push({ kind: "cue", text: op.query });

      const ids = usableIds(op.memories);
      // An empty frame would blank the field and read as data loss. A recall
      // that found nothing leaves the corpus framed and changes only the cue.
      if (ids.length > 0) commands.push({ kind: "frame", ids });

      return commands;
    }
    default:
      return [];
  }
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `cd front/ui && npx vitest run src/lib/view/fromSeatEvents.test.ts`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run the whole suite**

Run: `cd front/ui && npm test`
Expected: PASS — the three new files plus the pre-existing `src/lib/api/health.test.ts`.

- [ ] **Step 6: Commit**

```bash
git add front/ui/src/lib/view/fromSeatEvents.ts front/ui/src/lib/view/fromSeatEvents.test.ts
git commit -m "feat(ui): translate seat recall events into view commands

The seat already tells the browser which memories it retrieved; the browser
only printed it. This is the whole of implicit sync -- no new wire format, no
seat change, no prompt change.

Harness-scope recalls are ignored: moving the view for a question the user
never asked reads as the app twitching. A recall that found nothing changes
the cue but leaves the corpus framed, because an empty frame reads as data
loss rather than as an empty result."
```

---

### Task 5: Wire the adapter in, and make the field obey it

**Files:**
- Create: `front/ui/src/app/useViewSync.ts`
- Modify: `front/ui/src/app/App.tsx` (`Shell`, around line 60-74)
- Modify: `front/ui/src/features/recall/GraphCanvas.tsx` (transform init ~line 169, zoom wiring ~line 420)

**Interfaces:**
- Consumes: `useView` (Task 3), `commandsFromOp` (Task 4), `extentOf`/`fitTransform` (Task 2).
- Produces: `useViewSync(): void` — mounted exactly once.

- [ ] **Step 1: Write `useViewSync`**

Create `front/ui/src/app/useViewSync.ts`:

```ts
import { useEffect, useRef } from "react";
import { useChat } from "@/stores/chat";
import { useView } from "@/stores/view";
import { commandsFromOp } from "@/lib/view/fromSeatEvents";

/**
 * The single point where the agent's events become view commands.
 *
 * Mounted once, in Shell. EvidencePanel and MessageList keep their own
 * read-only consumption of the same events; this must not become a second
 * scattered consumer, because two translators would drift and the drift would
 * only show up in front of someone.
 *
 * Ops are consumed by position rather than by identity: the chat store appends
 * to `turns[n].ops` and never rewrites earlier entries, so a per-turn cursor is
 * sufficient and needs no id on the wire.
 */
export function useViewSync(): void {
  const dispatch = useView((s) => s.dispatch);
  const beginTurn = useView((s) => s.beginTurn);
  /** turn number -> ops already translated. */
  const cursor = useRef(new Map<number, number>());
  const lastTurn = useRef<number | null>(null);

  useEffect(() => {
    return useChat.subscribe((state) => {
      const id = state.activeId;
      if (!id) return;
      const convo = state.conversations[id];
      if (!convo) return;
      const turn = convo.turns.at(-1);
      if (!turn) return;

      // A new turn is a new negotiation over who is steering.
      if (lastTurn.current !== turn.turn) {
        lastTurn.current = turn.turn;
        beginTurn();
      }

      const seen = cursor.current.get(turn.turn) ?? 0;
      if (turn.ops.length <= seen) return;
      for (const op of turn.ops.slice(seen)) {
        for (const command of commandsFromOp(op)) dispatch(command, "model");
      }
      cursor.current.set(turn.turn, turn.ops.length);
    });
  }, [dispatch, beginTurn]);
}
```

- [ ] **Step 2: Mount it in `Shell`**

In `front/ui/src/app/App.tsx`, add the import beside the other app-local imports:

```ts
import { useViewSync } from "./useViewSync";
```

and call it as the first line of `Shell` (currently line 61):

```ts
function Shell({ reach }: { reach: Reachability }) {
  useViewSync();
  const { pathname } = useLocation();
```

- [ ] **Step 3: Make `GraphCanvas` frame on load and follow the store**

In `front/ui/src/features/recall/GraphCanvas.tsx`:

Add the imports:

```ts
import { extentOf, fitTransform } from "@/lib/view/fit";
import { useView } from "@/stores/view";
```

Read the framed set beside the existing selection subscription (near line 163):

```ts
const framed = useView((s) => s.frame);
```

After the zoom behaviour is created and `sel.call(zoomBehavior)` has run (around line 428), add the fit:

```ts
    /**
     * Frame the nodes rather than starting at the identity transform.
     *
     * This MUST go through `zoomBehavior.transform` and not by assigning
     * `transformRef.current`. d3-zoom keeps its own copy of the current
     * transform on the selection's `__zoom` property; writing the ref alone
     * leaves the two disagreeing and the first pan gesture jumps back to
     * wherever d3 still thought it was.
     */
    const frameNow = (): void => {
      const subject =
        framed === "all" ? nodes : nodes.filter((n) => framed.includes(n.id));
      // A frame naming nothing on screen must not blank the view. Falling back
      // to the whole set keeps the corpus visible, which is the rest state.
      const extent = extentOf(subject.length > 0 ? subject : nodes);
      if (!extent) return;
      const fit = fitTransform(extent, { width, height }, { padding: 32, scaleExtent: [0.2, 6] });
      sel.call(zoomBehavior.transform, zoomIdentity.translate(fit.x, fit.y).scale(fit.k));
    };
```

Call `frameNow()` once the layout has coordinates — inside the `reduceMotion`
branch immediately after the 300 synchronous ticks, and for the animated branch
from a `sim.on("end", frameNow)` handler registered beside the existing
`sim.on("tick", draw)`.

- [ ] **Step 4: Verify by hand, because this is the part no unit test covers**

Run: `cd front/ui && npm run dev` (serves on `:8788`; requires the backend on `:3030`)

Check each of these and write the result into the PR description:
1. `/recall` with results: every node is inside the viewport on first paint, with visible margin. Previously they overflowed.
2. Pan and zoom by hand: the first drag does **not** jump. A jump means Step 3's `zoomBehavior.transform` note was not followed.
3. Ask the conversation something that recalls memories: the field reframes to those memories, and the cue in the header updates.
4. Pan the field by hand, then ask again: the field does **not** move, and a Follow offer is recorded (`useView.getState().offered` in the console).
5. With OS "reduce motion" on: the reframe is instant, with no animated transition.

- [ ] **Step 5: Typecheck and full suite**

Run: `cd front/ui && npm run typecheck && npm test`
Expected: no type errors; all tests pass.

- [ ] **Step 6: Commit**

```bash
git add front/ui/src/app/useViewSync.ts front/ui/src/app/App.tsx front/ui/src/features/recall/GraphCanvas.tsx
git commit -m "feat(ui): field frames itself, and follows the agent's recalls

Replaces the identity transform both canvases started from with a computed
fit, which retires the long-standing 'graph does not fit the viewport'
defect rather than patching it -- framing is the rest state.

The fit is applied through zoomBehavior.transform, not by assigning the ref:
d3-zoom keeps its own transform on the selection's __zoom property, and
writing only the ref leaves the two disagreeing so the first pan jumps.

useViewSync is the single point where seat events become view commands.
EvidencePanel and MessageList keep their read-only consumption; a second
translator would drift."
```

---

### Task 6: The conversation dock

**Files:**
- Modify: `front/ui/package.json` (devDependencies)
- Modify: `front/ui/vite.config.ts` (add the `test` block)
- Create: `front/ui/src/test/setup.ts`
- Modify: `front/ui/src/features/chat/ConversationOverlay.tsx:154-158`
- Test: `front/ui/src/features/chat/ConversationDock.test.tsx`

**Interfaces:**
- Consumes: `useChat` from `@/stores/chat`.
- Produces: no new exports; changes `ConversationOverlay`'s mount conditions.

Spec §7. The lifecycle already works — `send()` is a store action, in-flight
aborts live in a module-level `Map` (`stores/chat.ts:265`), and only `forget()`
aborts (`:404`). Navigating does not kill a turn. What is missing is visibility.

This task needs a DOM, so it installs one. These are devDependencies and do not
enter the single-file bundle.

- [ ] **Step 1: Install the test environment**

Run: `cd front/ui && npm install --save-dev jsdom@^27 @testing-library/react@^16 @testing-library/dom@^10`

- [ ] **Step 2: Configure vitest for a DOM**

In `front/ui/vite.config.ts`, change the import so the config accepts a `test` block:

```ts
import { defineConfig } from "vitest/config";
```

and add, as a sibling of `build`:

```ts
  // vitest only. `vitest/config` re-exports vite's defineConfig, so the build
  // above is unaffected -- this block is stripped from a production build.
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
  },
```

Create `front/ui/src/test/setup.ts`:

```ts
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Without this, a component from one test is still mounted during the next and
// queries match the wrong tree -- which shows up as a passing test that proves
// nothing.
afterEach(cleanup);
```

- [ ] **Step 3: Write the failing test**

Create `front/ui/src/features/chat/ConversationDock.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it } from "vitest";
import { useChat } from "@/stores/chat";
import { ConversationOverlay } from "./ConversationOverlay";

const seatOnline = { state: "online", backendOk: true } as never;

function seedStreamingConversation(): void {
  useChat.setState({
    activeId: "c1",
    conversations: {
      c1: {
        turns: [
          {
            turn: 1,
            userText: "what did we decide about the bridge",
            ops: [],
            assistantText: "We decided",
            thinkingText: "",
            usage: null,
            pending: true,
          },
        ],
        streaming: true,
        model: { provider: "anthropic", model: "opus" } as never,
        totals: {
          input: 0,
          output: 0,
          cache_read: 0,
          cache_write: 0,
          reasoning: 0,
          total_tokens: 0,
          cost_total: 0,
        },
        transportError: null,
      },
    },
    selected: null,
    revertedLedgerIds: {},
    evidenceOpen: true,
    sessionsOpen: true,
  });
}

const at = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <ConversationOverlay seat={seatOnline} />
    </MemoryRouter>,
  );

describe("conversation dock", () => {
  beforeEach(seedStreamingConversation);

  it("is present on a non-chat destination", () => {
    at("/recall");
    expect(screen.getByLabelText(/conversation/i)).toBeTruthy();
  });

  it("is present on /chat too, so navigating never hides a live turn", () => {
    // Previously returned null here (ConversationOverlay.tsx:156), which made
    // the conversation vanish on exactly one route.
    at("/chat");
    expect(screen.getByLabelText(/conversation/i)).toBeTruthy();
  });

  it("stays present on every destination while a turn is streaming", () => {
    for (const path of ["/recall", "/graph", "/geo", "/anomalies", "/tasks"]) {
      const { unmount } = at(path);
      expect(screen.getByLabelText(/conversation/i)).toBeTruthy();
      unmount();
    }
  });

  it("hides only when the seat is offline", () => {
    render(
      <MemoryRouter initialEntries={["/recall"]}>
        <ConversationOverlay seat={{ state: "offline", detail: "down" } as never} />
      </MemoryRouter>,
    );
    expect(screen.queryByLabelText(/conversation/i)).toBeNull();
  });
});
```

- [ ] **Step 4: Run the test and verify it fails**

Run: `cd front/ui && npx vitest run src/features/chat/ConversationDock.test.tsx`
Expected: FAIL — the `/chat` case returns null, so `getByLabelText` throws
"Unable to find a label with the text of: /conversation/i".

- [ ] **Step 5: Change the mount conditions**

In `front/ui/src/features/chat/ConversationOverlay.tsx`, replace lines 154-158:

```ts
  // The /chat route IS the conversation, full width. A floating copy of it on
  // top of itself is a duplicate.
  if (pathname === "/chat") return null;
  if (seat.state !== "online") return null;
  if (dismissed) return null;
```

with:

```ts
  // Offline there is nothing to show and nothing to continue, so this is the
  // one condition that still removes the dock entirely.
  if (seat.state !== "online") return null;
```

and make `dismissed` collapse the dock to its strip rather than remove it: where
the component currently returns its full body, render the strip when `dismissed`
is true — the streaming state and the last line, with the same `aria-label` — and
the full panel otherwise. On `/chat`, render the strip rather than the panel, so
the route that IS the conversation does not carry a floating duplicate of
itself while a live turn still cannot become invisible.

Ensure the outermost rendered element in both branches carries
`aria-label="Conversation"`.

- [ ] **Step 6: Run the test and verify it passes**

Run: `cd front/ui && npx vitest run src/features/chat/ConversationDock.test.tsx`
Expected: PASS, 4 tests.

- [ ] **Step 7: Run the whole suite and typecheck**

Run: `cd front/ui && npm run typecheck && npm test`
Expected: no type errors; every test passes, including the pre-existing
`src/lib/api/health.test.ts` under the new jsdom environment.

- [ ] **Step 8: Verify the bundle did not grow a dependency**

Run: `cd front/ui && npm run build && ls -la dist/`
Expected: exactly one file, `dist/index.html`. No `.js` or `.css` siblings.
The three packages installed in Step 1 are devDependencies and must not appear
in the bundle.

- [ ] **Step 9: Commit**

```bash
git add front/ui/package.json front/ui/package-lock.json front/ui/vite.config.ts front/ui/src/test/setup.ts front/ui/src/features/chat/ConversationOverlay.tsx front/ui/src/features/chat/ConversationDock.test.tsx
git commit -m "feat(ui): conversation dock is present on every destination

The lifecycle already survived navigation -- send() is a store action and
only forget() aborts. What did not survive was visibility: the overlay
returned null on /chat and when dismissed, so a live turn could become
invisible on exactly the route that is about to show it.

Dismiss now collapses to a strip instead of removing the dock. Offline is
the one condition that still removes it, because there is nothing to show
and nothing to continue.

Adds jsdom and testing-library as devDependencies -- the first component
test in this project. They do not enter the single-file bundle; the build
still emits exactly dist/index.html."
```

---

## Self-review notes

**Spec coverage.** §3 field rest state → Tasks 2 and 5. §4 view bus → Tasks 1 and 3. §4.1 commands → Task 1. §4.2 single adapter → Task 4 and Task 5 Step 1. §5.2 implicit sync → Task 4. §6 authority → Task 3. §6.1 untrusted args → Task 1. §7 dock → Task 6. §9 error handling → the empty-frame and no-usable-id cases in Task 4, the fallback in Task 5 Step 3, the offline branch in Task 6. §10 testing → every task's test steps. §8 history, §8.4 actor, §5.3 explicit tools → **deliberately deferred**, stated under Scope.

**Known gaps carried forward to the second plan.** `EntityCanvas.tsx` gets the
same fit treatment as `GraphCanvas.tsx`; this plan changes only the recall
canvas so the pattern is reviewed once before it is repeated. The `Filters`
type is defined and stored but nothing reads it yet — it is consumed by the
history surface's filtering in the second plan.
