import { describe, expect, it } from "vitest";
import { coreExtentOf, extentOf, fitTransform } from "./fit";

describe("coreExtentOf", () => {
  /** 100 points packed in [0,10], plus one runaway — the shape a force layout
   *  produces when a degree-0 node drifts away from the mass. */
  const massPlusOutlier = () => {
    const pts = Array.from({ length: 100 }, (_, i) => ({ x: (i % 10), y: Math.floor(i / 10) }));
    return [...pts, { x: 5000, y: 5000 }];
  };

  it("returns null when nothing has coordinates", () => {
    expect(coreExtentOf([])).toBeNull();
    expect(coreExtentOf([{}, { x: 1 }])).toBeNull();
  });

  it("ignores a runaway node that would otherwise own the frame", () => {
    const core = coreExtentOf(massPlusOutlier(), 0.05)!;
    expect(core.maxX).toBeLessThanOrEqual(10);
    expect(core.maxY).toBeLessThanOrEqual(10);
  });

  it("is exactly extentOf when q is 0", () => {
    const pts = massPlusOutlier();
    expect(coreExtentOf(pts, 0)).toEqual(extentOf(pts));
  });

  it("does not trim small sets, where a fraction is a meaningful share", () => {
    // 8 points: trimming 5% either does nothing or throws away a real one.
    const pts = Array.from({ length: 8 }, (_, i) => ({ x: i, y: i }));
    expect(coreExtentOf(pts, 0.25)).toEqual(extentOf(pts));
  });

  it("trims each axis independently", () => {
    // This point is extreme in x and utterly ordinary in y. It should forfeit
    // its x and keep its y, not be dropped wholesale.
    const pts = Array.from({ length: 40 }, (_, i) => ({ x: i, y: 100 }));
    pts.push({ x: 99999, y: 100 });
    const core = coreExtentOf(pts, 0.05)!;
    expect(core.maxX).toBeLessThan(99999);
    expect(core.minY).toBe(100);
    expect(core.maxY).toBe(100);
  });

  it("clamps a nonsensical q rather than inverting the extent", () => {
    const pts = Array.from({ length: 40 }, (_, i) => ({ x: i, y: i }));
    const core = coreExtentOf(pts, 5)!;
    expect(core.minX).toBeLessThanOrEqual(core.maxX);
    expect(core.minY).toBeLessThanOrEqual(core.maxY);
  });
});

describe("extentOf", () => {
  it("returns null when no point has coordinates", () => {
    expect(extentOf([])).toBeNull();
    expect(extentOf([{}, { x: undefined, y: 3 }])).toBeNull();
  });

  it("ignores points missing either coordinate", () => {
    // d3-force nodes have optional x/y that are genuinely absent before the
    // first tick. Coercing a missing coordinate to 0 would drag the extent to
    // the origin and frame a box the nodes are not in.
    expect(extentOf([{ x: 1, y: 2 }, { x: 5 }, { y: 9 }])).toEqual({
      minX: 1,
      minY: 2,
      maxX: 1,
      maxY: 2,
    });
  });

  it("ignores non-finite coordinates", () => {
    expect(extentOf([{ x: 1, y: 2 }, { x: Number.NaN, y: 0 }, { x: Infinity, y: 0 }])).toEqual({
      minX: 1,
      minY: 2,
      maxX: 1,
      maxY: 2,
    });
  });

  it("spans every complete point", () => {
    expect(
      extentOf([
        { x: -4, y: 10 },
        { x: 6, y: 2 },
      ]),
    ).toEqual({ minX: -4, minY: 2, maxX: 6, maxY: 10 });
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

  it("honours padding, leaving the requested margin on the tight axis", () => {
    const extent = { minX: 0, minY: 0, maxX: 100, maxY: 100 };
    const t = fitTransform(extent, { width: 400, height: 400 }, { padding: 40 });
    expect(apply(t, { x: 0, y: 0 }).x).toBeCloseTo(40, 6);
    expect(apply(t, { x: 100, y: 100 }).x).toBeCloseTo(360, 6);
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
    // GraphCanvas uses scaleExtent([0.2, 6]) and EntityCanvas [0.15, 6]. A tiny
    // extent would otherwise want a scale far above 6, d3 would clamp it, and
    // the first pan gesture would jump to a transform we never drew.
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

  it("still centres correctly when the scale was clamped", () => {
    const t = fitTransform(
      { minX: 0, minY: 0, maxX: 1, maxY: 1 },
      { width: 800, height: 600 },
      { scaleExtent: [0.2, 2] },
    );
    expect(t.k).toBe(2);
    const centre = apply(t, { x: 0.5, y: 0.5 });
    expect(centre.x).toBeCloseTo(400, 6);
    expect(centre.y).toBeCloseTo(300, 6);
  });

  it("degrades to a finite transform on a zero-sized viewport", () => {
    // ResizeObserver fires before layout with width 0; NaN here would be
    // painted onto the canvas and the graph would vanish.
    const t = fitTransform({ minX: 0, minY: 0, maxX: 10, maxY: 10 }, { width: 0, height: 0 }, {});
    expect(Number.isFinite(t.k)).toBe(true);
    expect(t.k).toBeGreaterThan(0);
    expect(Number.isFinite(t.x)).toBe(true);
    expect(Number.isFinite(t.y)).toBe(true);
  });
});
