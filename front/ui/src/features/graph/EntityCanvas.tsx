import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  select,
  zoom,
  zoomIdentity,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
  type ZoomTransform,
} from "d3";
import { coreExtentOf, fitTransform } from "@/lib/view/fit";
import { useSession } from "@/stores/session";
import {
  cooccurFloor,
  entityTypeToken,
  isEdgeRendered,
  SMALL_GRAPH_MAX,
  tierToken,
  type EdgeTier,
  type UniverseModel,
} from "./universe";

/**
 * The entity knowledge-graph canvas.
 *
 * Canvas, not SVG: at universe scale the node count is in the thousands and an
 * SVG DOM node per entity is what makes a tab unresponsive. d3 supplies the
 * force layout and the zoom transform algebra; the painting is manual, which is
 * the same division the old vanilla dashboard used (front/index.html:596-598).
 *
 * That dashboard has since been deleted, so its line numbers here resolve in
 * git history rather than in the tree.
 *
 * TWO LEVELS, and which one shows is decided by corpus size rather than by a
 * preference:
 *
 *  - Above SMALL_GRAPH_MAX entities the overview draws CLUSTERS — Louvain
 *    communities as super-nodes, sized by member count — and clicking one
 *    drills into its members. A 5k-entity hairball is not a view of anything.
 *  - At or below that, the overview IS the entity constellation. Cluster
 *    bubbles only earn their keep at scale; on a demo corpus two opaque
 *    super-nodes hide the thing you came to look at (front/index.html:905-913).
 *
 * Both levels are the same renderer with different node sets, so there is one
 * hit-test, one zoom, one draw.
 */

/** Zoom limits, shared by the gesture handler and the auto-frame so the frame
 *  can never ask for a scale d3 would silently clamp. */
const SCALE_EXTENT: [number, number] = [0.15, 6];

/** Margin left around the framed corpus, in screen pixels. Entity labels are
 *  drawn outside the node radius, so a tight frame clips the words rather than
 *  the dots. */
const FRAME_PADDING = 56;

/** Fraction trimmed from each end of each axis before framing. A force layout
 *  strands weakly-connected nodes far from the mass; without this, twenty
 *  degree-0 entities set the camera for all 136 and the real corpus becomes an
 *  illegible knot. Trimmed nodes are still drawn, just outside the opening view. */
const FRAME_TRIM = 0.06;

type Level = "clusters" | "entities";

interface CanvasNode extends SimulationNodeDatum {
  /** Cluster index as a string, or an entity id. */
  id: string;
  label: string;
  r: number;
  kind: "cluster" | "entity";
  /** Entity type, or a cluster's dominant type. */
  type: string;
  /** Members, for a cluster node. */
  size: number;
  degree: number;
  salience: number;
  mentions: number;
  longTail: boolean;
}

interface CanvasLink extends SimulationLinkDatum<CanvasNode> {
  source: CanvasNode | string;
  target: CanvasNode | string;
  relation: string;
  strength: number;
  generic: boolean;
  tier: EdgeTier;
  /** Stroke width for aggregated cluster links. */
  width: number;
}

const DRAG_SLOP_PX = 4;

/** Below this many entities, every label is drawn at any zoom. Chosen so a
 *  drilled cluster and a small corpus both read as named things rather than
 *  anonymous dots; above it, labels would collide faster than they inform. */
const LABEL_ALWAYS_MAX = 200;

function readTokens(el: HTMLElement) {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    token: (name: string) => v(name, "#8a8f98"),
    // Selection ONLY. DIRECTION.md: one accent, and it marks focus, the primary
    // action and active nav — nothing else. Resting edges took this colour in
    // the first cut of this canvas, which made the whole graph read as
    // permanently active and left selection with nothing louder to say.
    active: v("--node-active", "#f4622e"),
    muted: v("--muted-foreground", "#8a8f98"),
    fg: v("--foreground", "#f7f8f8"),
    // Neutral greys that exist for exactly this: edge weight mapped to value,
    // no hue, so edges never compete with the categorical node hues.
    edgeStrong: v("--edge-strong", "#4a4d55"),
    edgeMedium: v("--edge-medium", "#35373d"),
    edgeWeak: v("--edge-weak", "#26282c"),
  };
}

function hexA(hex: string, a: number): string {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex.trim());
  if (!m) return `rgba(138,143,152,${a})`;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
}

interface Hover {
  node: CanvasNode;
  x: number;
  y: number;
}

export function EntityCanvas({
  model,
  level,
  clusterId,
  onDrillIn,
  onStats,
}: {
  model: UniverseModel;
  level: Level;
  clusterId: number | null;
  onDrillIn: (clusterId: number) => void;
  /** Reports what this level actually drew, so the footer states the same
   *  numbers the canvas used rather than recomputing them and drifting. */
  onStats?: (stats: { hiddenEdges: number; floor: number }) => void;
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const selectedEntityId = useSession((s) => s.selectedEntityId);
  const selectEntity = useSession((s) => s.selectEntity);
  const [hover, setHover] = useState<Hover | null>(null);

  const transformRef = useRef<ZoomTransform>(zoomIdentity);
  const hoverRef = useRef<CanvasNode | null>(null);
  const selectedRef = useRef<string | null>(selectedEntityId);
  const drawRef = useRef<() => void>(() => {});
  const simRef = useRef<Simulation<CanvasNode, CanvasLink> | null>(null);

  /** Corpus-derived threshold below which generic co-occurrence is not drawn. */
  const floor = useMemo(() => cooccurFloor(model), [model]);

  /** The node/link set for the current level. */
  const { nodes, links } = useMemo(() => {
    if (level === "clusters") {
      const maxSize = Math.max(1, ...model.clusters.map((c) => c.size));
      const nodes: CanvasNode[] = model.clusters.map((c) => ({
        id: String(c.id),
        label: c.label,
        // sqrt so radius is area-proportional to member count.
        r: 18 + 48 * Math.sqrt(c.size / maxSize),
        kind: "cluster",
        type: c.dominantType,
        size: c.size,
        degree: 0,
        salience: 0,
        mentions: 0,
        longTail: c.longTail,
      }));
      const maxW = Math.max(1, ...model.clusterLinks.map((l) => l.weight));
      const links: CanvasLink[] = model.clusterLinks.map((l) => ({
        source: String(l.source),
        target: String(l.target),
        relation: `${l.count} relation${l.count === 1 ? "" : "s"}`,
        strength: l.weight,
        generic: false,
        // An aggregate of many edges across many tiers has no single tier of
        // its own, so the overview draws it in the neutral edge grey rather
        // than picking one and implying a consolidation level it cannot know.
        tier: "L1Working",
        width: 0.5 + 3.5 * Math.sqrt(l.weight / maxW),
      }));
      return { nodes, links };
    }

    // Entity level: either one drilled cluster's members, or — on a small
    // corpus — every entity.
    const memberIdx =
      clusterId !== null
        ? (model.clusters[clusterId]?.members ?? [])
        : model.nodes.map((_, i) => i);
    const keep = new Set(memberIdx.map((i) => model.nodes[i].id));

    // Size carries MENTION COUNT, not salience. The read path serves salience
    // flat at 1.0 for most entities on this corpus, so sizing by it makes every
    // node identical and encodes nothing; mention_count survives the same path
    // with real variance. Sub-linear so a 60-mention hub does not swallow the
    // graph. The legend states which of the two is being drawn.
    const maxMentions = Math.max(1, ...memberIdx.map((i) => model.nodes[i].mentions));

    const nodes: CanvasNode[] = memberIdx.map((i) => {
      const n = model.nodes[i];
      return {
        id: n.id,
        label: n.name,
        r: 4.5 + 9 * Math.sqrt(Math.max(0, n.mentions) / maxMentions),
        kind: "entity",
        type: n.type,
        size: 0,
        degree: n.degree,
        salience: n.salience,
        mentions: n.mentions,
        longTail: false,
      };
    });
    const links: CanvasLink[] = model.edges
      .filter((e) => keep.has(e.source) && keep.has(e.target) && isEdgeRendered(e, floor))
      .map((e) => ({
        source: e.source,
        target: e.target,
        relation: e.relation,
        strength: e.strength,
        generic: e.generic,
        tier: e.tier,
        width: 0,
      }));
    return { nodes, links };
  }, [model, level, clusterId, floor]);

  /** What the floor hid at this level, reported upward so the footer can say
   *  so. Counted here because this is where the scope (whole graph vs one
   *  drilled cluster) is known. */
  const hiddenEdges = useMemo(() => {
    if (level === "clusters") return 0;
    const memberIdx =
      clusterId !== null ? (model.clusters[clusterId]?.members ?? []) : model.nodes.map((_, i) => i);
    const keep = new Set(memberIdx.map((i) => model.nodes[i].id));
    return model.edges.filter(
      (e) => keep.has(e.source) && keep.has(e.target) && !isEdgeRendered(e, floor),
    ).length;
  }, [model, level, clusterId, floor]);

  useEffect(() => {
    onStats?.({ hiddenEdges, floor });
  }, [hiddenEdges, floor, onStats]);

  useEffect(() => {
    selectedRef.current = selectedEntityId;
    drawRef.current();
  }, [selectedEntityId]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tokens = readTokens(wrap);
    const hueOf = (n: CanvasNode) => tokens.token(entityTypeToken(n.type));
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;

    function sizeCanvas() {
      const rect = wrap!.getBoundingClientRect();
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      width = rect.width;
      height = rect.height;
      canvas!.width = Math.round(width * dpr);
      canvas!.height = Math.round(height * dpr);
    }

    function draw() {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const t = transformRef.current;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx!.clearRect(0, 0, width, height);
      ctx!.save();
      ctx!.translate(t.x, t.y);
      ctx!.scale(t.k, t.k);

      const hovered = hoverRef.current;
      const sel = selectedRef.current;
      const focus = hovered ?? (sel ? (nodes.find((n) => n.id === sel) ?? null) : null);

      // Degree at or above which a node is named at rest. Small graphs label
      // everything; larger ones keep roughly the top slice legible instead of
      // stacking every name on top of its neighbour.
      const labelDegreeFloor =
        nodes.length <= LABEL_ALWAYS_MAX
          ? 0
          : Math.max(2, Math.round(nodes.reduce((a, n) => a + n.degree, 0) / nodes.length));
      const lit = new Set<string>();
      if (focus) {
        lit.add(focus.id);
        for (const l of links) {
          const s = l.source as CanvasNode;
          const g = l.target as CanvasNode;
          if (s.id === focus.id) lit.add(g.id);
          else if (g.id === focus.id) lit.add(s.id);
        }
      }
      const dimmed = focus !== null;

      ctx!.lineCap = "round";
      for (const l of links) {
        const s = l.source as CanvasNode;
        const g = l.target as CanvasNode;
        if (s.x == null || g.x == null) continue;
        const on = !dimmed || (lit.has(s.id) && lit.has(g.id));

        if (dimmed && on) {
          // The ONE place the accent appears on an edge: the selected node's
          // own connections. That is focus, which is what the accent is for.
          ctx!.strokeStyle = hexA(tokens.active, 0.9);
          ctx!.lineWidth = (1.2 + 1.6 * Math.min(1, l.strength)) / t.k;
        } else if (dimmed) {
          ctx!.strokeStyle = hexA(tokens.edgeWeak, 0.5);
          ctx!.lineWidth = 0.6 / t.k;
        } else if (l.width > 0) {
          // Aggregated cluster link: `width` was precomputed from the summed
          // weight of every relation crossing between the two communities.
          ctx!.strokeStyle = hexA(tokens.edgeMedium, 0.9);
          ctx!.lineWidth = l.width / t.k;
        } else {
          // THREE ENCODINGS, THREE CHANNELS, deliberately not overlapping:
          //   colour  = consolidation tier (L1 → L2 → L3), an ordinal ramp
          //   opacity = edge weight within that tier
          //   width   = typed relation vs generic co-occurrence
          //
          // Keeping them separate is what lets the graph be read: if tier also
          // changed width it would be indistinguishable from a strong
          // co-occurrence, and the question "is this settled knowledge or a
          // fresh guess" is a different question from "how sure are we".
          const w = Math.min(1, l.strength);
          ctx!.strokeStyle = hexA(tokens.token(tierToken(l.tier)), 0.5 + 0.5 * w);
          // Co-occurrence is background tissue — it says two entities appeared
          // together, which in a pairwise extractor is nearly free. A typed
          // relation is the ontology showing through, and reads above it.
          ctx!.lineWidth = (l.generic ? 0.5 + 0.4 * w : 1.3 + 1.1 * w) / t.k;
        }
        ctx!.beginPath();
        ctx!.moveTo(s.x, s.y!);
        ctx!.lineTo(g.x, g.y!);
        ctx!.stroke();
      }

      for (const n of nodes) {
        if (n.x == null || n.y == null) continue;
        const isLit = !dimmed || lit.has(n.id);
        const isSelected = n.id === sel;
        ctx!.globalAlpha = isLit ? 1 : 0.09;
        const hue = n.longTail ? tokens.muted : hueOf(n);

        ctx!.beginPath();
        ctx!.arc(n.x, n.y, n.r, 0, 2 * Math.PI);
        ctx!.fillStyle = hexA(hue, n.kind === "cluster" ? 0.2 : 0.3);
        ctx!.fill();
        ctx!.lineWidth = (isSelected ? 2.4 : 1.3) / t.k;
        ctx!.strokeStyle = isSelected ? tokens.active : hexA(hue, 0.95);
        ctx!.stroke();

        // Cluster labels are the whole point of the overview — an unlabelled
        // bubble is not navigable.
        //
        // Entity labels declutter by IMPORTANCE, not by a flat count rule. A
        // graph whose nodes are anonymous dots does not show what the corpus
        // knows, it shows that the corpus is busy — but labelling all of a
        // 5k-entity graph is a wordball. So the well-connected nodes are named
        // at rest (they are what someone is looking for), the long tail waits
        // for a zoom that separates it, and hover or selection always names
        // whatever is under the pointer whatever its degree.
        const labelWorthy = n.degree >= labelDegreeFloor;
        const showLabel =
          n.kind === "cluster" || isSelected || n === hovered || labelWorthy || t.k > 1.9;
        if (showLabel && isLit) {
          const fontPx = (n.kind === "cluster" ? 11 : 10) / t.k;
          ctx!.font = `${fontPx}px ui-sans-serif, system-ui, sans-serif`;
          ctx!.textAlign = "center";
          ctx!.textBaseline = "top";
          ctx!.fillStyle = hexA(tokens.fg, n.kind === "cluster" ? 0.92 : 0.75);
          const max = n.kind === "cluster" ? 22 : 26;
          const text = n.label.length > max ? `${n.label.slice(0, max - 1)}…` : n.label;
          ctx!.fillText(text, n.x, n.y + n.r + 3 / t.k);
        }
        ctx!.globalAlpha = 1;
      }

      ctx!.restore();
    }

    drawRef.current = draw;
    sizeCanvas();

    const sim = forceSimulation<CanvasNode, CanvasLink>(nodes)
      .force(
        "link",
        forceLink<CanvasNode, CanvasLink>(links)
          .id((d) => d.id)
          .distance((d) => (d.source as CanvasNode).r + (d.target as CanvasNode).r + 60)
          .strength((d) => Math.min(0.3, 0.06 + 0.5 * Math.min(1, d.strength))),
      )
      .force(
        "charge",
        forceManyBody<CanvasNode>()
          .strength((d) => -300 - 30 * d.r)
          .distanceMax(1400),
      )
      .force("center", forceCenter(width / 2, height / 2).strength(0.04))
      .force("x", forceX(width / 2).strength(0.012))
      .force("y", forceY(height / 2).strength(0.012))
      .force(
        "collide",
        forceCollide<CanvasNode>()
          .radius((d) => d.r + 14)
          .strength(1)
          .iterations(2),
      );
    simRef.current = sim;

    // Pre-warm off-screen so the first paint is already legible rather than a
    // pile at the centre, then let it settle live — the old file's "bounce".
    sim.stop();
    for (let i = 0; i < 60; i++) sim.tick();

    // `event.sourceEvent` is null for a transform we set ourselves and non-null
    // for a real wheel/drag. That is what separates "the view framed itself"
    // from "the user aimed it", and once the user has aimed it we stop moving
    // the camera under their hand.
    let userAimed = false;
    const zoomBehavior = zoom<HTMLCanvasElement, unknown>()
      .scaleExtent(SCALE_EXTENT)
      .on("zoom", (event) => {
        if (event.sourceEvent) userAimed = true;
        transformRef.current = event.transform;
        draw();
      });
    const sel = select(canvas);
    sel.call(zoomBehavior);

    /**
     * Frame every node, rather than opening at the identity transform.
     *
     * This MUST go through `zoomBehavior.transform` and not by assigning
     * `transformRef.current`. d3-zoom keeps its own copy of the current
     * transform on the selection's `__zoom` property; writing only the ref
     * leaves the two disagreeing, and the first pan gesture snaps back to
     * wherever d3 still believed the camera was.
     *
     * Applying the transform fires the zoom handler above, which repaints —
     * so callers must not also call `draw()`.
     */
    const frameNow = (): void => {
      const extent = coreExtentOf(nodes, FRAME_TRIM);
      if (!extent) return;
      const fit = fitTransform(extent, { width, height }, {
        padding: FRAME_PADDING,
        scaleExtent: SCALE_EXTENT,
      });
      sel.call(zoomBehavior.transform, zoomIdentity.translate(fit.x, fit.y).scale(fit.k));
    };

    if (reduceMotion) {
      for (let i = 0; i < 240; i++) sim.tick();
    } else {
      sim.alphaDecay(0.015).velocityDecay(0.3).alpha(0.9).restart();
      // Re-framing every tick keeps the whole corpus in view while the layout
      // spreads, instead of letting it grow off-screen and snapping at the end.
      // It costs one O(n) extent pass per tick against a full canvas repaint,
      // and it stops the moment the user takes the camera.
      sim.on("tick", () => {
        if (userAimed) draw();
        else frameNow();
      });
    }
    frameNow();
    // frameNow returns early on an empty extent, in which case nothing has
    // painted yet.
    if (nodes.length === 0) draw();

    const observer = new ResizeObserver(() => {
      sizeCanvas();
      sim.force("center", forceCenter(width / 2, height / 2).strength(0.04));
      if (!reduceMotion) sim.alpha(0.25).restart();
      draw();
    });
    observer.observe(wrap);

    return () => {
      observer.disconnect();
      sim.stop();
      sel.on(".zoom", null);
      simRef.current = null;
    };
  }, [nodes, links]);

  const nodeAt = useCallback(
    (sx: number, sy: number): CanvasNode | null => {
      const t = transformRef.current;
      const gx = (sx - t.x) / t.k;
      const gy = (sy - t.y) / t.k;
      let hit: CanvasNode | null = null;
      let best = Infinity;
      for (const n of nodes) {
        if (n.x == null || n.y == null) continue;
        const dx = n.x - gx;
        const dy = n.y - gy;
        const d = dx * dx + dy * dy;
        const rr = (n.r + 4) * (n.r + 4);
        if (d <= rr && d < best) {
          best = d;
          hit = n;
        }
      }
      return hit;
    },
    [nodes],
  );

  const downRef = useRef<[number, number] | null>(null);

  /** Entities at this level, ordered for keyboard traversal: strongest first,
   *  so arrowing through starts where an analyst would look. */
  const ordered = useMemo(
    () => (level === "entities" ? [...nodes].sort((a, b) => b.degree - a.degree) : []),
    [nodes, level],
  );

  const step = useCallback(
    (delta: number) => {
      if (ordered.length === 0) return;
      const at = ordered.findIndex((n) => n.id === selectedEntityId);
      const next =
        at === -1 ? (delta > 0 ? 0 : ordered.length - 1) : (at + delta + ordered.length) % ordered.length;
      selectEntity(ordered[next].id);
    },
    [ordered, selectedEntityId, selectEntity],
  );

  return (
    <div ref={wrapRef} className="absolute inset-0">
      <canvas
        ref={canvasRef}
        className="focus-visible:ring-ring size-full focus-visible:ring-2 focus-visible:outline-none"
        role="application"
        aria-label={
          level === "clusters"
            ? `Knowledge graph overview, ${nodes.length} clusters. Click a cluster to drill into its entities.`
            : `Knowledge graph, ${nodes.length} entities. Use the left and right arrow keys to move between them.`
        }
        tabIndex={0}
        onKeyDown={(e) => {
          if (level !== "entities") return;
          if (e.key === "ArrowRight" || e.key === "ArrowDown") {
            e.preventDefault();
            step(1);
          } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
            e.preventDefault();
            step(-1);
          } else if (e.key === "Escape") {
            selectEntity(null);
          }
        }}
        style={{ cursor: hover ? "pointer" : "grab" }}
        onPointerDown={(e) => {
          downRef.current = [e.nativeEvent.offsetX, e.nativeEvent.offsetY];
        }}
        onClick={(e) => {
          const down = downRef.current;
          const x = e.nativeEvent.offsetX;
          const y = e.nativeEvent.offsetY;
          if (down && (Math.abs(x - down[0]) > DRAG_SLOP_PX || Math.abs(y - down[1]) > DRAG_SLOP_PX)) {
            return;
          }
          const n = nodeAt(x, y);
          if (!n) {
            selectEntity(null);
            return;
          }
          if (n.kind === "cluster") onDrillIn(Number(n.id));
          else selectEntity(n.id === selectedEntityId ? null : n.id);
        }}
        onPointerMove={(e) => {
          const x = e.nativeEvent.offsetX;
          const y = e.nativeEvent.offsetY;
          const n = nodeAt(x, y);
          if (n !== hoverRef.current) {
            hoverRef.current = n;
            drawRef.current();
          }
          setHover(n ? { node: n, x, y } : null);
        }}
        onPointerLeave={() => {
          hoverRef.current = null;
          setHover(null);
          drawRef.current();
        }}
      />

      {hover ? (
        <div
          className="border-border bg-popover text-popover-foreground pointer-events-none absolute z-20 w-[230px] rounded-md border p-2.5 shadow-xl"
          style={{
            left: hover.x,
            top: hover.y,
            transform: `translate(${hover.x > 300 ? "calc(-100% - 14px)" : "14px"}, ${
              hover.y > 180 ? "calc(-100% - 14px)" : "14px"
            })`,
          }}
        >
          <p className="line-clamp-2 text-[12px] leading-relaxed font-medium">{hover.node.label}</p>
          {hover.node.kind === "cluster" ? (
            <p className="text-muted-foreground mono mt-1.5 text-[10px]">
              {hover.node.size} entities · mostly {hover.node.type}
              <br />
              click to drill in
            </p>
          ) : (
            <p className="text-muted-foreground mono mt-1.5 text-[10px]">
              {hover.node.type} · {hover.node.degree} connection
              {hover.node.degree === 1 ? "" : "s"}
              <br />
              salience {hover.node.salience.toFixed(2)} · {hover.node.mentions} mention
              {hover.node.mentions === 1 ? "" : "s"}
            </p>
          )}
        </div>
      ) : null}
    </div>
  );
}

export type { CanvasNode, CanvasLink, Level };

/**
 * Which level a corpus opens at.
 *
 * Size decides, not preference: cluster bubbles are an abstraction that only
 * pays for itself when the constellation underneath is unreadable.
 */
export function levelFor(model: UniverseModel): Level {
  return model.nodes.length > SMALL_GRAPH_MAX ? "clusters" : "entities";
}
