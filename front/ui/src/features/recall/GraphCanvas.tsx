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
import type { RecallMemory, RecallLineageEdge } from "@/lib/api";
import { coreExtentOf, fitTransform } from "@/lib/view/fit";
import { useSession } from "@/stores/session";
import { relClass, relName, type RelationClass } from "./relation";
import {
  memoryTier,
  MEMORY_TIER_LABEL,
  MEMORY_TIER_MARK,
  MEMORY_TIER_SELECTED_FILL,
  MEMORY_TIER_SELECTED_RING,
  type MemoryTier,
} from "./tier";

/**
 * The knowledge-graph canvas.
 *
 * WHAT IT DRAWS, and why that is not what the old dashboard drew.
 *
 * NOTE ON CITATIONS: front/index.html was the single-file vanilla dashboard and
 * has since been deleted. Every line number below resolves in git history, not
 * in the tree; they are kept because they are the provenance of what was ported
 * and, more importantly, of what deliberately was not.
 *
 * The vanilla dashboard (front/index.html) fed from
 * `GET /api/graph/{user}/universe`, whose nodes are ENTITIES — `UniverseStar`
 * is `{id, name, entity_type, salience, mention_count, is_proper_noun, …}`
 * (src/graph_memory.rs:7279-7291) and carries no memory id. It is also not
 * recall-scoped: `get_universe` returns every entity and relationship with no
 * cap (src/graph_memory.rs:7104-7220), which is why that file needed in-browser
 * Louvain clustering, a 40k edge budget and cluster drill-in — a 2k-node /
 * 87k-edge hairball is unreadable, and 126k raw edges froze its build for >75s
 * (front/index.html:728-735).
 *
 * This canvas draws the CURRENT RECALL RESULT instead: nodes are the returned
 * memories, edges are `RecallResponse.lineage`. Two things follow, and both are
 * requirements rather than preferences:
 *
 *  - `RecallLineageEdge.from`/`.to` ARE memory ids (src/handlers/recall.rs:982-1001),
 *    so clicking a node selects a real memory in the Inspector. Entity stars
 *    could never do that — there is no entity→memory mapping on that payload.
 *  - The server only emits a lineage edge when BOTH endpoints are in the
 *    recalled set (recall.rs:962-964, :981, :994). Every edge drawn here is
 *    therefore between two nodes that are both on screen; there are no dangling
 *    endpoints to resolve and no second request to make.
 *
 * The clustering machinery is deliberately NOT ported: a ~25-memory result set
 * is not the problem clustering solved. Retained from the old implementation
 * are the parts that were about canvas craft rather than corpus scale — the
 * d3.zoom transform driving a manual paint (front/index.html:596-598), HiDPI
 * sizing (:894-899), squared-distance hit-testing (:1552-1559) and the
 * drag-vs-click threshold (:1568).
 *
 * CONSOLIDATION is drawn, not just listed. Every node carries `RecallMemory.tier`
 * (src/handlers/types.rs:249, `format!("{:?}", m.tier)` at src/handlers/recall.rs:830),
 * and until now that reached the screen only as a text field in the Inspector —
 * one memory at a time, which is the one form in which it answers nothing. The
 * useful question is comparative ("is this cluster settled knowledge or this
 * morning's context?"), so it is encoded on the mark itself as how PRESENT the
 * node is drawn. ./tier.ts holds the ramp and the argument for that channel.
 *
 * PROVENANCE is the point. An edge is never just a line: hovering or selecting
 * surfaces the relation type and the confidence the server actually returned,
 * because "why is this connected" is the question a hairball cannot answer.
 * Nothing here is invented — `relation` and `confidence` are the only two
 * fields `RecallLineageEdge` has.
 */

/** Zoom limits, shared by the gesture handler and the auto-frame so the frame
 *  can never ask for a scale d3 would silently clamp. */
const SCALE_EXTENT: [number, number] = [0.2, 6];

/** Margin left around the framed corpus, in screen pixels. Enough that node
 *  labels, which are drawn outside the node radius, are not cut by the edge. */
const FRAME_PADDING = 48;

/** Fraction trimmed from each end of each axis before framing, so a few
 *  stranded nodes cannot set the camera for the whole result set. Trimmed
 *  nodes are still drawn, just outside the opening view. */
const FRAME_TRIM = 0.06;

interface GraphNode extends SimulationNodeDatum {
  id: string;
  label: string;
  /** Free-form `memory_type` from the wire; `null` when the writer set none. */
  type: string | null;
  score: number;
  r: number;
  color: string;
  degree: number;
  /** Consolidation tier, normalised from `RecallMemory.tier`. Encoded as how
   *  PRESENT the node is drawn — see ./tier.ts for why it is not a hue. */
  tier: MemoryTier;
}

interface GraphLink extends SimulationLinkDatum<GraphNode> {
  source: GraphNode | string;
  target: GraphNode | string;
  relation: string;
  confidence: number;
  cls: RelationClass;
}

/** Drag further than this between pointerdown and click and it was a pan. */
const DRAG_SLOP_PX = 4;

/** Resolve CSS custom properties to concrete colours.
 *
 *  The palette lives in index.css as design tokens (`--chart-1..5` at
 *  index.css:116-120, `--node-active` at :147) and must not be duplicated as
 *  literals here — a canvas cannot read `var()`, but it can read what the
 *  cascade computed. Read once against the live element so a token edit in CSS
 *  reaches the canvas with no code change. */
function readTokens(el: HTMLElement) {
  const cs = getComputedStyle(el);
  const v = (name: string, fallback: string) => cs.getPropertyValue(name).trim() || fallback;
  return {
    // Category hues for memory_type. Deliberately `--chart-*` and not
    // `--node-*`: the `--node-technology/org/location/person` tokens
    // (index.css:143-146) name ENTITY types, and these nodes are memories.
    // Reusing them would assert a mapping the data does not have — same hues,
    // honest name.
    chart: [
      v("--chart-1", "#a599ff"),
      v("--chart-2", "#4ea7fc"),
      v("--chart-3", "#4cb782"),
      v("--chart-4", "#ec6f9e"),
      v("--chart-5", "#39b8b0"),
    ],
    // Selection only. index.css:136-138 settles this: "node-active means the
    // current selection reached this — that is focus, so it takes the accent."
    active: v("--node-active", "#f4622e"),
    muted: v("--muted-foreground", "#8a8f98"),
    border: v("--border", "#23252a"),
  };
}

/** `#rrggbb` → `rgba(r,g,b,a)`. Ported from front/index.html:1548-1549. */
function hexA(hex: string, a: number): string {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex.trim());
  if (!m) return `rgba(138,143,152,${a})`;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
}

interface Hover {
  node: GraphNode;
  x: number;
  y: number;
}

export function GraphCanvas({
  memories,
  lineage,
}: {
  memories: RecallMemory[];
  lineage: RecallLineageEdge[];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const selectedId = useSession((s) => s.selectedMemoryId);
  const selectMemory = useSession((s) => s.select);
  const [hover, setHover] = useState<Hover | null>(null);

  // Mutable render state. These are refs, not state: the simulation ticks tens
  // of times a second and a setState per tick would re-render the tree for a
  // paint that happens on the canvas anyway.
  const transformRef = useRef<ZoomTransform>(zoomIdentity);
  const simRef = useRef<Simulation<GraphNode, GraphLink> | null>(null);
  const hoverRef = useRef<GraphNode | null>(null);
  const selectedRef = useRef<string | null>(selectedId);
  const drawRef = useRef<() => void>(() => {});

  /** Distinct `memory_type` values, sorted, each mapped to a chart hue.
   *
   *  The wire type is `Option<String>`, but the values are NOT free-form: recall
   *  serialises `format!("{:?}", experience_type)` (src/handlers/recall.rs:822),
   *  so they are the Debug renderings of a closed enum, Title-Cased —
   *  `Observation`, `Decision`, `Learning`, `Error`, `Discovery`, `Pattern`,
   *  `Context`, `Task`, `CodeEdit`, `FileAccess`, `Search`, `Command`,
   *  `Conversation`, `Intention` (the set `/api/remember` validates against).
   *
   *  Hues are still assigned from the values actually PRESENT rather than from
   *  that list. Fourteen categories against five chart hues would collide
   *  arbitrarily, and a legend naming eleven types a corpus does not contain
   *  explains nothing. Sorting what is present is deterministic — the same
   *  result set always yields the same legend — and degrades safely if the enum
   *  gains a variant. */
  const types = useMemo(() => {
    const seen = new Set<string>();
    for (const m of memories) if (m.experience.memory_type) seen.add(m.experience.memory_type);
    return [...seen].sort();
  }, [memories]);

  const { nodes, links } = useMemo(() => {
    const degree = new Map<string, number>();
    const present = new Set(memories.map((m) => m.id));
    // Guard the endpoints anyway. The handler only emits edges whose ends are
    // both in the recalled set, but a node the layout does not know would make
    // forceLink throw rather than skip, and a crash beats nothing on screen
    // only in a debugger.
    const usable = lineage.filter((e) => present.has(e.from) && present.has(e.to));
    for (const e of usable) {
      degree.set(e.from, (degree.get(e.from) ?? 0) + 1);
      degree.set(e.to, (degree.get(e.to) ?? 0) + 1);
    }

    const nodes: GraphNode[] = memories.map((m) => {
      const deg = degree.get(m.id) ?? 0;
      const typeIndex = m.experience.memory_type ? types.indexOf(m.experience.memory_type) : -1;
      return {
        id: m.id,
        label: m.experience.content,
        type: m.experience.memory_type,
        score: m.score,
        // Radius carries retrieval score, the one ranking signal every result
        // has. Sub-linear so a strong hit reads as larger without a weak one
        // vanishing to a dot.
        r: 6 + Math.sqrt(Math.max(0, m.score)) * 7,
        color: typeIndex >= 0 ? `chart:${typeIndex}` : "muted",
        degree: deg,
        tier: memoryTier(m.tier),
      };
    });

    const links: GraphLink[] = usable.map((e) => ({
      source: e.from,
      target: e.to,
      relation: e.relation,
      confidence: e.confidence,
      cls: relClass(e.relation),
    }));

    return { nodes, links };
  }, [memories, lineage, types]);

  useEffect(() => {
    selectedRef.current = selectedId;
    drawRef.current();
  }, [selectedId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tokens = readTokens(wrap);
    const hue = (n: GraphNode) =>
      n.color === "muted" ? tokens.muted : tokens.chart[Number(n.color.slice(6)) % tokens.chart.length];

    // A settled layout rather than a live one when motion is reduced. The
    // global rule in index.css:242-249 collapses CSS transitions, but a force
    // simulation is JS and would keep moving regardless; the honest equivalent
    // is to run it to rest off-screen and paint the result once.
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
      return dpr;
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
      // The focus node is what dims everything else: an explicit selection wins
      // over a transient hover, so moving the pointer away does not tear down
      // the neighbourhood someone just clicked to study.
      const focus = hovered ?? (sel ? nodes.find((n) => n.id === sel) ?? null : null);
      const litIds = new Set<string>();
      if (focus) {
        litIds.add(focus.id);
        for (const l of links) {
          const s = l.source as GraphNode;
          const tg = l.target as GraphNode;
          if (s.id === focus.id) litIds.add(tg.id);
          else if (tg.id === focus.id) litIds.add(s.id);
        }
      }
      const dimmed = focus !== null;

      ctx!.lineCap = "round";
      for (const l of links) {
        const s = l.source as GraphNode;
        const tg = l.target as GraphNode;
        if (s.x == null || tg.x == null) continue;
        const on = !dimmed || (litIds.has(s.id) && litIds.has(tg.id));

        if (dimmed && on) {
          // The lit pathway reads as energised, not merely recoloured — this is
          // the connection someone is asking about.
          ctx!.strokeStyle = hexA(tokens.active, 0.85);
          ctx!.lineWidth = (1 + Math.min(2, l.confidence * 2)) / transformRef.current.k;
        } else if (dimmed) {
          ctx!.strokeStyle = hexA(tokens.muted, 0.06);
          ctx!.lineWidth = 0.7 / transformRef.current.k;
        } else {
          // Colour by relation CLASS. The causal spine is the product's signal,
          // so it is bright; typed relations are cool; generic co-occurrence is
          // faint. Ported intent from front/index.html:1424-1437.
          ctx!.strokeStyle =
            l.cls === "causal"
              ? hexA(tokens.active, 0.85)
              : l.cls === "typed"
                ? hexA(tokens.chart[1], 0.5)
                : l.cls === "loc"
                  ? hexA(tokens.chart[2], 0.45)
                  : hexA(tokens.muted, 0.22);
          ctx!.lineWidth =
            (l.cls === "causal" ? 1.8 : 0.7 + Math.min(1.5, l.confidence * 1.4)) /
            transformRef.current.k;
        }

        ctx!.beginPath();
        ctx!.moveTo(s.x, s.y!);
        ctx!.lineTo(tg.x, tg.y!);
        ctx!.stroke();
      }

      for (const n of nodes) {
        if (n.x == null || n.y == null) continue;
        const isLit = !dimmed || litIds.has(n.id);
        const isSelected = n.id === sel;
        ctx!.globalAlpha = isLit ? 1 : 0.1;

        // Consolidation tier, as presence. A working memory is a faint outline
        // and a long-term one is filled and firmly ringed, so "how settled is
        // this?" is answerable without reading the legend. Selection ADDS to
        // the tier's step rather than overwriting it — the accent stroke is
        // what makes a selection unmistakable, so the tier survives being
        // clicked. Rationale and the ramp itself live in ./tier.ts.
        const mark = MEMORY_TIER_MARK[n.tier];

        ctx!.beginPath();
        ctx!.arc(n.x, n.y, n.r, 0, 2 * Math.PI);
        ctx!.fillStyle = hexA(
          hue(n),
          isSelected ? Math.min(1, mark.fill + MEMORY_TIER_SELECTED_FILL) : mark.fill,
        );
        ctx!.fill();
        ctx!.lineWidth =
          (isSelected ? mark.ring + MEMORY_TIER_SELECTED_RING : mark.ring) /
          transformRef.current.k;
        ctx!.strokeStyle = isSelected ? tokens.active : hexA(hue(n), mark.ringAlpha);
        ctx!.stroke();

        // An isolated memory is a real finding, not a rendering gap: it means
        // nothing in this result set causally connects to it. A dashed ring
        // says so without a label.
        if (n.degree === 0) {
          ctx!.setLineDash([3 / transformRef.current.k, 4 / transformRef.current.k]);
          ctx!.lineWidth = 1 / transformRef.current.k;
          ctx!.strokeStyle = hexA(tokens.muted, 0.5);
          ctx!.beginPath();
          ctx!.arc(n.x, n.y, n.r + 4 / transformRef.current.k, 0, 2 * Math.PI);
          ctx!.stroke();
          ctx!.setLineDash([]);
        }
        ctx!.globalAlpha = 1;
      }

      ctx!.restore();
    }

    drawRef.current = draw;

    const dpr = sizeCanvas();
    void dpr;

    const sim = forceSimulation<GraphNode, GraphLink>(nodes)
      .force(
        "link",
        forceLink<GraphNode, GraphLink>(links)
          .id((d) => d.id)
          // Confidence pulls: a relation the server is sure about holds its
          // endpoints closer than a tentative one, so spatial proximity means
          // something rather than being an artefact of iteration order.
          .distance((d) => (d.source as GraphNode).r + (d.target as GraphNode).r + 70)
          .strength((d) => Math.min(0.35, 0.06 + 0.5 * d.confidence)),
      )
      .force("charge", forceManyBody<GraphNode>().strength((d) => -300 - 30 * d.r).distanceMax(1200))
      .force("center", forceCenter(width / 2, height / 2).strength(0.05))
      .force("x", forceX(width / 2).strength(0.014))
      .force("y", forceY(height / 2).strength(0.014))
      .force("collide", forceCollide<GraphNode>().radius((d) => d.r + 14).strength(1).iterations(2));

    simRef.current = sim;

    // Zoom drives the transform; the paint is manual. d3 only supplies the
    // gesture handling and the transform algebra.
    //
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
      sim.stop();
      // 300 ticks is d3's own default run length to convergence at the default
      // alpha decay; running them synchronously yields the settled layout with
      // no animation frames at all.
      for (let i = 0; i < 300; i++) sim.tick();
      frameNow();
      // frameNow returns early on an empty extent, in which case nothing has
      // painted yet.
      if (nodes.length === 0) draw();
    } else {
      sim.alphaDecay(0.015).velocityDecay(0.32).alpha(1).restart();
      // Re-framing every tick keeps the whole corpus in view while the layout
      // spreads, instead of letting it grow off-screen and snapping at the end.
      // It costs one O(n) extent pass per tick against a full canvas repaint,
      // and it stops the moment the user takes the camera.
      sim.on("tick", () => {
        if (userAimed) draw();
        else frameNow();
      });
    }

    const observer = new ResizeObserver(() => {
      sizeCanvas();
      sim.force("center", forceCenter(width / 2, height / 2).strength(0.05));
      sim.force("x", forceX(width / 2).strength(0.014));
      sim.force("y", forceY(height / 2).strength(0.014));
      if (!reduceMotion) sim.alpha(0.3).restart();
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

  /** Hit-test in graph coordinates. Ported from front/index.html:1552-1559 —
   *  squared distance, nearest wins, with a few px of slack so a small node is
   *  still clickable. */
  const nodeAt = useCallback(
    (sx: number, sy: number): GraphNode | null => {
      const t = transformRef.current;
      const gx = (sx - t.x) / t.k;
      const gy = (sy - t.y) / t.k;
      let hit: GraphNode | null = null;
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

  return (
    <div ref={wrapRef} className="absolute inset-0">
      <canvas
        ref={canvasRef}
        className="size-full"
        style={{ cursor: hover ? "pointer" : "grab" }}
        onPointerDown={(e) => {
          downRef.current = [e.nativeEvent.offsetX, e.nativeEvent.offsetY];
        }}
        onClick={(e) => {
          const down = downRef.current;
          const x = e.nativeEvent.offsetX;
          const y = e.nativeEvent.offsetY;
          // A click that travelled is a pan, not a selection.
          if (down && (Math.abs(x - down[0]) > DRAG_SLOP_PX || Math.abs(y - down[1]) > DRAG_SLOP_PX)) {
            return;
          }
          const n = nodeAt(x, y);
          selectMemory(n ? n.id : null);
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

      {hover ? <GraphTooltip hover={hover} links={links} /> : null}
    </div>
  );
}

/**
 * The provenance tooltip.
 *
 * This is the "why is this connected" answer, and it is the reason the canvas
 * is worth having. It reports only what `RecallLineageEdge` carries — the
 * relation and the confidence — plus the direction, which is implied by which
 * end of the edge the hovered node sits on.
 */
function GraphTooltip({ hover, links }: { hover: Hover; links: GraphLink[] }) {
  const { node } = hover;
  const touching = links.filter((l) => {
    const s = l.source as GraphNode;
    const t = l.target as GraphNode;
    return s.id === node.id || t.id === node.id;
  });

  // Strongest evidence first; a tooltip that scrolls is a panel.
  const shown = [...touching].sort((a, b) => b.confidence - a.confidence).slice(0, 5);

  return (
    <div
      // Flip before the edge rather than after: at the right and bottom of the
      // stage a fixed offset would push this outside the clipped section and
      // silently truncate the provenance, which is the one thing here that
      // must not be lost.
      className="border-border bg-popover text-popover-foreground pointer-events-none absolute z-20 w-[260px] rounded-md border p-2.5 shadow-xl"
      style={{
        left: hover.x,
        top: hover.y,
        transform: `translate(${hover.x > 320 ? "calc(-100% - 14px)" : "14px"}, ${
          hover.y > 200 ? "calc(-100% - 14px)" : "14px"
        })`,
      }}
    >
      <p className="line-clamp-3 text-[12px] leading-relaxed">{node.label}</p>
      <p className="text-muted-foreground mono mt-1.5 text-[10px]">
        {/* The tier is named as well as drawn. The ramp answers "is this
            settled?" at a glance across the whole canvas; a reader who has
            singled one node out wants the step by name, not by comparison. */}
        {node.type ?? "untyped"} · {MEMORY_TIER_LABEL[node.tier]} · score{" "}
        {node.score.toFixed(3)}
      </p>

      {shown.length > 0 ? (
        <div className="border-border mt-2 border-t pt-2">
          <p className="text-muted-foreground/70 mb-1 text-[10px] tracking-wide uppercase">
            Why it connects
          </p>
          {shown.map((l, i) => {
            const s = l.source as GraphNode;
            const outgoing = s.id === node.id;
            return (
              <p key={i} className="mono text-[10px] leading-relaxed">
                <span className={l.cls === "causal" ? "text-primary" : "text-muted-foreground"}>
                  {outgoing ? "→" : "←"} {relName(l.relation)}
                </span>{" "}
                <span className="text-muted-foreground/70">{l.confidence.toFixed(2)}</span>
              </p>
            );
          })}
          {touching.length > shown.length ? (
            <p className="text-muted-foreground/60 mt-1 text-[10px]">
              +{touching.length - shown.length} more
            </p>
          ) : null}
        </div>
      ) : (
        <p className="text-muted-foreground/70 mt-2 text-[10px] leading-relaxed">
          Nothing in this result set causally connects to it.
        </p>
      )}
    </div>
  );
}

/** The legend, exported so the stage can render it beside the interaction
 *  hints. It lists the `memory_type` values actually present, in the same
 *  sorted order the canvas assigns hues in — the two cannot drift because both
 *  derive from the same sort. */
export function useMemoryTypes(memories: RecallMemory[]): string[] {
  return useMemo(() => {
    const seen = new Set<string>();
    for (const m of memories) if (m.experience.memory_type) seen.add(m.experience.memory_type);
    return [...seen].sort();
  }, [memories]);
}
