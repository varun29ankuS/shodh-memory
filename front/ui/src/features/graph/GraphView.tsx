import { useEffect, useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";
import type { Reachability } from "@/lib/api";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { useSession } from "@/stores/session";
import { useUniverse } from "./useUniverse";
import { EntityCanvas, levelFor, type Level } from "./EntityCanvas";
import {
  entityTypeToken,
  NAMED_ENTITY_TYPES,
  TIER_LABEL,
  TIER_ORDER,
  tierToken,
} from "./universe";

/**
 * Graph — what this corpus knows.
 *
 * This is NOT the lineage canvas on /recall, and the distinction is the reason
 * both exist. Lineage answers "how do these recalled memories cause each
 * other" and only exists once you have run a query. This answers "what does
 * this corpus know": the typed entities and the typed relations between them,
 * explorable before any question has been asked. An analyst arrives here to
 * find out what is in there; they go to Recall when they already know what to
 * ask.
 *
 * The ontology is the content, so it is visible rather than implied — node
 * colour is entity type, straight from the `--node-*` tokens the design system
 * defines for exactly these four classes, and the legend names them.
 */

function Breadcrumb({
  clusterLabel,
  onRoot,
}: {
  clusterLabel: string | null;
  onRoot: () => void;
}) {
  return (
    <nav
      aria-label="Graph level"
      className="text-muted-foreground absolute top-3 left-4 z-10 flex items-center gap-1 text-[12px]"
    >
      {clusterLabel === null ? (
        <span>What this corpus knows</span>
      ) : (
        <>
          <button
            type="button"
            onClick={onRoot}
            className="hover:text-foreground focus-visible:ring-ring rounded transition-colors focus-visible:ring-2 focus-visible:outline-none"
          >
            What this corpus knows
          </button>
          <ChevronRight aria-hidden="true" className="size-3 shrink-0" />
          <span className="text-foreground">{clusterLabel}</span>
        </>
      )}
    </nav>
  );
}

export function GraphView({ reach }: { reach: Reachability }) {
  const { model, error, isFetching, profile } = useUniverse(reach);
  const selectEntity = useSession((s) => s.selectEntity);

  // Drill state lives here rather than in the canvas: the breadcrumb and the
  // canvas are two views of it, and the canvas is remounted per level.
  const [clusterId, setClusterId] = useState<number | null>(null);
  /** Reported by the canvas so the footer states what was actually drawn. */
  const [stats, setStats] = useState<{
    hiddenEdges: number;
    floor: number;
    /** Set when the per-node edge budget cut lines. A picture that quietly
     *  dropped half its edges while looking complete is the thing the footer
     *  exists to prevent. */
    budget: string | null;
    tierCounts: Record<string, number> | null;
  }>({
    hiddenEdges: 0,
    floor: 0,
    budget: null,
    tierCounts: null,
  });

  // A new corpus invalidates a drill path taken through the old one.
  useEffect(() => {
    setClusterId(null);
    selectEntity(null);
  }, [profile, selectEntity]);

  const baseLevel: Level | null = model ? levelFor(model) : null;
  const level: Level = clusterId !== null ? "entities" : (baseLevel ?? "entities");

  /** Entity types actually present, ordered with the four the design system
   *  names first — those carry their own hue; everything else shares one. */
  const legend = useMemo(() => {
    if (!model) return [];
    const counts = new Map<string, number>();
    for (const n of model.nodes) counts.set(n.type, (counts.get(n.type) ?? 0) + 1);
    const named = NAMED_ENTITY_TYPES.filter((t) => counts.has(t)).map((t) => ({
      label: t as string,
      token: entityTypeToken(t),
      count: counts.get(t)!,
    }));
    const otherCount = [...counts.entries()]
      .filter(([t]) => !(NAMED_ENTITY_TYPES as readonly string[]).includes(t))
      .reduce((a, [, c]) => a + c, 0);
    return otherCount > 0
      ? [...named, { label: "Other", token: "--chart-5", count: otherCount }]
      : named;
  }, [model]);

  /**
   * Tier populations, counted over the edges that are actually DRAWN.
   *
   * Deliberately not the server's l1_edges/l2_edges/l3_edges: those live on
   * `/api/graph/data/{user_id}`, a different endpoint that truncates at 200
   * relationships per tier and reports the truncated numbers as totals
   * (src/handlers/visualization.rs:378-392, :458-459). Counting what this
   * canvas drew means the legend and the picture can never disagree.
   */
  /* COUNTED BY THE CANVAS, NOT RECOMPUTED HERE.
     This used to walk model.edges and count everything that passed the
     co-occurrence floor. That was correct until the per-node edge budget
     landed, at which point the key claimed 411 L3 edges over a picture drawing
     220 — the legend un-stating the cut the footer beside it had just stated.
     The canvas is the only thing that knows what survived BOTH filters, so it
     reports, and this reads. Null at cluster level, where a drawn line is an
     aggregate across many tiers and has no tier to count. */
  const tierCounts = stats.tierCounts;

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="The graph is built from the entity store, which needs the server running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile to show"
        body="The graph is per-profile, and none exists yet."
      />
    );
  }

  if (error) {
    return (
      <EmptyState
        size="page"
        title="Graph failed to load"
        body="The entity universe did not come back."
        more="The graph endpoint returns the whole corpus in one response, so it fails all at once rather than partially — there is no partial graph to show."
      />
    );
  }

  if (!model) {
    return (
      <EmptyState
        size="page"
        title={isFetching ? "Building the graph" : "Nothing to draw"}
        body={
          isFetching
            ? "One request for every entity and relation here."
            : "No entity universe came back for this profile."
        }
      />
    );
  }

  if (model.nodes.length === 0) {
    return (
      <EmptyState
        size="page"
        title="The graph shows what this memory knows about"
        body="Entities and their links appear here as memories are written and the extraction pipeline types what is in them."
        more="Entities appear as memories are written and the extraction pipeline types what is in them, so the graph fills in behind the corpus rather than being built separately."
      />
    );
  }

  const cluster = clusterId !== null ? (model.clusters[clusterId] ?? null) : null;
  // Split into the number and the noun so the strip can set the digits in the
  // mono face and the word in the text face — the number is what the eye is
  // hunting for, and "341 entities" rendered wholly in mono buries it.
  const shownCount =
    level === "clusters" ? model.clusters.length : (cluster?.size ?? model.nodes.length);
  const shownNoun =
    level === "clusters"
      ? `cluster${model.clusters.length === 1 ? "" : "s"} of ${model.totalEntities} entities`
      : `entit${shownCount === 1 ? "y" : "ies"} of ${model.totalEntities}`;

  return (
    <section className="relative h-full min-h-0 min-w-0 overflow-hidden">
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      <EntityCanvas
        // Remount on level change: the node set changes completely, and
        // reusing one simulation across that is how nodes end up carrying
        // positions from a graph they are no longer part of.
        key={`${level}:${clusterId ?? "root"}`}
        model={model}
        level={level}
        clusterId={clusterId}
        onDrillIn={(id) => {
          setClusterId(id);
          selectEntity(null);
        }}
        onStats={setStats}
      />

      <Breadcrumb
        clusterLabel={cluster ? cluster.label : null}
        onRoot={() => {
          setClusterId(null);
          selectEntity(null);
        }}
      />

      <div className="pointer-events-none absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center justify-between gap-x-6 gap-y-2"
      style={{ paddingRight: "var(--overlay-dock-inset, 0px)" }}>
        {/* The legend teaches the graph's three encodings, which is the only
            thing that makes them readable: node HUE is entity type, edge HUE is
            consolidation tier, and the two never share a palette — categorical
            colour belongs to nodes, the tier ramp is one desaturated cool
            family so it reads as a progression rather than as more categories. */}
        <div className="flex flex-col gap-1.5">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
            {legend.map((l) => (
              <span
                key={l.label}
                className="text-muted-foreground flex items-center gap-1.5 text-[11px]"
              >
                <span className="size-2 rounded-full" style={{ background: `var(${l.token})` }} />
                {l.label}
                <span className="text-muted-foreground/60">{l.count}</span>
              </span>
            ))}
            {/* The third encoding, moved here from the status strip on the
                right. It belongs with the other two: a key is where a reader
                decodes the picture, and "size = mentions" is a decoding, not a
                status. Two dots of different sizes carry it without a word. */}
            <span className="text-muted-foreground/70 flex items-center gap-1.5 text-[11px]">
              <span className="bg-muted-foreground/50 size-1.5 rounded-full" />
              <span className="bg-muted-foreground/50 size-2.5 rounded-full" />
              size = mentions
            </span>
          </div>
          {level === "entities" && tierCounts ? (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
              {TIER_ORDER.map((t) => (
                <span
                  key={t}
                  className="text-muted-foreground flex items-center gap-1.5 text-[11px]"
                >
                  {/* A bar, not a dot: these encode edges, and matching the
                      mark to what it describes is why the key is readable at
                      a glance beside the node dots above. */}
                  <span
                    className="h-[2px] w-4 rounded-full"
                    style={{ background: `var(${tierToken(t)})` }}
                  />
                  {TIER_LABEL[t]}
                  <span className="text-muted-foreground/60">{tierCounts[t] ?? 0}</span>
                </span>
              ))}
            </div>
          ) : null}
        </div>
        {/* EVERY REDUCTION BETWEEN THE CORPUS AND THE PIXELS IS STILL STATED —
            that rule has not moved. What has changed is that each reduction is
            now a token rather than a clause, and the exact co-occurrence floor
            that produced one of them sits behind the icon: "12 weak edges
            hidden" is the fact a reader needs on screen, "below 0.14" is how it
            was computed. The endpoint is uncapped, so any cut happened on this
            side, and a view that quietly drops most of a graph is worse than
            one that admits it.

            The two gestures and the size encoding left the strip entirely. Size
            is a legend entry — an unexplained size channel is decoration that
            looks like data — and it now sits with the other two encodings in
            the key on the left, where a reader decoding the picture is already
            looking. */}
        <Meta className="shrink-0">
          <Stat value={shownCount} label={shownNoun} />
          <Stat value={model.totalConnections} label="relations" />
          {model.edgesDropped > 0 ? (
            <Stat value={model.edgesDropped} label="dropped to budget" />
          ) : null}
          {stats.budget ? <span>{stats.budget}</span> : null}
          {stats.hiddenEdges > 0 ? (
            <Stat value={stats.hiddenEdges} label="weak edges hidden" />
          ) : null}
          <InfoHint label="canvas controls" align="right" side="up">
            Scroll to zoom, and{" "}
            {level === "clusters"
              ? "click a cluster to drill into the entities inside it."
              : "click an entity to inspect it."}
            {stats.hiddenEdges > 0 ? (
              <span className="mt-1.5 block">
                Hidden edges are co-occurrences weaker than {stats.floor.toFixed(2)} — the pairs
                that appear together too rarely to be worth a line.
              </span>
            ) : null}
          </InfoHint>
        </Meta>
      </div>
    </section>
  );
}
