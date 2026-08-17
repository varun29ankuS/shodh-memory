import { useMemo } from "react";
import type { Reachability } from "@/lib/api";
import { corpusToRecallMemory, useCorpus } from "@/lib/api/corpus";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { useRecall } from "@/features/recall/useRecall";
import { useMemoryTypes } from "@/features/recall/GraphCanvas";
import { GeoMap } from "./GeoMap";

/**
 * Geo — everywhere this profile's memory happened, with the current recall
 * result highlighted on top of it.
 *
 * The map is populated the moment the destination opens: the located memories
 * in the corpus page are drawn as quiet context points, because "where do I
 * have memory?" is a question with an answer before any search is typed.
 * Running a search does not swap the map out — it turns the matching points up
 * and the rest down, so a result is always seen against the corpus it came
 * from.
 *
 * The context layer is the newest `CORPUS_LIMIT` memories, not the whole
 * corpus — `useCorpus` fetches one page (see lib/api/corpus.ts). On a profile
 * larger than that page the map shows where recent memory happened, which is
 * not the same as everywhere memory has ever happened. Recall results are
 * plotted on top regardless of age, so a search still reaches the rest.
 *
 * Both data sources are shared cache entries (useCorpus, useRecall): this view
 * issues no retrieval of its own.
 */

export function GeoView({ reach }: { reach: Reachability }) {
  const corpus = useCorpus(reach);
  const { data, error, isFetching, profile, query } = useRecall(reach);

  const hasQuery = query.trim().length > 0;
  const results = useMemo(() => (hasQuery ? (data?.memories ?? []) : []), [hasQuery, data]);

  // The plotted set: recall results first (they carry scores and therefore
  // size), then the located memories from the corpus page that are not already
  // a result.
  const { plotted, dimmed } = useMemo(() => {
    const resultIds = new Set(results.map((m) => m.id));
    const context = (corpus.data?.memories ?? [])
      .filter((m) => m.geo_location)
      .filter((m) => !resultIds.has(m.id))
      .map(corpusToRecallMemory);
    if (!hasQuery || results.length === 0) {
      // No active answer: the corpus IS the map. Nothing is dimmed — these
      // points are not losing to anything.
      return { plotted: context, dimmed: undefined };
    }
    return {
      plotted: [...results, ...context],
      dimmed: new Set(context.map((m) => m.id)),
    };
  }, [corpus.data, results, hasQuery]);

  const types = useMemoryTypes(plotted);
  const located = plotted.filter((m) => m.experience.geo_location);
  const matched = results.filter((m) => m.experience.geo_location);

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="The map draws from memory, which needs the server running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile selected"
        body="The map is per-profile, and none exists yet."
      />
    );
  }

  if (corpus.isFetching && !corpus.data) {
    return <EmptyState size="page" title="Loading corpus" body="Placing located memories." />;
  }

  if (located.length === 0) {
    return (
      // Factual, not apologetic: the reason is a property of the data.
      <EmptyState
        size="page"
        title="No coordinates in this corpus"
        body="Nothing stored here carries a position."
        more="A memory only carries coordinates when whatever wrote it supplied them — imported corpora like GDELT do, session captures do not. The first one that does appears here without any search."
      />
    );
  }

  return (
    // `h-full`, not `flex-1` — this view is a direct child of `main`, which is
    // not a flex container (see the history of this comment in git blame).
    <section className="relative h-full min-h-0 min-w-0 overflow-hidden">
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      <GeoMap memories={plotted} types={types} dimmed={dimmed} />

      {/* NO TITLE HERE ANY MORE. "Where this profile's memory happened" was a
          restatement of the destination's own caption in the bar directly
          above it — the same words, twice, ten pixels apart. Duplicating a
          visible label is information pollution, and on a map it was competing
          with the only thing in this corner worth reading.

          What survives is the one thing the map does that is NOT visible from
          looking at it: a search does not swap the points out, it turns the
          matching ones up. Without that cue, a map that stays put after a
          search reads as a map that ignored it — a documented, real confusion,
          so it stays on screen rather than moving behind an affordance. It is
          shown only before a query, because after one the dimming and the
          matched count say it better than a sentence could. */}
      {!hasQuery ? (
        <div className="pointer-events-none absolute top-3 left-4 z-10">
          <span className="text-muted-foreground/70 text-[11px]">
            Search raises the points that match
          </span>
        </div>
      ) : null}

      <div
        className="pointer-events-none absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center justify-between gap-x-6 gap-y-2"
        style={{ paddingRight: "var(--overlay-dock-inset, 0px)" }}
      >
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {types.map((t, i) => (
            <span key={t} className="text-muted-foreground flex items-center gap-1.5 text-[11px]">
              <span
                className="size-2 rounded-full"
                style={{ background: `var(--chart-${(i % 5) + 1})` }}
              />
              {t}
            </span>
          ))}
        </div>
        {/* Counts on the right, gestures behind the icon beside them. What is
            ON the map is data and belongs on screen; how to move the camera is
            learned once, and printing it under every session is three of the
            strip's six words spent on something nobody reads twice. */}
        <Meta className="shrink-0">
          {/* "showing all" is not decoration: with no matched count beside it,
              a bare "search failed" leaves the reader to guess whether the
              points on screen are a partial answer or the whole corpus. */}
          {hasQuery && error ? <span className="text-warn">search failed — showing all</span> : null}
          {hasQuery && !error && isFetching && !data ? <span>searching…</span> : null}
          {hasQuery && !error && !(isFetching && !data) ? (
            <Stat value={matched.length} label="matched" />
          ) : null}
          <Stat value={located.length} label="located" />
          <InfoHint label="map controls" align="right" side="up">
            Scroll to zoom, drag to pan, click a point to inspect it. A search does not change
            which points are drawn — it raises the ones that match and dims the rest, so an answer
            is always seen against the corpus it came from.
          </InfoHint>
        </Meta>
      </div>
    </section>
  );
}
