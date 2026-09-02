import { useMemo } from "react";

import { useCorpus } from "@/lib/api/corpus";
import type { Reachability } from "@/lib/api/health";

import { AttentionPanel } from "./AttentionPanel";
import { DotMap } from "./DotMap";
import { WorkPanel } from "./WorkPanel";
import type { PlacedPoint } from "./dot-map";
import {
  INDIA,
  INDIA_BOUNDS,
  INDIA_CELL,
  WORLD,
  WORLD_BOUNDS,
  WORLD_CELL,
} from "./geo-shapes";

/**
 * The briefing.
 *
 * Every other destination answers a question you already had. This one is the
 * only surface that tells you something without being asked — what changed
 * since you last looked, and what is worth looking at now. That is why it is
 * the landing view.
 *
 * Everything except open work is DERIVED, not fetched: the maps and the anomaly
 * lenses both read the corpus cache entry geo and recall already populate, so
 * opening the briefing first makes those destinations instant rather than
 * making this one slow. Open work is the one fetch, and it strikes the same
 * bargain in the other direction -- it warms /tasks. (This comment used to say
 * "it adds no fetches", and kept saying it after the fetch landed. A file that
 * asserts its own behaviour instead of the behaviour being checked is how the
 * geo map came to carry a comment promising a zoom it did not have.)
 *
 * WHY THE MAPS ARE NOT THE WHOLE BRIEFING. They were, and on a profile whose
 * memories carry no coordinates that made the landing view two empty plates
 * under a headline reading "0 of them carrying a place" -- a briefing with
 * nothing to say on the majority of profiles, because geo-tagging is the
 * exception in this store, not the rule. The maps stay the centrepiece where
 * there is geography; where there is none they collapse to one line and the
 * panels that do not depend on coordinates carry the screen.
 */

/** Coordinates within ~55km collapse to one mark; below that the discs merge anyway. */
const CLUSTER_DEGREES = 0.5;

/** India's bounding box, used only to split the two maps' point sets. */
const INDIA_BOX = { minLon: 68, maxLon: 97.5, minLat: 6, maxLat: 37.5 };

type Located = { lat: number; lon: number };

/** Collapses nearby coordinates into weighted marks. */
function cluster(points: Located[]): PlacedPoint[] {
  const bins = new Map<string, { lon: number; lat: number; n: number }>();
  for (const { lat, lon } of points) {
    const key = `${Math.round(lat / CLUSTER_DEGREES)}:${Math.round(lon / CLUSTER_DEGREES)}`;
    const hit = bins.get(key);
    if (hit) hit.n += 1;
    else bins.set(key, { lon, lat, n: 1 });
  }
  // Sorted so the render order — and therefore which disc overlaps which — does
  // not depend on Map insertion order. Same discipline as the recall sorts.
  return [...bins.values()]
    .sort((a, b) => b.n - a.n || a.lon - b.lon || a.lat - b.lat)
    .map(({ lon, lat, n }) => [lon, lat, n] as PlacedPoint);
}

function inIndia({ lat, lon }: Located): boolean {
  return (
    lon >= INDIA_BOX.minLon &&
    lon <= INDIA_BOX.maxLon &&
    lat >= INDIA_BOX.minLat &&
    lat <= INDIA_BOX.maxLat
  );
}

export function BriefingView({ reach }: { reach: Reachability }) {
  const corpus = useCorpus(reach);
  const memories = corpus.data?.memories;

  const derived = useMemo(() => {
    const all = memories ?? [];
    const located: Located[] = [];
    for (const m of all) {
      // geo_location is [lat, lon, alt]; the renderer wants [lon, lat].
      if (m.geo_location) located.push({ lat: m.geo_location[0], lon: m.geo_location[1] });
    }

    const types = new Map<string, number>();
    for (const m of all) types.set(m.memory_type, (types.get(m.memory_type) ?? 0) + 1);

    return {
      total: all.length,
      located: located.length,
      world: cluster(located),
      india: cluster(located.filter(inIndia)),
      indiaCount: located.filter(inIndia).length,
      types: [...types.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])),
    };
  }, [memories]);

  // A FAILED FETCH IS NOT AN EMPTY CORPUS. Without this branch the briefing
  // renders `derived` over `memories ?? []` and states "0 memories in this
  // profile", three lenses reporting "not enough data", and no map -- a
  // confident description of a profile nobody managed to read. Caught live:
  // the backend returned 502 and this screen said the profile was empty
  // moments after drawing 28 memories from it.
  //
  // Reachability does not cover this. It is polled, so it still read
  // "Connected" while every corpus request was failing, and a per-request
  // error is the only thing that knows this request failed.
  if (corpus.error && !corpus.data) {
    return (
      <section className="grid h-full place-items-center p-8 text-center">
        <div className="max-w-md">
          <h2 className="text-base font-semibold">The corpus could not be read</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Nothing is being claimed about this profile until it can be. The memory server
            answered with an error — {String(corpus.error.message ?? corpus.error)}
          </p>
        </div>
      </section>
    );
  }

  // `isFetching` is true on background refetches too, so gate on "no data yet"
  // as well: a refresh must not blank a briefing the reader is already reading.
  if (corpus.isFetching && !corpus.data) {
    return (
      <section className="grid h-full place-items-center p-8 text-center">
        <div>
          <h2 className="text-base font-semibold">Reading the corpus</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            The briefing is derived, not stored — it is built from what is already here.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="h-full overflow-y-auto px-8 py-7">
      <header className="border-b border-border pb-5">
        {/* The place count leads only when there IS one. "0 of them carrying a
            place" as the first line of the landing view opens the product on
            an absence, and it is the wrong absence to lead with -- coordinates
            are the exception in this store, not the expectation. Where none
            exist the headline states what the profile holds, and the collapsed
            map section below says why there is no map. */}
        <h1 className="max-w-2xl text-2xl leading-snug font-semibold">
          <span className="text-primary">{derived.total.toLocaleString()}</span> memories
          {derived.located > 0 ? (
            <>
              ,{" "}
              <span className="text-primary">{derived.located.toLocaleString()}</span> of them
              carrying a place.
            </>
          ) : (
            " in this profile."
          )}
        </h1>

        {/* Memory kinds, not entity types. Labelled, because the two are one
            word apart and the mockup's headline figure was the other one. */}
        <ul className="mt-4 flex flex-wrap gap-x-6 gap-y-2 font-mono text-xs text-muted-foreground">
          {derived.types.slice(0, 6).map(([type, n]) => (
            <li key={type}>
              <span className="text-foreground">{n.toLocaleString()}</span> {type.toLowerCase()}
            </li>
          ))}
        </ul>
      </header>

      <div className="mt-7 grid gap-8 lg:grid-cols-2">
        <WorkPanel reach={reach} />
        <AttentionPanel memories={memories ?? []} />
      </div>

      {derived.located === 0 ? (
        /* One line, not two blank plates. An empty map is not a finding about
           the corpus, it is an absence of the input this graphic needs, and
           drawing it anyway spends the top of the screen saying so twice. */
        <p className="mt-8 border-t border-border pt-5 font-mono text-xs text-muted-foreground">
          No memory in this profile carries coordinates, so there is no map to draw. Geo-tagged
          profiles show the world and India here.
        </p>
      ) : (
      <>
      {/* The maps derive their height from their own width and the bounds'
          aspect, so an unbounded column gives India a ~900px plate. Cap the
          plate rather than the column: the captions and headings still want
          the full column width, and a briefing is read at arm's length. */}
      <div className="mt-7 grid gap-8 lg:grid-cols-2">
        <section>
          <h2 className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
            The world
          </h2>
          <div className="mt-3 max-w-[34rem]">
            <DotMap
              shapes={WORLD}
              bounds={WORLD_BOUNDS}
              points={derived.world}
              cell={WORLD_CELL}
              label={`World map, ${derived.located} located memories`}
            />
          </div>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            {derived.located} of {derived.total} memories carry a place
          </p>
        </section>

        <section>
          <h2 className="font-mono text-xs tracking-widest text-muted-foreground uppercase">
            India
          </h2>
          <div className="mt-3 max-w-[26rem]">
            <DotMap
              shapes={INDIA}
              bounds={INDIA_BOUNDS}
              points={derived.india}
              cell={INDIA_CELL}
              label={`Map of India, ${derived.indiaCount} located memories`}
            />
          </div>
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            {derived.indiaCount} memories · official boundary (LGD)
          </p>
        </section>
      </div>
      </>
      )}

      {/*
        DELIBERATELY ABSENT, and this is a note against re-adding it carelessly.

        The mockup's best element is the line that puts the corpus's worst
        property on the reader's first screen: "831 of 1,008 entities are typed
        Technology". That is GLiNER's ENTITY typing collapsing into one class,
        and it is invisible in the graph view because the graph draws whatever
        it is given.

        A first pass here rendered the same sentence from `memory_type`. That
        field is the memory KIND — Observation / Decision / Learning / Task, the
        four in the geo legend — so on the GDELT corpus it read "337 of 337
        typed Observation, 100% in one class, that is a typing failure". It is
        not. It is what a bulk import is, by construction. The instrument was
        reporting a defect that was not there, which is worse than showing
        nothing.

        Restoring it needs entity-type counts from the graph, not the corpus
        list — and the threshold has to be justified against a corpus where the
        typer is known to be working, or it just relabels normal skew as
        failure.
      */}

      <p className="mt-6 font-mono text-xs text-muted-foreground">
        Nothing here is generated — every figure is retrieved and traceable · Local · no network
        · Boundary: LGD via bharatlas · CC0-1.0 / CC-BY-4.0
      </p>
    </section>
  );
}
