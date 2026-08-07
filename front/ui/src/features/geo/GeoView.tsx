import type { Reachability } from "@/lib/api";
import { ActivityNotices } from "@/components/layout/ActivityNotices";
import { EmptyState } from "@/components/ui/empty-state";
import { useRecall } from "@/features/recall/useRecall";
import { useMemoryTypes } from "@/features/recall/GraphCanvas";
import { GeoMap } from "./GeoMap";

/**
 * Geo — where the current recall result set happened.
 *
 * It plots the SAME result set the Recall destination lists, read from the same
 * react-query cache entry (see useRecall), so this is a second view of one
 * retrieval rather than a second retrieval. Selecting a point selects the
 * memory, which is the same global selection the result list and the graph
 * canvas drive, and it opens in the same Inspector.
 *
 * The empty states carry most of this screen's honesty budget. Most memories
 * have no coordinates, and that is a property of how they were written rather
 * than a failure of this view — saying so plainly is the difference between a
 * user concluding "the map is broken" and "this corpus is not geotagged".
 */

export function GeoView({ reach }: { reach: Reachability }) {
  const { data, error, isFetching, profile, query } = useRecall(reach);
  const memories = data?.memories ?? [];
  const types = useMemoryTypes(memories);

  const located = memories.filter((m) => m.experience.geo_location);

  if (reach.state !== "online") {
    return (
      <EmptyState
        size="page"
        title="Not connected"
        body="The map draws from a recall result, which needs the memory server running."
      />
    );
  }

  if (profile === null) {
    return (
      <EmptyState
        size="page"
        title="No profile to search"
        body="Geo plots the results of a recall, and recall needs a profile that already exists."
      />
    );
  }

  if (!query.trim()) {
    return (
      <EmptyState
        size="page"
        title="Search to place results on the map"
        body="Geo shows where the current recall result happened. Run a search and any memory carrying coordinates appears here."
      />
    );
  }

  if (error) {
    return (
      <EmptyState
        size="page"
        title="Recall failed"
        body="The map plots a recall result, and that query did not complete."
      />
    );
  }

  if (isFetching && !data) {
    return <EmptyState size="page" title="Searching" body="Placing results as they arrive." />;
  }

  if (memories.length === 0) {
    return (
      <EmptyState
        size="page"
        title="Nothing surfaced"
        body="No memory in this profile activated strongly enough for that cue, so there is nothing to place."
      />
    );
  }

  if (located.length === 0) {
    return (
      // Factual, not apologetic. The reason is a property of the data and the
      // sentence says which data would behave differently, so it is actionable
      // rather than a shrug.
      <EmptyState
        size="page"
        title="No coordinates in these results"
        body={`All ${memories.length} results came back without coordinates. A memory only carries them when whatever wrote it supplied them — imported corpora like GDELT do, session captures do not.`}
      />
    );
  }

  return (
    // `h-full`, not `flex-1`. GraphStage can use `flex-1` because RecallView
    // wraps it in a flex row; this view is a direct child of `main`, which is
    // not a flex container, so `flex-1` resolves to nothing and the section
    // collapses to the height of its content — which is zero, since the map
    // and every overlay inside it are absolutely positioned. That renders a
    // blank destination with no error anywhere.
    <section className="relative h-full min-h-0 min-w-0 overflow-hidden">
      <div aria-hidden="true" className="graticule pointer-events-none absolute inset-0" />

      <GeoMap memories={memories} types={types} />

      <div className="text-muted-foreground pointer-events-none absolute top-3 left-4 z-10 text-[12px]">
        Where this happened
      </div>

      <ActivityNotices />

      <div className="pointer-events-none absolute inset-x-4 bottom-3 z-10 flex flex-wrap items-center justify-between gap-x-6 gap-y-2">
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
        <span className="text-muted-foreground/70 text-[11px]">
          {/* State the ratio, not just the count. "12 placed" alone invites the
              reading that the other 13 failed to draw; naming both makes the
              gap a fact about the corpus. */}
          {located.length} of {memories.length} carry coordinates · scroll to zoom · drag to pan ·
          click a point to inspect
        </span>
      </div>
    </section>
  );
}
