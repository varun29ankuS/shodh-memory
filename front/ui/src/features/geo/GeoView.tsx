import { useMemo } from "react";
import { ApiError, NetworkError, outageOf, type Reachability } from "@/lib/api";
import { corpusToRecallMemory, useCorpus } from "@/lib/api/corpus";
import { formatCount, sampleNote } from "@/lib/format";
import { EmptyState } from "@/components/ui/empty-state";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { useRecall } from "@/features/recall/useRecall";
import { useView } from "@/stores/view";
import { useSession } from "@/stores/session";
import { useMemoryTypes } from "@/features/recall/GraphCanvas";
import { GeoMap } from "./GeoMap";

/**
 * Geo — everywhere this profile's memory happened, with the current recall
 * result highlighted on top of it.
 *
 * The map is populated the moment the destination opens: every located memory
 * in the corpus is drawn as a quiet context point, because "where do I have
 * memory?" is a question with an answer before any search is typed. Running a
 * search does not swap the map out — it turns the matching points up and the
 * rest down, so a result is always seen against the corpus it came from.
 *
 * Both data sources are shared cache entries (useCorpus, useRecall): this view
 * issues no retrieval of its own.
 */

/**
 * Why the corpus listing failed, in the reader's terms.
 *
 * A READ THAT FAILED IS NOT A CORPUS THAT IS EMPTY, and every surface here has
 * one branch that says "there is none of this" and another that says "we could
 * not look". Routing a failed fetch into the first is the same defect as
 * asserting a negative from one page — worse, because it is a negative from
 * nothing at all. Shared with AnomaliesView, which already drew this line.
 */
function readFailure(error: unknown): string {
  if (error instanceof ApiError) {
    return error.isAuthFailure ? "The server rejected this key." : `The server answered ${error.status}.`;
  }
  if (error instanceof NetworkError) return "The server stopped responding mid-request.";
  return "Something went wrong loading this profile's memories.";
}

export function GeoView({ reach }: { reach: Reachability }) {
  const corpus = useCorpus(reach);
  const { data, error, isFetching, profile, query } = useRecall(reach);

  const hasQuery = query.trim().length > 0;
  const results = useMemo(() => (hasQuery ? (data?.memories ?? []) : []), [hasQuery, data]);

  // The plotted set: recall results first (they carry scores and therefore
  // size), then every located corpus memory that is not already a result.
  /* THE MAP CONSUMES A CUE, so "show me where this is" has an answer here.
     Two producers, same channel as the graph: the model's recall arrives as
     cue terms on the view bus, and the human's typing arrives as cueDraft.
     Whichever is live raises the points whose memory mentions a term and
     recedes the rest — and GeoMap re-fits to what was raised, so the map
     MOVES to the answer instead of dimming in place. A map that highlights
     without travelling still makes you hunt for the highlight. */
  const agentCue = useView((s) => s.cue);
  const typedCue = useSession((s) => s.cueDraft);

  const cueTerms = useMemo(() => {
    if (agentCue?.entities.length) return agentCue.entities.map((t) => t.toLowerCase());
    const typed = typedCue.trim().toLowerCase();
    return typed ? [typed] : [];
  }, [agentCue, typedCue]);

  const { plotted, dimmed } = useMemo(() => {
    const resultIds = new Set(results.map((m) => m.id));
    const context = (corpus.data?.memories ?? [])
      .filter((m) => m.geo_location)
      .filter((m) => !resultIds.has(m.id))
      .map(corpusToRecallMemory);

    const all = hasQuery && results.length > 0 ? [...results, ...context] : context;

    if (cueTerms.length > 0) {
      const hit = (m: (typeof all)[number]) => {
        const text = m.experience.content.toLowerCase();
        return cueTerms.some((t) => text.includes(t));
      };
      const missed = new Set(all.filter((m) => !hit(m)).map((m) => m.id));
      // Every point missing means the cue named nothing on this map. Dimming
      // all of them would blank it and read as data loss, so the corpus stays
      // as it was and the caption still says what was searched.
      if (missed.size < all.length) return { plotted: all, dimmed: missed };
      return { plotted: all, dimmed: undefined };
    }

    if (!hasQuery || results.length === 0) {
      // No active answer: the corpus IS the map. Nothing is dimmed — these
      // points are not losing to anything.
      return { plotted: context, dimmed: undefined };
    }
    return { plotted: all, dimmed: new Set(context.map((m) => m.id)) };
  }, [corpus.data, results, hasQuery, cueTerms]);

  const types = useMemoryTypes(plotted);
  const located = plotted.filter((m) => m.experience.geo_location);
  const matched = results.filter((m) => m.experience.geo_location);

  /* HOW MUCH OF THE PROFILE THIS SCREEN ACTUALLY LOOKED AT.
     `useCorpus` fetches ONE capped page — 500 rows — and every context point on
     this map comes out of it. On the `claude-code` profile that page is 2.6% of
     19,553 memories, and this view used to answer "does this profile have any
     located memory?" out of it with an unqualified no. A negative drawn from
     2.6% of a corpus is a statement about the request, not about the corpus.
     `sampleNote` is null when the page covered everything, so a profile smaller
     than the cap pays nothing for this and keeps the absolute claim it earns. */
  const read = corpus.data?.memories.length ?? 0;
  const heldTotal = corpus.data?.total ?? read;
  const sample = sampleNote(read, heldTotal);

  // A REJECTED KEY IS NOT A STOPPED SERVER — see `outageOf`. The sentence
  // below is the offline case only.
  const outage = outageOf(reach, "The map draws from memory, which needs the server running.");
  if (outage) return <EmptyState size="page" {...outage} />;

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
    return <EmptyState size="page" title="Loading corpus" body="Placing every located memory." />;
  }

  // BEFORE the "nothing carries a place" branch, never after it. On a failed
  // read `corpus.data` is undefined, so `read` and `heldTotal` are both zero,
  // `sampleNote` reports a complete read, and the empty state below would
  // state the absolute — "None in this profile does yet" — on the strength of
  // a request that never returned. That is the one confusion this screen is
  // not allowed to produce, and it is reachable whenever the API server is
  // restarting.
  if (corpus.error) {
    return (
      <EmptyState
        size="page"
        title="Could not read the corpus"
        body={readFailure(corpus.error)}
        more="The map draws from one listing request. Until it returns, this screen knows nothing about which memories carry coordinates — including whether any do."
      />
    );
  }

  if (located.length === 0) {
    return (
      // Factual, not apologetic: the reason is a property of the data — and the
      // claim is only ever as wide as the read behind it. The second sentence
      // of this body used to be "One appears here the moment a memory is
      // written with coordinates", which `more` already says in its last line;
      // dropping it pays for the denominator twice over.
      <EmptyState
        size="page"
        title="Geo plots the memories that carry a place"
        body={
          sample
            ? `None of the ${formatCount(read)} read from this profile's ${formatCount(heldTotal)} does.`
            : "None in this profile does yet."
        }
        more={
          sample
            ? `Geo reads one page of ${formatCount(read)} memories, so this says nothing about the other ${formatCount(heldTotal - read)}. A memory only carries coordinates when whatever wrote it supplied them — imported corpora like GDELT do, session captures do not — and the first one on this page appears here without any search.`
            : "A memory only carries coordinates when whatever wrote it supplied them — imported corpora like GDELT do, session captures do not. The first one that does appears here without any search."
        }
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
          {/* The denominator, immediately after the figure it qualifies. It
              scopes `located` and NOT `matched`: the context points come off
              one capped page, while a search runs against the whole profile,
              and collapsing the two would weaken a true claim to fix a false
              one. Absent entirely on a profile the page fully covers. */}
          {sample ? <span>{sample}</span> : null}
          <InfoHint label="map controls" align="right" side="up">
            Scroll to zoom, drag to pan, click a point to inspect it. A search does not change
            which points are drawn — it raises the ones that match and dims the rest, so an answer
            is always seen against the corpus it came from. The context points come from one page
            of the corpus; a search reaches all of it.
          </InfoHint>
        </Meta>
      </div>
    </section>
  );
}
