import { useQuery } from "@tanstack/react-query";
import {
  fetchConsolidationEvents,
  fetchLineageStats,
  fetchTierCensus,
} from "@/lib/api/consolidation";
import { cn } from "@/lib/utils";
import {
  eventCensus,
  eventLabel,
  lineageVerdict,
  prunableShare,
  tierBands,
} from "./consolidating";

/**
 * What the store has DONE with what it holds.
 *
 * The standfirst above answers "what is in here" and the ontology band answers
 * "what kinds of things". Neither says whether anything happens to a memory
 * after it is written — which is the entire claim that separates this product
 * from a search index over a folder, and until now it was answerable only by
 * reading RocksDB.
 *
 * WHY IT BELONGS ON THE FRONT PAGE AND NOT ON A NEW DESTINATION. A destination
 * means a route and a sidebar entry, which live in `app/` and are another
 * agent's files — but the placement is right on its own terms and not a
 * concession. The three reads here are a CENSUS of the corpus, not a workspace:
 * there is nothing to do with them, nothing to select, nowhere to drill. That
 * is exactly what a front page carries and exactly what a destination should
 * not be. It sits under the ontology band because it is the same kind of
 * statement in the same grammar — a row of counted things with their labels —
 * one level deeper: the ontology says what was extracted, this says what has
 * been happening to it since.
 *
 * THREE ROWS, THREE ENDPOINTS, EACH ALLOWED TO FAIL ALONE. `tier-census` had one
 * reference in the whole repository before this and none in the UI;
 * `lineage/stats` and `consolidation/events` had no UI reference at all. Each is
 * its own query so a profile whose graph is empty still reports its consolidation
 * activity, and a failed read renders nothing rather than a zero — the
 * Briefing's governing rule is that an empty briefing and a broken one must
 * never look the same, and a zero here would claim the store did nothing.
 *
 * NO CONTROLS, DELIBERATELY. The events window is the server's default of one
 * hour, sent by omission and stated in words. A range picker would be a channel
 * a person could drive and the agent could not, which is the failure this branch
 * exists to stop adding to.
 */

/** The Briefing's band grammar: a figure in the mono face, a label after it. */
function Band({
  value,
  label,
  emphasis,
}: {
  value: string;
  label: string;
  emphasis?: boolean;
}) {
  return (
    <span className="mono text-muted-foreground flex items-baseline gap-1.5 text-[11px] tracking-[0.06em]">
      <span
        className={cn("text-[15px] tabular-nums", emphasis ? "text-destructive" : "text-foreground")}
      >
        {value}
      </span>
      {label}
    </span>
  );
}

function Row({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-[1.1rem] gap-y-1.5">
      <span className="mono text-muted-foreground w-[8.5rem] shrink-0 text-[10px] tracking-[0.14em] uppercase opacity-80">
        {heading}
      </span>
      {children}
    </div>
  );
}

export function Learning({ profile }: { profile: string }) {
  const census = useQuery({
    queryKey: ["tier-census", profile],
    queryFn: ({ signal }) => fetchTierCensus(profile, signal),
    staleTime: 60_000,
  });
  const lineage = useQuery({
    queryKey: ["lineage-stats", profile],
    queryFn: ({ signal }) => fetchLineageStats(profile, signal),
    staleTime: 60_000,
  });
  const events = useQuery({
    queryKey: ["consolidation-events", profile],
    queryFn: ({ signal }) => fetchConsolidationEvents(profile, signal),
    staleTime: 60_000,
  });

  const bands = census.data ? tierBands(census.data) : [];
  const prunable = census.data ? prunableShare(census.data) : null;
  const verdict = lineage.data ? lineageVerdict(lineage.data) : null;
  const kinds = events.data ? eventCensus(events.data) : [];

  // Nothing measured on any of the three: render no section at all rather than
  // a heading over three empty rows. A profile with no graph and no maintenance
  // pass in the last hour has genuinely nothing to report here.
  if (bands.length === 0 && verdict === null && kinds.length === 0) return null;

  return (
    <section className="border-border flex flex-col gap-3 border-b py-4">
      {bands.length > 0 ? (
        <Row heading="Consolidated">
          {bands.map((band) => (
            <Band
              key={band.label}
              value={band.n.toLocaleString()}
              label={`${band.label} · ${band.strength.toFixed(2)}`}
            />
          ))}
          {/* The share of the graph already weak enough for maintenance to
              drop. `--destructive` because this is the one figure here that is
              a loss rather than a state — more than half of claude-code's edge
              set — and the token means wrong-or-going elsewhere in the product.
              Absent, not zero, on a corpus where nothing is near the floor. */}
          {prunable !== null ? <Band value={`${prunable}%`} label="near the floor" emphasis /> : null}
        </Row>
      ) : null}

      {verdict ? (
        <Row heading="Causal links">
          <Band value={verdict.total.toLocaleString()} label="edges" />
          <Band value={`${verdict.inferredShare}%`} label="inferred" />
          {/* THE FINDING THIS WHOLE SECTION EXISTS FOR. Every profile on this
              server reports zero confirmed and zero explicit causal edges: the
              confirm path and the explicit path both exist and neither has ever
              fired. Stated as a sentence rather than as two more zeroes in the
              band row, because "0 confirmed" reads as a metric at rest and this
              is a finding about the product. */}
          {verdict.allInferred ? (
            <span className="text-muted-foreground max-w-[52ch] text-[13px] leading-[1.45]">
              None was stated by a source or confirmed by a person — every link here was inferred.
            </span>
          ) : (
            <>
              <Band value={verdict.confirmed.toLocaleString()} label="confirmed" />
              <Band value={verdict.explicit.toLocaleString()} label="explicit" />
            </>
          )}
        </Row>
      ) : null}

      {kinds.length > 0 ? (
        <Row heading="Past hour">
          {kinds.map((kind) => (
            <Band key={kind.type} value={kind.count.toLocaleString()} label={eventLabel(kind.type, kind.count)} />
          ))}
        </Row>
      ) : null}
    </section>
  );
}
