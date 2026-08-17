import { useQuery } from "@tanstack/react-query";
import { Sigma } from "lucide-react";
import { fetchAnomalies, type AnomalyEntry } from "@/lib/api/anomalies";
import { useSession } from "@/stores/session";
import { InfoHint } from "@/components/ui/info-hint";
import { Meta, Stat } from "@/components/ui/meta";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import {
  axisLabel,
  drivingAxis,
  readSurprise,
  surpriseFacts,
  uniformMagnitude,
} from "./surprise";

/**
 * The fourth lens — the one the SERVER computed, on the screen named after it.
 *
 * `POST /api/anomalies` has existed with no client but the MCP tool. The screen
 * called Anomalies imported `useCorpus` and computed three different things
 * client-side, so two definitions of "anomaly" shipped under one label and the
 * richer of the two was invisible. This section is that endpoint, named for
 * what it actually measures.
 *
 * IT DOES NOT REPLACE THE OTHER THREE, AND SAYING SO IS THE POINT. The brief
 * anticipated the server detector might be thinner than the client lenses and
 * asked for both to be named honestly if so. Verified live, it is not thinner —
 * it is measuring a different object. The three client lenses read a memory's
 * CONTENT: where it says it is, what quantities it states, whether anything
 * else in the corpus shares its terms. This one reads the EXTRACTION's shape —
 * how strongly the entities it produced associate, how many were new, how many
 * relations went untyped — against the profile's own rolling baseline. A memory
 * can be perfectly ordinary on one and far out on the other, and neither is the
 * senior measure.
 *
 * NO CHART, AND THE REASON IS ON SCREEN. Every other lens here draws its
 * population before it draws a conclusion, on the principle that a flagged
 * value means nothing without the ordinary values it was judged against. This
 * one cannot: the endpoint returns the top 20 ranked by max |z|, not the 200 it
 * ranked them against, so any distribution drawn from the response would be a
 * plot of the tail presented as the whole. What it returns instead of a
 * population is the near-misses — the entries below the line, which the server
 * sends in the same feed — and those are rendered, dimmed, as the closest thing
 * to a comparison the response actually contains.
 *
 * NO CONTROLS. `window`, `limit` and `min_sigma` are all tunable on the wire
 * and none is exposed. A threshold slider in component state would be a channel
 * a person could drive and the agent could not, which is the precise failure
 * this branch exists to stop adding to. The parameters the server used are
 * printed instead.
 */

export function surpriseKey(profile: string | null) {
  return ["anomalies", profile] as const;
}

/**
 * One scored episode.
 *
 * Selecting sets the same global selection a client-lens finding does, so an
 * episode flagged by the server is the same kind of object as one flagged by
 * the arithmetic above it and travels to the Inspector the same way.
 *
 * THE EXPLANATION IS THE SERVER'S, VERBATIM. It is built deterministically from
 * the top deviating components (anomalies.rs:75-78) and is the auditable "why
 * was this flagged". Paraphrasing it client-side would put a second, unversioned
 * account of the flag on screen next to the one that can be checked.
 */
function SurpriseRow({ entry }: { entry: AnomalyEntry }) {
  const selected = useSession((s) => s.selectedMemoryId === entry.memory_id);
  const select = useSession((s) => s.select);
  const axis = drivingAxis(entry);

  return (
    <button
      type="button"
      onClick={() => select(entry.memory_id)}
      aria-current={selected ? "true" : undefined}
      className={cn(
        "border-border w-full border-b px-4 py-2 text-left transition-colors duration-100",
        "focus-visible:ring-ring focus-visible:-outline-offset-2 focus-visible:ring-2 focus-visible:outline-none",
        selected ? "bg-primary/10" : "hover:bg-accent/60",
        // Below the line is dimmer, not hidden and not coloured. `--warn` means
        // waiting-on and `--destructive` means late or wrong elsewhere in the
        // product; an ordinary episode is neither.
        entry.flagged ? "" : "opacity-55",
      )}
    >
      <Meta className="text-[12px]">
        <Stat value={`${entry.max_abs_z.toFixed(2)}σ`} />
        {axis ? <span>{axisLabel(axis)}</span> : null}
        {entry.flagged ? null : <span className="text-muted-foreground/70">below the line</span>}
      </Meta>
      <p
        className={cn(
          "mt-0.5 truncate text-[12px] leading-relaxed",
          selected ? "text-foreground/80" : "text-muted-foreground",
        )}
      >
        {entry.content_preview}
      </p>
      <p className="text-muted-foreground/60 mt-0.5 truncate text-[10px] leading-relaxed">
        {entry.explanation}
      </p>
    </button>
  );
}

export function SurpriseLens({ profile }: { profile: string }) {
  const { data, error, isFetching } = useQuery({
    queryKey: surpriseKey(profile),
    queryFn: ({ signal }) => fetchAnomalies(profile, signal),
    staleTime: 60_000,
  });

  const result = data ? readSurprise(data) : null;
  const flagged = result?.state === "findings" ? result.flagged : [];
  const ranked = result && result.state !== "insufficient" ? result.ranked : [];

  return (
    <section>
      <div className="border-border bg-muted/50 sticky top-0 z-10 flex items-center gap-2 border-b px-4 py-1.5 backdrop-blur-sm">
        <Sigma aria-hidden="true" className="text-muted-foreground size-3" strokeWidth={1.8} />
        <span className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          Statistically out of shape
        </span>
        {flagged.length > 0 ? (
          <span className="text-muted-foreground/60 mono text-[10px]">{flagged.length}</span>
        ) : null}
        <InfoHint label="Statistically out of shape">
          <span className="block">
            Episodes whose EXTRACTION deviates from this profile's own recent history — how strongly
            its entities associate, how many were unseen, how many relations went untyped. The three
            measures above read what a memory says; this one reads the shape of what was pulled out
            of it.
          </span>
          <span className="text-muted-foreground mt-1.5 block">
            The store records these statistics when a memory is written and scores the deviation
            when you ask, so the threshold can change without re-reading the corpus. Each z is the
            distance from the mean of the most recent scored episodes, in their own standard
            deviations. Computed on the server, component by component, with no model involved.
          </span>
        </InfoHint>
        <span className="text-muted-foreground/60 ml-auto text-[11px]">
          {result?.state === "clear"
            ? "Nothing flagged"
            : result?.state === "insufficient"
              ? "No baseline"
              : null}
        </span>
      </div>

      <div className="border-border border-b px-4 py-2">
        {isFetching && !data ? (
          <Skeleton className="h-3 w-[70%]" />
        ) : error ? (
          // A failed read is not a clean corpus, and this line is the only thing
          // stopping it from looking like one.
          <p className="text-muted-foreground text-[11px] leading-relaxed">
            The server's own deviation scoring did not load, so this section is reporting nothing
            rather than nothing found.
          </p>
        ) : data && result ? (
          <>
            <Meta>
              {surpriseFacts(data).map((fact) => (
                <span key={fact}>{fact}</span>
              ))}
            </Meta>
            {result.state === "insufficient" ? <Caveat>{result.reason}</Caveat> : null}
            {result.state !== "insufficient" ? (
              <Caveat>
                {/* The population caveat, always. It is the difference between
                    "these 14 are the unusual ones" and "these 14 are the most
                    unusual of the 20 the server chose to send". */}
                The server returns its ranked findings, not the {data.episodes_scored} episodes it
                ranked them against, so no distribution is drawn here. The dimmed rows are the
                nearest episodes that stayed below {data.min_sigma}σ.
                {uniformMagnitude(flagged)
                  ? ` All ${flagged.length} flagged episodes sit at exactly ${flagged[0].max_abs_z.toFixed(2)}σ, which means one repeated memory shape rather than ${flagged.length} independent findings.`
                  : ""}
              </Caveat>
            ) : null}
          </>
        ) : null}
      </div>

      {ranked.map((entry) => (
        <SurpriseRow key={entry.memory_id} entry={entry} />
      ))}
    </section>
  );
}

/** Matches `AnomaliesView`'s own caveat treatment — marked, indented, always
 *  rendered, never folded into the info panel. */
function Caveat({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-muted-foreground/80 border-warn/40 mt-2 border-l pl-2.5 text-[11px] leading-relaxed">
      {children}
    </p>
  );
}
