import { useQueryClient } from "@tanstack/react-query";
import type { RecallResponse } from "@/lib/api";
import { useSession } from "@/stores/session";
import { ScoreBreakdown } from "./ScoreBreakdown";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { relName } from "@/features/recall/relation";
import { memoryTier, MEMORY_TIER_LABEL, MEMORY_TIER_MEANING } from "@/features/recall/tier";
import { recallKey } from "@/features/recall/useRecall";
import { EntityDetail } from "./EntityDetail";

/**
 * The Inspector — the single object-detail surface.
 *
 * WORKFLOWS.md: every list in the product feeds this one pane, and it always
 * offers the next hop. That is what makes a click lead somewhere instead of
 * terminating.
 *
 * It reads the recall result from the react-query cache under the same key the
 * result list used, rather than taking it as a prop or refetching. Recall is
 * expensive and its response already contains everything shown here —
 * attribution, lineage, facts. Asking again to render a detail view would pay
 * the full retrieval cost twice for data already in memory.
 *
 * Positioned absolutely rather than as a flex column so the d3 canvas can later
 * take the stage's full width with this floated over its right edge. Until then
 * `main` reserves the same width, so nothing is occluded.
 */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <div className="text-muted-foreground/70 mb-1 text-[10px] tracking-wide uppercase">
        {label}
      </div>
      {children}
    </div>
  );
}

function Empty({ body }: { body: string }) {
  return (
    <div className="px-4 py-3">
      <p className="text-muted-foreground text-[12px] leading-relaxed">{body}</p>
    </div>
  );
}

export function Inspector() {
  const profile = useSession((s) => s.profile);
  const query = useSession((s) => s.activeQuery);
  const selectedId = useSession((s) => s.selectedMemoryId);
  const selectedEntityId = useSession((s) => s.selectedEntityId);
  const select = useSession((s) => s.select);

  // Same key as ResultPane, read straight out of the cache.
  //
  // Deliberately `getQueryData` and not `useQuery({ enabled: false })`. That
  // form still *registers* a query for the key, and a registration carrying no
  // queryFn throws when nothing else has populated it — which is exactly the
  // first-paint case here, before any search has run. A cache read has no such
  // failure mode and cannot issue a request by accident.
  const data = useQueryClient().getQueryData<RecallResponse>(recallKey(profile, query));

  const memory = data?.memories.find((m) => m.id === selectedId);

  // Chain 2: the causal edges that connect this memory to others in the same
  // result set. The server returns these already — no extra endpoint.
  const lineage = (data?.lineage ?? []).filter(
    (e) => e.from === selectedId || e.to === selectedId,
  );

  return (
    <aside
      aria-label="Detail"
      // `min(280px,36vw)` — kept in lockstep with `INSPECTOR_OFFSET` in
      // App.tsx, which reserves exactly this width in `main`. A bare 280px
      // here with a bare 280px there is how the two silently drift; see
      // RecallView.tsx for why a fixed width doesn't survive a narrow
      // viewport at all.
      className="border-border bg-card absolute top-12 right-0 bottom-0 z-20 flex w-[min(280px,36vw)] flex-col border-l"
    >
      <header className="border-border shrink-0 border-b px-4 py-3">
        <h2 className="text-[12px] font-medium tracking-tight">Detail</h2>
        <p className="text-muted-foreground mt-0.5 text-[11px] leading-relaxed">
          {/* The two kinds of object this pane holds answer different
              questions, so the subtitle names the one on screen rather than
              making one claim that is only true half the time. */}
          {selectedEntityId
            ? "What the corpus knows about this, and how it is connected."
            : "Every belief traces back to the sessions it came from."}
        </p>
      </header>

      <ScrollArea className="min-h-0 flex-1">
        {/* One Inspector, two kinds of object — WORKFLOWS.md's rule is that
            there is a single detail surface, not that everything in the product
            is a memory. An entity is reached from the knowledge graph and has
            no content, tier or score attribution; rendering it through the
            memory branch would mean inventing all three. Selection is mutually
            exclusive in the store, so exactly one of these can be live. */}
        {selectedEntityId ? (
          <EntityDetail />
        ) : !memory ? (
          <Empty
            body={
              data
                ? "Select a result to see what it is, why it surfaced, and what it connects to."
                : "Search, then select a result to see why it surfaced."
            }
          />
        ) : (
          <>
            <div className="px-4 py-3">
              <p className="text-[13px] leading-relaxed">{memory.experience.content}</p>

              {memory.experience.tags.length > 0 ? (
                <Field label="Entities">
                  <div className="flex flex-wrap gap-1">
                    {memory.experience.tags.map((t) => (
                      <Badge key={t}>{t}</Badge>
                    ))}
                  </div>
                </Field>
              ) : null}

              <Field label="Recorded">
                <p className="mono text-[11px]">
                  {new Date(memory.created_at).toLocaleString()}
                </p>
              </Field>

              {memory.experience.geo_location ? (
                <Field label="Coordinates">
                  <p className="mono text-[11px]">
                    {memory.experience.geo_location[0].toFixed(5)},{" "}
                    {memory.experience.geo_location[1].toFixed(5)}
                  </p>
                </Field>
              ) : null}

              {/* The same consolidation step the graph canvas encodes as node
                  presence, named here in words. `memory.tier` is a Rust Debug
                  rendering — `LongTerm` — and putting an identifier on screen
                  asks the reader to know the enum. This is the one surface with
                  room to say what the step MEANS, and it costs nothing here
                  because it is shown for one selected memory rather than on
                  every hover. */}
              <Field label="Consolidation">
                <p className="mono text-[11px]">
                  {MEMORY_TIER_LABEL[memoryTier(memory.tier)]}
                </p>
                <p className="text-muted-foreground/70 mt-0.5 text-[11px]">
                  {MEMORY_TIER_MEANING[memoryTier(memory.tier)]}
                </p>
              </Field>
            </div>

            {/* Chain 2 — the onward hop. Every edge here is a real causal
                relation the server returned, and selecting one moves the
                Inspector to the other end of it. */}
            {lineage.length > 0 ? (
              <section className="border-border border-t px-4 py-3">
                <h3 className="text-[12px] font-medium tracking-tight">What it connects to</h3>
                <div className="mt-2 flex flex-col gap-1">
                  {lineage.map((e, i) => {
                    const otherId = e.from === selectedId ? e.to : e.from;
                    const other = data?.memories.find((m) => m.id === otherId);
                    const outgoing = e.from === selectedId;
                    return (
                      <button
                        key={`${e.from}-${e.to}-${i}`}
                        type="button"
                        onClick={() => select(otherId)}
                        disabled={!other}
                        className="hover:bg-accent/60 focus-visible:ring-ring rounded px-2 py-1.5 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none disabled:opacity-50"
                      >
                        {/* `relation` arrives as a Rust Debug rendering —
                            `CausedBy`, or `Custom("depends on")` with the
                            quotes on the wire (src/handlers/recall.rs:985).
                            `relName` is the only thing standing between that
                            and Rust syntax on screen.

                            The confidence beside it is the server's own, and
                            it is the point: an edge without a strength is an
                            assertion, an edge with one is evidence. */}
                        <span className="text-primary mono text-[10px]">
                          {outgoing ? "→" : "←"} {relName(e.relation)}
                          <span className="text-muted-foreground/70">
                            {" "}
                            {e.confidence.toFixed(2)}
                          </span>
                        </span>
                        <span className="text-muted-foreground mt-0.5 line-clamp-2 block text-[11px] leading-relaxed">
                          {other?.experience.content ?? "Outside this result set"}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </section>
            ) : null}

            {memory.score_attribution ? (
              <ScoreBreakdown attr={memory.score_attribution} />
            ) : null}
          </>
        )}
      </ScrollArea>
    </aside>
  );
}
