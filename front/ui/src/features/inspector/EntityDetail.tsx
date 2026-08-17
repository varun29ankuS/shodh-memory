import { useQueryClient } from "@tanstack/react-query";
import { useSession } from "@/stores/session";
import { universeKey } from "@/features/graph/useUniverse";
import { entityTypeToken, type UniverseModel } from "@/features/graph/universe";
import { relName } from "@/features/recall/relation";
import { useEntityMemories } from "@/features/graph/useEntityMemories";
import { coOccurrenceOnly, observedWindow, typingLabel } from "@/features/graph/provenance";
import { Badge } from "@/components/ui/badge";
import { Meta, Stat } from "@/components/ui/meta";

/**
 * The Inspector's content for a selected ENTITY.
 *
 * Reads the built universe out of the react-query cache under the same key the
 * Graph destination writes (`universeKey`), exactly as the memory branch reads
 * the recall cache. It must not build its own model: Louvain over the full edge
 * set is the expensive part, and two models of one graph would disagree about
 * which community a node is in.
 *
 * What this pane is FOR is the edge provenance. A graph that only shows lines
 * asserts that two things are connected; naming the relation and its weight is
 * what makes the connection checkable. Both fields are the server's own —
 * `relation_type` and `strength` off `GravitationalConnection`
 * (src/graph_memory.rs:7293-7303) — and nothing here is inferred.
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

export function EntityDetail() {
  const profile = useSession((s) => s.profile);
  const selectedEntityId = useSession((s) => s.selectedEntityId);
  const selectEntity = useSession((s) => s.selectEntity);

  const model = useQueryClient().getQueryData<UniverseModel>(universeKey(profile));

  const idx = model && selectedEntityId ? model.index.get(selectedEntityId) : undefined;
  const entity = model && idx !== undefined ? model.nodes[idx] : undefined;

  if (!entity || !model || idx === undefined) {
    return (
      <div className="px-4 py-3">
        <p className="text-muted-foreground text-[12px] leading-relaxed">
          Select an entity to see what type it is and what it is connected to.
        </p>
      </div>
    );
  }

  // Neighbours with the relation that connects them, strongest first. The
  // adjacency is by index and carries only weight, so the relation comes from
  // the edge list keyed on the same pair.
  const neighbours = [...model.adjacency[idx].entries()]
    .map(([j, weight]) => {
      const other = model.nodes[j];
      const edge = model.edges.find(
        (e) =>
          (e.source === entity.id && e.target === other.id) ||
          (e.source === other.id && e.target === entity.id),
      );
      return { other, weight, relation: edge?.relation ?? null, generic: edge?.generic ?? true };
    })
    // Typed relations before generic co-occurrence, then by weight. A list
    // topped by twenty CoOccurs rows buries the one LocatedIn that matters.
    .sort((a, b) => Number(a.generic) - Number(b.generic) || b.weight - a.weight)
    .slice(0, 40);

  const cluster = model.clusters[entity.community] ?? null;

  return (
    <div>
      <div className="px-4 py-3">
        <p className="text-[13px] leading-relaxed font-medium">{entity.name}</p>

        <Field label="Type">
          <span className="flex items-center gap-1.5 text-[12px]">
            <span
              className="size-2 shrink-0 rounded-full"
              style={{ background: `var(${entityTypeToken(entity.type)})` }}
            />
            {entity.type}
            {entity.properNoun ? (
              <span className="text-muted-foreground/70 text-[11px]">proper noun</span>
            ) : null}
          </span>
        </Field>

        <Field label="Weight in the corpus">
          {/* Salience and mention count are the two things the entity store
              actually knows about how much this matters. They are not the same
              measure: salience is the extractor's own score, mentions is a
              count, and a high-mention low-salience entity is usually boilerplate. */}
          <p className="mono text-[11px]">
            salience {entity.salience.toFixed(3)} · {entity.mentions} mention
            {entity.mentions === 1 ? "" : "s"} · {entity.degree} connection
            {entity.degree === 1 ? "" : "s"}
          </p>
        </Field>

        {cluster ? (
          <Field label="Cluster">
            <p className="text-[12px]">
              {cluster.longTail ? "long tail" : cluster.label}
              <span className="text-muted-foreground/70"> · {cluster.size} entities</span>
            </p>
          </Field>
        ) : null}
      </div>

      {neighbours.length > 0 ? (
        <section className="border-border border-t px-4 py-3">
          <h3 className="text-[12px] font-medium tracking-tight">What it connects to</h3>
          <p className="text-muted-foreground/70 mt-0.5 text-[11px] leading-relaxed">
            Relation type and edge weight, as the graph stores them.
          </p>
          <div className="mt-2 flex flex-col gap-1">
            {neighbours.map(({ other, weight, relation, generic }) => (
              <button
                key={other.id}
                type="button"
                onClick={() => selectEntity(other.id)}
                className="hover:bg-accent/60 focus-visible:ring-ring rounded px-2 py-1.5 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
              >
                <span className="mono text-[10px]">
                  <span className={generic ? "text-muted-foreground" : "text-primary"}>
                    {relation ? relName(relation) : "related"}
                  </span>{" "}
                  <span className="text-muted-foreground/70">{weight.toFixed(2)}</span>
                </span>
                <span className="mt-0.5 flex items-center gap-1.5 text-[11px] leading-relaxed">
                  <span
                    className="size-1.5 shrink-0 rounded-full"
                    style={{ background: `var(${entityTypeToken(other.type)})` }}
                  />
                  <span className="truncate">{other.name}</span>
                </span>
              </button>
            ))}
          </div>
          {model.adjacency[idx].size > neighbours.length ? (
            <p className="text-muted-foreground/60 mt-1.5 text-[10px]">
              +{model.adjacency[idx].size - neighbours.length} more
            </p>
          ) : null}
        </section>
      ) : (
        <section className="border-border border-t px-4 py-3">
          <p className="text-muted-foreground text-[12px] leading-relaxed">
            No relations survived the edge budget for this entity. It is in the corpus but nothing
            links it strongly enough to draw.
          </p>
        </section>
      )}

      <SourceMemories profile={profile} name={entity.name} entityId={entity.id} />

      <section className="border-border border-t px-4 py-3">
        <div className="flex flex-wrap gap-1">
          <Badge>{entity.type}</Badge>
          {entity.properNoun ? <Badge>proper noun</Badge> : null}
        </div>
      </section>
    </div>
  );
}

/**
 * Chain 1's onward hop: the entity back to the memories it came from.
 *
 * This is real provenance, not a text search on the name. Every id here is a
 * `source_episode_id` off an edge incident to this entity, and an episode's
 * uuid IS the memory id (src/handlers/state.rs:3349-3350) — so these are the
 * memories whose extraction actually created this entity's relations.
 *
 * The limit is stated because it is a real limit: an edge needs two entities,
 * so a memory whose extraction yielded only this one produced no edge and
 * cannot appear. Saying "memories that connected it to something else" is
 * accurate; saying "everywhere it is mentioned" would not be.
 */
function SourceMemories({
  profile,
  name,
  entityId,
}: {
  profile: string | null;
  name: string;
  entityId: string;
}) {
  const { data, error, isFetching } = useEntityMemories(profile, name, entityId);
  const census = data?.census ?? [];

  return (
    <section className="border-border border-t px-4 py-3">
      <h3 className="text-[12px] font-medium tracking-tight">Where it came from</h3>

      {isFetching && !data ? (
        <p className="text-muted-foreground/70 mt-1 text-[11px]">Tracing provenance…</p>
      ) : error ? (
        <p className="text-muted-foreground mt-1 text-[11px] leading-relaxed">
          Provenance did not load for this entity.
        </p>
      ) : !data || data.sources.length === 0 ? (
        <p className="text-muted-foreground mt-1 text-[11px] leading-relaxed">
          No source episode is recorded for this entity's edges. An edge needs two entities, so a
          memory that mentioned only this one leaves no trace here.
        </p>
      ) : (
        <>
          <p className="text-muted-foreground/70 mt-0.5 text-[11px] leading-relaxed">
            Memories whose extraction connected this entity to another.
          </p>

          {/* HOW THE EDGES WERE DECIDED — the field that was arriving on every
              traverse and being dropped. It goes above the excerpts because it
              qualifies all of them at once: an entity attested only by
              co-occurrence has a neighbourhood built from adjacency, and a
              reader who learns that after reading eight excerpts has read them
              under the wrong assumption. */}
          {census.length > 0 ? (
            <div className="mt-2">
              <Meta className="text-[10px]">
                {census.map((c) => (
                  <Stat key={c.method} value={c.count} label={typingLabel(c.method)} />
                ))}
              </Meta>
              {coOccurrenceOnly(census) ? (
                // A caveat, so it is on screen rather than behind the info
                // affordance — it changes how every excerpt below reads.
                <p className="text-muted-foreground/80 border-warn/40 mt-1.5 border-l pl-2 text-[10px] leading-relaxed">
                  Nothing read the relation. Every attestation here is two names inside the same
                  window of text, so these links record adjacency, not a stated connection.
                </p>
              ) : null}
            </div>
          ) : null}

          <div className="mt-2 flex flex-col gap-2">
            {data.sources.map(({ episode, provenance }) => {
              const window = provenance ? observedWindow(provenance) : null;
              return (
                <div key={episode.uuid} className="border-border/60 border-l-2 pl-2">
                  <p className="text-muted-foreground text-[11px] leading-relaxed">
                    {episode.content.length > 240
                      ? `${episode.content.slice(0, 239)}…`
                      : episode.content}
                  </p>
                  {/* What this one source attests, from its own record. NOT
                      `evidence_span` and NOT `confidence`: both are structurally
                      empty in production and the reasoning is on the interface
                      in lib/api/graph.ts. A single observation says so instead
                      of printing a range whose ends are the same instant. */}
                  {window ? (
                    <Meta className="mt-0.5 text-[10px]">
                      {provenance?.typed_by ? <span>{typingLabel(provenance.typed_by)}</span> : null}
                      <Stat
                        value={window.mentions}
                        label={window.mentions === 1 ? "mention" : "mentions"}
                      />
                      <span>
                        {window.sameDay
                          ? observedDay(window.first)
                          : `${observedDay(window.first)} – ${observedDay(window.last)}`}
                      </span>
                    </Meta>
                  ) : null}
                </div>
              );
            })}
          </div>
          {data.totalSources > data.sources.length ? (
            <p className="text-muted-foreground/60 mt-1.5 text-[10px]">
              {data.sources.length} of {data.totalSources} attesting sources shown
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}

/** A provenance timestamp as a date. The time of day is noise here — what the
 *  record is being asked is when this entity was seen, not at which second. */
function observedDay(iso: string): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "";
  return at.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "2-digit" });
}
