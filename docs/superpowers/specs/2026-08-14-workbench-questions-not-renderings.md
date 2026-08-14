# Questions, not renderings — workbench design (revision 2)

**Date:** 2026-08-14
**Status:** approved in conversation, not yet implemented
**Supersedes:** `2026-08-14-workbench-field-and-view-bus-design.md` (revision 1, PR #507).
Revision 1's view bus, conversation dock and history surface survive intact.
Its **grammar** does not — see §2.
**Extends:** `front/ui/DIRECTION.md` on structure. **Overrides** it on ground: the
product is paper, not near-black.

---

## 1. What is actually wrong

Revision 1 diagnosed a flat, even interface with no dynamics. That was true and
it was not the root. Three findings from running the app against a real corpus
(`gdelt-bridge`, 337 memories; `defence`, 1,008 entities) replace it.

**1.1 — The rail is a list of drawing styles, not a list of questions.**
Recall, Graph and Geo are one result set rendered three ways: as a list, as a
node-link picture, as points on a map. Making each a destination means none of
them owns a question, so each feels thin, and the user must know which *picture*
they want before they know what they want to *ask*. Search compounds it: the
field already lives ambiently in the `TopBar`, so "Recall" is a place wrapped
around a verb that needed no place.

**1.2 — The largest surface in the product is an advertisement for itself.**
On `/recall` with no query, roughly 60% of the stage is `RecallDiagram` — a
four-step explainer (`01 Your cue → 02 Activation spreads → 03 What surfaced →
04 Where it came from`) with a *fake* search box, *fake* result rows, and a
paragraph of prose, animating on two infinite keyframes. The 337 real memories
are in a ~340px column beside it, truncating mid-word. An explainer is read
once; this one has permanent tenancy on the biggest surface there is.

**1.3 — The graph answers no question a person brings.**
Measured on screen: **18,059 relations** (human edge-tracing fails around
twenty); **831 of 1,008 entities share one type**, so five hues encode a
distinction the data does not have; edges are **unlabelled at rest** — the
relation type exists only in a hover tooltip — so "links" convey adjacency and
nothing else; node size is `mentions`, a popularity count, not importance; and
the footer admits `781 weak edges hidden`, so the picture is already filtered
without saying by what. It communicates exactly one fact: there is a dense part
and a sparse part.

## 2. The correction to revision 1

Revision 1 chose "the field": **the whole corpus, loudest at rest**. Against
§1.3 that rest state is worthless — it frames a picture that says nothing, very
precisely.

**The rest state is the corpus *described*, not drawn.** What is in it, in what
proportion, its largest clusters, its span in time — in plain words and honest
counts. On the defence corpus that opening line reads
`Technology 831 · Organization 90 · Other 47 · Location 31 · Person 9`, which
tells a reader more than the entire canvas does, *including* that the typing is
broken — a fact the hairball actively conceals.

The node-link canvas is promoted to the stage when there is a **specific** thing
to draw: a path, a neighbourhood, a result set of twenty. At that size it beats
a list. At a thousand nodes it loses to a sentence.

## 3. Information architecture

### 3.1 The rail carries questions

| survives | why |
|---|---|
| **Anomalies** | "What is odd here?" — a distinct question, already the least colour-dependent and most disciplined surface in the product |
| **Tasks** | "What is outstanding?" — a distinct question |

**Removed as destinations:** Recall, Graph, Geo — they become lenses (§3.3).
**Moved:** Providers is settings, and belongs behind a menu, not in the primary
rail. **Conversations** remains reachable, but as the ambient dock (§5), not as
a rail peer.

Seven rail items become two.

### 3.2 Ask is ambient, not a place

The cue field is present on every surface. Asking never navigates. The answer
arrives on the stage you are already looking at.

### 3.3 Lenses replace destinations

One answer, four ways to look at it — **List**, **Graph**, **Map**, **Time** —
as a small control on the stage, not as rail peers.

**The lens is pre-selected from the answer, not from memory of what you last
used.** Geotagged results open on the map; a result whose lineage forms a path
of three or more opens on the graph; everything else opens as a list. A lens
with nothing to show is disabled and says why ("no result in this set carries a
location") rather than rendering an empty frame — an empty map and a broken map
look identical.

### 3.4 The explainer earns its place by being asked for

`RecallDiagram` moves behind a **"How does this work"** control. It is good
work and it is the clearest statement of the product's claim; it is not a
resting state. Shown on request, dismissed on read, and — per revision 1 — it
already degrades correctly under `prefers-reduced-motion`.

## 3.5 Briefing, then workbench

The product is **not a dashboard**, and the word is barred from this spec
because of what it makes people build. A dashboard is pre-composed, glanceable
and monitored — a grid of tiles each with a fixed job. Twelve tiles in a grid is
exactly the even interface with no dynamics that §1 exists to fix. An analyst
does not monitor a memory; they interrogate it.

**You land on a briefing. It becomes a workbench.**

The briefing is three things, not twelve:

1. **What is in here** — the corpus described (§2), in plain words and honest counts.
2. **What changed since you left** — new memories, new links, what was reinforced.
3. **What looks wrong** — the anomalies.

Each is a **door, not a tile.** Selecting one does not open a panel beside it;
it converts the screen into the workbench focused on that thing, and the
briefing compresses to a strip that is the way back.

### 3.5.1 What the workbench holds

The answer (under whichever lens, §3.3), the conversation, the detail of a
selected object, and the trail. It is expected to hold more over time — saved
cues, a report being drafted, notes.

**That expectation is the risk.** "The workbench can also hold X" is the exact
path back to a partitioned dashboard, arrived at one reasonable addition at a
time. One rule prevents it, and it is not negotiable:

> **Exactly one primary at a time.** Everything else on the stage is compressed
> to its smallest legible form. Promoting anything demotes the current primary.
> There are never two co-equal panels.

Compressed does not mean hidden — a compressed conversation still shows that it
is streaming and its last line; a compressed briefing still shows the corpus in
one sentence. Compressed means *it has surrendered the width and kept its
meaning*.

This is checkable, which is the point of writing it down: if a screenshot of any
state shows two things competing for attention, the rule is broken and the
addition that broke it comes out. Adding a new kind of thing to the workbench
requires saying what it looks like compressed, in the same commit.

## 4. Fluid space

The rule, stated so it can be checked: **no surface holds width it is not
currently using.**

| state | stage | detail | list |
|---|---|---|---|
| answer, nothing selected | — | **absent** | full width |
| answer, something selected | — | one third | remainder |
| answer with a real path, graph lens | full stage | one third | compresses to a strip of what else matched |
| no answer yet | corpus description (§2) | absent | absent |

Already landed ahead of this spec, because it was a fifth of the stage: the
Inspector no longer reserves `min(280px,36vw)` when nothing is selected, where
it previously spent that width on the sentence "Select a memory or an entity"
(`app/App.tsx`).

Nothing in this table is a fixed pane. Every one of these is a transition of the
same stage, which is what makes the arrangement answer to the person rather than
to the route.

## 5. Carried over from revision 1, unchanged

These were not affected by the correction and stand as written:

- **The view bus** — one store, one `dispatch(command, author)`, two producers
  (the user's hands, and one adapter translating `SeatEvent`s). Commands named
  and serializable. **`lens` joins `destination`, `cue`, `frame`, `focus` and
  `filters` as a dimension**, so the agent can change how you are looking at an
  answer under the same authority rule.
- **Implicit agent sync** — `memory_recall` → `frame`, needing no seat change.
- **Authority** — the human always has the wheel; a declined agent command
  becomes a **Follow** offer, never a silent drop.
- **The conversation dock** — already survives navigation at the state layer
  (`stores/chat.ts`); it needs to stop returning `null` on `/chat` and to
  collapse to a strip rather than dismiss.
- **History** — the crumb trail expanded; three sources merged; built on
  `seat/src/ledger.ts`, which is already append-only JSONL with compensating
  reverts. Optional `actor` field; read-auditing deferred.

## 6. Ground: paper

Landed ahead of this spec (`front/ui/src/index.css`). Recorded here because it
changes constraints downstream.

The near-black console made every screen an instrument panel, which is a claim
about the work that is not true — the work is reading documents, provenance and
prose. Two rules survive verbatim because neither was about darkness: exactly
one accent, and the accent must never also mean anomaly (on paper they separate
by hue — accent orange-red, alarm crimson). Three things invert: depth (more
present is *darker*), ink is not black, and the data hues are re-derived rather
than converted.

**Known follow-up, not yet done:** both canvases composite node fills with
`hexA(hue, alpha)` at alphas tuned for a near-black ground. On paper those wash
out. The alpha ramps need re-tuning against the new ground; this is the one
place where "d3 reads the tokens, so re-tokening is enough" does not hold.

## 7. Testing

Beyond revision 1's suite:

- **Lens selection** — given a result set, the correct lens is chosen: geotagged
  → map; lineage path ≥ 3 → graph; otherwise list. Table-driven.
- **Lens availability** — a lens with nothing to show reports disabled with a
  reason, and never renders an empty frame.
- **Space allocation** — each row of §4's table asserted as a state, so a pane
  cannot silently reacquire permanent width.
- **Corpus description** — renders real counts from a fixture and states the
  dominant type honestly, including when one type holds 83% of the corpus.

Every test shown to fail before the code that makes it pass.

## 8. Sequencing

1. Corpus description as the rest state (§2) — replaces the explainer's tenancy.
2. Lenses (§3.3), and the rail reduced to questions (§3.1).
3. Ambient ask (§3.2).
4. Fluid space (§4).
5. View bus + implicit sync (§5) — unchanged from revision 1's plan.
6. Dock, then history.

## 9. Not in this pass

Session replay. Read-auditing as a durable ledger kind. Canvas alpha re-tuning
for paper (§6) is a follow-up, tracked, not deferred indefinitely.

## 10. The dependency this design does not fix

Every improvement above is bounded by entity typing. `Technology 831 of 1,008`
is why the legend teaches nothing, why the graph's colour channel is inert, and
why the corpus description in §2 currently describes a broken ontology. The
scattered degree-0 junk — hashes, bare floats, stray tokens — is why the field
looks empty and why it took a 6% frame trim to stop twenty of them dictating the
camera for all 136.

**The typing fix is a prerequisite for this design reading well, not a parallel
workstream.** Stated here so no one builds §2's opening screen against a corpus
that can only introduce itself as "mostly Technology".
