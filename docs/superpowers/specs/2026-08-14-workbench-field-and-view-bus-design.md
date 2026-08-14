# The field, the view bus, and the history — workbench design

**Date:** 2026-08-14
**Status:** approved in conversation, not yet implemented
**Supersedes:** nothing. Extends `front/ui/DIRECTION.md` (Gridline structure, 2026-08-07), which stays in force.

---

## 1. Why

The workbench is judged on first contact. Correctness is necessary and is not the
bar; a surface that is right and forgettable has failed the test it is actually
given, which is whether someone in the room leans in.

The current app does not fail on features or on palette. `front/ui/src/index.css`
defines four close greys for depth, one accent at a measured 6.29:1 on the ground,
and data hues deliberately fenced out of the 10–40° band so no category can be
mistaken for "selected". That is careful work.

It fails because **every surface is a fixed size**. A 48px bar, a 56px rail, a
280px Inspector, and whatever is left. Nothing swells when it matters and nothing
recedes when it stops mattering, so nothing is ever loudest. On the shipped build
at 1600×1000 roughly 95% of the screen is empty ground carrying a two-line
apology set at 13px.

An even interface has no music. This spec writes the dynamics.

## 2. Scope

**In:** the field grammar and its canvas, the view bus, the ambient conversation
dock, agent-driven view control, and the history surface.

**Out, this pass:** Anomalies, Tasks and Providers keep their current layouts.
Session replay (see §12). Read-auditing as a durable ledger kind (see §8.4).

**Vocabulary:** "tab" in the originating request means a rail destination —
Recall, Graph, Geo, Chat — not a browser tab.

## 3. The grammar: the field

Chosen from three candidates. The two rejected are recorded because the rejection
is load-bearing: *the aperture* (question loudest, corpus invisible at rest) and
*the watch floor* (recent activity loudest) both remain coherent designs, and a
blend of any two of the three reproduces the even interface described in §1.

**At rest:** the whole corpus is on the stage, framed to its actual extent, dense
and legible as a shape before a word has been read. Chrome drops to hairlines.
The prompt is one thin line at the bottom edge.

**Reaching:** the chosen neighbourhood swells toward the centre, the rest of the
field falls to a ghost, evidence rises on the right, and a crumb of where you came
from pins to the corner so the way back is never lost.

Two consequences that are not decoration:

1. **Framing is the rest state**, so it is computed rather than defaulted. Both
   canvases currently initialise `transformRef` to `zoomIdentity`
   (`features/graph/EntityCanvas.tsx:143`, `features/recall/GraphCanvas.tsx:169`)
   and neither computes a fit anywhere. The long-standing "graph doesn't fit the
   viewport" defect stops existing rather than being patched.
2. **The narrowing motion is the `frame` command** (§4). The animation is
   identical whether a human clicked a cluster or the model retrieved six
   memories. The agent does not get a second visual language.

`scaleExtent` stays as it is (`EntityCanvas.tsx:437`, `GraphCanvas.tsx:422`).
`prefers-reduced-motion` is already honoured in both canvases
(`EntityCanvas.tsx:260`, `GraphCanvas.tsx:258`) and every transition added here
must collapse to an instant state change under it.

## 4. Architecture: the view bus

A new zustand store, `stores/view.ts`, owning **everything about what is on
screen**:

| field | meaning |
|---|---|
| `destination` | which rail destination is showing |
| `cue` | the query the visible result set came from |
| `frame` | the extent currently framed — a set of ids, or `"all"` |
| `focus` | the promoted cluster or entity, if any |
| `filters` | time window, entity types, profile |

`selection` is **not** duplicated here. It already lives in `stores/session.ts`
as `selectedMemoryId` / `selectedEntityId` under the "one selected object at a
time" rule from `WORKFLOWS.md`, and a second copy would drift.

The canvases' private `transformRef` moves into `frame`. A canvas may still own
its *interpolation* toward the framed extent; it may not own *what* is framed.

### 4.1 Commands

Every mutation is a named, serializable command:

```ts
type ViewCommand =
  | { kind: "open";   view: Destination }
  | { kind: "cue";    text: string }
  | { kind: "frame";  ids: string[] | "all" }
  | { kind: "focus";  id: string; of: "memory" | "entity" }
  | { kind: "filter"; patch: Partial<Filters> }
```

Serializable is a requirement, not an accident: it is what makes the history in
§10 a record rather than a rendering, and what would later make replay possible
without a second mechanism.

### 4.2 Two producers, one bus

The store has exactly one entry point, `dispatch(command, author)`. Two things
call it:

- **Human interaction** — every click, drag and keystroke that changes what is on
  screen, with `author: "human"`.
- **A single adapter**, `useViewSync()`, mounted once in `Shell`
  (`app/App.tsx:60`), which subscribes to the chat store and translates arriving
  ops into commands with `author: "model"`.

One bus with two producers is what makes "both in sync" true by construction.
Two paths into the same screen state would drift, and the drift would be
invisible until a demo.

`useViewSync` is the **only** new consumer of `SeatEvent`s.
`features/chat/EvidencePanel.tsx` and `features/chat/MessageList.tsx` keep their
existing read-only consumption; the adapter must not become a second scattered
reader.

## 5. Data flow

### 5.1 The agent already narrates itself

No new transport is needed. `lib/seat/client.ts:223` streams `SeatEvent`s over
SSE for the duration of a turn, and the union (`lib/seat/types.ts:93-144`)
already carries both:

- `tool_call_start` with `tool_name` and `args` (`types.ts:98`)
- `memory_recall` with the memories actually retrieved (`types.ts:101`)

The browser receives all of this today and only prints it.

### 5.2 Implicit sync — phase 1, zero backend change

`memory_recall` → `frame(ids)`. You ask a question and the workbench visibly
narrows to what the model actually retrieved.

This requires no seat change, no new tool, and no prompt change. It is most of
the effect, and it is the one thing a competitor structurally cannot show,
because their retrieval is an opaque model call and this one is a deterministic
path.

### 5.3 Explicit sync — phase 2

A new `seat/src/view-tools.ts`, sibling to `seat/src/memory-tools.ts`, which
already registers first-party `AgentTool`s (`recall_memory` at `memory-tools.ts:161`,
`remember_memory` at `:248`, `record_seat_learning` at `:288`) assembled into the
agent's tool array at `seat/src/conversation.ts:393` and `:504`.

Tools: `open_view`, `focus_entity`, `frame_memories`, `set_cue`.

**The contract, stated because it is a real limitation:** there is no
browser→seat acknowledgement channel. The seat answers a view tool with
`requested` immediately. **A successful tool result is not confirmation that the
view changed** — the authority rule in §6 may have declined it. The tool
description must say so, so the model does not report a view change to the user
as an accomplished fact.

## 6. Authority and conflict

**The human always has the wheel.**

An agent command applies only if the user has not touched that dimension since
the current turn began. Dimensions are tracked independently: framing the field
by hand does not block the agent from opening a different destination.

When a command is declined it is **not** discarded. It surfaces as a **Follow**
affordance the user can accept. Silently dropping an agent's intent is worse than
either applying it or refusing it visibly, because the user is left with a model
that claims to have done something invisible.

Nothing ever yanks the view out from under a hand mid-gesture.

### 6.1 Untrusted arguments

Explicit tool arguments arrive as `unknown` from a language model
(`types.ts:98` types `args` as `unknown`; `seat/src/mcp.ts:630` shows the same
looseness for MCP tools). Every command is validated browser-side before
dispatch, **failing closed**. A rejected command appears in the history as
*unfulfilled*, with the reason. It is never silently swallowed.

## 7. The conversation dock

The requirement is that a conversation, once started, continues on every
destination without switching.

**This is already true at the state layer and must not be rebuilt.** `send()` is
an action on the chat store, in-flight aborts live in a module-level `Map`
outside React (`stores/chat.ts:265`), and only `forget()` aborts
(`stores/chat.ts:404`). Navigating away does not kill a turn.

What is missing is visibility. `features/chat/ConversationOverlay.tsx` returns
`null` on `/chat` (`:156`), when the seat is offline (`:157`), and when dismissed
(`:158`).

Changes:

1. Present on **every** destination, including `/chat`.
2. **Collapses to a strip; does not dismiss to nothing.** A live turn must never
   become invisible. The strip carries the streaming state and the last line;
   expanded is today's panel.
3. The offline branch (`:157`) stays. There is nothing to show and nothing to
   continue.

## 8. History

The **crumb trail is the history, minimized**. At rest it is the few chips
showing how the current view was reached; expanded it is the full timeline. This
keeps the min/max orchestration consistent and avoids an eighth unlabelled glyph
on the rail.

### 8.1 Three sources, one timeline

| source | records | durable |
|---|---|---|
| `learning-ledger.jsonl` | writes, reinforcements, implicit feedback, reverts | yes, on disk |
| conversation events | recalls, tool calls, model changes | yes, server-side |
| view commands | navigation, framing, focus | **no — session only** |

Every row carries its author. Rows are filterable by author, kind and time.

### 8.2 Build on the existing ledger, do not replace it

`seat/src/ledger.ts` is already the right substrate: append-only JSONL at
`<dataDir>/learning-ledger.jsonl`, appends serialized through a write chain so
entries cannot interleave, and **reverts implemented as compensating entries
referencing the original** (`kind: "revert"`, `data.of`) rather than mutation.
Its header is honest that an EMA reinforcement is not bitwise-invertible, so a
revert is recorded as compensation and not as undo.

It is read at `/seat/v1/learning/events` and reverted at
`/seat/v1/learning/revert` (`lib/seat/client.ts:145-153`). Revert stays available
from the history row.

### 8.3 View commands stay out of the durable ledger

Where the camera pointed is not a fact about the corpus. Mixing UI navigation
into an evidentiary record weakens precisely the property the product is sold on.
Session-scoped rows are marked as such and vanish on reload.

### 8.4 Two gaps, named rather than papered over

**No `actor` field.** `LedgerEntryBase` (`ledger.ts:77-87`) carries
`id, ts, kind, scope, user_id, conversation_id, turn, data`. Nothing separates a
model-initiated write from a human-initiated revert except the kind. Add
`actor: "model" | "human" | "system"`, **optional**, so existing lines still
parse. Without it, "who did this" is inference dressed as record.

**The ledger records mutations, not reads.** There is no recall kind
(`ledger.ts:71-75`). Recalls are therefore read from the durable conversation
events rather than duplicated into the ledger, which would raise ledger volume by
roughly an order of magnitude for data that already exists. For a defence buyer
*who read what* is often the more sensitive question, so if read-auditing is
wanted as a first-class durable kind it is a deliberate decision with real
storage cost — **explicitly deferred, not silently included.**

### 8.5 Export

NDJSON — the ledger's own on-disk format. What leaves the screen is byte-identical
to what is on disk, so there is no "did the export transform it" question to
answer.

## 9. Error handling

| condition | behaviour |
|---|---|
| seat offline | dock hides (existing `:157` branch); field and history still render from the backend |
| backend offline | field shows the offline state; no command may claim a frame it cannot draw |
| malformed tool args | validation fails closed; row logged *unfulfilled* with reason |
| agent command declined by authority rule | **Follow** affordance; row logged *offered* |
| ledger read fails | history shows the session-scoped rows and states that durable rows are unavailable — never an empty timeline implying nothing happened |
| stream drops mid-turn | existing `transportError` path (`stores/chat.ts:356`); dock strip shows it |

The rule behind the last two: **an empty audit surface must never be
indistinguishable from a working one with nothing to show.**

## 10. Testing

Frontend tests run under vitest (`front/ui`, `npm test`).

- **View store** — reducer tests per command; author tracking; the authority rule,
  including that per-dimension touch does not over-block.
- **Adapter** — a recorded `SeatEvent` sequence in, an expected command sequence
  out. This is the sync contract and it is the test that must not be weakened.
- **Validation** — malformed and hostile `args` fail closed and produce an
  `unfulfilled` row rather than throwing or dispatching.
- **Fit** — framing a known node extent produces a transform that contains every
  node, at several viewport aspect ratios. This is the regression guard for the
  defect in §3.
- **Dock** — a turn started on one destination is still streaming, and still
  visible, after navigating to two others.
- **History** — ledger rows, conversation rows and session rows merge in
  timestamp order; a failed ledger read degrades per §9.

Every test must be shown to fail before the change that makes it pass. A test
that cannot fail is invisible to a failing-test sweep.

## 11. Sequencing

1. View store + adapter + implicit sync (`memory_recall` → `frame`). No backend
   change. Delivers most of the effect.
2. Field canvas: computed fit as rest state; the swell/ghost transition.
3. Conversation dock: always present, collapse-not-dismiss.
4. History surface: crumb expanded, three sources merged, export, revert.
5. Ledger `actor` field.
6. Explicit view tools in the seat.

## 12. Deferred

**Session replay.** Because every command is serializable and every row is
timestamped, scrubbing a session and re-animating the field is a consequence of
this design rather than a new mechanism. It is not built in this pass.

**Read-auditing as a durable ledger kind.** See §8.4.
