# Agent run record — canonicalize merge provenance (M3/M6)

**Date:** 2026-08-06 · **Branch:** `fix/canonicalize-merge-provenance`
**Commits:** `27eee599` (implementation), `58c37395` (verification fixes)

This file exists because the *code* survives in git but the *reasoning* does not.
Two months from now the diff will still be here; what was checked, what was proved,
and what was deliberately left unproved would otherwise be gone.

## What was fixed

- **M3** — canonicalize-merge destroyed merged entities' episode provenance.
  `delete_entity` wiped `entity_episodes`; nothing re-indexed under the canonical
  UUID; `entity_refs` dangled; mention counts / labels / attributes were discarded.
- **M6** — edge repoint was `let _ = delete(...)` then `add(...)`, both results
  ignored, no `WriteBatch`. An add failure destroyed the edge permanently.

Approach: an `entity_merge_audit` column family holding a tombstone snapshot of the
merged member, plus `merge_entity_into` / `repoint_edge_onto` doing all index
mutations in single `WriteBatch`es with every result propagated.

Prior art that shaped it: TOKI (arXiv 2606.06240) names "audit erasure" as a
write-time anomaly and prescribes a dual-row current/audit schema; `semantica-agi/semantica`
(MIT) ships the same idea as `invalidated` tombstones. Neither was copied — the shape
was adapted to this codebase's CF model.

## Verification — how it was checked, not just that it was

An adversarial pass was run with a brief to **refute**, defaulting to "not proven".

| Claim | Verdict | Evidence |
|---|---|---|
| New CF breaks existing DBs | **REFUTED (safe)** | `create_missing_column_families(true)` at `graph_memory.rs:2399` + `open_cf_descriptors` over full `GRAPH_CF_NAMES`. Proved empirically by `graph_db_from_before_merge_audit_cf_opens_and_gets_cf_created`, which builds a DB lacking the CF and reopens via the production constructor. |
| Fix reachable from production | **CONFIRMED** | `canonicalize_entities` → `merge_entity_into` at `:3163`; reached from the 6-hourly cycle (`handlers/state.rs:2165`) and `handlers/graph.rs:295`. Not dead code. |
| M6 atomicity | **CONFIRMED** | Edge record + both `entity_edges` rows + `entity_pair_index` in one `WriteBatch`; all reads/writes `?`-propagated. No `let _ =` remains on any delete/add in the merge path. |
| Idempotence over the 6h timer | **CONFIRMED** | Completed merge → member deleted → `get_entity` None → `Ok(0)`. No re-fold, no strength drift. |
| Self-merge / cycles | **CONFIRMED safe** | `member == canonical` → early `Ok(0)` (`:3234`); A→B then B→A → hard error rather than destruction (`:3240-3244`). |
| Ranking neutrality | **CONFIRMED at merge time, NOT forward-in-time** | See below — this is the important one. |
| Audit rows pruned | **CONFIRMED unbounded** | Only writer `merge_entity_into`, only reader `get_merge_audit`. KBs–low-MBs/year. Accepted, not fixed. |

### Defect found *by* the verification

Phase 1 blindly overwrote an existing audit tombstone on retry-after-partial-failure,
destroying the only snapshot of edges the first attempt had already consumed — the exact
data-loss class this change exists to close, hiding inside the fix for it. Now unions by
UUID. Regression test: `merge_retry_unions_audit_row_instead_of_overwriting`.

### The spec was wrong, and that matters

The original brief said the change "must NOT change recall rankings." Verification showed
that is unattainable while fixing M3, because three behaviour changes *are* the fix:

1. Surviving `mention_count` feeds `frequency_boost` on salience (`:3830`), and salience
   feeds proactive-context ordering (`state.rs:3196`) and entity filters.
2. Plain repoint no longer resets edge `created_at`, so old active edges can now reach
   time-aware Full LTP (`:1268`) → slower decay → higher effective strength.
3. Preserved edge UUID retains per-edge activation state.

All three follow from *not destroying data*. A version of this fix that left rankings
bit-identical forever would not be fixing M3.

## Known-unproved (do not re-derive; these were decided, not missed)

- `WriteBatch` **write**-failure atomicity is proved by construction only. The failure test
  injects a **read** failure; testing a write failure needs fault injection.
- Crash between phase-4 canonical write and phase-5 `delete_entity` can double
  `mention_count` on retry. One put+delete wide. A real fix drags `delete_entity`'s
  in-memory index maintenance into scope — judged out of proportion for a stat counter.
- Episodes referencing the member in `entity_refs` with no `entity_episodes` row keep a
  dangling ref. Pre-existing; such episodes were already invisible to
  `get_episodes_by_entity`.
- CF addition is **one-way**: after this binary opens a DB, an older binary with an explicit
  CF list cannot. Same as when `entity_alias` shipped.
- `src/migration.rs:514-548` lists CFs independently and already omits
  `relation_stats`/`entity_alias` on main — pre-existing gap in the legacy migration tool.
- Live-eval confirmation that recall metrics are unmoved was **not run**. Neutrality
  analysis above is code-level only.

## Environment notes for whoever runs this next

- `cargo` builds inside `.claude/worktrees/` hit the Windows **MAX_PATH** limit (~262 chars).
  An external `CARGO_TARGET_DIR` is required. This is also the **confirmed** cause of the
  `librocksdb-sys` build failures other agents hit in worktrees: `fatal error C1083`, object
  path measured at 262 chars. Two wrong diagnoses were recorded before that measurement —
  OneDrive (inferred from the directory name; no OneDrive process runs here and the folder
  is not a sync point) and Defender (real-time scanning is on, but was never observed doing
  anything). Neither was ever true. Measure the path length before blaming a process.
- `cargo clippy --all-targets` is **red on `main`**, independent of this branch: two
  `absurd_extreme_comparisons` deny-errors in `tests/brutal_stress_tests.rs` (`:687`, `:1303`)
  — vacuous `unsigned >= 0` assertions, same class as the one fixed in #426.
