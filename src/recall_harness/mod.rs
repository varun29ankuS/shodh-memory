//! Recall Harness
//!
//! Quality measurement for the full `MemorySystem::recall_*` pipeline.
//! Designed to drive baseline-comparison CI gates and embedder swap decisions
//! (see project `recall-harness`, issues #263–#270).
//!
//! Submodules are added incrementally as the project lands:
//! - `metrics` — NDCG@k, recall@k, precision@k, MRR, P@1, MAP (RH-2)
//! - Future: `fixtures` (RH-3, RH-7), `runner` (RH-4), `report` (RH-4)

pub mod metrics;
