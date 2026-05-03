//! Recall Harness
//!
//! Quality measurement for the full `MemorySystem::recall_*` pipeline.
//! Designed to drive baseline-comparison CI gates and embedder swap decisions
//! (see project `recall-harness`, issues #263–#270).
//!
//! Submodules are added incrementally as the project lands:
//! - `metrics` — NDCG@k, recall@k, precision@k, MRR, P@1, MAP (RH-2)
//! - `fixtures` — L1 smoke suite loader + structural validation (RH-3)
//! - `runner` — end-to-end smoke runner against `MemorySystem` (RH-4)
//! - `report` — JSON schema + baseline comparison (RH-4)

pub mod fixtures;
pub mod metrics;
pub mod report;
pub mod runner;
