//! Memory Consolidation Tests
//!
//! Tests for semantic consolidation and memory compression:
//! - Fact extraction from episodic memories
//! - Compression of old memories
//! - Tier migration (Working -> Session -> LongTerm -> Archive)
//! - Importance-based retention
//! - Auto-compression triggers

use std::sync::Arc;

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::graph_memory::GraphMemory;
use shodh_memory::memory::{
    Experience, ExperienceType, Memory, MemoryConfig, MemoryId, MemorySystem, MemoryTier,
};
use shodh_memory::uuid::Uuid;
use tempfile::TempDir;

/// Create fallback NER for testing (rule-based, no ONNX required)
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Create experience with NER entity extraction
fn create_experience_with_ner(content: &str, ner: &NeuralNer) -> Experience {
    let entities = ner.extract(content).unwrap_or_default();
    let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
    Experience {
        experience_type: ExperienceType::Observation,
        content: content.to_string(),
        entities: entity_names,
        ..Default::default()
    }
}

/// Create test memory system with knowledge graph
fn setup_memory_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 50,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 200,
        auto_compress: false,
        compression_age_days: 1,
        importance_threshold: 0.7,
    };

    let mut memory_system =
        MemorySystem::new(config, None).expect("Failed to create memory system");
    let graph_path = temp_dir.path().join("graph");
    let graph = GraphMemory::new(&graph_path, None).expect("Failed to create graph memory");
    memory_system.set_graph_memory(Arc::new(shodh_memory::parking_lot::RwLock::new(graph)));
    (memory_system, temp_dir)
}

fn setup_auto_compress_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 10, // Small to trigger eviction
        session_memory_size_mb: 1,
        max_heap_per_user_mb: 10,
        auto_compress: true,
        compression_age_days: 0, // Immediately eligible
        importance_threshold: 0.5,
    };

    let memory_system = MemorySystem::new(config, None).expect("Failed to create memory system");
    (memory_system, temp_dir)
}

fn create_experience(content: &str) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        ..Default::default()
    }
}

// =============================================================================
// TIER MIGRATION TESTS
// =============================================================================

#[test]
fn test_memory_starts_in_working() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );
    assert_eq!(memory.tier, MemoryTier::Working);
}

#[test]
fn test_promote_working_to_session() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
}

#[test]
fn test_promote_session_to_longterm() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );
    memory.tier = MemoryTier::Session;
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);
}

#[test]
fn test_promote_longterm_is_terminal() {
    // The LongTerm → Archive arrow was removed. No production path drove it,
    // and Archive's documented job (compressed archival) already happens on
    // entry to LongTerm via `should_compress` → `MemoryCompressor::compress`.
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );
    memory.tier = MemoryTier::LongTerm;
    memory.promote();
    assert_eq!(
        memory.tier,
        MemoryTier::LongTerm,
        "LongTerm is terminal — promote() must not invent an Archive"
    );
}

#[test]
fn test_promote_archive_stays_archive() {
    // Archive is a RETIRED variant, kept only so that a postcard record
    // carrying discriminant 3 still decodes. Nothing assigns it, and promote()
    // must leave it alone rather than silently rewriting it to another tier.
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );
    memory.tier = MemoryTier::Archive;
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Archive);
}

// `demote()` was removed — it had zero production callers, nothing in the
// system produces the signal it would need, and tier is a retrieval input
// (the graph-leg multiplier), so demotion would flap a memory's rank between
// recalls for any memory sitting near a threshold. The four per-step demotion
// tests that lived here tested only that removed method.

// =============================================================================
// TIER TRANSITIONS MUST BE DURABLE
//
// Every memory is written to long-term storage at insert, so the persisted
// record starts life with `tier: Working`. Working→Session used to be a pure
// in-memory map move with NO storage write, while the one place recall reads
// tier — the memory-tier graph multiplier in `graph_retrieval` — materializes
// memories from STORAGE. The persisted tier was therefore almost always
// `Working`, so nearly everything scored at the 0.3 multiplier and the 0.6
// `Session` branch was effectively unreachable.
//
// These tests pin the write, not the in-memory move.
// =============================================================================

/// Build an experience that comfortably clears the promotion importance bar.
///
/// `TIER_PROMOTION_WORKING_IMPORTANCE` is 0.35. A `Decision` contributes 0.3 on
/// the type factor and >50 words contributes 0.25 on richness, so this clears it
/// with margin — the test is about the storage write, and it must not become
/// vacuous because the importance heuristic drifted a little.
fn promotable_experience(content_seed: &str) -> Experience {
    let body = std::iter::repeat(content_seed)
        .take(60)
        .collect::<Vec<_>>()
        .join(" ");
    Experience {
        content: format!("Decision: {body}"),
        experience_type: ExperienceType::Decision,
        ..Default::default()
    }
}

#[test]
fn working_to_session_promotion_is_persisted_not_just_moved_in_memory() {
    let (system, _tmp) = setup_memory_system();

    // Control: a memory younger than TIER_PROMOTION_WORKING_AGE_SECS (1800s) is
    // not eligible, so it must still persist as Working. Before this fix, EVERY
    // memory looked like this forever, whatever its real tier.
    let fresh_id = system
        .remember(promotable_experience("not yet eligible"), None)
        .expect("remember failed");
    assert_eq!(
        system
            .get_memory(&fresh_id)
            .expect("memory should be in storage")
            .tier,
        MemoryTier::Working,
        "a memory inside the 30-minute window persists as Working"
    );

    // Subject: backdate past the age threshold so the memory is eligible.
    // `remember` itself runs `consolidate_if_needed` on the request path, so
    // this is already promoted by the time it returns; `run_maintenance` is the
    // background path and must agree.
    let created_at = chrono::Utc::now() - chrono::Duration::seconds(3600);
    let id = system
        .remember(
            promotable_experience("durable tier transition"),
            Some(created_at),
        )
        .expect("remember failed");

    system
        .run_maintenance(1.0, "default", false)
        .expect("maintenance failed");

    // The assertion that matters: STORAGE, not the in-memory map, says Session.
    // `get_memory` reads long-term storage, which is exactly what recall's tier
    // multiplier materializes from.
    let after = system
        .get_memory(&id)
        .expect("memory should still be stored");
    assert_eq!(
        after.tier,
        MemoryTier::Session,
        "Working→Session must be written through to storage, not only moved between maps"
    );

    // ...and the control is still Working, so the write is driven by the
    // promotion criteria rather than being unconditional.
    assert_eq!(
        system
            .get_memory(&fresh_id)
            .expect("memory should be in storage")
            .tier,
        MemoryTier::Working,
        "an ineligible memory must not be swept along"
    );
}

#[test]
fn session_tier_survives_a_serialization_round_trip() {
    // `tier` is already a field of `MemoryFlat`, so persisting the transition
    // needed no schema change — but nothing pinned the Session value, only
    // LongTerm. If Session ever failed to round-trip, the promotion write above
    // would be silently undone on read.
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None,
    );
    memory.tier = MemoryTier::Session;

    let bytes =
        bincode::serde::encode_to_vec(&memory, bincode::config::standard()).expect("serialize");
    let (decoded, _): (Memory, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .expect("deserialize");
    assert_eq!(decoded.tier, MemoryTier::Session);
}

#[test]
fn every_live_tier_maps_to_a_distinct_graph_multiplier() {
    // The declared ladder: three live tiers, three DIFFERENT graph-leg weights,
    // strictly increasing with consolidation.
    //
    // NOTE what this pins and what it does not. It pins the CONSTANTS, not the
    // retrieval path. `graph_retrieval::memory_tier_graph_trust` deliberately
    // holds `Session` at the `Working` value: the promotion write above made the
    // 0.6 rung reachable for the very first time, and measured on the
    // LoCoMo-100 gate it costs recall (multi_hop ndcg 0.2652 -> 0.2516,
    // open_domain recall@10 0.2750 -> 0.2250). Promoting the middle rung is a
    // measured decision; `session_graph_trust_is_held_at_the_working_value` is
    // the tripwire that forces it to be made deliberately. The ladder stays
    // declared here so the value survives for whoever calibrates it.
    use shodh_memory::constants::{
        MEMORY_TIER_GRAPH_MULT_LONGTERM, MEMORY_TIER_GRAPH_MULT_SESSION,
        MEMORY_TIER_GRAPH_MULT_WORKING,
    };
    assert!(
        MEMORY_TIER_GRAPH_MULT_WORKING < MEMORY_TIER_GRAPH_MULT_SESSION
            && MEMORY_TIER_GRAPH_MULT_SESSION < MEMORY_TIER_GRAPH_MULT_LONGTERM,
        "tier trust must be strictly increasing: {MEMORY_TIER_GRAPH_MULT_WORKING} < \
         {MEMORY_TIER_GRAPH_MULT_SESSION} < {MEMORY_TIER_GRAPH_MULT_LONGTERM}"
    );
}

// =============================================================================
// FACT CORRECTION — END TO END
// =============================================================================

/// A creation timestamp `days` in the past, pinned to WHOLE MILLISECONDS.
///
/// The incremental fact-extraction watermark is persisted as
/// `created_at.timestamp_millis()` and the next cycle filters with a strict
/// `created_at > watermark`. `Utc::now()` carries sub-millisecond digits, so a
/// memory always compares as newer than the watermark it just produced and is
/// re-consolidated on every subsequent cycle.
///
/// In production that is merely wasteful — dedup absorbs the re-derivation. In a
/// two-cycle test it is fatal and silently so: cycle two re-ingests the claim
/// memories alongside the correction, and since the two share most of their
/// content stems they land in ONE Jaccard cluster, producing a single merged
/// fact (observed: one fact with four source memories) instead of a claim plus a
/// contradicting correction. Pinning the fixture to whole milliseconds makes the
/// watermark round-trip exactly, so each cycle sees only its own cohort.
fn days_ago_ms(days: i64) -> chrono::DateTime<chrono::Utc> {
    let t = chrono::Utc::now() - chrono::Duration::days(days);
    chrono::DateTime::from_timestamp_millis(t.timestamp_millis()).expect("valid timestamp")
}

/// Store a plain declarative memory with a pinned importance and creation time.
///
/// `importance_override` is set so cluster-representative selection (highest
/// candidate confidence wins) is deterministic rather than a function of the
/// importance heuristic.
fn declarative_memory(content: &str, importance: f32) -> Experience {
    Experience {
        content: content.to_string(),
        experience_type: ExperienceType::Observation,
        importance_override: Some(importance),
        ..Default::default()
    }
}

/// remember -> extraction -> contradiction -> invalidation, in one run.
///
/// This test existed once, passed VACUOUSLY, and was deleted. The harness
/// extracted ZERO facts, so every assertion about contradictions was trivially
/// satisfied. The cause was the candidate extractor, not the invalidation work:
/// three gates in series on the only route a plain declarative sentence had —
/// `ExperienceType::Observation` only, `importance >= 0.5`, and a hard
/// requirement that the sentence literally contain one of the NER entities —
/// each of which fails CLOSED. `remember` defaults to `Observation`,
/// `calculate_importance` gives `Observation` the 0.05 catch-all weight so the
/// default type could not reach 0.5, and NER emission is empty for whole classes
/// of records. See `SemanticConsolidator::extract_fact_candidates`.
///
/// It is resurrected with the non-vacuity assertion FIRST: `facts > 0` is
/// checked, and the identity of the extracted fact pinned, before anything about
/// arbitration is asserted. The test cannot pass again by exercising nothing.
///
/// TWO maintenance cycles, deliberately. Facts minted in one `consolidate` batch
/// are all written AFTER the arbitration loop (`store_batch` runs once the loop
/// finishes), so no fact in a batch can see its batch-mates in the store.
/// Contradiction can only fire when the correction arrives on a LATER cycle than
/// the claim — which is also how a correction arrives in the world. The
/// incremental watermark cooperates: it advances to the newest processed
/// memory's `created_at`, so the claim memories are excluded from cycle two by
/// the same mechanism that admits the correction.
#[test]
fn a_corrected_report_supersedes_the_initial_claim_end_to_end() {
    use shodh_memory::similarity::cosine_similarity;

    let (system, _dir) = setup_memory_system();
    let user = "e2e-correction";

    // Both cohorts are older than CONSOLIDATION_MIN_AGE_DAYS (7). The correction
    // is NEWER than the claim so it clears the watermark that cycle one leaves
    // behind, and the claim does not (the filter is a strict `>` — see
    // `days_ago_ms` for why the millisecond pinning matters).
    let claim_at = days_ago_ms(40);
    let correction_at = days_ago_ms(20);

    // CONSOLIDATION_MIN_SUPPORT is 2, so each claim needs two DISTINCT memories
    // whose extracted sentences cluster together. Corroboration is what mints a
    // fact — that is the filter that replaced the entity gate.
    const CLAIM: &str =
        "Initial reports said four crew members were injured in the bridge collapse";
    const CLAIM_ECHO: &str =
        "Early reports said four crew members were injured in the bridge collapse";
    // The correction is worded as a minimal negation of the claim on purpose.
    // `find_contradiction` reuses dedup's thresholds — cosine >= 0.80 AND
    // Jaccard >= 0.30 AND a shared entity — with polarity INVERTED, so a
    // contradiction only registers between two claims the system would otherwise
    // have considered the same fact. Those thresholds are load-bearing for the
    // invalidation semantics and are not to be relaxed to make a test pass; the
    // corpus wording is what has to clear them.
    const CORRECTION: &str =
        "Corrected reports confirm no crew members were injured in the bridge collapse";
    const CORRECTION_ECHO: &str =
        "Later reports confirm no crew members were injured in the bridge collapse";

    // ── Cycle 1: the claim ──────────────────────────────────────────────────
    system
        .remember(
            declarative_memory(&format!("{CLAIM}."), 0.9),
            Some(claim_at),
        )
        .expect("remember claim");
    system
        .remember(
            declarative_memory(&format!("{CLAIM_ECHO}."), 0.8),
            Some(claim_at),
        )
        .expect("remember claim echo");

    system
        .run_maintenance(1.0, user, true)
        .expect("heavy maintenance");

    let after_claim = system.get_facts(user, 100).expect("facts");
    // NON-VACUITY. Everything below is meaningless if this is empty, which is
    // exactly the state the deleted version of this test shipped in.
    assert!(
        !after_claim.is_empty(),
        "extraction produced no facts from two corroborating declarative memories — \
         the candidate extractor is silent again and every assertion below is vacuous"
    );
    let claim_fact = after_claim
        .iter()
        .find(|f| f.fact.contains("crew members were injured"))
        .unwrap_or_else(|| {
            panic!("expected the claim to be extracted verbatim, got {after_claim:?}")
        })
        .clone();
    assert!(
        claim_fact.is_active(),
        "a freshly extracted fact must be active"
    );

    // ── Cycle 2: the correction ─────────────────────────────────────────────
    system
        .remember(
            declarative_memory(&format!("{CORRECTION}."), 0.9),
            Some(correction_at),
        )
        .expect("remember correction");
    system
        .remember(
            declarative_memory(&format!("{CORRECTION_ECHO}."), 0.8),
            Some(correction_at),
        )
        .expect("remember correction echo");

    system
        .run_maintenance(1.0, user, true)
        .expect("heavy maintenance");

    let after_correction = system.get_facts(user, 100).expect("facts");
    let correction_fact = after_correction
        .iter()
        .find(|f| f.fact.contains("no crew members were injured"))
        .unwrap_or_else(|| panic!("the correction was never extracted, got {after_correction:?}"))
        .clone();

    // ── Arbitration ─────────────────────────────────────────────────────────
    let store = system.fact_store();
    let settled_claim = store
        .get(user, &claim_fact.id)
        .expect("read claim")
        .expect("claim still stored — invalidation retains, never deletes");

    // A failure here is almost always one of `find_contradiction`'s three gates
    // rather than the arbitration rule, so report all three.
    if settled_claim.is_active() {
        let claim_emb = store.get_embedding(user, &claim_fact.id).ok().flatten();
        let corr_emb = store
            .get_embedding(user, &correction_fact.id)
            .ok()
            .flatten();
        let cosine = match (&claim_emb, &corr_emb) {
            (Some(a), Some(b)) => cosine_similarity(a, b),
            _ => f32::NAN,
        };
        let shared: Vec<&String> = claim_fact
            .related_entities
            .iter()
            .filter(|e| correction_fact.related_entities.contains(e))
            .collect();
        panic!(
            "the correction did not supersede the claim.\n  claim: {:?}\n  correction: {:?}\n  \
             cosine (needs >= FACT_DEDUP_COSINE_THRESHOLD 0.80): {cosine}\n  \
             claim entities: {:?}\n  correction entities: {:?}\n  shared entities \
             (needs >= 1): {shared:?}",
            settled_claim.fact,
            correction_fact.fact,
            claim_fact.related_entities,
            correction_fact.related_entities,
        );
    }

    assert_eq!(
        settled_claim.invalidated_by.as_deref(),
        Some(correction_fact.id.as_str()),
        "the superseded claim must name its victor"
    );
    assert!(
        settled_claim.contradicts.contains(&correction_fact.id),
        "the link must be recorded on the superseded side"
    );
    assert!(
        !settled_claim.source_memories.is_empty(),
        "invalidation must not break the trust chain back to the episodes"
    );

    let settled_correction = store
        .get(user, &correction_fact.id)
        .expect("read correction")
        .expect("correction stored");
    assert!(
        settled_correction.is_active(),
        "the winning correction must remain active"
    );
    assert!(
        settled_correction.contradicts.contains(&claim_fact.id),
        "the winner must remember what it displaced"
    );
}

/// The ON-DEMAND distillation path must arbitrate contradictions exactly as the
/// timer-driven maintenance path does.
///
/// It did not. `distill_facts` called `find_similar` and, on a miss, pushed the
/// fact straight onto the "genuinely new" pile — `find_contradiction` was never
/// reached. So a correction distilled on demand did not supersede the claim it
/// corrected: both rows stayed ACTIVE and unlinked, each ratcheting its own
/// confidence and extending its own half-life, which is precisely the state the
/// invalidation increment exists to prevent. Whether the system ended up
/// believing one thing or two contradictory things depended on nothing more
/// principled than whether a human hit the consolidate endpoint or a timer fired
/// first.
///
/// This is the maintenance-path correction test (`a_corrected_report_supersedes_
/// the_initial_claim_end_to_end`) re-run through `distill_facts`, and it must
/// reach the same verdict. Both now route through
/// `SemanticFactStore::ingest_candidate`.
///
/// Non-vacuity is asserted first: if extraction is silent, every arbitration
/// assertion below is trivially satisfied — the exact way the original version
/// of the maintenance test shipped green while testing nothing.
#[test]
fn on_demand_distillation_arbitrates_a_contradiction_like_maintenance_does() {
    let (system, _dir) = setup_memory_system();
    let user = "e2e-distill-correction";

    let claim_at = days_ago_ms(40);
    let correction_at = days_ago_ms(20);

    const CLAIM: &str =
        "Initial reports said four crew members were injured in the bridge collapse";
    const CLAIM_ECHO: &str =
        "Early reports said four crew members were injured in the bridge collapse";
    const CORRECTION: &str =
        "Corrected reports confirm no crew members were injured in the bridge collapse";
    const CORRECTION_ECHO: &str =
        "Later reports confirm no crew members were injured in the bridge collapse";

    // ── Cycle 1: the claim ──────────────────────────────────────────────────
    system
        .remember(declarative_memory(&format!("{CLAIM}."), 0.9), Some(claim_at))
        .expect("remember claim");
    system
        .remember(
            declarative_memory(&format!("{CLAIM_ECHO}."), 0.8),
            Some(claim_at),
        )
        .expect("remember claim echo");

    system.distill_facts(user, 2, 7).expect("distill claim");

    let after_claim = system.get_facts(user, 100).expect("facts");
    assert!(
        !after_claim.is_empty(),
        "on-demand distillation produced no facts — every assertion below is vacuous"
    );
    let claim_fact = after_claim
        .iter()
        .find(|f| f.fact.contains("crew members were injured"))
        .unwrap_or_else(|| panic!("expected the claim to be extracted, got {after_claim:?}"))
        .clone();
    assert!(claim_fact.is_active(), "a fresh fact must be active");

    // ── Cycle 2: the correction ─────────────────────────────────────────────
    system
        .remember(
            declarative_memory(&format!("{CORRECTION}."), 0.9),
            Some(correction_at),
        )
        .expect("remember correction");
    system
        .remember(
            declarative_memory(&format!("{CORRECTION_ECHO}."), 0.8),
            Some(correction_at),
        )
        .expect("remember correction echo");

    system.distill_facts(user, 2, 7).expect("distill correction");

    let store = system.fact_store();
    let all = store.list(user, 100).expect("list");
    let correction_fact = all
        .iter()
        .find(|f| f.fact.contains("no crew members were injured"))
        .unwrap_or_else(|| panic!("the correction was never extracted, got {all:?}"))
        .clone();

    // ── Arbitration ─────────────────────────────────────────────────────────
    let active: Vec<_> = all.iter().filter(|f| f.is_active()).collect();
    assert_eq!(
        active.len(),
        1,
        "a claim and its negation must not both be believed. Active: {active:?}"
    );
    assert_eq!(
        active[0].id, correction_fact.id,
        "the NEWER claim wins on equal support — recency is the documented default \
         because a contradiction arriving later is usually a correction"
    );

    let settled_claim = store
        .get(user, &claim_fact.id)
        .expect("read claim")
        .expect("claim still stored — invalidation retains, never deletes");
    assert!(!settled_claim.is_active());
    assert_eq!(
        settled_claim.invalidated_by.as_deref(),
        Some(correction_fact.id.as_str()),
        "the superseded claim must name its victor"
    );
    assert!(
        settled_claim.contradicts.contains(&correction_fact.id),
        "the link must be recorded on the superseded side"
    );
    assert!(
        !settled_claim.source_memories.is_empty(),
        "invalidation must not break the trust chain back to the episodes"
    );

    let settled_correction = store
        .get(user, &correction_fact.id)
        .expect("read correction")
        .expect("correction stored");
    assert!(
        settled_correction.contradicts.contains(&claim_fact.id),
        "the winner must remember what it displaced"
    );

    // ── Stability: re-distilling must not oscillate ─────────────────────────
    // The wrong claim is still in the corpus, so it is re-extracted forever.
    // Repeated distillation must keep landing it on its own dead row.
    for round in 0..3 {
        system
            .distill_facts(user, 2, 7)
            .unwrap_or_else(|e| panic!("round {round}: distill failed: {e}"));
        let now_active: Vec<_> = store
            .list(user, 100)
            .expect("list")
            .into_iter()
            .filter(|f| f.is_active())
            .collect();
        assert_eq!(
            now_active.len(),
            1,
            "round {round}: exactly one fact may be active. Got {now_active:?}"
        );
        assert_eq!(
            now_active[0].id, correction_fact.id,
            "round {round}: the verdict must not flip-flop"
        );
    }
}

/// A superseded fact must not be resurrected by the ON-DEMAND distillation path.
///
/// `find_similar` deliberately still MATCHES invalidated rows — that is what
/// stops a corrected claim and its correction from trading places on every
/// cycle. `run_maintenance` therefore checks `is_active()` before reinforcing.
/// `distill_facts` did not, so a wrong claim (which normally stays in the corpus
/// and is re-extracted forever) would get its `last_reinforced` refreshed and,
/// since the disuse half-life grows with support, become MORE durable each time
/// someone asked for a distillation.
///
/// The bug was unreachable until now for the same reason this whole branch
/// exists: the candidate extractor produced nothing, so the reinforcement branch
/// never ran with real input.
///
/// Non-vacuity: the fact count must stay at 1. If `find_similar` had failed to
/// match the dead row at all, the re-derivation would be stored as a SECOND
/// fact, and the "unchanged timestamp" assertions would pass while testing
/// nothing.
#[test]
fn on_demand_distillation_does_not_resurrect_an_invalidated_fact() {
    let (system, _dir) = setup_memory_system();
    let user = "e2e-no-resurrect";

    let first_at = days_ago_ms(40);
    let later_at = days_ago_ms(20);

    system
        .remember(
            declarative_memory(
                "Initial reports said four crew members were injured in the bridge collapse.",
                0.9,
            ),
            Some(first_at),
        )
        .expect("remember claim");
    system
        .remember(
            declarative_memory(
                "Early reports said four crew members were injured in the bridge collapse.",
                0.8,
            ),
            Some(first_at),
        )
        .expect("remember claim echo");

    let distilled = system.distill_facts(user, 2, 7).expect("distill");
    assert!(
        distilled.facts_extracted >= 1,
        "on-demand distillation extracted nothing — the rest of this test is vacuous"
    );

    let store = system.fact_store();
    let facts = store.list(user, 100).expect("list");
    assert_eq!(facts.len(), 1, "expected exactly one distilled fact");
    let mut dead = facts[0].clone();

    // Supersede it, then record the state that must not move.
    dead.invalidate(Some("some-later-correction"), chrono::Utc::now());
    store.update(user, &dead).expect("invalidate");
    let frozen_reinforced = dead.last_reinforced;
    let frozen_support = dead.support_count;

    // The wrong claim is still in the corpus and gets re-asserted.
    system
        .remember(
            declarative_memory(
                "Initial dispatches said four crew members were injured in the bridge collapse.",
                0.9,
            ),
            Some(later_at),
        )
        .expect("remember re-derivation");
    system
        .remember(
            declarative_memory(
                "Early dispatches said four crew members were injured in the bridge collapse.",
                0.8,
            ),
            Some(later_at),
        )
        .expect("remember re-derivation echo");

    system.distill_facts(user, 2, 7).expect("distill again");

    let after = store.list(user, 100).expect("list");
    assert_eq!(
        after.len(),
        1,
        "the re-derivation must have been RECOGNISED as the dead fact, not stored as a \
         second row — otherwise the assertions below prove nothing. Got: {after:?}"
    );

    let settled = store.get(user, &dead.id).expect("read").expect("stored");
    assert!(
        !settled.is_active(),
        "a superseded fact must stay superseded"
    );
    assert_eq!(
        settled.last_reinforced, frozen_reinforced,
        "re-deriving a dead fact must not extend its half-life"
    );
    assert_eq!(
        settled.support_count, frozen_support,
        "a dead fact must not accrue support"
    );
}

#[test]
fn test_tier_full_cycle() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );

    // Promote all the way — the ladder is monotonic and saturates at LongTerm.
    assert_eq!(memory.tier, MemoryTier::Working);
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::LongTerm);
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::LongTerm, "terminal, not Archive");

    // There is no way back down: `demote()` was removed (see above).
}

// =============================================================================
// TIER PRESERVATION TESTS
// =============================================================================

#[test]
fn test_tier_preserved_on_serialization() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );
    memory.tier = MemoryTier::LongTerm;

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard()).unwrap();

    assert_eq!(deserialized.tier, MemoryTier::LongTerm);
}

#[test]
fn test_all_tiers_serialize() {
    for tier in [
        MemoryTier::Working,
        MemoryTier::Session,
        MemoryTier::LongTerm,
        MemoryTier::Archive,
    ] {
        let mut memory = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );
        memory.tier = tier;

        let serialized =
            bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
        let (deserialized, _): (Memory, _) =
            bincode::serde::decode_from_slice(&serialized, bincode::config::standard()).unwrap();

        assert_eq!(deserialized.tier, tier);
    }
}

// =============================================================================
// TIER EQUALITY TESTS
// =============================================================================

#[test]
fn test_tier_equality() {
    assert_eq!(MemoryTier::Working, MemoryTier::Working);
    assert_eq!(MemoryTier::Session, MemoryTier::Session);
    assert_eq!(MemoryTier::LongTerm, MemoryTier::LongTerm);
    assert_eq!(MemoryTier::Archive, MemoryTier::Archive);
}

#[test]
fn test_tier_inequality() {
    assert_ne!(MemoryTier::Working, MemoryTier::Session);
    assert_ne!(MemoryTier::Session, MemoryTier::LongTerm);
    assert_ne!(MemoryTier::LongTerm, MemoryTier::Archive);
}

#[test]
fn test_tier_default() {
    assert_eq!(MemoryTier::default(), MemoryTier::Working);
}

// =============================================================================
// IMPORTANCE-BASED CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_high_importance_memories_preserved() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.9, // High importance
        None,
        None,
        None,
        None, // created_at
    );

    assert!(memory.importance() >= 0.7);
}

#[test]
fn test_low_importance_eligible_for_forget() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.2, // Low importance
        None,
        None,
        None,
        None, // created_at
    );

    assert!(memory.importance() < 0.5);
}

#[test]
fn test_importance_affects_retention() {
    let (mut system, _temp) = setup_memory_system();

    // Record high importance memory
    let high_id = system
        .remember(
            Experience {
                content: "Very important observation".to_string(),
                experience_type: ExperienceType::Decision, // Decisions get higher importance
                ..Default::default()
            },
            None,
        )
        .unwrap();

    // Record low importance memory
    let low_id = system
        .remember(
            Experience {
                content: "Random observation".to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            },
            None,
        )
        .unwrap();

    // Both should exist
    assert!(system.get_memory(&high_id).is_ok());
    assert!(system.get_memory(&low_id).is_ok());
}

// =============================================================================
// MEMORY SYSTEM CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_working_memory_capacity() {
    let (mut system, _temp) = setup_auto_compress_system();

    // Record more memories than working memory capacity
    for i in 0..20 {
        system
            .remember(create_experience(&format!("Memory {}", i)), None)
            .unwrap();
    }

    // System should have handled the overflow
    let stats = system.stats();
    assert!(stats.total_memories > 0);
}

#[test]
fn test_session_memory_used() {
    let (mut system, _temp) = setup_auto_compress_system();

    // Fill up working memory
    for i in 0..15 {
        system
            .remember(create_experience(&format!("Overflow memory {}", i)), None)
            .unwrap();
    }

    let stats = system.stats();
    // Should have some memories distributed across tiers
    assert!(stats.total_memories > 0);
}

#[test]
fn test_graph_maintenance_succeeds() {
    let (system, _temp) = setup_memory_system();

    // Record some memories
    for i in 0..5 {
        system
            .remember(create_experience(&format!("To consolidate {}", i)), None)
            .unwrap();
    }

    // Graph maintenance should not panic
    system.graph_maintenance();
}

// =============================================================================
// COMPRESSION STATE TESTS
// =============================================================================

#[test]
fn test_new_memory_not_compressed() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );

    assert!(!memory.compressed);
}

#[test]
fn test_compressed_flag_serializes() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );
    memory.compressed = true;

    let serialized = bincode::serde::encode_to_vec(&memory, bincode::config::standard()).unwrap();
    let (deserialized, _): (Memory, _) =
        bincode::serde::decode_from_slice(&serialized, bincode::config::standard()).unwrap();

    assert!(deserialized.compressed);
}

// =============================================================================
// ACCESS PATTERN CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_frequently_accessed_stays_active() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );

    // Multiple accesses
    for _ in 0..10 {
        memory.record_access();
    }

    assert_eq!(memory.access_count(), 10);
}

#[test]
fn test_rarely_accessed_eligible_for_archive() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.3,
        None,
        None,
        None,
        None, // created_at
    );

    // No accesses
    assert_eq!(memory.access_count(), 0);
}

#[test]
fn test_access_updates_timestamp() {
    let memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );

    let before = memory.last_accessed();
    std::thread::sleep(std::time::Duration::from_millis(10));
    memory.record_access();
    let after = memory.last_accessed();

    assert!(after >= before);
}

// =============================================================================
// MULTI-MEMORY CONSOLIDATION TESTS
// =============================================================================

#[test]
fn test_many_memories_graph_maintenance() {
    let (system, _temp) = setup_memory_system();

    // Create many memories
    for i in 0..50 {
        system
            .remember(create_experience(&format!("Bulk memory {}", i)), None)
            .unwrap();
    }

    // Graph maintenance should work
    system.graph_maintenance();
}

#[test]
fn test_empty_system_graph_maintenance() {
    let (system, _temp) = setup_memory_system();

    // Graph maintenance on empty system
    system.graph_maintenance();
}

#[test]
fn test_multiple_graph_maintenance_calls() {
    let (system, _temp) = setup_memory_system();

    for i in 0..10 {
        system
            .remember(create_experience(&format!("Memory {}", i)), None)
            .unwrap();
    }

    // Multiple graph maintenance calls should be idempotent
    for _ in 0..5 {
        system.graph_maintenance();
    }
}

// =============================================================================
// TIER AND IMPORTANCE COMBINATION TESTS
// =============================================================================

#[test]
fn test_high_importance_working_memory() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.9,
        None,
        None,
        None,
        None, // created_at
    );

    assert_eq!(memory.tier, MemoryTier::Working);
    assert!(memory.importance() > 0.8);

    // Promote should work
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
    // Importance preserved
    assert!(memory.importance() > 0.8);
}

#[test]
fn test_low_importance_archive() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.1,
        None,
        None,
        None,
        None, // created_at
    );
    memory.tier = MemoryTier::Archive;

    assert_eq!(memory.tier, MemoryTier::Archive);
    assert!(memory.importance() < 0.2);
}

#[test]
fn test_tier_independent_of_importance() {
    let mut high = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.9,
        None,
        None,
        None,
        None, // created_at
    );

    let mut low = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.1,
        None,
        None,
        None,
        None, // created_at
    );

    // Both can be promoted
    high.promote();
    low.promote();

    assert_eq!(high.tier, MemoryTier::Session);
    assert_eq!(low.tier, MemoryTier::Session);
}

// =============================================================================
// BATCH TIER OPERATIONS
// =============================================================================

#[test]
fn test_batch_promote() {
    let mut memories: Vec<Memory> = (0..10)
        .map(|_| {
            Memory::new(
                MemoryId(Uuid::new_v4()),
                Experience::default(),
                0.5,
                None,
                None,
                None,
                None, // created_at
            )
        })
        .collect();

    for m in &mut memories {
        m.promote();
    }

    for m in &memories {
        assert_eq!(m.tier, MemoryTier::Session);
    }
}

// `test_batch_demote` removed with `demote()` — see the rationale above.

#[test]
fn test_heterogeneous_tier_batch() {
    let mut memories = Vec::new();

    for (i, tier) in [
        MemoryTier::Working,
        MemoryTier::Session,
        MemoryTier::LongTerm,
        MemoryTier::Archive,
    ]
    .iter()
    .enumerate()
    {
        let mut m = Memory::new(
            MemoryId(Uuid::new_v4()),
            Experience::default(),
            0.5,
            None,
            None,
            None,
            None, // created_at
        );
        m.tier = *tier;
        memories.push((i, m));
    }

    // Promote all
    for (_, m) in &mut memories {
        m.promote();
    }

    // Each moved up one step, except the two terminal/retired tiers which hold.
    assert_eq!(memories[0].1.tier, MemoryTier::Session);
    assert_eq!(memories[1].1.tier, MemoryTier::LongTerm);
    assert_eq!(memories[2].1.tier, MemoryTier::LongTerm, "terminal");
    assert_eq!(
        memories[3].1.tier,
        MemoryTier::Archive,
        "retired, untouched"
    );
}

// =============================================================================
// EDGE CASES
// =============================================================================

#[test]
fn test_tier_after_many_operations() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    );

    // Many promotions saturate at the terminal tier and stay there.
    for _ in 0..100 {
        memory.promote();
    }
    assert_eq!(memory.tier, MemoryTier::LongTerm);
}

#[test]
fn test_tier_with_zero_importance() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.0,
        None,
        None,
        None,
        None, // created_at
    );

    // Tier operations still work
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
}

#[test]
fn test_tier_with_max_importance() {
    let mut memory = Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        1.0,
        None,
        None,
        None,
        None, // created_at
    );

    // Tier operations still work
    memory.promote();
    assert_eq!(memory.tier, MemoryTier::Session);
}

// =============================================================================
// GRAPH STATS TESTS
// =============================================================================

#[test]
fn test_stats_report_accurate() {
    let (system, _temp) = setup_memory_system();

    for i in 0..10 {
        system
            .remember(create_experience(&format!("Stats test {}", i)), None)
            .unwrap();
    }

    let stats = system.stats();
    assert!(stats.total_memories >= 10);
}

#[test]
fn test_empty_system_stats() {
    let (system, _temp) = setup_memory_system();

    let stats = system.stats();
    assert_eq!(stats.total_memories, 0);
}

// =============================================================================
// CONCURRENT TIER OPERATIONS
// =============================================================================

#[test]
fn test_concurrent_tier_reads() {
    use std::sync::Arc;
    use std::thread;

    let memory = Arc::new(Memory::new(
        MemoryId(Uuid::new_v4()),
        Experience::default(),
        0.5,
        None,
        None,
        None,
        None, // created_at
    ));

    let handles: Vec<_> = (0..10)
        .map(|_| {
            let m = Arc::clone(&memory);
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = m.tier;
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// =============================================================================
// CONSOLIDATION INTROSPECTION TESTS (SHO-28)
// =============================================================================
//
// SHIPPED-SEMANTICS NOTE for the introspection tests below that used co-retrieval
// as their event source.
//
// `f6b730ee` (2026-07-10) flipped `SHODH_COACT_STRENGTHEN_ONLY` to default-ON
// ("strengthen-not-create"): `GraphMemory::record_memory_coactivation` only
// STRENGTHENS a memory-to-memory edge that already exists between a co-active
// pair, it no longer mints a `CoRetrieved` edge per co-retrieved pair (that
// un-gated all-pairs minting was ~80% of all graph edges and the recall-time
// OOM driver). The mint branch is the only writer of the `mem_edge:` pair index
// the strengthen branch reads, so with minting off the lookup always misses and
// `record_memory_coactivation` returns 0 for freshly stored memories.
//
// The introspection consequence: the recall path only records
// `ConsolidationEvent::EdgeStrengthened` when that call returns `> 0`
// (src/memory/mod.rs:5210-5228). Under the shipped default it never does, so
// recall produces no association events, `formed_associations` and
// `strengthened_associations` stay empty, and — because these scenarios happen
// to produce no `MemoryStrengthened` events either — recall alone leaves the
// event buffer empty. This is the current, intentional contract; a
// remove-vs-revive decision on the memory-to-memory coactivation layer is
// PENDING, so these tests pin CURRENT behavior and pre-empt nothing.
//
// The both-modes coactivation contract is pinned by the unit tests in
// src/graph_memory.rs (`coactivation_strengthen_only_creates_no_new_edges`,
// `coactivation_strengthen_only_still_strengthens_existing`,
// `coactivation_strengthen_only_actually_increments_activation_and_strength`,
// plus the three `coactivation_*_survives_graph_reopen` durability tests).
// Those reach the mint path by PARAMETER (`record_memory_coactivation_impl`);
// integration tests only have the `SHODH_COACT_STRENGTHEN_ONLY` env var, and
// `std::env::set_var` is process-global — it would corrupt sibling tests
// running concurrently in this binary, so it is deliberately not used here.
//
// Tests below whose real subject is the EVENT BUFFER API rather than
// coactivation now source their events from `run_maintenance`, which emits
// `ConsolidationEvent::MaintenanceCycleCompleted` into the same buffer
// (src/memory/mod.rs:9132 -> `record_consolidation_event` -> the buffer read by
// `get_all_consolidation_events`). That keeps their assertions real instead of
// degenerating into "the buffer is empty", which would pass forever.

use chrono::{Duration, TimeZone, Utc};
use shodh_memory::memory::{ConsolidationEvent, Query};

/// Helper to get a "beginning of time" DateTime for all-time queries
fn epoch() -> chrono::DateTime<chrono::Utc> {
    Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap()
}

#[test]
fn test_consolidation_report_empty_system() {
    let (system, _temp) = setup_memory_system();

    // Empty system should produce valid report with zero counts
    let report = system.get_consolidation_report(epoch(), None);

    assert!(report.strengthened_memories.is_empty());
    assert!(report.decayed_memories.is_empty());
    assert!(report.formed_associations.is_empty());
    assert!(report.pruned_associations.is_empty());
    assert_eq!(report.statistics.maintenance_cycles, 0);
}

#[test]
fn test_consolidation_report_after_retrieval() {
    let (system, _temp) = setup_memory_system();

    // Record some memories
    let _id1 = system
        .remember(create_experience("Paris is the capital of France"), None)
        .unwrap();
    let _id2 = system
        .remember(create_experience("The Eiffel Tower is in Paris"), None)
        .unwrap();

    // Retrieve memories multiple times to trigger strengthening
    // (importance boosts after 5+ accesses)
    for _ in 0..7 {
        let query = Query {
            query_text: Some("Paris".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report (epoch = all time)
    let report = system.get_consolidation_report(epoch(), None);

    // Association events: none. Both of these memories are fresh, so there is
    // no pre-existing memory-to-memory edge for the recall-path coactivation to
    // strengthen, and minting is off by default (see the module note above).
    assert!(
        report.formed_associations.is_empty(),
        "strengthen-only default (f6b730ee, 2026-07-10) forms no associations \
         on recall: {} formed",
        report.formed_associations.len()
    );
    assert!(
        report.strengthened_associations.is_empty(),
        "strengthen-only default strengthens no memory-to-memory associations \
         for pairs with no prior edge: {} strengthened",
        report.strengthened_associations.len()
    );

    // `strengthened_memories` is deliberately NOT pinned to a count here: it is
    // driven by `update_access_count_instrumented`, which only emits when
    // importance actually moves (multiplicative `boost_importance`, gated on
    // access_count > 5) — an independent axis from the coactivation gate. What
    // IS pinned is that the report stays internally consistent whichever way
    // that goes: the statistics counters must match the vectors they summarize.
    assert_eq!(
        report.statistics.memories_strengthened,
        report.strengthened_memories.len(),
        "memories_strengthened stat must match its vector"
    );
    assert_eq!(
        report.statistics.edges_formed,
        report.formed_associations.len(),
        "edges_formed stat must match its vector"
    );
    assert_eq!(
        report.statistics.edges_strengthened,
        report.strengthened_associations.len(),
        "edges_strengthened stat must match its vector"
    );
}

#[test]
fn test_consolidation_report_after_maintenance() {
    let (system, _temp) = setup_memory_system();

    // Record memories
    for i in 0..10 {
        system
            .remember(create_experience(&format!("Test memory {}", i)), None)
            .unwrap();
    }

    // Run maintenance to potentially trigger decay events (0.95 = standard decay factor)
    system.run_maintenance(0.95, "test-user", false).unwrap();

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Maintenance should have been recorded
    assert!(
        report.statistics.maintenance_cycles >= 1,
        "Expected at least 1 maintenance cycle"
    );
}

#[test]
fn test_consolidation_report_hebbian_learning() {
    let (system, _temp) = setup_memory_system();

    // Record memories with related content
    let _id1 = system
        .remember(
            create_experience("Rust is a systems programming language"),
            None,
        )
        .unwrap();
    let _id2 = system
        .remember(create_experience("Rust has ownership and borrowing"), None)
        .unwrap();
    let _id3 = system
        .remember(create_experience("Rust prevents memory leaks"), None)
        .unwrap();

    // Retrieve related memories together multiple times
    // This should trigger Hebbian co-activation (fire together, wire together)
    for _ in 0..3 {
        let query = Query {
            query_text: Some("Rust programming".to_string()),
            max_results: 3,
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Before 2026-07-10, retrieving 2+ memories together minted an edge per
    // pair and this asserted the resulting formation events existed. Under the
    // shipped strengthen-only default (see the module note above) recall-time
    // Hebbian learning is inert for memories with no prior edge between them:
    // `record_memory_coactivation` returns 0, so src/memory/mod.rs:5210 never
    // reaches its `EdgeStrengthened` emit, and no association event of any kind
    // is recorded. Pinned positively so a future revive of the layer will trip
    // this test rather than silently passing.
    assert!(
        report.formed_associations.is_empty(),
        "shipped strengthen-only default must record zero edge-formation \
         events on recall: {}",
        report.formed_associations.len()
    );
    assert!(
        report.strengthened_associations.is_empty(),
        "shipped strengthen-only default must record zero edge-strengthening \
         events for co-retrieved memories with no prior edge: {}",
        report.strengthened_associations.len()
    );
    assert_eq!(
        report.statistics.edges_formed + report.statistics.edges_strengthened,
        0,
        "the association statistics counters must agree with the empty vectors"
    );
}

#[test]
fn test_consolidation_report_time_filtering() {
    let (system, _temp) = setup_memory_system();

    // Record and retrieve memories
    let _id1 = system
        .remember(create_experience("Time-filtered test memory"), None)
        .unwrap();

    for _ in 0..7 {
        let query = Query {
            query_text: Some("Time-filtered".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get reports for different time periods
    let all_time = system.get_consolidation_report(epoch(), None);
    let one_hour_ago = Utc::now() - Duration::hours(1);
    let last_hour = system.get_consolidation_report(one_hour_ago, None);

    // All events should be within the last hour for this test, so they should be equal
    // (Since we just created them, last_hour should have same events as all_time)
    let all_time_count = all_time.strengthened_memories.len()
        + all_time.formed_associations.len()
        + all_time.statistics.maintenance_cycles;
    let last_hour_count = last_hour.strengthened_memories.len()
        + last_hour.formed_associations.len()
        + last_hour.statistics.maintenance_cycles;

    assert!(
        all_time_count >= last_hour_count,
        "AllTime ({}) should have >= events than LastHour ({})",
        all_time_count,
        last_hour_count
    );
}

#[test]
fn test_consolidation_event_buffer_clear() {
    let (system, _temp) = setup_memory_system();

    // This test's subject is the buffer's clear() semantics, not coactivation.
    // It used to source its events from co-retrieval, which is inert under the
    // shipped strengthen-only default (see the module note) — leaving
    // `before=0, after=0` and making the clear assertion untestable. Events are
    // now sourced from `run_maintenance`, which emits
    // `MaintenanceCycleCompleted` into the same buffer, so "clear actually
    // clears a NON-EMPTY buffer" is genuinely exercised.
    system
        .remember(
            create_experience("Buffer clear test with important data"),
            None,
        )
        .unwrap();
    system
        .remember(
            create_experience("Buffer overflow prevention in systems"),
            None,
        )
        .unwrap();
    let query = Query {
        query_text: Some("Buffer".to_string()),
        ..Default::default()
    };
    for _ in 0..7 {
        let _ = system.recall(&query);
    }
    for _ in 0..3 {
        system.run_maintenance(0.95, "test-user", false).unwrap();
    }

    // Count events before clear
    let events_before = system.get_all_consolidation_events().len();
    assert!(
        events_before > 0,
        "the buffer must be non-empty before the clear, otherwise this test is \
         vacuous: {events_before}"
    );

    // Clear the buffer
    system.clear_consolidation_events();

    // Count events after clear
    let events_after = system.get_all_consolidation_events().len();

    assert!(
        events_after < events_before,
        "Events should be cleared: before={}, after={}",
        events_before,
        events_after
    );
    assert_eq!(events_after, 0, "Events should be completely cleared");
    assert_eq!(
        system.consolidation_event_count(),
        0,
        "the count accessor must agree with the emptied event list"
    );
}

#[test]
fn test_consolidation_report_stats_consistency() {
    let (system, _temp) = setup_memory_system();

    // Record and interact with memories
    for i in 0..5 {
        system
            .remember(
                create_experience(&format!("Stats consistency test {}", i)),
                None,
            )
            .unwrap();
    }

    // Multiple retrieval rounds
    for _ in 0..3 {
        let query = Query {
            query_text: Some("Stats consistency".to_string()),
            max_results: 5,
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Run maintenance with standard decay factor
    system.run_maintenance(0.95, "test-user", false).unwrap();

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Verify stats are internally consistent
    // statistics counters should match the vector lengths
    assert_eq!(
        report.statistics.memories_strengthened,
        report.strengthened_memories.len(),
        "memories_strengthened stat should match vector length"
    );
    assert_eq!(
        report.statistics.memories_decayed,
        report.decayed_memories.len(),
        "memories_decayed stat should match vector length"
    );
    assert_eq!(
        report.statistics.edges_formed,
        report.formed_associations.len(),
        "edges_formed stat should match vector length"
    );
    assert_eq!(
        report.statistics.edges_strengthened,
        report.strengthened_associations.len(),
        "edges_strengthened stat should match vector length"
    );
}

#[test]
fn test_memory_strengthening_records_before_after() {
    let (system, _temp) = setup_memory_system();

    // Record a memory with moderate initial importance
    let _id = system
        .remember(
            Experience {
                content: "Strengthening before/after test".to_string(),
                experience_type: ExperienceType::Observation,
                ..Default::default()
            },
            None,
        )
        .unwrap();

    // Access the memory many times to trigger importance boost
    for _ in 0..10 {
        let query = Query {
            query_text: Some("before/after test".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Verify strengthening events have valid before/after values
    for change in &report.strengthened_memories {
        assert!(
            change.activation_after >= change.activation_before,
            "activation_after ({}) should be >= activation_before ({})",
            change.activation_after,
            change.activation_before
        );
    }
}

#[test]
fn test_edge_events_have_strength_values() {
    let (system, _temp) = setup_memory_system();

    // Record related memories
    let _id1 = system
        .remember(create_experience("Edge test: topic A related"), None)
        .unwrap();
    let _id2 = system
        .remember(create_experience("Edge test: topic A connected"), None)
        .unwrap();

    // Retrieve together to form edges
    for _ in 0..5 {
        let query = Query {
            query_text: Some("Edge test: topic A".to_string()),
            max_results: 2,
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get report
    let report = system.get_consolidation_report(epoch(), None);

    // Verify edge events have valid strength values
    for assoc in &report.formed_associations {
        // strength_after contains the initial_strength for newly formed associations
        assert!(
            assoc.strength_after > 0.0 && assoc.strength_after <= 1.0,
            "strength_after (initial) should be in (0, 1]"
        );
    }

    for assoc in &report.strengthened_associations {
        // strength_before is Option<f32>, so we check if it exists
        if let Some(before) = assoc.strength_before {
            assert!(
                assoc.strength_after >= before,
                "strength_after ({}) should be >= strength_before ({}) for strengthening",
                assoc.strength_after,
                before
            );
        }
    }
}

#[test]
fn test_consolidation_events_list() {
    let (system, _temp) = setup_memory_system();

    // Record multiple memories (coactivation needs 2+)
    system
        .remember(
            create_experience("Test consolidation events list for tracking"),
            None,
        )
        .unwrap();
    system
        .remember(
            create_experience("Consolidation events are important for monitoring"),
            None,
        )
        .unwrap();

    for _ in 0..7 {
        let query = Query {
            query_text: Some("consolidation events".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }
    // Recall alone records nothing under the shipped strengthen-only default
    // (see the module note), so the event source for this buffer-API test is
    // maintenance, which emits `MaintenanceCycleCompleted` into the same buffer.
    system.run_maintenance(0.95, "test-user", false).unwrap();

    // Get all events directly
    let events = system.get_all_consolidation_events();

    // Should have some events recorded
    assert!(
        !events.is_empty(),
        "Expected some consolidation events to be recorded"
    );
    assert_eq!(
        events.len(),
        system.consolidation_event_count(),
        "get_all_consolidation_events() and consolidation_event_count() must agree"
    );

    // Verify each event has a valid timestamp
    for event in &events {
        let timestamp = event.timestamp();
        let now = Utc::now();
        // Event should have been created within the last minute
        assert!(
            timestamp <= now,
            "Event timestamp should not be in the future"
        );
    }
}

#[test]
fn test_consolidation_events_since_filter() {
    let (system, _temp) = setup_memory_system();

    // Record a memory and generate some events
    system
        .remember(create_experience("Test events since filter"), None)
        .unwrap();

    let start_time = Utc::now();

    for _ in 0..5 {
        let query = Query {
            query_text: Some("events since".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }

    // Get events since the start time
    let recent_events = system.get_consolidation_events_since(start_time);

    // All returned events should be >= start_time
    for event in &recent_events {
        assert!(
            event.timestamp() >= start_time,
            "Event timestamp should be >= filter time"
        );
    }
}

#[test]
fn test_consolidation_event_count() {
    let (system, _temp) = setup_memory_system();

    // Initially should have zero events
    let initial_count = system.consolidation_event_count();
    assert_eq!(initial_count, 0, "Initial event count should be zero");

    // Record memories and do some retrievals (coactivation needs 2+)
    system
        .remember(create_experience("Test event count tracking system"), None)
        .unwrap();
    system
        .remember(
            create_experience("Event count should increase with operations"),
            None,
        )
        .unwrap();

    for _ in 0..7 {
        let query = Query {
            query_text: Some("event count".to_string()),
            ..Default::default()
        };
        let _ = system.recall(&query);
    }
    // Recall alone records nothing under the shipped strengthen-only default
    // (see the module note); maintenance is the event source that makes the
    // "count increases" contract testable. Two cycles, so the counter is shown
    // to accumulate rather than merely become non-zero once.
    system.run_maintenance(0.95, "test-user", false).unwrap();
    let after_one_cycle = system.consolidation_event_count();
    system.run_maintenance(0.95, "test-user", false).unwrap();

    // Should have more events now
    let final_count = system.consolidation_event_count();
    assert!(
        after_one_cycle > initial_count,
        "Event count should increase after operations: {initial_count} -> {after_one_cycle}"
    );
    assert!(
        final_count > after_one_cycle,
        "Event count should keep accumulating across cycles: \
         {after_one_cycle} -> {final_count}"
    );
}
