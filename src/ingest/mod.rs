//! The enrichment write path, shared by `POST /api/remember` and every connector.
//!
//! # Why this module exists
//!
//! `MemorySystem::remember_detailed` stores a memory. It does not extract
//! entities, it does not build a graph episode, it does not emit an SSE event
//! and it does not record a metric. All of that lived inside the
//! `POST /api/remember` handler, which meant anything that was not an HTTP
//! request could not have it: a caller reaching `remember_detailed` directly
//! writes a memory that is in RocksDB and BM25 but absent from the knowledge
//! graph, and the graph leg is a measurable share of recall.
//!
//! The alternative — having an in-process ingestion run call its own server
//! over HTTP — needs an API key in-process, pays a JSON round trip per item,
//! and makes ingestion depend on the listening socket. On a machine that is
//! deliberately offline that is the one dependency that is not allowed.
//!
//! So the enrichment core is a function, and the handler is one of its callers.
//!
//! # What is deliberately NOT here
//!
//! `SessionStore::add_event(SessionEvent::MemoryCreated)` stays in the HTTP
//! handler. A session is a human or agent work period; an ingestion run is not
//! one. Folding the session event in here would have every connector run
//! fabricate entries in a store that is in-process only and never hydrated
//! from disk. A connector's session-equivalent is its run record, which is
//! durable.
//!
//! Request-shape validation also stays with the caller. The HTTP handler
//! validates a request body; a connector validates a file. They do not share a
//! failure vocabulary, and a shared one would be a union of two unrelated
//! sets.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::errors::AppError;
use crate::handlers::router::AppState;
use crate::handlers::types::MemoryEvent;
use crate::memory::types::{MemoryOrigin, NerEntityRecord};
use crate::memory::{Experience, ExperienceType, MemoryId};
use crate::metrics;
use crate::validation;

/// One item to write, with everything the caller knows and nothing this module
/// derives.
///
/// The enrichment fields — `entities`, `tags`, `declared_entities`,
/// `ner_entities` and `toponyms` on [`Self::experience`] — are **overwritten**
/// by [`ingest_experience`]. Setting them on the way in has no effect; that is
/// what [`Self::tags`] is for.
pub struct IngestRequest {
    pub user_id: String,

    /// Server-observed write path, stamped by the caller from what the *server*
    /// knows — which endpoint ran, or which connector walked. Never read out of
    /// a request body.
    ///
    /// [`MemoryOrigin::Unknown`] is rejected. `Unknown` means "this record
    /// predates the field"; assigning it to a new write manufactures the one
    /// thing the field exists to prevent.
    pub origin: MemoryOrigin,

    /// Caller-asserted tags, before the NER and YAKE merge. Seeds the merged
    /// entity list and, separately, the `declared_entities` set that tells the
    /// graph "something asserted this names a thing" as distinct from "a
    /// keyphrase extractor scored it highly".
    pub tags: Vec<String>,

    /// Backdated write time. `None` means now.
    pub created_at: Option<DateTime<Utc>>,

    /// Caller-declared write identity, stored verbatim and never verified.
    pub agent_id: Option<String>,
    pub run_id: Option<String>,
    pub actor_id: Option<String>,

    /// Parent memory id or id-prefix for hierarchical organisation. Resolved
    /// in the background pass, exactly as the handler resolved it.
    pub parent_id: Option<String>,

    /// The experience to store. `content`, `experience_type`, `metadata`,
    /// `importance_override` and `context` are the caller's; the robotics and
    /// multimodal fields are the caller's too and are `Default` for every
    /// connector. The five enrichment fields listed on this struct's doc are
    /// not.
    pub experience: Experience,
}

impl IngestRequest {
    /// The minimum a connector needs: who it is for, what the text is, what
    /// kind of experience it is, and which write path produced it.
    pub fn new(
        user_id: impl Into<String>,
        content: impl Into<String>,
        experience_type: ExperienceType,
        origin: MemoryOrigin,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            origin,
            tags: Vec::new(),
            created_at: None,
            agent_id: None,
            run_id: None,
            actor_id: None,
            parent_id: None,
            experience: Experience {
                content: content.into(),
                experience_type,
                origin,
                ..Default::default()
            },
        }
    }

    /// Attach caller tags. They seed both the merged entity list and the
    /// declared-entity set.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Attach a metadata entry. Namespaced keys only, by convention
    /// (`shodh.source.*` for ingestion).
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.experience.metadata.insert(key.into(), value.into());
        self
    }
}

/// What the write produced, in the terms a caller has to act on.
pub struct IngestOutcome {
    /// On a content-hash dedup hit this is the **existing** memory's id. A
    /// merge never mints a new id and never changes `created_at`.
    pub memory_id: MemoryId,

    /// True when the content already existed and enrichment was merged into
    /// the stored memory rather than a new memory being created.
    pub deduped: bool,

    /// The merged entity set actually stored — caller tags, NER surfaces and
    /// YAKE keyphrases, capped at [`validation::MAX_ENTITIES_PER_MEMORY`], and
    /// on a dedup merge the *stored* memory's set rather than this request's.
    pub entities: Vec<String>,

    /// `Debug` rendering of the experience type, which is the string the SSE
    /// event and the session event both carry.
    pub experience_type_label: String,
}

/// Run the enrichment write path for one experience.
///
/// In order: NER and YAKE in parallel, entity merge and cap, toponym
/// resolution, `remember_detailed` (which is where the content-hash dedup
/// lives), metrics, the SSE `CREATE` event, and a background pass on
/// `state.task_tracker` doing entity-name embeddings, the graph episode,
/// parent resolution, temporal facts and causal lineage.
///
/// The background pass is fire-and-forget by design: every task in it logs and
/// continues on failure, none of their results reach the caller, and running
/// them inline is what caused the 31% duplication rate in issue #109 (a handler
/// slower than the MCP client's 10 s timeout is a handler that gets retried).
/// `task_tracker` rather than a bare `tokio::spawn` so shutdown can await the
/// in-flight graph writes.
pub async fn ingest_experience(
    state: &AppState,
    req: IngestRequest,
) -> Result<IngestOutcome, AppError> {
    let op_start = std::time::Instant::now();

    if req.origin == MemoryOrigin::Unknown {
        // Not an InvalidInput: no request field carries this, so no caller can
        // fix it by changing what they sent. It is a programming error in a
        // write path that forgot to say what it was.
        return Err(AppError::Internal(anyhow::anyhow!(
            "ingest_experience called with MemoryOrigin::Unknown; every write path must \
             declare itself"
        )));
    }

    let IngestRequest {
        user_id,
        origin,
        tags,
        created_at,
        agent_id,
        run_id,
        actor_id,
        parent_id,
        mut experience,
    } = req;

    let content = experience.content.clone();
    let experience_type_label = format!("{:?}", experience.experience_type);

    // PERF: NER and YAKE are both CPU-bound and independent. Running them
    // concurrently on the blocking pool is worth ~40% of this call's latency.
    let ner = state.get_neural_ner();
    let yake = state.get_keyword_extractor();
    let content_for_ner = content.clone();
    let content_for_yake = content.clone();

    let (ner_result, yake_result) = tokio::join!(
        tokio::task::spawn_blocking(move || {
            match ner.extract(&content_for_ner) {
                Ok(entities) => entities
                    .into_iter()
                    .map(|e| NerEntityRecord {
                        text: e.text,
                        entity_type: e.entity_type.as_str().to_string(),
                        confidence: e.confidence,
                        start_char: Some(e.start),
                        end_char: Some(e.end),
                        fine_label: e.fine_label,
                    })
                    .collect::<Vec<NerEntityRecord>>(),
                // Reaching here means BOTH the neural typer and the rule-based
                // fallback failed — the memory is about to be stored with no
                // typed entities at all, which starves graph labelling and the
                // toponym gazetteer. That is not a debug-level event.
                Err(e) => {
                    tracing::warn!(
                        "NER extraction failed on ingest — storing memory with NO typed \
                         entities: {e}"
                    );
                    Vec::new()
                }
            }
        }),
        tokio::task::spawn_blocking(move || yake.extract_texts(&content_for_yake))
    );

    let ner_entities = match ner_result {
        Ok(entities) => entities,
        Err(e) => {
            if e.is_panic() {
                tracing::error!("NER extraction task panicked: {:?}", e);
            } else {
                tracing::debug!("NER extraction task cancelled: {:?}", e);
            }
            Vec::new()
        }
    };
    let extracted_keywords = match yake_result {
        Ok(keywords) => keywords,
        Err(e) => {
            if e.is_panic() {
                tracing::error!("YAKE extraction task panicked: {:?}", e);
            } else {
                tracing::debug!("YAKE extraction task cancelled: {:?}", e);
            }
            Vec::new()
        }
    };

    let merged_entities = merge_entities(&tags, &ner_entities, extracted_keywords);

    // Resolve place mentions to coordinates. Deliberately NOT written to
    // geo_location: that field means "recorded here" and feeds the geohash
    // radius index, while these are places the content merely talks about.
    let toponyms = crate::gazetteer::resolve_ner_locations(&ner_entities);
    let declared_entities = crate::handlers::remember::declared_entities_from(&tags);

    experience.entities = merged_entities.clone();
    experience.tags = merged_entities;
    experience.declared_entities = declared_entities;
    experience.ner_entities = ner_entities;
    experience.toponyms = toponyms;
    experience.origin = origin;

    let memory = state.get_user_memory(&user_id).map_err(AppError::Internal)?;

    // `_detailed` so a content-hash dedup hit reports whether it MERGED
    // enrichment into the stored memory. Without that signal the merged entity
    // set would reach RocksDB and BM25 but never the graph, because the
    // episode already exists and the graph pass is idempotent.
    let outcome = {
        let memory = memory.clone();
        let exp_clone = experience.clone();
        let agent_id = agent_id.clone();
        let run_id = run_id.clone();
        let actor_id = actor_id.clone();

        tokio::task::spawn_blocking(move || {
            let memory_guard = memory.read();
            if agent_id.is_some() || run_id.is_some() || actor_id.is_some() {
                memory_guard.remember_with_agent_detailed(
                    exp_clone, created_at, agent_id, run_id, actor_id,
                )
            } else {
                memory_guard.remember_detailed(exp_clone, created_at)
            }
        })
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
        .map_err(AppError::Internal)?
    };

    let memory_id = outcome.id.clone();
    let deduped = outcome.deduped;

    // A merge that added entities makes the memory's stored experience RICHER
    // than the one this request carried, so every downstream pass (graph, NER
    // embeddings, temporal facts) must run on the merged copy, not the request.
    let needs_graph_rebuild = outcome.needs_graph_rebuild();
    let experience = outcome.merged_experience.clone().unwrap_or(experience);
    let stored_entities = experience.entities.clone();

    let duration = op_start.elapsed().as_secs_f64();
    metrics::MEMORY_STORE_DURATION.observe(duration);
    metrics::MEMORY_STORE_TOTAL
        .with_label_values(&["success"])
        .inc();

    state.emit_event(MemoryEvent {
        event_type: "CREATE".to_string(),
        timestamp: Utc::now(),
        user_id: user_id.clone(),
        memory_id: Some(memory_id.0.to_string()),
        content_preview: Some(content.chars().take(500).collect()),
        memory_type: Some(experience_type_label.clone()),
        importance: None,
        count: None,
        entities: if tags.is_empty() {
            None
        } else {
            Some(tags.clone())
        },
        results: None,
    });

    spawn_post_write_pass(
        state.clone(),
        memory,
        user_id,
        content,
        experience,
        memory_id.clone(),
        needs_graph_rebuild,
        parent_id,
        created_at,
    );

    Ok(IngestOutcome {
        memory_id,
        deduped,
        entities: stored_entities,
        experience_type_label,
    })
}

/// Caller tags, then NER surfaces, then YAKE keyphrases — de-duplicated
/// case-insensitively in that precedence order and capped.
///
/// Order is the contract: the caller's own words come first, so a cap that
/// truncates drops inferred terms before asserted ones.
fn merge_entities(
    tags: &[String],
    ner_entities: &[NerEntityRecord],
    extracted_keywords: Vec<String>,
) -> Vec<String> {
    let mut merged: Vec<String> = tags.to_vec();
    let mut seen: std::collections::HashSet<String> =
        merged.iter().map(|t| t.to_lowercase()).collect();

    for record in ner_entities {
        if seen.insert(record.text.to_lowercase()) {
            if validation::validate_entity(&record.text).is_ok() {
                merged.push(record.text.clone());
            } else {
                tracing::debug!(
                    entity = %record.text,
                    "Skipping invalid NER entity (too long or invalid chars)"
                );
            }
        }
    }
    for keyword in extracted_keywords {
        if seen.insert(keyword.to_lowercase()) {
            if validation::validate_entity(&keyword).is_ok() {
                merged.push(keyword);
            } else {
                tracing::debug!(
                    entity = %keyword,
                    "Skipping invalid YAKE keyword (too long or invalid chars)"
                );
            }
        }
    }
    if merged.len() > validation::MAX_ENTITIES_PER_MEMORY {
        tracing::debug!(
            count = merged.len(),
            max = validation::MAX_ENTITIES_PER_MEMORY,
            "Capping entities to maximum allowed"
        );
        merged.truncate(validation::MAX_ENTITIES_PER_MEMORY);
    }
    merged
}

/// The four background tasks that follow every write: entity-name embeddings
/// feeding the graph episode, parent resolution, temporal fact extraction and
/// causal lineage inference.
///
/// Every one of them logs and continues on failure — none is allowed to make a
/// stored memory look unstored.
#[allow(clippy::too_many_arguments)]
fn spawn_post_write_pass(
    state: AppState,
    memory: std::sync::Arc<parking_lot::RwLock<crate::memory::MemorySystem>>,
    user_id: String,
    content: String,
    experience: Experience,
    memory_id: MemoryId,
    needs_graph_rebuild: bool,
    parent_id: Option<String>,
    created_at: Option<DateTime<Utc>>,
) {
    let tracker = state.task_tracker.clone();
    tracker.spawn(async move {
        // Pre-compute entity name embeddings for Tier 4 concept merge
        let entity_embeddings = {
            let mem = memory.clone();
            let names: Vec<String> = experience
                .ner_entities
                .iter()
                .map(|e| e.text.clone())
                .chain(experience.tags.iter().cloned())
                .collect();
            if names.is_empty() {
                None
            } else {
                match tokio::task::spawn_blocking(move || {
                    let guard = mem.read();
                    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
                    guard.get_embedder().encode_batch(&refs).map(|vecs| {
                        names
                            .into_iter()
                            .zip(vecs)
                            .collect::<HashMap<String, Vec<f32>>>()
                    })
                })
                .await
                {
                    Ok(Ok(map)) => Some(map),
                    Ok(Err(e)) => {
                        tracing::debug!("Entity name embedding failed (non-fatal): {}", e);
                        None
                    }
                    Err(e) => {
                        tracing::debug!("Entity name embedding task panicked: {}", e);
                        None
                    }
                }
            }
        };

        // Task 1: Build episodic graph (entities + episode + relationships).
        // On success the episode's surprise components come back — emit them
        // on the SSE stream so live consumers (dashboard anomaly feed) see
        // each episode's statistical shape as it lands.
        let graph_pass = if needs_graph_rebuild {
            // The episode exists and was built from the pre-merge entity set,
            // so the idempotency guard would skip it. Demolish and rebuild
            // from the MERGED experience.
            state.rebuild_experience_graph(
                &user_id,
                &experience,
                &memory_id,
                entity_embeddings.as_ref(),
            )
        } else {
            state.process_experience_into_graph(
                &user_id,
                &experience,
                &memory_id,
                entity_embeddings.as_ref(),
            )
        };

        match graph_pass {
            Ok(Some(surprise)) => {
                state.emit_event(MemoryEvent {
                    event_type: "surprise".to_string(),
                    timestamp: Utc::now(),
                    user_id: user_id.clone(),
                    memory_id: Some(memory_id.0.to_string()),
                    content_preview: Some(experience.content.chars().take(100).collect()),
                    memory_type: None,
                    importance: None,
                    count: None,
                    entities: None,
                    results: serde_json::to_value(&surprise).ok(),
                });
            }
            Ok(None) => {}
            Err(e) => tracing::debug!("Graph processing failed (non-fatal): {}", e),
        }

        // Task 2: Set parent_id for hierarchical organization
        if let Some(ref parent_id_str) = parent_id {
            let resolved_parent = if let Ok(parent_uuid) = uuid::Uuid::parse_str(parent_id_str) {
                Some(MemoryId(parent_uuid))
            } else {
                let mem = memory.clone();
                let prefix = parent_id_str.clone();
                match tokio::task::spawn_blocking(move || {
                    let guard = mem.read();
                    guard
                        .find_memory_by_prefix(&prefix)
                        .ok()
                        .flatten()
                        .map(|m| m.id.clone())
                })
                .await
                {
                    Ok(result) => result,
                    Err(e) => {
                        tracing::warn!("Parent resolve panicked (non-fatal): {e}");
                        None
                    }
                }
            };

            if let Some(resolved) = resolved_parent {
                let mem = memory.clone();
                let mid = memory_id.clone();
                if let Err(e) = tokio::task::spawn_blocking(move || {
                    let guard = mem.read();
                    guard.set_memory_parent(&mid, Some(resolved))
                })
                .await
                {
                    tracing::warn!("Parent set task panicked (non-fatal): {e}");
                }
            } else {
                tracing::warn!("Could not resolve parent_id: {}", parent_id_str);
            }
        }

        // Task 3: Extract and store temporal facts
        {
            let mem = memory.clone();
            let uid = user_id.clone();
            let cnt = content.clone();
            let ents = experience.entities.clone();
            let ts = created_at.unwrap_or_else(Utc::now);
            let mid = memory_id.clone();

            if let Err(e) = tokio::task::spawn_blocking(move || {
                let guard = mem.read();
                guard.store_temporal_facts_for_memory(&uid, &mid, &cnt, &ents, ts)
            })
            .await
            {
                tracing::warn!("Temporal fact extraction panicked (non-fatal): {e}");
            }
        }

        // Task 4: Infer causal lineage (runs after graph processing)
        crate::handlers::remember::spawn_lineage_inference(state, user_id, memory_id);
    });
}
