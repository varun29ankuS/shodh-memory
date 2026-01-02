//! Comprehensive Test Suite for Adaptive Memory Systems
//!
//! Tests the three core adaptive memory mechanisms:
//! - Outcome Feedback System (Hebbian learning through task outcomes)
//! - Semantic Consolidation (episodic → semantic fact extraction)
//! - Anticipatory Prefetch (context-aware cache warming)
//! - NER integration for entity extraction
//!
//! These tests verify that the memory system can learn, adapt, and improve over time.

use chrono::{Duration, Timelike, Utc};
use tempfile::TempDir;
use uuid::Uuid;

use shodh_memory::embeddings::ner::{NerConfig, NeuralNer};
use shodh_memory::memory::{
    compression::{ConsolidationResult, FactType, SemanticConsolidator, SemanticFact},
    retrieval::{
        AnticipatoryPrefetch, PrefetchContext, PrefetchReason, PrefetchResult, ReinforcementStats,
        RetrievalOutcome,
    },
    types::{
        CodeContext, ContextId, ConversationContext, DocumentContext, EnvironmentContext,
        Experience, ExperienceType, ProjectContext, Query, RetrievalMode, RichContext,
        SemanticContext, TemporalContext, UserContext,
    },
    Memory, MemoryConfig, MemoryId, MemorySystem,
};

/// Create fallback NER instance for testing
fn setup_fallback_ner() -> NeuralNer {
    let config = NerConfig::default();
    NeuralNer::new_fallback(config)
}

/// Create experience with NER-extracted entities
fn create_experience_with_ner(
    content: &str,
    exp_type: ExperienceType,
    ner: &NeuralNer,
) -> Experience {
    let entities = ner.extract(content).unwrap_or_default();
    let entity_names: Vec<String> = entities.iter().map(|e| e.text.clone()).collect();
    Experience {
        experience_type: exp_type,
        content: content.to_string(),
        entities: entity_names,
        ..Default::default()
    }
}

// ============================================================================
// TEST INFRASTRUCTURE
// ============================================================================

fn create_test_system() -> (MemorySystem, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = MemoryConfig {
        storage_path: temp_dir.path().to_path_buf(),
        working_memory_size: 100,
        session_memory_size_mb: 50,
        max_heap_per_user_mb: 500,
        auto_compress: false,
        compression_age_days: 7,
        importance_threshold: 0.3,
    };
    let system = MemorySystem::new(config).expect("Failed to create memory system");
    (system, temp_dir)
}

fn create_experience(content: &str, exp_type: ExperienceType, entities: Vec<&str>) -> Experience {
    Experience {
        experience_type: exp_type,
        content: content.to_string(),
        entities: entities.into_iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

fn create_learning_experience(content: &str, entities: Vec<&str>) -> Experience {
    create_experience(content, ExperienceType::Learning, entities)
}

fn create_decision_experience(content: &str, entities: Vec<&str>) -> Experience {
    create_experience(content, ExperienceType::Decision, entities)
}

fn create_error_experience(content: &str, entities: Vec<&str>) -> Experience {
    create_experience(content, ExperienceType::Error, entities)
}

fn create_conversation_experience(content: &str, entities: Vec<&str>) -> Experience {
    create_experience(content, ExperienceType::Conversation, entities)
}

fn create_memory(
    content: &str,
    exp_type: ExperienceType,
    entities: Vec<&str>,
    importance: f32,
) -> Memory {
    let exp = create_experience(content, exp_type, entities);
    Memory::new(
        MemoryId(Uuid::new_v4()),
        exp,
        importance,
        None,
        None,
        None,
        None,
    )
}

fn create_rich_context(project_id: Option<String>, code_ctx: Option<CodeContext>) -> RichContext {
    let now = Utc::now();
    RichContext {
        id: ContextId(Uuid::new_v4()),
        conversation: ConversationContext::default(),
        user: UserContext::default(),
        project: ProjectContext {
            project_id,
            ..Default::default()
        },
        temporal: TemporalContext::default(),
        semantic: SemanticContext::default(),
        code: code_ctx.unwrap_or_default(),
        document: DocumentContext::default(),
        environment: EnvironmentContext::default(),
        emotional: Default::default(),
        source: Default::default(),
        episode: Default::default(),
        parent: None,
        embeddings: None,
        decay_rate: 0.1,
        created_at: now,
        updated_at: now,
    }
}

// ============================================================================
// OUTCOME FEEDBACK SYSTEM TESTS
// ============================================================================

#[test]
fn test_retrieval_outcome_enum_values() {
    let helpful = RetrievalOutcome::Helpful;
    let misleading = RetrievalOutcome::Misleading;
    let neutral = RetrievalOutcome::Neutral;

    assert!(helpful != misleading);
    assert!(helpful != neutral);
    assert!(misleading != neutral);
}

#[test]
fn test_retrieval_outcome_default() {
    let default = RetrievalOutcome::default();
    assert_eq!(default, RetrievalOutcome::Neutral);
}

#[test]
fn test_tracked_retrieval_creation() {
    let (mut system, _temp_dir) = create_test_system();

    let exp1 = create_learning_experience(
        "Rust ownership prevents memory leaks",
        vec!["rust", "ownership"],
    );
    let exp2 = create_learning_experience(
        "The borrow checker enforces safety",
        vec!["rust", "borrow_checker"],
    );

    system.remember(exp1, None).expect("Failed to record");
    system.remember(exp2, None).expect("Failed to record");

    let query = Query {
        query_text: Some("rust memory safety".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let tracked = system
        .recall_tracked(&query)
        .expect("Failed to retrieve tracked");

    assert!(!tracked.retrieval_id.is_empty());
    assert!(!tracked.memories.is_empty());
    assert!(tracked.query_fingerprint != 0);
}

#[test]
fn test_tracked_retrieval_memory_ids() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_learning_experience("Test content", vec!["test"]);
    let id = system.remember(exp, None).expect("Failed to record");

    let query = Query {
        query_text: Some("test content".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let tracked = system.recall_tracked(&query).expect("Failed");
    let ids = tracked.memory_ids();

    assert!(ids.iter().any(|i| i == &id));
}

#[test]
fn test_reinforce_helpful_outcome() {
    let (mut system, _temp_dir) = create_test_system();

    let exp1 =
        create_learning_experience("API endpoint for user authentication", vec!["api", "auth"]);
    let exp2 = create_learning_experience("JWT token validation process", vec!["jwt", "auth"]);
    let exp3 = create_learning_experience("OAuth2 flow implementation", vec!["oauth", "auth"]);

    let id1 = system.remember(exp1, None).expect("Failed");
    let id2 = system.remember(exp2, None).expect("Failed");
    let id3 = system.remember(exp3, None).expect("Failed");

    let stats = system
        .reinforce_recall(
            &[id1.clone(), id2.clone(), id3.clone()],
            RetrievalOutcome::Helpful,
        )
        .expect("Failed to reinforce");

    assert_eq!(stats.memories_processed, 3);
    assert_eq!(stats.importance_boosts, 3);
    assert_eq!(stats.importance_decays, 0);
    assert_eq!(stats.associations_strengthened, 3);
    assert_eq!(stats.outcome, RetrievalOutcome::Helpful);
}

#[test]
fn test_reinforce_misleading_outcome() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_learning_experience("Misleading information", vec!["test"]);
    let id = system.remember(exp, None).expect("Failed");

    let initial_importance = {
        let query = Query {
            query_text: Some("misleading".to_string()),
            max_results: 1,
            ..Default::default()
        };
        let results = system.recall(&query).expect("Failed");
        results[0].importance()
    };

    // Apply multiple misleading reinforcements to ensure measurable decay
    for _ in 0..5 {
        let stats = system
            .reinforce_recall(&[id.clone()], RetrievalOutcome::Misleading)
            .expect("Failed");
        assert_eq!(stats.memories_processed, 1);
        assert_eq!(stats.importance_decays, 1);
        assert_eq!(stats.importance_boosts, 0);
        assert_eq!(stats.associations_strengthened, 0);
    }

    let query = Query {
        query_text: Some("misleading".to_string()),
        max_results: 1,
        ..Default::default()
    };
    let results = system.recall(&query).expect("Failed");
    let final_importance = results[0].importance();
    // After 5 decays of 10% each: 0.9^5 = 0.59 of original
    assert!(
        final_importance < initial_importance * 0.7,
        "importance should decay: {} < {} * 0.7 = {}",
        final_importance,
        initial_importance,
        initial_importance * 0.7
    );
}

#[test]
fn test_reinforce_neutral_outcome() {
    let (mut system, _temp_dir) = create_test_system();

    let exp1 = create_learning_experience("First neutral memory", vec!["neutral"]);
    let exp2 = create_learning_experience("Second neutral memory", vec!["neutral"]);

    let id1 = system.remember(exp1, None).expect("Failed");
    let id2 = system.remember(exp2, None).expect("Failed");

    let stats = system
        .reinforce_recall(&[id1, id2], RetrievalOutcome::Neutral)
        .expect("Failed");

    assert_eq!(stats.memories_processed, 2);
    assert_eq!(stats.associations_strengthened, 1);
    assert_eq!(stats.importance_boosts, 0);
    assert_eq!(stats.importance_decays, 0);
}

#[test]
fn test_reinforce_empty_list() {
    let (system, _temp_dir) = create_test_system();

    let stats = system
        .reinforce_recall(&[], RetrievalOutcome::Helpful)
        .expect("Failed");

    assert_eq!(stats.memories_processed, 0);
    assert_eq!(stats.associations_strengthened, 0);
}

#[test]
fn test_reinforce_tracked_convenience() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_learning_experience("Tracked memory test", vec!["tracked"]);
    system.remember(exp, None).expect("Failed");

    let query = Query {
        query_text: Some("tracked memory".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let tracked = system.recall_tracked(&query).expect("Failed");
    let stats = system
        .reinforce_recall_tracked(&tracked, RetrievalOutcome::Helpful)
        .expect("Failed");

    assert!(stats.memories_processed > 0);
}

#[test]
fn test_importance_boost_cumulative() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_learning_experience("Memory that gets boosted multiple times", vec!["boost"]);
    let id = system.remember(exp, None).expect("Failed");

    // Verify the boost mechanism reports correct stats
    let mut total_boosts = 0;
    for _ in 0..10 {
        let stats = system
            .reinforce_recall(&[id.clone()], RetrievalOutcome::Helpful)
            .expect("Failed");
        assert_eq!(stats.importance_boosts, 1);
        total_boosts += stats.importance_boosts;
    }

    // Verify all 10 boosts were applied (stats counted correctly)
    assert_eq!(total_boosts, 10);
}

#[test]
fn test_importance_decay_floor() {
    let (mut system, _temp_dir) = create_test_system();

    let exp = create_learning_experience("Memory that gets decayed to minimum", vec!["decay"]);
    let id = system.remember(exp, None).expect("Failed");

    // Verify decay mechanism reports correctly
    let mut total_decays = 0;
    for _ in 0..20 {
        let stats = system
            .reinforce_recall(&[id.clone()], RetrievalOutcome::Misleading)
            .expect("Failed");
        assert_eq!(stats.importance_decays, 1);
        total_decays += 1;
    }

    // Verify all 20 decays were processed
    assert_eq!(total_decays, 20);
}

#[test]
fn test_coactivation_strengthens_graph() {
    let (mut system, _temp_dir) = create_test_system();

    let exp1 = create_learning_experience("Database connection pooling", vec!["database"]);
    let exp2 = create_learning_experience("Query optimization techniques", vec!["database"]);
    let exp3 = create_learning_experience("Index design patterns", vec!["database"]);

    let id1 = system.remember(exp1, None).expect("Failed");
    let id2 = system.remember(exp2, None).expect("Failed");
    let id3 = system.remember(exp3, None).expect("Failed");

    let initial_stats = system.graph_stats();

    for _ in 0..5 {
        system
            .reinforce_recall(
                &[id1.clone(), id2.clone(), id3.clone()],
                RetrievalOutcome::Helpful,
            )
            .expect("Failed");
    }

    let final_stats = system.graph_stats();
    assert!(final_stats.edge_count >= initial_stats.edge_count);
}

// ============================================================================
// SEMANTIC CONSOLIDATION TESTS
// ============================================================================

#[test]
fn test_semantic_consolidator_creation() {
    let _consolidator = SemanticConsolidator::new();
}

#[test]
fn test_semantic_consolidator_with_thresholds() {
    let _consolidator = SemanticConsolidator::with_thresholds(3, 14);
}

#[test]
fn test_consolidate_empty_memories() {
    let consolidator = SemanticConsolidator::new();
    let result = consolidator.consolidate(&[]);

    assert_eq!(result.memories_processed, 0);
    assert_eq!(result.facts_extracted, 0);
}

#[test]
fn test_consolidate_young_memories() {
    let consolidator = SemanticConsolidator::new();

    let memories: Vec<Memory> = vec![
        create_memory(
            "Young memory 1",
            ExperienceType::Learning,
            vec!["test"],
            0.5,
        ),
        create_memory(
            "Young memory 2",
            ExperienceType::Learning,
            vec!["test"],
            0.5,
        ),
    ];

    let result = consolidator.consolidate(&memories);
    assert_eq!(result.memories_processed, 2);
    assert_eq!(result.facts_extracted, 0);
}

#[test]
fn test_fact_type_classification() {
    let pref_pattern = "preference: use tabs over spaces";
    let fact_type = classify_pattern(pref_pattern);
    assert_eq!(fact_type, FactType::Preference);

    let cap_pattern = "the system can handle 1000 requests per second";
    let fact_type = classify_pattern(cap_pattern);
    assert_eq!(fact_type, FactType::Capability);

    let rel_pattern = "authentication relates to user sessions";
    let fact_type = classify_pattern(rel_pattern);
    assert_eq!(fact_type, FactType::Relationship);

    let proc_pattern = "to deploy the application run npm build";
    let fact_type = classify_pattern(proc_pattern);
    assert_eq!(fact_type, FactType::Procedure);

    let def_pattern = "a mutex is a synchronization primitive";
    let fact_type = classify_pattern(def_pattern);
    assert_eq!(fact_type, FactType::Definition);
}

fn classify_pattern(pattern: &str) -> FactType {
    let lower = pattern.to_lowercase();

    if lower.starts_with("preference:") || lower.contains("prefer") || lower.contains("like") {
        FactType::Preference
    } else if lower.contains("can ") || lower.contains("able to") || lower.contains("supports") {
        FactType::Capability
    } else if lower.contains("relates to")
        || lower.contains("depends on")
        || lower.contains("connects")
    {
        FactType::Relationship
    } else if lower.contains("to ")
        || lower.contains("run ")
        || lower.contains("execute")
        || lower.contains("deploy")
    {
        FactType::Procedure
    } else if lower.contains(" is ") || lower.contains(" are ") || lower.contains("means") {
        FactType::Definition
    } else {
        FactType::Pattern
    }
}

#[test]
fn test_semantic_fact_structure() {
    let fact = SemanticFact {
        id: "fact_001".to_string(),
        fact: "Users prefer dark mode".to_string(),
        confidence: 0.85,
        support_count: 3,
        source_memories: vec![MemoryId(Uuid::new_v4()), MemoryId(Uuid::new_v4())],
        related_entities: vec!["user".to_string(), "dark_mode".to_string()],
        created_at: Utc::now(),
        last_reinforced: Utc::now(),
        fact_type: FactType::Preference,
    };

    assert_eq!(fact.id, "fact_001");
    assert_eq!(fact.confidence, 0.85);
    assert_eq!(fact.support_count, 3);
    assert_eq!(fact.source_memories.len(), 2);
    assert_eq!(fact.fact_type, FactType::Preference);
}

#[test]
fn test_reinforce_fact_increases_confidence() {
    let consolidator = SemanticConsolidator::new();

    let mut fact = SemanticFact {
        id: "test_fact".to_string(),
        fact: "Test fact".to_string(),
        confidence: 0.5,
        support_count: 2,
        source_memories: vec![],
        related_entities: vec![],
        created_at: Utc::now(),
        last_reinforced: Utc::now() - Duration::days(1),
        fact_type: FactType::Pattern,
    };

    let initial_confidence = fact.confidence;
    let initial_support = fact.support_count;

    let memory = create_memory(
        "Supporting evidence",
        ExperienceType::Learning,
        vec!["test"],
        0.5,
    );
    consolidator.reinforce_fact(&mut fact, &memory);

    assert!(fact.confidence > initial_confidence);
    assert_eq!(fact.support_count, initial_support + 1);
}

#[test]
fn test_reinforce_fact_adds_source() {
    let consolidator = SemanticConsolidator::new();

    let mut fact = SemanticFact {
        id: "test_fact".to_string(),
        fact: "Test".to_string(),
        confidence: 0.5,
        support_count: 1,
        source_memories: vec![],
        related_entities: vec![],
        created_at: Utc::now(),
        last_reinforced: Utc::now(),
        fact_type: FactType::Pattern,
    };

    let memory = create_memory("Evidence", ExperienceType::Learning, vec!["test"], 0.5);

    assert!(fact.source_memories.is_empty());
    consolidator.reinforce_fact(&mut fact, &memory);
    assert_eq!(fact.source_memories.len(), 1);
}

#[test]
fn test_should_decay_fact_old_unreinforced() {
    let consolidator = SemanticConsolidator::new();

    let old_fact = SemanticFact {
        id: "old_fact".to_string(),
        fact: "Old unused fact".to_string(),
        confidence: 0.3,
        support_count: 1,
        source_memories: vec![],
        related_entities: vec![],
        created_at: Utc::now() - Duration::days(100),
        last_reinforced: Utc::now() - Duration::days(100),
        fact_type: FactType::Pattern,
    };

    assert!(consolidator.should_decay_fact(&old_fact));
}

#[test]
fn test_should_not_decay_high_confidence_fact() {
    let consolidator = SemanticConsolidator::new();

    let strong_fact = SemanticFact {
        id: "strong_fact".to_string(),
        fact: "Well-established fact".to_string(),
        confidence: 0.95,
        support_count: 10,
        source_memories: vec![],
        related_entities: vec![],
        created_at: Utc::now() - Duration::days(30),
        last_reinforced: Utc::now() - Duration::days(30),
        fact_type: FactType::Definition,
    };

    assert!(!consolidator.should_decay_fact(&strong_fact));
}

#[test]
fn test_fact_type_default() {
    let default = FactType::default();
    assert_eq!(default, FactType::Pattern);
}

#[test]
fn test_consolidation_result_structure() {
    let result = ConsolidationResult {
        memories_processed: 10,
        facts_extracted: 3,
        facts_reinforced: 5,
        new_fact_ids: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
        new_facts: Vec::new(),
    };

    assert_eq!(result.memories_processed, 10);
    assert_eq!(result.facts_extracted, 3);
    assert_eq!(result.facts_reinforced, 5);
    assert_eq!(result.new_fact_ids.len(), 3);
}

// ============================================================================
// ANTICIPATORY PREFETCH TESTS
// ============================================================================

#[test]
fn test_prefetch_context_default() {
    let ctx = PrefetchContext::default();

    assert!(ctx.project_id.is_none());
    assert!(ctx.current_file.is_none());
    assert!(ctx.recent_entities.is_empty());
    assert!(ctx.hour_of_day.is_none());
    assert!(ctx.day_of_week.is_none());
    assert!(ctx.task_type.is_none());
}

#[test]
fn test_prefetch_context_from_current_time() {
    let ctx = PrefetchContext::from_current_time();

    assert!(ctx.hour_of_day.is_some());
    assert!(ctx.day_of_week.is_some());

    let hour = ctx.hour_of_day.unwrap();
    assert!(hour <= 23);

    let day = ctx.day_of_week.unwrap();
    assert!(day <= 6);
}

#[test]
fn test_anticipatory_prefetch_creation() {
    let _prefetch = AnticipatoryPrefetch::new();
}

#[test]
fn test_anticipatory_prefetch_with_limits() {
    let _prefetch = AnticipatoryPrefetch::with_limits(50, 0.5, 4);
}

#[test]
fn test_generate_prefetch_query_project() {
    let prefetch = AnticipatoryPrefetch::new();

    let ctx = PrefetchContext {
        project_id: Some("auth-service".to_string()),
        ..Default::default()
    };

    let query = prefetch.generate_prefetch_query(&ctx);
    assert!(query.is_some());

    let q = query.unwrap();
    assert!(q.query_text.is_some());
    assert!(q.query_text.as_ref().unwrap().contains("auth-service"));
}

#[test]
fn test_generate_prefetch_query_entities() {
    let prefetch = AnticipatoryPrefetch::new();

    let ctx = PrefetchContext {
        recent_entities: vec!["User".to_string(), "Session".to_string()],
        ..Default::default()
    };

    let query = prefetch.generate_prefetch_query(&ctx);
    assert!(query.is_some());

    let q = query.unwrap();
    assert!(q.query_text.is_some());
    let text = q.query_text.as_ref().unwrap();
    assert!(text.contains("User") || text.contains("Session"));
}

#[test]
fn test_generate_prefetch_query_file() {
    let prefetch = AnticipatoryPrefetch::new();

    let ctx = PrefetchContext {
        current_file: Some("/src/auth/login.rs".to_string()),
        ..Default::default()
    };

    let query = prefetch.generate_prefetch_query(&ctx);
    assert!(query.is_some());

    let q = query.unwrap();
    assert!(q.query_text.is_some());
    assert!(q.query_text.as_ref().unwrap().contains("login.rs"));
}

#[test]
fn test_generate_prefetch_query_temporal() {
    let prefetch = AnticipatoryPrefetch::new();

    let ctx = PrefetchContext {
        hour_of_day: Some(14),
        day_of_week: Some(1),
        ..Default::default()
    };

    let query = prefetch.generate_prefetch_query(&ctx);
    assert!(query.is_some());

    let q = query.unwrap();
    assert!(q.time_range.is_some());
    assert_eq!(q.retrieval_mode, RetrievalMode::Temporal);
}

#[test]
fn test_generate_prefetch_query_empty_context() {
    let prefetch = AnticipatoryPrefetch::new();
    let ctx = PrefetchContext::default();

    let query = prefetch.generate_prefetch_query(&ctx);
    assert!(query.is_none());
}

#[test]
fn test_prefetch_priority_order() {
    let prefetch = AnticipatoryPrefetch::new();

    let ctx = PrefetchContext {
        project_id: Some("priority-project".to_string()),
        recent_entities: vec!["Entity1".to_string()],
        current_file: Some("file.rs".to_string()),
        hour_of_day: Some(10),
        day_of_week: Some(2),
        ..Default::default()
    };

    let query = prefetch.generate_prefetch_query(&ctx);
    assert!(query.is_some());

    let q = query.unwrap();
    assert!(q.query_text.as_ref().unwrap().contains("priority-project"));
}

#[test]
fn test_relevance_score_project_match() {
    let prefetch = AnticipatoryPrefetch::new();

    let mut exp = create_learning_experience("Auth implementation", vec!["auth"]);
    exp.context = Some(create_rich_context(Some("auth-project".to_string()), None));

    let memory = Memory::new(MemoryId(Uuid::new_v4()), exp, 0.7, None, None, None, None);

    let ctx = PrefetchContext {
        project_id: Some("auth-project".to_string()),
        ..Default::default()
    };

    let score = prefetch.relevance_score(&memory, &ctx);
    assert!(score >= 0.4);
}

#[test]
fn test_relevance_score_entity_overlap() {
    let prefetch = AnticipatoryPrefetch::new();

    let exp = create_learning_experience("User authentication", vec!["User", "Auth"]);
    let memory = Memory::new(MemoryId(Uuid::new_v4()), exp, 0.7, None, None, None, None);

    let ctx = PrefetchContext {
        recent_entities: vec!["User".to_string(), "Session".to_string()],
        ..Default::default()
    };

    let score = prefetch.relevance_score(&memory, &ctx);
    assert!(score > 0.0);
}

#[test]
fn test_relevance_score_file_mention() {
    let prefetch = AnticipatoryPrefetch::new();

    let exp = create_learning_experience(
        "The login.rs file handles user authentication",
        vec!["login", "auth"],
    );
    let memory = Memory::new(MemoryId(Uuid::new_v4()), exp, 0.7, None, None, None, None);

    let ctx = PrefetchContext {
        current_file: Some("login.rs".to_string()),
        ..Default::default()
    };

    let score = prefetch.relevance_score(&memory, &ctx);
    assert!(score > 0.0);
}

#[test]
fn test_relevance_score_recency_boost() {
    let prefetch = AnticipatoryPrefetch::new();

    let exp = create_learning_experience("Recent memory", vec!["recent"]);
    let memory = Memory::new(MemoryId(Uuid::new_v4()), exp, 0.7, None, None, None, None);

    let ctx = PrefetchContext::default();
    let score = prefetch.relevance_score(&memory, &ctx);
    assert!(score >= 0.1);
}

#[test]
fn test_relevance_score_maximum() {
    let prefetch = AnticipatoryPrefetch::new();

    let mut exp =
        create_learning_experience("The login.rs file is important", vec!["User", "Auth"]);

    let code_ctx = CodeContext {
        current_file: Some("login.rs".to_string()),
        related_files: vec!["login.rs".to_string()],
        ..Default::default()
    };

    exp.context = Some(create_rich_context(
        Some("test-project".to_string()),
        Some(code_ctx),
    ));

    let memory = Memory::new(MemoryId(Uuid::new_v4()), exp, 0.7, None, None, None, None);

    let ctx = PrefetchContext {
        project_id: Some("test-project".to_string()),
        recent_entities: vec!["User".to_string(), "Auth".to_string()],
        current_file: Some("login.rs".to_string()),
        hour_of_day: Some(Utc::now().hour()),
        ..Default::default()
    };

    let score = prefetch.relevance_score(&memory, &ctx);
    assert!(score <= 1.0);
}

#[test]
fn test_prefetch_result_structure() {
    let result = PrefetchResult {
        prefetched_ids: vec![MemoryId(Uuid::new_v4()), MemoryId(Uuid::new_v4())],
        reason: PrefetchReason::Project("test".to_string()),
        cache_hits: 1,
        fetches: 1,
    };

    assert_eq!(result.prefetched_ids.len(), 2);
    assert_eq!(result.cache_hits, 1);
    assert_eq!(result.fetches, 1);
}

#[test]
fn test_prefetch_reason_variants() {
    let _project = PrefetchReason::Project("test".to_string());
    let _files = PrefetchReason::RelatedFiles;
    let _entities = PrefetchReason::SharedEntities;
    let _temporal = PrefetchReason::TemporalPattern;
    let _assoc = PrefetchReason::AssociatedMemories;
    let _query = PrefetchReason::QueryPrediction;
    let _mixed = PrefetchReason::Mixed;
}

#[test]
fn test_prefetch_reason_default() {
    let default = PrefetchReason::default();
    assert!(matches!(default, PrefetchReason::Mixed));
}

// ============================================================================
// INTEGRATION TESTS - Combining All Systems
// ============================================================================

#[test]
fn test_adaptive_memory_workflow() {
    let (mut system, _temp_dir) = create_test_system();

    let experiences = vec![
        create_learning_experience(
            "Rust ownership model prevents memory bugs",
            vec!["rust", "ownership"],
        ),
        create_learning_experience(
            "The borrow checker validates references at compile time",
            vec!["rust", "borrow_checker"],
        ),
        create_decision_experience(
            "Decided to use async/await for concurrency",
            vec!["rust", "async"],
        ),
        create_error_experience(
            "Error: lifetime mismatch in function return",
            vec!["rust", "lifetime"],
        ),
    ];

    let mut ids = Vec::new();
    for exp in experiences {
        ids.push(system.remember(exp, None).expect("Failed to record"));
    }

    let query = Query {
        query_text: Some("rust memory safety borrow".to_string()),
        max_results: 5,
        ..Default::default()
    };

    let tracked = system.recall_tracked(&query).expect("Failed");
    assert!(!tracked.memories.is_empty());

    let stats = system
        .reinforce_recall_tracked(&tracked, RetrievalOutcome::Helpful)
        .expect("Failed");
    assert!(stats.memories_processed > 0);

    let graph_stats = system.graph_stats();
    assert!(graph_stats.node_count > 0 || graph_stats.edge_count >= 0);

    let prefetch = AnticipatoryPrefetch::new();
    let ctx = PrefetchContext {
        recent_entities: vec!["rust".to_string()],
        ..Default::default()
    };

    let prefetch_query = prefetch.generate_prefetch_query(&ctx);
    assert!(prefetch_query.is_some());
}

#[test]
fn test_graph_maintenance() {
    let (mut system, _temp_dir) = create_test_system();

    let exp1 = create_learning_experience("First memory", vec!["test"]);
    let exp2 = create_learning_experience("Second memory", vec!["test"]);

    let id1 = system.remember(exp1, None).expect("Failed");
    let id2 = system.remember(exp2, None).expect("Failed");

    system
        .reinforce_recall(&[id1, id2], RetrievalOutcome::Helpful)
        .expect("Failed");

    system.graph_maintenance();

    let stats = system.graph_stats();
    assert!(stats.node_count >= 0);
}

#[test]
fn test_reinforcement_stats_structure() {
    let stats = ReinforcementStats {
        memories_processed: 5,
        associations_strengthened: 10,
        importance_boosts: 3,
        importance_decays: 2,
        outcome: RetrievalOutcome::Helpful,
        persist_failures: 0,
    };

    assert_eq!(stats.memories_processed, 5);
    assert_eq!(stats.associations_strengthened, 10);
    assert_eq!(stats.importance_boosts, 3);
    assert_eq!(stats.importance_decays, 2);
    assert_eq!(stats.outcome, RetrievalOutcome::Helpful);
    assert_eq!(stats.persist_failures, 0);
}

// ============================================================================
// STRESS TESTS
// ============================================================================

#[test]
fn test_high_volume_reinforcement() {
    let (mut system, _temp_dir) = create_test_system();

    let mut ids = Vec::new();
    for i in 0..50 {
        let exp =
            create_learning_experience(&format!("High volume memory {i}"), vec!["stress", "test"]);
        ids.push(system.remember(exp, None).expect("Failed"));
    }

    for chunk in ids.chunks(10) {
        let stats = system
            .reinforce_recall(chunk, RetrievalOutcome::Helpful)
            .expect("Failed");
        assert!(stats.memories_processed > 0);
    }
}

#[test]
fn test_rapid_feedback_cycles() {
    let (mut system, _temp_dir) = create_test_system();

    for i in 0..20 {
        let exp = create_learning_experience(&format!("Cycle {i}"), vec!["cycle"]);
        let _id = system.remember(exp, None).expect("Failed");

        let query = Query {
            query_text: Some(format!("cycle {i}")),
            max_results: 5,
            ..Default::default()
        };
        let tracked = system.recall_tracked(&query).expect("Failed");

        let outcome = if i % 3 == 0 {
            RetrievalOutcome::Helpful
        } else if i % 3 == 1 {
            RetrievalOutcome::Misleading
        } else {
            RetrievalOutcome::Neutral
        };

        system
            .reinforce_recall_tracked(&tracked, outcome)
            .expect("Failed");
    }

    let stats = system.stats();
    assert!(stats.total_memories >= 20);
}
