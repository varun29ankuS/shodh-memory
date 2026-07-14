//! Named Entity Recognition with schema-driven fine typing.
//!
//! Production path: the GLiNER bi-edge span typer ([`GlinerTyper`]) — a
//! schema-driven bi-encoder that predicts 141 fine labels and rolls each up to
//! one of 18 coarse [`EntityLabel`] classes. Every recognized surface carries
//! the fine label of its **top-scoring** span, so downstream ingest can set
//! `EntityNode.fine_type` and pick a precise primary label instead of the old
//! "everything is one bucket" MISC funnel.
//!
//! Degradation: when the GLiNER model assets are absent, extraction falls back
//! to the rule-based [`EntityExtractor`] keyword matcher (logged once at init).
//! The fallback yields coarse 4-class types only (no fine label).
//!
//! The legacy bert-tiny 4-class BIO tagger (`extract_neural`) and its MISC→regex
//! typer (`classify_misc_entity`) have been removed — GLiNER is the sole neural
//! typer, and the schema rollup replaces the keyword heuristics.
//!
//! # Edge Device Optimizations
//! - GLiNER bi-edge fp32 ONNX (~150MB) shares the process ORT runtime with MiniLM.
//! - Lazy loading — the model loads on first inference.
//! - LRU cache keyed by text hash avoids re-processing identical inputs.

use anyhow::Result;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::embeddings::gliner::GlinerTyper;
use crate::graph_memory::EntityLabel;

/// Coarse entity types surfaced to downstream query analysis and filtering.
///
/// This is the stable 4-class view every existing consumer already reads. The
/// GLiNER production path rolls its richer 18-class [`EntityLabel`] down to one
/// of these via [`NerEntityType::from_coarse`]; the precise class survives on
/// [`NerEntity::fine_label`] and is re-expanded at graph-insertion time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NerEntityType {
    Person,
    Organization,
    Location,
    Misc,
}

impl NerEntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NerEntityType::Person => "PER",
            NerEntityType::Organization => "ORG",
            NerEntityType::Location => "LOC",
            NerEntityType::Misc => "MISC",
        }
    }

    /// Roll a coarse [`EntityLabel`] down to the stable 4-class view. Named-entity
    /// classes with a clear PER/ORG/LOC home map through; everything else (dates,
    /// money, works, cyber, abstract concepts, …) is MISC — the fine label on the
    /// [`NerEntity`] preserves the exact class for the graph.
    pub fn from_coarse(label: &EntityLabel) -> Self {
        match label {
            EntityLabel::Person | EntityLabel::Title | EntityLabel::Role => Self::Person,
            EntityLabel::Organization | EntityLabel::Team | EntityLabel::Norp => Self::Organization,
            EntityLabel::Location
            | EntityLabel::Gpe
            | EntityLabel::Facility
            | EntityLabel::Environment => Self::Location,
            _ => Self::Misc,
        }
    }
}

/// A recognized entity from NER.
#[derive(Debug, Clone)]
pub struct NerEntity {
    /// The entity text (e.g., "Microsoft", "New York").
    pub text: String,
    /// Coarse 4-class type (drives existing query analysis / filtering).
    pub entity_type: NerEntityType,
    /// Confidence score (0.0 - 1.0). GLiNER sigmoid probability, or fallback salience.
    pub confidence: f32,
    /// Start character offset in original text.
    pub start: usize,
    /// End character offset in original text.
    pub end: usize,
    /// GLiNER fine label (schema leaf, e.g. "cargo ship", "bridge") of the
    /// top-scoring span for this surface. `None` on the rule-based fallback path.
    pub fine_label: Option<String>,
}

/// Configuration for the NER stage.
///
/// The GLiNER production path is configured from the environment via
/// [`GlinerConfig::from_env`](crate::embeddings::gliner::GlinerConfig::from_env)
/// (`SHODH_GLINER_MODEL_PATH`, default `./models/gliner-bi-edge`); the fields
/// here carry the fallback confidence floor and are retained for config-API and
/// call-site stability.
#[derive(Debug, Clone)]
pub struct NerConfig {
    /// Legacy model-dir path (unused by the GLiNER path; kept for API stability).
    pub model_path: PathBuf,
    /// Legacy tokenizer path (unused by the GLiNER path; kept for API stability).
    pub tokenizer_path: PathBuf,
    /// Legacy max sequence length (unused by the GLiNER path; kept for stability).
    pub max_length: usize,
    /// Minimum confidence threshold for the rule-based fallback path.
    pub confidence_threshold: f32,
}

impl Default for NerConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl NerConfig {
    /// Create configuration from environment variables.
    pub fn from_env() -> Self {
        let base_path = std::env::var("SHODH_NER_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| super::downloader::get_ner_models_dir());

        let confidence_threshold = std::env::var("SHODH_NER_CONFIDENCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);

        Self {
            model_path: base_path.join("model.onnx"),
            tokenizer_path: base_path.join("tokenizer.json"),
            max_length: 128,
            confidence_threshold,
        }
    }
}

/// Cache size for NER results (number of unique texts).
const NER_CACHE_SIZE: u64 = 1000;

/// Neural NER model — GLiNER bi-edge production typer with a rule-based fallback.
pub struct NeuralNer {
    /// GLiNER bi-edge span typer (lazy-loaded on first inference).
    gliner: GlinerTyper,
    /// True when GLiNER assets are absent — extraction degrades to rule-based.
    use_fallback: bool,
    /// Lazy-loaded EntityExtractor for comprehensive rule-based fallback.
    entity_extractor: OnceLock<crate::graph_memory::EntityExtractor>,
    /// Minimum confidence for fallback-path entities.
    fallback_confidence_threshold: f32,
    /// LRU cache for extracted entities (keyed by text hash).
    entity_cache: moka::sync::Cache<u64, Vec<NerEntity>>,
}

// ── NER replay hook (offline-entity ablation; env-gated, default OFF) ────────
// When SHODH_NER_REPLAY points at a JSON map {text -> [entities]}, extract()/
// extract_batch() return those entities instead of running the model. Lets us
// ablate a different NER through the REAL pipeline without porting it to Rust.
// JSON: {"<text>": [{"text","type"("PER"|"ORG"|"LOC"|"MISC"),"start","end","conf"}]}.
// A miss falls through to the model.
#[derive(serde::Deserialize)]
struct ReplayEntity {
    text: String,
    #[serde(rename = "type")]
    ty: String,
    start: usize,
    end: usize,
    conf: f32,
}

static NER_REPLAY_MAP: std::sync::OnceLock<
    Option<std::collections::HashMap<String, Vec<NerEntity>>>,
> = std::sync::OnceLock::new();

fn ner_replay_map() -> Option<&'static std::collections::HashMap<String, Vec<NerEntity>>> {
    NER_REPLAY_MAP
        .get_or_init(|| {
            let path = std::env::var("SHODH_NER_REPLAY").ok()?;
            let data = std::fs::read_to_string(&path)
                .map_err(|e| tracing::error!("SHODH_NER_REPLAY read {path} failed: {e}"))
                .ok()?;
            let raw: std::collections::HashMap<String, Vec<ReplayEntity>> =
                serde_json::from_str(&data)
                    .map_err(|e| tracing::error!("SHODH_NER_REPLAY parse failed: {e}"))
                    .ok()?;
            let map: std::collections::HashMap<String, Vec<NerEntity>> = raw
                .into_iter()
                .map(|(k, ents)| {
                    let v = ents
                        .into_iter()
                        .map(|e| NerEntity {
                            text: e.text,
                            entity_type: match e.ty.as_str() {
                                "PER" => NerEntityType::Person,
                                "ORG" => NerEntityType::Organization,
                                "LOC" => NerEntityType::Location,
                                _ => NerEntityType::Misc,
                            },
                            confidence: e.conf,
                            start: e.start,
                            end: e.end,
                            fine_label: None,
                        })
                        .collect();
                    (k, v)
                })
                .collect();
            tracing::warn!(
                "SHODH_NER_REPLAY ACTIVE: {} texts loaded from {} (model NER bypassed on hits)",
                map.len(),
                path
            );
            Some(map)
        })
        .as_ref()
}

fn ner_replay_lookup(text: &str) -> Option<Vec<NerEntity>> {
    ner_replay_map()?.get(text).cloned()
}

fn build_entity_cache() -> moka::sync::Cache<u64, Vec<NerEntity>> {
    moka::sync::Cache::builder()
        .max_capacity(NER_CACHE_SIZE)
        .time_to_live(std::time::Duration::from_secs(3600)) // 1 hour TTL
        .build()
}

impl NeuralNer {
    /// Create a NER stage. Uses the GLiNER bi-edge production typer when its
    /// assets are present, otherwise degrades to the rule-based fallback (logged).
    /// Never fails — the `Result` is retained for call-site stability.
    pub fn new(config: NerConfig) -> Result<Self> {
        let gliner = GlinerTyper::from_env();
        let use_fallback = !gliner.is_available();
        if use_fallback {
            tracing::warn!(
                "GLiNER bi-edge assets not found — NER degrading to rule-based fallback"
            );
        } else {
            tracing::info!("Neural NER initialized (GLiNER bi-edge production typer)");
        }
        Ok(Self {
            gliner,
            use_fallback,
            entity_extractor: OnceLock::new(),
            fallback_confidence_threshold: config.confidence_threshold,
            entity_cache: build_entity_cache(),
        })
    }

    /// Create a NER stage forced into rule-based fallback mode (GLiNER bypassed).
    pub fn new_fallback(config: NerConfig) -> Self {
        Self {
            gliner: GlinerTyper::from_env(),
            use_fallback: true,
            entity_extractor: OnceLock::new(),
            fallback_confidence_threshold: config.confidence_threshold,
            entity_cache: build_entity_cache(),
        }
    }

    /// Check if using rule-based fallback mode (GLiNER assets absent or forced).
    pub fn is_fallback_mode(&self) -> bool {
        self.use_fallback
    }

    /// Compute cache key from text (FNV-1a-style hash for speed).
    fn cache_key(text: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Extract entities (with caching and the optional replay hook).
    pub fn extract(&self, text: &str) -> Result<Vec<NerEntity>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        // NER replay hook (SHODH_NER_REPLAY) — offline-entity ablation, default off.
        if let Some(ents) = ner_replay_lookup(text) {
            return Ok(ents);
        }

        // Check cache first.
        let cache_key = Self::cache_key(text);
        if let Some(cached) = self.entity_cache.get(&cache_key) {
            return Ok(cached);
        }

        let entities = if self.use_fallback {
            self.extract_fallback(text)?
        } else {
            self.extract_gliner(text)
        };

        self.entity_cache.insert(cache_key, entities.clone());
        Ok(entities)
    }

    /// Extract entities from multiple texts.
    ///
    /// GLiNER types one text per inference, so this is a cache-aware loop over
    /// [`extract`](Self::extract) — replay, empties, and caching are handled there.
    pub fn extract_batch(&self, texts: &[&str]) -> Result<Vec<Vec<NerEntity>>> {
        let mut results = Vec::with_capacity(texts.len());
        for &text in texts {
            results.push(self.extract(text)?);
        }
        Ok(results)
    }

    /// GLiNER production typing: fine-typed spans → coarse-view [`NerEntity`]s.
    ///
    /// GLiNER performs flat, non-overlapping span selection internally, so each
    /// surface already carries its single top-scoring fine label — that is the
    /// label that lands on the graph node.
    fn extract_gliner(&self, text: &str) -> Vec<NerEntity> {
        let spans = self.gliner.extract(text);
        let entities: Vec<NerEntity> = spans
            .into_iter()
            .filter_map(|span| {
                if span.text.trim().is_empty() {
                    return None;
                }
                Some(NerEntity {
                    entity_type: NerEntityType::from_coarse(&span.coarse),
                    confidence: span.score,
                    start: span.start,
                    end: span.end,
                    fine_label: Some(span.fine_label),
                    text: span.text,
                })
            })
            .collect();
        // GLiNER already yields non-overlapping spans; dedup guards identical surfaces.
        self.deduplicate_entities(entities)
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (u64, u64) {
        (self.entity_cache.entry_count(), NER_CACHE_SIZE)
    }

    /// Clear the entity cache.
    pub fn clear_cache(&self) {
        self.entity_cache.invalidate_all();
    }

    /// Deduplicate entities (prefer longer spans, drop overlaps).
    fn deduplicate_entities(&self, mut entities: Vec<NerEntity>) -> Vec<NerEntity> {
        if entities.len() <= 1 {
            return entities;
        }

        // Sort by start position, then by length (descending).
        entities.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
        });

        let mut result = Vec::new();
        let mut seen_spans: HashSet<(usize, usize)> = HashSet::new();

        for entity in entities {
            // Check if this span overlaps with any seen span.
            let overlaps = seen_spans
                .iter()
                .any(|&(s, e)| entity.start < e && entity.end > s);

            if !overlaps {
                seen_spans.insert((entity.start, entity.end));
                result.push(entity);
            }
        }

        result
    }

    /// Rule-based fallback extraction using the comprehensive [`EntityExtractor`].
    ///
    /// Used only when GLiNER assets are absent. Provides coarse 4-class types with
    /// no fine label (`fine_label = None`) — the graph then defaults the entity's
    /// primary label from the coarse type.
    fn extract_fallback(&self, text: &str) -> Result<Vec<NerEntity>> {
        use crate::graph_memory::{EntityExtractor, EntityLabel};

        // Lazy-load the EntityExtractor (1000+ lines of dictionaries, only init once).
        let extractor = self.entity_extractor.get_or_init(EntityExtractor::new);

        // Extract entities with salience information.
        let extracted = extractor.extract_with_salience(text);

        // Convert EntityLabel to the coarse NerEntityType view and build NerEntity structs.
        let entities: Vec<NerEntity> = extracted
            .into_iter()
            .map(|e| {
                let entity_type = match e.label {
                    EntityLabel::Person => NerEntityType::Person,
                    EntityLabel::Organization | EntityLabel::Team => NerEntityType::Organization,
                    EntityLabel::Location | EntityLabel::Environment => NerEntityType::Location,
                    _ => NerEntityType::Misc,
                };

                // Use salience as confidence (EntityExtractor returns 0.6-0.9).
                let confidence = (e.base_salience * 0.9).min(0.85);

                // Find position in original text (case-insensitive byte-offset search).
                // Use the original name's byte length for slicing into `text`, since
                // to_lowercase() can change byte lengths for non-ASCII characters.
                let name_len = e.name.len();
                let (start, end) = text
                    .char_indices()
                    .find(|&(i, _)| {
                        text[i..]
                            .get(..name_len)
                            .is_some_and(|slice| slice.eq_ignore_ascii_case(&e.name))
                    })
                    .map(|(pos, _)| (pos, pos + name_len))
                    // Not found in the source text (e.g. the extractor normalised the
                    // surface form): emit a zero-width span at offset 0 to signal
                    // "position unknown" rather than fabricating a (0, name_len) span.
                    .unwrap_or((0, 0));

                NerEntity {
                    text: e.name,
                    entity_type,
                    confidence,
                    start,
                    end,
                    fine_label: None,
                }
            })
            .filter(|e| e.confidence >= self.fallback_confidence_threshold)
            .collect();

        // Dedup — the heuristic extractor can emit the same surface more than once.
        let entities = self.deduplicate_entities(entities);

        Ok(entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== NerEntityType Tests ====================

    #[test]
    fn test_entity_type_as_str() {
        assert_eq!(NerEntityType::Person.as_str(), "PER");
        assert_eq!(NerEntityType::Organization.as_str(), "ORG");
        assert_eq!(NerEntityType::Location.as_str(), "LOC");
        assert_eq!(NerEntityType::Misc.as_str(), "MISC");
    }

    #[test]
    fn test_from_coarse_rolls_named_classes_to_four_class_view() {
        // Location family
        assert_eq!(
            NerEntityType::from_coarse(&EntityLabel::Gpe),
            NerEntityType::Location
        );
        assert_eq!(
            NerEntityType::from_coarse(&EntityLabel::Facility),
            NerEntityType::Location
        );
        // Organization family
        assert_eq!(
            NerEntityType::from_coarse(&EntityLabel::Norp),
            NerEntityType::Organization
        );
        // Person family
        assert_eq!(
            NerEntityType::from_coarse(&EntityLabel::Title),
            NerEntityType::Person
        );
        // Everything else → MISC (fine label preserves the precise class).
        assert_eq!(
            NerEntityType::from_coarse(&EntityLabel::Money),
            NerEntityType::Misc
        );
        assert_eq!(
            NerEntityType::from_coarse(&EntityLabel::Vehicle),
            NerEntityType::Misc
        );
    }

    // ==================== NerConfig Tests ====================

    #[test]
    fn test_ner_config_default() {
        let config = NerConfig::default();
        assert_eq!(config.max_length, 128);
    }

    // ==================== NeuralNer Fallback Tests ====================

    fn fallback_ner() -> NeuralNer {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };
        NeuralNer::new_fallback(config)
    }

    #[test]
    fn test_fallback_mode_detection() {
        assert!(fallback_ner().is_fallback_mode());
    }

    #[test]
    fn test_fallback_extraction_organizations() {
        let ner = fallback_ner();
        let test_cases = vec![
            ("Microsoft is a company", "Microsoft"),
            ("I work at Google", "Google"),
            ("Apple released a new product", "Apple"),
            ("Infosys reported earnings", "Infosys"),
        ];
        for (text, expected_entity) in test_cases {
            let entities = ner.extract(text).unwrap();
            let found = entities.iter().find(|e| e.text == expected_entity);
            assert!(found.is_some(), "Should find {expected_entity} in '{text}'");
            assert_eq!(
                found.unwrap().entity_type,
                NerEntityType::Organization,
                "Wrong type for {expected_entity} in '{text}'"
            );
            // Fallback path never sets a fine label.
            assert!(found.unwrap().fine_label.is_none());
        }
    }

    #[test]
    fn test_fallback_extraction_locations() {
        let ner = fallback_ner();
        let test_cases = vec![
            ("The office is in Seattle", "Seattle"),
            ("I visited Mumbai last week", "Mumbai"),
            ("Tokyo is beautiful", "Tokyo"),
            ("Moving to Bangalore", "Bangalore"),
        ];
        for (text, expected_entity) in test_cases {
            let entities = ner.extract(text).unwrap();
            let found = entities.iter().find(|e| e.text == expected_entity);
            assert!(found.is_some(), "Should find {expected_entity} in '{text}'");
            assert_eq!(
                found.unwrap().entity_type,
                NerEntityType::Location,
                "Wrong type for {expected_entity} in '{text}'"
            );
        }
    }

    #[test]
    fn test_fallback_extraction_mixed() {
        let ner = fallback_ner();
        let entities = ner
            .extract("Microsoft is headquartered in Seattle")
            .unwrap();
        let microsoft = entities.iter().find(|e| e.text == "Microsoft");
        let seattle = entities.iter().find(|e| e.text == "Seattle");
        assert!(microsoft.is_some());
        assert!(seattle.is_some());
        assert_eq!(microsoft.unwrap().entity_type, NerEntityType::Organization);
        assert_eq!(seattle.unwrap().entity_type, NerEntityType::Location);
    }

    #[test]
    fn test_fallback_extraction_empty_text() {
        let ner = fallback_ner();
        assert!(ner.extract("").unwrap().is_empty());
    }

    #[test]
    fn test_fallback_extraction_whitespace_only() {
        let ner = fallback_ner();
        assert!(ner.extract("   \t\n  ").unwrap().is_empty());
    }

    #[test]
    fn test_fallback_extraction_stop_words_only() {
        let ner = fallback_ner();
        let entities = ner.extract("the a an and or is are was were").unwrap();
        assert!(
            entities.is_empty(),
            "Expected no entities from stop words but got: {entities:?}"
        );
    }

    #[test]
    fn test_fallback_deduplication() {
        let ner = fallback_ner();
        let entities = ner
            .extract("Microsoft partnered with Microsoft Azure")
            .unwrap();
        let microsoft_count = entities.iter().filter(|e| e.text == "Microsoft").count();
        assert_eq!(microsoft_count, 1, "Microsoft should appear only once");
    }

    #[test]
    fn test_fallback_confidence_scores() {
        let ner = fallback_ner();
        let entities = ner.extract("Microsoft Google Apple").unwrap();
        for entity in &entities {
            assert!(
                entity.confidence >= 0.5 && entity.confidence <= 1.0,
                "Confidence {} out of expected range",
                entity.confidence
            );
        }
    }

    // ==================== NerEntity Tests ====================

    #[test]
    fn test_ner_entity_clone() {
        let entity = NerEntity {
            text: "Microsoft".to_string(),
            entity_type: NerEntityType::Organization,
            confidence: 0.95,
            start: 0,
            end: 9,
            fine_label: Some("company".to_string()),
        };
        let cloned = entity.clone();
        assert_eq!(cloned.text, entity.text);
        assert_eq!(cloned.entity_type, entity.entity_type);
        assert_eq!(cloned.fine_label, entity.fine_label);
        assert!((cloned.confidence - entity.confidence).abs() < 1e-5);
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_punctuation_handling() {
        let ner = fallback_ner();
        let entities = ner.extract("Microsoft, Google, and Apple!").unwrap();
        for entity in &entities {
            assert!(!entity.text.contains(','));
            assert!(!entity.text.contains('!'));
        }
    }

    #[test]
    fn test_indian_cities() {
        let ner = fallback_ner();
        let indian_cities = vec!["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"];
        for city in indian_cities {
            let entities = ner.extract(&format!("Office in {city}")).unwrap();
            let found = entities.iter().find(|e| e.text == city);
            assert!(found.is_some(), "Should find Indian city: {city}");
            assert_eq!(
                found.unwrap().entity_type,
                NerEntityType::Location,
                "{city} should be Location"
            );
        }
    }
}
