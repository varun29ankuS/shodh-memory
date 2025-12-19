//! Neural Named Entity Recognition using ONNX Runtime
//!
//! Implements lightweight NER optimized for edge devices:
//! - Model: bert-tiny-NER (ONNX exported, ~17MB)
//! - Labels: PER, ORG, LOC, MISC with BIO tagging
//! - Accuracy: ~85% F1 on CoNLL-2003
//! - Latency: ~10-15ms per inference
//!
//! This provides neural NER quality while staying lightweight enough
//! for edge deployment alongside MiniLM embeddings.
//!
//! # Architecture
//! - Input: Raw text
//! - Tokenization: WordPiece (BERT tokenizer)
//! - Model: TinyBERT for token classification (4.4M params)
//! - Output: BIO-tagged entities with confidence scores
//!
//! # Supported Entity Types
//! - PER: Person names (maps to EntityLabel::Person)
//! - ORG: Organizations (maps to EntityLabel::Organization)
//! - LOC: Locations (maps to EntityLabel::Location)
//! - MISC: Miscellaneous entities (maps to EntityLabel::Other)
//!
//! # Edge Device Optimizations
//! - Quantized INT8 model (~17MB vs 400MB for bert-base)
//! - Max sequence length: 128 (vs 512 for base)
//! - Shared ONNX runtime with embeddings model
//! - Lazy loading - only loads when first used

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tokenizers::Tokenizer;

/// BIO tag labels from TinyBERT-finetuned-NER-ONNX
/// Index mapping: O=0, B-MISC=1, I-MISC=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6, B-PER=7, I-PER=8
/// Note: This ordering differs from bert-base-NER (dslim) which uses MISC, PER, ORG, LOC
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NerTag {
    Outside,
    BeginMisc,
    InsideMisc,
    BeginOrg,
    InsideOrg,
    BeginLoc,
    InsideLoc,
    BeginPerson,
    InsidePerson,
}

impl NerTag {
    fn from_index(idx: usize) -> Self {
        match idx {
            0 => NerTag::Outside,
            1 => NerTag::BeginMisc,
            2 => NerTag::InsideMisc,
            3 => NerTag::BeginOrg,
            4 => NerTag::InsideOrg,
            5 => NerTag::BeginLoc,
            6 => NerTag::InsideLoc,
            7 => NerTag::BeginPerson,
            8 => NerTag::InsidePerson,
            _ => NerTag::Outside,
        }
    }

    fn is_begin(&self) -> bool {
        matches!(
            self,
            NerTag::BeginMisc | NerTag::BeginPerson | NerTag::BeginOrg | NerTag::BeginLoc
        )
    }

    fn is_inside(&self) -> bool {
        matches!(
            self,
            NerTag::InsideMisc | NerTag::InsidePerson | NerTag::InsideOrg | NerTag::InsideLoc
        )
    }

    fn entity_type(&self) -> Option<NerEntityType> {
        match self {
            NerTag::BeginPerson | NerTag::InsidePerson => Some(NerEntityType::Person),
            NerTag::BeginOrg | NerTag::InsideOrg => Some(NerEntityType::Organization),
            NerTag::BeginLoc | NerTag::InsideLoc => Some(NerEntityType::Location),
            NerTag::BeginMisc | NerTag::InsideMisc => Some(NerEntityType::Misc),
            NerTag::Outside => None,
        }
    }

    fn matches_type(&self, other: &NerTag) -> bool {
        self.entity_type() == other.entity_type()
    }
}

/// Entity types from NER model
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
}

/// A recognized entity from neural NER
#[derive(Debug, Clone)]
pub struct NerEntity {
    /// The entity text (e.g., "Microsoft", "New York")
    pub text: String,
    /// Entity type
    pub entity_type: NerEntityType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Start character offset in original text
    pub start: usize,
    /// End character offset in original text
    pub end: usize,
}

/// Configuration for NER model
#[derive(Debug, Clone)]
pub struct NerConfig {
    /// Path to ONNX model file
    pub model_path: PathBuf,
    /// Path to tokenizer file
    pub tokenizer_path: PathBuf,
    /// Maximum sequence length (BERT default: 512)
    pub max_length: usize,
    /// Minimum confidence threshold for entity extraction
    pub confidence_threshold: f32,
}

impl Default for NerConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl NerConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let base_path = std::env::var("SHODH_NER_MODEL_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                // Try common locations - bundled package dir has highest priority
                let candidates: Vec<Option<PathBuf>> = vec![
                    // Bundled in Python package (highest priority for pip install)
                    std::env::var("SHODH_PACKAGE_DIR")
                        .ok()
                        .map(|p| PathBuf::from(p).join("models/bert-tiny-ner")),
                    // Local development paths
                    Some(PathBuf::from("./models/bert-tiny-ner")),
                    Some(PathBuf::from("../models/bert-tiny-ner")),
                    // Downloaded models cache
                    Some(super::downloader::get_ner_models_dir()),
                    // System data directory
                    dirs::data_dir().map(|p| p.join("shodh-memory/models/bert-tiny-ner")),
                ];

                candidates
                    .into_iter()
                    .flatten()
                    .find(|p| p.join("model.onnx").exists())
                    .unwrap_or_else(super::downloader::get_ner_models_dir)
            });

        let confidence_threshold = std::env::var("SHODH_NER_CONFIDENCE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);

        Self {
            model_path: base_path.join("model.onnx"),
            tokenizer_path: base_path.join("tokenizer.json"),
            // bert-tiny uses shorter sequences for speed (128 vs 512)
            max_length: 128,
            confidence_threshold,
        }
    }
}

/// Lazily initialized NER model
struct LazyNerModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl LazyNerModel {
    fn new(config: &NerConfig) -> Result<Self> {
        let num_threads = std::env::var("SHODH_ONNX_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);

        tracing::info!(
            "Loading BERT-NER model from {:?} with {} threads",
            config.model_path,
            num_threads
        );

        let session = Session::builder()
            .context("Failed to create NER session builder")?
            .with_intra_threads(num_threads)
            .context("Failed to set NER thread count")?
            .commit_from_file(&config.model_path)
            .context("Failed to load NER ONNX model")?;

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load NER tokenizer: {e}"))?;

        tracing::info!("BERT-NER model loaded successfully");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }
}

/// Neural NER model using BERT + ONNX Runtime
pub struct NeuralNer {
    config: NerConfig,
    lazy_model: OnceLock<Result<Arc<LazyNerModel>, String>>,
    /// Fallback: rule-based extraction when model unavailable
    use_fallback: bool,
    /// Lazy-loaded EntityExtractor for comprehensive rule-based fallback
    entity_extractor: OnceLock<crate::graph_memory::EntityExtractor>,
}

impl NeuralNer {
    /// Create new NER model with lazy loading
    pub fn new(config: NerConfig) -> Result<Self> {
        let model_available = config.model_path.exists() && config.tokenizer_path.exists();

        if !model_available {
            tracing::warn!(
                "NER model not found at {:?}. Using rule-based fallback.",
                config.model_path
            );
            return Ok(Self {
                config,
                lazy_model: OnceLock::new(),
                use_fallback: true,
                entity_extractor: OnceLock::new(),
            });
        }

        Ok(Self {
            config,
            lazy_model: OnceLock::new(),
            use_fallback: false,
            entity_extractor: OnceLock::new(),
        })
    }

    /// Create NER model with explicit fallback mode
    pub fn new_fallback(config: NerConfig) -> Self {
        Self {
            config,
            lazy_model: OnceLock::new(),
            use_fallback: true,
            entity_extractor: OnceLock::new(),
        }
    }

    /// Ensure model is loaded
    fn ensure_model_loaded(&self) -> Result<&Arc<LazyNerModel>> {
        if self.use_fallback {
            anyhow::bail!("NER model in fallback mode");
        }

        let result = self.lazy_model.get_or_init(|| {
            LazyNerModel::new(&self.config)
                .map(Arc::new)
                .map_err(|e| e.to_string())
        });

        match result {
            Ok(model) => Ok(model),
            Err(e) => Err(anyhow::anyhow!("Failed to load NER model: {e}")),
        }
    }

    /// Check if using fallback mode
    pub fn is_fallback_mode(&self) -> bool {
        self.use_fallback
    }

    /// Extract entities using neural NER
    pub fn extract(&self, text: &str) -> Result<Vec<NerEntity>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        if self.use_fallback {
            return self.extract_fallback(text);
        }

        match self.extract_neural(text) {
            Ok(entities) => Ok(entities),
            Err(e) => {
                tracing::warn!("Neural NER failed: {}. Using fallback.", e);
                self.extract_fallback(text)
            }
        }
    }

    /// Neural extraction using ONNX model
    fn extract_neural(&self, text: &str) -> Result<Vec<NerEntity>> {
        let model = self.ensure_model_loaded()?;
        let mut session = model.session.lock();

        // Tokenize input
        let encoding = model
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("NER tokenization failed: {e}"))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let offsets = encoding.get_offsets();
        let max_length = self.config.max_length;

        // Prepare input tensors
        let mut input_ids = vec![0i64; max_length];
        let mut attention = vec![0i64; max_length];

        for (i, &token) in tokens.iter().take(max_length).enumerate() {
            input_ids[i] = token as i64;
        }
        for (i, &mask) in attention_mask.iter().take(max_length).enumerate() {
            attention[i] = mask as i64;
        }

        // Create ONNX input tensors
        // token_type_ids: all zeros for single sentence (BERT segment embedding)
        let token_type_ids = vec![0i64; max_length];

        let input_ids_value = Value::from_array((vec![1, max_length], input_ids))
            .context("Failed to create input_ids tensor")?;
        let attention_mask_value = Value::from_array((vec![1, max_length], attention.clone()))
            .context("Failed to create attention_mask tensor")?;
        let token_type_ids_value = Value::from_array((vec![1, max_length], token_type_ids))
            .context("Failed to create token_type_ids tensor")?;

        // Run inference
        let outputs = session
            .run(ort::inputs![
                "input_ids" => &input_ids_value,
                "attention_mask" => &attention_mask_value,
                "token_type_ids" => &token_type_ids_value,
            ])
            .context("NER inference failed")?;

        // Extract logits - shape: [1, seq_len, num_labels]
        let output_tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract NER output tensor")?;
        let (_shape, logits) = output_tensor;

        // Decode BIO tags to entities
        let num_labels = 9; // O, B-MISC, I-MISC, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
        let seq_len = tokens.len().min(max_length);

        let mut entities = Vec::new();
        let mut current_entity: Option<(NerTag, Vec<usize>, f32)> = None;

        for i in 0..seq_len {
            // Skip [CLS] and [SEP] tokens
            if i == 0 || attention[i] == 0 {
                continue;
            }

            // Get logits for this position
            let start_idx = i * num_labels;
            let token_logits = &logits[start_idx..start_idx + num_labels];

            // Softmax to get probabilities
            let probs = softmax(token_logits);

            // Find best label
            let (best_idx, best_prob) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            let tag = NerTag::from_index(best_idx);

            // Handle BIO tagging
            match (&current_entity, tag.is_begin(), tag.is_inside()) {
                // Begin new entity
                (None, true, _) => {
                    current_entity = Some((tag, vec![i], *best_prob));
                }
                // Continue current entity
                (Some((prev_tag, indices, acc_prob)), _, true) if tag.matches_type(prev_tag) => {
                    let mut new_indices = indices.clone();
                    new_indices.push(i);
                    current_entity = Some((*prev_tag, new_indices, acc_prob + best_prob));
                }
                // End current entity, possibly start new
                (Some((prev_tag, indices, acc_prob)), _, _) => {
                    // Save previous entity
                    if let Some(entity) =
                        self.build_entity(text, prev_tag, indices, *acc_prob, offsets)
                    {
                        if entity.confidence >= self.config.confidence_threshold {
                            entities.push(entity);
                        }
                    }

                    // Start new entity if this is a B- tag
                    if tag.is_begin() {
                        current_entity = Some((tag, vec![i], *best_prob));
                    } else {
                        current_entity = None;
                    }
                }
                _ => {}
            }
        }

        // Don't forget the last entity
        if let Some((tag, indices, acc_prob)) = current_entity {
            if let Some(entity) = self.build_entity(text, &tag, &indices, acc_prob, offsets) {
                if entity.confidence >= self.config.confidence_threshold {
                    entities.push(entity);
                }
            }
        }

        // Deduplicate and merge overlapping entities
        let entities = self.deduplicate_entities(entities);

        Ok(entities)
    }

    /// Build entity from token indices
    fn build_entity(
        &self,
        text: &str,
        tag: &NerTag,
        token_indices: &[usize],
        accumulated_prob: f32,
        offsets: &[(usize, usize)],
    ) -> Option<NerEntity> {
        if token_indices.is_empty() {
            return None;
        }

        let entity_type = tag.entity_type()?;

        // Get character offsets
        let first_idx = token_indices[0];
        let last_idx = token_indices[token_indices.len() - 1];

        if first_idx >= offsets.len() || last_idx >= offsets.len() {
            return None;
        }

        let start = offsets[first_idx].0;
        let end = offsets[last_idx].1;

        if start >= end || end > text.len() {
            return None;
        }

        let entity_text = text[start..end].trim().to_string();
        if entity_text.is_empty() {
            return None;
        }

        // Average confidence over all tokens
        let confidence = accumulated_prob / token_indices.len() as f32;

        Some(NerEntity {
            text: entity_text,
            entity_type,
            confidence,
            start,
            end,
        })
    }

    /// Deduplicate entities (prefer longer spans with higher confidence)
    fn deduplicate_entities(&self, mut entities: Vec<NerEntity>) -> Vec<NerEntity> {
        if entities.len() <= 1 {
            return entities;
        }

        // Sort by start position, then by length (descending)
        entities.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
        });

        let mut result = Vec::new();
        let mut seen_spans: HashSet<(usize, usize)> = HashSet::new();

        for entity in entities {
            // Check if this span overlaps with any seen span
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

    /// Fallback rule-based extraction using comprehensive EntityExtractor
    ///
    /// Uses the sophisticated EntityExtractor from graph_memory which provides:
    /// - 100+ organization keywords (Indian companies, global tech, startups)
    /// - 50+ location keywords (cities, countries, regions)
    /// - Person name detection with indicators (Mr, Dr, etc.)
    /// - Technology keyword matching (Rust, Python, AWS, etc.)
    /// - Proper noun detection based on capitalization patterns
    /// - Salience scoring based on entity type and context
    fn extract_fallback(&self, text: &str) -> Result<Vec<NerEntity>> {
        use crate::graph_memory::{EntityExtractor, EntityLabel};

        // Lazy-load the EntityExtractor (1000+ lines of dictionaries, only init once)
        let extractor = self.entity_extractor.get_or_init(EntityExtractor::new);

        // Extract entities with salience information
        let extracted = extractor.extract_with_salience(text);

        // Convert EntityLabel to NerEntityType and build NerEntity structs
        let entities: Vec<NerEntity> = extracted
            .into_iter()
            .map(|e| {
                let entity_type = match e.label {
                    EntityLabel::Person => NerEntityType::Person,
                    EntityLabel::Organization => NerEntityType::Organization,
                    EntityLabel::Location => NerEntityType::Location,
                    EntityLabel::Technology
                    | EntityLabel::Concept
                    | EntityLabel::Event
                    | EntityLabel::Date
                    | EntityLabel::Product
                    | EntityLabel::Skill
                    | EntityLabel::Other(_) => NerEntityType::Misc,
                };

                // Use salience as confidence (scaled appropriately)
                // EntityExtractor returns salience 0.6-0.9, map to confidence 0.5-0.85
                let confidence = (e.base_salience * 0.9).min(0.85);

                // Find position in original text (case-insensitive search)
                let (start, end) = text
                    .to_lowercase()
                    .find(&e.name.to_lowercase())
                    .map(|pos| (pos, pos + e.name.len()))
                    .unwrap_or((0, e.name.len()));

                NerEntity {
                    text: e.name,
                    entity_type,
                    confidence,
                    start,
                    end,
                }
            })
            .collect();

        Ok(entities)
    }
}

/// Softmax function for converting logits to probabilities
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    logits
        .iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== NerTag Tests ====================

    #[test]
    fn test_ner_tag_from_index() {
        // TinyBERT-finetuned-NER-ONNX label order: O, MISC, ORG, LOC, PER
        assert_eq!(NerTag::from_index(0), NerTag::Outside);
        assert_eq!(NerTag::from_index(1), NerTag::BeginMisc);
        assert_eq!(NerTag::from_index(2), NerTag::InsideMisc);
        assert_eq!(NerTag::from_index(3), NerTag::BeginOrg);
        assert_eq!(NerTag::from_index(4), NerTag::InsideOrg);
        assert_eq!(NerTag::from_index(5), NerTag::BeginLoc);
        assert_eq!(NerTag::from_index(6), NerTag::InsideLoc);
        assert_eq!(NerTag::from_index(7), NerTag::BeginPerson);
        assert_eq!(NerTag::from_index(8), NerTag::InsidePerson);
        // Out of bounds should default to Outside
        assert_eq!(NerTag::from_index(99), NerTag::Outside);
    }

    #[test]
    fn test_tag_is_begin() {
        assert!(NerTag::BeginPerson.is_begin());
        assert!(NerTag::BeginOrg.is_begin());
        assert!(NerTag::BeginLoc.is_begin());
        assert!(NerTag::BeginMisc.is_begin());
        assert!(!NerTag::InsidePerson.is_begin());
        assert!(!NerTag::Outside.is_begin());
    }

    #[test]
    fn test_tag_is_inside() {
        assert!(NerTag::InsidePerson.is_inside());
        assert!(NerTag::InsideOrg.is_inside());
        assert!(NerTag::InsideLoc.is_inside());
        assert!(NerTag::InsideMisc.is_inside());
        assert!(!NerTag::BeginPerson.is_inside());
        assert!(!NerTag::Outside.is_inside());
    }

    #[test]
    fn test_tag_entity_type() {
        assert_eq!(
            NerTag::BeginPerson.entity_type(),
            Some(NerEntityType::Person)
        );
        assert_eq!(
            NerTag::InsidePerson.entity_type(),
            Some(NerEntityType::Person)
        );
        assert_eq!(
            NerTag::BeginOrg.entity_type(),
            Some(NerEntityType::Organization)
        );
        assert_eq!(
            NerTag::BeginLoc.entity_type(),
            Some(NerEntityType::Location)
        );
        assert_eq!(NerTag::BeginMisc.entity_type(), Some(NerEntityType::Misc));
        assert_eq!(NerTag::Outside.entity_type(), None);
    }

    #[test]
    fn test_tag_matching() {
        let b_per = NerTag::BeginPerson;
        let i_per = NerTag::InsidePerson;
        let b_org = NerTag::BeginOrg;
        let i_org = NerTag::InsideOrg;

        // Same entity type should match
        assert!(b_per.matches_type(&i_per));
        assert!(b_org.matches_type(&i_org));

        // Different entity types should not match
        assert!(!b_per.matches_type(&b_org));
        assert!(!i_per.matches_type(&i_org));
    }

    // ==================== NerEntityType Tests ====================

    #[test]
    fn test_entity_type_as_str() {
        assert_eq!(NerEntityType::Person.as_str(), "PER");
        assert_eq!(NerEntityType::Organization.as_str(), "ORG");
        assert_eq!(NerEntityType::Location.as_str(), "LOC");
        assert_eq!(NerEntityType::Misc.as_str(), "MISC");
    }

    // ==================== Softmax Tests ====================

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Highest logit should have highest prob
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![1.0, 1.0, 1.0];
        let probs = softmax(&logits);

        // Uniform logits should give uniform probabilities
        for prob in &probs {
            assert!((*prob - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        // Test numerical stability with large values
        let logits = vec![100.0, 101.0, 102.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-1.0, 0.0, 1.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    // ==================== NerConfig Tests ====================

    #[test]
    fn test_ner_config_default() {
        let config = NerConfig::default();
        assert_eq!(config.max_length, 128); // bert-tiny uses 128
        assert!((config.confidence_threshold - 0.7).abs() < 1e-5);
    }

    // ==================== NeuralNer Fallback Tests ====================

    #[test]
    fn test_fallback_mode_detection() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        assert!(ner.is_fallback_mode());
    }

    #[test]
    fn test_fallback_extraction_organizations() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);

        // Test various organizations
        let test_cases = vec![
            (
                "Microsoft is a company",
                "Microsoft",
                NerEntityType::Organization,
            ),
            ("I work at Google", "Google", NerEntityType::Organization),
            (
                "Apple released a new product",
                "Apple",
                NerEntityType::Organization,
            ),
            (
                "Tata group is expanding",
                "Tata",
                NerEntityType::Organization,
            ),
            (
                "Infosys reported earnings",
                "Infosys",
                NerEntityType::Organization,
            ),
        ];

        for (text, expected_entity, expected_type) in test_cases {
            let entities = ner.extract(text).unwrap();
            let found = entities.iter().find(|e| e.text == expected_entity);
            assert!(
                found.is_some(),
                "Should find {} in '{}'",
                expected_entity,
                text
            );
            assert_eq!(
                found.unwrap().entity_type,
                expected_type,
                "Wrong type for {} in '{}'",
                expected_entity,
                text
            );
        }
    }

    #[test]
    fn test_fallback_extraction_locations() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);

        // Test various locations
        let test_cases = vec![
            (
                "The office is in Seattle",
                "Seattle",
                NerEntityType::Location,
            ),
            (
                "I visited Mumbai last week",
                "Mumbai",
                NerEntityType::Location,
            ),
            ("Tokyo is beautiful", "Tokyo", NerEntityType::Location),
            ("Moving to Bangalore", "Bangalore", NerEntityType::Location),
            ("India is growing", "India", NerEntityType::Location),
        ];

        for (text, expected_entity, expected_type) in test_cases {
            let entities = ner.extract(text).unwrap();
            let found = entities.iter().find(|e| e.text == expected_entity);
            assert!(
                found.is_some(),
                "Should find {} in '{}'",
                expected_entity,
                text
            );
            assert_eq!(
                found.unwrap().entity_type,
                expected_type,
                "Wrong type for {} in '{}'",
                expected_entity,
                text
            );
        }
    }

    #[test]
    fn test_fallback_extraction_mixed() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        let entities = ner
            .extract("Microsoft is headquartered in Seattle")
            .unwrap();

        // Should find both Microsoft (Org) and Seattle (Loc)
        let microsoft = entities.iter().find(|e| e.text == "Microsoft");
        let seattle = entities.iter().find(|e| e.text == "Seattle");

        assert!(microsoft.is_some());
        assert!(seattle.is_some());
        assert_eq!(microsoft.unwrap().entity_type, NerEntityType::Organization);
        assert_eq!(seattle.unwrap().entity_type, NerEntityType::Location);
    }

    #[test]
    fn test_fallback_extraction_empty_text() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        let entities = ner.extract("").unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_fallback_extraction_whitespace_only() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        let entities = ner.extract("   \t\n  ").unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_fallback_extraction_no_entities() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        let entities = ner.extract("the quick brown fox jumps").unwrap();

        // All lowercase, no entities expected
        assert!(entities.is_empty());
    }

    #[test]
    fn test_fallback_deduplication() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        // Microsoft mentioned twice
        let entities = ner
            .extract("Microsoft partnered with Microsoft Azure")
            .unwrap();

        // Should only have one Microsoft entry (deduplicated)
        let microsoft_count = entities.iter().filter(|e| e.text == "Microsoft").count();
        assert_eq!(microsoft_count, 1, "Microsoft should appear only once");
    }

    #[test]
    fn test_fallback_confidence_scores() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        let entities = ner.extract("Microsoft Google Apple").unwrap();

        for entity in &entities {
            // Fallback confidence should be reasonable (0.5-0.8 range)
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
        };

        let cloned = entity.clone();
        assert_eq!(cloned.text, entity.text);
        assert_eq!(cloned.entity_type, entity.entity_type);
        assert!((cloned.confidence - entity.confidence).abs() < 1e-5);
    }

    // ==================== Edge Case Tests ====================

    #[test]
    fn test_single_character_words() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        // Single character words should be skipped
        let entities = ner.extract("I A B C").unwrap();
        // Single chars are too short to be meaningful entities
        assert!(entities.is_empty() || entities.iter().all(|e| e.text.len() >= 2));
    }

    #[test]
    fn test_punctuation_handling() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);
        let entities = ner.extract("Microsoft, Google, and Apple!").unwrap();

        // Should extract entities without punctuation
        for entity in &entities {
            assert!(!entity.text.contains(','));
            assert!(!entity.text.contains('!'));
        }
    }

    #[test]
    fn test_indian_companies() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);

        // Test Indian companies specifically
        let indian_companies = vec!["Flipkart", "Zomato", "Swiggy", "Paytm"];

        for company in indian_companies {
            let entities = ner.extract(&format!("{} is growing", company)).unwrap();
            let found = entities.iter().find(|e| e.text == company);
            assert!(found.is_some(), "Should find Indian company: {}", company);
        }
    }

    #[test]
    fn test_indian_cities() {
        let config = NerConfig {
            model_path: PathBuf::from("nonexistent.onnx"),
            tokenizer_path: PathBuf::from("nonexistent.json"),
            max_length: 128,
            confidence_threshold: 0.5,
        };

        let ner = NeuralNer::new_fallback(config);

        // Test Indian cities specifically
        let indian_cities = vec!["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"];

        for city in indian_cities {
            let entities = ner.extract(&format!("Office in {}", city)).unwrap();
            let found = entities.iter().find(|e| e.text == city);
            assert!(found.is_some(), "Should find Indian city: {}", city);
            assert_eq!(
                found.unwrap().entity_type,
                NerEntityType::Location,
                "{} should be Location",
                city
            );
        }
    }
}
