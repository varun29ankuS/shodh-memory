pub mod config;
pub mod maximal_munch;
pub mod metadata;
pub mod nlp;
pub mod types;
pub mod verb_dictionary;
#[cfg(test)]
mod tests;

use aho_corasick::AhoCorasick;
pub use config::ExtractionConfig;
pub use types::{ExtractedEntity, ExtractedTriple, ExtractionResult, ExtractionSource};

pub struct Extractor {
    ac_dict: Option<AhoCorasick>,
    ac_entities: Vec<ExtractedEntity>,
    config: ExtractionConfig,
    compiled_patterns: Vec<config::CompiledPattern>,
}

impl Extractor {
    pub fn new(ac_dict: Option<AhoCorasick>, ac_entities: Vec<ExtractedEntity>, config: ExtractionConfig) -> Self {
        let compiled_patterns = config.compile_patterns();
        Self {
            ac_dict,
            ac_entities,
            config,
            compiled_patterns,
        }
    }

    pub fn extract(&self, text: &str, tags: &[String], issue_ids: &[String]) -> ExtractionResult {
        // Source 0: Metadata
        let metadata_entities = metadata::extract_metadata(tags, issue_ids);

        // Source 1: Maximal Munch
        let mut munch_entities = maximal_munch::run_maximal_munch(
            text,
            self.ac_dict.as_ref(),
            &self.ac_entities,
            &self.compiled_patterns,
        );

        // Build consumed spans
        let mut consumed_spans = Vec::new();
        for ent in &munch_entities {
            // Include parent span, not just derived
            for span in &ent.spans {
                consumed_spans.push(*span);
            }
        }
        
        // Remove Url intermediate entities from munch_entities as per spec
        munch_entities.retain(|e| e.entity_type != "Url");

        // Source 2: NLP
        let (nlp_entities, mut nlp_triples) = nlp::run_nlp_parse(text, &consumed_spans, &munch_entities, &self.config.stopwords);

        // Check canonical dictionary for triples
        for t in &mut nlp_triples {
            if let Some(_relation) = verb_dictionary::get_canonical_relation(&t.verb) {
                if t.confidence >= 0.5 {
                    t.promoted = true;
                }
            }
        }

        // Merge
        let mut all_entities = metadata_entities;
        all_entities.extend(munch_entities);
        all_entities.extend(nlp_entities);

        ExtractionResult {
            entities: all_entities,
            triples: nlp_triples,
        }
    }
}
