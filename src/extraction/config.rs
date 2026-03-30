use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    pub entity_type: String,
    pub match_pattern: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractionConfig {
    pub patterns: Vec<PatternConfig>,
    pub stopwords: Vec<String>,
}

#[derive(Debug)]
pub struct CompiledPattern {
    pub entity_type: String,
    pub regex: Regex,
}

impl ExtractionConfig {
    pub fn compile_patterns(&self) -> Vec<CompiledPattern> {
        self.patterns
            .iter()
            .filter_map(|p| {
                match Regex::new(&p.match_pattern) {
                    Ok(regex) => Some(CompiledPattern {
                        entity_type: p.entity_type.clone(),
                        regex,
                    }),
                    Err(e) => {
                        tracing::warn!("Failed to compile extraction pattern '{}': {}", p.match_pattern, e);
                        None
                    }
                }
            })
            .collect()
    }
}
