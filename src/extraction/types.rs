use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractionSource {
    Metadata,
    MaximalMunch,
    NlpParse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub text: String,
    pub entity_type: String,       // "Domain", "OperatorId", "NounPhrase", "ProperNoun", etc.
    pub confidence: f32,           // 0.0-1.0
    pub spans: Vec<(usize, usize)>, // Byte indices (start, end) into the original un-normalized text
    pub source: ExtractionSource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTriple {
    pub subject: Option<String>,
    pub subject_span: Option<(usize, usize)>, // Byte indices to bind to specific entity instance
    pub verb: String,              // lemmatized form
    pub object: String,
    pub object_span: Option<(usize, usize)>,  // Byte indices to bind to specific entity instance
    pub confidence: f32,
    pub passive: bool,
    pub promoted: bool,            // true if verb matched canonical dictionary -> typed edge created
}

#[derive(Debug, Clone, Default)]
pub struct ExtractionResult {
    pub entities: Vec<ExtractedEntity>,
    pub triples: Vec<ExtractedTriple>,
}
