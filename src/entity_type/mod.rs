//! In-repo entity-type schema — coarse OntoNotes rollup over a FIGER+domain fine
//! taxonomy (D12). The JSON is generated offline (see git history for the
//! now-deleted Python generator) but is checked in as the single source of
//! truth and embedded into the binary via `include_str!`, so the schema ships
//! with the crate and needs no filesystem access or Python at runtime.
//!
//! GLiNER predicts FINE labels only (coarse and fine must not co-appear in one
//! inference set — a coarse label suppresses the fine call); `coarse_of`
//! performs the rollup after typing. `wikidata_qid` anchors a fine label to a
//! Wikidata QID for KB linking (see `kb.rs`) where one is known.

use serde::Deserialize;
use std::sync::OnceLock;

const SCHEMA_JSON: &str = include_str!("entity-type-schema.json");

/// Process-wide entity-type schema, parsed once from the embedded JSON asset.
static GLOBAL_SCHEMA: OnceLock<Schema> = OnceLock::new();

/// The full coarse/fine entity-type schema.
pub fn schema() -> &'static Schema {
    GLOBAL_SCHEMA.get_or_init(|| {
        serde_json::from_str(SCHEMA_JSON)
            .expect("entity-type-schema.json is a compiled-in asset validated by tests")
    })
}

/// One coarse (OntoNotes-grounded) entity class — used for blocking + colour.
#[derive(Debug, Clone, Deserialize)]
pub struct CoarseDef {
    /// Coarse id (`"person"`, `"organization"`, …) — referenced by `FineDef::coarse`.
    pub id: String,
    /// OntoNotes tag this coarse class maps to (`"PERSON"`, `"ORG"`, …).
    pub onto: String,
    /// Provenance of this coarse class (`"ontonotes"`, `"domain"`, `"ic"`).
    pub source: String,
    /// Wikidata QID anchor for this coarse class, when one is known.
    pub wikidata: Option<String>,
}

/// One fine-grained entity type — what GLiNER predicts. Rolls up to exactly
/// one coarse class via [`FineDef::coarse`].
#[derive(Debug, Clone, Deserialize)]
pub struct FineDef {
    /// Fine label (`"politician"`, `"river"`, …) — the GLiNER inference-set entry.
    pub label: String,
    /// The coarse id this fine label rolls up to (must be a real [`CoarseDef::id`]).
    pub coarse: String,
    /// Provenance of this fine label (`"figer"`, `"ic-geopol"`, `"disaster"`, …).
    pub source: String,
    /// Wikidata QID anchor for this fine label, when one is known.
    pub wikidata: Option<String>,
}

/// The coarse/fine entity-type schema, parsed from the embedded JSON asset.
#[derive(Debug, Clone, Deserialize)]
pub struct Schema {
    /// Schema format version (`"1.0"`).
    pub version: String,
    /// Free-text design note carried over from the JSON asset.
    pub note: String,
    /// Coarse (OntoNotes-grounded) entity classes.
    pub coarse: Vec<CoarseDef>,
    /// Fine-grained entity types — the GLiNER inference set.
    pub fine: Vec<FineDef>,
}

/// The set of fine labels GLiNER will predict, in schema order.
pub fn fine_labels() -> Vec<&'static str> {
    schema().fine.iter().map(|f| f.label.as_str()).collect()
}

/// Roll a fine label up to its coarse id (`"river"` -> `"location"`).
pub fn coarse_of(fine: &str) -> Option<&'static str> {
    schema()
        .fine
        .iter()
        .find(|f| f.label == fine)
        .map(|f| f.coarse.as_str())
}

/// The Wikidata QID anchor for a fine label, when one is known.
pub fn wikidata_qid(fine: &str) -> Option<&'static str> {
    schema()
        .fine
        .iter()
        .find(|f| f.label == fine)
        .and_then(|f| f.wikidata.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ports the Python generator's own integrity assertion: every fine label's
    /// `coarse` id must resolve to a real coarse class, and the tier sizes must
    /// match the designed taxonomy (141 fine labels over 18 coarse classes).
    #[test]
    fn every_fine_rolls_up_to_a_real_coarse() {
        let s = schema();
        assert_eq!(s.coarse.len(), 18, "coarse tier must have exactly 18 classes");
        assert_eq!(s.fine.len(), 141, "fine tier must have exactly 141 labels");

        let coarse_ids: std::collections::HashSet<&str> =
            s.coarse.iter().map(|c| c.id.as_str()).collect();
        for f in &s.fine {
            assert!(
                coarse_ids.contains(f.coarse.as_str()),
                "fine label {:?} rolls up to unknown coarse id {:?}",
                f.label,
                f.coarse
            );
        }
    }

    #[test]
    fn coarse_of_rolls_up_bridge_to_facility() {
        assert_eq!(coarse_of("bridge"), Some("facility"));
    }

    #[test]
    fn wikidata_qid_resolves_river() {
        assert_eq!(wikidata_qid("river"), Some("Q4022"));
    }

    #[test]
    fn fine_labels_matches_schema_len() {
        assert_eq!(fine_labels().len(), schema().fine.len());
    }

    #[test]
    fn unknown_fine_label_resolves_to_none() {
        assert_eq!(coarse_of("not-a-real-label"), None);
        assert_eq!(wikidata_qid("not-a-real-label"), None);
    }
}
