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

/// Process-wide entity-type schema, parsed once from the embedded JSON asset
/// or from the operator override.
static GLOBAL_SCHEMA: OnceLock<Schema> = OnceLock::new();

/// Env var pointing at a replacement schema JSON, same shape as the embedded one.
pub const SCHEMA_PATH_ENV: &str = "SHODH_ENTITY_SCHEMA_PATH";

/// The full coarse/fine entity-type schema.
///
/// Defaults to the compiled-in asset, so the schema ships with the crate and
/// needs no filesystem access — that is what lets the binary run air-gapped
/// with nothing beside it.
///
/// `SHODH_ENTITY_SCHEMA_PATH` replaces it, which is how a domain taxonomy
/// (GeoNames feature codes, STIX object types, an analyst's own ontology) gets
/// in without a rebuild. Two rules make the override safe to hand to an
/// operator:
///
/// 1. **A bad override is fatal, never a silent fallback.** If the file is
///    missing, unparseable or structurally invalid, this panics with the reason
///    instead of quietly running the default. A system that reports "loaded your
///    ontology" while typing against a different one is worse than one that
///    refuses to start.
/// 2. **The label tower must still match.** GLiNER maps a predicted class index
///    to a fine label BY SCHEMA ORDER against `label_embeddings` computed
///    offline, so a schema with a different label count is rejected by
///    `gliner.rs` ("schema/asset mismatch"). Changing labels means regenerating
///    that asset — the count check catches a resized set, but nothing can catch
///    a same-size set with different labels, so regenerate rather than assume.
pub fn schema() -> &'static Schema {
    GLOBAL_SCHEMA.get_or_init(|| {
        let (json, origin) = match std::env::var(SCHEMA_PATH_ENV) {
            Ok(p) if !p.trim().is_empty() => {
                let text = std::fs::read_to_string(&p)
                    .unwrap_or_else(|e| panic!("{SCHEMA_PATH_ENV}={p} could not be read: {e}"));
                (text, p)
            }
            _ => (SCHEMA_JSON.to_string(), "<compiled-in>".to_string()),
        };

        let parsed: Schema = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("entity-type schema from {origin} is not valid JSON: {e}"));

        if let Err(e) = parsed.validate() {
            panic!("entity-type schema from {origin} is invalid: {e}");
        }

        parsed
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

impl Schema {
    /// Structural checks an operator-supplied schema must pass.
    ///
    /// Each one corresponds to a way a hand-written taxonomy silently degrades
    /// typing rather than failing: a fine label pointing at a coarse class that
    /// does not exist makes `coarse_of` return None and the entity untyped
    /// downstream; a duplicate label means the bi-encoder has two rows
    /// competing for the same string and the argmax picks arbitrarily; an empty
    /// set produces a model that types nothing while still reporting success.
    pub fn validate(&self) -> Result<(), String> {
        if self.coarse.is_empty() {
            return Err("no coarse classes defined".into());
        }
        if self.fine.is_empty() {
            return Err("no fine labels defined".into());
        }

        let mut coarse_ids = std::collections::HashSet::new();
        for c in &self.coarse {
            if c.id.trim().is_empty() {
                return Err("a coarse class has an empty id".into());
            }
            if !coarse_ids.insert(c.id.as_str()) {
                return Err(format!("duplicate coarse id {:?}", c.id));
            }
        }

        let mut fine_labels = std::collections::HashSet::new();
        for f in &self.fine {
            if f.label.trim().is_empty() {
                return Err("a fine label is empty".into());
            }
            if !fine_labels.insert(f.label.as_str()) {
                return Err(format!(
                    "duplicate fine label {:?} — the label tower would have two rows \
                     competing for the same string",
                    f.label
                ));
            }
            if !coarse_ids.contains(f.coarse.as_str()) {
                return Err(format!(
                    "fine label {:?} rolls up to unknown coarse id {:?}",
                    f.label, f.coarse
                ));
            }
        }

        Ok(())
    }
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

    /// Builds a minimal valid schema, then breaks it one way at a time.
    ///
    /// `validate` is the only thing standing between an operator's hand-written
    /// taxonomy and a system that types badly while reporting success, so each
    /// rejection is pinned individually rather than trusting one happy path.
    fn ok_schema() -> Schema {
        serde_json::from_str(
            r#"{
              "version": "1.0",
              "note": "test",
              "coarse": [{"id":"person","onto":"PERSON","source":"ontonotes","wikidata":null}],
              "fine":   [{"label":"politician","coarse":"person","source":"figer","wikidata":null}]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn validate_accepts_a_wellformed_schema() {
        assert!(ok_schema().validate().is_ok());
    }

    #[test]
    fn validate_rejects_fine_label_pointing_at_unknown_coarse() {
        // Silently produces untyped entities downstream: coarse_of returns None.
        let mut s = ok_schema();
        s.fine[0].coarse = "nonexistent".into();
        let err = s.validate().unwrap_err();
        assert!(err.contains("unknown coarse id"), "got: {err}");
    }

    #[test]
    fn validate_rejects_duplicate_fine_labels() {
        // Two rows in the label tower competing for one string — argmax picks
        // arbitrarily and typing becomes non-reproducible across builds.
        let mut s = ok_schema();
        let dup = s.fine[0].clone();
        s.fine.push(dup);
        let err = s.validate().unwrap_err();
        assert!(err.contains("duplicate fine label"), "got: {err}");
    }

    #[test]
    fn validate_rejects_duplicate_coarse_ids() {
        let mut s = ok_schema();
        let dup = s.coarse[0].clone();
        s.coarse.push(dup);
        let err = s.validate().unwrap_err();
        assert!(err.contains("duplicate coarse id"), "got: {err}");
    }

    #[test]
    fn validate_rejects_empty_tiers() {
        let mut s = ok_schema();
        s.fine.clear();
        assert!(s.validate().unwrap_err().contains("no fine labels"));

        let mut s = ok_schema();
        s.coarse.clear();
        assert!(s.validate().unwrap_err().contains("no coarse classes"));
    }

    #[test]
    fn the_shipped_schema_passes_its_own_validation() {
        // The compiled-in asset must satisfy the same rules an override does.
        schema()
            .validate()
            .expect("shipped entity-type-schema.json must be valid");
    }

    /// Ports the Python generator's own integrity assertion: every fine label's
    /// `coarse` id must resolve to a real coarse class, and the tier sizes must
    /// match the designed taxonomy (141 fine labels over 18 coarse classes).
    #[test]
    fn every_fine_rolls_up_to_a_real_coarse() {
        let s = schema();
        assert_eq!(
            s.coarse.len(),
            18,
            "coarse tier must have exactly 18 classes"
        );
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

        let fine_labels: std::collections::HashSet<&str> =
            s.fine.iter().map(|f| f.label.as_str()).collect();
        assert_eq!(
            fine_labels.len(),
            s.fine.len(),
            "fine label set has duplicates — a dup desyncs the label-embedding row count"
        );
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
