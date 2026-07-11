//! Domain knowledge base for retrieval-based entity linking — ER Plan Task 3.2,
//! GATED (`SHODH_KB_LINKING`).
//!
//! A **static, CC0** asset (Wikidata-derived) loaded from JSONL where each entity
//! carries its label, aliases, a short description, a coarse type, AND a
//! **precomputed MiniLM embedding of `label + description`**. Because the
//! embeddings are baked in offline (see `scripts/build_wikidata_kb.py`), the
//! runtime needs no embedder and stays deterministic + on-device: linking is a
//! type-blocked cosine nearest-neighbour of a mention's embedding against the KB,
//! plus an exact alias fast-path. This resolves world-knowledge merges no
//! string/embedding matcher over the corpus can reach (`Google` ↔ `Alphabet`),
//! and the KB grows by indexing — no retrain.

use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Process-wide domain KB, lazily loaded once from `SHODH_KB_PATH` (a JSONL asset).
/// `None` when the variable is unset or the file is missing/empty — callers then
/// degrade to the in-corpus matchers exactly as before (KB linking is additive).
static GLOBAL_KB: OnceLock<Option<DomainKb>> = OnceLock::new();

/// The loaded domain KB, or `None` if not configured.
pub fn global() -> Option<&'static DomainKb> {
    GLOBAL_KB
        .get_or_init(|| {
            let path = std::env::var_os("SHODH_KB_PATH")?;
            match DomainKb::from_jsonl(std::path::Path::new(&path)) {
                Ok(kb) if !kb.is_empty() => {
                    tracing::info!(entities = kb.len(), "domain KB loaded (SHODH_KB_PATH)");
                    Some(kb)
                }
                Ok(_) => None,
                Err(e) => {
                    tracing::warn!("domain KB load failed: {e}");
                    None
                }
            }
        })
        .as_ref()
}

/// True when a domain KB is configured and loaded.
pub fn is_available() -> bool {
    global().is_some()
}

/// One canonical KB entity. `embedding` is MiniLM(`label` + " " + `description`),
/// precomputed offline so the runtime carries no embedder.
#[derive(Debug, Clone, Deserialize)]
pub struct KbEntity {
    /// Canonical id (e.g. a Wikidata QID) — stable across surface forms.
    pub id: String,
    /// Canonical label ("Alphabet Inc.").
    pub label: String,
    /// Known surface variants ("Google", "Alphabet", "GOOGL").
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Short gloss used as the retrieval text ("American multinational…").
    #[serde(default)]
    pub description: String,
    /// Coarse type for blocking (`person`/`organization`/`location`/…), lowercased.
    #[serde(default)]
    pub entity_type: String,
    /// Precomputed MiniLM embedding of `label + description`.
    #[serde(default)]
    pub embedding: Vec<f32>,
}

/// An in-memory domain KB with a type-blocked layout and an alias fast-path.
#[derive(Debug, Default)]
pub struct DomainKb {
    entities: Vec<KbEntity>,
    /// lowercased entity_type → entity indices (blocking).
    by_type: HashMap<String, Vec<usize>>,
    /// lowercased alias / label → entity index (exact fast-path).
    by_alias: HashMap<String, usize>,
}

impl DomainKb {
    /// Load a KB from a JSONL file (one [`KbEntity`] per line). Malformed lines are
    /// skipped (logged at debug), so a partial/streamed asset still loads.
    pub fn from_jsonl(path: &std::path::Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut kb = DomainKb::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<KbEntity>(line) {
                Ok(e) => kb.push(e),
                Err(err) => tracing::debug!("KB: skipped malformed line: {err}"),
            }
        }
        Ok(kb)
    }

    /// Build from entities directly (tests / programmatic loads).
    pub fn from_entities(entities: Vec<KbEntity>) -> Self {
        let mut kb = DomainKb::default();
        for e in entities {
            kb.push(e);
        }
        kb
    }

    fn push(&mut self, e: KbEntity) {
        let idx = self.entities.len();
        let ty = e.entity_type.to_lowercase();
        self.by_type.entry(ty).or_default().push(idx);
        self.by_alias.insert(e.label.trim().to_lowercase(), idx);
        for a in &e.aliases {
            // First writer wins so a canonical label isn't clobbered by a shared alias.
            self.by_alias.entry(a.trim().to_lowercase()).or_insert(idx);
        }
        self.entities.push(e);
    }

    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Exact alias/label link (Tier 0): case-insensitive surface hit. Free, precise.
    pub fn link_by_alias(&self, surface: &str) -> Option<&KbEntity> {
        let key = surface.trim().to_lowercase();
        self.by_alias.get(&key).map(|&i| &self.entities[i])
    }

    /// Retrieval link: the nearest KB entity of a COMPATIBLE type whose cosine to
    /// `mention_emb` is ≥ `min`. Type-blocking is the precision guard — a Location
    /// mention never links to an Organization KB entry. `mention_type` empty →
    /// search all types. Returns `(entity, cosine)`.
    pub fn link_by_embedding(
        &self,
        mention_emb: &[f32],
        mention_type: &str,
        min: f32,
    ) -> Option<(&KbEntity, f32)> {
        if mention_emb.is_empty() {
            return None;
        }
        let ty = mention_type.trim().to_lowercase();
        let candidates: Vec<usize> = if ty.is_empty() {
            (0..self.entities.len()).collect()
        } else {
            match self.by_type.get(&ty) {
                Some(v) => v.clone(),
                None => return None,
            }
        };
        let mut best: Option<(usize, f32)> = None;
        for i in candidates {
            let e = &self.entities[i];
            if e.embedding.len() != mention_emb.len() {
                continue;
            }
            let sim = crate::similarity::cosine_similarity(mention_emb, &e.embedding);
            if sim >= min && best.is_none_or(|(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }
        best.map(|(i, s)| (&self.entities[i], s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(id: &str, label: &str, aliases: &[&str], ty: &str, emb: Vec<f32>) -> KbEntity {
        KbEntity {
            id: id.into(),
            label: label.into(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            description: String::new(),
            entity_type: ty.into(),
            embedding: emb,
        }
    }

    fn kb() -> DomainKb {
        DomainKb::from_entities(vec![
            ent("Q20800404", "Alphabet Inc.", &["Google", "Alphabet", "GOOGL"], "organization", vec![1.0, 0.0, 0.0]),
            ent("Q312", "Apple Inc.", &["Apple", "the iPhone maker"], "organization", vec![0.0, 1.0, 0.0]),
            ent("Q1297", "Chicago", &["the Windy City"], "location", vec![1.0, 0.0, 0.0]),
        ])
    }

    #[test]
    fn alias_fast_path_resolves_google_to_alphabet() {
        let kb = kb();
        let e = kb.link_by_alias("google").expect("alias hit");
        assert_eq!(e.id, "Q20800404");
        assert_eq!(e.label, "Alphabet Inc.");
    }

    #[test]
    fn embedding_link_is_type_blocked() {
        let kb = kb();
        // A mention embedding identical to Alphabet's, typed as a LOCATION, must NOT
        // link to Alphabet (org) — even though "Chicago" (location) shares the vector.
        let (loc, _) = kb
            .link_by_embedding(&[1.0, 0.0, 0.0], "location", 0.8)
            .expect("location link");
        assert_eq!(loc.id, "Q1297", "type-blocking must keep org/location apart");
        let (org, _) = kb
            .link_by_embedding(&[1.0, 0.0, 0.0], "organization", 0.8)
            .expect("org link");
        assert_eq!(org.id, "Q20800404");
    }

    #[test]
    fn embedding_link_respects_threshold() {
        let kb = kb();
        // Orthogonal to every org embedding → no link above 0.8.
        assert!(kb
            .link_by_embedding(&[0.0, 0.0, 1.0], "organization", 0.8)
            .is_none());
    }

    #[test]
    fn loads_from_jsonl() {
        let dir = std::env::temp_dir();
        let path = dir.join("shodh_kb_test.jsonl");
        std::fs::write(
            &path,
            "{\"id\":\"Q312\",\"label\":\"Apple Inc.\",\"aliases\":[\"Apple\"],\"entity_type\":\"organization\",\"embedding\":[0.0,1.0,0.0]}\n\ngarbage-line\n",
        )
        .unwrap();
        let kb = DomainKb::from_jsonl(&path).unwrap();
        assert_eq!(kb.len(), 1, "malformed/blank lines skipped");
        assert_eq!(kb.link_by_alias("apple").unwrap().id, "Q312");
        let _ = std::fs::remove_file(&path);
    }
}
