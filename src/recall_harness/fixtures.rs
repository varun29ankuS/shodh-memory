//! L1 smoke-suite fixtures.
//!
//! Loads a fixed corpus of shodh-flavoured memories and 30 hand-crafted
//! queries against that corpus. The fixture format is JSONL, one record per
//! line, so individual cases are easy to add, diff, and review.
//!
//! Items reference each other by stable string handles (`ssm-NNN` for corpus
//! items). The harness binary maps these handles to assigned `Uuid`s at
//! ingest time, sidestepping the need for deterministic UUID generation.
//!
//! See issue #265 for the suite's design and acceptance criteria.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Canonical relative path to the smoke-suite corpus inside the repo.
pub const SMOKE_CORPUS_PATH: &str = "tests/recall/corpora/shodh-smoke.jsonl";

/// Canonical relative path to the smoke-suite query cases inside the repo.
pub const SMOKE_CASES_PATH: &str = "tests/recall/smoke_cases.jsonl";

/// Identifier used by every smoke-suite case in `fixture_corpus_id`.
pub const SMOKE_CORPUS_ID: &str = "shodh-smoke";

/// Required category counts in the L1 suite.
///
/// Five cases per category, six categories — total 30. The constants are
/// asserted by `validate_smoke_suite` so adding or removing cases without
/// rebalancing the categories fails CI.
pub const CASES_PER_CATEGORY: usize = 5;
pub const TOTAL_SMOKE_CASES: usize = CASES_PER_CATEGORY * 6;

/// One memory in the smoke corpus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CorpusItem {
    /// Stable handle, e.g. `ssm-001`.
    pub id: String,
    /// Memory text. Ingested verbatim.
    pub content: String,
    /// Memory type tag (matches `MemoryType` enum names lowercased).
    pub memory_type: String,
    /// Free-form tags for filtering and BM25 boosts.
    pub tags: Vec<String>,
    /// Authoring timestamp. Used to drive temporal queries.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// One graded relevance judgement for a query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RelevanceJudgement {
    /// Refers to a `CorpusItem::id`.
    pub corpus_item_id: String,
    /// 1 = marginally relevant, 2 = relevant, 3 = highly relevant.
    /// Used directly as the graded relevance for NDCG.
    pub grade: u8,
}

/// One smoke-suite query case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SmokeCase {
    /// Stable handle, e.g. `smoke-001`.
    pub id: String,
    pub category: SmokeCategory,
    pub query: String,
    /// MUST equal [`SMOKE_CORPUS_ID`] for L1 cases.
    pub fixture_corpus_id: String,
    pub relevant: Vec<RelevanceJudgement>,
}

/// The six categories the L1 suite exercises.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum SmokeCategory {
    Decision,
    Code,
    Temporal,
    Entity,
    MultiHop,
    Negation,
}

impl SmokeCategory {
    pub const ALL: [SmokeCategory; 6] = [
        SmokeCategory::Decision,
        SmokeCategory::Code,
        SmokeCategory::Temporal,
        SmokeCategory::Entity,
        SmokeCategory::MultiHop,
        SmokeCategory::Negation,
    ];
}

/// Errors surfaced by the fixture loaders.
#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("json parse error in {path} at line {line}: {source}")]
    Parse {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
}

/// Read a JSONL file, parsing one record per non-empty line.
fn load_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>, FixtureError> {
    let file = File::open(path).map_err(|source| FixtureError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (i, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|source| FixtureError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let record: T = serde_json::from_str(trimmed).map_err(|source| FixtureError::Parse {
            path: path.to_path_buf(),
            line: i + 1,
            source,
        })?;
        out.push(record);
    }
    Ok(out)
}

/// Load the smoke corpus from a JSONL file.
pub fn load_corpus(path: &Path) -> Result<Vec<CorpusItem>, FixtureError> {
    load_jsonl(path)
}

/// Load the smoke query cases from a JSONL file.
pub fn load_smoke_cases(path: &Path) -> Result<Vec<SmokeCase>, FixtureError> {
    load_jsonl(path)
}

/// Resolve a path under the crate root (`CARGO_MANIFEST_DIR`).
///
/// Used by tests and the `recall-eval` binary to locate the canonical
/// fixture files without assuming the caller's working directory.
pub fn manifest_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Errors surfaced by [`validate_smoke_suite`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("expected {TOTAL_SMOKE_CASES} cases, found {0}")]
    WrongTotal(usize),
    #[error("category {0:?} has {1} cases, expected {CASES_PER_CATEGORY}")]
    WrongCategoryCount(SmokeCategory, usize),
    #[error("duplicate case id: {0}")]
    DuplicateCaseId(String),
    #[error("duplicate corpus item id: {0}")]
    DuplicateCorpusId(String),
    #[error("case {case} references unknown corpus item: {corpus_item}")]
    UnknownCorpusReference { case: String, corpus_item: String },
    #[error("case {0} has no relevant items")]
    EmptyRelevant(String),
    #[error("case {case}: relevance grade {grade} is out of range (expected 1..=3)")]
    BadGrade { case: String, grade: u8 },
    #[error("case {0}: fixture_corpus_id must equal `{SMOKE_CORPUS_ID}`")]
    WrongCorpusId(String),
    #[error("case {case}: duplicate corpus item in relevant list: {corpus_item}")]
    DuplicateRelevant { case: String, corpus_item: String },
}

/// Structural integrity check for the L1 smoke suite.
///
/// Runs at unit-test time so any malformed addition to the fixture files
/// fails CI before it can mask a recall regression.
pub fn validate_smoke_suite(
    corpus: &[CorpusItem],
    cases: &[SmokeCase],
) -> Result<(), ValidationError> {
    use std::collections::{HashMap, HashSet};

    // Corpus item IDs must be unique.
    let mut corpus_ids = HashSet::with_capacity(corpus.len());
    for item in corpus {
        if !corpus_ids.insert(item.id.clone()) {
            return Err(ValidationError::DuplicateCorpusId(item.id.clone()));
        }
    }

    // Total case count is fixed.
    if cases.len() != TOTAL_SMOKE_CASES {
        return Err(ValidationError::WrongTotal(cases.len()));
    }

    // Each category has exactly `CASES_PER_CATEGORY` cases.
    let mut by_cat: HashMap<SmokeCategory, usize> = HashMap::new();
    for case in cases {
        *by_cat.entry(case.category).or_insert(0) += 1;
    }
    for cat in SmokeCategory::ALL {
        let count = by_cat.get(&cat).copied().unwrap_or(0);
        if count != CASES_PER_CATEGORY {
            return Err(ValidationError::WrongCategoryCount(cat, count));
        }
    }

    // Per-case validation.
    let mut case_ids = HashSet::with_capacity(cases.len());
    for case in cases {
        if !case_ids.insert(case.id.clone()) {
            return Err(ValidationError::DuplicateCaseId(case.id.clone()));
        }
        if case.fixture_corpus_id != SMOKE_CORPUS_ID {
            return Err(ValidationError::WrongCorpusId(case.id.clone()));
        }
        if case.relevant.is_empty() {
            return Err(ValidationError::EmptyRelevant(case.id.clone()));
        }
        let mut seen_in_case = HashSet::with_capacity(case.relevant.len());
        for rel in &case.relevant {
            if !(1..=3).contains(&rel.grade) {
                return Err(ValidationError::BadGrade {
                    case: case.id.clone(),
                    grade: rel.grade,
                });
            }
            if !corpus_ids.contains(&rel.corpus_item_id) {
                return Err(ValidationError::UnknownCorpusReference {
                    case: case.id.clone(),
                    corpus_item: rel.corpus_item_id.clone(),
                });
            }
            if !seen_in_case.insert(rel.corpus_item_id.clone()) {
                return Err(ValidationError::DuplicateRelevant {
                    case: case.id.clone(),
                    corpus_item: rel.corpus_item_id.clone(),
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_path() -> PathBuf {
        manifest_path(SMOKE_CORPUS_PATH)
    }

    fn cases_path() -> PathBuf {
        manifest_path(SMOKE_CASES_PATH)
    }

    #[test]
    fn corpus_loads() {
        let corpus = load_corpus(&corpus_path()).expect("corpus must load");
        assert!(!corpus.is_empty(), "corpus must contain items");
    }

    #[test]
    fn smoke_cases_load() {
        let cases = load_smoke_cases(&cases_path()).expect("cases must load");
        assert_eq!(cases.len(), TOTAL_SMOKE_CASES);
    }

    #[test]
    fn smoke_suite_passes_structural_validation() {
        let corpus = load_corpus(&corpus_path()).expect("corpus must load");
        let cases = load_smoke_cases(&cases_path()).expect("cases must load");
        validate_smoke_suite(&corpus, &cases).expect("smoke suite must validate");
    }

    #[test]
    fn smoke_suite_covers_all_six_categories_evenly() {
        use std::collections::HashMap;
        let cases = load_smoke_cases(&cases_path()).expect("cases must load");
        let mut by_cat: HashMap<SmokeCategory, usize> = HashMap::new();
        for case in &cases {
            *by_cat.entry(case.category).or_insert(0) += 1;
        }
        for cat in SmokeCategory::ALL {
            assert_eq!(
                by_cat.get(&cat).copied().unwrap_or(0),
                CASES_PER_CATEGORY,
                "category {cat:?} should have exactly {CASES_PER_CATEGORY} cases"
            );
        }
    }

    // ---- Validation logic unit tests (using synthetic data) -----------------

    fn now() -> chrono::DateTime<chrono::Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
            .expect("static timestamp is valid")
            .with_timezone(&chrono::Utc)
    }

    fn make_corpus_item(id: &str) -> CorpusItem {
        CorpusItem {
            id: id.to_string(),
            content: format!("content for {id}"),
            memory_type: "decision".to_string(),
            tags: vec![],
            created_at: now(),
        }
    }

    fn make_case(id: &str, category: SmokeCategory, refs: &[(&str, u8)]) -> SmokeCase {
        SmokeCase {
            id: id.to_string(),
            category,
            query: format!("query for {id}"),
            fixture_corpus_id: SMOKE_CORPUS_ID.to_string(),
            relevant: refs
                .iter()
                .map(|(cid, grade)| RelevanceJudgement {
                    corpus_item_id: cid.to_string(),
                    grade: *grade,
                })
                .collect(),
        }
    }

    /// Build a fully-valid 30-case suite with single-item ground truth so we
    /// can mutate one piece at a time in negative tests.
    fn valid_synthetic_suite() -> (Vec<CorpusItem>, Vec<SmokeCase>) {
        let corpus: Vec<CorpusItem> = (1..=TOTAL_SMOKE_CASES)
            .map(|i| make_corpus_item(&format!("c{i:03}")))
            .collect();
        let mut cases = Vec::with_capacity(TOTAL_SMOKE_CASES);
        let mut idx = 1usize;
        for cat in SmokeCategory::ALL {
            for _ in 0..CASES_PER_CATEGORY {
                cases.push(make_case(
                    &format!("smoke-{idx:03}"),
                    cat,
                    &[(&format!("c{idx:03}"), 3)],
                ));
                idx += 1;
            }
        }
        (corpus, cases)
    }

    #[test]
    fn validation_accepts_well_formed_synthetic_suite() {
        let (corpus, cases) = valid_synthetic_suite();
        validate_smoke_suite(&corpus, &cases).expect("synthetic suite must validate");
    }

    #[test]
    fn validation_rejects_wrong_total() {
        let (corpus, mut cases) = valid_synthetic_suite();
        cases.pop();
        assert_eq!(
            validate_smoke_suite(&corpus, &cases).unwrap_err(),
            ValidationError::WrongTotal(TOTAL_SMOKE_CASES - 1)
        );
    }

    #[test]
    fn validation_rejects_uneven_categories() {
        let (corpus, mut cases) = valid_synthetic_suite();
        // Re-categorise the first case to a different category, breaking the balance.
        cases[0].category = SmokeCategory::Negation;
        let err = validate_smoke_suite(&corpus, &cases).unwrap_err();
        assert!(matches!(err, ValidationError::WrongCategoryCount(_, _)));
    }

    #[test]
    fn validation_rejects_duplicate_case_id() {
        let (corpus, mut cases) = valid_synthetic_suite();
        cases[1].id = cases[0].id.clone();
        assert_eq!(
            validate_smoke_suite(&corpus, &cases).unwrap_err(),
            ValidationError::DuplicateCaseId(cases[0].id.clone())
        );
    }

    #[test]
    fn validation_rejects_duplicate_corpus_id() {
        let (mut corpus, cases) = valid_synthetic_suite();
        corpus[1].id = corpus[0].id.clone();
        assert_eq!(
            validate_smoke_suite(&corpus, &cases).unwrap_err(),
            ValidationError::DuplicateCorpusId(corpus[0].id.clone())
        );
    }

    #[test]
    fn validation_rejects_dangling_corpus_reference() {
        let (corpus, mut cases) = valid_synthetic_suite();
        cases[0].relevant[0].corpus_item_id = "does-not-exist".to_string();
        let err = validate_smoke_suite(&corpus, &cases).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::UnknownCorpusReference { .. }
        ));
    }

    #[test]
    fn validation_rejects_empty_relevant_list() {
        let (corpus, mut cases) = valid_synthetic_suite();
        cases[0].relevant.clear();
        assert_eq!(
            validate_smoke_suite(&corpus, &cases).unwrap_err(),
            ValidationError::EmptyRelevant(cases[0].id.clone())
        );
    }

    #[test]
    fn validation_rejects_out_of_range_grade() {
        let (corpus, mut cases) = valid_synthetic_suite();
        cases[0].relevant[0].grade = 0;
        let err = validate_smoke_suite(&corpus, &cases).unwrap_err();
        assert!(matches!(err, ValidationError::BadGrade { grade: 0, .. }));

        let (corpus, mut cases) = valid_synthetic_suite();
        cases[0].relevant[0].grade = 4;
        let err = validate_smoke_suite(&corpus, &cases).unwrap_err();
        assert!(matches!(err, ValidationError::BadGrade { grade: 4, .. }));
    }

    #[test]
    fn validation_rejects_wrong_corpus_id() {
        let (corpus, mut cases) = valid_synthetic_suite();
        cases[0].fixture_corpus_id = "other-corpus".to_string();
        assert_eq!(
            validate_smoke_suite(&corpus, &cases).unwrap_err(),
            ValidationError::WrongCorpusId(cases[0].id.clone())
        );
    }

    #[test]
    fn validation_rejects_duplicate_relevant_within_case() {
        let (corpus, mut cases) = valid_synthetic_suite();
        let dup = cases[0].relevant[0].clone();
        cases[0].relevant.push(dup);
        let err = validate_smoke_suite(&corpus, &cases).unwrap_err();
        assert!(matches!(err, ValidationError::DuplicateRelevant { .. }));
    }
}
