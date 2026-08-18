//! The lexical leg must match a query to its morphological variants.
//!
//! Measured on the LoCoMo gate corpus (629 docs, `tests/recall/corpora/`): of the
//! 76 `multi_hop` (query, gold) pairs, **53 share no content word with the query**
//! once the two speaker names are removed — the benchmark is largely a paraphrase
//! task. Of those 53, **18 become lexically reachable under an English stemmer**
//! and only 2 under a WordNet hypernym walk. Stemming is the single largest
//! lexical lever in that set by an order of magnitude.
//!
//! The cases below are taken verbatim from that corpus rather than invented, so a
//! regression here is a regression in a number we actually gate on.
//!
//! `rust_stemmers` is already used by the entity index (`graph_memory.rs`), the
//! query parser, compression and temporal facts. The BM25 leg was the one place
//! that did not stem, because `tantivy::schema::TEXT` binds the `default`
//! analyzer (`SimpleTokenizer + RemoveLongFilter + LowerCaser`) with no stemmer.

use shodh_memory::memory::hybrid_search::BM25Index;
use shodh_memory::memory::MemoryId;

fn indexed(docs: &[(&str, &str)]) -> (tempfile::TempDir, BM25Index, Vec<MemoryId>) {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = BM25Index::new(dir.path()).expect("index");
    let mut ids = Vec::new();
    for (content, entity) in docs {
        let id = MemoryId(uuid::Uuid::new_v4());
        let entities = if entity.is_empty() {
            Vec::new()
        } else {
            vec![(*entity).to_string()]
        };
        index.upsert(&id, content, &[], &entities).expect("upsert");
        ids.push(id);
    }
    index.commit().expect("commit");
    index.reload().expect("reload");
    (dir, index, ids)
}

/// `conv-42_q27` asks "What places has Joanna **submitted** her work to?" and its
/// only gold (`conv-42:D16:1`) says "just got done **submitting** my recent
/// screenplay". Without stemming the lexical leg contributes nothing to a case
/// whose gold is otherwise a clean keyword match.
#[test]
fn query_verb_matches_the_inflected_form_in_the_document() {
    let (_dir, index, ids) = indexed(&[
        (
            "Hey Nate, long time no see! I just got done submitting my recent screenplay to a film contest",
            "",
        ),
        ("Thanks! The turtles might be small, but both sure have big personalities", ""),
    ]);

    let results = index.search("submitted", 10).expect("search");
    assert!(
        results.iter().any(|(id, _)| id == &ids[0]),
        "\"submitted\" must reach a document that says \"submitting\" — this is \
         conv-42_q27 in the LoCoMo gate, and 18 of 76 multi_hop gold pairs turn \
         on exactly this. Got {} result(s).",
        results.len()
    );
}

/// The same failure in the other direction: `conv-42_q30` asks about Joanna's
/// "**writings**" and the gold says she "**writes**". Plural/verb inflection is
/// the most common shape in the 18.
#[test]
fn noun_plural_and_verb_forms_share_a_stem() {
    let (_dir, index, ids) = indexed(&[
        (
            "I have been busy with writing projects and really going all out",
            "",
        ),
        (
            "Playing video games and watching movies are my main hobbies",
            "",
        ),
    ]);

    let results = index.search("writings", 10).expect("search");
    assert!(
        results.iter().any(|(id, _)| id == &ids[0]),
        "\"writings\" must reach a document that says \"writing\". Got {} result(s).",
        results.len()
    );
}

/// Reopening an index the current binary wrote must NOT trigger the schema
/// migration. The migration deletes the index directory and leans on the startup
/// backfill to refill it from RocksDB; if the schema comparison in
/// `BM25Index::new` were ever spuriously unequal, every single restart would
/// discard the whole lexical index and serve empty results until backfill
/// completed. This is the regression guard for that.
#[test]
fn reopening_a_current_index_preserves_its_documents() {
    let dir = tempfile::tempdir().expect("tempdir");
    let id = MemoryId(uuid::Uuid::new_v4());

    {
        let index = BM25Index::new(dir.path()).expect("create");
        index
            .upsert(
                &id,
                "submitting my recent screenplay to a film contest",
                &[],
                &[],
            )
            .expect("upsert");
        index.commit().expect("commit");
        index.reload().expect("reload");
        assert!(
            !index.search("submitted", 10).expect("search").is_empty(),
            "precondition: the document is findable before reopen"
        );
    }

    let reopened = BM25Index::new(dir.path()).expect("reopen");
    reopened.reload().expect("reload");
    let results = reopened.search("submitted", 10).expect("search");
    assert!(
        results.iter().any(|(found, _)| found == &id),
        "reopening an index written by this same binary must not wipe it — a \
         spurious schema mismatch would silently empty the lexical leg on every \
         restart. Got {} result(s).",
        results.len()
    );
}

/// Stemming must not be bought at the cost of collapsing distinct proper nouns.
/// `graph_memory.rs` deliberately keeps proper nouns out of its stemmed entity
/// index "to prevent 'Paris' → 'pari' merging with 'Parison'"; the lexical leg
/// must not reintroduce that collision through the `entities` field.
#[test]
fn distinct_proper_nouns_do_not_collide_through_stemming() {
    let (_dir, index, ids) = indexed(&[
        (
            "The team met in Paris last spring to review the plan",
            "Paris",
        ),
        (
            "Parison discussed the glass forming process at length",
            "Parison",
        ),
    ]);

    let results = index.search("Paris", 10).expect("search");
    let top = results.first().map(|(id, _)| id);
    assert_eq!(
        top,
        Some(&ids[0]),
        "\"Paris\" must rank the Paris document first, not the Parison one — \
         stemming the entities field would merge these two distinct names."
    );
}
