//! The lexical leg must match a query to its morphological variants.
//!
//! `tantivy::schema::TEXT` binds the `default` analyzer (`SimpleTokenizer +
//! RemoveLongFilter + LowerCaser`) with no stemmer, so BM25 could not match
//! "letters" to "letter" or "submitted" to "submitting". Every other stemmed
//! surface in this codebase (the entity index in `graph_memory.rs`, the query
//! parser, compression, temporal facts) already stems.
//!
//! The documents and queries below are taken verbatim from the recall corpus
//! (`tests/recall/corpora/locomo.jsonl`) rather than invented, so a regression
//! here is a regression in cases the recall gate scores.

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

/// `conv-42_q62`: "How many letters has Joanna recieved?" Both gold documents
/// say "letter"; no document in the corpus says "letters", and the typo matches
/// nothing. Unstemmed, the only matching query words are "how", "many", "has"
/// and "Joanna", so the distractors (which say "how" and "Joanna") outrank both
/// golds. This is the case the recall gate lost when #509 stopped keyphrases
/// becoming graph nodes: the graph's stemmed entity match had been the only
/// route from "letters" to "letter".
///
/// The fixture keeps the corpus statistics the case depends on. In the corpus
/// "how" occurs in 741 of 5,882 documents and the stem "letter" in 7; with only
/// a handful of documents the two words would be equally rare, and BM25 would
/// weigh a stray "how" as heavily as the one discriminating word. So most
/// distractors say "how", as conversational turns do.
#[test]
fn plural_query_reaches_both_singular_gold_documents() {
    let (_dir, index, ids) = indexed(&[
        (
            "Joanna: Nate, after finishing my screenplay I got a rejection letter from a major company. It really bummed me out.",
            "",
        ),
        (
            "Joanna: Yep. Last week, someone wrote me a letter after reading an online blog post I made about a hard moment in my life. Their words touched me; they said my story had brought them comfort. It was awesome to realize my words had that kind of power. It reminded me why I love writing.",
            "",
        ),
        (
            "Joanna: Hey Nate, how's it going? I took your reccomendation and watched \"The Lord of the Rings\" Trilogy last night! It was awesome!",
            "",
        ),
        (
            "Joanna: Nice! That must have been a surprise. How did it feel to finally win one?",
            "",
        ),
        (
            "Joanna: Wow, Nate! I'm proud of what you did. Your gaming room looks great - have you been gaming a lot recently?",
            "",
        ),
        (
            "Nate: It was! How about you? Do you have any hobbies you love?",
            "",
        ),
        (
            "Nate: Wow, that's amazing! How do you feel now that it's finished? Do you have any new plans for it?",
            "",
        ),
        ("Joanna: Awww! How long have you had them?", ""),
        (
            "Nate: Wow, Joanna, that takes guts! I can't wait to see it all come together. I'm also pumped to see how your first one will do!",
            "",
        ),
        (
            "Joanna: They look so peaceful! It's amazing how these creatures bring so much calm and joy. Is taking care of them tough?",
            "",
        ),
        (
            "Nate: Hey Joanna! Long time no talk, how's it going? Crazy stuff's been happening since we last chatted.",
            "",
        ),
        ("Nate: Congrats! How did it go? Are you excited?", ""),
        (
            "Joanna: Wow, Nate! Can't wait to see it. Must feel so liberating! How're you feeling?",
            "",
        ),
    ]);

    let results = index
        .search("How many letters has Joanna recieved?", 10)
        .expect("search");
    let top_two: Vec<&MemoryId> = results.iter().take(2).map(|(id, _)| id).collect();
    assert!(
        top_two.contains(&&ids[0]) && top_two.contains(&&ids[1]),
        "both \"letter\" documents must outrank documents that only share \"how\" and \
         \"Joanna\" with the query. Top two: {top_two:?}, golds: {:?}",
        &ids[..2]
    );
}

/// `conv-42_q27` asks "What places has Joanna **submitted** her work to?" and its
/// gold says "just got done **submitting** my recent screenplay".
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
        "\"submitted\" must reach a document that says \"submitting\". Got {} result(s).",
        results.len()
    );
}

/// `conv-42_q30` asks about Joanna's "**writings**" and the gold says she is
/// busy with "**writing**" projects.
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

/// Reopening an index this binary wrote must not trigger the schema migration.
/// The migration deletes the index and leans on the startup backfill to refill
/// it; a spurious schema mismatch would empty the lexical leg on every restart.
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
        "reopening an index written by this binary must not wipe it. Got {} result(s).",
        results.len()
    );
}

/// An index written by an earlier binary, with the unstemmed `TEXT` analyzer on
/// every field, must be discarded on open rather than kept: `Index::open`
/// restores the analyzers named in the index's own `meta.json`, so a kept index
/// would go on indexing and querying unstemmed for good. Discarded, it opens
/// empty, which is the signal the startup backfill refills from.
#[test]
fn an_index_with_the_old_unstemmed_schema_is_rebuilt_empty_and_stemmed() {
    use tantivy::schema::{Schema, STORED, STRING, TEXT};

    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut builder = Schema::builder();
        let id_field = builder.add_text_field("id", STRING | STORED);
        let content_field = builder.add_text_field("content", TEXT | STORED);
        builder.add_text_field("tags", TEXT);
        builder.add_text_field("entities", TEXT);
        let old = tantivy::Index::create_in_dir(dir.path(), builder.build()).expect("old index");
        let mut writer: tantivy::IndexWriter = old.writer(15_000_000).expect("writer");
        let mut doc = tantivy::TantivyDocument::default();
        doc.add_text(id_field, uuid::Uuid::new_v4().to_string());
        doc.add_text(content_field, "written before stemming existed");
        writer.add_document(doc).expect("add");
        writer.commit().expect("commit");
    }

    let index = BM25Index::new(dir.path()).expect("open over the old index");
    index.reload().expect("reload");
    assert!(
        index.is_empty(),
        "an index with the old schema must be discarded, not kept; it still holds {} \
         document(s)",
        index.len()
    );

    let id = MemoryId(uuid::Uuid::new_v4());
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
        index
            .search("submitted", 10)
            .expect("search")
            .iter()
            .any(|(found, _)| found == &id),
        "the rebuilt index must use the stemmed analyzer"
    );
}

/// Stemming must not collapse distinct proper nouns. `graph_memory.rs` keeps
/// proper nouns out of its stemmed entity index "to prevent 'Paris' → 'pari'
/// merging with 'Parison'"; the `entities` field stays unstemmed for the same
/// reason.
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
        "\"Paris\" must rank the Paris document first, not the Parison one."
    );
}
