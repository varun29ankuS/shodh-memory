//! In-engine syntactic dependency parser (vendored `spacy-rusty`, en_core_web_sm).
//!
//! Provides POS + dependency-head + lemma for short entity mentions so entity
//! resolution can span-clean and pick the syntactic head of a mention
//! (`Port of Baltimore` → `Port`; `ship crashed` → verb-headed fragment). This
//! is the FROZEN base of the entity-resolution roadmap (Phase 1): the parser
//! never updates at runtime; only the resolver/matcher layers above it learn.
//!
//! The model bundle (`model.json` + `model.safetensors`, ~15 MB) is NOT shipped
//! in the repo. It is loaded once from `SHODH_SPACY_MODEL_PATH`, mirroring the
//! GLiNER/MiniLM asset convention. When the variable is unset or the bundle is
//! missing, every entry point degrades to `None` so callers fall back exactly as
//! they do today — the parser is additive, never load-bearing for a request.
//!
//! Concurrency: the vendored crate holds its static-vector table behind `Arc`
//! (see the crate NOTICE), so a single loaded `Pipeline` is `Send + Sync` and is
//! shared read-only across all async workers via a process-wide `OnceLock`.
//! `Pipeline::process` takes `&self`, so concurrent parses are safe.

use std::path::Path;
use std::sync::{Arc, OnceLock};

use spacy_rusty::pipeline::Pipeline;

/// One parsed token: the fields entity resolution consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToken {
    /// Token index within the parsed text.
    pub i: usize,
    /// Surface text of the token.
    pub text: String,
    /// Index of this token's syntactic head; equals `i` for a sentence root.
    pub head: usize,
    /// Dependency label (`"ROOT"` for roots).
    pub dep: String,
    /// Coarse-grained part of speech (`PROPN`, `NOUN`, `VERB`, ...).
    pub pos: String,
    /// Fine-grained POS tag.
    pub tag: String,
    /// Rule-lemmatized form.
    pub lemma: String,
}

impl ParsedToken {
    /// True when this token is a syntactic root (`head == i`).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.head == self.i
    }
}

/// Process-wide, lazily-initialized pipeline. `None` means "no model available";
/// once resolved it never changes, so the 15 MB bundle is loaded at most once.
static PIPELINE: OnceLock<Option<Arc<Pipeline>>> = OnceLock::new();

/// Build a pipeline from an explicit bundle directory. Pure (no env, no global)
/// so it is unit-testable by path — the env/`OnceLock` layer lives in
/// [`pipeline`]. Returns `None` if the manifest/weights are missing or the
/// bundle fails to construct (corrupt data), so a misconfigured model can never
/// panic a request thread.
fn load_from_dir(dir: &Path) -> Option<Arc<Pipeline>> {
    let manifest = std::fs::read_to_string(dir.join("model.json")).ok()?;
    let safetensors = std::fs::read(dir.join("model.safetensors")).ok()?;
    // Optional: enables readable vector neighbors; parsing/POS/lemma do not need
    // it, so `en_core_web_sm` (no static vectors) loads fine without it.
    let key2row = std::fs::read_to_string(dir.join("vectors_key2row.json")).ok();

    // `Pipeline::from_bytes` unwraps internally on malformed input. Contain any
    // such panic here so a bad bundle degrades to `None` instead of aborting.
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Pipeline::from_bytes(&manifest, &safetensors, key2row.as_deref())
    }));
    match built {
        Ok(pipeline) => Some(Arc::new(pipeline)),
        Err(_) => {
            tracing::warn!(
                dir = %dir.display(),
                "spacy-rusty bundle failed to load; dependency parser disabled"
            );
            None
        }
    }
}

/// Resolve the bundle directory from `SHODH_SPACY_MODEL_PATH` and load it.
fn load_from_env() -> Option<Arc<Pipeline>> {
    let dir = std::env::var_os("SHODH_SPACY_MODEL_PATH")?;
    load_from_dir(Path::new(&dir))
}

/// The shared pipeline, or `None` when no model is configured/available.
pub fn pipeline() -> Option<&'static Arc<Pipeline>> {
    PIPELINE.get_or_init(load_from_env).as_ref()
}

/// Whether the dependency parser is available in this process.
pub fn is_available() -> bool {
    pipeline().is_some()
}

/// Parse `text` with an explicitly-provided pipeline (no global state). This is
/// the core used by both [`parse`] and the parity tests.
pub fn parse_with(pipeline: &Pipeline, text: &str) -> Vec<ParsedToken> {
    pipeline
        .process(text)
        .tokens
        .into_iter()
        .map(|t| ParsedToken {
            i: t.i,
            text: t.text,
            head: t.head,
            dep: t.dep,
            pos: t.pos,
            tag: t.tag,
            lemma: t.lemma,
        })
        .collect()
}

/// Parse `text` with the shared pipeline. `None` if no model is available.
pub fn parse(text: &str) -> Option<Vec<ParsedToken>> {
    let pipeline = pipeline()?;
    Some(parse_with(pipeline, text))
}

/// The syntactic head of `text`: the first sentence root (`head == i`). For a
/// short entity mention this is its head word, and the returned `pos` tells the
/// resolver whether the mention is nominal (`PROPN`/`NOUN` → keep) or a
/// verb-headed fragment (`VERB` → strip/reject). Falls back to the first token
/// for degenerate inputs. `None` if no model is available.
pub fn head_token(text: &str) -> Option<ParsedToken> {
    let tokens = parse(text)?;
    Some(head_of(&tokens))
}

/// Select the head token from an already-parsed token list: the first root, else
/// the first token. Split out (pure) so head selection is testable without a model.
fn head_of(tokens: &[ParsedToken]) -> ParsedToken {
    tokens
        .iter()
        .find(|t| t.is_root())
        .or_else(|| tokens.first())
        .cloned()
        .unwrap_or_else(|| ParsedToken {
            i: 0,
            text: String::new(),
            head: 0,
            dep: String::new(),
            pos: String::new(),
            tag: String::new(),
            lemma: String::new(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(i: usize, text: &str, head: usize, pos: &str) -> ParsedToken {
        ParsedToken {
            i,
            text: text.to_string(),
            head,
            dep: if head == i {
                "ROOT".into()
            } else {
                "dep".into()
            },
            pos: pos.to_string(),
            tag: pos.to_string(),
            lemma: text.to_lowercase(),
        }
    }

    #[test]
    fn missing_bundle_dir_yields_none() {
        // Pure loader, no env, no global: a nonexistent bundle degrades to None.
        assert!(load_from_dir(Path::new("does/not/exist/en_core_web_sm")).is_none());
    }

    #[test]
    fn head_of_picks_the_root_not_the_first_token() {
        // "Port of Baltimore" shape: root is "Port" (i=0 here) — but prove we key
        // on head==i, not position, by making the root the middle token.
        let toks = vec![
            tok(0, "the", 1, "DET"),
            tok(1, "ship", 1, "NOUN"), // root
            tok(2, "Dali", 1, "PROPN"),
        ];
        let head = head_of(&toks);
        assert_eq!(head.text, "ship");
        assert!(head.is_root());
    }

    #[test]
    fn head_of_falls_back_to_first_token_when_no_root() {
        let toks = vec![tok(0, "alpha", 1, "NOUN"), tok(1, "beta", 2, "NOUN")];
        let head = head_of(&toks);
        assert_eq!(head.text, "alpha");
    }

    #[test]
    fn head_of_empty_is_safe() {
        assert_eq!(head_of(&[]).text, "");
    }
}
