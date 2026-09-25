//! Record and replay of model outputs, so a measurement does not depend on the CPU.
//!
//! ONNX Runtime picks its kernels by CPUID, so the embedder, GLiNER and the
//! cross-encoder return different low bits on runners that expose different CPU
//! features, and near-ties in the ranking flip (#566, #569). Everything after the
//! models is deterministic on any x86_64 machine with AVX2. A tape records every
//! model output of one run, keyed by the exact input. A later run replays them, and
//! then no model arithmetic happens at all, so the result is the same on every CPU.
//!
//! Off unless one of two variables is set:
//! - `SHODH_MODEL_TAPE_RECORD=<path>`: run the models, and record what they return.
//! - `SHODH_MODEL_TAPE_REPLAY=<path>`: return recorded outputs, and never run a model.
//!
//! **A replay miss must never be absorbed.** This codebase degrades on purpose when a
//! model fails: the embedder falls back to simplified embeddings, NER to the rule-based
//! extractor, and `recall()` skips the reranker. Each of those would turn a miss into a
//! quietly different measurement. So a miss returns an error, and it is also added to a
//! process-wide list that the harness checks before it reports anything
//! ([`take_problems`]). Recording refuses degraded outputs for the same reason: a
//! simplified embedding or a fallback entity list is never written to a tape.
//!
//! Keys are the exact text each model saw, after any instruction prefix. The
//! cross-encoder's key is the (query, candidate) pair. Values are kept per input, not
//! per batch: the first value recorded for a key is the one kept, and a later value
//! that differs is counted as a conflict. Batches pad to their longest member, so the
//! conflict count measures whether batch composition changes outputs.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::ner::{NerEntity, NerEntityType};

/// Environment variable naming a tape to record into.
pub const RECORD_ENV: &str = "SHODH_MODEL_TAPE_RECORD";
/// Environment variable naming a tape to replay from.
pub const REPLAY_ENV: &str = "SHODH_MODEL_TAPE_REPLAY";

/// Bumped when the file layout changes, so an old tape is refused rather than misread.
const FORMAT_VERSION: u32 = 1;

/// The three models a tape covers, named as they appear in a tape's header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Model {
    Embedder,
    Ner,
    CrossEncoder,
}

impl Model {
    fn as_str(self) -> &'static str {
        match self {
            Model::Embedder => "embedder",
            Model::Ner => "ner",
            Model::CrossEncoder => "cross_encoder",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Record,
    Replay,
}

/// One recorded NER entity. The confidence travels as its f32 bit pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TapedEntity {
    text: String,
    #[serde(rename = "type")]
    entity_type: String,
    confidence: u32,
    start: usize,
    end: usize,
    fine_label: Option<String>,
}

impl TapedEntity {
    fn from_entity(e: &NerEntity) -> Self {
        Self {
            text: e.text.clone(),
            entity_type: e.entity_type.as_str().to_string(),
            confidence: e.confidence.to_bits(),
            start: e.start,
            end: e.end,
            fine_label: e.fine_label.clone(),
        }
    }

    fn to_entity(&self) -> Result<NerEntity> {
        let entity_type = match self.entity_type.as_str() {
            "PER" => NerEntityType::Person,
            "ORG" => NerEntityType::Organization,
            "LOC" => NerEntityType::Location,
            "MISC" => NerEntityType::Misc,
            other => bail!("model tape: unknown NER type {other:?}"),
        };
        Ok(NerEntity {
            text: self.text.clone(),
            entity_type,
            confidence: f32::from_bits(self.confidence),
            start: self.start,
            end: self.end,
            fine_label: self.fine_label.clone(),
        })
    }
}

/// One line of a tape file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Line {
    Header {
        version: u32,
        /// sha256 of each model's files, by [`Model::as_str`].
        models: HashMap<String, String>,
        /// Free-form provenance: git sha, CPU, conflict counts. Informational only.
        provenance: serde_json::Value,
    },
    Embed {
        text: String,
        bits: Vec<u32>,
    },
    Ner {
        text: String,
        entities: Vec<TapedEntity>,
    },
    Ce {
        query: String,
        doc: String,
        bits: u32,
    },
}

#[derive(Debug, Default)]
struct Tables {
    embed: HashMap<String, Vec<f32>>,
    ner: HashMap<String, Vec<TapedEntity>>,
    ce: HashMap<(String, String), f32>,
    /// Record order, so a written tape is stable for identical runs.
    order: Vec<Key>,
}

#[derive(Debug, Clone)]
enum Key {
    Embed(String),
    Ner(String),
    Ce(String, String),
}

struct Tape {
    mode: Mode,
    path: PathBuf,
    tables: Tables,
    /// Model identities: from the tape header on replay, from registration on record.
    models: HashMap<String, String>,
    /// Record mode: later values for an existing key that differed from the kept one.
    conflicts: HashMap<&'static str, usize>,
    /// Misses and identity mismatches. The harness must see this empty.
    problems: Vec<String>,
    /// Replay mode: models whose loaded files matched the tape header.
    verified: std::collections::HashSet<&'static str>,
}

fn state() -> Option<&'static Mutex<Tape>> {
    static TAPE: OnceLock<Option<Mutex<Tape>>> = OnceLock::new();
    TAPE.get_or_init(|| match init_from_env() {
        Ok(t) => t.map(Mutex::new),
        Err(e) => {
            // A tape that was asked for and cannot be used is fatal for a
            // measurement. Keep a tape object in replay mode with the failure on
            // its problem list, so every lookup misses and the harness refuses.
            tracing::error!("model tape: {e:#}");
            Some(Mutex::new(Tape {
                mode: Mode::Replay,
                path: PathBuf::new(),
                tables: Tables::default(),
                models: HashMap::new(),
                conflicts: HashMap::new(),
                problems: vec![format!("model tape could not be opened: {e:#}")],
                verified: Default::default(),
            }))
        }
    })
    .as_ref()
}

fn init_from_env() -> Result<Option<Tape>> {
    let record = std::env::var_os(RECORD_ENV).filter(|v| !v.is_empty());
    let replay = std::env::var_os(REPLAY_ENV).filter(|v| !v.is_empty());
    if (record.is_some() || replay.is_some())
        && std::env::var_os("SHODH_NER_REPLAY").is_some_and(|v| !v.is_empty())
    {
        bail!(
            "SHODH_NER_REPLAY and the model tape are both set; the NER replay hook \
             drops GLiNER fine labels, so the two would disagree about entities"
        );
    }
    match (record, replay) {
        (Some(_), Some(_)) => bail!("{RECORD_ENV} and {REPLAY_ENV} are both set"),
        (Some(p), None) => Ok(Some(Tape {
            mode: Mode::Record,
            path: PathBuf::from(p),
            tables: Tables::default(),
            models: HashMap::new(),
            conflicts: HashMap::new(),
            problems: Vec::new(),
            verified: Default::default(),
        })),
        (None, Some(p)) => {
            let path = PathBuf::from(p);
            let (models, tables) = read_tape(&path)?;
            Ok(Some(Tape {
                mode: Mode::Replay,
                path,
                tables,
                models,
                conflicts: HashMap::new(),
                problems: Vec::new(),
                verified: Default::default(),
            }))
        }
        (None, None) => Ok(None),
    }
}

/// Whether a tape is in use in this process.
pub fn active() -> bool {
    state().is_some()
}

/// Whether this process replays from a tape (and so must not run any model).
pub fn replaying() -> bool {
    state().is_some_and(|t| t.lock().mode == Mode::Replay)
}

/// Whether this process records into a tape.
pub fn recording() -> bool {
    state().is_some_and(|t| t.lock().mode == Mode::Record)
}

/// Declare the files a model was loaded from.
///
/// Recording keeps their hash for the tape header. Replaying compares it with the
/// header: a tape recorded from another model version would return that model's
/// outputs, so a mismatch is a problem the harness reports. A no-op with no tape.
pub fn register_model(model: Model, files: &[&Path]) {
    let Some(tape) = state() else { return };
    let digest = match hash_files(files) {
        Ok(d) => d,
        Err(e) => {
            tape.lock().problems.push(format!(
                "{}: cannot hash model files: {e:#}",
                model.as_str()
            ));
            return;
        }
    };
    let mut t = tape.lock();
    match t.mode {
        Mode::Record => match t.models.get(model.as_str()) {
            Some(existing) if *existing != digest => {
                let msg = format!(
                    "{}: two different model files were loaded in one recording ({existing} and {digest})",
                    model.as_str()
                );
                t.problems.push(msg);
            }
            _ => {
                t.models.insert(model.as_str().to_string(), digest);
            }
        },
        Mode::Replay => match t.models.get(model.as_str()) {
            Some(expected) if *expected == digest => {
                t.verified.insert(model.as_str());
            }
            Some(expected) => {
                let msg = format!(
                    "{}: the tape was recorded from model files with sha256 {expected}, \
                     but this process loaded {digest}; re-record the tape",
                    model.as_str()
                );
                t.problems.push(msg);
            }
            None => {
                let msg = format!("{}: the tape has no outputs for this model", model.as_str());
                t.problems.push(msg);
            }
        },
    }
}

fn hash_files(files: &[&Path]) -> Result<String> {
    // Hashing the 149 MB GLiNER graph costs about a second, and a harness builds a
    // new manager per repeat, so each path is hashed once per process.
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, String>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut combined = Sha256::new();
    for path in files {
        let digest = if let Some(d) = cache.lock().get(*path).cloned() {
            d
        } else {
            let mut f =
                std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
            let mut h = Sha256::new();
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = f.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                h.update(&buf[..n]);
            }
            let d = hex::encode(h.finalize());
            cache.lock().insert(path.to_path_buf(), d.clone());
            d
        };
        combined.update(digest.as_bytes());
    }
    Ok(hex::encode(combined.finalize()))
}

fn miss(t: &mut Tape, what: String) -> anyhow::Error {
    let msg = format!("model tape miss: {what}");
    t.problems.push(msg.clone());
    anyhow::anyhow!(msg)
}

fn clip(s: &str) -> String {
    const MAX: usize = 80;
    if s.chars().count() <= MAX {
        s.to_string()
    } else {
        let head: String = s.chars().take(MAX).collect();
        format!("{head}…")
    }
}

/// Embed `text` (the exact text the model would see, after any prefix) through the tape.
///
/// `None` when no tape is in use, and the caller runs the model as usual. Otherwise
/// replay returns the recorded vector or an error, and record runs `live`, which must
/// be the real model and never a degraded fallback, and keeps its result.
pub fn embed(text: &str, live: impl FnOnce() -> Result<Vec<f32>>) -> Option<Result<Vec<f32>>> {
    let tape = state()?;
    if tape.lock().mode == Mode::Replay {
        let mut t = tape.lock();
        return Some(match t.tables.embed.get(text) {
            Some(v) => Ok(v.clone()),
            None => Err(miss(&mut t, format!("embedding of {:?}", clip(text)))),
        });
    }
    Some(
        record_failure(tape, live(), || format!("embedding of {:?}", clip(text)))
            .inspect(|v| record_embed(&mut tape.lock(), text, v)),
    )
}

/// A live model call that fails while recording leaves a hole in the tape, and the
/// caller may well absorb the error into a fallback. So the failure is also a problem.
fn record_failure<T>(
    tape: &Mutex<Tape>,
    result: Result<T>,
    what: impl FnOnce() -> String,
) -> Result<T> {
    result.map_err(|e| {
        tape.lock()
            .problems
            .push(format!("model failed while recording {}: {e:#}", what()));
        e
    })
}

fn record_embed(t: &mut Tape, text: &str, v: &[f32]) {
    match t.tables.embed.get(text) {
        Some(kept) => {
            if kept
                .iter()
                .map(|x| x.to_bits())
                .ne(v.iter().map(|x| x.to_bits()))
            {
                *t.conflicts.entry("embed").or_default() += 1;
            }
        }
        None => {
            t.tables.embed.insert(text.to_string(), v.to_vec());
            t.tables.order.push(Key::Embed(text.to_string()));
        }
    }
}

/// Extract entities from `text` through the tape. `None` when no tape is in use.
pub fn ner(
    text: &str,
    live: impl FnOnce() -> Result<Vec<NerEntity>>,
) -> Option<Result<Vec<NerEntity>>> {
    let tape = state()?;
    if tape.lock().mode == Mode::Replay {
        let mut t = tape.lock();
        return Some(match t.tables.ner.get(text) {
            Some(ents) => ents.iter().map(TapedEntity::to_entity).collect(),
            None => Err(miss(&mut t, format!("NER of {:?}", clip(text)))),
        });
    }
    Some(
        record_failure(tape, live(), || format!("NER of {:?}", clip(text))).inspect(|ents| {
            let taped: Vec<TapedEntity> = ents.iter().map(TapedEntity::from_entity).collect();
            let mut t = tape.lock();
            match t.tables.ner.get(text) {
                Some(kept) => {
                    if *kept != taped {
                        *t.conflicts.entry("ner").or_default() += 1;
                    }
                }
                None => {
                    t.tables.ner.insert(text.to_string(), taped);
                    t.tables.order.push(Key::Ner(text.to_string()));
                }
            }
        }),
    )
}

/// Score (query, candidate) pairs through the tape. `None` when no tape is in use.
///
/// Replay looks each pair up on its own, so the scores do not depend on which other
/// candidates share the batch. A batch with any missing pair fails as a whole.
pub fn cross_encode(
    query: &str,
    docs: &[&str],
    live: impl FnOnce() -> Result<Vec<f32>>,
) -> Option<Result<Vec<f32>>> {
    let tape = state()?;
    if tape.lock().mode == Mode::Replay {
        let mut t = tape.lock();
        let mut out = Vec::with_capacity(docs.len());
        let mut missing = 0usize;
        for doc in docs {
            match t.tables.ce.get(&(query.to_string(), (*doc).to_string())) {
                Some(s) => out.push(*s),
                None => missing += 1,
            }
        }
        return Some(if missing == 0 {
            Ok(out)
        } else {
            Err(miss(
                &mut t,
                format!(
                    "{missing} of {} cross-encoder pairs for query {:?}",
                    docs.len(),
                    clip(query)
                ),
            ))
        });
    }
    let what = || {
        format!(
            "{} cross-encoder pairs for query {:?}",
            docs.len(),
            clip(query)
        )
    };
    Some(record_failure(tape, live(), what).inspect(|scores| {
        let mut t = tape.lock();
        for (doc, s) in docs.iter().zip(scores) {
            let key = (query.to_string(), (*doc).to_string());
            match t.tables.ce.get(&key) {
                Some(kept) => {
                    if kept.to_bits() != s.to_bits() {
                        *t.conflicts.entry("ce").or_default() += 1;
                    }
                }
                None => {
                    t.tables.ce.insert(key.clone(), *s);
                    t.tables.order.push(Key::Ce(key.0, key.1));
                }
            }
        }
    }))
}

/// Whether the tape already holds a cross-encoder score for this pair.
pub fn has_cross_encoding(query: &str, doc: &str) -> bool {
    state().is_some_and(|t| {
        t.lock()
            .tables
            .ce
            .contains_key(&(query.to_string(), doc.to_string()))
    })
}

/// Every distinct cross-encoder query in the tape, in first-recorded order.
pub fn cross_encoder_queries() -> Vec<String> {
    let Some(tape) = state() else {
        return Vec::new();
    };
    let t = tape.lock();
    let mut seen = std::collections::HashSet::new();
    t.tables
        .order
        .iter()
        .filter_map(|k| match k {
            Key::Ce(q, _) if seen.insert(q.clone()) => Some(q.clone()),
            _ => None,
        })
        .collect()
}

/// Every distinct cross-encoder candidate in the tape, in first-recorded order.
pub fn cross_encoder_docs() -> Vec<String> {
    let Some(tape) = state() else {
        return Vec::new();
    };
    let t = tape.lock();
    let mut seen = std::collections::HashSet::new();
    t.tables
        .order
        .iter()
        .filter_map(|k| match k {
            Key::Ce(_, d) if seen.insert(d.clone()) => Some(d.clone()),
            _ => None,
        })
        .collect()
}

/// Number of entries recorded so far. A harness compares it across repeats: a repeat
/// that adds entries saw inputs an earlier one did not, so coverage is not stable.
pub fn entry_count() -> usize {
    state().map_or(0, |t| t.lock().tables.order.len())
}

/// On replay, add a problem for each model whose files were never checked against the
/// tape header. A model that never loaded cannot prove the tape was recorded from the
/// version this build uses. A no-op when recording or with no tape.
pub fn require_verified_models() {
    let Some(tape) = state() else { return };
    let mut t = tape.lock();
    if t.mode != Mode::Replay {
        return;
    }
    for model in [Model::Embedder, Model::Ner, Model::CrossEncoder] {
        if !t.verified.contains(model.as_str()) {
            let msg = format!(
                "{}: this process never loaded the model, so the tape's record of which                  model produced it was not checked",
                model.as_str()
            );
            t.problems.push(msg);
        }
    }
}

/// Misses, identity mismatches and load failures since the last call. A measurement
/// taken with a non-empty list did not run on the tape it claims to, and must be
/// refused.
pub fn take_problems() -> Vec<String> {
    state().map_or_else(Vec::new, |t| std::mem::take(&mut t.lock().problems))
}

/// Conflicts seen while recording, by table.
pub fn conflicts() -> HashMap<&'static str, usize> {
    state().map_or_else(HashMap::new, |t| t.lock().conflicts.clone())
}

/// Write the recorded tape to the path it was opened with. `provenance` goes into the
/// header as-is. Returns the path and the entry count.
pub fn write_recording(provenance: serde_json::Value) -> Result<(PathBuf, usize)> {
    let Some(tape) = state() else {
        bail!("no model tape is in use");
    };
    let t = tape.lock();
    if t.mode != Mode::Record {
        bail!("the model tape is replaying, not recording");
    }
    for model in [Model::Embedder, Model::Ner, Model::CrossEncoder] {
        if !t.models.contains_key(model.as_str()) {
            bail!(
                "no {} was registered while recording, so the tape cannot say which \
                 model produced its outputs",
                model.as_str()
            );
        }
    }
    let file =
        std::fs::File::create(&t.path).with_context(|| format!("creating {}", t.path.display()))?;
    let gz = flate2::write::GzEncoder::new(BufWriter::new(file), flate2::Compression::default());
    let mut w = BufWriter::new(gz);
    let header = Line::Header {
        version: FORMAT_VERSION,
        models: t.models.clone(),
        provenance,
    };
    writeln!(w, "{}", serde_json::to_string(&header)?)?;
    for key in &t.tables.order {
        let line = match key {
            Key::Embed(text) => Line::Embed {
                text: text.clone(),
                bits: t.tables.embed[text].iter().map(|x| x.to_bits()).collect(),
            },
            Key::Ner(text) => Line::Ner {
                text: text.clone(),
                entities: t.tables.ner[text].clone(),
            },
            Key::Ce(q, d) => Line::Ce {
                query: q.clone(),
                doc: d.clone(),
                bits: t.tables.ce[&(q.clone(), d.clone())].to_bits(),
            },
        };
        writeln!(w, "{}", serde_json::to_string(&line)?)?;
    }
    w.into_inner()
        .map_err(|e| anyhow::anyhow!("flushing model tape: {e}"))?
        .finish()?
        .flush()?;
    Ok((t.path.clone(), t.tables.order.len()))
}

fn read_tape(path: &Path) -> Result<(HashMap<String, String>, Tables)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening model tape {}", path.display()))?;
    let reader = BufReader::new(flate2::read::GzDecoder::new(file));
    let mut lines = reader.lines();
    let first = lines
        .next()
        .context("model tape is empty")?
        .context("reading model tape header")?;
    let models = match serde_json::from_str::<Line>(&first).context("parsing model tape header")? {
        Line::Header {
            version, models, ..
        } => {
            if version != FORMAT_VERSION {
                bail!("model tape format {version}; this build reads {FORMAT_VERSION}");
            }
            models
        }
        _ => bail!("model tape does not start with a header"),
    };
    let mut tables = Tables::default();
    for (n, line) in lines.enumerate() {
        let line = line.with_context(|| format!("reading model tape line {}", n + 2))?;
        match serde_json::from_str::<Line>(&line)
            .with_context(|| format!("parsing model tape line {}", n + 2))?
        {
            Line::Header { .. } => bail!("model tape has a second header at line {}", n + 2),
            Line::Embed { text, bits } => {
                tables
                    .embed
                    .insert(text, bits.into_iter().map(f32::from_bits).collect());
            }
            Line::Ner { text, entities } => {
                tables.ner.insert(text, entities);
            }
            Line::Ce { query, doc, bits } => {
                tables.ce.insert((query, doc), f32::from_bits(bits));
            }
        }
    }
    Ok((models, tables))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tape's own state is process-global and driven by env, so these tests
    /// exercise the pieces that do not need it: entity round-trips, file format, and
    /// the read path's refusals.
    #[test]
    fn entity_round_trips_through_its_taped_form() {
        let e = NerEntity {
            text: "Nate".to_string(),
            entity_type: NerEntityType::Person,
            confidence: 0.307_654_3,
            start: 12,
            end: 16,
            fine_label: Some("diplomat".to_string()),
        };
        let back = TapedEntity::from_entity(&e).to_entity().unwrap();
        assert_eq!(back.text, e.text);
        assert_eq!(back.entity_type, e.entity_type);
        assert_eq!(back.confidence.to_bits(), e.confidence.to_bits());
        assert_eq!((back.start, back.end), (e.start, e.end));
        assert_eq!(back.fine_label, e.fine_label);
    }

    fn write_lines(lines: &[Line]) -> tempfile::NamedTempFile {
        let f = tempfile::NamedTempFile::new().unwrap();
        let gz = flate2::write::GzEncoder::new(
            std::fs::File::create(f.path()).unwrap(),
            flate2::Compression::fast(),
        );
        let mut w = BufWriter::new(gz);
        for l in lines {
            writeln!(w, "{}", serde_json::to_string(l).unwrap()).unwrap();
        }
        w.into_inner().unwrap().finish().unwrap();
        f
    }

    fn header(version: u32) -> Line {
        Line::Header {
            version,
            models: HashMap::from([("embedder".to_string(), "abc".to_string())]),
            provenance: serde_json::json!({}),
        }
    }

    #[test]
    fn a_written_tape_reads_back_bit_for_bit() {
        let v = [0.1f32, -2.5e-8, f32::MIN_POSITIVE];
        let f = write_lines(&[
            header(FORMAT_VERSION),
            Line::Embed {
                text: "Nate invited Joanna.".to_string(),
                bits: v.iter().map(|x| x.to_bits()).collect(),
            },
            Line::Ce {
                query: "who did Nate invite?".to_string(),
                doc: "Nate invited Joanna.".to_string(),
                bits: (-1.630_57f32).to_bits(),
            },
        ]);
        let (models, t) = read_tape(f.path()).unwrap();
        assert_eq!(models["embedder"], "abc");
        let got = &t.embed["Nate invited Joanna."];
        assert!(got.iter().zip(&v).all(|(a, b)| a.to_bits() == b.to_bits()));
        let key = (
            "who did Nate invite?".to_string(),
            "Nate invited Joanna.".to_string(),
        );
        assert_eq!(t.ce[&key].to_bits(), (-1.630_57f32).to_bits());
    }

    #[test]
    fn a_tape_from_another_format_version_is_refused() {
        let f = write_lines(&[header(FORMAT_VERSION + 1)]);
        let err = read_tape(f.path()).unwrap_err().to_string();
        assert!(err.contains("format"), "{err}");
    }

    #[test]
    fn a_tape_without_a_header_is_refused() {
        let f = write_lines(&[Line::Embed {
            text: "x".to_string(),
            bits: vec![0],
        }]);
        assert!(read_tape(f.path()).is_err());
    }

    #[test]
    fn an_unknown_entity_type_is_an_error_not_a_default() {
        let t = TapedEntity {
            text: "x".to_string(),
            entity_type: "DATE".to_string(),
            confidence: 0,
            start: 0,
            end: 1,
            fine_label: None,
        };
        assert!(t.to_entity().is_err());
    }
}
