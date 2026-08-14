//! Token-aware, structure-aware text chunking for embeddings.
//!
//! # Why token-aware
//!
//! The embedding window is enforced by the TOKENIZER, not by this module: the
//! shipped MiniLM tokenizer.json truncates every sequence to
//! [`MODEL_TOKEN_WINDOW`] (128) tokens including `[CLS]`/`[SEP]`. The previous
//! chunker measured CHARACTERS and targeted ~800 chars ("~200 tokens"), so
//! every full-size chunk exceeded the real window and silently lost its last
//! ~35% of tokens before the model ever saw them. Chunk budgets are therefore
//! expressed in REAL tokens, counted by the same tokenizer that feeds the
//! model (`Embedder::count_tokens`), never by a chars/4 heuristic.
//!
//! # Strategy: structural segmentation + token-budget packing
//!
//! 1. Segment on structure — dialogue turns (never split mid-turn), then
//!    paragraphs, then sentences/lines. Splitting mid-sentence measurably
//!    degrades sentence-embedding quality; boundaries follow the text's own
//!    units.
//! 2. Greedily pack segments into chunks whose full sequence length
//!    (content + special tokens) fits `ChunkConfig::max_tokens`.
//! 3. Overlap is a single trailing sentence, carried only across SOFT
//!    (sentence) boundaries and only when it is at most
//!    `ChunkConfig::overlap_tokens`. Rationale: with sentence-aligned
//!    boundaries no fact is severed mid-clause, so the only residual boundary
//!    loss is anaphora ("it", "she") whose antecedent is the previous
//!    sentence. One sentence of carry-over restores that context; the old
//!    fixed 25% char overlap inflated vector count without a measured recall
//!    gain. Hard boundaries (paragraph / dialogue turn) carry no overlap —
//!    those units are self-contained by construction.
//! 4. A verification pass re-counts every emitted chunk with the real counter
//!    and hard-splits any that still exceed the budget (defence against
//!    non-additive tokenization edge cases), so no emitted chunk can exceed
//!    the window.
//!
//! # Rejected alternatives (surveyed 2026-08)
//!
//! - **Late chunking** (embed long context once, pool per chunk): requires a
//!   long-context token-embedding model; degenerate at a 128-token window.
//! - **Embedding-distance semantic breakpoints**: needs per-sentence
//!   embeddings at ingest (~Nx embed cost). At this window a memory yields
//!   1–3 chunks and a breakpoint can move a boundary by at most a sentence —
//!   the cost is not earned. Revisit only with an ingest-side ablation.
//! - **Long-context embedder swap**: out of scope; our own bake-off found the
//!   embedder is not the lever (one comparison bit-identical).
//! - **Parent-document retrieval**: already present — chunk vectors map to the
//!   parent memory id (`IdMapping::insert_chunks`), and retrieval returns the
//!   whole memory.

use regex::Regex;
use std::sync::LazyLock;

/// The real encoder sequence window, in tokens, INCLUDING the `[CLS]`/`[SEP]`
/// special tokens. This is the single source of truth that the chunk budget,
/// the tokenizer truncation (enforced at model load), and the ONNX tensor
/// length are all validated against at startup
/// (`MiniLMEmbedder::validate_sequence_contract`). The shipped MiniLM
/// tokenizer.json truncates at exactly this length; content past it never
/// reaches the model.
pub const MODEL_TOKEN_WINDOW: usize = 128;

/// Tokens consumed by `[CLS]` + `[SEP]` in a single-sequence BERT encode.
pub const SPECIAL_TOKEN_OVERHEAD: usize = 2;

/// Token counter contract: returns the FULL encoded sequence length of `text`
/// as the model would see it — including special tokens, WITHOUT truncation.
/// `MiniLMEmbedder` implements this with the real tokenizer; the default
/// trait fallback uses the calibrated heuristic in `token_estimation`.
pub type TokenCounter<'a> = dyn Fn(&str) -> usize + 'a;

/// Pattern to detect dialogue turns (e.g., "Alice:", "User:", "Speaker 1:")
static DIALOGUE_TURN_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^([A-Z][a-zA-Z0-9_\- ]{0,30})\s*:").unwrap());

/// Sentence terminator followed by whitespace (or end): `.`, `!`, `?` runs,
/// optionally closed by quotes/brackets.
static SENTENCE_END_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"[.!?]+["')\]]*(?:\s+|$)"#).unwrap());

/// Chunk configuration, in REAL tokens (full sequence, including special
/// tokens — the same unit the encoder window is measured in).
pub struct ChunkConfig {
    /// Maximum full-sequence tokens per chunk (content + special tokens).
    /// MUST be ≤ the tokenizer truncation window or content is silently lost;
    /// this invariant is asserted at embedder construction.
    pub max_tokens: usize,
    /// Maximum CONTENT tokens the trailing sentence of a chunk may have to be
    /// carried into the next chunk as overlap.
    pub overlap_tokens: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_tokens: MODEL_TOKEN_WINDOW,
            overlap_tokens: 24,
        }
    }
}

impl ChunkConfig {
    /// Config for an embedder-specific budget (e.g. reduced by an asymmetric
    /// document instruction prefix). Clamped to a sane floor so a
    /// misconfigured prefix can never produce degenerate one-word chunks.
    pub fn for_budget(budget_tokens: usize) -> Self {
        Self {
            max_tokens: budget_tokens.max(32),
            ..Self::default()
        }
    }
}

/// Result of chunking a text
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// The chunked text segments
    pub chunks: Vec<String>,
    /// Original text length in chars
    pub original_length: usize,
    /// Whether the text was split into more than one chunk
    pub was_chunked: bool,
}

/// How a segment attaches to the segment before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Boundary {
    /// Paragraph or dialogue-turn start: self-contained unit, no overlap
    /// carried across it, joined with a newline inside a chunk.
    Hard,
    /// Sentence/line continuation: overlap may be carried, joined with a
    /// space inside a chunk.
    Soft,
}

/// A structural unit with its pre-counted CONTENT token length.
struct Unit {
    text: String,
    boundary: Boundary,
    /// Content tokens (full count minus special-token overhead).
    tokens: usize,
}

/// Chunk `text` into segments that each fit the token budget.
///
/// `counter` must implement the [`TokenCounter`] contract (full sequence
/// length including specials, no truncation). Every returned chunk satisfies
/// `counter(chunk) <= config.max_tokens`.
pub fn chunk_text(text: &str, config: &ChunkConfig, counter: &TokenCounter) -> ChunkResult {
    let text = text.trim();
    let original_length = text.len();

    // Fast path: the whole text fits the window.
    if counter(text) <= config.max_tokens {
        return ChunkResult {
            chunks: vec![text.to_string()],
            original_length,
            was_chunked: false,
        };
    }

    let content_budget = config.max_tokens.saturating_sub(SPECIAL_TOKEN_OVERHEAD);
    let units = split_units(text, content_budget, counter);
    let mut chunks = pack_units(units, config, content_budget);

    // Verification pass: re-count each emitted chunk with the REAL counter.
    // Wordpiece token counts are additive across whitespace joins for BERT
    // tokenizers, but this module must hold its guarantee for ANY counter, so
    // any chunk that still exceeds the budget is hard-split here.
    let mut verified = Vec::with_capacity(chunks.len());
    for chunk in chunks.drain(..) {
        if counter(&chunk) > config.max_tokens {
            verified.extend(hard_split(&chunk, content_budget, counter));
        } else {
            verified.push(chunk);
        }
    }

    let was_chunked = verified.len() > 1;
    ChunkResult {
        chunks: verified,
        original_length,
        was_chunked,
    }
}

/// Detect if text appears to be dialogue/conversation format
pub fn is_dialogue_format(text: &str) -> bool {
    DIALOGUE_TURN_PATTERN.is_match(text)
}

/// Split text into structural units: dialogue turns > paragraphs > sentences.
/// Any single unit larger than `content_budget` is word-split so every unit
/// fits the budget on its own.
fn split_units(text: &str, content_budget: usize, counter: &TokenCounter) -> Vec<Unit> {
    let mut units = Vec::new();

    // Top-level spans: dialogue turns when the text is a conversation,
    // otherwise paragraphs (blank-line separated).
    let spans: Vec<&str> = if is_dialogue_format(text) {
        let starts: Vec<usize> = DIALOGUE_TURN_PATTERN
            .find_iter(text)
            .map(|m| m.start())
            .collect();
        let mut spans = Vec::with_capacity(starts.len() + 1);
        if let Some(&first) = starts.first() {
            if first > 0 {
                spans.push(&text[..first]);
            }
        }
        for (i, &start) in starts.iter().enumerate() {
            let end = starts.get(i + 1).copied().unwrap_or(text.len());
            spans.push(&text[start..end]);
        }
        if spans.is_empty() {
            spans.push(text);
        }
        spans
    } else {
        static PARAGRAPH_PATTERN: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"\n\s*\n").unwrap());
        let mut spans = Vec::new();
        let mut last = 0;
        for m in PARAGRAPH_PATTERN.find_iter(text) {
            if m.start() > last {
                spans.push(&text[last..m.start()]);
            }
            last = m.end();
        }
        if last < text.len() {
            spans.push(&text[last..]);
        }
        if spans.is_empty() {
            spans.push(text);
        }
        spans
    };

    for span in spans {
        let span = span.trim();
        if span.is_empty() {
            continue;
        }
        let mut first_in_span = true;
        for sentence in split_sentences(span) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let boundary = if first_in_span {
                Boundary::Hard
            } else {
                Boundary::Soft
            };
            first_in_span = false;

            let tokens = counter(sentence).saturating_sub(SPECIAL_TOKEN_OVERHEAD);
            if tokens > content_budget {
                // Oversized single sentence (rare: log lines, minified blobs).
                for (i, piece) in hard_split(sentence, content_budget, counter)
                    .into_iter()
                    .enumerate()
                {
                    let tokens = counter(&piece).saturating_sub(SPECIAL_TOKEN_OVERHEAD);
                    units.push(Unit {
                        text: piece,
                        boundary: if i == 0 { boundary } else { Boundary::Soft },
                        tokens,
                    });
                }
            } else {
                units.push(Unit {
                    text: sentence.to_string(),
                    boundary,
                    tokens,
                });
            }
        }
    }

    units
}

/// Split a span into sentences; single newlines also terminate a unit so
/// logs / lists / code lines segment on their natural lines.
fn split_sentences(span: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for line in span.split('\n') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut last = 0;
        for m in SENTENCE_END_PATTERN.find_iter(line) {
            out.push(&line[last..m.end()]);
            last = m.end();
        }
        if last < line.len() {
            out.push(&line[last..]);
        }
    }
    out
}

/// Greedily pack units into chunks that fit the content budget, carrying a
/// single small trailing sentence as overlap across soft boundaries.
fn pack_units(units: Vec<Unit>, config: &ChunkConfig, content_budget: usize) -> Vec<String> {
    let mut chunks: Vec<(String, usize)> = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;
    // Trailing unit of `current`, kept for overlap carry-over.
    let mut last_unit: Option<(String, usize)> = None;

    for unit in units {
        let fits = current.is_empty() || current_tokens + unit.tokens <= content_budget;
        if !fits {
            // Close the current chunk and start the next one. The carried
            // sentence must leave room for the unit that forced the split:
            // without this check a 24-token overlap in front of a
            // budget-sized unit overflows the window, and the verification
            // pass would have to rescue it with a WORD-level hard split —
            // destroying exactly the sentence boundaries this module exists
            // to preserve. When it does not fit, correctness wins: drop the
            // overlap, keep the boundary.
            let overlap = if unit.boundary == Boundary::Soft {
                last_unit.take().filter(|(_, t)| {
                    *t <= config.overlap_tokens && *t + unit.tokens <= content_budget
                })
            } else {
                None
            };
            chunks.push((std::mem::take(&mut current), current_tokens));
            current_tokens = 0;
            if let Some((text, tokens)) = overlap {
                current.push_str(&text);
                current_tokens = tokens;
            }
        }

        if !current.is_empty() {
            current.push(if unit.boundary == Boundary::Hard {
                '\n'
            } else {
                ' '
            });
        }
        current.push_str(&unit.text);
        current_tokens += unit.tokens;
        last_unit = Some((unit.text, unit.tokens));
    }

    if !current.is_empty() {
        // NOTE: no minimum-size merge. Greedy packing guarantees a trailing
        // fragment only becomes its own chunk when the previous chunk had no
        // room for it — so any merge would overflow the budget and lose
        // tokens. A small final chunk is correct; an overflowing one is not.
        chunks.push((current, current_tokens));
    }

    chunks.into_iter().map(|(text, _)| text).collect()
}

/// Split text that exceeds the content budget at word boundaries, greedily
/// packing words up to the budget. A single word larger than the budget
/// (pathological: base64/minified blobs) is bisected at char boundaries.
fn hard_split(text: &str, content_budget: usize, counter: &TokenCounter) -> Vec<String> {
    let budget = content_budget.max(1);
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0usize;

    let mut queue: std::collections::VecDeque<String> =
        text.split_whitespace().map(str::to_string).collect();

    while let Some(word) = queue.pop_front() {
        let word_tokens = counter(&word).saturating_sub(SPECIAL_TOKEN_OVERHEAD);
        if word_tokens > budget {
            // Bisect the oversized word at a char boundary and re-queue.
            let mid = word.len() / 2;
            let mut split_at = mid;
            while split_at > 0 && !word.is_char_boundary(split_at) {
                split_at -= 1;
            }
            if split_at == 0 || split_at == word.len() {
                // Cannot split further; emit as-is (tokenizer truncation is
                // the final backstop for a single indivisible token run).
                if !current.is_empty() {
                    out.push(std::mem::take(&mut current));
                    current_tokens = 0;
                }
                out.push(word);
                continue;
            }
            let (a, b) = word.split_at(split_at);
            queue.push_front(b.to_string());
            queue.push_front(a.to_string());
            continue;
        }

        if !current.is_empty() && current_tokens + word_tokens > budget {
            out.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&word);
        current_tokens += word_tokens;
    }

    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        out.push(text.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heuristic counter matching the `Embedder::count_tokens` default:
    /// whitespace-word based, plus special-token overhead. Deterministic and
    /// additive across whitespace joins, like real wordpiece counts.
    fn word_counter(text: &str) -> usize {
        text.split_whitespace().count() + SPECIAL_TOKEN_OVERHEAD
    }

    fn cfg(max_tokens: usize, overlap: usize) -> ChunkConfig {
        ChunkConfig {
            max_tokens,
            overlap_tokens: overlap,
        }
    }

    #[test]
    fn chunk_budget_never_exceeds_model_window() {
        // The invariant the startup validation enforces; pinned here so a
        // config drift fails the unit suite too, not only server startup.
        assert!(ChunkConfig::default().max_tokens <= MODEL_TOKEN_WINDOW);
    }

    #[test]
    fn short_text_single_chunk() {
        let result = chunk_text(
            "This is a short text.",
            &ChunkConfig::default(),
            &word_counter,
        );
        assert_eq!(result.chunks.len(), 1);
        assert!(!result.was_chunked);
        assert_eq!(result.chunks[0], "This is a short text.");
    }

    /// Regression: a small trailing sentence carried as overlap must never be
    /// prepended to a unit that already fills the budget. Before the fits
    /// check in `pack_units`, `overlap + unit` overflowed the window and the
    /// verification pass rescued it with a WORD-level `hard_split`, cutting
    /// mid-sentence — the one thing structural chunking must not do.
    #[test]
    fn overlap_never_overflows_and_never_forces_a_word_split() {
        // max_tokens 22 => content budget 20; overlap allowance 8.
        let config = cfg(22, 8);
        // Three sentences: 12-token filler, a 4-token carry candidate, then a
        // sentence that alone exactly fills the 20-token content budget.
        // Carrying the 4-token sentence in front of it would need 24.
        let filler = (1..=12)
            .map(|i| format!("f{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let small = "Alpha beta gamma delta.";
        let big = (1..=20)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = format!("{filler}. {small} {big}.");
        let result = chunk_text(&text, &config, &word_counter);

        for chunk in &result.chunks {
            assert!(
                word_counter(chunk) <= config.max_tokens,
                "chunk exceeds budget: {} tokens: {chunk:?}",
                word_counter(chunk)
            );
        }
        // The oversized sentence must survive intact in ONE chunk — proof no
        // word-level hard split was needed to rescue an overlap overflow.
        assert!(
            result
                .chunks
                .iter()
                .any(|c| c.contains("w1 ") && c.contains("w20")),
            "the budget-filling sentence was split apart: {:?}",
            result.chunks
        );
    }

    #[test]
    fn every_chunk_fits_token_budget() {
        let config = cfg(30, 8);
        let text = (1..=40)
            .map(|i| format!("Sentence number {i} contains unique information."))
            .collect::<Vec<_>>()
            .join(" ");
        let result = chunk_text(&text, &config, &word_counter);
        assert!(result.was_chunked);
        for chunk in &result.chunks {
            assert!(
                word_counter(chunk) <= config.max_tokens,
                "chunk exceeds token budget: {} tokens: {chunk:?}",
                word_counter(chunk)
            );
        }
    }

    #[test]
    fn coverage_no_content_lost() {
        let config = cfg(30, 8);
        let text = (1..=40)
            .map(|i| format!("Sentence number {i} contains unique information."))
            .collect::<Vec<_>>()
            .join(" ");
        let result = chunk_text(&text, &config, &word_counter);
        for i in 1..=40 {
            let marker = format!("number {i} ");
            let marker_end = format!("number {i} contains");
            let found = result
                .chunks
                .iter()
                .any(|c| c.contains(&marker) || c.contains(&marker_end));
            assert!(found, "sentence {i} missing from all chunks");
        }
    }

    #[test]
    fn unique_markers_beginning_middle_end_searchable() {
        let config = cfg(40, 8);
        let beginning = "ALPHA_BEGINNING_MARKER is a unique identifier at the start.";
        let filler_a = "This is filler content to push things apart. ".repeat(20);
        let middle = "BETA_MIDDLE_MARKER represents content in the center of the document.";
        let filler_b = "More filler content for separation between sections. ".repeat(20);
        let end = "GAMMA_END_MARKER signifies the conclusion of this memory content.";
        let text = format!("{beginning} {filler_a} {middle} {filler_b} {end}");

        let result = chunk_text(&text, &config, &word_counter);
        assert!(result.was_chunked);
        assert!(result.chunks.iter().any(|c| c.contains("ALPHA_BEGINNING")));
        assert!(result.chunks.iter().any(|c| c.contains("BETA_MIDDLE")));
        assert!(result.chunks.iter().any(|c| c.contains("GAMMA_END")));
    }

    #[test]
    fn sentence_boundaries_respected() {
        let config = cfg(12, 2);
        let text = "First sentence here. Second sentence follows on. Third sentence ends it. \
                    Fourth sentence too. Fifth sentence closes.";
        let result = chunk_text(&text, &config, &word_counter);
        assert!(result.chunks.len() > 1);
        for chunk in &result.chunks[..result.chunks.len() - 1] {
            let last = chunk.trim_end().chars().last().unwrap();
            assert!(
                last == '.' || last == '!' || last == '?',
                "chunk does not end at a sentence boundary: {chunk:?}"
            );
        }
    }

    #[test]
    fn overlap_carries_small_trailing_sentence() {
        let config = cfg(20, 10);
        // Sentences of 8 words each: two fit per 18-content-token chunk; the
        // trailing sentence (8 tokens ≤ overlap 10) must be carried over.
        let text = (1..=6)
            .map(|i| format!("Overlap test sentence {i} has exactly eight words."))
            .collect::<Vec<_>>()
            .join(" ");
        let result = chunk_text(&text, &config, &word_counter);
        assert!(result.chunks.len() >= 2);
        for w in result.chunks.windows(2) {
            let last_sentence = w[0].split(". ").last().unwrap_or("").trim();
            assert!(
                w[1].contains(last_sentence.trim_end_matches('.')),
                "no overlap between consecutive chunks:\n prev: {:?}\n next: {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn dialogue_turns_not_split_when_they_fit() {
        let config = cfg(25, 4);
        let text = "Alice: I went to the market this morning and bought fresh vegetables.\n\
                    Bob: That sounds great, did you find good tomatoes there?\n\
                    Alice: Yes, and I also picked up some basil for the sauce.\n\
                    Bob: Perfect, let us cook dinner together tonight then.";
        let result = chunk_text(text, &config, &word_counter);
        // Every turn is <= budget, so no chunk may contain a partial turn:
        // each speaker prefix in a chunk starts at a line boundary.
        for chunk in &result.chunks {
            for (i, _) in chunk
                .match_indices("Alice:")
                .chain(chunk.match_indices("Bob:"))
            {
                assert!(
                    i == 0 || chunk.as_bytes()[i - 1] == b'\n',
                    "turn split mid-chunk: {chunk:?}"
                );
            }
        }
    }

    #[test]
    fn oversized_single_sentence_is_word_split() {
        let config = cfg(12, 2);
        let text = "word ".repeat(100); // one 100-word "sentence", no terminator
        let result = chunk_text(text.trim(), &config, &word_counter);
        assert!(result.chunks.len() >= 10);
        for chunk in &result.chunks {
            assert!(word_counter(chunk) <= config.max_tokens);
        }
        let total_words: usize = result
            .chunks
            .iter()
            .map(|c| c.split_whitespace().count())
            .sum();
        assert_eq!(total_words, 100, "hard split lost words");
    }

    #[test]
    fn pathological_single_token_run_is_bisected() {
        let config = cfg(10, 2);
        // Counter that charges 1 token per 4 chars — a 400-char unbroken blob.
        let char_counter = |t: &str| t.chars().count().div_ceil(4) + SPECIAL_TOKEN_OVERHEAD;
        let blob = "x".repeat(400);
        let result = chunk_text(&blob, &config, &char_counter);
        for chunk in &result.chunks {
            assert!(char_counter(chunk) <= config.max_tokens);
        }
        let total: usize = result.chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, 400, "bisection lost characters");
    }

    #[test]
    fn trailing_fragment_never_lost_and_never_overflows() {
        let config = cfg(20, 4);
        // Two 9-word sentences fill a chunk to its 18-content-token budget;
        // the 3-word tail cannot merge without overflowing, so it must stand
        // alone rather than be dropped or force a lossy merge.
        let text = "This first sentence is exactly nine words long okay.                     This second sentence is also exactly nine words long. Tiny tail here";
        let result = chunk_text(text, &config, &word_counter);
        assert!(
            result.chunks.iter().any(|c| c.contains("Tiny tail here")),
            "trailing fragment lost"
        );
        for chunk in &result.chunks {
            assert!(
                word_counter(chunk) <= config.max_tokens,
                "chunk exceeds budget: {chunk:?}"
            );
        }
    }

    #[test]
    fn empty_text_single_empty_chunk() {
        let result = chunk_text("", &ChunkConfig::default(), &word_counter);
        assert_eq!(result.chunks.len(), 1);
        assert!(!result.was_chunked);
    }
}
