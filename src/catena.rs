//! CATENA-lite — the EVENT arm of the causal spine (event→event causation).
//!
//! The counterpart to the entity-relation arms (GLiREL, OpenIE), which cannot
//! represent inchoative pivots (`collapse`, `blackout`, `power-loss`) because those
//! have no agent-patient entity pair. Following CATENA (Causal And Temporal
//! relation Extraction), it extracts **event triggers** (main verbs + deverbal-noun
//! events) and links event pairs with **causal/temporal SIGNALS**, orienting the
//! arrow by signal direction, with temporal precedence (cause before effect) as the
//! fallback clock. This is the arm that produced the only clean narrative chain:
//! `failure → collision → collapse → rescue`.
//!
//! Model-free over an already-parsed token list ([`crate::dep_parser`] supplies POS
//! + lemma); sentences are segmented on terminal punctuation and signals matched in
//! token space, so no character offsets are needed.

use crate::causal_vocab::{self, LinkRelation, SignalDir};
use crate::dep_parser::{self, ParsedToken};

/// An event trigger within a sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct EventTrigger {
    /// Event lemma (lowercased).
    pub lemma: String,
    /// Token index within the parsed text.
    pub index: usize,
    /// `"verb"` or `"nominal"` (a deverbal-noun event).
    pub kind: &'static str,
}

/// A directed event→event link — either causal (`Causes`) or temporal
/// (`Precedes`). Kept distinct so temporal sequence is never reported as causation.
#[derive(Debug, Clone, PartialEq)]
pub struct EventLink {
    /// Head event lemma — the cause (if `Causes`) or the earlier event (`Precedes`).
    pub source: String,
    /// Tail event lemma — the effect, or the later event.
    pub target: String,
    /// Whether the link is causal or purely temporal.
    pub relation: LinkRelation,
    /// The signal marker that licensed and oriented the link.
    pub signal: String,
}

/// Split a token list into sentences on terminal punctuation (`.`/`!`/`?`).
fn sentences(tokens: &[ParsedToken]) -> Vec<&[ParsedToken]> {
    let mut sents = Vec::new();
    let mut start = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        if matches!(t.text.as_str(), "." | "!" | "?") {
            if i + 1 > start {
                sents.push(&tokens[start..=i]);
            }
            start = i + 1;
        }
    }
    if start < tokens.len() {
        sents.push(&tokens[start..]);
    }
    sents
}

/// Event triggers in a sentence: main verbs (minus light verbs) + deverbal-noun
/// events. The index is the token's index within the *sentence slice*.
pub fn event_triggers(sent: &[ParsedToken]) -> Vec<EventTrigger> {
    let mut evs = Vec::new();
    for (i, t) in sent.iter().enumerate() {
        let lemma = t.lemma.to_lowercase();
        if t.pos == "VERB" && !causal_vocab::is_light_verb(&lemma) {
            evs.push(EventTrigger {
                lemma,
                index: i,
                kind: "verb",
            });
        } else if t.pos == "NOUN" && causal_vocab::is_nominal_event(&lemma) {
            evs.push(EventTrigger {
                lemma,
                index: i,
                kind: "nominal",
            });
        }
    }
    evs
}

/// Signal occurrences: `(direction, relation, start_token, end_token, marker)`.
/// Matched in token space, longest-first (the signal list is length-ordered), so
/// `as a result of` beats its prefix `as a result` and the arrow can't invert.
fn signal_hits(sent: &[ParsedToken]) -> Vec<(SignalDir, LinkRelation, usize, usize, String)> {
    let lower: Vec<String> = sent.iter().map(|t| t.text.to_lowercase()).collect();
    let mut hits = Vec::new();
    let mut i = 0usize;
    while i < lower.len() {
        let mut matched = false;
        for (marker, dir, relation) in causal_vocab::signals() {
            let words: Vec<&str> = marker.split(' ').collect();
            if i + words.len() <= lower.len()
                && words.iter().enumerate().all(|(k, w)| lower[i + k] == *w)
            {
                hits.push((*dir, *relation, i, i + words.len(), (*marker).to_string()));
                i += words.len();
                matched = true;
                break;
            }
        }
        if !matched {
            i += 1;
        }
    }
    hits
}

fn nearest_left(evs: &[EventTrigger], before: usize) -> Option<&EventTrigger> {
    evs.iter()
        .filter(|e| e.index < before)
        .max_by_key(|e| e.index)
}
fn nearest_right(evs: &[EventTrigger], at_or_after: usize) -> Option<&EventTrigger> {
    evs.iter()
        .filter(|e| e.index >= at_or_after)
        .min_by_key(|e| e.index)
}

/// Link the events in one sentence via its causal/temporal signals. Pure — no model.
pub fn link_events_in_sentence(sent: &[ParsedToken]) -> Vec<EventLink> {
    let evs = event_triggers(sent);
    if evs.len() < 2 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (dir, relation, start, end, marker) in signal_hits(sent) {
        let (Some(left), Some(right)) = (nearest_left(&evs, start), nearest_right(&evs, end))
        else {
            continue;
        };
        if left.lemma == right.lemma {
            continue;
        }
        // The head is the preceding clause (Forward) or the following one
        // (Backward) — the cause for a causal signal, the earlier event for a
        // temporal one. The relation (`Causes`/`Precedes`) comes from the signal.
        let (source, target) = match dir {
            SignalDir::Forward => (&left.lemma, &right.lemma),
            SignalDir::Backward => (&right.lemma, &left.lemma),
        };
        out.push(EventLink {
            source: source.clone(),
            target: target.clone(),
            relation,
            signal: marker,
        });
    }
    out
}

/// Extract event→event links (causal + temporal) across all sentences,
/// de-duplicated by (source, signal, target). Pure — no model.
pub fn links_from_tokens(tokens: &[ParsedToken]) -> Vec<EventLink> {
    let mut out: Vec<EventLink> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let sents = sentences(tokens);

    for sent in &sents {
        for link in link_events_in_sentence(sent) {
            let key = (
                link.source.clone(),
                link.signal.clone(),
                link.target.clone(),
            );
            if seen.insert(key) {
                out.push(link);
            }
        }
    }

    // INTER-SENTENTIAL pass. Everything above is bounded by a sentence, which
    // silently loses the dominant construction in news prose: the cause is one
    // sentence and the connective OPENS the next.
    //
    //   "The ship rammed the bridge. As a result, the bridge collapsed."
    //
    // Measured, that yields zero links while the identical clause pair inside
    // one sentence yields `ram --as a result--> collapse`. The vocabulary and
    // the extraction were never the problem — the segmentation boundary was,
    // and on a news corpus (where "As a result," / "Consequently," /
    // "Therefore," are conventionally sentence-initial) that is most of the
    // causal structure there is.
    //
    // Bounded deliberately: only a signal in the LEADING position of a sentence
    // qualifies, and it links the last event of the previous sentence to the
    // first event of this one. A signal buried mid-sentence already had its
    // chance intra-sententially, and reaching backwards from there would invent
    // links across clauses that the writer joined for other reasons.
    for pair in sents.windows(2) {
        let (prev, cur) = (pair[0], pair[1]);
        let Some((dir, relation, start, end, marker)) = leading_signal(cur) else {
            continue;
        };
        // A leading signal has no left operand in its own sentence, so direction
        // resolves against the previous one: Forward = prev caused cur (the
        // common case), Backward = cur caused prev ("Because of this, ...").
        let prev_evs = event_triggers(prev);
        let cur_evs = event_triggers(cur);
        let (Some(prev_ev), Some(cur_ev)) = (prev_evs.last(), nearest_right(&cur_evs, end)) else {
            continue;
        };
        let _ = start;
        let (source, target) = match dir {
            SignalDir::Forward => (prev_ev.lemma.clone(), cur_ev.lemma.clone()),
            SignalDir::Backward => (cur_ev.lemma.clone(), prev_ev.lemma.clone()),
        };
        if source == target {
            continue;
        }
        let key = (source.clone(), marker.clone(), target.clone());
        if seen.insert(key) {
            out.push(EventLink {
                source,
                target,
                relation,
                signal: marker,
            });
        }
    }

    out
}

/// A signal occupying the LEADING position of a sentence — the first token, or
/// preceded only by punctuation. Returns `(dir, relation, start, end, marker)`.
///
/// Restricted to the leading position on purpose: that is the position where a
/// signal has no left operand in its own sentence and therefore genuinely
/// refers back to the previous one.
fn leading_signal(sent: &[ParsedToken]) -> Option<(SignalDir, LinkRelation, usize, usize, String)> {
    let first_word = sent
        .iter()
        .position(|t| t.text.chars().any(|c| c.is_alphanumeric()))?;
    signal_hits(sent)
        .into_iter()
        .find(|(_, _, start, _, _)| *start == first_word)
}

/// Extract the event causal links from raw text via the shared parser. `None` if
/// the parser model is unavailable.
pub fn extract_event_links(text: &str) -> Option<Vec<EventLink>> {
    let tokens = dep_parser::parse(text)?;
    Some(links_from_tokens(&tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk(i: usize, text: &str, pos: &str, lemma: &str) -> ParsedToken {
        ParsedToken {
            i,
            text: text.to_string(),
            head: i,
            dep: "dep".to_string(),
            pos: pos.to_string(),
            tag: pos.to_string(),
            lemma: lemma.to_string(),
        }
    }

    #[test]
    fn forward_signal_links_left_causes_right() {
        // "the failure caused the collapse" → failure --caused--> collapse
        let toks = vec![
            tk(0, "the", "DET", "the"),
            tk(1, "failure", "NOUN", "failure"),
            tk(2, "caused", "VERB", "cause"),
            tk(3, "the", "DET", "the"),
            tk(4, "collapse", "NOUN", "collapse"),
        ];
        // "caused" is both a VERB event and a FORWARD signal; the signal binds the
        // nominal events on each side (failure ← left, collapse → right).
        let links = link_events_in_sentence(&toks);
        assert!(
            links
                .iter()
                .any(|l| l.source == "failure" && l.target == "collapse"),
            "expected failure→collapse, got {links:?}"
        );
    }

    #[test]
    fn backward_signal_inverts_direction() {
        // "the collapse occurred due to the collision" → collision --due to--> collapse
        let toks = vec![
            tk(0, "the", "DET", "the"),
            tk(1, "collapse", "NOUN", "collapse"),
            tk(2, "occurred", "VERB", "occur"),
            tk(3, "due", "ADP", "due"),
            tk(4, "to", "ADP", "to"),
            tk(5, "the", "DET", "the"),
            tk(6, "collision", "NOUN", "collision"),
        ];
        let links = link_events_in_sentence(&toks);
        assert!(
            links
                .iter()
                .any(|l| l.source == "collision" && l.target == "collapse"),
            "backward signal must put the following clause as cause, got {links:?}"
        );
    }

    #[test]
    fn temporal_signal_is_precedes_not_causes() {
        // "rescue after collapse" → collapse PRECEDES rescue. Temporal order, NOT
        // causation — the arm must not claim the collapse caused the rescue.
        let toks = vec![
            tk(0, "rescue", "NOUN", "rescue"),
            tk(1, "after", "ADP", "after"),
            tk(2, "collapse", "NOUN", "collapse"),
        ];
        let links = link_events_in_sentence(&toks);
        let l = links
            .iter()
            .find(|l| l.signal == "after")
            .expect("an 'after' link");
        assert_eq!(
            l.relation,
            LinkRelation::Precedes,
            "temporal signal is not causal"
        );
        assert_eq!(
            l.source, "collapse",
            "`X after Y` → Y is earlier (the head)"
        );
        assert_eq!(l.target, "rescue");
    }

    #[test]
    fn nominal_pivots_are_events_even_without_a_verb() {
        // The inchoative pivots (blackout, collapse) are nouns — no entity arm sees
        // them, but the event arm does.
        let toks = vec![
            tk(0, "blackout", "NOUN", "blackout"),
            tk(1, "led", "VERB", "lead"),
            tk(2, "to", "ADP", "to"),
            tk(3, "evacuation", "NOUN", "evacuation"),
        ];
        let evs = event_triggers(&toks);
        assert!(evs
            .iter()
            .any(|e| e.lemma == "blackout" && e.kind == "nominal"));
        assert!(evs
            .iter()
            .any(|e| e.lemma == "evacuation" && e.kind == "nominal"));
        let links = link_events_in_sentence(&toks);
        assert!(links
            .iter()
            .any(|l| l.source == "blackout" && l.target == "evacuation"));
    }

    #[test]
    fn sentences_are_scoped_by_punctuation() {
        // Events in different sentences must NOT link across the boundary.
        // "the ship sank. the bridge opened." — no signal, no cross links anyway,
        // but verify segmentation splits them.
        let toks = vec![
            tk(0, "ship", "NOUN", "ship"),
            tk(1, "sank", "VERB", "sink"),
            tk(2, ".", "PUNCT", "."),
            tk(3, "bridge", "NOUN", "bridge"),
            tk(4, "opened", "VERB", "open"),
            tk(5, ".", "PUNCT", "."),
        ];
        assert_eq!(sentences(&toks).len(), 2);
    }

    #[test]
    fn leading_signal_links_across_a_sentence_boundary() {
        // "The ship rammed the bridge. As a result, the bridge collapsed."
        //
        // The dominant causal construction in news prose: cause in one
        // sentence, connective opening the next. Measured against the real
        // parser this produced ZERO links before the inter-sentential pass,
        // while the identical clause pair inside one sentence produced
        // `ram --as a result--> collapse`. Neither the vocabulary nor the
        // extraction was at fault — `links_from_tokens` simply never looked
        // across the boundary, which on a news corpus is most of the causal
        // structure that exists.
        let toks = vec![
            tk(0, "ship", "NOUN", "ship"),
            tk(1, "rammed", "VERB", "ram"),
            tk(2, "bridge", "NOUN", "bridge"),
            tk(3, ".", "PUNCT", "."),
            tk(4, "As", "ADP", "as"),
            tk(5, "a", "DET", "a"),
            tk(6, "result", "NOUN", "result"),
            tk(7, ",", "PUNCT", ","),
            tk(8, "bridge", "NOUN", "bridge"),
            tk(9, "collapsed", "VERB", "collapse"),
            tk(10, ".", "PUNCT", "."),
        ];
        let links = links_from_tokens(&toks);
        assert!(
            links
                .iter()
                .any(|l| l.source == "ram" && l.target == "collapse"),
            "leading signal must link the previous sentence's last event to this \
             sentence's first, got {links:?}"
        );
    }

    #[test]
    fn mid_sentence_signal_does_not_reach_into_the_previous_sentence() {
        // The bound on the pass above. A signal buried mid-sentence already had
        // its chance intra-sententially; letting it reach backwards would invent
        // links across clauses the writer joined for other reasons. Here the
        // second sentence is self-contained, so the only link is its own.
        let toks = vec![
            tk(0, "ship", "NOUN", "ship"),
            tk(1, "docked", "VERB", "dock"),
            tk(2, ".", "PUNCT", "."),
            tk(3, "failure", "NOUN", "failure"),
            tk(4, "caused", "VERB", "cause"),
            tk(5, "collapse", "NOUN", "collapse"),
            tk(6, ".", "PUNCT", "."),
        ];
        let links = links_from_tokens(&toks);
        assert!(
            links
                .iter()
                .any(|l| l.source == "failure" && l.target == "collapse"),
            "intra-sentence link must still fire, got {links:?}"
        );
        assert!(
            !links.iter().any(|l| l.source == "dock"),
            "a mid-sentence signal must NOT reach back to the previous sentence, got {links:?}"
        );
    }

    #[test]
    fn chain_reconstructs_across_sentences() {
        // Two sentences, each one link, forming a chain failure→collision→collapse.
        let toks = vec![
            // "failure caused collision."
            tk(0, "failure", "NOUN", "failure"),
            tk(1, "caused", "VERB", "cause"),
            tk(2, "collision", "NOUN", "collision"),
            tk(3, ".", "PUNCT", "."),
            // "collision led to collapse."
            tk(4, "collision", "NOUN", "collision"),
            tk(5, "led", "VERB", "lead"),
            tk(6, "to", "ADP", "to"),
            tk(7, "collapse", "NOUN", "collapse"),
            tk(8, ".", "PUNCT", "."),
        ];
        let links = links_from_tokens(&toks);
        assert!(links
            .iter()
            .any(|l| l.source == "failure" && l.target == "collision"));
        assert!(links
            .iter()
            .any(|l| l.source == "collision" && l.target == "collapse"));
    }
}
