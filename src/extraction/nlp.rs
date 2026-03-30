use unicode_segmentation::UnicodeSegmentation;
use std::collections::HashSet;
use std::sync::LazyLock;

use super::types::{ExtractedEntity, ExtractedTriple, ExtractionSource};

#[derive(Debug, Clone, PartialEq)]
enum PosTag {
    NNP, // Proper noun
    NN,  // Common noun
    VB,  // Verb
    VBN, // Past participle (heuristically ending in 'ed')
    VBG, // Gerund/present participle (heuristically ending in 'ing')
    JJ,  // Adjective
    CD,  // Cardinal number
    IN,  // Preposition
    DT,  // Determiner
    OTHER,
}

#[derive(Debug)]
struct Token {
    text: String,
    tag: PosTag,
    start: usize,
    end: usize,
}

static GLOBAL_STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Generic nouns
        "thing", "way", "lot", "result", "case", "kind", "type", "part", "point", "fact",
        "issue", "problem", "question", "reason", "example", "number", "time", "place",
        "area", "end", "side", "use", "set", "group", "level", "step", "item", "change",
        "output", "input", "value", "name", "list", "count", "status", "state", "mode",
        "test", "error", "check", "note", "line", "run", "task", "work", "fix",
        // Report/section headers
        "summary", "findings", "issues", "details", "results", "overview", "analysis",
        "approach", "actions", "updates", "observations", "playbook", "schema", "script",
        "batch", "table", "category", "method", "phase", "cycle", "round",
        // Domain-generic words (useful in multi-word NPs, noise as singletons)
        "domains", "subdomains", "trace", "update", "handle", "operator", "cluster",
        "chain", "queue", "queries", "outputs", "context", "session", "module",
        "options", "configs", "settings", "format", "pattern", "version", "model",
        "artifacts", "nodes", "edges", "json", "data", "report", "entry",
        // Pronouns and function words that survive POS tagger
        "me", "him", "her", "them", "us", "something", "anything", "nothing", "everything",
    ].iter().cloned().collect()
});

static DOMAIN_NOUNS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    ["server", "system", "host", "service", "network", "port", "file", "data", "user", "client", "request", "response", "connection", "process", "node"].iter().cloned().collect()
});

static DETERMINERS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    ["the", "a", "an", "this", "that", "these", "those"].iter().cloned().collect()
});

static PREPOSITIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    ["by", "in", "on", "at", "to", "for", "with", "from", "of", "about", "as", "into", "like", "through", "after", "over", "between", "out", "against", "during", "without", "before", "under", "around", "among"].iter().cloned().collect()
});

static AUXILIARIES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    ["is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "can", "could", "should", "must", "will", "would"].iter().cloned().collect()
});

/// Common English words that should NOT be proper nouns even when capitalized
/// (typically capitalized only because they start a sentence).
static FALSE_PROPER_NOUNS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Common verbs/actions at sentence start
        "let", "see", "look", "get", "got", "set", "put", "run", "ran", "did",
        "made", "had", "found", "used", "added", "fixed", "based", "need", "want",
        "tried", "moved", "checked", "confirmed", "updated", "removed", "changed",
        // Common adjectives/adverbs
        "new", "old", "good", "bad", "big", "top", "low", "high", "full", "same",
        "key", "very", "just", "also", "still", "only", "well", "more", "most",
        "complete", "single", "near", "next", "last", "first", "other",
        // Interrogatives
        "what", "which", "who", "whom", "whose", "how", "when", "where", "why",
        // Pronouns/determiners
        "here", "there", "now", "then", "both", "all", "no", "not", "but", "so",
        "this", "that", "these", "those", "they", "them", "we", "it", "its",
        "one", "each", "some", "any", "every", "many", "few", "much",
        // Conjunctions
        "and", "or", "yet", "nor",
        // Adjectives not caught by suffix rules
        "true", "false", "real", "raw", "free", "open", "live", "dead", "safe", "sure",
        "able", "main", "clear", "done", "final", "total", "right", "left",
        // Adverbs
        "already", "actually", "currently", "recently", "usually", "often",
        "always", "never", "almost", "quite", "rather", "enough",
        "instead", "however", "therefore", "otherwise", "perhaps", "probably",
        "maybe", "likely", "mostly", "basically", "simply", "essentially",
        // Number words
        "zero", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "twenty", "hundred", "thousand",
        "several", "multiple", "various", "numerous",
        // Generic adjectives/misc non-entities
        "thin", "thick", "ones", "own", "such", "else",
        // Prefixes/misc non-entities
        "auto", "semi", "self", "non",
        // Report/meta words
        "note", "per", "via", "yes",
        "summary", "findings", "issues", "status", "think", "script",
    ].iter().cloned().collect()
});

/// Short entities (< 4 chars) are noise unless they're known acronyms.
static SHORT_ENTITY_ALLOWLIST: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "TOR", "PHP", "OJS", "WAF", "VPN", "TLS", "DNS", "SQL", "SEO",
        "ASN", "CDN", "CMS", "XSS", "API", "SSH", "FTP", "RCE",
    ].iter().cloned().collect()
});

static KNOWN_VERBS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Canonical dictionary verbs
        "infect", "compromise", "breach", "exploit",
        "redirect", "forward", "proxy",
        "host", "serve", "run",
        "contain", "include", "embed",
        "block", "filter", "deny",
        // Common threat-intel / report verbs
        "assign", "resolve", "scan", "drop", "map", "load",
        "inject", "target", "distribute", "connect", "send",
        "receive", "download", "upload", "execute", "install",
        "deliver", "observe", "detect", "report", "indicate",
        "use", "show", "point", "link", "associate",
    ].iter().cloned().collect()
});

fn heuristic_pos_tag(word: &str) -> PosTag {
    // Contractions and possessives are never entities
    if word.contains('\'') || word.contains('\u{2019}') {
        return PosTag::OTHER;
    }
    let lower = word.to_lowercase();
    if word.chars().all(|c| c.is_ascii_digit()) {
        return PosTag::CD;
    }
    if DETERMINERS.contains(lower.as_str()) {
        return PosTag::DT;
    }
    if PREPOSITIONS.contains(lower.as_str()) {
        return PosTag::IN;
    }
    if AUXILIARIES.contains(lower.as_str()) {
        return PosTag::VB; // treat aux as verb
    }

    // Check known verbs (catches 3rd-person "assigns", "hosts", "blocks", etc.)
    if KNOWN_VERBS.contains(lower.as_str()) {
        return PosTag::VB;
    }
    {
        let lemma = lemmatize_verb(&lower);
        if lemma != lower && KNOWN_VERBS.contains(lemma.as_str()) {
            return PosTag::VB;
        }
    }

    // Simplistic heuristics
    if lower.ends_with("ed") && lower.len() > 3 {
        return PosTag::VBN;
    }
    if lower.ends_with("ing") && lower.len() > 4 {
        return PosTag::VBG;
    }
    if lower.ends_with("ous") || lower.ends_with("al") || lower.ends_with("ic") || lower.ends_with("ive") || lower.ends_with("ble") || lower.ends_with("ful") {
        return PosTag::JJ;
    }
    
    // False-NNP filter: common words capitalized at sentence start
    if FALSE_PROPER_NOUNS.contains(lower.as_str()) {
        return if GLOBAL_STOPWORDS.contains(lower.as_str()) || DOMAIN_NOUNS.contains(lower.as_str()) {
            PosTag::NN
        } else {
            PosTag::OTHER // Don't even let these into NP chunks
        };
    }

    // Capitalized word mid-sentence usually NNP
    let first_char = word.chars().next().unwrap_or('a');
    if first_char.is_uppercase() {
        return PosTag::NNP;
    }

    if DOMAIN_NOUNS.contains(lower.as_str()) || GLOBAL_STOPWORDS.contains(lower.as_str()) {
        return PosTag::NN;
    }

    // Default to common noun for alphabetical strings, with guards:
    // - Must be at least 2 chars
    // - Must start and end with a letter (no leading/trailing hyphens)
    // - Must contain at least one letter (no pure punctuation)
    if word.len() >= 2
        && word.chars().all(|c| c.is_alphabetic() || c == '-')
        && word.chars().next().map_or(false, |c| c.is_alphabetic())
        && word.chars().last().map_or(false, |c| c.is_alphabetic())
    {
        PosTag::NN
    } else {
        PosTag::OTHER
    }
}

fn lemmatize_verb(verb: &str) -> String {
    let v = verb.to_lowercase();
    if v.ends_with("ies") { return format!("{}y", &v[..v.len()-3]); }
    if v.ends_with("ied") { return format!("{}y", &v[..v.len()-3]); }
    if v.ends_with("ing") {
        let root = &v[..v.len()-3];
        if root == "compromis" || root == "includ" || root == "requir" || root == "serv" || root == "us" || root == "mak" {
            return format!("{}e", root);
        }
        let bytes = root.as_bytes();
        let len = bytes.len();
        if len >= 2 && bytes[len-1] == bytes[len-2] && bytes[len-1] != b's' {
            return root[..len-1].to_string();
        }
        return root.to_string();
    }
    if v.ends_with("ed") {
        let root = &v[..v.len()-2];
        if root == "compromis" || root == "includ" || root == "requir" || root == "serv" || root == "us" {
            return format!("{}e", root);
        }
        if root == "infect" || root == "redirect" || root == "host" || root == "embed" || root == "block" || root == "exploit" || root == "forward" {
            return root.to_string();
        }
        if v.ends_with("eed") { return v[..v.len()-1].to_string(); }
        let bytes = root.as_bytes();
        let len = bytes.len();
        if len >= 2 && bytes[len-1] == bytes[len-2] && bytes[len-1] != b's' {
            return root[..len-1].to_string();
        }
        return root.to_string();
    }
    if v.ends_with("es") {
        let root = &v[..v.len()-2];
        if root.ends_with("ch") || root.ends_with("sh") || root.ends_with("x") || root.ends_with("s") {
            return root.to_string();
        }
        if root == "compromis" || root == "includ" || root == "requir" || root == "serv" || root == "us" {
            return format!("{}e", root);
        }
    }
    if v.ends_with('s') && !v.ends_with("ss") {
        return v[..v.len()-1].to_string();
    }
    v
}

pub fn run_nlp_parse(
    text: &str,
    consumed_spans: &[(usize, usize)],
    source1_entities: &[ExtractedEntity],
    config_stopwords: &[String],
) -> (Vec<ExtractedEntity>, Vec<ExtractedTriple>) {
    let mut entities = Vec::new();
    let mut triples = Vec::new();

    // 1. Tokenization using UAX #29 (word bounds)
    // Filter non-content segments (whitespace, standalone punctuation) so that
    // adjacent content words are consecutive in the token array, enabling
    // multi-word NP chunking.
    let mut tokens = Vec::new();
    for (start, word) in text.split_word_bound_indices() {
        let end = start + word.len();

        // Skip whitespace but keep punctuation — punctuation gets tagged OTHER
        // and naturally breaks NP chunks at sentence boundaries.
        if word.chars().all(|c| c.is_whitespace()) {
            continue;
        }

        // Skip if inside consumed span
        let in_consumed = consumed_spans.iter().any(|(s, e)| start >= *s && end <= *e);
        if in_consumed {
            continue;
        }

        let tag = heuristic_pos_tag(word);
        tokens.push(Token {
            text: word.to_string(),
            tag,
            start,
            end,
        });
    }

    // 1b. Merge hyphenated compounds: UAX#29 splits "chain-tracer" into
    // ["chain", "-", "tracer"]. Re-join when hyphen is directly between words.
    let raw_tokens = tokens;
    let mut tokens = Vec::with_capacity(raw_tokens.len());
    let mut ti = 0;
    while ti < raw_tokens.len() {
        if raw_tokens[ti].text.chars().any(|c| c.is_alphanumeric()) {
            let mut compound_end = ti;
            while compound_end + 2 < raw_tokens.len()
                && raw_tokens[compound_end + 1].text == "-"
                && raw_tokens[compound_end].end == raw_tokens[compound_end + 1].start
                && raw_tokens[compound_end + 1].end == raw_tokens[compound_end + 2].start
                && raw_tokens[compound_end + 2].text.chars().any(|c| c.is_alphanumeric())
            {
                compound_end += 2;
            }
            if compound_end > ti {
                let compound_text = text[raw_tokens[ti].start..raw_tokens[compound_end].end].to_string();
                let tag = heuristic_pos_tag(&compound_text);
                tokens.push(Token { text: compound_text, tag, start: raw_tokens[ti].start, end: raw_tokens[compound_end].end });
                ti = compound_end + 1;
            } else {
                tokens.push(Token { text: raw_tokens[ti].text.clone(), tag: raw_tokens[ti].tag.clone(), start: raw_tokens[ti].start, end: raw_tokens[ti].end });
                ti += 1;
            }
        } else {
            tokens.push(Token { text: raw_tokens[ti].text.clone(), tag: raw_tokens[ti].tag.clone(), start: raw_tokens[ti].start, end: raw_tokens[ti].end });
            ti += 1;
        }
    }

    // 2. NP Chunking (Simplified)
    // We look for sequences of (JJ|VBN|VBG|NN|NNP)+ ending in NN or NNP.
    // CD can extend but not start a chunk — prevents "6 operators" quantity phrases.
    let mut nps = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let tag = &tokens[i].tag;
        if *tag == PosTag::NNP || *tag == PosTag::NN || *tag == PosTag::VBN || *tag == PosTag::VBG || *tag == PosTag::JJ {
            let start_idx = i;
            let mut end_idx = i;
            let mut is_nnp = *tag == PosTag::NNP;
            let mut has_noun = *tag == PosTag::NN || *tag == PosTag::NNP;

            i += 1;
            while i < tokens.len() {
                let t = &tokens[i].tag;
                if *t == PosTag::NNP || *t == PosTag::NN || *t == PosTag::VBN || *t == PosTag::VBG || *t == PosTag::JJ || *t == PosTag::CD {
                    if *t == PosTag::NNP { is_nnp = true; }
                    if *t == PosTag::NN || *t == PosTag::NNP { has_noun = true; }
                    end_idx = i;
                    i += 1;
                } else {
                    break;
                }
            }

            // Must contain at least one noun to be a valid noun phrase
            if has_noun {
                // backtrack end_idx if it doesn't end in NN or NNP
                while end_idx >= start_idx {
                    let last_tag = &tokens[end_idx].tag;
                    if *last_tag == PosTag::NN || *last_tag == PosTag::NNP {
                        break;
                    }
                    if end_idx == 0 { break; }
                    end_idx -= 1;
                }
                
                if end_idx >= start_idx && end_idx < tokens.len() { // ensure bounds
                    let np_start = tokens[start_idx].start;
                    let np_end = tokens[end_idx].end;
                    let np_text = text[np_start..np_end].to_string();
                    
                    let word_count = end_idx - start_idx + 1;
                    let lower_text = np_text.to_lowercase();
                    
                    let is_tier1 = word_count == 1 && (GLOBAL_STOPWORDS.contains(lower_text.as_str()) || config_stopwords.iter().any(|s| s.to_lowercase() == lower_text));
                    let is_tier2_bare = word_count == 1 && DOMAIN_NOUNS.contains(lower_text.as_str());
                    // Multi-word NPs where every token is a stopword are noise ("me check", "data output")
                    let all_stopwords = word_count > 1 && (start_idx..=end_idx).all(|j| {
                        let lw = tokens[j].text.to_lowercase();
                        GLOBAL_STOPWORDS.contains(lw.as_str()) || DOMAIN_NOUNS.contains(lw.as_str())
                    });

                    // Tighter cap for all-common-noun phrases (no proper nouns)
                    let max_words: usize = if is_nnp { 4 } else { 2 };
                    if !is_tier1 && !is_tier2_bare && !all_stopwords && word_count <= max_words {
                        let entity_type = if is_nnp { "ProperNoun" } else { "NounPhrase" };
                        
                        let cap = if is_nnp {
                            if word_count > 1 { 0.9 } else { 0.7 }
                        } else {
                            if word_count > 1 { 0.6 } else { 0.4 }
                        };
                        
                        let mut conf: f32 = 0.4;
                        if is_nnp { conf += 0.3; }
                        if word_count > 1 { conf += 0.2; }
                        conf = conf.min(cap);
                        
                        nps.push((np_text, entity_type.to_string(), np_start, np_end, conf));
                    }
                }
            }
        } else {
            i += 1;
        }
    }

    // PP attachment merge ("of"-linking)
    let mut merged_nps = Vec::new();
    let mut n = 0;
    while n < nps.len() {
        if n + 1 < nps.len() {
            let (_, _, _, end1, _) = &nps[n];
            let (_, _, start2, _, _) = &nps[n+1];
            let between_text = text[*end1..*start2].trim().to_lowercase();
            if between_text == "of" || between_text == "for" || between_text == "in" {
                // Only merge if at least one NP contains a proper noun
                // ("Ministry of Agriculture" yes, "briefing instead of two thin ones" no)
                if nps[n].1 == "ProperNoun" || nps[n+1].1 == "ProperNoun" {
                    let np_start = nps[n].2;
                    let np_end = nps[n+1].3;
                    let np_text = text[np_start..np_end].to_string();
                    merged_nps.push((np_text, "ProperNoun".to_string(), np_start, np_end, 0.75));
                    n += 2;
                    continue;
                }
            }
        }
        merged_nps.push(nps[n].clone());
        n += 1;
    }

    for (np_text, entity_type, np_start, np_end, conf) in merged_nps.clone() {
        // Skip very short entities unless they're known acronyms
        if np_text.len() < 4 && !SHORT_ENTITY_ALLOWLIST.contains(np_text.to_uppercase().as_str()) {
            continue;
        }
        entities.push(ExtractedEntity {
            text: np_text,
            entity_type,
            confidence: conf,
            spans: vec![(np_start, np_end)],
            source: ExtractionSource::NlpParse,
        });
    }

    // 3. SVO Extraction
    // Combine merged_nps and source1_entities for SVO
    let mut all_nps: Vec<(String, usize, usize, f32)> = merged_nps.into_iter().map(|(t, _, s, e, c)| (t, s, e, c)).collect();
    for ent in source1_entities {
        for &(s, e) in &ent.spans {
            all_nps.push((ent.text.clone(), s, e, ent.confidence));
        }
    }
    all_nps.sort_by_key(|k| k.1);

    // Look for verbs (VB, VBN, VBG) and find nearest NP before and after
    for (idx, tok) in tokens.iter().enumerate() {
        if tok.tag == PosTag::VB || tok.tag == PosTag::VBN || tok.tag == PosTag::VBG {
            // Found a verb. Attempt SVO.
            let verb_text = lemmatize_verb(&tok.text);
            
            // Check for passive (was + VBN)
            let mut is_passive = false;
            if tok.tag == PosTag::VBN && idx > 0 && AUXILIARIES.contains(tokens[idx-1].text.to_lowercase().as_str()) {
                is_passive = true;
            }

            let prev_np = all_nps.iter().rev().find(|&&(_, _, e, _)| e <= tok.start);
            let next_np = all_nps.iter().find(|&&(_, s, _, _)| s >= tok.end);

            if let Some(prev) = prev_np {
                let mut subject = None;
                let mut subj_span = None;
                let mut object = None;
                let mut obj_span = None;
                let mut is_valid = false;
                
                let mut prepositional_object = false;

                if is_passive {
                    // Agentless passive is possible
                    object = Some(prev.0.clone());
                    obj_span = Some((prev.1, prev.2));
                    is_valid = true;
                    
                    if let Some(next) = next_np {
                        let by_token_exists = tokens.iter().any(|t| t.start >= tok.end && t.end <= next.1 && t.text.to_lowercase() == "by");
                        if by_token_exists {
                            subject = Some(next.0.clone());
                            subj_span = Some((next.1, next.2));
                        }
                    }
                } else if let Some(next) = next_np {
                    subject = Some(prev.0.clone());
                    subj_span = Some((prev.1, prev.2));
                    object = Some(next.0.clone());
                    obj_span = Some((next.1, next.2));
                    is_valid = true;
                    
                    let prep_exists = tokens.iter().any(|t| t.start >= tok.end && t.end <= next.1 && ["via", "through", "with", "from"].contains(&t.text.to_lowercase().as_str()));
                    if prep_exists {
                        prepositional_object = true;
                    }
                }

                if is_valid {
                    let mut confidence: f32 = 0.4;
                    if !is_passive { confidence += 0.3; } // Active voice boost
                    
                    let subj_conf = if is_passive {
                        if subject.is_some() { next_np.unwrap().3 } else { 0.0 }
                    } else { prev.3 };
                    
                    let obj_conf = if is_passive { prev.3 } else { next_np.unwrap().3 };
                    
                    if subj_conf >= 0.7 && obj_conf >= 0.7 {
                        confidence += 0.2;
                    }
                    
                    if prepositional_object { confidence -= 0.4; }
                    
                    // Coordination penalty
                    let (min_s, max_e) = if is_passive { 
                        if let Some(next) = next_np { (prev.1, next.2) } else { (prev.1, tok.end) }
                    } else {
                        (prev.1, next_np.unwrap().2)
                    };
                    
                    let multiple_verbs = tokens.iter().filter(|t| t.start >= min_s && t.end <= max_e && (t.tag == PosTag::VB || t.tag == PosTag::VBN || t.tag == PosTag::VBG)).count() > 1;
                    let intermediate_nps_total = all_nps.iter().filter(|&&(_, s, e, _)| s > min_s && e < max_e).count();
                    if multiple_verbs || intermediate_nps_total > 0 {
                        confidence -= 0.3;
                    }
                    
                    // Distance penalty > 8 tokens
                    let tokens_between = tokens.iter().filter(|t| t.start >= prev.2 && t.end <= next_np.map(|n| n.1).unwrap_or(tok.end)).count();
                    if tokens_between > 8 {
                        confidence -= 0.2;
                    }

                    triples.push(ExtractedTriple {
                        subject,
                        subject_span: subj_span,
                        verb: verb_text,
                        object: object.unwrap(),
                        object_span: obj_span,
                        confidence: confidence.max(0.0),
                        passive: is_passive,
                        promoted: false, // Updated later
                    });
                }
            }
        }
    }

    (entities, triples)
}
