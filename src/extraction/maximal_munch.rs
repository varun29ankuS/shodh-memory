use std::cmp::Ordering;
use aho_corasick::AhoCorasick;
use regex::Regex;
use std::sync::LazyLock;
use url::Url;

use super::types::{ExtractedEntity, ExtractionSource};
use super::config::CompiledPattern;

/// TLDs and common domain prefixes that are noise as standalone DomainLabel entities.
static SKIP_DOMAIN_LABELS: LazyLock<std::collections::HashSet<&'static str>> = LazyLock::new(|| {
    [
        // Generic TLDs
        "com", "org", "net", "edu", "gov", "io", "co", "us", "uk", "de", "xyz", "ai",
        "dev", "info", "biz", "site", "tech", "online", "local", "network",
        // Country codes (ASEAN focus + common)
        "vn", "th", "my", "id", "ph", "sg", "mm", "la", "kh", "ru", "cn", "jp", "br",
        "kr", "ir", "in", "au", "nz",
        // Second-level country domains
        "ac", "go", "or", "ne",
        // Common prefixes
        "www", "mail", "ftp", "api",
    ].iter().cloned().collect()
});

static IP_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap());
static URL_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bhttps?://[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|/))").unwrap());
static OPERATOR_ID_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b[A-Z]+(?:-[A-Z]+)+\b").unwrap());
static PREFIX_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(?:CLUSTER-[A-Z0-9-]+|CVE-\d{4}-\d{4,7})\b").unwrap());
// Simplistic domain regex with explicit common TLDs to avoid file.tar.gz matches
static DOMAIN_REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|us|uk|de|xyz|ai|dev|info|biz|site|tech|online|local|network|ru|cn|jp|br|kr|ir|in)\b").unwrap());

#[derive(Debug, Clone, PartialEq, Eq)]
enum CandidateSource {
    Config,
    BuiltIn,
    AhoCorasick,
}

#[derive(Debug, Clone)]
struct Candidate {
    start: usize, // Byte index
    end: usize,   // Byte index
    text: String,
    entity_type: String,
    source: CandidateSource,
    confidence: f32,
    original_ac_source: Option<ExtractionSource>, // If AC match, preserve its original source
}

pub fn run_maximal_munch(
    text: &str,
    ac: Option<&AhoCorasick>,
    ac_entities: &[ExtractedEntity], // Parallel array to AC patterns
    config_patterns: &[CompiledPattern],
) -> Vec<ExtractedEntity> {
    let mut candidates = Vec::new();

    // 1. Config Patterns
    for pattern in config_patterns {
        for cap in pattern.regex.captures_iter(text) {
            let m = if let Some(m) = cap.get(1) { m } else { cap.get(0).unwrap() };
            candidates.push(Candidate {
                start: m.start(),
                end: m.end(),
                text: m.as_str().to_string(),
                entity_type: pattern.entity_type.clone(),
                source: CandidateSource::Config,
                confidence: 0.9,
                original_ac_source: None,
            });
        }
    }

    // 2. Built-in Heuristics
    let mut add_builtin = |regex: &Regex, etype: &str| {
        for m in regex.find_iter(text) {
            // Verify IP octets if IpAddress
            if etype == "IpAddress" {
                let parts: Vec<&str> = m.as_str().split('.').collect();
                if parts.len() != 4 || !parts.iter().all(|p| p.parse::<u8>().is_ok()) {
                    continue;
                }
            }
            if etype == "OperatorId" && m.as_str().len() < 6 {
                continue;
            }
            let actual_etype = if etype == "Prefix" {
                if m.as_str().starts_with("CVE") { "Cve" } else { "ClusterName" }
            } else {
                etype
            };

            candidates.push(Candidate {
                start: m.start(),
                end: m.end(),
                text: m.as_str().to_string(),
                entity_type: actual_etype.to_string(),
                source: CandidateSource::BuiltIn,
                confidence: 0.9, // Built-ins base confidence
                original_ac_source: None,
            });
        }
    };

    add_builtin(&IP_REGEX, "IpAddress");
    add_builtin(&URL_REGEX, "Url");
    add_builtin(&OPERATOR_ID_REGEX, "OperatorId");
    add_builtin(&PREFIX_REGEX, "Prefix");
    add_builtin(&DOMAIN_REGEX, "Domain");

    // 3. Aho-Corasick Matches
    if let Some(ac) = ac {
        for mat in ac.find_iter(text) {
            let ac_ent = &ac_entities[mat.pattern().as_usize()];
            candidates.push(Candidate {
                start: mat.start(),
                end: mat.end(),
                text: text[mat.start()..mat.end()].to_string(),
                entity_type: ac_ent.entity_type.clone(),
                source: CandidateSource::AhoCorasick,
                confidence: ac_ent.confidence,
                original_ac_source: Some(ac_ent.source),
            });
        }
    }

    // 4. Resolve Overlaps (Sort & filter)
    candidates.sort_by(|a, b| {
        let len_a = a.end - a.start;
        let len_b = b.end - b.start;
        if len_a != len_b {
            return len_b.cmp(&len_a); // Longest first
        }
        // Tie-breaker 1: Source precedence
        let source_score = |s: &CandidateSource| match s {
            CandidateSource::Config => 3,
            CandidateSource::BuiltIn => 2,
            CandidateSource::AhoCorasick => 1,
        };
        let ss_a = source_score(&a.source);
        let ss_b = source_score(&b.source);
        if ss_a != ss_b {
            return ss_b.cmp(&ss_a);
        }
        // Tie-breaker 2: Alphabetical by entity type
        let cmp_type = a.entity_type.cmp(&b.entity_type);
        if cmp_type != Ordering::Equal {
            return cmp_type;
        }
        // Tie-breaker 3: Earlier in text
        a.start.cmp(&b.start)
    });

    let mut accepted = Vec::new();
    let mut consumed_intervals: Vec<(usize, usize)> = Vec::new();

    for cand in candidates {
        let overlaps = consumed_intervals.iter().any(|(s, e)| {
            cand.start < *e && cand.end > *s
        });
        if !overlaps {
            consumed_intervals.push((cand.start, cand.end));
            accepted.push(cand);
        }
    }

    // 5. Containment Extraction and Emitting
    let mut final_entities = Vec::new();

    // Reconstruct the system words dictionary logic later; for now mock it.
    let is_dictionary_word = |word: &str| -> bool {
        // Read /usr/share/dict/words logic would go here.
        // We can load it once into a global HashSet.
        is_in_system_dict(word)
    };

    for cand in accepted {
        let mut derived = Vec::new();

        if cand.entity_type == "Url" {
            // It's an intermediate entity, we will drop it during Merge phase (Phase 1),
            // but we still extract it here and mark it.
            if let Ok(parsed) = Url::parse(&cand.text) {
                if let Some(host) = parsed.host_str() {
                    // Find the host in the original text to get correct byte span
                    if let Some(host_idx) = cand.text.find(host) {
                        let h_start = cand.start + host_idx;
                        let h_end = h_start + host.len();
                        
                        let is_ip = IP_REGEX.is_match(host);
                        derived.push(ExtractedEntity {
                            text: host.to_string(),
                            entity_type: if is_ip { "IpAddress".to_string() } else { "Domain".to_string() },
                            confidence: 1.0,
                            spans: vec![(h_start, h_end)],
                            source: ExtractionSource::MaximalMunch,
                        });

                        // If Domain, check DomainLabels (skip TLDs and common prefixes)
                        if !is_ip {
                            let mut offset = 0;
                            for label in host.split(|c| c == '.' || c == '-') {
                                let l_len = label.len();
                                if is_dictionary_word(label) && !SKIP_DOMAIN_LABELS.contains(label.to_lowercase().as_str()) {
                                    let l_start = h_start + offset;
                                    let l_end = l_start + l_len;
                                    derived.push(ExtractedEntity {
                                        text: label.to_string(),
                                        entity_type: "DomainLabel".to_string(),
                                        confidence: 0.5,
                                        spans: vec![(l_start, l_end)],
                                        source: ExtractionSource::MaximalMunch,
                                    });
                                }
                                offset += l_len + 1; // +1 for the separator
                            }
                        }
                    }
                }
                
                let path = parsed.path();
                let slashes = path.chars().filter(|c| *c == '/').count();
                let has_ext = path.contains('.');
                if slashes >= 2 || (slashes >= 1 && has_ext) {
                    // It's a non-trivial path. We must not emit a span because of URL decoding drift.
                    derived.push(ExtractedEntity {
                        text: path.to_string(),
                        entity_type: "Path".to_string(),
                        confidence: 0.9,
                        spans: vec![], // Omitted to avoid decoding drift panics
                        source: ExtractionSource::MaximalMunch,
                    });
                }
            }
        } else if cand.entity_type == "Domain" {
            let mut offset = 0;
            for label in cand.text.split(|c| c == '.' || c == '-') {
                let l_len = label.len();
                if is_dictionary_word(label) && !SKIP_DOMAIN_LABELS.contains(label.to_lowercase().as_str()) {
                    let l_start = cand.start + offset;
                    let l_end = l_start + l_len;
                    derived.push(ExtractedEntity {
                        text: label.to_string(),
                        entity_type: "DomainLabel".to_string(),
                        confidence: 0.5,
                        spans: vec![(l_start, l_end)],
                        source: ExtractionSource::MaximalMunch,
                    });
                }
                offset += l_len + 1;
            }
        }

        final_entities.push(ExtractedEntity {
            text: cand.text,
            entity_type: cand.entity_type,
            confidence: cand.confidence,
            spans: vec![(cand.start, cand.end)],
            source: cand.original_ac_source.unwrap_or(ExtractionSource::MaximalMunch),
        });
        
        final_entities.extend(derived);
    }

    final_entities
}

static SYSTEM_DICT: LazyLock<std::collections::HashSet<String>> = LazyLock::new(|| {
    let mut set = std::collections::HashSet::new();
    if let Ok(content) = std::fs::read_to_string("/usr/share/dict/words") {
        for line in content.lines() {
            set.insert(line.trim().to_lowercase());
        }
    }
    set
});

fn is_in_system_dict(word: &str) -> bool {
    SYSTEM_DICT.contains(&word.to_lowercase())
}
