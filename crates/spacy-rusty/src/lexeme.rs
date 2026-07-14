//! Lexical attributes feeding tok2vec: NORM, PREFIX, SUFFIX, SHAPE, SPACY,
//! IS_SPACE — computed to match spaCy's Lexeme / lang.lex_attrs.

use std::collections::HashMap;

/// NORM: lowercased orth, overridden by the `lexeme_norm` lookup table when
/// present. (Per-token special-case NORM overrides are applied at the token
/// level, not here.)
pub fn norm(text: &str, norm_table: &HashMap<String, String>) -> String {
    if let Some(n) = norm_table.get(text) {
        return n.clone();
    }
    text.to_lowercase()
}

/// PREFIX: first character of the orth.
pub fn prefix(text: &str) -> String {
    text.chars().next().map(|c| c.to_string()).unwrap_or_default()
}

/// SUFFIX: last 3 characters of the orth.
pub fn suffix(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let start = chars.len().saturating_sub(3);
    chars[start..].iter().collect()
}

/// SHAPE: spaCy `word_shape` — alpha->X/x, digit->d, other->self, runs of the
/// same shape-char capped at 4.
pub fn shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".to_string();
    }
    let mut out = String::new();
    let mut last: Option<char> = None;
    let mut seq = 0usize;
    for c in text.chars() {
        let sc = if c.is_alphabetic() {
            if c.is_uppercase() {
                'X'
            } else {
                'x'
            }
        } else if c.is_ascii_digit() {
            'd'
        } else {
            c
        };
        if last == Some(sc) {
            seq += 1;
        } else {
            seq = 0;
            last = Some(sc);
        }
        if seq < 4 {
            out.push(sc);
        }
    }
    out
}

/// IS_SPACE: orth is non-empty and entirely whitespace.
pub fn is_space(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_whitespace())
}

/// IS_ALPHA: non-empty and every char is alphabetic (Python `str.isalpha`).
pub fn is_alpha(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_alphabetic())
}

/// IS_DIGIT: non-empty and every char is an ASCII digit (Python `str.isdigit`).
pub fn is_digit(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| c.is_ascii_digit())
}

/// IS_PUNCT: non-empty and every char is punctuation.
pub fn is_punct(text: &str) -> bool {
    !text.is_empty()
        && text.chars().all(|c| {
            c.is_ascii_punctuation()
                || matches!(c, '\u{00A1}' | '\u{00BF}' | '\u{2010}'..='\u{2027}' | '\u{2030}'..='\u{205E}')
        })
}

/// IS_LOWER: has a cased char and all cased chars are lowercase (`str.islower`).
pub fn is_lower(text: &str) -> bool {
    let mut cased = false;
    for c in text.chars() {
        if c.is_uppercase() {
            return false;
        }
        if c.is_lowercase() {
            cased = true;
        }
    }
    cased
}

/// IS_UPPER: has a cased char and all cased chars are uppercase (`str.isupper`).
pub fn is_upper(text: &str) -> bool {
    let mut cased = false;
    for c in text.chars() {
        if c.is_lowercase() {
            return false;
        }
        if c.is_uppercase() {
            cased = true;
        }
    }
    cased
}

/// IS_TITLE: Python `str.istitle` — every run of cased chars starts uppercase
/// and continues lowercase, with at least one such run.
pub fn is_title(text: &str) -> bool {
    let mut cased = false;
    let mut prev_cased = false; // previous char was a cased letter
    for c in text.chars() {
        if c.is_uppercase() {
            if prev_cased {
                return false; // uppercase following a cased letter
            }
            cased = true;
            prev_cased = true;
        } else if c.is_lowercase() {
            if !prev_cased {
                return false; // lowercase starting a word
            }
            cased = true;
            prev_cased = true;
        } else {
            prev_cased = false;
        }
    }
    cased
}

const NUM_WORDS: &[&str] = &[
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
    "sextillion", "septillion", "octillion", "nonillion", "decillion", "gajillion", "bazillion",
];
const ORDINAL_WORDS: &[&str] = &[
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
    "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
    "seventieth", "eightieth", "ninetieth", "hundredth", "thousandth", "millionth", "billionth",
    "trillionth", "quadrillionth", "quintillionth", "sextillionth", "septillionth", "octillionth",
    "nonillionth", "decillionth", "gajillionth", "bazillionth",
];

fn all_ascii_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// LIKE_NUM: faithful port of spaCy `lang/en/lex_attrs.like_num`.
/// spaCy strips a leading sign, then commas/dots, and runs every subsequent
/// check on that stripped string.
pub fn like_num(text: &str) -> bool {
    let mut t = text;
    if let Some(c) = t.chars().next() {
        if matches!(c, '+' | '-' | '±' | '~') {
            t = &t[c.len_utf8()..];
        }
    }
    let s: String = t.chars().filter(|&c| c != ',' && c != '.').collect();
    if all_ascii_digits(&s) {
        return true;
    }
    if s.matches('/').count() == 1 {
        let mut it = s.split('/');
        if let (Some(a), Some(b)) = (it.next(), it.next()) {
            if all_ascii_digits(a) && all_ascii_digits(b) {
                return true;
            }
        }
    }
    let lower = s.to_lowercase();
    if NUM_WORDS.contains(&lower.as_str()) || ORDINAL_WORDS.contains(&lower.as_str()) {
        return true;
    }
    if lower.ends_with("st") || lower.ends_with("nd") || lower.ends_with("rd") || lower.ends_with("th") {
        let head = &lower[..lower.len() - 2];
        if all_ascii_digits(head) {
            return true;
        }
    }
    false
}
