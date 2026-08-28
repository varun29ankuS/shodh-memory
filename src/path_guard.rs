//! The filesystem chokepoint for caller-supplied identifiers.
//!
//! `validation::validate_user_id` already states this codebase's identifier contract — and its
//! rules are exactly what a path component needs (no separators in the charset, `..` rejected,
//! no leading/trailing dot, bounded length). What was missing is ENFORCEMENT AT THE JOIN: the
//! validator is called in some HTTP handlers and skipped in others, while every per-user
//! directory is `base.join(user_id)` regardless. This module re-exports the same contract as the
//! guard the state manager's and backup engine's join helpers route through, so an unvalidated
//! id cannot reach a filesystem path whichever handler it arrived by.
//!
//! Deliberately a delegation, not a second vocabulary: two validators drift, and an id accepted
//! at the API layer must never fail at the filesystem layer.

/// Validate `value` as a safe single path component, under the SAME rules as
/// `validation::validate_user_id`. Returns the same slice on success so call sites stay
/// borrow-friendly; rejects rather than normalizes, since silently rewriting `../x` would hide
/// the attempt.
pub fn sanitize_component<'a>(value: &'a str, what: &str) -> anyhow::Result<&'a str> {
    crate::validation::validate_user_id(value)
        .map_err(|e| anyhow::anyhow!("invalid {what}: {e}"))?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::sanitize_component;

    #[test]
    fn accepts_everything_the_api_layer_accepts() {
        // Parity with validate_user_id is the point — an id accepted at the API boundary must
        // never fail at the filesystem boundary. Unicode-alphanumeric ids are valid upstream.
        for ok in [
            "all",
            "user-1",
            "a.b",
            "X_9",
            "someone@example.com",
            "名前",
            "müller",
        ] {
            assert!(sanitize_component(ok, "user id").is_ok(), "{ok}");
        }
    }

    #[test]
    fn rejects_everything_that_could_leave_the_directory() {
        for bad in [
            "",
            ".",
            "..",
            "../x",
            "a/b",
            "a\\b",
            "/etc",
            "a\0b",
            ".hidden",
            "a..b",
            "trailing.",
            "a b",
            &"x".repeat(129),
        ] {
            assert!(sanitize_component(bad, "user id").is_err(), "{bad:?}");
        }
    }
}
