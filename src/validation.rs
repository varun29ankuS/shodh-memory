//! Input validation for enterprise security
//! Prevents injection attacks, ensures data integrity, protects against ReDoS

use anyhow::{Result, anyhow};
use regex::Regex;

/// Maximum lengths for security
pub const MAX_USER_ID_LENGTH: usize = 128;
pub const MAX_CONTENT_LENGTH: usize = 50_000;  // 50KB
pub const MAX_PATTERN_LENGTH: usize = 256;     // Max regex pattern length
pub const MAX_ENTITY_LENGTH: usize = 256;      // Max entity name length
pub const MAX_METADATA_SIZE: usize = 10_000;   // Max metadata JSON size (10KB)
pub const MAX_ENTITIES_PER_MEMORY: usize = 50; // Max entities per memory

/// Validate user_id
pub fn validate_user_id(user_id: &str) -> Result<()> {
    if user_id.is_empty() {
        return Err(anyhow!("user_id cannot be empty"));
    }

    if user_id.len() > MAX_USER_ID_LENGTH {
        return Err(anyhow!(
            "user_id too long: {} chars (max: {})",
            user_id.len(),
            MAX_USER_ID_LENGTH
        ));
    }

    // Only allow alphanumeric, dash, underscore
    if !user_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '@' || c == '.') {
        return Err(anyhow!(
            "user_id contains invalid characters (allowed: alphanumeric, -, _, @, .)"
        ));
    }

    Ok(())
}

/// Validate memory_id (UUID format)
pub fn validate_memory_id(memory_id: &str) -> Result<uuid::Uuid> {
    uuid::Uuid::parse_str(memory_id)
        .map_err(|e| anyhow!("Invalid memory_id UUID format: {e}"))
}

/// Validate content
pub fn validate_content(content: &str, allow_empty: bool) -> Result<()> {
    if !allow_empty && content.trim().is_empty() {
        return Err(anyhow!("content cannot be empty"));
    }

    if content.len() > MAX_CONTENT_LENGTH {
        return Err(anyhow!(
            "content too long: {} chars (max: {})",
            content.len(),
            MAX_CONTENT_LENGTH
        ));
    }

    Ok(())
}

/// Validate embeddings vector
pub fn validate_embeddings(embeddings: &[f32]) -> Result<()> {
    if embeddings.is_empty() {
        return Err(anyhow!("embeddings cannot be empty"));
    }

    // Common embedding dimensions: 384, 512, 768, 1024, 1536
    let valid_dims = [128, 256, 384, 512, 768, 1024, 1536, 2048];
    if !valid_dims.contains(&embeddings.len()) {
        return Err(anyhow!(
            "Unusual embedding dimension: {}. Common dimensions: {:?}",
            embeddings.len(),
            valid_dims
        ));
    }

    // Check for NaN or Inf
    if embeddings.iter().any(|&v| !v.is_finite()) {
        return Err(anyhow!("embeddings contain NaN or Inf values"));
    }

    Ok(())
}

/// Validate importance threshold
pub fn validate_importance_threshold(threshold: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(anyhow!(
            "importance_threshold must be between 0.0 and 1.0, got: {threshold}"
        ));
    }
    Ok(())
}

/// Validate max_results
pub fn validate_max_results(max_results: usize) -> Result<()> {
    if max_results == 0 {
        return Err(anyhow!("max_results must be greater than 0"));
    }

    if max_results > 10_000 {
        return Err(anyhow!(
            "max_results too large: {max_results} (max: 10,000)"
        ));
    }

    Ok(())
}

/// Validate and compile a regex pattern with ReDoS protection
///
/// Prevents regex denial-of-service attacks by:
/// 1. Limiting pattern length
/// 2. Detecting potentially catastrophic backtracking patterns
/// 3. Using regex crate which has built-in protections
pub fn validate_and_compile_pattern(pattern: &str) -> Result<Regex> {
    // Length check
    if pattern.is_empty() {
        return Err(anyhow!("Pattern cannot be empty"));
    }

    if pattern.len() > MAX_PATTERN_LENGTH {
        return Err(anyhow!(
            "Pattern too long: {} chars (max: {})",
            pattern.len(),
            MAX_PATTERN_LENGTH
        ));
    }

    // Detect potentially dangerous patterns using state machine approach
    // Track groups and quantifiers to detect nested quantifier patterns like (a+)+
    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;

    // Stack to track if each group level has a quantifier inside
    let mut group_has_quantifier: Vec<bool> = Vec::new();

    while i < chars.len() {
        // Skip escaped characters
        if chars[i] == '\\' && i + 1 < chars.len() {
            i += 2;
            continue;
        }

        match chars[i] {
            '(' => {
                // Starting a new group
                group_has_quantifier.push(false);
            }
            ')' => {
                // Closing a group - check if next char is a quantifier
                let group_had_quantifier = group_has_quantifier.pop().unwrap_or(false);

                // Look ahead for quantifier after the closing paren
                if i + 1 < chars.len() {
                    let next = chars[i + 1];
                    if (next == '+' || next == '*') && group_had_quantifier {
                        // Nested quantifier detected: (something+)+ or (something*)*
                        return Err(anyhow!(
                            "Pattern contains nested quantifiers (e.g., (a+)+) which may cause catastrophic backtracking"
                        ));
                    }
                }
            }
            '+' | '*' => {
                // Mark current group (if any) as having a quantifier
                if let Some(last) = group_has_quantifier.last_mut() {
                    *last = true;
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Compile with default limits (regex crate has built-in size limits)
    Regex::new(pattern).map_err(|e| anyhow!("Invalid regex pattern: {}", e))
}

/// Validate entity name
pub fn validate_entity(entity: &str) -> Result<()> {
    if entity.is_empty() {
        return Err(anyhow!("Entity name cannot be empty"));
    }

    if entity.len() > MAX_ENTITY_LENGTH {
        return Err(anyhow!(
            "Entity name too long: {} chars (max: {})",
            entity.len(),
            MAX_ENTITY_LENGTH
        ));
    }

    // Only allow printable characters, no control characters
    if entity.chars().any(|c| c.is_control()) {
        return Err(anyhow!("Entity name contains invalid control characters"));
    }

    // No path traversal patterns
    if entity.contains("..") || entity.contains('/') || entity.contains('\\') {
        return Err(anyhow!("Entity name contains invalid path characters"));
    }

    Ok(())
}

/// Validate entities list
pub fn validate_entities(entities: &[String]) -> Result<()> {
    if entities.len() > MAX_ENTITIES_PER_MEMORY {
        return Err(anyhow!(
            "Too many entities: {} (max: {})",
            entities.len(),
            MAX_ENTITIES_PER_MEMORY
        ));
    }

    for entity in entities {
        validate_entity(entity)?;
    }

    Ok(())
}

/// Validate metadata JSON size
pub fn validate_metadata(metadata: &serde_json::Value) -> Result<()> {
    let size = metadata.to_string().len();
    if size > MAX_METADATA_SIZE {
        return Err(anyhow!(
            "Metadata too large: {} bytes (max: {})",
            size,
            MAX_METADATA_SIZE
        ));
    }
    Ok(())
}

/// Validate relationship strength
pub fn validate_relationship_strength(strength: f32) -> Result<()> {
    if !(0.0..=1.0).contains(&strength) {
        return Err(anyhow!(
            "Relationship strength must be between 0.0 and 1.0, got: {strength}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_user_id() {
        assert!(validate_user_id("alice").is_ok());
        assert!(validate_user_id("user-123").is_ok());
        assert!(validate_user_id("test_user").is_ok());
        assert!(validate_user_id("user@example.com").is_ok());
    }

    #[test]
    fn test_invalid_user_id() {
        assert!(validate_user_id("").is_err());  // empty
        assert!(validate_user_id("user/123").is_err());  // invalid char
        assert!(validate_user_id(&"a".repeat(200)).is_err());  // too long
    }

    #[test]
    fn test_valid_content() {
        assert!(validate_content("Hello world", false).is_ok());
        assert!(validate_content("", true).is_ok());  // allowed when allow_empty=true
    }

    #[test]
    fn test_invalid_content() {
        assert!(validate_content("", false).is_err());  // empty not allowed
        assert!(validate_content(&"x".repeat(100_000), false).is_err());  // too long
    }

    #[test]
    fn test_valid_embeddings() {
        let emb_384 = vec![0.5_f32; 384];
        assert!(validate_embeddings(&emb_384).is_ok());

        let emb_768 = vec![0.5_f32; 768];
        assert!(validate_embeddings(&emb_768).is_ok());
    }

    #[test]
    fn test_invalid_embeddings() {
        assert!(validate_embeddings(&[]).is_err());  // empty
        assert!(validate_embeddings(&[f32::NAN, 0.5]).is_err());  // NaN
        assert!(validate_embeddings(&vec![0.5; 999]).is_err());  // unusual dimension
    }

    #[test]
    fn test_importance_threshold() {
        assert!(validate_importance_threshold(0.0).is_ok());
        assert!(validate_importance_threshold(0.5).is_ok());
        assert!(validate_importance_threshold(1.0).is_ok());
        assert!(validate_importance_threshold(-0.1).is_err());
        assert!(validate_importance_threshold(1.5).is_err());
    }

    #[test]
    fn test_max_results() {
        assert!(validate_max_results(1).is_ok());
        assert!(validate_max_results(100).is_ok());
        assert!(validate_max_results(10_000).is_ok());
        assert!(validate_max_results(0).is_err());
        assert!(validate_max_results(20_000).is_err());
    }

    #[test]
    fn test_valid_patterns() {
        // Simple patterns should work
        assert!(validate_and_compile_pattern("hello").is_ok());
        assert!(validate_and_compile_pattern("user.*").is_ok());
        assert!(validate_and_compile_pattern("[a-z]+").is_ok());
        assert!(validate_and_compile_pattern("^start").is_ok());
        assert!(validate_and_compile_pattern("end$").is_ok());
    }

    #[test]
    fn test_redos_patterns_rejected() {
        // These patterns could cause catastrophic backtracking
        assert!(validate_and_compile_pattern("(a+)+").is_err());
        assert!(validate_and_compile_pattern("(.*)*").is_err());
        assert!(validate_and_compile_pattern("(.+)+").is_err());
        // Pattern too long
        assert!(validate_and_compile_pattern(&"a".repeat(300)).is_err());
        // Empty pattern
        assert!(validate_and_compile_pattern("").is_err());
    }

    #[test]
    fn test_valid_entity() {
        assert!(validate_entity("user").is_ok());
        assert!(validate_entity("John Doe").is_ok());
        assert!(validate_entity("entity-123").is_ok());
    }

    #[test]
    fn test_invalid_entity() {
        assert!(validate_entity("").is_err());  // empty
        assert!(validate_entity(&"a".repeat(300)).is_err());  // too long
        assert!(validate_entity("../etc/passwd").is_err());  // path traversal
        assert!(validate_entity("entity\x00null").is_err());  // control char
    }

    #[test]
    fn test_entities_list() {
        let valid: Vec<String> = vec!["a".to_string(), "b".to_string()];
        assert!(validate_entities(&valid).is_ok());

        // Too many entities
        let too_many: Vec<String> = (0..100).map(|i| format!("entity{}", i)).collect();
        assert!(validate_entities(&too_many).is_err());
    }

    #[test]
    fn test_relationship_strength() {
        assert!(validate_relationship_strength(0.0).is_ok());
        assert!(validate_relationship_strength(0.5).is_ok());
        assert!(validate_relationship_strength(1.0).is_ok());
        assert!(validate_relationship_strength(-0.1).is_err());
        assert!(validate_relationship_strength(1.1).is_err());
    }
}
