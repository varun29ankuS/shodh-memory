//! Semantic Fact Storage
//!
//! Persistent storage for semantic facts extracted from episodic memories.
//! Facts represent durable knowledge distilled from multiple experiences.
//!
//! Storage schema:
//! - `facts:{user_id}:{fact_id}` - Primary fact storage
//! - `facts_by_entity:{user_id}:{entity}:{fact_id}` - Entity index for fast lookup
//! - `facts_by_type:{user_id}:{type}:{fact_id}` - Type index

use anyhow::Result;
use rocksdb::{IteratorMode, DB};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::compression::{FactType, SemanticFact};

/// Response for fact queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactQueryResponse {
    pub facts: Vec<SemanticFact>,
    pub total: usize,
}

/// Statistics about semantic facts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactStats {
    pub total_facts: usize,
    pub by_type: std::collections::HashMap<String, usize>,
    pub avg_confidence: f32,
    pub avg_support: f32,
}

/// Storage for semantic facts with indexing
pub struct SemanticFactStore {
    db: Arc<DB>,
}

impl SemanticFactStore {
    /// Create a new fact store backed by RocksDB
    pub fn new(db: Arc<DB>) -> Self {
        Self { db }
    }

    /// Store a semantic fact
    pub fn store(&self, user_id: &str, fact: &SemanticFact) -> Result<()> {
        // Primary storage
        let key = format!("facts:{}:{}", user_id, fact.id);
        let value = bincode::serialize(fact)?;
        self.db.put(key.as_bytes(), &value)?;

        // Entity index - index by each related entity
        for entity in &fact.related_entities {
            let entity_key = format!(
                "facts_by_entity:{}:{}:{}",
                user_id,
                entity.to_lowercase(),
                fact.id
            );
            self.db.put(entity_key.as_bytes(), fact.id.as_bytes())?;
        }

        // Type index
        let type_name = format!("{:?}", fact.fact_type);
        let type_key = format!("facts_by_type:{}:{}:{}", user_id, type_name, fact.id);
        self.db.put(type_key.as_bytes(), fact.id.as_bytes())?;

        Ok(())
    }

    /// Store multiple facts in a batch
    pub fn store_batch(&self, user_id: &str, facts: &[SemanticFact]) -> Result<usize> {
        let mut stored = 0;
        for fact in facts {
            if self.store(user_id, fact).is_ok() {
                stored += 1;
            }
        }
        Ok(stored)
    }

    /// Get a fact by ID
    pub fn get(&self, user_id: &str, fact_id: &str) -> Result<Option<SemanticFact>> {
        let key = format!("facts:{}:{}", user_id, fact_id);
        match self.db.get(key.as_bytes())? {
            Some(data) => {
                let fact: SemanticFact = bincode::deserialize(&data)?;
                Ok(Some(fact))
            }
            None => Ok(None),
        }
    }

    /// Update an existing fact (for reinforcement)
    pub fn update(&self, user_id: &str, fact: &SemanticFact) -> Result<()> {
        // Simply overwrite - indices stay valid since ID doesn't change
        let key = format!("facts:{}:{}", user_id, fact.id);
        let value = bincode::serialize(fact)?;
        self.db.put(key.as_bytes(), &value)?;
        Ok(())
    }

    /// Delete a fact
    pub fn delete(&self, user_id: &str, fact_id: &str) -> Result<bool> {
        // Get fact first to clean up indices
        if let Some(fact) = self.get(user_id, fact_id)? {
            // Delete entity indices
            for entity in &fact.related_entities {
                let entity_key = format!(
                    "facts_by_entity:{}:{}:{}",
                    user_id,
                    entity.to_lowercase(),
                    fact_id
                );
                self.db.delete(entity_key.as_bytes())?;
            }

            // Delete type index
            let type_name = format!("{:?}", fact.fact_type);
            let type_key = format!("facts_by_type:{}:{}:{}", user_id, type_name, fact_id);
            self.db.delete(type_key.as_bytes())?;

            // Delete primary record
            let key = format!("facts:{}:{}", user_id, fact_id);
            self.db.delete(key.as_bytes())?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all facts for a user
    pub fn list(&self, user_id: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        let prefix = format!("facts:{}:", user_id);
        let mut facts = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            // Stop when we leave the prefix
            if !key_str.starts_with(&prefix) {
                break;
            }

            // Skip index keys (they contain extra colons)
            if key_str.matches(':').count() > 2 {
                continue;
            }

            if let Ok(fact) = bincode::deserialize::<SemanticFact>(&value) {
                facts.push(fact);
                if facts.len() >= limit {
                    break;
                }
            }
        }

        // Sort by confidence (highest first)
        facts.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(facts)
    }

    /// Find facts by related entity
    pub fn find_by_entity(&self, user_id: &str, entity: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        let prefix = format!("facts_by_entity:{}:{}:", user_id, entity.to_lowercase());
        let mut facts = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let fact_id = String::from_utf8_lossy(&value);
            if seen_ids.insert(fact_id.to_string()) {
                if let Some(fact) = self.get(user_id, &fact_id)? {
                    facts.push(fact);
                    if facts.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(facts)
    }

    /// Find facts by type
    pub fn find_by_type(&self, user_id: &str, fact_type: FactType, limit: usize) -> Result<Vec<SemanticFact>> {
        let type_name = format!("{:?}", fact_type);
        let prefix = format!("facts_by_type:{}:{}:", user_id, type_name);
        let mut facts = Vec::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(&prefix) {
                break;
            }

            let fact_id = String::from_utf8_lossy(&value);
            if let Some(fact) = self.get(user_id, &fact_id)? {
                facts.push(fact);
                if facts.len() >= limit {
                    break;
                }
            }
        }

        Ok(facts)
    }

    /// Search facts by keyword in fact content
    pub fn search(&self, user_id: &str, query: &str, limit: usize) -> Result<Vec<SemanticFact>> {
        let query_lower = query.to_lowercase();
        let all_facts = self.list(user_id, 1000)?; // Get all facts

        let mut matching: Vec<SemanticFact> = all_facts
            .into_iter()
            .filter(|f| f.fact.to_lowercase().contains(&query_lower))
            .collect();

        matching.truncate(limit);
        Ok(matching)
    }

    /// Get statistics about stored facts
    pub fn stats(&self, user_id: &str) -> Result<FactStats> {
        let facts = self.list(user_id, 10000)?;

        if facts.is_empty() {
            return Ok(FactStats::default());
        }

        let mut by_type: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        let mut total_confidence: f32 = 0.0;
        let mut total_support: usize = 0;

        for fact in &facts {
            let type_name = format!("{:?}", fact.fact_type);
            *by_type.entry(type_name).or_insert(0) += 1;
            total_confidence += fact.confidence;
            total_support += fact.support_count;
        }

        let count = facts.len();
        Ok(FactStats {
            total_facts: count,
            by_type,
            avg_confidence: total_confidence / count as f32,
            avg_support: total_support as f32 / count as f32,
        })
    }

    /// Find facts that should decay (no reinforcement for too long)
    pub fn find_decaying_facts(&self, user_id: &str, max_age_days: i64) -> Result<Vec<SemanticFact>> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days);
        let all_facts = self.list(user_id, 10000)?;

        let decaying: Vec<SemanticFact> = all_facts
            .into_iter()
            .filter(|f| f.last_reinforced < cutoff)
            .collect();

        Ok(decaying)
    }

    /// Check if a similar fact already exists (for deduplication)
    pub fn find_similar(&self, user_id: &str, fact_content: &str, threshold: f32) -> Result<Option<SemanticFact>> {
        let facts = self.list(user_id, 1000)?;
        let query_lower = fact_content.to_lowercase();

        // Simple substring matching for now (could use embeddings for semantic similarity)
        for fact in facts {
            let fact_lower = fact.fact.to_lowercase();

            // Check for significant overlap
            let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
            let fact_words: std::collections::HashSet<&str> = fact_lower.split_whitespace().collect();

            let intersection = query_words.intersection(&fact_words).count();
            let union = query_words.union(&fact_words).count();

            if union > 0 {
                let jaccard = intersection as f32 / union as f32;
                if jaccard >= threshold {
                    return Ok(Some(fact));
                }
            }
        }

        Ok(None)
    }

    /// List all unique user IDs that have facts
    pub fn list_users(&self, limit: usize) -> Result<Vec<String>> {
        let prefix = "facts:";
        let mut users = std::collections::HashSet::new();

        let iter = self.db.iterator(IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, _) = item?;
            let key_str = String::from_utf8_lossy(&key);

            if !key_str.starts_with(prefix) {
                break;
            }

            // Key format: facts:{user_id}:{fact_id}
            // Skip index keys (facts_by_entity, facts_by_type)
            if key_str.starts_with("facts_by_") {
                continue;
            }

            // Extract user_id from key
            let parts: Vec<&str> = key_str.splitn(3, ':').collect();
            if parts.len() >= 2 {
                users.insert(parts[1].to_string());
                if users.len() >= limit {
                    break;
                }
            }
        }

        Ok(users.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_store() -> (SemanticFactStore, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db = Arc::new(DB::open_default(temp_dir.path()).unwrap());
        (SemanticFactStore::new(db), temp_dir)
    }

    fn create_test_fact(id: &str, content: &str) -> SemanticFact {
        SemanticFact {
            id: id.to_string(),
            fact: content.to_string(),
            confidence: 0.8,
            support_count: 3,
            source_memories: vec![],
            related_entities: vec!["rust".to_string(), "memory".to_string()],
            created_at: chrono::Utc::now(),
            last_reinforced: chrono::Utc::now(),
            fact_type: FactType::Pattern,
        }
    }

    #[test]
    fn test_store_and_get() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Rust is a systems programming language");

        store.store("user-1", &fact).unwrap();
        let retrieved = store.get("user-1", "fact-1").unwrap();

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().fact, "Rust is a systems programming language");
    }

    #[test]
    fn test_find_by_entity() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Rust has efficient memory management");

        store.store("user-1", &fact).unwrap();
        let results = store.find_by_entity("user-1", "rust", 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "fact-1");
    }

    #[test]
    fn test_find_by_type() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Pattern detected in codebase");

        store.store("user-1", &fact).unwrap();
        let results = store.find_by_type("user-1", FactType::Pattern, 10).unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_delete() {
        let (store, _dir) = create_test_store();
        let fact = create_test_fact("fact-1", "Test fact");

        store.store("user-1", &fact).unwrap();
        assert!(store.get("user-1", "fact-1").unwrap().is_some());

        store.delete("user-1", "fact-1").unwrap();
        assert!(store.get("user-1", "fact-1").unwrap().is_none());

        // Entity index should also be cleaned up
        let by_entity = store.find_by_entity("user-1", "rust", 10).unwrap();
        assert!(by_entity.is_empty());
    }

    #[test]
    fn test_stats() {
        let (store, _dir) = create_test_store();

        store.store("user-1", &create_test_fact("fact-1", "Fact one")).unwrap();
        store.store("user-1", &create_test_fact("fact-2", "Fact two")).unwrap();

        let stats = store.stats("user-1").unwrap();
        assert_eq!(stats.total_facts, 2);
        assert!(stats.avg_confidence > 0.0);
    }
}
