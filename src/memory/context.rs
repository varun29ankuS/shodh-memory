//! Context Management - Building and merging rich contexts

use super::types::*;
use anyhow::Result;
use chrono::{Datelike, Timelike, Utc};
use std::collections::HashMap;
use uuid::Uuid;

/// Context builder for creating rich contexts
///
/// Public API for external callers to construct RichContext objects.
/// Used internally by ContextManager and available for custom context creation.
#[allow(unused)] // Public API
pub struct ContextBuilder {
    context: RichContext,
}

#[allow(unused)] // Public API methods
impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unused)] // Public API methods
impl ContextBuilder {
    /// Create new context builder
    pub fn new() -> Self {
        Self {
            context: RichContext {
                id: ContextId(Uuid::new_v4()),
                conversation: ConversationContext::default(),
                user: UserContext::default(),
                project: ProjectContext::default(),
                temporal: TemporalContext::default(),
                semantic: SemanticContext::default(),
                code: CodeContext::default(),
                document: DocumentContext::default(),
                environment: EnvironmentContext::default(),
                // SHO-104: Richer context encoding
                emotional: EmotionalContext::default(),
                source: SourceContext::default(),
                episode: EpisodeContext::default(),
                parent: None,
                embeddings: None,
                decay_rate: 0.1, // Default decay
                created_at: Utc::now(),
                updated_at: Utc::now(),
            },
        }
    }

    /// Set conversation context
    pub fn with_conversation(mut self, conv: ConversationContext) -> Self {
        self.context.conversation = conv;
        self
    }

    /// Set user context
    pub fn with_user(mut self, user: UserContext) -> Self {
        self.context.user = user;
        self
    }

    /// Set project context
    pub fn with_project(mut self, project: ProjectContext) -> Self {
        self.context.project = project;
        self
    }

    /// Set code context
    pub fn with_code(mut self, code: CodeContext) -> Self {
        self.context.code = code;
        self
    }

    /// Set semantic context
    pub fn with_semantic(mut self, semantic: SemanticContext) -> Self {
        self.context.semantic = semantic;
        self
    }

    /// Set document context
    pub fn with_document(mut self, doc: DocumentContext) -> Self {
        self.context.document = doc;
        self
    }

    /// Set parent context
    pub fn with_parent(mut self, parent: RichContext) -> Self {
        self.context.parent = Some(Box::new(parent));
        self
    }

    /// Set decay rate
    pub fn with_decay_rate(mut self, rate: f32) -> Self {
        self.context.decay_rate = rate;
        self
    }

    /// Set temporal context
    pub fn with_temporal(mut self, temporal: TemporalContext) -> Self {
        self.context.temporal = temporal;
        self
    }

    // SHO-104: Builder methods for richer context encoding

    /// Set emotional context (valence, arousal, dominant emotion)
    pub fn with_emotional(mut self, emotional: EmotionalContext) -> Self {
        self.context.emotional = emotional;
        self
    }

    /// Set source context (source type, credibility)
    pub fn with_source(mut self, source: SourceContext) -> Self {
        self.context.source = source;
        self
    }

    /// Set episode context (episode ID, sequence, temporal chain)
    pub fn with_episode(mut self, episode: EpisodeContext) -> Self {
        self.context.episode = episode;
        self
    }

    /// Build the context
    pub fn build(self) -> RichContext {
        self.context
    }
}

/// Context manager for automatic context capture
///
/// Public API for managing conversation context across sessions.
/// Tracks user profile, project state, and temporal patterns.
#[allow(unused)] // Public API
pub struct ContextManager {
    /// Current active context
    current_context: Option<RichContext>,

    /// Context history (sliding window)
    context_history: Vec<RichContext>,
    max_history: usize,

    /// User profile (accumulated over time)
    user_profile: UserContext,

    /// Project state
    project_state: HashMap<String, ProjectContext>,
}

#[allow(unused)] // Public API
impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unused)] // Public API methods
impl ContextManager {
    /// Create new context manager
    pub fn new() -> Self {
        Self {
            current_context: None,
            context_history: Vec::new(),
            max_history: 50,
            user_profile: UserContext::default(),
            project_state: HashMap::new(),
        }
    }

    /// Capture current context automatically
    pub fn capture_context(&mut self) -> Result<RichContext> {
        let mut builder = ContextBuilder::new();

        // Capture temporal context
        let now = Utc::now();
        let temporal = TemporalContext {
            time_of_day: Some(format!("{:02}:{:02}", now.hour(), now.minute())),
            day_of_week: Some(now.weekday().to_string()),
            session_duration_minutes: self.calculate_session_duration(),
            time_since_last_interaction: self.calculate_time_since_last(),
            patterns: self.detect_temporal_patterns(),
            trends: Vec::new(),
        };

        builder = builder
            .with_temporal(temporal)
            .with_user(self.user_profile.clone());

        // Add parent context if available
        if let Some(parent) = &self.current_context {
            builder = builder.with_parent(parent.clone());
        }

        // Build context
        let context = builder.build();

        // Update current context
        self.current_context = Some(context.clone());

        // Add to history
        self.context_history.push(context.clone());
        if self.context_history.len() > self.max_history {
            self.context_history.remove(0);
        }

        Ok(context)
    }

    /// Update user profile based on interactions
    pub fn update_user_profile(&mut self, experience: &Experience) {
        // Extract entities mentioned
        for entity in &experience.entities {
            if !self.user_profile.expertise.contains(entity) {
                // Check if mentioned frequently
                let count = self
                    .context_history
                    .iter()
                    .filter(|ctx| ctx.conversation.mentioned_entities.contains(entity))
                    .count();

                if count > 5 {
                    self.user_profile.expertise.push(entity.clone());
                }
            }
        }

        // Extract preferences from metadata
        if let Some(ctx) = &experience.context {
            for (key, value) in &ctx.user.preferences {
                self.user_profile
                    .preferences
                    .insert(key.clone(), value.clone());
            }
        }
    }

    /// Update conversation context
    pub fn update_conversation(
        &mut self,
        conv_id: String,
        message: String,
        topic: Option<String>,
    ) -> Result<()> {
        if let Some(ref mut ctx) = self.current_context {
            ctx.conversation.conversation_id = Some(conv_id);
            ctx.conversation.recent_messages.push(message.clone());

            // Keep only last 10 messages
            if ctx.conversation.recent_messages.len() > 10 {
                ctx.conversation.recent_messages.remove(0);
            }

            if let Some(t) = topic {
                ctx.conversation.topic = Some(t);
            }

            ctx.updated_at = Utc::now();
        }
        Ok(())
    }

    /// Update code context
    pub fn update_code_context(&mut self, file: String, scope: Option<String>) -> Result<()> {
        if let Some(ref mut ctx) = self.current_context {
            ctx.code.current_file = Some(file.clone());
            ctx.code.current_scope = scope;

            // Track recent edits
            if !ctx.code.recent_edits.contains(&file) {
                ctx.code.recent_edits.push(file);
            }

            // Keep only last 20 edits
            if ctx.code.recent_edits.len() > 20 {
                ctx.code.recent_edits.remove(0);
            }

            ctx.updated_at = Utc::now();
        }
        Ok(())
    }

    /// Update project context
    pub fn update_project_context(
        &mut self,
        project_id: String,
        project: ProjectContext,
    ) -> Result<()> {
        self.project_state
            .insert(project_id.clone(), project.clone());

        if let Some(ref mut ctx) = self.current_context {
            ctx.project = project;
            ctx.updated_at = Utc::now();
        }
        Ok(())
    }

    /// Get current context
    pub fn get_current_context(&self) -> Option<&RichContext> {
        self.current_context.as_ref()
    }

    /// Merge multiple contexts
    pub fn merge_contexts(&self, contexts: Vec<RichContext>) -> RichContext {
        if contexts.is_empty() {
            return ContextBuilder::new().build();
        }

        let mut merged = contexts[0].clone();

        for ctx in contexts.iter().skip(1) {
            // Merge conversation contexts
            merged
                .conversation
                .recent_messages
                .extend(ctx.conversation.recent_messages.clone());
            merged
                .conversation
                .mentioned_entities
                .extend(ctx.conversation.mentioned_entities.clone());
            merged
                .conversation
                .active_intents
                .extend(ctx.conversation.active_intents.clone());

            // Merge code contexts
            merged
                .code
                .related_files
                .extend(ctx.code.related_files.clone());
            merged
                .code
                .recent_edits
                .extend(ctx.code.recent_edits.clone());
            merged.code.patterns.extend(ctx.code.patterns.clone());

            // Merge semantic contexts
            merged
                .semantic
                .concepts
                .extend(ctx.semantic.concepts.clone());
            merged
                .semantic
                .related_concepts
                .extend(ctx.semantic.related_concepts.clone());
            merged.semantic.tags.extend(ctx.semantic.tags.clone());

            // Deduplicate
            merged.conversation.mentioned_entities.sort();
            merged.conversation.mentioned_entities.dedup();
            merged.code.related_files.sort();
            merged.code.related_files.dedup();
            merged.semantic.concepts.sort();
            merged.semantic.concepts.dedup();
        }

        merged.updated_at = Utc::now();
        merged
    }

    /// Find similar contexts from history
    pub fn find_similar_contexts(
        &self,
        query_context: &RichContext,
        top_k: usize,
    ) -> Vec<RichContext> {
        let mut scored_contexts: Vec<(f32, RichContext)> = self
            .context_history
            .iter()
            .map(|ctx| {
                let score = self.calculate_context_similarity(query_context, ctx);
                (score, ctx.clone())
            })
            .collect();

        scored_contexts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored_contexts
            .into_iter()
            .take(top_k)
            .map(|(_, ctx)| ctx)
            .collect()
    }

    /// Calculate similarity between two contexts
    fn calculate_context_similarity(&self, ctx1: &RichContext, ctx2: &RichContext) -> f32 {
        let mut score = 0.0;

        // Conversation similarity
        let conv_overlap = ctx1
            .conversation
            .mentioned_entities
            .iter()
            .filter(|e| ctx2.conversation.mentioned_entities.contains(e))
            .count();
        score += conv_overlap as f32 * 0.2;

        // Project similarity
        if ctx1.project.project_id == ctx2.project.project_id {
            score += 0.3;
        }

        // Code similarity
        let code_overlap = ctx1
            .code
            .related_files
            .iter()
            .filter(|f| ctx2.code.related_files.contains(f))
            .count();
        score += code_overlap as f32 * 0.15;

        // Semantic similarity
        let concept_overlap = ctx1
            .semantic
            .concepts
            .iter()
            .filter(|c| ctx2.semantic.concepts.contains(c))
            .count();
        score += concept_overlap as f32 * 0.25;

        // Temporal decay
        let time_diff = (Utc::now() - ctx2.created_at).num_hours() as f32;
        let decay_factor = (-ctx2.decay_rate * time_diff / 24.0).exp();
        score *= decay_factor;

        score
    }

    /// Helper: Calculate session duration
    fn calculate_session_duration(&self) -> Option<u32> {
        self.context_history.first().map(|first| {
            let duration = Utc::now() - first.created_at;
            duration.num_minutes() as u32
        })
    }

    /// Helper: Calculate time since last interaction
    fn calculate_time_since_last(&self) -> Option<i64> {
        self.context_history
            .last()
            .map(|last| (Utc::now() - last.updated_at).num_seconds())
    }

    /// Helper: Detect temporal patterns
    fn detect_temporal_patterns(&self) -> Vec<TimePattern> {
        // Simple pattern detection - can be enhanced
        Vec::new()
    }

    /// Get user expertise areas
    pub fn get_user_expertise(&self) -> &Vec<String> {
        &self.user_profile.expertise
    }

    /// Add entity to user expertise
    pub fn add_user_expertise(&mut self, entity: String) {
        if !self.user_profile.expertise.contains(&entity) {
            self.user_profile.expertise.push(entity);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_builder() {
        let ctx = ContextBuilder::new()
            .with_conversation(ConversationContext {
                topic: Some("Testing".to_string()),
                ..Default::default()
            })
            .build();

        assert_eq!(ctx.conversation.topic, Some("Testing".to_string()));
    }

    #[test]
    fn test_context_manager() {
        let mut manager = ContextManager::new();
        let ctx = manager.capture_context().unwrap();

        assert!(ctx.temporal.time_of_day.is_some());
    }
}
