//! Linear integration for syncing issues to Shodh memory
//!
//! Provides:
//! - Webhook receiver for real-time issue updates
//! - Bulk sync for importing existing issues
//! - HMAC-SHA256 signature verification

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

// =============================================================================
// LINEAR WEBHOOK TYPES
// =============================================================================

/// Linear webhook payload structure
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinearWebhookPayload {
    /// Action type: "create", "update", "remove"
    pub action: String,
    /// Actor who triggered the webhook (user info)
    #[serde(default)]
    pub actor: Option<LinearActor>,
    /// Timestamp of the webhook
    #[serde(default)]
    pub created_at: Option<String>,
    /// The data payload (varies by type)
    pub data: LinearIssueData,
    /// Type of entity: "Issue", "Comment", "Project", etc.
    #[serde(rename = "type")]
    pub entity_type: String,
    /// URL to the entity in Linear
    #[serde(default)]
    pub url: Option<String>,
    /// Organization ID
    #[serde(default)]
    pub organization_id: Option<String>,
    /// Webhook ID
    #[serde(default)]
    pub webhook_id: Option<String>,
    /// Webhook timestamp
    #[serde(default)]
    pub webhook_timestamp: Option<i64>,
}

/// Actor who triggered the webhook
#[derive(Debug, Clone, Deserialize)]
pub struct LinearActor {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(rename = "type")]
    #[serde(default)]
    pub actor_type: Option<String>,
}

/// Linear issue data from webhook
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LinearIssueData {
    /// UUID of the issue
    pub id: String,
    /// Human-readable identifier (e.g., "SHO-39")
    #[serde(default)]
    pub identifier: Option<String>,
    /// Issue title
    #[serde(default)]
    pub title: Option<String>,
    /// Issue description (markdown)
    #[serde(default)]
    pub description: Option<String>,
    /// Priority (0-4, where 1=Urgent, 4=Low, 0=None)
    #[serde(default)]
    pub priority: Option<i32>,
    /// Priority label
    #[serde(default)]
    pub priority_label: Option<String>,
    /// Issue state
    #[serde(default)]
    pub state: Option<LinearState>,
    /// Assignee
    #[serde(default)]
    pub assignee: Option<LinearUser>,
    /// Creator
    #[serde(default)]
    pub creator: Option<LinearUser>,
    /// Labels
    #[serde(default)]
    pub labels: Vec<LinearLabel>,
    /// Team
    #[serde(default)]
    pub team: Option<LinearTeam>,
    /// Project
    #[serde(default)]
    pub project: Option<LinearProject>,
    /// Cycle
    #[serde(default)]
    pub cycle: Option<LinearCycle>,
    /// Parent issue (for sub-issues)
    #[serde(default)]
    pub parent: Option<Box<LinearIssueData>>,
    /// Due date
    #[serde(default)]
    pub due_date: Option<String>,
    /// Estimate (story points)
    #[serde(default)]
    pub estimate: Option<f32>,
    /// Created timestamp
    #[serde(default)]
    pub created_at: Option<String>,
    /// Updated timestamp
    #[serde(default)]
    pub updated_at: Option<String>,
    /// Completed timestamp
    #[serde(default)]
    pub completed_at: Option<String>,
    /// Canceled timestamp
    #[serde(default)]
    pub canceled_at: Option<String>,
    /// URL to the issue
    #[serde(default)]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearState {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(rename = "type")]
    #[serde(default)]
    pub state_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearUser {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearLabel {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub color: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearTeam {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub key: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearProject {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LinearCycle {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub number: Option<i32>,
}

// =============================================================================
// WEBHOOK HANDLER
// =============================================================================

/// Linear webhook handler
pub struct LinearWebhook {
    /// Webhook signing secret for HMAC verification
    signing_secret: Option<String>,
}

impl LinearWebhook {
    /// Create a new webhook handler
    pub fn new(signing_secret: Option<String>) -> Self {
        Self { signing_secret }
    }

    /// Verify webhook signature using HMAC-SHA256
    ///
    /// Linear sends the signature in the `Linear-Signature` header
    pub fn verify_signature(&self, body: &[u8], signature: &str) -> Result<bool> {
        let secret = match &self.signing_secret {
            Some(s) => s,
            None => {
                tracing::warn!("No signing secret configured, skipping signature verification");
                return Ok(true);
            }
        };

        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).context("Invalid signing secret")?;
        mac.update(body);

        // Linear signature format: "sha256=<hex>"
        let expected_sig = signature.strip_prefix("sha256=").unwrap_or(signature);

        let expected_bytes = hex::decode(expected_sig).context("Invalid signature format")?;

        Ok(mac.verify_slice(&expected_bytes).is_ok())
    }

    /// Parse webhook payload
    pub fn parse_payload(&self, body: &[u8]) -> Result<LinearWebhookPayload> {
        serde_json::from_slice(body).context("Failed to parse Linear webhook payload")
    }

    /// Transform Linear issue to memory content
    ///
    /// Creates a structured text representation suitable for semantic search
    pub fn issue_to_content(issue: &LinearIssueData) -> String {
        let mut parts = Vec::new();

        // Header: Identifier and title
        if let Some(id) = &issue.identifier {
            if let Some(title) = &issue.title {
                parts.push(format!("{}: {}", id, title));
            } else {
                parts.push(id.clone());
            }
        } else if let Some(title) = &issue.title {
            parts.push(title.clone());
        }

        // Metadata section
        let mut metadata = Vec::new();

        if let Some(state) = &issue.state {
            metadata.push(format!("Status: {}", state.name));
        }

        if let Some(assignee) = &issue.assignee {
            if let Some(name) = &assignee.name {
                metadata.push(format!("Assignee: {}", name));
            }
        }

        if let Some(priority) = &issue.priority_label {
            metadata.push(format!("Priority: {}", priority));
        }

        if !issue.labels.is_empty() {
            let label_names: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
            metadata.push(format!("Labels: {}", label_names.join(", ")));
        }

        if let Some(project) = &issue.project {
            if let Some(name) = &project.name {
                metadata.push(format!("Project: {}", name));
            }
        }

        if let Some(cycle) = &issue.cycle {
            if let Some(name) = &cycle.name {
                metadata.push(format!("Cycle: {}", name));
            } else if let Some(num) = cycle.number {
                metadata.push(format!("Cycle: #{}", num));
            }
        }

        if let Some(due) = &issue.due_date {
            metadata.push(format!("Due: {}", due));
        }

        if let Some(estimate) = issue.estimate {
            metadata.push(format!("Estimate: {} points", estimate));
        }

        if !metadata.is_empty() {
            parts.push(metadata.join(" | "));
        }

        // Description
        if let Some(desc) = &issue.description {
            if !desc.is_empty() {
                parts.push(String::new()); // Empty line
                parts.push(desc.clone());
            }
        }

        parts.join("\n")
    }

    /// Extract tags from Linear issue
    pub fn issue_to_tags(issue: &LinearIssueData) -> Vec<String> {
        let mut tags = vec!["linear".to_string()];

        // Add identifier as tag
        if let Some(id) = &issue.identifier {
            tags.push(id.clone());
        }

        // Add labels as tags
        for label in &issue.labels {
            tags.push(label.name.clone());
        }

        // Add state as tag
        if let Some(state) = &issue.state {
            tags.push(state.name.clone());
        }

        // Add team key as tag
        if let Some(team) = &issue.team {
            if let Some(key) = &team.key {
                tags.push(key.clone());
            }
        }

        // Add project name as tag
        if let Some(project) = &issue.project {
            if let Some(name) = &project.name {
                tags.push(name.clone());
            }
        }

        tags
    }

    /// Determine change type from webhook action and issue state
    pub fn determine_change_type(action: &str, issue: &LinearIssueData) -> String {
        match action {
            "create" => "created".to_string(),
            "remove" => "content_updated".to_string(), // Treat removal as update (soft delete)
            "update" => {
                // Try to determine what changed
                if issue.completed_at.is_some() || issue.canceled_at.is_some() {
                    "status_changed".to_string()
                } else {
                    "content_updated".to_string()
                }
            }
            _ => "content_updated".to_string(),
        }
    }
}

// =============================================================================
// BULK SYNC TYPES
// =============================================================================

/// Request for bulk syncing Linear issues
#[derive(Debug, Deserialize)]
pub struct LinearSyncRequest {
    /// User ID to associate memories with
    pub user_id: String,
    /// Linear API key
    pub api_key: String,
    /// Optional team ID to filter issues
    #[serde(default)]
    pub team_id: Option<String>,
    /// Optional: only sync issues updated after this date (ISO 8601)
    #[serde(default)]
    pub updated_after: Option<String>,
    /// Optional: limit number of issues to sync
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response from bulk sync
#[derive(Debug, Serialize)]
pub struct LinearSyncResponse {
    /// Number of issues synced
    pub synced_count: usize,
    /// Number of issues created (new)
    pub created_count: usize,
    /// Number of issues updated (existing)
    pub updated_count: usize,
    /// Number of issues that failed
    pub error_count: usize,
    /// Error messages if any
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
}

// =============================================================================
// LINEAR API CLIENT
// =============================================================================

/// Simple Linear GraphQL API client for bulk sync
pub struct LinearClient {
    api_key: String,
    api_url: String,
    client: reqwest::Client,
}

impl LinearClient {
    const DEFAULT_API_URL: &'static str = "https://api.linear.app/graphql";

    pub fn new(api_key: String) -> Self {
        let api_url =
            std::env::var("LINEAR_API_URL").unwrap_or_else(|_| Self::DEFAULT_API_URL.to_string());
        Self {
            api_key,
            api_url,
            client: reqwest::Client::new(),
        }
    }

    /// Fetch issues from Linear using GraphQL
    pub async fn fetch_issues(
        &self,
        team_id: Option<&str>,
        updated_after: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<LinearIssueData>> {
        let limit = limit.unwrap_or(250);

        // Build filter
        let mut filters = Vec::new();
        if let Some(tid) = team_id {
            filters.push(format!(r#"team: {{ id: {{ eq: "{}" }} }}"#, tid));
        }
        if let Some(after) = updated_after {
            filters.push(format!(r#"updatedAt: {{ gte: "{}" }}"#, after));
        }

        let filter_str = if filters.is_empty() {
            String::new()
        } else {
            format!("filter: {{ {} }}", filters.join(", "))
        };

        let query = format!(
            r#"
            query {{
                issues(first: {}, {}) {{
                    nodes {{
                        id
                        identifier
                        title
                        description
                        priority
                        priorityLabel
                        url
                        createdAt
                        updatedAt
                        completedAt
                        canceledAt
                        dueDate
                        estimate
                        state {{
                            id
                            name
                            color
                            type
                        }}
                        assignee {{
                            id
                            name
                            email
                        }}
                        creator {{
                            id
                            name
                            email
                        }}
                        labels {{
                            nodes {{
                                id
                                name
                                color
                            }}
                        }}
                        team {{
                            id
                            name
                            key
                        }}
                        project {{
                            id
                            name
                        }}
                        cycle {{
                            id
                            name
                            number
                        }}
                        parent {{
                            id
                            identifier
                            title
                        }}
                    }}
                }}
            }}
        "#,
            limit, filter_str
        );

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "query": query }))
            .send()
            .await
            .context("Failed to send request to Linear API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Linear API error: {} - {}", status, body);
        }

        let body: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse Linear API response")?;

        // Check for GraphQL errors
        if let Some(errors) = body.get("errors") {
            anyhow::bail!("Linear GraphQL errors: {:?}", errors);
        }

        // Parse issues from response
        let issues_raw = body
            .get("data")
            .and_then(|d| d.get("issues"))
            .and_then(|i| i.get("nodes"))
            .context("Unexpected Linear API response structure")?;

        // Transform to our structure (handling nested labels)
        let issues: Vec<LinearIssueData> = issues_raw
            .as_array()
            .context("Expected issues array")?
            .iter()
            .filter_map(|issue| {
                // Handle labels.nodes -> labels transformation
                let mut issue_obj = issue.clone();
                if let Some(labels) = issue_obj.get("labels").and_then(|l| l.get("nodes")) {
                    issue_obj["labels"] = labels.clone();
                }
                serde_json::from_value(issue_obj).ok()
            })
            .collect();

        Ok(issues)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_issue_to_content() {
        let issue = LinearIssueData {
            id: "uuid".to_string(),
            identifier: Some("SHO-39".to_string()),
            title: Some("Test Issue".to_string()),
            description: Some("This is a test".to_string()),
            priority: Some(2),
            priority_label: Some("High".to_string()),
            state: Some(LinearState {
                id: "state-id".to_string(),
                name: "In Progress".to_string(),
                color: None,
                state_type: None,
            }),
            assignee: Some(LinearUser {
                id: "user-id".to_string(),
                name: Some("Varun".to_string()),
                email: None,
            }),
            creator: None,
            labels: vec![LinearLabel {
                id: "label-id".to_string(),
                name: "Feature".to_string(),
                color: None,
            }],
            team: None,
            project: None,
            cycle: None,
            parent: None,
            due_date: None,
            estimate: None,
            created_at: None,
            updated_at: None,
            completed_at: None,
            canceled_at: None,
            url: None,
        };

        let content = LinearWebhook::issue_to_content(&issue);
        assert!(content.contains("SHO-39: Test Issue"));
        assert!(content.contains("Status: In Progress"));
        assert!(content.contains("Assignee: Varun"));
        assert!(content.contains("Labels: Feature"));
        assert!(content.contains("This is a test"));
    }

    #[test]
    fn test_issue_to_tags() {
        let issue = LinearIssueData {
            id: "uuid".to_string(),
            identifier: Some("SHO-39".to_string()),
            title: None,
            description: None,
            priority: None,
            priority_label: None,
            state: Some(LinearState {
                id: "state-id".to_string(),
                name: "In Progress".to_string(),
                color: None,
                state_type: None,
            }),
            assignee: None,
            creator: None,
            labels: vec![LinearLabel {
                id: "label-id".to_string(),
                name: "Feature".to_string(),
                color: None,
            }],
            team: Some(LinearTeam {
                id: "team-id".to_string(),
                name: Some("Shodh".to_string()),
                key: Some("SHO".to_string()),
            }),
            project: None,
            cycle: None,
            parent: None,
            due_date: None,
            estimate: None,
            created_at: None,
            updated_at: None,
            completed_at: None,
            canceled_at: None,
            url: None,
        };

        let tags = LinearWebhook::issue_to_tags(&issue);
        assert!(tags.contains(&"linear".to_string()));
        assert!(tags.contains(&"SHO-39".to_string()));
        assert!(tags.contains(&"Feature".to_string()));
        assert!(tags.contains(&"In Progress".to_string()));
        assert!(tags.contains(&"SHO".to_string()));
    }
}
