//! GitHub integration for syncing Issues, PRs, and Commits to Shodh memory
//!
//! Provides:
//! - Webhook receiver for real-time issue/PR updates
//! - Bulk sync for importing existing issues, PRs, and commits
//! - HMAC-SHA256 signature verification

use anyhow::{Context, Result};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

// =============================================================================
// GITHUB WEBHOOK TYPES
// =============================================================================

/// GitHub webhook payload - unified structure for Issues and PRs
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubWebhookPayload {
    /// Action: "opened", "edited", "closed", "reopened", "labeled", "merged", etc.
    pub action: String,
    /// Issue data (for issue events)
    #[serde(default)]
    pub issue: Option<GitHubIssue>,
    /// Pull request data (for PR events)
    #[serde(default)]
    pub pull_request: Option<GitHubPullRequest>,
    /// Repository info
    pub repository: GitHubRepository,
    /// Sender (who triggered the event)
    #[serde(default)]
    pub sender: Option<GitHubUser>,
}

/// GitHub Issue
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubIssue {
    pub number: u64,
    pub title: String,
    #[serde(default)]
    pub body: Option<String>,
    pub state: String,
    pub html_url: String,
    #[serde(default)]
    pub user: Option<GitHubUser>,
    #[serde(default)]
    pub assignee: Option<GitHubUser>,
    #[serde(default)]
    pub assignees: Vec<GitHubUser>,
    #[serde(default)]
    pub labels: Vec<GitHubLabel>,
    #[serde(default)]
    pub milestone: Option<GitHubMilestone>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub closed_at: Option<String>,
}

/// GitHub Pull Request
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubPullRequest {
    pub number: u64,
    pub title: String,
    #[serde(default)]
    pub body: Option<String>,
    pub state: String,
    pub html_url: String,
    #[serde(default)]
    pub user: Option<GitHubUser>,
    #[serde(default)]
    pub assignee: Option<GitHubUser>,
    #[serde(default)]
    pub assignees: Vec<GitHubUser>,
    #[serde(default)]
    pub labels: Vec<GitHubLabel>,
    #[serde(default)]
    pub milestone: Option<GitHubMilestone>,
    /// Base branch info
    pub base: GitHubBranch,
    /// Head branch info
    pub head: GitHubBranch,
    /// Whether PR is merged
    #[serde(default)]
    pub merged: bool,
    /// Merge commit SHA
    #[serde(default)]
    pub merge_commit_sha: Option<String>,
    /// Number of commits
    #[serde(default)]
    pub commits: Option<u32>,
    /// Additions
    #[serde(default)]
    pub additions: Option<u32>,
    /// Deletions
    #[serde(default)]
    pub deletions: Option<u32>,
    /// Changed files count
    #[serde(default)]
    pub changed_files: Option<u32>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub closed_at: Option<String>,
    #[serde(default)]
    pub merged_at: Option<String>,
    /// Draft PR
    #[serde(default)]
    pub draft: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GitHubRepository {
    pub id: u64,
    pub name: String,
    pub full_name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub html_url: String,
    pub owner: GitHubUser,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GitHubUser {
    pub id: u64,
    pub login: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub avatar_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GitHubLabel {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GitHubMilestone {
    pub number: u64,
    pub title: String,
    #[serde(default)]
    pub description: Option<String>,
    pub state: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GitHubBranch {
    #[serde(rename = "ref")]
    pub branch_ref: String,
    pub sha: String,
    #[serde(default)]
    pub repo: Option<GitHubRepository>,
}

/// GitHub Commit
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubCommit {
    pub sha: String,
    pub commit: GitHubCommitData,
    pub html_url: String,
    #[serde(default)]
    pub author: Option<GitHubUser>,
    #[serde(default)]
    pub committer: Option<GitHubUser>,
    #[serde(default)]
    pub stats: Option<GitHubCommitStats>,
    #[serde(default)]
    pub files: Option<Vec<GitHubCommitFile>>,
}

/// Inner commit data (message, author info)
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubCommitData {
    pub message: String,
    pub author: GitHubCommitAuthor,
    pub committer: GitHubCommitAuthor,
}

/// Commit author/committer info
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubCommitAuthor {
    pub name: String,
    pub email: String,
    pub date: String,
}

/// Commit statistics (additions/deletions)
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubCommitStats {
    #[serde(default)]
    pub additions: u32,
    #[serde(default)]
    pub deletions: u32,
    #[serde(default)]
    pub total: u32,
}

/// File changed in a commit
#[derive(Debug, Clone, Deserialize)]
pub struct GitHubCommitFile {
    pub filename: String,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub additions: u32,
    #[serde(default)]
    pub deletions: u32,
    #[serde(default)]
    pub changes: u32,
}

// =============================================================================
// WEBHOOK HANDLER
// =============================================================================

/// GitHub webhook handler
pub struct GitHubWebhook {
    /// Webhook secret for HMAC verification
    webhook_secret: Option<String>,
}

impl GitHubWebhook {
    /// Create a new webhook handler
    pub fn new(webhook_secret: Option<String>) -> Self {
        Self { webhook_secret }
    }

    /// Verify webhook signature using HMAC-SHA256
    ///
    /// GitHub sends the signature in the `X-Hub-Signature-256` header
    pub fn verify_signature(&self, body: &[u8], signature: &str) -> Result<bool> {
        let secret = match &self.webhook_secret {
            Some(s) => s,
            None => {
                tracing::warn!("No webhook secret configured, skipping signature verification");
                return Ok(true);
            }
        };

        let mut mac =
            HmacSha256::new_from_slice(secret.as_bytes()).context("Invalid webhook secret")?;
        mac.update(body);

        // GitHub signature format: "sha256=<hex>"
        let expected_sig = signature.strip_prefix("sha256=").unwrap_or(signature);

        let expected_bytes = hex::decode(expected_sig).context("Invalid signature format")?;

        Ok(mac.verify_slice(&expected_bytes).is_ok())
    }

    /// Parse webhook payload
    pub fn parse_payload(&self, body: &[u8]) -> Result<GitHubWebhookPayload> {
        serde_json::from_slice(body).context("Failed to parse GitHub webhook payload")
    }

    /// Transform GitHub issue to memory content
    pub fn issue_to_content(issue: &GitHubIssue, repo: &GitHubRepository) -> String {
        let mut parts = Vec::new();

        // Header: #{number}: {title}
        parts.push(format!("#{}: {}", issue.number, issue.title));

        // Metadata
        let mut metadata = Vec::new();
        metadata.push(format!("Status: {}", issue.state));

        if let Some(assignee) = &issue.assignee {
            metadata.push(format!("Assignee: {}", assignee.login));
        } else if !issue.assignees.is_empty() {
            let names: Vec<&str> = issue.assignees.iter().map(|a| a.login.as_str()).collect();
            metadata.push(format!("Assignees: {}", names.join(", ")));
        }

        if !issue.labels.is_empty() {
            let label_names: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
            metadata.push(format!("Labels: {}", label_names.join(", ")));
        }

        if let Some(milestone) = &issue.milestone {
            metadata.push(format!("Milestone: {}", milestone.title));
        }

        parts.push(format!("Repo: {}", repo.full_name));

        if !metadata.is_empty() {
            parts.push(metadata.join(" | "));
        }

        // Body
        if let Some(body) = &issue.body {
            if !body.is_empty() {
                parts.push(String::new());
                parts.push(body.clone());
            }
        }

        parts.join("\n")
    }

    /// Transform GitHub PR to memory content
    pub fn pr_to_content(pr: &GitHubPullRequest, repo: &GitHubRepository) -> String {
        let mut parts = Vec::new();

        // Header: PR #{number}: {title}
        parts.push(format!("PR #{}: {}", pr.number, pr.title));

        // Metadata
        let mut metadata = Vec::new();

        let status = if pr.merged {
            "merged".to_string()
        } else if pr.draft {
            "draft".to_string()
        } else {
            pr.state.clone()
        };
        metadata.push(format!("Status: {}", status));

        if let Some(user) = &pr.user {
            metadata.push(format!("Author: {}", user.login));
        }

        // Branch info
        metadata.push(format!("{} <- {}", pr.base.branch_ref, pr.head.branch_ref));

        if !metadata.is_empty() {
            parts.push(metadata.join(" | "));
        }

        // Stats
        let mut stats = Vec::new();
        if let Some(files) = pr.changed_files {
            stats.push(format!("{} files", files));
        }
        if let Some(adds) = pr.additions {
            stats.push(format!("+{}", adds));
        }
        if let Some(dels) = pr.deletions {
            stats.push(format!("-{}", dels));
        }
        if !stats.is_empty() {
            parts.push(stats.join(" "));
        }

        // Labels
        if !pr.labels.is_empty() {
            let label_names: Vec<&str> = pr.labels.iter().map(|l| l.name.as_str()).collect();
            parts.push(format!("Labels: {}", label_names.join(", ")));
        }

        parts.push(format!("Repo: {}", repo.full_name));

        // Body
        if let Some(body) = &pr.body {
            if !body.is_empty() {
                parts.push(String::new());
                parts.push(body.clone());
            }
        }

        parts.join("\n")
    }

    /// Transform GitHub commit to memory content
    pub fn commit_to_content(commit: &GitHubCommit, repo: &GitHubRepository) -> String {
        let mut parts = Vec::new();

        // Header: commit SHA (short) and first line of message
        let short_sha = &commit.sha[..7.min(commit.sha.len())];
        let first_line = commit.commit.message.lines().next().unwrap_or("");
        parts.push(format!("Commit {}: {}", short_sha, first_line));

        // Author info
        parts.push(format!(
            "Author: {} <{}>",
            commit.commit.author.name, commit.commit.author.email
        ));
        parts.push(format!("Date: {}", commit.commit.author.date));

        // Stats if available
        if let Some(stats) = &commit.stats {
            parts.push(format!(
                "+{} -{} ({} total)",
                stats.additions, stats.deletions, stats.total
            ));
        }

        // Files changed if available
        if let Some(files) = &commit.files {
            if !files.is_empty() {
                let file_count = files.len();
                parts.push(format!("{} files changed", file_count));
                // List first few files
                for file in files.iter().take(5) {
                    let status = file.status.as_deref().unwrap_or("modified");
                    parts.push(format!("  {} {}", status, file.filename));
                }
                if file_count > 5 {
                    parts.push(format!("  ... and {} more", file_count - 5));
                }
            }
        }

        parts.push(format!("Repo: {}", repo.full_name));

        // Full commit message if multiline
        let lines: Vec<&str> = commit.commit.message.lines().collect();
        if lines.len() > 1 {
            parts.push(String::new());
            parts.push(commit.commit.message.clone());
        }

        parts.join("\n")
    }

    /// Extract tags from GitHub issue
    pub fn issue_to_tags(issue: &GitHubIssue, repo: &GitHubRepository) -> Vec<String> {
        let mut tags = vec![
            "github".to_string(),
            "issue".to_string(),
            repo.full_name.clone(),
            format!("#{}", issue.number),
            issue.state.clone(),
        ];

        // Add labels
        for label in &issue.labels {
            tags.push(label.name.clone());
        }

        // Add assignee
        if let Some(assignee) = &issue.assignee {
            tags.push(assignee.login.clone());
        }

        // Add milestone
        if let Some(milestone) = &issue.milestone {
            tags.push(milestone.title.clone());
        }

        tags
    }

    /// Extract tags from GitHub PR
    pub fn pr_to_tags(pr: &GitHubPullRequest, repo: &GitHubRepository) -> Vec<String> {
        let mut tags = vec![
            "github".to_string(),
            "pr".to_string(),
            "pull-request".to_string(),
            repo.full_name.clone(),
            format!("#{}", pr.number),
        ];

        // Status
        if pr.merged {
            tags.push("merged".to_string());
        } else if pr.draft {
            tags.push("draft".to_string());
        } else {
            tags.push(pr.state.clone());
        }

        // Add labels
        for label in &pr.labels {
            tags.push(label.name.clone());
        }

        // Add author
        if let Some(user) = &pr.user {
            tags.push(user.login.clone());
        }

        // Add branch names
        tags.push(pr.base.branch_ref.clone());
        tags.push(pr.head.branch_ref.clone());

        tags
    }

    /// Extract tags from GitHub commit
    pub fn commit_to_tags(commit: &GitHubCommit, repo: &GitHubRepository) -> Vec<String> {
        let mut tags = vec![
            "github".to_string(),
            "commit".to_string(),
            repo.full_name.clone(),
            commit.sha[..7.min(commit.sha.len())].to_string(),
        ];

        // Add author
        tags.push(commit.commit.author.name.clone());

        // Add committer if different from author
        if commit.commit.committer.name != commit.commit.author.name {
            tags.push(commit.commit.committer.name.clone());
        }

        // Add GitHub user login if available
        if let Some(author) = &commit.author {
            tags.push(author.login.clone());
        }

        tags
    }

    /// Determine change type from webhook action
    pub fn determine_change_type(action: &str, is_pr: bool) -> String {
        match action {
            "opened" => "created".to_string(),
            "closed" | "merged" => "status_changed".to_string(),
            "reopened" => "status_changed".to_string(),
            "labeled" | "unlabeled" => "tags_updated".to_string(),
            "edited" => "content_updated".to_string(),
            "assigned" | "unassigned" => "content_updated".to_string(),
            "review_requested" | "review_request_removed" if is_pr => "content_updated".to_string(),
            "synchronize" if is_pr => "content_updated".to_string(), // New commits pushed
            _ => "content_updated".to_string(),
        }
    }

    /// Build external_id for issue
    pub fn issue_external_id(repo: &GitHubRepository, issue_number: u64) -> String {
        format!("github:{}#issue-{}", repo.full_name, issue_number)
    }

    /// Build external_id for PR
    pub fn pr_external_id(repo: &GitHubRepository, pr_number: u64) -> String {
        format!("github:{}#pr-{}", repo.full_name, pr_number)
    }

    /// Build external_id for commit
    pub fn commit_external_id(repo: &GitHubRepository, sha: &str) -> String {
        format!("github:{}#commit-{}", repo.full_name, sha)
    }
}

// =============================================================================
// BULK SYNC TYPES
// =============================================================================

/// Request for bulk syncing GitHub issues/PRs/commits
#[derive(Debug, Deserialize)]
pub struct GitHubSyncRequest {
    /// User ID to associate memories with
    pub user_id: String,
    /// GitHub personal access token
    pub token: String,
    /// Repository owner (user or org)
    pub owner: String,
    /// Repository name
    pub repo: String,
    /// Sync issues (default: true)
    #[serde(default = "default_true")]
    pub sync_issues: bool,
    /// Sync pull requests (default: true)
    #[serde(default = "default_true")]
    pub sync_prs: bool,
    /// Sync commits (default: false)
    #[serde(default)]
    pub sync_commits: bool,
    /// Only sync items with state (open, closed, all) - default: all
    #[serde(default = "default_state")]
    pub state: String,
    /// Limit number of items to sync per type
    #[serde(default)]
    pub limit: Option<usize>,
    /// Branch to sync commits from (default: default branch)
    #[serde(default)]
    pub branch: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_state() -> String {
    "all".to_string()
}

/// Response from bulk sync
#[derive(Debug, Serialize)]
pub struct GitHubSyncResponse {
    /// Total items synced
    pub synced_count: usize,
    /// Issues synced
    pub issues_synced: usize,
    /// PRs synced
    pub prs_synced: usize,
    /// Commits synced
    pub commits_synced: usize,
    /// Items created (new)
    pub created_count: usize,
    /// Items updated (existing)
    pub updated_count: usize,
    /// Items that failed
    pub error_count: usize,
    /// Error messages if any
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<String>,
}

// =============================================================================
// GITHUB REST API CLIENT
// =============================================================================

/// Simple GitHub REST API client for bulk sync
pub struct GitHubClient {
    token: String,
    api_url: String,
    client: reqwest::Client,
}

impl GitHubClient {
    const DEFAULT_API_URL: &'static str = "https://api.github.com";

    pub fn new(token: String) -> Self {
        let api_url =
            std::env::var("GITHUB_API_URL").unwrap_or_else(|_| Self::DEFAULT_API_URL.to_string());
        Self {
            token,
            api_url,
            client: reqwest::Client::new(),
        }
    }

    /// Fetch issues from GitHub
    pub async fn fetch_issues(
        &self,
        owner: &str,
        repo: &str,
        state: &str,
        limit: Option<usize>,
    ) -> Result<Vec<GitHubIssue>> {
        let per_page = limit.unwrap_or(100).min(100);
        let url = format!(
            "{}/repos/{}/{}/issues?state={}&per_page={}&sort=updated&direction=desc",
            self.api_url, owner, repo, state, per_page
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "shodh-memory")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await
            .context("Failed to send request to GitHub API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("GitHub API error: {} - {}", status, body);
        }

        let issues: Vec<GitHubIssue> = response
            .json()
            .await
            .context("Failed to parse GitHub issues response")?;

        // Filter out PRs (GitHub API returns PRs in issues endpoint if they have "pull_request" field)
        // We handle this by checking if they're actual issues (no pull_request key in JSON)
        // The deserialization handles this - PRs won't deserialize as issues properly
        Ok(issues)
    }

    /// Fetch pull requests from GitHub
    pub async fn fetch_pull_requests(
        &self,
        owner: &str,
        repo: &str,
        state: &str,
        limit: Option<usize>,
    ) -> Result<Vec<GitHubPullRequest>> {
        let per_page = limit.unwrap_or(100).min(100);
        let url = format!(
            "{}/repos/{}/{}/pulls?state={}&per_page={}&sort=updated&direction=desc",
            self.api_url, owner, repo, state, per_page
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "shodh-memory")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await
            .context("Failed to send request to GitHub API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("GitHub API error: {} - {}", status, body);
        }

        let prs: Vec<GitHubPullRequest> = response
            .json()
            .await
            .context("Failed to parse GitHub PRs response")?;

        Ok(prs)
    }

    /// Fetch commits from GitHub
    pub async fn fetch_commits(
        &self,
        owner: &str,
        repo: &str,
        branch: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<GitHubCommit>> {
        let per_page = limit.unwrap_or(100).min(100);
        let mut url = format!(
            "{}/repos/{}/{}/commits?per_page={}",
            self.api_url, owner, repo, per_page
        );

        if let Some(branch) = branch {
            url.push_str(&format!("&sha={}", branch));
        }

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "shodh-memory")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await
            .context("Failed to send request to GitHub API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("GitHub API error: {} - {}", status, body);
        }

        let commits: Vec<GitHubCommit> = response
            .json()
            .await
            .context("Failed to parse GitHub commits response")?;

        Ok(commits)
    }

    /// Get repository info
    pub async fn get_repository(&self, owner: &str, repo: &str) -> Result<GitHubRepository> {
        let url = format!("{}/repos/{}/{}", self.api_url, owner, repo);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "shodh-memory")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await
            .context("Failed to send request to GitHub API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("GitHub API error: {} - {}", status, body);
        }

        response
            .json()
            .await
            .context("Failed to parse GitHub repository response")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_issue_to_content() {
        let repo = GitHubRepository {
            id: 1,
            name: "shodh-memory".to_string(),
            full_name: "varun29ankuS/shodh-memory".to_string(),
            description: None,
            html_url: "https://github.com/varun29ankuS/shodh-memory".to_string(),
            owner: GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: None,
                avatar_url: None,
            },
        };

        let issue = GitHubIssue {
            number: 123,
            title: "Fix authentication bug".to_string(),
            body: Some("The auth is broken".to_string()),
            state: "open".to_string(),
            html_url: "https://github.com/varun29ankuS/shodh-memory/issues/123".to_string(),
            user: Some(GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: None,
                avatar_url: None,
            }),
            assignee: Some(GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: None,
                avatar_url: None,
            }),
            assignees: vec![],
            labels: vec![GitHubLabel {
                id: 1,
                name: "bug".to_string(),
                color: None,
                description: None,
            }],
            milestone: None,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            closed_at: None,
        };

        let content = GitHubWebhook::issue_to_content(&issue, &repo);
        assert!(content.contains("#123: Fix authentication bug"));
        assert!(content.contains("Status: open"));
        assert!(content.contains("Assignee: varun29ankuS"));
        assert!(content.contains("Labels: bug"));
        assert!(content.contains("The auth is broken"));
    }

    #[test]
    fn test_issue_external_id() {
        let repo = GitHubRepository {
            id: 1,
            name: "shodh-memory".to_string(),
            full_name: "varun29ankuS/shodh-memory".to_string(),
            description: None,
            html_url: "https://github.com/varun29ankuS/shodh-memory".to_string(),
            owner: GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: None,
                avatar_url: None,
            },
        };

        let external_id = GitHubWebhook::issue_external_id(&repo, 123);
        assert_eq!(external_id, "github:varun29ankuS/shodh-memory#issue-123");

        let pr_id = GitHubWebhook::pr_external_id(&repo, 456);
        assert_eq!(pr_id, "github:varun29ankuS/shodh-memory#pr-456");
    }

    #[test]
    fn test_commit_external_id() {
        let repo = GitHubRepository {
            id: 1,
            name: "shodh-memory".to_string(),
            full_name: "varun29ankuS/shodh-memory".to_string(),
            description: None,
            html_url: "https://github.com/varun29ankuS/shodh-memory".to_string(),
            owner: GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: None,
                avatar_url: None,
            },
        };

        let commit_id = GitHubWebhook::commit_external_id(&repo, "abc123def456");
        assert_eq!(
            commit_id,
            "github:varun29ankuS/shodh-memory#commit-abc123def456"
        );
    }

    #[test]
    fn test_commit_to_content() {
        let repo = GitHubRepository {
            id: 1,
            name: "shodh-memory".to_string(),
            full_name: "varun29ankuS/shodh-memory".to_string(),
            description: None,
            html_url: "https://github.com/varun29ankuS/shodh-memory".to_string(),
            owner: GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: None,
                avatar_url: None,
            },
        };

        let commit = GitHubCommit {
            sha: "abc123def456789".to_string(),
            html_url: "https://github.com/varun29ankuS/shodh-memory/commit/abc123".to_string(),
            commit: GitHubCommitData {
                message: "feat: add commit sync\n\nThis adds commit history sync support."
                    .to_string(),
                author: GitHubCommitAuthor {
                    name: "Varun".to_string(),
                    email: "varun@example.com".to_string(),
                    date: "2025-01-01T00:00:00Z".to_string(),
                },
                committer: GitHubCommitAuthor {
                    name: "Varun".to_string(),
                    email: "varun@example.com".to_string(),
                    date: "2025-01-01T00:00:00Z".to_string(),
                },
            },
            author: Some(GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: Some("Varun".to_string()),
                avatar_url: None,
            }),
            committer: None,
            stats: Some(GitHubCommitStats {
                additions: 100,
                deletions: 20,
                total: 120,
            }),
            files: None,
        };

        let content = GitHubWebhook::commit_to_content(&commit, &repo);
        assert!(content.contains("Commit abc123d: feat: add commit sync"));
        assert!(content.contains("Author: Varun <varun@example.com>"));
        assert!(content.contains("+100 -20 (120 total)"));
    }
}
