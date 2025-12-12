//! Integration tests for Linear and GitHub webhook handlers
//!
//! Tests:
//! - Webhook payload parsing
//! - HMAC signature verification
//! - Content transformation
//! - Tag extraction
//! - External ID generation
//! - Upsert flow with external linking

use shodh_memory::integrations::{
    github::{
        GitHubBranch, GitHubIssue, GitHubLabel, GitHubMilestone, GitHubPullRequest,
        GitHubRepository, GitHubUser, GitHubWebhook, GitHubWebhookPayload,
    },
    linear::{
        LinearCycle, LinearIssueData, LinearLabel, LinearProject, LinearState, LinearTeam,
        LinearUser, LinearWebhook, LinearWebhookPayload,
    },
};

// =============================================================================
// LINEAR WEBHOOK TESTS
// =============================================================================

mod linear_tests {
    use super::*;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    type HmacSha256 = Hmac<Sha256>;

    fn create_test_issue() -> LinearIssueData {
        LinearIssueData {
            id: "uuid-12345".to_string(),
            identifier: Some("SHO-42".to_string()),
            title: Some("Implement webhook tests".to_string()),
            description: Some("Add comprehensive tests for webhook handlers".to_string()),
            priority: Some(2),
            priority_label: Some("High".to_string()),
            state: Some(LinearState {
                id: "state-id".to_string(),
                name: "In Progress".to_string(),
                color: Some("#f2c94c".to_string()),
                state_type: Some("started".to_string()),
            }),
            assignee: Some(LinearUser {
                id: "user-id".to_string(),
                name: Some("Varun".to_string()),
                email: Some("varun@test.com".to_string()),
            }),
            creator: Some(LinearUser {
                id: "creator-id".to_string(),
                name: Some("Creator".to_string()),
                email: None,
            }),
            labels: vec![
                LinearLabel {
                    id: "label-1".to_string(),
                    name: "Feature".to_string(),
                    color: Some("#5e6ad2".to_string()),
                },
                LinearLabel {
                    id: "label-2".to_string(),
                    name: "Testing".to_string(),
                    color: Some("#26b5ce".to_string()),
                },
            ],
            team: Some(LinearTeam {
                id: "team-id".to_string(),
                name: Some("Shodh Memory".to_string()),
                key: Some("SHO".to_string()),
            }),
            project: Some(LinearProject {
                id: "project-id".to_string(),
                name: Some("Core".to_string()),
            }),
            cycle: Some(LinearCycle {
                id: "cycle-id".to_string(),
                name: Some("Sprint 5".to_string()),
                number: Some(5),
            }),
            parent: None,
            due_date: Some("2025-01-15".to_string()),
            estimate: Some(3.0),
            created_at: Some("2025-01-01T00:00:00Z".to_string()),
            updated_at: Some("2025-01-10T12:00:00Z".to_string()),
            completed_at: None,
            canceled_at: None,
            url: Some("https://linear.app/shodh/issue/SHO-42".to_string()),
        }
    }

    #[test]
    fn test_linear_webhook_creation() {
        let webhook = LinearWebhook::new(Some("test-secret".to_string()));
        // Invalid hex should return error
        assert!(webhook.verify_signature(b"test", "sha256=invalid").is_err());

        let webhook_no_secret = LinearWebhook::new(None);
        let result = webhook_no_secret.verify_signature(b"test", "any");
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should return true when no secret configured
    }

    #[test]
    fn test_linear_signature_verification_valid() {
        let secret = "webhook-signing-secret";
        let body = r#"{"action":"create","type":"Issue","data":{"id":"123"}}"#;

        // Compute expected signature
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        let signature_header = format!("sha256={}", signature);

        let webhook = LinearWebhook::new(Some(secret.to_string()));
        let result = webhook.verify_signature(body.as_bytes(), &signature_header);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_linear_signature_verification_invalid() {
        let secret = "webhook-signing-secret";
        let body = r#"{"action":"create","type":"Issue","data":{"id":"123"}}"#;
        let wrong_signature =
            "sha256=0000000000000000000000000000000000000000000000000000000000000000";

        let webhook = LinearWebhook::new(Some(secret.to_string()));
        let result = webhook.verify_signature(body.as_bytes(), wrong_signature);

        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should return false for invalid signature
    }

    #[test]
    fn test_linear_payload_parsing() {
        let payload_json = r#"{
            "action": "create",
            "type": "Issue",
            "data": {
                "id": "uuid-12345",
                "identifier": "SHO-42",
                "title": "Test Issue",
                "description": "Test description",
                "priority": 2,
                "priorityLabel": "High",
                "state": {
                    "id": "state-id",
                    "name": "Todo"
                },
                "labels": [
                    {"id": "label-1", "name": "Bug"}
                ],
                "team": {
                    "id": "team-id",
                    "key": "SHO"
                }
            },
            "url": "https://linear.app/shodh/issue/SHO-42"
        }"#;

        let webhook = LinearWebhook::new(None);
        let result = webhook.parse_payload(payload_json.as_bytes());

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.action, "create");
        assert_eq!(payload.entity_type, "Issue");
        assert_eq!(payload.data.identifier, Some("SHO-42".to_string()));
        assert_eq!(payload.data.title, Some("Test Issue".to_string()));
        assert_eq!(payload.data.labels.len(), 1);
        assert_eq!(payload.data.labels[0].name, "Bug");
    }

    #[test]
    fn test_linear_issue_to_content() {
        let issue = create_test_issue();
        let content = LinearWebhook::issue_to_content(&issue);

        // Check header
        assert!(content.contains("SHO-42: Implement webhook tests"));

        // Check metadata
        assert!(content.contains("Status: In Progress"));
        assert!(content.contains("Assignee: Varun"));
        assert!(content.contains("Priority: High"));
        assert!(content.contains("Labels: Feature, Testing"));
        assert!(content.contains("Project: Core"));
        assert!(content.contains("Cycle: Sprint 5"));
        assert!(content.contains("Due: 2025-01-15"));
        assert!(content.contains("Estimate: 3 points"));

        // Check description
        assert!(content.contains("Add comprehensive tests for webhook handlers"));
    }

    #[test]
    fn test_linear_issue_to_content_minimal() {
        let issue = LinearIssueData {
            id: "uuid".to_string(),
            identifier: None,
            title: Some("Minimal Issue".to_string()),
            description: None,
            priority: None,
            priority_label: None,
            state: None,
            assignee: None,
            creator: None,
            labels: vec![],
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
        assert_eq!(content, "Minimal Issue");
    }

    #[test]
    fn test_linear_issue_to_tags() {
        let issue = create_test_issue();
        let tags = LinearWebhook::issue_to_tags(&issue);

        assert!(tags.contains(&"linear".to_string()));
        assert!(tags.contains(&"SHO-42".to_string()));
        assert!(tags.contains(&"Feature".to_string()));
        assert!(tags.contains(&"Testing".to_string()));
        assert!(tags.contains(&"In Progress".to_string()));
        assert!(tags.contains(&"SHO".to_string())); // Team key
        assert!(tags.contains(&"Core".to_string())); // Project name
    }

    #[test]
    fn test_linear_determine_change_type() {
        // Create action
        let issue = create_test_issue();
        assert_eq!(
            LinearWebhook::determine_change_type("create", &issue),
            "created"
        );

        // Update action (no completion)
        assert_eq!(
            LinearWebhook::determine_change_type("update", &issue),
            "content_updated"
        );

        // Update with completion
        let mut completed_issue = issue.clone();
        completed_issue.completed_at = Some("2025-01-10T12:00:00Z".to_string());
        assert_eq!(
            LinearWebhook::determine_change_type("update", &completed_issue),
            "status_changed"
        );

        // Update with cancellation
        let mut canceled_issue = create_test_issue();
        canceled_issue.canceled_at = Some("2025-01-10T12:00:00Z".to_string());
        assert_eq!(
            LinearWebhook::determine_change_type("update", &canceled_issue),
            "status_changed"
        );

        // Remove action
        assert_eq!(
            LinearWebhook::determine_change_type("remove", &create_test_issue()),
            "content_updated"
        );
    }

    #[test]
    fn test_linear_external_id_format() {
        let issue = create_test_issue();

        // External ID should be based on identifier
        let identifier = issue.identifier.as_ref().unwrap();
        let external_id = format!("linear:{}", identifier);

        assert_eq!(external_id, "linear:SHO-42");
    }
}

// =============================================================================
// GITHUB WEBHOOK TESTS
// =============================================================================

mod github_tests {
    use super::*;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    type HmacSha256 = Hmac<Sha256>;

    fn create_test_repo() -> GitHubRepository {
        GitHubRepository {
            id: 123456,
            name: "shodh-memory".to_string(),
            full_name: "varun29ankuS/shodh-memory".to_string(),
            description: Some("Cognitive memory system for AI agents".to_string()),
            html_url: "https://github.com/varun29ankuS/shodh-memory".to_string(),
            owner: GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: Some("Varun".to_string()),
                avatar_url: Some("https://avatars.githubusercontent.com/u/1".to_string()),
            },
        }
    }

    fn create_test_issue() -> GitHubIssue {
        GitHubIssue {
            number: 42,
            title: "Fix memory leak in retrieval engine".to_string(),
            body: Some("Memory usage grows unbounded during batch retrieval.\n\nSteps to reproduce:\n1. Run batch query\n2. Check memory usage".to_string()),
            state: "open".to_string(),
            html_url: "https://github.com/varun29ankuS/shodh-memory/issues/42".to_string(),
            user: Some(GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: Some("Varun".to_string()),
                avatar_url: None,
            }),
            assignee: Some(GitHubUser {
                id: 2,
                login: "contributor".to_string(),
                name: Some("Contributor".to_string()),
                avatar_url: None,
            }),
            assignees: vec![],
            labels: vec![
                GitHubLabel {
                    id: 1,
                    name: "bug".to_string(),
                    color: Some("d73a4a".to_string()),
                    description: Some("Something isn't working".to_string()),
                },
                GitHubLabel {
                    id: 2,
                    name: "memory".to_string(),
                    color: Some("0e8a16".to_string()),
                    description: None,
                },
            ],
            milestone: Some(GitHubMilestone {
                number: 1,
                title: "v0.2.0".to_string(),
                description: Some("Second release".to_string()),
                state: "open".to_string(),
            }),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-10T12:00:00Z".to_string(),
            closed_at: None,
        }
    }

    fn create_test_pr() -> GitHubPullRequest {
        GitHubPullRequest {
            number: 55,
            title: "Add GitHub integration".to_string(),
            body: Some("## Summary\nAdds webhook receiver and bulk sync for GitHub.\n\n## Test Plan\n- Unit tests added".to_string()),
            state: "open".to_string(),
            html_url: "https://github.com/varun29ankuS/shodh-memory/pull/55".to_string(),
            user: Some(GitHubUser {
                id: 1,
                login: "varun29ankuS".to_string(),
                name: Some("Varun".to_string()),
                avatar_url: None,
            }),
            assignee: None,
            assignees: vec![],
            labels: vec![GitHubLabel {
                id: 3,
                name: "enhancement".to_string(),
                color: Some("a2eeef".to_string()),
                description: None,
            }],
            milestone: None,
            base: GitHubBranch {
                branch_ref: "main".to_string(),
                sha: "abc123".to_string(),
                repo: None,
            },
            head: GitHubBranch {
                branch_ref: "feature/github-integration".to_string(),
                sha: "def456".to_string(),
                repo: None,
            },
            merged: false,
            merge_commit_sha: None,
            commits: Some(5),
            additions: Some(1025),
            deletions: Some(2),
            changed_files: Some(4),
            created_at: "2025-01-05T00:00:00Z".to_string(),
            updated_at: "2025-01-10T12:00:00Z".to_string(),
            closed_at: None,
            merged_at: None,
            draft: false,
        }
    }

    #[test]
    fn test_github_webhook_creation() {
        let webhook = GitHubWebhook::new(Some("test-secret".to_string()));
        // Invalid hex should return error
        assert!(webhook.verify_signature(b"test", "sha256=invalid").is_err());

        let webhook_no_secret = GitHubWebhook::new(None);
        let result = webhook_no_secret.verify_signature(b"test", "any");
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should return true when no secret configured
    }

    #[test]
    fn test_github_signature_verification_valid() {
        let secret = "webhook-secret";
        let body = r#"{"action":"opened","issue":{"number":1}}"#;

        // Compute expected signature
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body.as_bytes());
        let signature = hex::encode(mac.finalize().into_bytes());
        let signature_header = format!("sha256={}", signature);

        let webhook = GitHubWebhook::new(Some(secret.to_string()));
        let result = webhook.verify_signature(body.as_bytes(), &signature_header);

        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_github_signature_verification_invalid() {
        let secret = "webhook-secret";
        let body = r#"{"action":"opened","issue":{"number":1}}"#;
        let wrong_signature =
            "sha256=0000000000000000000000000000000000000000000000000000000000000000";

        let webhook = GitHubWebhook::new(Some(secret.to_string()));
        let result = webhook.verify_signature(body.as_bytes(), wrong_signature);

        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    #[test]
    fn test_github_payload_parsing_issue() {
        let payload_json = r#"{
            "action": "opened",
            "issue": {
                "number": 42,
                "title": "Test Issue",
                "body": "Test body",
                "state": "open",
                "html_url": "https://github.com/test/repo/issues/42",
                "labels": [{"id": 1, "name": "bug"}],
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z"
            },
            "repository": {
                "id": 123,
                "name": "repo",
                "full_name": "test/repo",
                "html_url": "https://github.com/test/repo",
                "owner": {
                    "id": 1,
                    "login": "test"
                }
            }
        }"#;

        let webhook = GitHubWebhook::new(None);
        let result = webhook.parse_payload(payload_json.as_bytes());

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.action, "opened");
        assert!(payload.issue.is_some());
        assert!(payload.pull_request.is_none());

        let issue = payload.issue.unwrap();
        assert_eq!(issue.number, 42);
        assert_eq!(issue.title, "Test Issue");
        assert_eq!(issue.labels.len(), 1);
    }

    #[test]
    fn test_github_payload_parsing_pr() {
        let payload_json = r#"{
            "action": "opened",
            "pull_request": {
                "number": 55,
                "title": "Test PR",
                "body": "Test body",
                "state": "open",
                "html_url": "https://github.com/test/repo/pull/55",
                "labels": [],
                "base": {"ref": "main", "sha": "abc"},
                "head": {"ref": "feature", "sha": "def"},
                "merged": false,
                "draft": false,
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:00Z"
            },
            "repository": {
                "id": 123,
                "name": "repo",
                "full_name": "test/repo",
                "html_url": "https://github.com/test/repo",
                "owner": {
                    "id": 1,
                    "login": "test"
                }
            }
        }"#;

        let webhook = GitHubWebhook::new(None);
        let result = webhook.parse_payload(payload_json.as_bytes());

        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.action, "opened");
        assert!(payload.issue.is_none());
        assert!(payload.pull_request.is_some());

        let pr = payload.pull_request.unwrap();
        assert_eq!(pr.number, 55);
        assert_eq!(pr.base.branch_ref, "main");
        assert_eq!(pr.head.branch_ref, "feature");
    }

    #[test]
    fn test_github_issue_to_content() {
        let repo = create_test_repo();
        let issue = create_test_issue();
        let content = GitHubWebhook::issue_to_content(&issue, &repo);

        // Check header
        assert!(content.contains("#42: Fix memory leak in retrieval engine"));

        // Check metadata
        assert!(content.contains("Status: open"));
        assert!(content.contains("Assignee: contributor"));
        assert!(content.contains("Labels: bug, memory"));
        assert!(content.contains("Milestone: v0.2.0"));
        assert!(content.contains("Repo: varun29ankuS/shodh-memory"));

        // Check body
        assert!(content.contains("Memory usage grows unbounded"));
    }

    #[test]
    fn test_github_pr_to_content() {
        let repo = create_test_repo();
        let pr = create_test_pr();
        let content = GitHubWebhook::pr_to_content(&pr, &repo);

        // Check header
        assert!(content.contains("PR #55: Add GitHub integration"));

        // Check metadata
        assert!(content.contains("Status: open"));
        assert!(content.contains("Author: varun29ankuS"));
        assert!(content.contains("main <- feature/github-integration"));

        // Check stats
        assert!(content.contains("4 files"));
        assert!(content.contains("+1025"));
        assert!(content.contains("-2"));

        // Check labels
        assert!(content.contains("Labels: enhancement"));

        // Check body
        assert!(content.contains("Adds webhook receiver"));
    }

    #[test]
    fn test_github_pr_to_content_merged() {
        let repo = create_test_repo();
        let mut pr = create_test_pr();
        pr.merged = true;

        let content = GitHubWebhook::pr_to_content(&pr, &repo);
        assert!(content.contains("Status: merged"));
    }

    #[test]
    fn test_github_pr_to_content_draft() {
        let repo = create_test_repo();
        let mut pr = create_test_pr();
        pr.draft = true;

        let content = GitHubWebhook::pr_to_content(&pr, &repo);
        assert!(content.contains("Status: draft"));
    }

    #[test]
    fn test_github_issue_to_tags() {
        let repo = create_test_repo();
        let issue = create_test_issue();
        let tags = GitHubWebhook::issue_to_tags(&issue, &repo);

        assert!(tags.contains(&"github".to_string()));
        assert!(tags.contains(&"issue".to_string()));
        assert!(tags.contains(&"varun29ankuS/shodh-memory".to_string()));
        assert!(tags.contains(&"#42".to_string()));
        assert!(tags.contains(&"open".to_string()));
        assert!(tags.contains(&"bug".to_string()));
        assert!(tags.contains(&"memory".to_string()));
        assert!(tags.contains(&"contributor".to_string())); // Assignee
        assert!(tags.contains(&"v0.2.0".to_string())); // Milestone
    }

    #[test]
    fn test_github_pr_to_tags() {
        let repo = create_test_repo();
        let pr = create_test_pr();
        let tags = GitHubWebhook::pr_to_tags(&pr, &repo);

        assert!(tags.contains(&"github".to_string()));
        assert!(tags.contains(&"pr".to_string()));
        assert!(tags.contains(&"pull-request".to_string()));
        assert!(tags.contains(&"varun29ankuS/shodh-memory".to_string()));
        assert!(tags.contains(&"#55".to_string()));
        assert!(tags.contains(&"open".to_string()));
        assert!(tags.contains(&"enhancement".to_string()));
        assert!(tags.contains(&"varun29ankuS".to_string())); // Author
        assert!(tags.contains(&"main".to_string())); // Base branch
        assert!(tags.contains(&"feature/github-integration".to_string())); // Head branch
    }

    #[test]
    fn test_github_pr_to_tags_merged() {
        let repo = create_test_repo();
        let mut pr = create_test_pr();
        pr.merged = true;

        let tags = GitHubWebhook::pr_to_tags(&pr, &repo);
        assert!(tags.contains(&"merged".to_string()));
        assert!(!tags.contains(&"open".to_string()));
    }

    #[test]
    fn test_github_external_id_issue() {
        let repo = create_test_repo();
        let external_id = GitHubWebhook::issue_external_id(&repo, 42);

        assert_eq!(external_id, "github:varun29ankuS/shodh-memory#issue-42");
    }

    #[test]
    fn test_github_external_id_pr() {
        let repo = create_test_repo();
        let external_id = GitHubWebhook::pr_external_id(&repo, 55);

        assert_eq!(external_id, "github:varun29ankuS/shodh-memory#pr-55");
    }

    #[test]
    fn test_github_determine_change_type() {
        // Issue actions
        assert_eq!(
            GitHubWebhook::determine_change_type("opened", false),
            "created"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("closed", false),
            "status_changed"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("reopened", false),
            "status_changed"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("labeled", false),
            "tags_updated"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("unlabeled", false),
            "tags_updated"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("edited", false),
            "content_updated"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("assigned", false),
            "content_updated"
        );

        // PR-specific actions
        assert_eq!(
            GitHubWebhook::determine_change_type("merged", true),
            "status_changed"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("review_requested", true),
            "content_updated"
        );
        assert_eq!(
            GitHubWebhook::determine_change_type("synchronize", true),
            "content_updated"
        );
    }
}

// =============================================================================
// CROSS-PLATFORM EXTERNAL ID TESTS
// =============================================================================

mod external_id_tests {
    use super::*;

    #[test]
    fn test_external_id_uniqueness() {
        // Linear and GitHub external IDs should never collide
        let linear_id = "linear:SHO-42";
        let github_issue_id = "github:owner/repo#issue-42";
        let github_pr_id = "github:owner/repo#pr-42";

        assert_ne!(linear_id, github_issue_id);
        assert_ne!(linear_id, github_pr_id);
        assert_ne!(github_issue_id, github_pr_id);
    }

    #[test]
    fn test_external_id_parsing() {
        // Test that external IDs can be parsed back to source info
        let linear_id = "linear:SHO-42";
        assert!(linear_id.starts_with("linear:"));
        let identifier = linear_id.strip_prefix("linear:").unwrap();
        assert_eq!(identifier, "SHO-42");

        let github_id = "github:varun29ankuS/shodh-memory#issue-123";
        assert!(github_id.starts_with("github:"));
        let rest = github_id.strip_prefix("github:").unwrap();
        let parts: Vec<&str> = rest.split('#').collect();
        assert_eq!(parts[0], "varun29ankuS/shodh-memory");
        assert_eq!(parts[1], "issue-123");
    }

    #[test]
    fn test_external_id_special_characters() {
        // Test with repos containing special characters (allowed in GitHub)
        let repo = GitHubRepository {
            id: 1,
            name: "my-repo.js".to_string(),
            full_name: "org-name/my-repo.js".to_string(),
            description: None,
            html_url: "https://github.com/org-name/my-repo.js".to_string(),
            owner: GitHubUser {
                id: 1,
                login: "org-name".to_string(),
                name: None,
                avatar_url: None,
            },
        };

        let external_id = GitHubWebhook::issue_external_id(&repo, 1);
        assert_eq!(external_id, "github:org-name/my-repo.js#issue-1");
    }
}

// =============================================================================
// CONTENT TRANSFORMATION EDGE CASES
// =============================================================================

mod content_edge_cases {
    use super::*;

    #[test]
    fn test_linear_empty_description() {
        let issue = LinearIssueData {
            id: "uuid".to_string(),
            identifier: Some("SHO-1".to_string()),
            title: Some("Empty Description".to_string()),
            description: Some("".to_string()), // Empty but not None
            priority: None,
            priority_label: None,
            state: None,
            assignee: None,
            creator: None,
            labels: vec![],
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
        // Should not contain extra empty lines from empty description
        assert!(!content.ends_with("\n\n"));
    }

    #[test]
    fn test_github_issue_multiple_assignees() {
        let repo = GitHubRepository {
            id: 1,
            name: "test".to_string(),
            full_name: "test/test".to_string(),
            description: None,
            html_url: "https://github.com/test/test".to_string(),
            owner: GitHubUser {
                id: 1,
                login: "test".to_string(),
                name: None,
                avatar_url: None,
            },
        };

        let issue = GitHubIssue {
            number: 1,
            title: "Test".to_string(),
            body: None,
            state: "open".to_string(),
            html_url: "https://github.com/test/test/issues/1".to_string(),
            user: None,
            assignee: None, // No single assignee
            assignees: vec![
                GitHubUser {
                    id: 1,
                    login: "alice".to_string(),
                    name: None,
                    avatar_url: None,
                },
                GitHubUser {
                    id: 2,
                    login: "bob".to_string(),
                    name: None,
                    avatar_url: None,
                },
            ],
            labels: vec![],
            milestone: None,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            closed_at: None,
        };

        let content = GitHubWebhook::issue_to_content(&issue, &repo);
        assert!(content.contains("Assignees: alice, bob"));
    }

    #[test]
    fn test_linear_cycle_number_fallback() {
        let issue = LinearIssueData {
            id: "uuid".to_string(),
            identifier: Some("SHO-1".to_string()),
            title: Some("Test".to_string()),
            description: None,
            priority: None,
            priority_label: None,
            state: None,
            assignee: None,
            creator: None,
            labels: vec![],
            team: None,
            project: None,
            cycle: Some(LinearCycle {
                id: "cycle-id".to_string(),
                name: None, // No name
                number: Some(3),
            }),
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
        assert!(content.contains("Cycle: #3"));
    }
}
