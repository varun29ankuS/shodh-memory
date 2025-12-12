//! Benchmarks for Linear and GitHub integration performance
//!
//! Measures:
//! - Webhook payload parsing
//! - HMAC signature verification
//! - Content transformation
//! - Tag extraction

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use hmac::{Hmac, Mac};
use serde_json;
use sha2::Sha256;

use shodh_memory::integrations::github::{
    GitHubBranch, GitHubIssue, GitHubLabel, GitHubPullRequest, GitHubRepository, GitHubUser,
    GitHubWebhook,
};
use shodh_memory::integrations::linear::{
    LinearCycle, LinearIssueData, LinearLabel, LinearProject, LinearState, LinearTeam, LinearUser,
    LinearWebhook,
};

type HmacSha256 = Hmac<Sha256>;

// =============================================================================
// TEST DATA GENERATORS
// =============================================================================

fn create_linear_issue() -> LinearIssueData {
    LinearIssueData {
        id: "uuid-12345-abcde-67890".to_string(),
        identifier: Some("SHO-42".to_string()),
        title: Some("Implement webhook integration for external services".to_string()),
        description: Some(
            "Add comprehensive webhook support for Linear and GitHub integrations.\n\n\
            ## Requirements\n\
            - HMAC signature verification\n\
            - Payload parsing\n\
            - Content transformation\n\
            - Tag extraction\n\n\
            ## Acceptance Criteria\n\
            - All webhook events properly handled\n\
            - Tests passing\n\
            - Documentation updated"
                .to_string(),
        ),
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
            name: Some("Varun Sharma".to_string()),
            email: Some("varun@test.com".to_string()),
        }),
        creator: Some(LinearUser {
            id: "creator-id".to_string(),
            name: Some("Product Manager".to_string()),
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
                name: "Integration".to_string(),
                color: Some("#26b5ce".to_string()),
            },
            LinearLabel {
                id: "label-3".to_string(),
                name: "High Priority".to_string(),
                color: Some("#eb5757".to_string()),
            },
        ],
        team: Some(LinearTeam {
            id: "team-id".to_string(),
            name: Some("Shodh Memory".to_string()),
            key: Some("SHO".to_string()),
        }),
        project: Some(LinearProject {
            id: "project-id".to_string(),
            name: Some("Core Platform".to_string()),
        }),
        cycle: Some(LinearCycle {
            id: "cycle-id".to_string(),
            name: Some("Sprint 5".to_string()),
            number: Some(5),
        }),
        parent: None,
        due_date: Some("2025-01-15".to_string()),
        estimate: Some(5.0),
        created_at: Some("2025-01-01T00:00:00Z".to_string()),
        updated_at: Some("2025-01-10T12:00:00Z".to_string()),
        completed_at: None,
        canceled_at: None,
        url: Some("https://linear.app/shodh/issue/SHO-42".to_string()),
    }
}

fn create_github_repo() -> GitHubRepository {
    GitHubRepository {
        id: 123456789,
        name: "shodh-memory".to_string(),
        full_name: "varun29ankuS/shodh-memory".to_string(),
        description: Some("Cognitive memory system for AI agents".to_string()),
        html_url: "https://github.com/varun29ankuS/shodh-memory".to_string(),
        owner: GitHubUser {
            id: 1,
            login: "varun29ankuS".to_string(),
            name: Some("Varun Sharma".to_string()),
            avatar_url: Some("https://avatars.githubusercontent.com/u/1".to_string()),
        },
    }
}

fn create_github_issue() -> GitHubIssue {
    GitHubIssue {
        number: 42,
        title: "Fix memory leak in retrieval engine during batch operations".to_string(),
        body: Some(
            "## Description\n\
            Memory usage grows unbounded during batch retrieval operations.\n\n\
            ## Steps to Reproduce\n\
            1. Start the server\n\
            2. Run batch query with 1000+ results\n\
            3. Observe memory growth\n\n\
            ## Expected Behavior\n\
            Memory should be released after query completes.\n\n\
            ## Environment\n\
            - OS: Ubuntu 22.04\n\
            - Rust: 1.75\n\
            - shodh-memory: 0.1.5"
                .to_string(),
        ),
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
            GitHubLabel {
                id: 3,
                name: "priority:high".to_string(),
                color: Some("b60205".to_string()),
                description: None,
            },
        ],
        milestone: None,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        updated_at: "2025-01-10T12:00:00Z".to_string(),
        closed_at: None,
    }
}

fn create_github_pr() -> GitHubPullRequest {
    GitHubPullRequest {
        number: 55,
        title: "feat(integrations): Add GitHub webhook receiver and bulk sync".to_string(),
        body: Some(
            "## Summary\n\
            Adds comprehensive GitHub integration for syncing issues and PRs to Shodh memory.\n\n\
            ## Changes\n\
            - Add webhook handler with HMAC verification\n\
            - Add REST API client for bulk sync\n\
            - Add content transformation\n\
            - Add tag extraction\n\n\
            ## Test Plan\n\
            - [x] Unit tests added\n\
            - [x] Integration tests added\n\
            - [x] Manual testing completed"
                .to_string(),
        ),
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
        labels: vec![
            GitHubLabel {
                id: 3,
                name: "enhancement".to_string(),
                color: Some("a2eeef".to_string()),
                description: None,
            },
            GitHubLabel {
                id: 4,
                name: "integration".to_string(),
                color: Some("0052cc".to_string()),
                description: None,
            },
        ],
        milestone: None,
        base: GitHubBranch {
            branch_ref: "main".to_string(),
            sha: "abc123def456".to_string(),
            repo: None,
        },
        head: GitHubBranch {
            branch_ref: "feature/github-integration".to_string(),
            sha: "789xyz012".to_string(),
            repo: None,
        },
        merged: false,
        merge_commit_sha: None,
        commits: Some(12),
        additions: Some(1025),
        deletions: Some(15),
        changed_files: Some(8),
        created_at: "2025-01-05T00:00:00Z".to_string(),
        updated_at: "2025-01-10T12:00:00Z".to_string(),
        closed_at: None,
        merged_at: None,
        draft: false,
    }
}

fn create_linear_webhook_payload() -> String {
    serde_json::json!({
        "action": "update",
        "type": "Issue",
        "data": {
            "id": "uuid-12345-abcde-67890",
            "identifier": "SHO-42",
            "title": "Implement webhook integration for external services",
            "description": "Add comprehensive webhook support for Linear and GitHub integrations.",
            "priority": 2,
            "priorityLabel": "High",
            "state": {
                "id": "state-id",
                "name": "In Progress",
                "color": "#f2c94c",
                "type": "started"
            },
            "assignee": {
                "id": "user-id",
                "name": "Varun Sharma",
                "email": "varun@test.com"
            },
            "labels": [
                {"id": "label-1", "name": "Feature", "color": "#5e6ad2"},
                {"id": "label-2", "name": "Integration", "color": "#26b5ce"}
            ],
            "team": {
                "id": "team-id",
                "name": "Shodh Memory",
                "key": "SHO"
            },
            "project": {
                "id": "project-id",
                "name": "Core Platform"
            },
            "dueDate": "2025-01-15",
            "estimate": 5.0,
            "createdAt": "2025-01-01T00:00:00Z",
            "updatedAt": "2025-01-10T12:00:00Z"
        },
        "url": "https://linear.app/shodh/issue/SHO-42"
    })
    .to_string()
}

fn create_github_webhook_payload() -> String {
    serde_json::json!({
        "action": "opened",
        "issue": {
            "number": 42,
            "title": "Fix memory leak in retrieval engine",
            "body": "Memory usage grows unbounded during batch retrieval.",
            "state": "open",
            "html_url": "https://github.com/varun29ankuS/shodh-memory/issues/42",
            "user": {"id": 1, "login": "varun29ankuS"},
            "assignee": {"id": 2, "login": "contributor"},
            "labels": [
                {"id": 1, "name": "bug", "color": "d73a4a"},
                {"id": 2, "name": "memory", "color": "0e8a16"}
            ],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-10T12:00:00Z"
        },
        "repository": {
            "id": 123456789,
            "name": "shodh-memory",
            "full_name": "varun29ankuS/shodh-memory",
            "html_url": "https://github.com/varun29ankuS/shodh-memory",
            "owner": {"id": 1, "login": "varun29ankuS"}
        },
        "sender": {"id": 1, "login": "varun29ankuS"}
    })
    .to_string()
}

// =============================================================================
// BENCHMARKS
// =============================================================================

fn bench_linear_payload_parsing(c: &mut Criterion) {
    let payload = create_linear_webhook_payload();
    let webhook = LinearWebhook::new(None);

    let mut group = c.benchmark_group("linear_integration");
    group.throughput(Throughput::Bytes(payload.len() as u64));

    group.bench_function("payload_parsing", |b| {
        b.iter(|| webhook.parse_payload(black_box(payload.as_bytes())))
    });

    group.finish();
}

fn bench_github_payload_parsing(c: &mut Criterion) {
    let payload = create_github_webhook_payload();
    let webhook = GitHubWebhook::new(None);

    let mut group = c.benchmark_group("github_integration");
    group.throughput(Throughput::Bytes(payload.len() as u64));

    group.bench_function("payload_parsing", |b| {
        b.iter(|| webhook.parse_payload(black_box(payload.as_bytes())))
    });

    group.finish();
}

fn bench_linear_signature_verification(c: &mut Criterion) {
    let secret = "webhook-signing-secret-for-linear-integration";
    let payload = create_linear_webhook_payload();

    // Compute valid signature
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(payload.as_bytes());
    let signature = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

    let webhook = LinearWebhook::new(Some(secret.to_string()));

    let mut group = c.benchmark_group("linear_integration");

    group.bench_function("signature_verification", |b| {
        b.iter(|| webhook.verify_signature(black_box(payload.as_bytes()), black_box(&signature)))
    });

    group.finish();
}

fn bench_github_signature_verification(c: &mut Criterion) {
    let secret = "webhook-signing-secret-for-github-integration";
    let payload = create_github_webhook_payload();

    // Compute valid signature
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(payload.as_bytes());
    let signature = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

    let webhook = GitHubWebhook::new(Some(secret.to_string()));

    let mut group = c.benchmark_group("github_integration");

    group.bench_function("signature_verification", |b| {
        b.iter(|| webhook.verify_signature(black_box(payload.as_bytes()), black_box(&signature)))
    });

    group.finish();
}

fn bench_linear_content_transformation(c: &mut Criterion) {
    let issue = create_linear_issue();

    let mut group = c.benchmark_group("linear_integration");

    group.bench_function("content_transformation", |b| {
        b.iter(|| LinearWebhook::issue_to_content(black_box(&issue)))
    });

    group.finish();
}

fn bench_github_content_transformation(c: &mut Criterion) {
    let repo = create_github_repo();
    let issue = create_github_issue();
    let pr = create_github_pr();

    let mut group = c.benchmark_group("github_integration");

    group.bench_function("issue_content_transformation", |b| {
        b.iter(|| GitHubWebhook::issue_to_content(black_box(&issue), black_box(&repo)))
    });

    group.bench_function("pr_content_transformation", |b| {
        b.iter(|| GitHubWebhook::pr_to_content(black_box(&pr), black_box(&repo)))
    });

    group.finish();
}

fn bench_linear_tag_extraction(c: &mut Criterion) {
    let issue = create_linear_issue();

    let mut group = c.benchmark_group("linear_integration");

    group.bench_function("tag_extraction", |b| {
        b.iter(|| LinearWebhook::issue_to_tags(black_box(&issue)))
    });

    group.finish();
}

fn bench_github_tag_extraction(c: &mut Criterion) {
    let repo = create_github_repo();
    let issue = create_github_issue();
    let pr = create_github_pr();

    let mut group = c.benchmark_group("github_integration");

    group.bench_function("issue_tag_extraction", |b| {
        b.iter(|| GitHubWebhook::issue_to_tags(black_box(&issue), black_box(&repo)))
    });

    group.bench_function("pr_tag_extraction", |b| {
        b.iter(|| GitHubWebhook::pr_to_tags(black_box(&pr), black_box(&repo)))
    });

    group.finish();
}

fn bench_full_linear_pipeline(c: &mut Criterion) {
    let secret = "webhook-signing-secret";
    let payload = create_linear_webhook_payload();

    // Compute valid signature
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(payload.as_bytes());
    let signature = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

    let webhook = LinearWebhook::new(Some(secret.to_string()));

    let mut group = c.benchmark_group("linear_integration");

    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            // 1. Verify signature
            let valid = webhook
                .verify_signature(black_box(payload.as_bytes()), black_box(&signature))
                .unwrap();
            assert!(valid);

            // 2. Parse payload
            let parsed = webhook
                .parse_payload(black_box(payload.as_bytes()))
                .unwrap();

            // 3. Transform content
            let content = LinearWebhook::issue_to_content(&parsed.data);

            // 4. Extract tags
            let tags = LinearWebhook::issue_to_tags(&parsed.data);

            // 5. Determine change type
            let change_type = LinearWebhook::determine_change_type(&parsed.action, &parsed.data);

            (content, tags, change_type)
        })
    });

    group.finish();
}

fn bench_full_github_pipeline(c: &mut Criterion) {
    let secret = "webhook-signing-secret";
    let payload = create_github_webhook_payload();

    // Compute valid signature
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
    mac.update(payload.as_bytes());
    let signature = format!("sha256={}", hex::encode(mac.finalize().into_bytes()));

    let webhook = GitHubWebhook::new(Some(secret.to_string()));

    let mut group = c.benchmark_group("github_integration");

    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            // 1. Verify signature
            let valid = webhook
                .verify_signature(black_box(payload.as_bytes()), black_box(&signature))
                .unwrap();
            assert!(valid);

            // 2. Parse payload
            let parsed = webhook
                .parse_payload(black_box(payload.as_bytes()))
                .unwrap();

            // 3. Transform content (issue)
            let content = if let Some(issue) = &parsed.issue {
                GitHubWebhook::issue_to_content(issue, &parsed.repository)
            } else {
                String::new()
            };

            // 4. Extract tags
            let tags = if let Some(issue) = &parsed.issue {
                GitHubWebhook::issue_to_tags(issue, &parsed.repository)
            } else {
                vec![]
            };

            // 5. Determine change type
            let change_type = GitHubWebhook::determine_change_type(&parsed.action, false);

            (content, tags, change_type)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_linear_payload_parsing,
    bench_github_payload_parsing,
    bench_linear_signature_verification,
    bench_github_signature_verification,
    bench_linear_content_transformation,
    bench_github_content_transformation,
    bench_linear_tag_extraction,
    bench_github_tag_extraction,
    bench_full_linear_pipeline,
    bench_full_github_pipeline,
);

criterion_main!(benches);
