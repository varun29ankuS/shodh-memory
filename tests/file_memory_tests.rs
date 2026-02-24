//! Tests for FileMemoryStore - Codebase Integration (MEM-36)
//!
//! Tests cover:
//! - CRUD operations for FileMemory
//! - Codebase scanning with limits
//! - File indexing and key item extraction
//! - File type detection
//! - Heat score calculation
//!
//! Run with: cargo test --test file_memory_tests -- --nocapture

use std::fs;
use std::sync::Arc;
use tempfile::TempDir;

use rocksdb::Options;
use shodh_memory::memory::files::FileMemoryStore;
use shodh_memory::memory::types::{
    CodebaseConfig, FileMemory, FileMemoryId, FileType, LearnedFrom, ProjectId,
};
use uuid::Uuid;

// ============================================================================
// TEST HELPERS
// ============================================================================

fn create_test_store() -> (FileMemoryStore, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("shared_db");

    let mut opts = Options::default();
    opts.create_if_missing(true);
    opts.create_missing_column_families(true);

    let cfs = FileMemoryStore::cf_descriptors();

    let db = rocksdb::DB::open_cf_descriptors(&opts, &db_path, cfs).expect("Failed to open DB");
    let db = Arc::new(db);

    let store = FileMemoryStore::new(db, temp_dir.path()).expect("Failed to create store");
    (store, temp_dir)
}

fn create_test_codebase() -> TempDir {
    let temp_dir = TempDir::new().expect("Failed to create temp codebase");

    // Create Rust files
    fs::create_dir_all(temp_dir.path().join("src")).unwrap();
    fs::write(
        temp_dir.path().join("src/main.rs"),
        r#"
pub fn main() {
    println!("Hello, world!");
}

pub struct Config {
    pub name: String,
    pub value: i32,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Config {
            name: name.to_string(),
            value: 0,
        }
    }
}

fn helper_function() {
    // private helper
}
"#,
    )
    .unwrap();

    fs::write(
        temp_dir.path().join("src/lib.rs"),
        r#"
pub mod utils;

pub struct Library {
    pub data: Vec<u8>,
}

pub trait Processor {
    fn process(&self, input: &str) -> String;
}

pub enum Status {
    Active,
    Inactive,
    Pending,
}
"#,
    )
    .unwrap();

    // Create TypeScript file
    fs::write(
        temp_dir.path().join("index.ts"),
        r#"
export class UserService {
    private users: Map<string, User>;

    constructor() {
        this.users = new Map();
    }

    public addUser(user: User): void {
        this.users.set(user.id, user);
    }
}

export interface User {
    id: string;
    name: string;
}

export function validateUser(user: User): boolean {
    return user.id.length > 0;
}
"#,
    )
    .unwrap();

    // Create Python file
    fs::write(
        temp_dir.path().join("utils.py"),
        r#"
class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        return data.upper()

def helper_function(x, y):
    return x + y

async def fetch_data(url):
    pass
"#,
    )
    .unwrap();

    // Create Go file
    fs::write(
        temp_dir.path().join("server.go"),
        r#"
package main

type Server struct {
    Port int
    Host string
}

func NewServer(port int) *Server {
    return &Server{Port: port}
}

func (s *Server) Start() error {
    return nil
}

type Handler interface {
    Handle(req Request) Response
}
"#,
    )
    .unwrap();

    // Create a binary file (should be skipped)
    fs::write(temp_dir.path().join("image.png"), &[0x89, 0x50, 0x4E, 0x47]).unwrap();

    // Create a large file (should be skipped with default config)
    let large_content = "x".repeat(600_000); // 600KB
    fs::write(temp_dir.path().join("large.txt"), large_content).unwrap();

    // Create nested directory
    fs::create_dir_all(temp_dir.path().join("tests")).unwrap();
    fs::write(
        temp_dir.path().join("tests/test_main.rs"),
        r#"
#[test]
fn test_example() {
    assert!(true);
}
"#,
    )
    .unwrap();

    temp_dir
}

// ============================================================================
// FILE TYPE DETECTION TESTS
// ============================================================================

#[test]
fn test_file_type_from_extension() {
    assert_eq!(FileType::from_extension("rs"), FileType::Rust);
    assert_eq!(FileType::from_extension("ts"), FileType::TypeScript);
    assert_eq!(FileType::from_extension("tsx"), FileType::TypeScript);
    assert_eq!(FileType::from_extension("js"), FileType::JavaScript);
    assert_eq!(FileType::from_extension("jsx"), FileType::JavaScript);
    assert_eq!(FileType::from_extension("py"), FileType::Python);
    assert_eq!(FileType::from_extension("go"), FileType::Go);
    assert_eq!(FileType::from_extension("java"), FileType::Java);
    assert_eq!(FileType::from_extension("cpp"), FileType::Cpp);
    assert_eq!(FileType::from_extension("c"), FileType::C);
    assert_eq!(FileType::from_extension("md"), FileType::Markdown);
    assert_eq!(FileType::from_extension("json"), FileType::Json);
    assert_eq!(FileType::from_extension("yaml"), FileType::Yaml);
    assert_eq!(FileType::from_extension("toml"), FileType::Toml);

    // Unknown extension returns Other with the extension
    match FileType::from_extension("unknown") {
        FileType::Other(ext) => assert_eq!(ext, "unknown"),
        _ => panic!("Expected FileType::Other"),
    }
}

#[test]
fn test_file_type_is_code() {
    assert!(FileType::Rust.is_code());
    assert!(FileType::TypeScript.is_code());
    assert!(FileType::Python.is_code());
    assert!(FileType::Go.is_code());
    assert!(FileType::Shell.is_code());

    // Config/doc files are not code
    assert!(!FileType::Json.is_code());
    assert!(!FileType::Yaml.is_code());
    assert!(!FileType::Markdown.is_code());
}

// ============================================================================
// CRUD TESTS
// ============================================================================

#[test]
fn test_store_and_get_file_memory() {
    let (store, _temp) = create_test_store();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let file_memory = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: project_id.clone(),
        user_id: user_id.to_string(),
        path: "src/main.rs".to_string(),
        absolute_path: "/project/src/main.rs".to_string(),
        file_hash: "abc123".to_string(),
        summary: "Main entry point".to_string(),
        key_items: vec!["fn main()".to_string(), "struct Config".to_string()],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 1024,
        line_count: 50,
        learned_from: LearnedFrom::ManualIndex,
        access_count: 0,
        last_accessed: chrono::Utc::now(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    // Store
    store.store(&file_memory).expect("Failed to store");

    // Get by ID
    let retrieved = store
        .get(user_id, &file_memory.id)
        .expect("Failed to get")
        .expect("File not found");

    assert_eq!(retrieved.path, "src/main.rs");
    assert_eq!(retrieved.summary, "Main entry point");
    assert_eq!(retrieved.key_items.len(), 2);
}

#[test]
fn test_list_by_project() {
    let (store, _temp) = create_test_store();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    // Store multiple files
    for i in 0..5 {
        let file_memory = FileMemory {
            id: FileMemoryId(Uuid::new_v4()),
            project_id: project_id.clone(),
            user_id: user_id.to_string(),
            path: format!("src/file{}.rs", i),
            absolute_path: format!("/project/src/file{}.rs", i),
            file_hash: format!("hash{}", i),
            summary: format!("File {}", i),
            key_items: vec![],
            purpose: None,
            connections: vec![],
            embedding: None,
            file_type: FileType::Rust,
            size_bytes: 100,
            line_count: 10,
            learned_from: LearnedFrom::ManualIndex,
            access_count: 0,
            last_accessed: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.store(&file_memory).expect("Failed to store");
    }

    // List all
    let files = store
        .list_by_project(user_id, &project_id, None)
        .expect("Failed to list");
    assert_eq!(files.len(), 5);

    // List with limit
    let limited = store
        .list_by_project(user_id, &project_id, Some(3))
        .expect("Failed to list");
    assert_eq!(limited.len(), 3);
}

#[test]
fn test_delete_file_memory() {
    let (store, _temp) = create_test_store();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let file_memory = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: project_id.clone(),
        user_id: user_id.to_string(),
        path: "src/delete_me.rs".to_string(),
        absolute_path: "/project/src/delete_me.rs".to_string(),
        file_hash: "deletehash".to_string(),
        summary: "To be deleted".to_string(),
        key_items: vec![],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 100,
        line_count: 10,
        learned_from: LearnedFrom::ManualIndex,
        access_count: 0,
        last_accessed: chrono::Utc::now(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    store.store(&file_memory).expect("Failed to store");

    // Verify exists
    assert!(store.get(user_id, &file_memory.id).unwrap().is_some());

    // Delete
    store
        .delete(user_id, &file_memory.id)
        .expect("Failed to delete");

    // Verify gone
    assert!(store.get(user_id, &file_memory.id).unwrap().is_none());
}

#[test]
fn test_delete_project_files() {
    let (store, _temp) = create_test_store();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());
    let other_project_id = ProjectId(Uuid::new_v4());

    // Store files for main project
    for i in 0..3 {
        let file_memory = FileMemory {
            id: FileMemoryId(Uuid::new_v4()),
            project_id: project_id.clone(),
            user_id: user_id.to_string(),
            path: format!("src/file{}.rs", i),
            absolute_path: format!("/project/src/file{}.rs", i),
            file_hash: format!("hash{}", i),
            summary: format!("File {}", i),
            key_items: vec![],
            purpose: None,
            connections: vec![],
            embedding: None,
            file_type: FileType::Rust,
            size_bytes: 100,
            line_count: 10,
            learned_from: LearnedFrom::ManualIndex,
            access_count: 0,
            last_accessed: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.store(&file_memory).expect("Failed to store");
    }

    // Store file for other project
    let other_file = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: other_project_id.clone(),
        user_id: user_id.to_string(),
        path: "other.rs".to_string(),
        absolute_path: "/other/other.rs".to_string(),
        file_hash: "otherhash".to_string(),
        summary: "Other project".to_string(),
        key_items: vec![],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 100,
        line_count: 10,
        learned_from: LearnedFrom::ManualIndex,
        access_count: 0,
        last_accessed: chrono::Utc::now(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.store(&other_file).expect("Failed to store");

    // Delete main project files
    let deleted = store
        .delete_project_files(user_id, &project_id)
        .expect("Failed to delete");
    assert_eq!(deleted, 3);

    // Main project files gone
    let main_files = store
        .list_by_project(user_id, &project_id, None)
        .expect("Failed to list");
    assert_eq!(main_files.len(), 0);

    // Other project file still exists
    let other_files = store
        .list_by_project(user_id, &other_project_id, None)
        .expect("Failed to list");
    assert_eq!(other_files.len(), 1);
}

// ============================================================================
// CODEBASE SCANNING TESTS
// ============================================================================

#[test]
fn test_scan_codebase() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();

    let result = store
        .scan_codebase(codebase.path(), None)
        .expect("Failed to scan");

    // Should find: main.rs, lib.rs, index.ts, utils.py, server.go, test_main.rs
    // Should skip: image.png (binary), large.txt (too big)
    assert!(result.total_files >= 6, "Expected at least 6 files");
    assert!(
        result.eligible_files >= 6,
        "Expected at least 6 eligible files"
    );
    assert!(!result.limit_reached, "Should not hit limit");

    // Check file paths include expected files
    let paths = result.file_paths.join(" ");
    assert!(paths.contains("main.rs") || paths.contains("lib.rs"));
}

#[test]
fn test_scan_codebase_with_limit() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();

    let config = CodebaseConfig {
        max_files_per_project: 2, // Very low limit
        ..Default::default()
    };

    let result = store
        .scan_codebase(codebase.path(), Some(&config))
        .expect("Failed to scan");

    // With only 2 max files, should hit limit since we have 6+ files
    assert!(result.limit_reached, "Should hit limit with max_files=2");
    // eligible_files may be >= limit due to how scan works
    assert!(
        result.file_paths.len() >= 2,
        "Should have at least 2 eligible files"
    );
}

#[test]
fn test_scan_codebase_with_excludes() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();

    let config = CodebaseConfig {
        exclude_patterns: vec!["tests/**".to_string(), "*.py".to_string()],
        ..Default::default()
    };

    let result = store
        .scan_codebase(codebase.path(), Some(&config))
        .expect("Failed to scan");

    // Should not include tests/test_main.rs or utils.py
    let paths = result.file_paths.join(" ");
    assert!(!paths.contains("test_main"), "Should exclude tests");
    assert!(!paths.contains(".py"), "Should exclude Python files");
}

// ============================================================================
// FILE INDEXING TESTS
// ============================================================================

#[test]
fn test_index_file() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let file_memory = store
        .index_file(codebase.path(), "src/main.rs", &project_id, user_id)
        .expect("Failed to index file");

    assert_eq!(file_memory.path, "src/main.rs");
    assert_eq!(file_memory.file_type, FileType::Rust);
    // Note: summary may be empty until AI-generated

    // Check key items extracted
    let key_items_str = file_memory.key_items.join(" ");
    assert!(
        key_items_str.contains("main") || key_items_str.contains("Config"),
        "Should extract pub fn or struct"
    );
}

#[test]
fn test_index_codebase() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let result = store
        .index_codebase(codebase.path(), &project_id, user_id, None)
        .expect("Failed to index codebase");

    assert!(result.indexed_files >= 5, "Should index at least 5 files");
    assert_eq!(result.errors.len(), 0, "Should have no errors");

    // Verify files are stored
    let files = store
        .list_by_project(user_id, &project_id, None)
        .expect("Failed to list");
    assert_eq!(files.len(), result.indexed_files);
}

// ============================================================================
// KEY ITEM EXTRACTION TESTS
// ============================================================================

#[test]
fn test_key_item_extraction_rust() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let file_memory = store
        .index_file(codebase.path(), "src/lib.rs", &project_id, user_id)
        .expect("Failed to index");

    // lib.rs contains: pub struct Library, pub trait Processor, pub enum Status
    let items = file_memory.key_items.join(" ");
    assert!(
        items.contains("Library") || items.contains("Processor") || items.contains("Status"),
        "Should extract Rust definitions: {}",
        items
    );
}

#[test]
fn test_key_item_extraction_typescript() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let file_memory = store
        .index_file(codebase.path(), "index.ts", &project_id, user_id)
        .expect("Failed to index");

    // index.ts contains: export class UserService, export interface User, export function validateUser
    let items = file_memory.key_items.join(" ");
    assert!(
        items.contains("UserService") || items.contains("User") || items.contains("validateUser"),
        "Should extract TypeScript definitions: {}",
        items
    );
}

#[test]
fn test_key_item_extraction_python() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let file_memory = store
        .index_file(codebase.path(), "utils.py", &project_id, user_id)
        .expect("Failed to index");

    // utils.py contains: class DataProcessor, def helper_function, async def fetch_data
    let items = file_memory.key_items.join(" ");
    assert!(
        items.contains("DataProcessor")
            || items.contains("helper_function")
            || items.contains("fetch_data"),
        "Should extract Python definitions: {}",
        items
    );
}

#[test]
fn test_key_item_extraction_go() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let file_memory = store
        .index_file(codebase.path(), "server.go", &project_id, user_id)
        .expect("Failed to index");

    // server.go contains: type Server struct, func NewServer, type Handler interface
    let items = file_memory.key_items.join(" ");
    assert!(
        items.contains("Server") || items.contains("NewServer") || items.contains("Handler"),
        "Should extract Go definitions: {}",
        items
    );
}

// ============================================================================
// HEAT SCORE TESTS
// ============================================================================

#[test]
fn test_heat_score_calculation() {
    let now = chrono::Utc::now();

    // Low access count -> heat 1
    let cold_file = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: ProjectId(Uuid::new_v4()),
        user_id: "test".to_string(),
        path: "cold.rs".to_string(),
        absolute_path: "/cold.rs".to_string(),
        file_hash: "hash".to_string(),
        summary: "Cold file".to_string(),
        key_items: vec![],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 100,
        line_count: 10,
        learned_from: LearnedFrom::ManualIndex,
        access_count: 1,
        last_accessed: now,
        created_at: now,
        updated_at: now,
    };

    // Medium access count -> heat 2
    let warm_file = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: ProjectId(Uuid::new_v4()),
        user_id: "test".to_string(),
        path: "warm.rs".to_string(),
        absolute_path: "/warm.rs".to_string(),
        file_hash: "hash".to_string(),
        summary: "Warm file".to_string(),
        key_items: vec![],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 100,
        line_count: 10,
        learned_from: LearnedFrom::ReadAccess,
        access_count: 5,
        last_accessed: now,
        created_at: now,
        updated_at: now,
    };

    // High access count -> heat 3
    let hot_file = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: ProjectId(Uuid::new_v4()),
        user_id: "test".to_string(),
        path: "hot.rs".to_string(),
        absolute_path: "/hot.rs".to_string(),
        file_hash: "hash".to_string(),
        summary: "Hot file".to_string(),
        key_items: vec![],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 100,
        line_count: 10,
        learned_from: LearnedFrom::EditAccess,
        access_count: 100,
        last_accessed: now,
        created_at: now,
        updated_at: now,
    };

    assert_eq!(cold_file.heat_score(), 1, "Low access = heat 1");
    assert_eq!(warm_file.heat_score(), 2, "Medium access = heat 2");
    assert_eq!(hot_file.heat_score(), 3, "High access = heat 3");
}

// ============================================================================
// STATS TESTS
// ============================================================================

#[test]
fn test_file_memory_stats() {
    let (store, _temp) = create_test_store();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    // Store files of different types
    let file_types = [FileType::Rust, FileType::TypeScript, FileType::Python];
    for (i, file_type) in file_types.iter().enumerate() {
        let file_memory = FileMemory {
            id: FileMemoryId(Uuid::new_v4()),
            project_id: project_id.clone(),
            user_id: user_id.to_string(),
            path: format!("file{}", i),
            absolute_path: format!("/file{}", i),
            file_hash: format!("hash{}", i),
            summary: format!("File {}", i),
            key_items: vec!["item1".to_string(), "item2".to_string()],
            purpose: None,
            connections: vec![],
            embedding: None,
            file_type: file_type.clone(),
            size_bytes: 1000 * (i + 1) as u64,
            line_count: 100 * (i + 1),
            learned_from: LearnedFrom::ManualIndex,
            access_count: 0,
            last_accessed: chrono::Utc::now(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        store.store(&file_memory).expect("Failed to store");
    }

    let stats = store.stats(user_id).expect("Failed to get stats");

    assert_eq!(stats.total_files, 3);
    assert_eq!(stats.total_size_bytes, 1000 + 2000 + 3000);
    assert_eq!(stats.by_type.len(), 3);
}

// ============================================================================
// EDGE CASES
// ============================================================================

#[test]
fn test_empty_codebase() {
    let (store, _temp) = create_test_store();
    let empty_dir = TempDir::new().expect("Failed to create temp dir");

    let result = store
        .scan_codebase(empty_dir.path(), None)
        .expect("Failed to scan");

    assert_eq!(result.total_files, 0);
    assert_eq!(result.eligible_files, 0);
}

#[test]
fn test_nonexistent_file() {
    let (store, _temp) = create_test_store();
    let codebase = create_test_codebase();
    let user_id = "test-user";
    let project_id = ProjectId(Uuid::new_v4());

    let result = store.index_file(codebase.path(), "nonexistent.rs", &project_id, user_id);

    assert!(result.is_err(), "Should fail for nonexistent file");
}

#[test]
fn test_user_isolation() {
    let (store, _temp) = create_test_store();
    let project_id = ProjectId(Uuid::new_v4());

    // Store file for user1
    let file1 = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: project_id.clone(),
        user_id: "user1".to_string(),
        path: "file.rs".to_string(),
        absolute_path: "/file.rs".to_string(),
        file_hash: "hash1".to_string(),
        summary: "User 1 file".to_string(),
        key_items: vec![],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 100,
        line_count: 10,
        learned_from: LearnedFrom::ManualIndex,
        access_count: 0,
        last_accessed: chrono::Utc::now(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.store(&file1).expect("Failed to store");

    // Store file for user2
    let file2 = FileMemory {
        id: FileMemoryId(Uuid::new_v4()),
        project_id: project_id.clone(),
        user_id: "user2".to_string(),
        path: "file.rs".to_string(),
        absolute_path: "/file.rs".to_string(),
        file_hash: "hash2".to_string(),
        summary: "User 2 file".to_string(),
        key_items: vec![],
        purpose: None,
        connections: vec![],
        embedding: None,
        file_type: FileType::Rust,
        size_bytes: 100,
        line_count: 10,
        learned_from: LearnedFrom::ManualIndex,
        access_count: 0,
        last_accessed: chrono::Utc::now(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    store.store(&file2).expect("Failed to store");

    // User1 should only see their file
    let user1_files = store
        .list_by_project("user1", &project_id, None)
        .expect("Failed to list");
    assert_eq!(user1_files.len(), 1);
    assert_eq!(user1_files[0].summary, "User 1 file");

    // User2 should only see their file
    let user2_files = store
        .list_by_project("user2", &project_id, None)
        .expect("Failed to list");
    assert_eq!(user2_files.len(), 1);
    assert_eq!(user2_files[0].summary, "User 2 file");
}
