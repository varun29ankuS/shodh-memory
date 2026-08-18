//! The watched-folder connector: enumerate a directory, ingest what changed,
//! commit a cursor per item, and leave a run record behind.
//!
//! # Poll, not watch
//!
//! Event-based watching is not used and is not a future upgrade in disguise.
//! Both platform APIs lose events *silently* under load — inotify drops past
//! `max_user_watches` and signals `IN_Q_OVERFLOW`; `ReadDirectoryChangesW`
//! discards its whole buffer and still returns success with
//! `lpBytesReturned == 0` — and Microsoft's own documented recovery from
//! `ERROR_NOTIFY_ENUM_DIR` is *"compute the changes by enumerating the
//! directory or subtree"*. A poller runs that recovery unconditionally. Over
//! NFS no events are emitted at all. And a missed poll is visible as a stale
//! `last_run`, where a missed event is visible as nothing.
//!
//! # The security invariant
//!
//! > **A caller-supplied string is never used to construct a path.**
//!
//! The walk *enumerates*; a caller never names a file. Globs are filters
//! applied to already-enumerated root-relative paths, never path constructors.
//! No endpoint in this subsystem accepts a file path and reads it. The only
//! caller-supplied path anywhere is a source's `root`, and it is canonicalised
//! once at registration by [`validate_root`], stored resolved, and never
//! re-resolved.
//!
//! # Why this is not `scan_project_codebase`
//!
//! `POST /api/projects/{id}/scan` writes `FileMemory` records into the `files`
//! column family — a code-navigation index with a heat score and an access
//! count. Those are not memories: not in the vector index as memories, not in
//! the graph, not recallable, no cursor, no provenance. Different product,
//! different store. This module reuses the *patterns* from
//! `crate::memory::files` (the excluded-directory list, compile-globs-once,
//! the binary extension list) and none of its code path.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use chrono::Utc;
use sha2::{Digest, Sha256};

use crate::errors::AppError;
use crate::handlers::router::AppState;
use crate::memory::sources::{
    ItemCursor, ItemState, RunLease, RunStatus, RunTrigger, SourceDefinition, SourceId, SourceRun,
    SourceRuntime, WatchedFolderConfig, QUARANTINE_THRESHOLD, SUPERSEDED_KEEP,
};
use crate::memory::types::MemoryOrigin;
use crate::validation;

/// Metadata keys, namespaced like the existing `shodh.surprise`.
pub const META_SOURCE_ID: &str = "shodh.source.id";
pub const META_SOURCE_KIND: &str = "shodh.source.kind";
pub const META_RUN_ID: &str = "shodh.source.run_id";
pub const META_ITEM: &str = "shodh.source.item";
pub const META_ITEM_SHA256: &str = "shodh.source.item_sha256";
pub const META_PART: &str = "shodh.source.part";

/// Largest body handed to a single memory. `validation::MAX_CONTENT_LENGTH` is
/// 50,000 **bytes**; the margin absorbs the fact that a paragraph boundary
/// rarely lands on the limit.
const MAX_PART_BYTES: usize = 45_000;

/// Rewrite the run record and refresh the lease at least this often.
const HEARTBEAT_EVERY_ITEMS: u32 = 100;
const HEARTBEAT_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

/// Bytes sniffed for a NUL before deciding a file is binary.
const BINARY_SNIFF_BYTES: usize = 8 * 1024;

/// Environment switch: allow a UNC / network root.
const ENV_ALLOW_NETWORK: &str = "SHODH_SOURCE_ALLOW_NETWORK";
/// Environment switch: restrict every source root to these trees.
const ENV_SOURCE_ROOTS: &str = "SHODH_SOURCE_ROOTS";

/// Directory names never descended into. Mirrors the list in
/// `crate::memory::files`, which declares its own inside a method body and so
/// cannot be shared. `.git`, `.ssh` and `.gnupg` are here for a second reason:
/// they hold credential material, and entering them is counted as a policy
/// denial rather than an ordinary skip.
const EXCLUDED_DIR_NAMES: &[&str] = &[
    ".git",
    ".svn",
    ".hg",
    ".bzr",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "virtualenv",
    "site-packages",
    "target",
    "dist",
    "build",
    ".next",
    ".cache",
    ".idea",
    ".vscode",
];

/// Directories whose contents are credential material.
const CREDENTIAL_DIR_NAMES: &[&str] = &[".git", ".ssh", ".gnupg", ".aws"];

/// Filename patterns refused before any read, matched case-insensitively
/// against the file name.
///
/// **Not user-editable, by design.** A deny-list a user can edit is a
/// deny-list that gets edited, and this one is the difference between a
/// connector that reads notes and a connector that reads private keys. These
/// are applied on top of `exclude_globs`, never instead of them.
fn is_credential_shaped(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    const SUFFIXES: &[&str] = &[
        ".pem", ".key", ".pfx", ".p12", ".kdbx", ".ovpn", ".jks", ".keystore", ".asc", ".gpg",
    ];
    const PREFIXES: &[&str] = &["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", ".env"];
    const EXACT: &[&str] = &[
        "credentials",
        ".netrc",
        "_netrc",
        ".htpasswd",
        ".pgpass",
        "known_hosts",
        "authorized_keys",
    ];
    SUFFIXES.iter().any(|s| lower.ends_with(s))
        || PREFIXES.iter().any(|p| lower.starts_with(p))
        || EXACT.contains(&lower.as_str())
}

/// Extensions that are binary regardless of content. Mirrors the list in
/// `crate::memory::files`, which is private to a method there.
const BINARY_EXTENSIONS: &[&str] = &[
    "exe", "dll", "so", "dylib", "bin", "obj", "o", "a", "lib", "png", "jpg", "jpeg", "gif", "bmp",
    "ico", "webp", "mp3", "mp4", "avi", "mov", "mkv", "wav", "flac", "zip", "tar", "gz", "rar",
    "7z", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "woff", "woff2", "ttf", "otf", "eot",
    "class", "pyc", "pyo", "wasm", "db", "sqlite", "sst",
];

fn has_binary_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| BINARY_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Lowercase hex sha256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// The stable identity of one item under one source: the sha256 of its
/// root-relative, `/`-separated path.
///
/// Hashed rather than embedded in the key so the key is fixed-length and
/// separator-safe — a Windows path carries both `:` and `\`, either of which
/// would corrupt a delimited key — and so key size is bounded regardless of
/// path depth. The readable path lives inside the value.
pub fn item_hash(relative_path: &str) -> String {
    sha256_hex(relative_path.as_bytes())
}

/// The external id a single-part item's memory is bound to. Stable across
/// edits, unique per item, and the shape `external_id` was built for
/// (`linear:SHO-39`, `github:pr-123`).
pub fn item_external_id(source_id: &SourceId, item_hash: &str) -> String {
    format!("src:{}:{}", source_id.0, item_hash)
}

// ---------------------------------------------------------------------------
// Registration-time root validation
// ---------------------------------------------------------------------------

fn invalid_root(reason: impl Into<String>) -> AppError {
    AppError::InvalidInput {
        field: "config.root".to_string(),
        reason: reason.into(),
    }
}

/// Canonical paths that a source root may never be, contain, or sit inside.
fn deny_roots(base_path: &Path) -> Vec<PathBuf> {
    let mut denied: Vec<PathBuf> = Vec::new();

    // Our own storage. Ingesting the store's RocksDB into the store is a
    // corruption-and-blowup path, not merely a waste.
    push_canonical(&mut denied, base_path.to_path_buf());

    for var in [
        "WINDIR",
        "SystemRoot",
        "ProgramFiles",
        "ProgramFiles(x86)",
        "ProgramData",
    ] {
        if let Ok(v) = std::env::var(var) {
            push_canonical(&mut denied, PathBuf::from(v));
        }
    }

    for fixed in [
        "/etc", "/proc", "/sys", "/dev", "/boot", "/usr", "/bin", "/sbin", "/var/lib",
    ] {
        push_canonical(&mut denied, PathBuf::from(fixed));
    }

    if let Some(home) = home_dir() {
        for rel in [
            ".ssh",
            ".gnupg",
            ".aws",
            ".config/gcloud",
            ".kube",
            ".docker",
            "AppData/Roaming/gcloud",
        ] {
            push_canonical(&mut denied, home.join(rel));
        }
    }

    denied
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .ok()
        .map(PathBuf::from)
}

fn push_canonical(out: &mut Vec<PathBuf>, path: PathBuf) {
    // A deny entry that does not exist cannot contain anything, and comparing
    // an un-canonicalised entry against a canonical root would silently never
    // match on Windows, where canonicalisation adds a `\\?\` prefix.
    if let Ok(c) = std::fs::canonicalize(&path) {
        out.push(c);
    }
}

fn is_related(a: &Path, b: &Path) -> bool {
    a == b || a.starts_with(b) || b.starts_with(a)
}

/// Resolve and vet a caller-supplied root exactly once.
///
/// Every rejection is a distinct `InvalidInput` on `config.root`, mirroring
/// the `exists()` -> `is_dir()` -> `canonicalize()` ladder
/// `files::scan_project_codebase` already uses. The canonical form is what is
/// returned, stored and walked; the string the caller typed is never used
/// again.
pub fn validate_root(
    raw: &str,
    base_path: &Path,
    existing: &[SourceDefinition],
) -> Result<PathBuf, AppError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(invalid_root("root is required"));
    }
    let requested = PathBuf::from(trimmed);
    if !requested.is_absolute() {
        return Err(invalid_root(
            "root must be an absolute path; a relative path would resolve against the \
             server's working directory, which is not something a caller can see",
        ));
    }

    // A symlinked root would be resolved away by canonicalize, so the vetting
    // below would apply to the target while the caller believes it applies to
    // the link. Refuse the link instead of resolving it.
    match std::fs::symlink_metadata(&requested) {
        Ok(md) if md.file_type().is_symlink() || is_reparse_point(&md) => {
            return Err(invalid_root(
                "root is a symlink, junction or mount point; register the directory it \
                 points at so the path that is vetted is the path that is read",
            ));
        }
        Ok(_) => {}
        Err(e) => return Err(invalid_root(format!("root cannot be read: {e}"))),
    }

    let canonical = std::fs::canonicalize(&requested)
        .map_err(|e| invalid_root(format!("root cannot be resolved: {e}")))?;

    if !canonical.is_dir() {
        return Err(invalid_root("root is not a directory"));
    }

    // A volume or filesystem root has one non-prefix component (the root
    // separator) and no name. Pointing a connector at `C:\` or `/` is refused
    // at registration rather than survived at run time.
    let depth = canonical
        .components()
        .filter(|c| !matches!(c, Component::Prefix(_)))
        .count();
    if depth < 2 {
        return Err(invalid_root(
            "root is a filesystem or volume root; choose a directory inside it",
        ));
    }

    let canonical_str = canonical.to_string_lossy().to_string();
    let is_unc = canonical_str.starts_with(r"\\?\UNC\")
        || (canonical_str.starts_with(r"\\") && !canonical_str.starts_with(r"\\?\"));
    if is_unc && !env_flag(ENV_ALLOW_NETWORK) {
        return Err(invalid_root(format!(
            "root is a network path; every poll would become an unbounded-latency network \
             operation. Set {ENV_ALLOW_NETWORK}=1 to allow it."
        )));
    }

    for denied in deny_roots(base_path) {
        if is_related(&canonical, &denied) {
            return Err(invalid_root(format!(
                "root overlaps a protected directory ({})",
                denied.display()
            )));
        }
    }

    if let Ok(allow) = std::env::var(ENV_SOURCE_ROOTS) {
        if !allow.trim().is_empty() {
            let mut permitted = false;
            for entry in std::env::split_paths(&allow) {
                if let Ok(c) = std::fs::canonicalize(&entry) {
                    if canonical.starts_with(&c) {
                        permitted = true;
                        break;
                    }
                }
            }
            if !permitted {
                return Err(invalid_root(format!(
                    "root is outside every tree listed in {ENV_SOURCE_ROOTS}"
                )));
            }
        }
    }

    for def in existing {
        let other = PathBuf::from(&def.config.as_watched_folder().root);
        if is_related(&canonical, &other) {
            return Err(invalid_root(format!(
                "root overlaps the existing source '{}'; two sources over the same files \
                 produce two cursors pointing at one memory",
                def.name
            )));
        }
    }

    Ok(canonical)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Windows junctions and mount points are not symlinks to
/// `FileType::is_symlink`, but they redirect exactly the same way. They carry
/// `FILE_ATTRIBUTE_REPARSE_POINT`.
#[cfg(windows)]
fn is_reparse_point(md: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
    md.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_reparse_point(_md: &std::fs::Metadata) -> bool {
    false
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct WalkEntry {
    /// Root-relative, `/`-separated.
    rel: String,
    abs: PathBuf,
    size: u64,
    mtime_unix_nanos: Option<i64>,
}

#[derive(Debug, Default)]
struct WalkOutcome {
    entries: Vec<WalkEntry>,
    /// Entries a security rule refused before any read.
    denied: u32,
    /// Directories that could not be listed. Retryable.
    failures: Vec<(String, String)>,
    truncated_by: Option<String>,
}

fn compile_globs(patterns: &[String], field: &str) -> Result<Vec<glob::Pattern>, AppError> {
    patterns
        .iter()
        .map(|p| {
            glob::Pattern::new(p).map_err(|e| AppError::InvalidInput {
                field: field.to_string(),
                reason: format!("invalid glob '{p}': {e}"),
            })
        })
        .collect()
}

fn mtime_nanos(md: &std::fs::Metadata) -> Option<i64> {
    let modified = md.modified().ok()?;
    let dt: chrono::DateTime<Utc> = modified.into();
    dt.timestamp_nanos_opt()
}

/// Depth-first enumeration under `root`, applying every walk-time control.
///
/// Blocking: filesystem metadata calls are blocking, so the caller runs this
/// inside `spawn_blocking`. Only metadata is collected here — bounded by
/// `max_files_per_run` — so a run never holds a folder's worth of file bytes.
fn walk_root(
    root: &Path,
    cfg: &WatchedFolderConfig,
    include: &[glob::Pattern],
    exclude: &[glob::Pattern],
) -> WalkOutcome {
    let mut out = WalkOutcome::default();
    let mut stack: Vec<(PathBuf, u16)> = vec![(root.to_path_buf(), 0)];

    while let Some((dir, depth)) = stack.pop() {
        let read = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(e) => {
                out.failures.push((
                    display_relative(root, &dir),
                    format!("directory cannot be listed: {e}"),
                ));
                continue;
            }
        };

        // `read_dir` order is filesystem-defined. Sorting makes a truncated run
        // truncate the same way twice, which is the difference between "stopped
        // at 2,000 files" being resumable and being a lottery.
        let mut children: Vec<PathBuf> = Vec::new();
        for entry in read {
            match entry {
                Ok(e) => children.push(e.path()),
                Err(e) => out
                    .failures
                    .push((display_relative(root, &dir), format!("entry unreadable: {e}"))),
            }
        }
        children.sort();

        for path in children {
            let md = match std::fs::symlink_metadata(&path) {
                Ok(md) => md,
                Err(e) => {
                    out.failures.push((
                        display_relative(root, &path),
                        format!("metadata unreadable: {e}"),
                    ));
                    continue;
                }
            };

            // Refuse the link rather than resolve it. Refusing is unambiguous;
            // canonicalise-then-compare on a followed link is the version that
            // gets subtly wrong.
            if md.file_type().is_symlink() || is_reparse_point(&md) {
                out.denied += 1;
                continue;
            }

            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => {
                    out.denied += 1;
                    continue;
                }
            };

            if md.is_dir() {
                if CREDENTIAL_DIR_NAMES.contains(&name.as_str()) {
                    out.denied += 1;
                    continue;
                }
                if EXCLUDED_DIR_NAMES.contains(&name.as_str()) {
                    continue;
                }
                if depth + 1 <= cfg.max_depth {
                    stack.push((path, depth + 1));
                }
                continue;
            }

            if !md.is_file() {
                continue;
            }

            let Some(rel) = relative_slash_path(root, &path) else {
                // Outside the root, or unrepresentable. Either is a control
                // failure, not an ordinary skip.
                out.denied += 1;
                continue;
            };

            if !include.iter().any(|p| p.matches(&rel)) {
                continue;
            }
            if exclude.iter().any(|p| p.matches(&rel)) {
                continue;
            }
            // Checked after the include/exclude filters so this count means
            // "credential-shaped files this configuration would otherwise have
            // read", which is the number worth showing.
            if is_credential_shaped(&name) {
                out.denied += 1;
                continue;
            }

            if out.entries.len() as u32 >= cfg.max_files_per_run {
                out.truncated_by = Some("max_files_per_run".to_string());
                return out;
            }

            out.entries.push(WalkEntry {
                rel,
                abs: path,
                size: md.len(),
                mtime_unix_nanos: mtime_nanos(&md),
            });
        }
    }

    out.entries.sort_by(|a, b| a.rel.cmp(&b.rel));
    out
}

fn relative_slash_path(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let mut parts: Vec<String> = Vec::new();
    for c in rel.components() {
        match c {
            Component::Normal(n) => parts.push(n.to_str()?.to_string()),
            // A `..` or a root inside a stripped relative path means the path
            // did not actually sit under the root.
            _ => return None,
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("/"))
}

fn display_relative(root: &Path, path: &Path) -> String {
    relative_slash_path(root, path).unwrap_or_else(|| path.display().to_string())
}

// ---------------------------------------------------------------------------
// Reading one item
// ---------------------------------------------------------------------------

/// Why an item was not ingested. Every variant maps to a cursor state, so a
/// refusal is always visible in the item list rather than only in a log.
enum ReadRefusal {
    /// Deliberate and not an error. Retried when size or mtime move.
    Skipped(String),
    /// Transient. Retried on the next run.
    Failed(String),
    /// A security control refused it. Counted separately and logged loudly,
    /// because reaching one of these means an earlier control did not hold.
    Denied(String),
}

/// Canonicalise, confirm containment, read, and decode — in that order.
///
/// Blocking. Canonicalisation happens *before* the open, not after: a file
/// that resolves outside the root must never be read at all, and reading it
/// first and checking afterwards is a control that has already failed by the
/// time it fires.
fn read_item(entry: &WalkEntry, root: &Path, max_file_bytes: u64) -> Result<String, ReadRefusal> {
    if entry.size > max_file_bytes {
        return Err(ReadRefusal::Skipped(format!(
            "file is {} bytes, over max_file_bytes ({max_file_bytes})",
            entry.size
        )));
    }
    if has_binary_extension(&entry.abs) {
        return Err(ReadRefusal::Skipped("binary file extension".to_string()));
    }

    let canonical = std::fs::canonicalize(&entry.abs)
        .map_err(|e| ReadRefusal::Failed(format!("cannot resolve: {e}")))?;
    if !canonical.starts_with(root) {
        return Err(ReadRefusal::Denied(
            "resolves outside the source root".to_string(),
        ));
    }

    let bytes = std::fs::read(&canonical).map_err(|e| ReadRefusal::Failed(format!("{e}")))?;

    if bytes
        .iter()
        .take(BINARY_SNIFF_BYTES)
        .any(|b| *b == 0)
    {
        return Err(ReadRefusal::Skipped("binary content".to_string()));
    }

    // Never a lossy decode. Mojibake in the embedding index is unrecoverable
    // and silent; a skip is neither.
    String::from_utf8(bytes).map_err(|_| ReadRefusal::Skipped("not valid UTF-8".to_string()))
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

/// Split a document into parts no larger than [`MAX_PART_BYTES`], preferring
/// paragraph boundaries, then line boundaries, then a character boundary.
///
/// A document longer than `validation::MAX_CONTENT_LENGTH` cannot be one
/// memory. `crate::embeddings::chunking` does not solve this: it chunks for the
/// 128-token embedding window and maps chunk vectors back to one parent memory,
/// which makes a large memory *retrievable*, not *storable*.
pub fn split_content(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    if trimmed.len() <= MAX_PART_BYTES {
        return vec![trimmed.to_string()];
    }

    let mut parts: Vec<String> = Vec::new();
    let mut rest = trimmed;

    while !rest.is_empty() {
        if rest.len() <= MAX_PART_BYTES {
            parts.push(rest.to_string());
            break;
        }
        let window = &rest[..ceil_char_boundary(rest, MAX_PART_BYTES)];
        let cut = window
            .rfind("\n\n")
            .map(|i| i + 2)
            .or_else(|| window.rfind('\n').map(|i| i + 1))
            .filter(|i| *i > 0)
            .unwrap_or_else(|| floor_char_boundary(rest, MAX_PART_BYTES));

        let (head, tail) = rest.split_at(cut);
        let head = head.trim();
        if !head.is_empty() {
            parts.push(head.to_string());
        }
        rest = tail.trim_start();
    }

    // A trailing sliver is not a document; fold it into its predecessor rather
    // than storing a memory too short to mean anything.
    if parts.len() > 1 {
        let last_too_small = parts
            .last()
            .map(|p| p.trim().len() < validation::MIN_MEANINGFUL_CONTENT_LENGTH)
            .unwrap_or(false);
        if last_too_small {
            let tail = parts.pop().unwrap_or_default();
            if let Some(prev) = parts.last_mut() {
                prev.push('\n');
                prev.push_str(tail.trim());
            }
        }
    }

    parts
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i.max(1)
}

fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// What a single item's write produced.
struct ItemWrite {
    memory_ids: Vec<uuid::Uuid>,
    memories_written: u32,
    /// Every part was already in the store with identical content.
    all_deduped: bool,
    /// The external id the item's memory is bound to, when written as a
    /// single part.
    external_id: Option<String>,
}

/// Execute one run of a watched-folder source, start to finish.
///
/// Returns the final run record. Errors only when the registry itself cannot
/// be written — every *ingestion* failure is recorded on the run and on the
/// item's cursor, because a connector whose failures are exceptions is a
/// connector that stops on the first locked file.
pub async fn execute_run(
    state: &AppState,
    def: &SourceDefinition,
    trigger: RunTrigger,
    force: bool,
) -> Result<SourceRun, AppError> {
    let store = state.source_store.clone();
    let cfg = def.config.as_watched_folder().clone();
    let root = PathBuf::from(&cfg.root);

    let started_at = Utc::now();
    let mut run = SourceRun::start(def, trigger, started_at);
    let mut lease = RunLease {
        run_id: run.run_id,
        started_at,
        run_started_nanos: run.started_nanos,
        heartbeat_at: started_at,
        pid: std::process::id(),
    };

    // Record the intent before the side effect. After this write a crash is
    // detectable as "started, never finished"; before it, a crash is invisible.
    store.begin_run(&run, &lease).map_err(AppError::Internal)?;

    let mut runtime = store
        .get_runtime(&def.user_id, &def.id)
        .map_err(AppError::Internal)?;
    let rehash_due = cfg.rehash_every_n_runs > 0
        && (runtime.run_count + 1) % u64::from(cfg.rehash_every_n_runs) == 0;
    let full_read = force || rehash_due;

    // A run-level fatal still has to leave a run record behind, which is why
    // the lease was taken first.
    if let Err(reason) = revalidate_root(&root) {
        run.status = RunStatus::Failed;
        run.error = Some(reason);
        return finish(state, def, run, runtime).await;
    }

    let include = match compile_globs(&cfg.include_globs, "config.include_globs") {
        Ok(g) => g,
        Err(e) => {
            run.status = RunStatus::Failed;
            run.error = Some(e.message());
            return finish(state, def, run, runtime).await;
        }
    };
    let exclude = match compile_globs(&cfg.exclude_globs, "config.exclude_globs") {
        Ok(g) => g,
        Err(e) => {
            run.status = RunStatus::Failed;
            run.error = Some(e.message());
            return finish(state, def, run, runtime).await;
        }
    };

    let walk = {
        let root = root.clone();
        let cfg = cfg.clone();
        tokio::task::spawn_blocking(move || walk_root(&root, &cfg, &include, &exclude))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Walk task panicked: {e}")))?
    };

    run.items_denied_by_policy = walk.denied;
    run.truncated_by = walk.truncated_by.clone();
    for (item, reason) in &walk.failures {
        run.items_failed += 1;
        run.push_failure(item, reason, true);
    }

    let experience_type =
        crate::handlers::remember::parse_experience_type(Some(&cfg.memory_type)).map_err(|e| {
            AppError::InvalidInput {
                field: "config.memory_type".to_string(),
                reason: e.message(),
            }
        })?;

    let mut last_heartbeat = std::time::Instant::now();
    let mut since_heartbeat = 0u32;

    for entry in walk.entries {
        let hash = item_hash(&entry.rel);
        let existing = store
            .get_cursor(&def.id, &hash)
            .map_err(AppError::Internal)?;

        run.items_seen += 1;

        if let Some(prior) = &existing {
            if matches!(prior.state, ItemState::Quarantined { .. }) && !force {
                run.items_skipped += 1;
                let mut cursor = prior.clone();
                cursor.last_seen_at = Utc::now();
                cursor.last_run_id = run.run_id;
                store
                    .touch_cursor(&def.id, &hash, &cursor)
                    .map_err(AppError::Internal)?;
                continue;
            }

            // Layer 1: size and mtime. A miss costs a read, never a duplicate;
            // a platform that reports no mtime forces the slow path, because a
            // missing signal must never be read as "unchanged".
            let fast_path_hit = !full_read
                && prior.size_bytes == entry.size
                && prior.mtime_unix_nanos.is_some()
                && prior.mtime_unix_nanos == entry.mtime_unix_nanos;
            if fast_path_hit {
                run.items_unchanged += 1;
                let mut cursor = prior.clone();
                cursor.last_seen_at = Utc::now();
                cursor.last_run_id = run.run_id;
                store
                    .touch_cursor(&def.id, &hash, &cursor)
                    .map_err(AppError::Internal)?;
                continue;
            }
        }

        if run.bytes_read.saturating_add(entry.size) > cfg.max_run_bytes {
            run.truncated_by = Some("max_run_bytes".to_string());
            run.items_seen -= 1;
            break;
        }

        let read = {
            let entry = entry.clone();
            let root = root.clone();
            let max_file_bytes = cfg.max_file_bytes;
            tokio::task::spawn_blocking(move || read_item(&entry, &root, max_file_bytes))
                .await
                .map_err(|e| AppError::Internal(anyhow::anyhow!("Read task panicked: {e}")))?
        };

        let text = match read {
            Ok(text) => text,
            Err(refusal) => {
                apply_refusal(&store, def, &mut run, &entry, &hash, existing.as_ref(), refusal)?;
                since_heartbeat += 1;
                continue;
            }
        };

        run.bytes_read = run.bytes_read.saturating_add(entry.size);
        let content_sha = sha256_hex(text.as_bytes());

        // Layer 2: the file's own content hash. A `touch` that did not change
        // the bytes lands here and costs a read, not a duplicate memory.
        if let Some(prior) = &existing {
            if prior.content_sha256 == content_sha
                && matches!(prior.state, ItemState::Ingested | ItemState::Deduped)
            {
                run.items_unchanged += 1;
                let mut cursor = prior.clone();
                cursor.size_bytes = entry.size;
                cursor.mtime_unix_nanos = entry.mtime_unix_nanos;
                cursor.last_seen_at = Utc::now();
                cursor.last_run_id = run.run_id;
                store
                    .touch_cursor(&def.id, &hash, &cursor)
                    .map_err(AppError::Internal)?;
                continue;
            }
        }

        let parts = split_content(&text);
        if parts.is_empty()
            || parts[0].trim().len() < validation::MIN_MEANINGFUL_CONTENT_LENGTH
        {
            apply_refusal(
                &store,
                def,
                &mut run,
                &entry,
                &hash,
                existing.as_ref(),
                ReadRefusal::Skipped("file holds no meaningful text".to_string()),
            )?;
            since_heartbeat += 1;
            continue;
        }

        match write_item(
            state,
            def,
            &run,
            &entry.rel,
            &hash,
            &content_sha,
            &experience_type,
            &cfg,
            parts,
        )
        .await
        {
            Ok(write) => {
                let now = Utc::now();
                let superseded = supersede(existing.as_ref(), &write.memory_ids);
                let cursor = ItemCursor {
                    path: entry.rel.clone(),
                    size_bytes: entry.size,
                    mtime_unix_nanos: entry.mtime_unix_nanos,
                    content_sha256: content_sha,
                    memory_ids: write.memory_ids,
                    superseded_memory_ids: superseded,
                    first_ingested_at: existing
                        .as_ref()
                        .map(|c| c.first_ingested_at)
                        .unwrap_or(now),
                    last_ingested_at: now,
                    last_seen_at: now,
                    last_run_id: run.run_id,
                    state: if write.all_deduped {
                        ItemState::Deduped
                    } else {
                        ItemState::Ingested
                    },
                    consecutive_failures: 0,
                    external_id: write.external_id,
                };
                if write.all_deduped {
                    run.items_deduped += 1;
                } else {
                    run.items_ingested += 1;
                }
                run.memories_written += write.memories_written;
                // The cursor advance, and ONLY after every part is durable.
                store
                    .commit_item(&def.id, &hash, &cursor, &run)
                    .map_err(AppError::Internal)?;
            }
            Err(e) => {
                apply_refusal(
                    &store,
                    def,
                    &mut run,
                    &entry,
                    &hash,
                    existing.as_ref(),
                    ReadRefusal::Failed(e.message()),
                )?;
            }
        }

        since_heartbeat += 1;
        if since_heartbeat >= HEARTBEAT_EVERY_ITEMS || last_heartbeat.elapsed() >= HEARTBEAT_EVERY {
            store
                .heartbeat(&run, &mut lease)
                .map_err(AppError::Internal)?;
            since_heartbeat = 0;
            last_heartbeat = std::time::Instant::now();
        }
    }

    // One walk of the cursors: disappearance detection and the item totals the
    // dashboard reads, so the listing endpoint never has to scan them.
    let cursors = store
        .list_cursors(&def.id)
        .map_err(AppError::Internal)?;
    let mut tracked = 0u64;
    let mut failed = 0u64;
    let mut quarantined = 0u64;
    for cursor in &cursors {
        tracked += 1;
        match cursor.state {
            ItemState::Failed { .. } => failed += 1,
            ItemState::Quarantined { .. } => quarantined += 1,
            _ => {}
        }
        // A file that disappeared is counted and its cursor is kept. Its
        // memories are NOT deleted: a memory store records what was observed,
        // and silently deleting a corpus because a share was unmounted is
        // unrecoverable.
        if cursor.last_seen_at < run.started_at {
            run.items_disappeared += 1;
        }
    }
    runtime.items_tracked = tracked;
    runtime.items_failed = failed;
    runtime.items_quarantined = quarantined;

    run.status = if run.items_failed > 0 {
        RunStatus::PartiallyFailed
    } else {
        RunStatus::Succeeded
    };

    finish(state, def, run, runtime).await
}

fn revalidate_root(root: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(root) {
        Ok(md) if md.file_type().is_symlink() || is_reparse_point(&md) => {
            Err("root became a symlink, junction or mount point since registration".to_string())
        }
        Ok(md) if !md.is_dir() => Err("root is no longer a directory".to_string()),
        Ok(_) => Ok(()),
        Err(e) => Err(format!("root cannot be read: {e}")),
    }
}

/// Record a refusal on the item's cursor and the run.
///
/// The cursor **is** the dead-letter queue: a durable failure record with the
/// reason attached, queryable per state. There is no second store, and there
/// is no tolerance switch in v1 — a partial run that reports itself beats a
/// nightly ingest halted by one locked file.
#[allow(clippy::too_many_arguments)]
fn apply_refusal(
    store: &crate::memory::sources::SourceStore,
    def: &SourceDefinition,
    run: &mut SourceRun,
    entry: &WalkEntry,
    hash: &str,
    existing: Option<&ItemCursor>,
    refusal: ReadRefusal,
) -> Result<(), AppError> {
    let now = Utc::now();
    let prior_failures = existing.map(|c| c.consecutive_failures).unwrap_or(0);

    let (state, retryable) = match refusal {
        ReadRefusal::Skipped(reason) => {
            run.items_skipped += 1;
            (ItemState::Skipped { reason }, false)
        }
        ReadRefusal::Denied(reason) => {
            run.items_denied_by_policy += 1;
            tracing::warn!(
                source = %def.id,
                item = %entry.rel,
                reason = %reason,
                "Ingestion refused an item at read time: a walk-time control did not hold"
            );
            (ItemState::Skipped { reason }, false)
        }
        ReadRefusal::Failed(reason) => {
            run.items_failed += 1;
            run.push_failure(&entry.rel, &reason, true);
            let failures = prior_failures.saturating_add(1);
            if failures >= QUARANTINE_THRESHOLD {
                (ItemState::Quarantined { reason }, true)
            } else {
                (ItemState::Failed { reason }, true)
            }
        }
    };

    // On a failure the previous size, mtime and content hash are KEPT. Writing
    // the new ones would let the fast path skip the file on the next run, which
    // would turn a transient read error into a permanent one.
    let (size_bytes, mtime, content_sha256) = if retryable {
        existing
            .map(|c| {
                (
                    c.size_bytes,
                    c.mtime_unix_nanos,
                    c.content_sha256.clone(),
                )
            })
            .unwrap_or((0, None, String::new()))
    } else {
        (entry.size, entry.mtime_unix_nanos, String::new())
    };

    let cursor = ItemCursor {
        path: entry.rel.clone(),
        size_bytes,
        mtime_unix_nanos: mtime,
        content_sha256,
        memory_ids: existing.map(|c| c.memory_ids.clone()).unwrap_or_default(),
        superseded_memory_ids: existing
            .map(|c| c.superseded_memory_ids.clone())
            .unwrap_or_default(),
        first_ingested_at: existing.map(|c| c.first_ingested_at).unwrap_or(now),
        last_ingested_at: existing.map(|c| c.last_ingested_at).unwrap_or(now),
        last_seen_at: now,
        last_run_id: run.run_id,
        state,
        consecutive_failures: if retryable {
            prior_failures.saturating_add(1)
        } else {
            0
        },
        external_id: existing.and_then(|c| c.external_id.clone()),
    };

    store
        .commit_item(&def.id, hash, &cursor, run)
        .map_err(AppError::Internal)
}

/// Memory ids from earlier versions of an item, newest first.
///
/// Without this the overwrite of `memory_ids` is the moment the registry loses
/// the only link between a stale memory and the file that produced it. Under
/// the upsert path the id does not change and nothing is superseded; under the
/// append path, and across a single-part/multi-part transition, it does.
fn supersede(existing: Option<&ItemCursor>, new_ids: &[uuid::Uuid]) -> Vec<uuid::Uuid> {
    let Some(prior) = existing else {
        return Vec::new();
    };
    let current: HashSet<uuid::Uuid> = new_ids.iter().copied().collect();
    let mut out: Vec<uuid::Uuid> = prior
        .memory_ids
        .iter()
        .copied()
        .filter(|id| !current.contains(id))
        .collect();
    for id in &prior.superseded_memory_ids {
        if !current.contains(id) && !out.contains(id) {
            out.push(*id);
        }
    }
    out.truncate(SUPERSEDED_KEEP);
    out
}

/// Write one item's parts.
///
/// # The changed-file rule
///
/// **A single-part version upserts; a split version appends.** Upsert keys the
/// memory on the item's external id, so an edited document stays one
/// always-current memory with its previous text on that memory's own version
/// history — no accumulating pile of near-identical versions competing in
/// retrieval. A split document cannot use that, because a version with five
/// parts followed by one with three would leave two orphaned part-memories
/// that upsert alone will not remove; its parts are appended instead, and the
/// previous version's ids move to `superseded_memory_ids`.
///
/// The rule is applied per *version*, not per item, so a document that grows
/// past the split threshold moves from upsert to append and back without any
/// reconciliation step. What it costs is stated plainly: after an edit, the old
/// text of a single-part document is a history read, not a recall.
#[allow(clippy::too_many_arguments)]
async fn write_item(
    state: &AppState,
    def: &SourceDefinition,
    run: &SourceRun,
    rel: &str,
    hash: &str,
    content_sha: &str,
    experience_type: &crate::memory::ExperienceType,
    cfg: &WatchedFolderConfig,
    parts: Vec<String>,
) -> Result<ItemWrite, AppError> {
    let total = parts.len();

    if total == 1 {
        let external_id = item_external_id(&def.id, hash);
        let content = parts.into_iter().next().unwrap_or_default();

        // Replay guard. A crash between the memory write and the cursor commit
        // leaves the memory in the store with no cursor; the next run re-reads
        // the same bytes. Without this check `upsert` would find the memory by
        // external id and push an identical version onto its history — turning
        // an at-least-once delivery into a visible version bump. The content
        // hash cannot catch it, because `upsert` does not consult that index.
        let existing = {
            let memory = state
                .get_user_memory(&def.user_id)
                .map_err(AppError::Internal)?;
            let eid = external_id.clone();
            tokio::task::spawn_blocking(move || {
                let guard = memory.read();
                guard.find_by_external_id(&eid)
            })
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Blocking task panicked: {e}")))?
            .map_err(AppError::Internal)?
        };
        if let Some(memory) = existing {
            if memory.experience.content == content {
                return Ok(ItemWrite {
                    memory_ids: vec![memory.id.0],
                    memories_written: 0,
                    all_deduped: true,
                    external_id: Some(external_id),
                });
            }
        }

        let mut req = base_request(def, run, rel, hash, content_sha, experience_type, cfg, content);
        req.external_id = Some(external_id.clone());
        req.change_reason = Some("source item content changed".to_string());
        let outcome = crate::ingest::ingest_experience(state, req).await?;

        return Ok(ItemWrite {
            memory_ids: vec![outcome.memory_id.0],
            memories_written: 1,
            all_deduped: false,
            external_id: Some(external_id),
        });
    }

    // Split document: the parts become an episode, which is a structure that
    // already ships and already carries ordering.
    let episode_id = item_external_id(&def.id, hash);
    let mut memory_ids: Vec<uuid::Uuid> = Vec::with_capacity(total);
    let mut written = 0u32;
    let mut all_deduped = true;
    let mut preceding: Option<String> = None;

    for (index, content) in parts.into_iter().enumerate() {
        let mut req = base_request(def, run, rel, hash, content_sha, experience_type, cfg, content);
        req.experience.metadata.insert(
            META_PART.to_string(),
            format!("{}/{}", index + 1, total),
        );
        req.experience.context = crate::handlers::remember::build_rich_context(
            None,
            None,
            None,
            None,
            None,
            Some(episode_id.clone()),
            Some(index as u32),
            preceding.clone(),
        );

        // A failure part-way through does NOT advance the cursor, so the next
        // run re-reads the file and re-derives the same parts; the parts
        // already written are absorbed by the content-hash index.
        let outcome = crate::ingest::ingest_experience(state, req).await?;
        if !outcome.deduped {
            all_deduped = false;
            written += 1;
        }
        preceding = Some(outcome.memory_id.0.to_string());
        memory_ids.push(outcome.memory_id.0);
    }

    Ok(ItemWrite {
        memory_ids,
        memories_written: written,
        all_deduped,
        external_id: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn base_request(
    def: &SourceDefinition,
    run: &SourceRun,
    rel: &str,
    hash: &str,
    content_sha: &str,
    experience_type: &crate::memory::ExperienceType,
    cfg: &WatchedFolderConfig,
    content: String,
) -> crate::ingest::IngestRequest {
    // The memory's content is the file text and nothing else. Prepending a path
    // header would make two copies of one document into two memories, defeat
    // the content-hash layer, and put a filesystem path into the embeddable,
    // retrievable text where it pollutes both NER and BM25. Path, source, run
    // and hash live in metadata.
    crate::ingest::IngestRequest::new(
        def.user_id.clone(),
        content,
        experience_type.clone(),
        MemoryOrigin::Connector,
    )
    .with_tags(cfg.tags.clone())
    .with_metadata(META_SOURCE_ID, def.id.0.to_string())
    .with_metadata(META_SOURCE_KIND, def.kind.as_str())
    .with_metadata(META_RUN_ID, run.run_id.to_string())
    .with_metadata(META_ITEM, rel)
    .with_metadata(META_ITEM_SHA256, content_sha)
    .with_metadata("shodh.source.item_id", hash)
}

/// Close the run out: final record, runtime counters, lease release, history
/// prune, and the human-timeline audit entry.
async fn finish(
    state: &AppState,
    def: &SourceDefinition,
    mut run: SourceRun,
    mut runtime: SourceRuntime,
) -> Result<SourceRun, AppError> {
    run.finished_at = Some(Utc::now());

    runtime.last_run_id = Some(run.run_id);
    runtime.last_run_started_at = Some(run.started_at);
    runtime.last_run_finished_at = run.finished_at;
    runtime.last_run_status = Some(run.status);
    runtime.run_count = runtime.run_count.saturating_add(1);
    runtime.memories_written_total = runtime
        .memories_written_total
        .saturating_add(u64::from(run.memories_written));
    if matches!(run.status, RunStatus::Failed) {
        runtime.consecutive_failures = runtime.consecutive_failures.saturating_add(1);
    } else {
        runtime.consecutive_failures = 0;
        runtime.last_success_at = run.finished_at;
    }

    state
        .source_store
        .finish_run(&run, &runtime)
        .map_err(AppError::Internal)?;

    // The audit entry is the human timeline, interleaved with every other
    // mutation and subject to rotation. The run record is the authority. Never
    // read the audit trail to answer "is my data current".
    let summary = format!(
        "source '{}' ({}) run {}: seen {}, ingested {}, unchanged {}, deduped {}, skipped {}, \
         failed {}, denied {}, memories {}",
        def.name,
        run.status.as_str(),
        run.run_id,
        run.items_seen,
        run.items_ingested,
        run.items_unchanged,
        run.items_deduped,
        run.items_skipped,
        run.items_failed,
        run.items_denied_by_policy,
        run.memories_written,
    );
    state.log_event(&def.user_id, "SOURCE_RUN", &def.id.0.to_string(), &summary);

    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_shapes_are_refused_by_name() {
        for name in [
            "id_rsa",
            "id_ed25519.pub",
            "server.pem",
            "client.key",
            "vault.kdbx",
            ".env",
            ".env.production",
            "credentials",
            "backup.PFX",
        ] {
            assert!(
                is_credential_shaped(name),
                "{name} must be refused before any read"
            );
        }
        for name in ["notes.md", "README.txt", "keyboard-shortcuts.md", "envoy.md"] {
            assert!(
                !is_credential_shaped(name),
                "{name} is ordinary text and must not be refused"
            );
        }
    }

    #[test]
    fn split_keeps_every_part_storable_and_loses_nothing() {
        let paragraph = "x".repeat(1_000);
        let doc = std::iter::repeat(paragraph.as_str())
            .take(120)
            .collect::<Vec<_>>()
            .join("\n\n");
        assert!(doc.len() > validation::MAX_CONTENT_LENGTH);

        let parts = split_content(&doc);
        assert!(parts.len() > 1, "a document over the limit must be split");
        for part in &parts {
            assert!(
                part.len() <= validation::MAX_CONTENT_LENGTH,
                "a part that exceeds MAX_CONTENT_LENGTH cannot be stored at all"
            );
            assert!(part.trim().len() >= validation::MIN_MEANINGFUL_CONTENT_LENGTH);
        }
        let rejoined: String = parts.join("");
        assert_eq!(
            rejoined.matches('x').count(),
            doc.matches('x').count(),
            "splitting must not drop content"
        );
    }

    #[test]
    fn split_leaves_a_short_document_alone() {
        let parts = split_content("  a short note about Baltimore  ");
        assert_eq!(parts, vec!["a short note about Baltimore".to_string()]);
        assert!(split_content("   \n  ").is_empty());
    }

    #[test]
    fn split_never_cuts_a_multibyte_character() {
        // No paragraph or line breaks anywhere, so every cut lands on the hard
        // character-boundary fallback.
        let doc = "é".repeat(60_000);
        let parts = split_content(&doc);
        assert!(parts.len() > 1);
        for part in &parts {
            assert!(part.len() <= validation::MAX_CONTENT_LENGTH);
            assert!(
                part.chars().all(|c| c == 'é'),
                "a cut inside a multi-byte character would produce a replacement char"
            );
        }
        assert_eq!(
            parts.iter().map(|p| p.chars().count()).sum::<usize>(),
            60_000
        );
    }
}
