//! Best-effort process and cgroup memory diagnostics.
//!
//! These counters are intentionally observational. They make memory growth
//! visible through health/metrics endpoints without changing readiness status.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SystemMemoryDiagnostics {
    pub process_rss_bytes: Option<u64>,
    pub process_peak_rss_bytes: Option<u64>,
    pub process_virtual_bytes: Option<u64>,
    pub cgroup_memory_current_bytes: Option<u64>,
    pub cgroup_memory_peak_bytes: Option<u64>,
}

pub fn read_system_memory_diagnostics() -> SystemMemoryDiagnostics {
    SystemMemoryDiagnostics {
        process_rss_bytes: read_proc_status_bytes("VmRSS"),
        process_peak_rss_bytes: read_proc_status_bytes("VmHWM"),
        process_virtual_bytes: read_proc_status_bytes("VmSize"),
        cgroup_memory_current_bytes: read_cgroup_memory_current_bytes(),
        cgroup_memory_peak_bytes: read_cgroup_memory_peak_bytes(),
    }
}

#[cfg(target_os = "linux")]
fn read_proc_status_bytes(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_proc_status_bytes(&status, key)
}

#[cfg(not(target_os = "linux"))]
fn read_proc_status_bytes(_key: &str) -> Option<u64> {
    None
}

fn parse_proc_status_bytes(status: &str, key: &str) -> Option<u64> {
    for line in status.lines() {
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let rest = rest.trim_start().strip_prefix(':')?.trim_start();
        let mut parts = rest.split_whitespace();
        let value = parts.next()?.parse::<u64>().ok()?;
        let unit = parts.next().unwrap_or("B");
        return match unit {
            "kB" => value.checked_mul(1024),
            "mB" | "MB" => value.checked_mul(1024 * 1024),
            "gB" | "GB" => value.checked_mul(1024 * 1024 * 1024),
            "B" => Some(value),
            _ => None,
        };
    }
    None
}

#[cfg(target_os = "linux")]
fn read_cgroup_memory_current_bytes() -> Option<u64> {
    read_cgroup_memory_metric("memory.current", "memory.usage_in_bytes")
}

#[cfg(not(target_os = "linux"))]
fn read_cgroup_memory_current_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn read_cgroup_memory_peak_bytes() -> Option<u64> {
    read_cgroup_memory_metric("memory.peak", "memory.max_usage_in_bytes")
}

#[cfg(not(target_os = "linux"))]
fn read_cgroup_memory_peak_bytes() -> Option<u64> {
    None
}

#[cfg(target_os = "linux")]
fn read_cgroup_memory_metric(v2_file: &str, v1_file: &str) -> Option<u64> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in cgroup.lines() {
        let mut parts = line.splitn(3, ':');
        let _hierarchy = parts.next()?;
        let controllers = parts.next()?;
        let relative_path = parts.next()?;

        if controllers.is_empty() {
            if let Some(value) =
                read_u64_file(cgroup_file_path("/sys/fs/cgroup", relative_path, v2_file))
            {
                return Some(value);
            }
        } else if controllers
            .split(',')
            .any(|controller| controller == "memory")
        {
            if let Some(value) = read_u64_file(cgroup_file_path(
                "/sys/fs/cgroup/memory",
                relative_path,
                v1_file,
            )) {
                return Some(value);
            }
            if let Some(value) =
                read_u64_file(cgroup_file_path("/sys/fs/cgroup", relative_path, v1_file))
            {
                return Some(value);
            }
        }
    }

    read_u64_file(Path::new("/sys/fs/cgroup").join(v2_file))
        .or_else(|| read_u64_file(Path::new("/sys/fs/cgroup/memory").join(v1_file)))
}

fn cgroup_file_path(root: impl AsRef<Path>, relative_path: &str, file_name: &str) -> PathBuf {
    let relative_path = relative_path.trim_start_matches('/');
    if relative_path.is_empty() {
        root.as_ref().join(file_name)
    } else {
        root.as_ref().join(relative_path).join(file_name)
    }
}

fn read_u64_file(path: impl AsRef<Path>) -> Option<u64> {
    let raw = std::fs::read_to_string(path).ok()?;
    parse_u64_file_value(&raw)
}

fn parse_u64_file_value(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed == "max" {
        None
    } else {
        trimmed.parse::<u64>().ok()
    }
}

/// RocksDB in-process memory, decomposed — the instrument for #90.
///
/// Process RSS growth (see [`SystemMemoryDiagnostics`]) tells you THAT memory
/// grows; this tells you WHERE inside RocksDB it sits:
/// - `shared_block_cache_*`: the single LRU pool shared by every DB instance
///   (per-user memory DBs, per-user graph DBs, the global shared DB). Bounded
///   by capacity — growth here plateaus by construction.
/// - `user_memtables_bytes` / `user_table_readers_bytes`: per-column-family
///   write buffers and index/filter readers, summed across CACHED users only
///   (never loads cold users — the #362 lesson). These are the unbounded-ish
///   suspects: they scale with open CFs per cached user.
///
/// The decisive read: if RSS climbs while all of these plateau, the growth is
/// OUTSIDE RocksDB (allocator retention, our own caches); if memtables/readers
/// climb with RSS, it's RocksDB tuning.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RocksDbMemoryDiagnostics {
    /// Bytes currently used in the shared LRU block cache.
    pub shared_block_cache_usage_bytes: u64,
    /// Bytes pinned in the shared cache (in active use, can't be evicted).
    pub shared_block_cache_pinned_bytes: u64,
    /// Configured capacity of the shared cache (the hard ceiling).
    pub shared_block_cache_capacity_bytes: u64,
    /// Sum of active+immutable memtable bytes across all CFs of all CACHED
    /// users' DBs (memory storage + graph).
    pub user_memtables_bytes: u64,
    /// Sum of estimated table-reader (index/filter) bytes outside the block
    /// cache, same scope as `user_memtables_bytes`.
    pub user_table_readers_bytes: u64,
    /// How many cached users the per-user sums cover (cold users on disk are
    /// deliberately not opened by this diagnostic).
    pub users_counted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_proc_status_memory_values_as_bytes() {
        let status = "\
Name:\tshodh-memory\n\
VmSize:\t  4096 kB\n\
VmHWM:\t     7 kB\n\
VmRSS:\t  1234 kB\n";

        assert_eq!(parse_proc_status_bytes(status, "VmRSS"), Some(1234 * 1024));
        assert_eq!(parse_proc_status_bytes(status, "VmHWM"), Some(7 * 1024));
        assert_eq!(parse_proc_status_bytes(status, "VmSize"), Some(4096 * 1024));
        assert_eq!(parse_proc_status_bytes(status, "VmSwap"), None);
    }

    #[test]
    fn parse_u64_file_value_handles_cgroup_limits() {
        assert_eq!(parse_u64_file_value("12345\n"), Some(12345));
        assert_eq!(parse_u64_file_value("max\n"), None);
        assert_eq!(parse_u64_file_value("not-a-number\n"), None);
    }

    #[test]
    fn cgroup_file_path_handles_root_and_nested_paths() {
        assert_eq!(
            cgroup_file_path("/sys/fs/cgroup", "/", "memory.current"),
            PathBuf::from("/sys/fs/cgroup/memory.current")
        );
        assert_eq!(
            cgroup_file_path(
                "/sys/fs/cgroup",
                "/user.slice/user-1000.slice/app.slice",
                "memory.current"
            ),
            PathBuf::from("/sys/fs/cgroup/user.slice/user-1000.slice/app.slice/memory.current")
        );
    }
}
