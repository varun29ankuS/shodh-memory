//! Shared test-only scaffolding for process-global state.
//!
//! Rust runs a crate's unit tests as threads inside ONE process. Anything a
//! test mutates that is process-wide — environment variables above all — is
//! mutated for every other test running at that moment and for every test that
//! runs afterwards. Two failure modes follow, and this module addresses one of
//! them:
//!
//! * **Leak.** A test sets a variable and never puts it back, so every later
//!   test runs under a configuration nobody chose. This is fixable, and
//!   [`ScopedEnv`] fixes it: record the prior value, restore it on drop.
//! * **Exposure.** While a test holds a variable at a non-default value, an
//!   unrelated test on another thread reads it. This is NOT fixable with a
//!   lock — a mutex serialises writers, and the racing party here is a reader
//!   that takes nothing. The only real fix is to stop having the configuration
//!   be process-global; see `Query::read_only`, which is what the recall
//!   harness now uses instead of `SHODH_RECALL_READONLY`.
//!
//! So: reach for [`ScopedEnv`] when a variable genuinely has to stay global
//! (an env var read by production code that takes no config parameter), and
//! treat it as containment, not as isolation. When the value can be injected
//! instead, inject it.

use std::ffi::OsString;

/// Process-global lock for tests that mutate the server/auth configuration
/// environment: `SHODH_ENV`, `SHODH_API_KEYS`, `SHODH_DEV_API_KEY`,
/// `SHODH_PORT`, `SHODH_MAX_USERS`, `SHODH_WRITE_MODE`.
///
/// ONE lock for that whole family, deliberately: module-local mutexes would not
/// exclude each other, so an `auth` test clearing `SHODH_API_KEYS` could still
/// race a `local_ipc` test setting it.
///
/// The recall-determinism family (`SHODH_RECALL_READONLY`, `SHODH_EVAL_NOW`,
/// …) has its own lock, `crate::memory::RECALL_ENV_LOCK`. The two variable sets
/// are disjoint and no test touches both, so a single lock would only serialise
/// fast auth tests behind multi-minute harness runs for no correctness gain.
///
/// Reentrant so a helper that acquires a [`ScopedEnv`] can be called from a test
/// that already holds one.
pub(crate) static ENV_LOCK: parking_lot::ReentrantMutex<()> =
    parking_lot::const_reentrant_mutex(());

/// Holds [`ENV_LOCK`] and restores every variable it touched when dropped.
///
/// Restoring rather than clearing matters more than it looks. The handler test
/// harness sets `SHODH_API_KEYS` once, for the whole process, because every
/// handler test needs an accepted key. An auth test that ends by *removing*
/// that variable — which is what `clear_auth_env` used to do — leaves the key
/// unset for every handler test that starts afterwards, which is a 401/503 with
/// no relationship to the code under test. That exact failure was diagnosed in
/// CI once already (see the note on `local_ipc`'s former `ScopedEnvVar`).
pub(crate) struct ScopedEnv {
    _lock: parking_lot::ReentrantMutexGuard<'static, ()>,
    /// Original values, in the order the keys were first touched. Only the
    /// FIRST value seen for a key is recorded, so repeated `set` calls within
    /// one test still restore the value the test started with.
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl ScopedEnv {
    /// Take the server/auth config lock ([`ENV_LOCK`]) without touching
    /// anything yet.
    pub(crate) fn acquire() -> Self {
        Self::with_lock(&ENV_LOCK)
    }

    /// Take the recall-determinism lock (`crate::memory::RECALL_ENV_LOCK`)
    /// instead, for the retrieval feature flags the harness and its tests flip:
    /// `SHODH_FUSION_V2`, `SHODH_TYPED_WALK`, `SHODH_FUSION_FEATURE_EXPORT`.
    ///
    /// Same lock the harness pin takes, so a test that flips one of these and
    /// then runs a suite is serialised against every other recall-env test. The
    /// lock is reentrant, so the nesting composes.
    pub(crate) fn acquire_recall() -> Self {
        Self::with_lock(&crate::memory::RECALL_ENV_LOCK)
    }

    fn with_lock(lock: &'static parking_lot::ReentrantMutex<()>) -> Self {
        Self {
            _lock: lock.lock(),
            saved: Vec::new(),
        }
    }

    fn remember(&mut self, key: &'static str) {
        if !self.saved.iter().any(|(k, _)| *k == key) {
            self.saved.push((key, std::env::var_os(key)));
        }
    }

    /// Set `key` for the lifetime of this guard.
    pub(crate) fn set(&mut self, key: &'static str, value: &str) {
        self.remember(key);
        std::env::set_var(key, value);
    }

    /// Unset `key` for the lifetime of this guard.
    pub(crate) fn remove(&mut self, key: &'static str) {
        self.remember(key);
        std::env::remove_var(key);
    }

    /// Set `key` for the rest of the process, holding [`ENV_LOCK`] only for the
    /// write itself.
    ///
    /// For the narrow case where a value must outlive any single test — the
    /// handler harness's API key, which every handler test needs present while
    /// its requests are in flight on other threads. Use [`ScopedEnv::acquire`]
    /// for everything else; a permanent write is a leak by construction and is
    /// only defensible when the alternative is a guard whose drop breaks
    /// sibling tests.
    ///
    /// Do NOT `mem::forget` a `ScopedEnv` to get this: the guard owns a
    /// reentrant lock, and leaking it leaves the lock held forever by whichever
    /// thread ran first, deadlocking every other test that needs the env.
    pub(crate) fn set_for_process(key: &'static str, value: &str) {
        let _lock = ENV_LOCK.lock();
        std::env::set_var(key, value);
    }
}

impl Drop for ScopedEnv {
    fn drop(&mut self) {
        for (key, prior) in self.saved.drain(..).rev() {
            match prior {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Every file in `src/` that mutates a process environment variable, with the
/// number of live mutation calls it contains and why they are allowed to be
/// there. `env_mutation_sites_are_accounted_for` fails if reality drifts from
/// this table.
///
/// The point is not the count. The point is that adding a mutation site becomes
/// a decision someone has to write down, because the alternative — noticing it
/// later — has already failed once at scale: a single unrestored
/// `SHODH_RECALL_READONLY=1` in the recall harness silently disabled
/// reinforcement for every other test in the process, for the entire history of
/// this suite, and no test failed.
const ENV_MUTATION_INVENTORY: &[(&str, usize, &str)] = &[
    (
        "src/bin/recall_eval.rs",
        2,
        "process startup, before any thread or ONNX/rayon pool exists; the          whole process IS the harness, so there is nobody else to affect",
    ),
    (
        "src/embeddings/downloader.rs",
        1,
        "ORT_DYLIB_PATH, written once while resolving the ONNX runtime before          a session exists; ort reads it at session construction",
    ),
    (
        "src/embeddings/minilm.rs",
        3,
        "ORT_DYLIB_PATH behind a OnceLock, same reason; read-once at model load",
    ),
    (
        "src/memory/mod.rs",
        3,
        "RecallEnvPin: one set and its two restore arms, all under          RECALL_ENV_LOCK, test-only",
    ),
    (
        "src/recall_harness/runner.rs",
        5,
        "HarnessEnvPin's set and its two restore arms, plus analyze_ablation's          per-arm config which runs only in the recall_eval binary; all under          RECALL_ENV_LOCK",
    ),
    (
        "src/integrations/mod.rs",
        6,
        "https_default_tests: one set + one remove for SHODH_ENFORCE_HTTPS and          for the URL override, plus both restore arms. Test-only, and taken          under RECALL_ENV_LOCK rather than a module-local mutex, because a          private lock excludes only this module's tests",
    ),
    (
        "src/memory/ablation.rs",
        3,
        "ablation tests: one set + one remove for SHODH_DISABLE_BOOSTS and its          restore arm. Under RECALL_ENV_LOCK because this flag changes what the          RECALL PATH scores -- a module-local lock would let it flip mid-query          in a sibling module's test",
    ),
    (
        "src/server.rs",
        8,
        "server bootstrap, documented as running before the tokio runtime          spawns any thread",
    ),
    (
        "src/test_support.rs",
        5,
        "ScopedEnv itself: set, remove, set_for_process, and the two restore          arms",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "SHODH_TEST_SUPPORT_SCOPED_ENV";

    /// Count live environment-mutation calls in one file's source text.
    ///
    /// Lines that are entirely a comment do not count — the codebase discusses
    /// these functions in prose a lot, and a doc paragraph is not a mutation.
    /// The needles are assembled from fragments so this file's own source does
    /// not match the patterns it searches for.
    fn count_env_mutations(text: &str) -> usize {
        let needles = [
            concat!("env::", "set_var("),
            concat!("env::", "remove_var("),
        ];
        text.lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .map(|l| needles.iter().map(|n| l.matches(n).count()).sum::<usize>())
            .sum()
    }

    /// The scanner behind `env_mutation_sites_are_accounted_for` must actually
    /// see a new mutation site, and must not be fooled by prose.
    ///
    /// Without this, the inventory test could pass because the scanner returns
    /// zero for everything and the inventory happened to be empty — the same
    /// class of failure as a mutation harness reporting survivors because
    /// nothing ran at all.
    ///
    /// The fixtures are assembled from fragments for the same reason the
    /// needles are: a literal call written here would be counted as a real site
    /// in this file. That is not hypothetical — the first draft of this test
    /// spelled them out and the inventory test failed on it, which is the
    /// clearest evidence available that the inventory notices new sites.
    #[test]
    fn the_env_mutation_scanner_counts_calls_and_ignores_comments() {
        let set = concat!("env::", "set_var(");
        let remove = concat!("env::", "remove_var(");

        let prose = format!(
            "/// see std::{set}) for why this is unsound
// std::{remove})
"
        );
        assert_eq!(
            count_env_mutations(&prose),
            0,
            "prose about the functions must not count as a call site"
        );

        let real = format!(
            "fn f() {{
    std::{set}\"A\", \"1\");
    std::{remove}\"A\");
}}
"
        );
        assert_eq!(
            count_env_mutations(&real),
            2,
            "both a set and a remove must be counted"
        );

        assert_eq!(
            count_env_mutations(&format!("{prose}{real}")),
            2,
            "prose plus two calls is two — this is what makes an unaccounted              site fail the inventory"
        );
    }

    /// Walk `src/`, counting live mutation calls per file, and compare against
    /// [`ENV_MUTATION_INVENTORY`].
    #[test]
    fn env_mutation_sites_are_accounted_for() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut found: std::collections::BTreeMap<String, usize> = Default::default();
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("reading src/") {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("reading a source file");
                let hits = count_env_mutations(&text);
                if hits > 0 {
                    let rel = path
                        .strip_prefix(root.parent().expect("src has a parent"))
                        .expect("path under the crate root")
                        .to_string_lossy()
                        .replace(std::path::MAIN_SEPARATOR, "/");
                    found.insert(rel, hits);
                }
            }
        }

        let expected: std::collections::BTreeMap<String, usize> = ENV_MUTATION_INVENTORY
            .iter()
            .map(|(f, n, _)| ((*f).to_string(), *n))
            .collect();

        assert_eq!(
            found,
            expected,
            "

The set of process-environment mutation sites in src/ changed.

             Found:    {found:#?}
             Expected: {expected:#?}

             A new site is not automatically wrong, but it is never free: Rust              runs this crate's tests as threads in ONE process, so a variable              set by one test is read by every other test running at that moment              and by every test that runs afterwards. Before updating              ENV_MUTATION_INVENTORY, answer three questions.

             1. Can the value be injected instead? A field on the request, an              argument, a config struct. An injected value cannot be observed by              an unrelated thread; that is a fix, not a mitigation.
             2. If it must stay global, is it written under one of the two              crate-wide env locks and restored on drop (ScopedEnv /              RecallEnvPin / HarnessEnvPin)?
             3. Does the variable change what OTHER code does while it is set?              If yes, a lock does not save you — a mutex serialises writers, and              the party you are racing is a reader that takes nothing. Say so in              the inventory rationale rather than implying the lock covers it.
"
        );
    }

    #[test]
    fn scoped_env_restores_a_previously_unset_variable() {
        {
            let mut env = ScopedEnv::acquire();
            env.set(KEY, "set-by-test");
            assert_eq!(std::env::var(KEY).as_deref(), Ok("set-by-test"));
        }
        assert!(
            std::env::var_os(KEY).is_none(),
            "a variable that was unset before the guard must be unset after it"
        );
    }

    #[test]
    fn scoped_env_restores_the_value_the_test_started_with_not_the_last_one_set() {
        let outer_key = "SHODH_TEST_SUPPORT_SCOPED_ENV_OUTER";
        let mut outer = ScopedEnv::acquire();
        outer.set(outer_key, "original");

        {
            let mut env = ScopedEnv::acquire();
            env.set(outer_key, "first");
            env.set(outer_key, "second");
            env.remove(outer_key);
            assert!(std::env::var_os(outer_key).is_none());
        }

        assert_eq!(
            std::env::var(outer_key).as_deref(),
            Ok("original"),
            "restoring must return the value present when the guard was taken, \
             not the most recent value it wrote"
        );
    }

    #[test]
    fn scoped_env_is_reentrant() {
        // A plain mutex would deadlock here. Helpers that need the env lock get
        // called from tests that already hold it; that must compose.
        let _outer = ScopedEnv::acquire();
        let mut inner = ScopedEnv::acquire();
        inner.set(KEY, "nested");
        assert_eq!(std::env::var(KEY).as_deref(), Ok("nested"));
    }
}
