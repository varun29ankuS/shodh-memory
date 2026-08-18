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
pub(crate) static ENV_LOCK: parking_lot::ReentrantMutex<()> = parking_lot::const_reentrant_mutex(());

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
    /// Take the lock without touching anything yet.
    pub(crate) fn acquire() -> Self {
        Self {
            _lock: ENV_LOCK.lock(),
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

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "SHODH_TEST_SUPPORT_SCOPED_ENV";

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
