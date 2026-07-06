//! `.env` reload support for the `/reload` slash command.
//!
//! Mirrors the upstream's `reload.env` RPC (`tui_gateway/server.py:4147-4165`)
//! which calls `the upstream's env-reload helper` to refresh
//! environment variables in the running process so credentials
//! changed on disk take effect on the next provider call without
//! a restart.
//!
//! Lives in its own module (rather than inline in `main.rs`) so
//! the parsing + override loop is testable as part of the lib
//! crate.

use anyhow::{Context, Result};
use std::path::Path;

/// Re-read `path` as a `.env` file, applying every key to the
/// current process via [`std::env::set_var`]. Returns the count
/// of vars applied. Missing file is *not* an error — returns 0,
/// matching the upstream's silent-no-op behavior when `its env file`
/// doesn't exist.
///
/// Already-set env vars are *overwritten* (the upstream's default; the
/// rationale is that the user explicitly asked to reload and
/// expects their on-disk value to win).
pub fn reload_env_file(path: &Path) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let mut count = 0usize;
    let iter =
        dotenvy::from_path_iter(path).with_context(|| format!("reading {}", path.display()))?;
    for entry in iter {
        let (k, v) = entry.with_context(|| format!("parsing {}", path.display()))?;
        // SAFETY: set_var is unsafe in newer Rust due to threading
        // concerns. /reload runs on the submit task while no other
        // thread mutates the env, and the user has explicitly asked
        // for the refresh — same contract the upstream uses with os.environ.
        unsafe {
            std::env::set_var(&k, &v);
        }
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn reload_missing_file_returns_zero_no_error() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("does-not-exist.env");
        let n = reload_env_file(&path).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn reload_sets_each_key_in_process_env() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(".env");
        std::fs::write(
            &path,
            "FENNEC_TEST_RELOAD_KEY_A=alpha\nFENNEC_TEST_RELOAD_KEY_B=beta\n",
        )
        .unwrap();
        let n = reload_env_file(&path).unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            std::env::var("FENNEC_TEST_RELOAD_KEY_A").unwrap(),
            "alpha"
        );
        assert_eq!(
            std::env::var("FENNEC_TEST_RELOAD_KEY_B").unwrap(),
            "beta"
        );
    }

    #[test]
    fn reload_overwrites_existing_value() {
        // SAFETY: see module-level safety note. This test runs
        // single-threaded and we're explicitly verifying overwrite
        // behavior.
        unsafe {
            std::env::set_var("FENNEC_TEST_RELOAD_OVERWRITE", "old");
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join(".env");
        std::fs::write(&path, "FENNEC_TEST_RELOAD_OVERWRITE=new\n").unwrap();
        reload_env_file(&path).unwrap();
        assert_eq!(
            std::env::var("FENNEC_TEST_RELOAD_OVERWRITE").unwrap(),
            "new"
        );
    }
}
