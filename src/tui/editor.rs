//! `$EDITOR` integration for long prompts.
//!
//! `/edit` (or `Ctrl-G`) suspends the TUI, hands the terminal to
//! the user's editor with a tempfile pre-filled with the current
//! composer contents, and pastes the saved result back into the
//! input buffer when the editor exits.
//!
//! Mirrors the upstream's `lib/editor.ts` + `useComposerState.ts:267-297`
//! flow: `$VISUAL > $EDITOR > nano/pico/vi/emacs > vi` priority,
//! shell-tokenised so `EDITOR="code --wait"` works, tempdir
//! cleaned unconditionally on Drop.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, anyhow};

/// Resolve which editor to launch + its argv tail.
///
/// Priority:
/// 1. `$VISUAL` (whitespace-tokenised — `code --wait` → `["code", "--wait"]`)
/// 2. `$EDITOR` (same tokenisation)
/// 3. First match on `$PATH` from the candidate list:
///    - Unix: `editor`, `nano`, `pico`, `vi`, `emacs`
///    - Windows: `notepad.exe`
/// 4. Fallback floor: `vi` (Unix) / `notepad.exe` (Windows)
///
/// Empty / whitespace-only env vars are treated as unset, matching
/// the upstream's `editor.ts:29-47`.
pub fn resolve_editor() -> Vec<String> {
    for env_var in ["VISUAL", "EDITOR"] {
        if let Ok(v) = std::env::var(env_var) {
            let parts = tokenise(&v);
            if !parts.is_empty() {
                return parts;
            }
        }
    }
    let (candidates, floor) = if cfg!(target_os = "windows") {
        (vec!["notepad.exe"], "notepad.exe".to_string())
    } else {
        (
            vec!["editor", "nano", "pico", "vi", "emacs"],
            "vi".to_string(),
        )
    };
    for cand in &candidates {
        if find_on_path(cand).is_some() {
            return vec![(*cand).to_string()];
        }
    }
    vec![floor]
}

/// Whitespace tokeniser for `$EDITOR`-style values. the upstream uses
/// the same simple split — no shell-quoting handling, so something
/// like `EDITOR="code 'with space'"` doesn't survive correctly,
/// matching upstream.
fn tokenise(s: &str) -> Vec<String> {
    s.trim()
        .split_whitespace()
        .map(|p| p.to_string())
        .collect()
}

/// Walk `$PATH` looking for an executable named `name`. Returns
/// the first match. We don't use the `which` crate here —
/// platform-specific extension handling we need (notepad.exe vs
/// notepad) is handled by the caller picking the right candidate
/// list per OS.
fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Open `$EDITOR` with `initial` pre-loaded into a tempfile,
/// blocking the calling thread until the editor exits, and
/// return the saved contents.
///
/// Returns:
/// - `Ok(Some(text))` — editor exited with status 0 AND the file
///   was non-empty after a `trim_end`. Submit-ready text.
/// - `Ok(None)` — editor exited non-zero (`vim :cq`), or the file
///   came back empty / whitespace-only. Composer state is left
///   untouched.
/// - `Err(_)` — editor binary not found, tempdir creation failed,
///   or the spawn errored. Surfaces to the user as a system message.
///
/// Caller is responsible for suspending the TUI's terminal before
/// calling this and restoring after — see
/// `crate::tui::run_editor_request` in the event loop.
pub fn open_editor_for_input(initial: &str) -> Result<Option<String>> {
    let dir = tempfile::tempdir().context("creating tempdir for editor")?;
    let file = dir.path().join("prompt.md");
    std::fs::write(&file, initial).context("seeding tempfile with composer text")?;

    let argv = resolve_editor();
    if argv.is_empty() {
        return Err(anyhow!("no editor resolved (VISUAL/EDITOR unset, no fallback found)"));
    }
    let mut cmd = Command::new(&argv[0]);
    for arg in &argv[1..] {
        cmd.arg(arg);
    }
    cmd.arg(&file);

    let status = cmd
        .status()
        .with_context(|| format!("spawning editor: {}", argv.join(" ")))?;

    if !status.success() {
        // Editor aborted (e.g. `:cq`). Drop the result.
        return Ok(None);
    }

    let body = std::fs::read_to_string(&file)
        .with_context(|| format!("reading {}", file.display()))?;
    let trimmed = body.trim_end_matches('\n').to_string();
    if trimmed.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Save / restore env vars across a test so parallel runs
    /// don't leak state into each other. Uses unsafe set_var/
    /// remove_var (Rust 2024 marks them unsafe due to thread
    /// concerns); this test serialises state changes via a
    /// process-wide Mutex.
    fn with_env<F: FnOnce()>(env: &[(&str, Option<&str>)], f: F) {
        use std::sync::Mutex;
        static GUARD: Mutex<()> = Mutex::new(());
        let _g = GUARD.lock().unwrap();
        let prior: Vec<(String, Option<String>)> = env
            .iter()
            .map(|(k, _)| (k.to_string(), std::env::var(k).ok()))
            .collect();
        for (k, v) in env {
            // SAFETY: serialised via GUARD; whole test surface
            // doesn't read $VISUAL/$EDITOR concurrently.
            unsafe {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
        f();
        for (k, v) in prior {
            unsafe {
                match v {
                    Some(val) => std::env::set_var(&k, val),
                    None => std::env::remove_var(&k),
                }
            }
        }
    }

    #[test]
    fn resolve_prefers_visual_over_editor() {
        with_env(
            &[("VISUAL", Some("vim")), ("EDITOR", Some("nano"))],
            || {
                let argv = resolve_editor();
                assert_eq!(argv, vec!["vim".to_string()]);
            },
        );
    }

    #[test]
    fn resolve_falls_back_to_editor_when_visual_unset() {
        with_env(&[("VISUAL", None), ("EDITOR", Some("nano"))], || {
            let argv = resolve_editor();
            assert_eq!(argv, vec!["nano".to_string()]);
        });
    }

    #[test]
    fn resolve_tokenises_editor_with_args() {
        with_env(
            &[("VISUAL", None), ("EDITOR", Some("code --wait"))],
            || {
                let argv = resolve_editor();
                assert_eq!(argv, vec!["code".to_string(), "--wait".to_string()]);
            },
        );
    }

    #[test]
    fn resolve_treats_whitespace_only_as_unset() {
        with_env(
            &[("VISUAL", Some("   ")), ("EDITOR", Some("nano"))],
            || {
                let argv = resolve_editor();
                assert_eq!(argv, vec!["nano".to_string()]);
            },
        );
    }

    #[test]
    fn resolve_fallback_when_neither_set() {
        with_env(&[("VISUAL", None), ("EDITOR", None)], || {
            // Result depends on what's installed; just assert
            // we got a non-empty argv with one entry.
            let argv = resolve_editor();
            assert!(!argv.is_empty());
            assert!(!argv[0].is_empty());
        });
    }

    /// Smoke-test the open_editor flow using `cat` as the
    /// "editor" — it just dumps the file to stdout and exits.
    /// Verifies the tempfile lifecycle + read-back, without
    /// requiring an interactive editor. Wrapped in `with_env`
    /// so it serialises against the resolve_editor tests; two
    /// tests racing on `$EDITOR` would otherwise pick up each
    /// other's value mid-run.
    #[test]
    fn open_editor_with_cat_returns_initial_unchanged() {
        with_env(&[("VISUAL", None), ("EDITOR", Some("cat"))], || {
            let r = open_editor_for_input("hello world");
            match r {
                Ok(Some(text)) => assert_eq!(text, "hello world"),
                Ok(None) => panic!("expected Some, got None"),
                Err(e) => panic!("expected Ok, got Err: {e}"),
            }
        });
    }

    #[test]
    fn open_editor_with_false_returns_none() {
        // `false` exits with status 1 → editor cancelled.
        with_env(&[("VISUAL", None), ("EDITOR", Some("false"))], || {
            let r = open_editor_for_input("anything");
            match r {
                Ok(None) => {} // expected
                other => panic!("expected Ok(None), got {other:?}"),
            }
        });
    }
}
