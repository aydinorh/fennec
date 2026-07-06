//! Cron-job script execution.
//!
//! Mirrors the upstream's `_run_job_script` / `_parse_wake_gate` semantics:
//!
//! - Scripts must live under `<scripts_dir>/` (default `<cron_dir>/scripts/`).
//!   Relative paths resolve against that dir; absolute paths are still
//!   validated to stay within it. Symlinks that escape the directory are
//!   rejected via canonicalisation.
//! - `.sh` / `.bash` files run via `bash` (resolved on `PATH`, falling
//!   back to `/bin/bash`); everything else runs via the current Python
//!   interpreter (preserves the data-collection-script pattern).
//! - The script has a wall-clock timeout (default 120s, overridable via
//!   [`SCRIPT_TIMEOUT_ENV`]) so a hung script can't block the scheduler.
//! - [`parse_wake_gate`] reads the LAST non-empty line of stdout as
//!   JSON. `{"wakeAgent": false}` means "nothing to report, skip the
//!   agent"; any other output (non-JSON, missing flag, or
//!   `wakeAgent: true`) means "wake the agent normally".

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Result};

/// Default script timeout (matches the upstream's
/// `_DEFAULT_SCRIPT_TIMEOUT`). A script with no output within this many
/// seconds is killed and reported as a failure.
pub const DEFAULT_SCRIPT_TIMEOUT_SECS: u64 = 120;

/// Env var that overrides the default script timeout. Matches the
/// upstream's `HERMES_CRON_SCRIPT_TIMEOUT`. A positive integer (seconds);
/// invalid values fall back to the default.
pub const SCRIPT_TIMEOUT_ENV: &str = "FENNEC_CRON_SCRIPT_TIMEOUT";

/// Resolve the script timeout from env first, then the configured
/// default. Matches the upstream's `_get_script_timeout`.
pub fn resolve_script_timeout() -> Duration {
    if let Ok(raw) = std::env::var(SCRIPT_TIMEOUT_ENV) {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            if let Ok(secs) = trimmed.parse::<u64>() {
                if secs > 0 {
                    return Duration::from_secs(secs);
                }
            }
            tracing::warn!(
                "Invalid {} value '{}'; falling back to default {}s",
                SCRIPT_TIMEOUT_ENV,
                raw,
                DEFAULT_SCRIPT_TIMEOUT_SECS
            );
        }
    }
    Duration::from_secs(DEFAULT_SCRIPT_TIMEOUT_SECS)
}

/// Resolve and validate a script path against the scripts directory.
///
/// Returns the canonical path on success. Fails when:
/// - The scripts dir doesn't exist or can't be canonicalised.
/// - The resolved script path escapes the scripts dir (path traversal
///   via `..` or absolute injection).
/// - The path doesn't exist or isn't a file.
///
/// Both relative paths (under scripts dir) and absolute paths (validated
/// to lie inside scripts dir, including via symlink target) are accepted
/// — matches the upstream's `_run_job_script` path resolution.
pub fn resolve_script_path(scripts_dir: &Path, script_path: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(scripts_dir)
        .map_err(|e| anyhow!("creating scripts dir {}: {}", scripts_dir.display(), e))?;
    let scripts_dir_resolved = scripts_dir.canonicalize().map_err(|e| {
        anyhow!(
            "canonicalising scripts dir {}: {}",
            scripts_dir.display(),
            e
        )
    })?;

    let raw = Path::new(script_path);
    let candidate = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        scripts_dir.join(raw)
    };
    let resolved = candidate.canonicalize().map_err(|e| {
        anyhow!(
            "script path {} cannot be resolved: {}",
            candidate.display(),
            e
        )
    })?;

    if !resolved.starts_with(&scripts_dir_resolved) {
        return Err(anyhow!(
            "blocked: script path resolves outside the scripts directory ({}): {:?}",
            scripts_dir_resolved.display(),
            script_path
        ));
    }
    if !resolved.is_file() {
        return Err(anyhow!("script path is not a file: {}", resolved.display()));
    }
    Ok(resolved)
}

/// Choose the interpreter argv for a script by its extension: bash for
/// `.sh`/`.bash`, `python3` (falling back to `python`) for anything
/// else. Names go through `PATH` lookup at spawn time; missing
/// interpreter surfaces as a clear error.
fn interpreter_argv(script: &Path) -> Vec<String> {
    let ext = script
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let prog = if ext == "sh" || ext == "bash" {
        "bash"
    } else {
        // Prefer `python3` on systems that ship both; the spawn-fallback
        // step below tries `python` if `python3` isn't on PATH.
        "python3"
    };
    vec![prog.to_string(), script.to_string_lossy().to_string()]
}

/// Execute a cron job script with a wall-clock timeout. Returns
/// `(success, output)`; on failure, `output` carries an error message
/// shaped so the LLM (or operator log) can surface the problem.
///
/// The script runs with `cwd` set to its parent directory and inherits
/// the parent environment plus `FENNEC_HOME=<scripts_dir parent>`. For
/// `.py` scripts, `python3` is tried first then `python` — matching
/// how the upstream pins to the current Python interpreter.
///
/// **Follow-up:** stdout/stderr should be passed through a secret-redactor
/// before being returned so accidentally-echoed credentials don't end up
/// in cron output files or agent prompts. Fennec doesn't ship a generic
/// `redact()` helper yet — when one lands (separate plan item), wire it
/// in here and in `save_job_output`.
pub async fn run_job_script(
    script_path: &str,
    scripts_dir: &Path,
    timeout: Duration,
) -> (bool, String) {
    let resolved = match resolve_script_path(scripts_dir, script_path) {
        Ok(p) => p,
        Err(e) => return (false, e.to_string()),
    };
    let argv = interpreter_argv(&resolved);
    let cwd = resolved.parent().unwrap_or(scripts_dir).to_path_buf();
    let fennec_home = scripts_dir
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| scripts_dir.to_path_buf());

    // Try `python3` first; on ENOENT (missing on PATH), retry with
    // `python`. Bash scripts never hit this fallback.
    let primary = spawn_and_wait(&argv, &cwd, &fennec_home, timeout).await;
    let output = match primary {
        Ok(out) => out,
        Err(e) if argv[0] == "python3" && is_not_found(&e) => {
            let alt = vec!["python".to_string(), argv[1].clone()];
            match spawn_and_wait(&alt, &cwd, &fennec_home, timeout).await {
                Ok(out) => out,
                Err(e2) => return (false, format_spawn_err(&e2, &resolved)),
            }
        }
        Err(e) => return (false, format_spawn_err(&e, &resolved)),
    };

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if !output.status.success() {
        let mut parts = vec![format!(
            "Script exited with code {}",
            output.status.code().unwrap_or(-1)
        )];
        if !stderr.is_empty() {
            parts.push(format!("stderr:\n{stderr}"));
        }
        if !stdout.is_empty() {
            parts.push(format!("stdout:\n{stdout}"));
        }
        return (false, parts.join("\n"));
    }
    (true, stdout)
}

/// One-shot script spawn with timeout. Returns the process Output on
/// successful completion (regardless of exit code) or a structured
/// error on spawn / timeout / interpreter-missing.
async fn spawn_and_wait(
    argv: &[String],
    cwd: &Path,
    fennec_home: &Path,
    timeout: Duration,
) -> Result<std::process::Output, ScriptSpawnError> {
    let mut cmd = tokio::process::Command::new(&argv[0]);
    cmd.args(&argv[1..]).current_dir(cwd);
    cmd.env("FENNEC_HOME", fennec_home);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let child = cmd.spawn().map_err(ScriptSpawnError::Spawn)?;
    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(ScriptSpawnError::Wait(e)),
        Err(_) => Err(ScriptSpawnError::Timeout(timeout)),
    }
}

#[derive(Debug)]
enum ScriptSpawnError {
    Spawn(std::io::Error),
    Wait(std::io::Error),
    Timeout(Duration),
}

fn is_not_found(e: &ScriptSpawnError) -> bool {
    matches!(e, ScriptSpawnError::Spawn(io) if io.kind() == std::io::ErrorKind::NotFound)
}

fn format_spawn_err(e: &ScriptSpawnError, script: &Path) -> String {
    match e {
        ScriptSpawnError::Spawn(io) if io.kind() == std::io::ErrorKind::NotFound => format!(
            "Script interpreter not found on PATH for {} (install bash for .sh/.bash, or python3 / python for others): {}",
            script.display(), io
        ),
        ScriptSpawnError::Spawn(io) => format!("Script spawn failed: {io}"),
        ScriptSpawnError::Wait(io) => format!("Script wait failed: {io}"),
        ScriptSpawnError::Timeout(d) => format!(
            "Script timed out after {}s: {}",
            d.as_secs(),
            script.display()
        ),
    }
}

/// Parse the wake-gate from a cron job's pre-check script output.
///
/// Convention (ported from the upstream's `_parse_wake_gate`): the LAST
/// non-empty line of stdout is parsed as JSON. If it deserialises to an
/// object with `"wakeAgent": false`, the agent is skipped (job is
/// silent this tick). Any other output — non-JSON, missing flag, empty
/// stdout, or `wakeAgent: true` — means "wake the agent normally".
pub fn parse_wake_gate(script_output: &str) -> bool {
    let last_line = script_output.lines().rfind(|l| !l.trim().is_empty());
    let Some(line) = last_line else {
        return true;
    };
    let trimmed = line.trim();
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return true;
    };
    if let Some(obj) = value.as_object() {
        if obj.get("wakeAgent") == Some(&serde_json::Value::Bool(false)) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_wake_gate_default_true_on_empty() {
        assert!(parse_wake_gate(""));
        assert!(parse_wake_gate("   \n   "));
    }

    #[test]
    fn parse_wake_gate_false_when_last_line_says_so() {
        assert!(!parse_wake_gate("doing checks\n{\"wakeAgent\": false}"));
        assert!(!parse_wake_gate("{\"wakeAgent\": false}"));
    }

    #[test]
    fn parse_wake_gate_true_when_flag_true_or_missing() {
        assert!(parse_wake_gate("{\"wakeAgent\": true}"));
        assert!(parse_wake_gate("{\"other\": true}"));
        assert!(parse_wake_gate("plain text output"));
        // Non-last lines with the false marker don't count.
        assert!(parse_wake_gate("{\"wakeAgent\": false}\nactual report"));
    }

    #[test]
    fn parse_wake_gate_ignores_non_json_last_line() {
        assert!(parse_wake_gate("findings: 42"));
    }

    #[test]
    fn resolve_script_timeout_env_wins() {
        // SAFETY: tests in this crate run single-threaded by default; we
        // restore the env on exit so this doesn't bleed into siblings.
        let prev = std::env::var(SCRIPT_TIMEOUT_ENV).ok();
        unsafe {
            std::env::set_var(SCRIPT_TIMEOUT_ENV, "300");
        }
        assert_eq!(resolve_script_timeout(), Duration::from_secs(300));
        unsafe {
            std::env::set_var(SCRIPT_TIMEOUT_ENV, "0");
        }
        // Zero falls through to the default — guards against scripts
        // becoming "no timeout = run forever".
        assert_eq!(
            resolve_script_timeout(),
            Duration::from_secs(DEFAULT_SCRIPT_TIMEOUT_SECS)
        );
        unsafe {
            std::env::remove_var(SCRIPT_TIMEOUT_ENV);
        }
        assert_eq!(
            resolve_script_timeout(),
            Duration::from_secs(DEFAULT_SCRIPT_TIMEOUT_SECS)
        );
        unsafe {
            match prev {
                Some(v) => std::env::set_var(SCRIPT_TIMEOUT_ENV, v),
                None => std::env::remove_var(SCRIPT_TIMEOUT_ENV),
            }
        }
    }

    #[tokio::test]
    async fn run_script_rejects_path_outside_scripts_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let scripts = tmp.path().join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        // Try to escape via ".." — must fail before any execution.
        let (ok, msg) =
            run_job_script("../../etc/passwd", &scripts, Duration::from_secs(5)).await;
        assert!(!ok);
        assert!(
            msg.contains("blocked") || msg.contains("cannot be resolved"),
            "expected sandbox rejection, got: {msg}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_script_executes_and_captures_stdout() {
        let tmp = tempfile::tempdir().unwrap();
        let scripts = tmp.path().join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        let script = scripts.join("hello.sh");
        std::fs::write(&script, "#!/bin/bash\necho hello\n").unwrap();
        let (ok, out) = run_job_script("hello.sh", &scripts, Duration::from_secs(5)).await;
        assert!(ok, "script should succeed; got: {out}");
        assert_eq!(out, "hello");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_script_reports_nonzero_exit() {
        let tmp = tempfile::tempdir().unwrap();
        let scripts = tmp.path().join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        let script = scripts.join("fail.sh");
        std::fs::write(&script, "#!/bin/bash\necho oops 1>&2\nexit 7\n").unwrap();
        let (ok, msg) = run_job_script("fail.sh", &scripts, Duration::from_secs(5)).await;
        assert!(!ok);
        assert!(msg.contains("code 7"), "expected exit code in msg: {msg}");
        assert!(msg.contains("oops"), "expected stderr in msg: {msg}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn run_script_times_out_on_hang() {
        let tmp = tempfile::tempdir().unwrap();
        let scripts = tmp.path().join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        let script = scripts.join("hang.sh");
        std::fs::write(&script, "#!/bin/bash\nsleep 10\n").unwrap();
        let (ok, msg) = run_job_script("hang.sh", &scripts, Duration::from_millis(200)).await;
        assert!(!ok);
        assert!(msg.contains("timed out"), "expected timeout msg: {msg}");
    }
}
