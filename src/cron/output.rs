//! Per-job cron output persistence.
//!
//! Each fired cron job writes a timestamped `.md` file under
//! `<output_dir>/<job_id>/`. This mirrors the upstream's
//! `~/.hermes/cron/output/{job_id}/{ts}.md` layout and is what
//! `context_from` reads to inject preceding-job outputs into the next
//! prompt.
//!
//! Job IDs are filesystem path components. To avoid path-escape via
//! crafted or legacy IDs (`../escape`, absolute paths, nested separators),
//! [`job_output_dir`] rejects anything that isn't a single safe path
//! component — matches the upstream's `_job_output_dir` guard.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

/// Maximum characters injected from a preceding job's output into a
/// downstream job's prompt — caps prompt bloat regardless of how
/// chatty the upstream job got. Matches the upstream's
/// `_MAX_CONTEXT_CHARS`.
pub const MAX_CONTEXT_CHARS: usize = 8000;

/// Default cron output directory derived from a `jobs.json` path.
/// Lives as a sibling of the jobs file so a single Fennec home has a
/// single output tree.
pub fn default_output_dir_for(jobs_path: &Path) -> PathBuf {
    jobs_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("cron_output")
}

/// Resolve a job's output directory under `output_dir`. Rejects job
/// IDs that contain path separators, `..`, absolute prefixes, or empty
/// strings — mirrors the upstream's `_job_output_dir` sandbox guard.
pub fn job_output_dir(output_dir: &Path, job_id: &str) -> Result<PathBuf> {
    let trimmed = job_id.trim();
    if trimmed.is_empty() || trimmed == "." || trimmed == ".." {
        return Err(anyhow!("invalid cron job id for output path: {job_id:?}"));
    }
    if trimmed.contains('/') || trimmed.contains('\\') {
        return Err(anyhow!("invalid cron job id for output path: {job_id:?}"));
    }
    let candidate = Path::new(trimmed);
    if candidate.is_absolute() || candidate.components().count() != 1 {
        return Err(anyhow!("invalid cron job id for output path: {job_id:?}"));
    }
    Ok(output_dir.join(trimmed))
}

/// Save a job's run output to `<output_dir>/<job_id>/<ts>.md`. Creates
/// the per-job directory if needed and tightens it to owner-only
/// (0700) on Unix; the file is written 0600. Returns the file path on
/// success.
pub fn save_job_output(output_dir: &Path, job_id: &str, output: &str) -> Result<PathBuf> {
    let job_dir = job_output_dir(output_dir, job_id)?;
    std::fs::create_dir_all(&job_dir)
        .with_context(|| format!("creating job output dir {}", job_dir.display()))?;
    secure_dir(&job_dir);

    let ts = chrono::Utc::now().format("%Y-%m-%d_%H-%M-%S");
    let out_path = job_dir.join(format!("{ts}.md"));

    // Atomic write: tempfile sibling + rename.
    let tmp_path = job_dir.join(format!(".output.tmp.{}", std::process::id()));
    let result = (|| -> Result<()> {
        std::fs::write(&tmp_path, output)
            .with_context(|| format!("writing tempfile {}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, &out_path).with_context(|| {
            format!(
                "renaming {} to {}",
                tmp_path.display(),
                out_path.display()
            )
        })?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            secure_file(&out_path);
            Ok(out_path)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

/// Return the most recent saved output for a job (the newest file in
/// `<output_dir>/<job_id>/` by mtime). Returns `None` when the job has
/// no recorded output yet. Output is truncated at
/// [`MAX_CONTEXT_CHARS`] with a clear "[... output truncated ...]"
/// marker so prompt size stays predictable.
pub fn latest_job_output(output_dir: &Path, job_id: &str) -> Option<String> {
    let job_dir = job_output_dir(output_dir, job_id).ok()?;
    if !job_dir.exists() {
        return None;
    }
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&job_dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let mtime = entry.metadata().ok()?.modified().ok()?;
        match &newest {
            Some((cur, _)) if *cur >= mtime => {}
            _ => newest = Some((mtime, path)),
        }
    }
    let (_, path) = newest?;
    let raw = std::fs::read_to_string(&path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().count() > MAX_CONTEXT_CHARS {
        let head: String = trimmed.chars().take(MAX_CONTEXT_CHARS).collect();
        Some(format!("{head}\n\n[... output truncated ...]"))
    } else {
        Some(trimmed.to_string())
    }
}

/// Remove a job's entire output directory. Called after a job is
/// deleted so orphaned outputs don't accumulate. Best-effort: a
/// missing dir is success; a removal error is logged but not
/// propagated (the job is already gone from `jobs.json`).
pub fn cleanup_job_output(output_dir: &Path, job_id: &str) {
    let Ok(job_dir) = job_output_dir(output_dir, job_id) else {
        return;
    };
    if !job_dir.exists() {
        return;
    }
    if let Err(e) = std::fs::remove_dir_all(&job_dir) {
        tracing::warn!(
            "cleanup_job_output: failed to remove {}: {}",
            job_dir.display(),
            e
        );
    }
}

#[cfg(unix)]
fn secure_dir(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn secure_dir(_path: &Path) {}

#[cfg(unix)]
fn secure_file(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if path.exists() {
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

#[cfg(not(unix))]
fn secure_file(_path: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_output_dir_rejects_path_escape() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(job_output_dir(tmp.path(), "../etc").is_err());
        assert!(job_output_dir(tmp.path(), "a/b").is_err());
        assert!(job_output_dir(tmp.path(), "/etc/passwd").is_err());
        assert!(job_output_dir(tmp.path(), "").is_err());
        assert!(job_output_dir(tmp.path(), ".").is_err());
        assert!(job_output_dir(tmp.path(), "..").is_err());
    }

    #[test]
    fn job_output_dir_accepts_safe_ids() {
        let tmp = tempfile::tempdir().unwrap();
        let p = job_output_dir(tmp.path(), "abc123").unwrap();
        assert_eq!(p, tmp.path().join("abc123"));
        // UUID-shaped IDs (which is what cron_tool emits) are fine.
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let p = job_output_dir(tmp.path(), uuid).unwrap();
        assert_eq!(p.file_name().unwrap(), uuid);
    }

    #[test]
    fn save_and_read_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = save_job_output(tmp.path(), "j1", "hello world").unwrap();
        assert!(path.exists());
        let read = latest_job_output(tmp.path(), "j1").unwrap();
        assert_eq!(read, "hello world");
    }

    #[test]
    fn latest_picks_newest_when_multiple_files() {
        let tmp = tempfile::tempdir().unwrap();
        save_job_output(tmp.path(), "j1", "old output").unwrap();
        // Force a different second so the second save has a strictly
        // newer mtime even on filesystems with 1-second mtime granularity.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        save_job_output(tmp.path(), "j1", "newer output").unwrap();
        let read = latest_job_output(tmp.path(), "j1").unwrap();
        assert_eq!(read, "newer output");
    }

    #[test]
    fn latest_truncates_oversized_output() {
        let tmp = tempfile::tempdir().unwrap();
        let big = "x".repeat(MAX_CONTEXT_CHARS + 500);
        save_job_output(tmp.path(), "j1", &big).unwrap();
        let read = latest_job_output(tmp.path(), "j1").unwrap();
        assert!(read.ends_with("[... output truncated ...]"));
        // Truncated chunk + the marker + the blank-line separator.
        assert!(read.chars().count() > MAX_CONTEXT_CHARS);
        assert!(read.chars().count() < MAX_CONTEXT_CHARS + 100);
    }

    #[test]
    fn latest_returns_none_when_no_outputs() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(latest_job_output(tmp.path(), "never").is_none());
    }

    #[test]
    fn cleanup_removes_job_output_dir() {
        let tmp = tempfile::tempdir().unwrap();
        save_job_output(tmp.path(), "j1", "stuff").unwrap();
        assert!(tmp.path().join("j1").exists());
        cleanup_job_output(tmp.path(), "j1");
        assert!(!tmp.path().join("j1").exists());
    }

    #[test]
    fn default_output_dir_derived_from_jobs_path() {
        let jobs = Path::new("/home/me/.fennec/cron_jobs.json");
        assert_eq!(
            default_output_dir_for(jobs),
            Path::new("/home/me/.fennec/cron_output")
        );
    }

    #[cfg(unix)]
    #[test]
    fn save_secures_output_file_to_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = save_job_output(tmp.path(), "j1", "secret cron output").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let dir_mode = std::fs::metadata(tmp.path().join("j1"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(dir_mode, 0o700);
    }
}
