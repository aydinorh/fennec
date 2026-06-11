//! File-write helpers for files that must not be world-readable.
//!
//! `std::fs::write` creates files with the umask-defined mode (typically
//! 0644), so the classic "write plaintext secret, then chmod 0600 afterward"
//! sequence has a race window where the file is readable by other local
//! users. This module writes to a same-directory tempfile opened with
//! `mode(0o600)` up front, then atomically renames it into place — so the
//! target path never exists with loose permissions, even for an instant.
//!
//! On non-Unix platforms we fall back to `std::fs::write` (Windows ACL
//! hardening is a separate job).

use std::path::Path;

use anyhow::{Context, Result};

/// Atomically write `content` to `path` with 0600 permissions on Unix.
///
/// - Writes to `<parent>/.<basename>.tmp.<pid>` with `O_CREAT|O_WRONLY|O_TRUNC`
///   and mode 0600, then `rename(2)`s into place (same filesystem, so the
///   rename is atomic).
/// - Creates missing parent directories.
/// - On failure, removes the tempfile so we don't leak `.tmp.*` files.
/// - On Windows, falls back to `std::fs::write` (no ACL handling yet).
pub fn write_secure(path: &Path, content: &[u8]) -> Result<()> {
    #[cfg(unix)]
    {
        use std::fs::OpenOptions;
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;

        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("write_secure: path has no parent directory"))?;
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating parent dir {}", parent.display()))?;
        }

        let base = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");
        let temp_path = parent.join(format!(".{}.tmp.{}", base, std::process::id()));

        let result = (|| -> Result<()> {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&temp_path)
                .with_context(|| {
                    format!("creating secure tempfile {}", temp_path.display())
                })?;
            f.write_all(content)
                .with_context(|| format!("writing {}", temp_path.display()))?;
            f.sync_all()
                .with_context(|| format!("syncing {}", temp_path.display()))?;
            Ok(())
        })();

        match result {
            Ok(()) => std::fs::rename(&temp_path, path)
                .with_context(|| format!("renaming into place: {}", path.display())),
            Err(e) => {
                let _ = std::fs::remove_file(&temp_path);
                Err(e)
            }
        }
    }

    #[cfg(not(unix))]
    {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating parent dir {}", parent.display()))?;
            }
        }
        std::fs::write(path, content)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

/// Atomically write `content` to `path`, preserving the target's
/// existing permissions (new files get umask defaults).
///
/// For WORKSPACE files written by the agent's file tools: writes to a
/// same-directory tempfile, fsyncs, then `rename(2)`s into place — a
/// crash or concurrent reader never observes a truncated/half-written
/// file, and an overwrite that fails midway leaves the original
/// intact. The upstream's write_file does the same (temp + rename,
/// preserving the existing file's mode).
///
/// Secret-bearing Fennec-internal files should use [`write_secure`]
/// (0600 from the first byte) instead — this helper deliberately does
/// NOT clamp permissions, because a user's project file written 0600
/// breaks anything else that reads it.
pub fn write_atomic(path: &Path, content: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("write_atomic: path has no parent directory"))?;
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating parent dir {}", parent.display()))?;
    }

    let base = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("file");
    let temp_path = parent.join(format!(".{}.tmp.{}", base, std::process::id()));

    let result = (|| -> Result<()> {
        std::fs::write(&temp_path, content)
            .with_context(|| format!("writing tempfile {}", temp_path.display()))?;
        // Preserve the existing target's permissions across the swap
        // (Unix). New files keep the tempfile's umask-default mode.
        #[cfg(unix)]
        if let Ok(meta) = std::fs::metadata(path) {
            let _ = std::fs::set_permissions(&temp_path, meta.permissions());
        }
        let f = std::fs::File::open(&temp_path)
            .with_context(|| format!("reopening {}", temp_path.display()))?;
        f.sync_all()
            .with_context(|| format!("syncing {}", temp_path.display()))?;
        std::fs::rename(&temp_path, path)
            .with_context(|| format!("renaming into place: {}", path.display()))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

/// Create `path` (and missing parents) with 0700 permissions on Unix.
///
/// Unlike `std::fs::create_dir_all` which uses the umask default (typically
/// 0755), this sets mode 0700 on the deepest-created directory. Parent
/// directories that already exist are left untouched; missing parents are
/// created with umask defaults (matching `create_dir_all` semantics).
///
/// On non-Unix platforms, delegates to `create_dir_all`.
pub fn create_dir_private(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::fs::DirBuilder;
        use std::os::unix::fs::DirBuilderExt;

        if path.exists() {
            return Ok(());
        }
        DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(path)
            .with_context(|| format!("creating private dir {}", path.display()))?;
        Ok(())
    }

    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(path)
            .with_context(|| format!("creating dir {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_secure_creates_file_with_content() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("a.txt");
        write_secure(&path, b"hello").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"hello");
    }

    #[test]
    fn write_secure_overwrites_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("a.txt");
        std::fs::write(&path, b"first").unwrap();
        write_secure(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn write_secure_creates_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("deep/nested/file.txt");
        write_secure(&path, b"x").unwrap();
        assert!(path.exists());
    }

    #[cfg(unix)]
    #[test]
    fn write_secure_sets_mode_600() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("secret.txt");
        write_secure(&path, b"top secret").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0o600 perms, got {:o}", mode);
    }

    #[cfg(unix)]
    #[test]
    fn write_secure_overwrite_keeps_600() {
        // Even if the existing file has looser perms, the rename-in-place
        // replaces it with a fresh 0600 file.
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("secret.txt");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        write_secure(&path, b"new").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn write_atomic_writes_and_overwrites() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("file.txt");
        write_atomic(&path, b"first").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        write_atomic(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
    }

    /// Unlike write_secure, write_atomic preserves the target's
    /// existing permissions — a user's 0755 script must stay 0755
    /// after the agent edits it.
    #[cfg(unix)]
    #[test]
    fn write_atomic_preserves_existing_mode() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("script.sh");
        std::fs::write(&path, b"#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        write_atomic(&path, b"#!/bin/sh\necho hi\n").unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "existing mode must survive the atomic swap");
    }

    #[test]
    fn write_atomic_cleans_up_on_failure() {
        let tmp = tempfile::tempdir().unwrap();
        // Target is an existing directory — the rename must fail.
        let path = tmp.path().to_path_buf();
        let _ = write_atomic(&path, b"x");
        let orphans: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(orphans.is_empty(), "tempfiles left behind: {:?}", orphans);
    }

    #[test]
    fn write_secure_cleans_up_on_failure() {
        // Writing to a directory (not a file) should fail AND leave no
        // orphan tempfiles behind.
        let tmp = tempfile::tempdir().unwrap();
        // Path is an existing directory; open should fail.
        let path = tmp.path().to_path_buf();
        let _ = write_secure(&path, b"x");
        // Count any .tmp files left over.
        let orphans: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .contains(".tmp.")
            })
            .collect();
        assert!(orphans.is_empty(), "tempfiles left behind: {:?}", orphans);
    }

    #[cfg(unix)]
    #[test]
    fn create_dir_private_sets_mode_700() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("priv");
        create_dir_private(&path).unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "expected 0o700 perms, got {:o}", mode);
    }

    #[test]
    fn create_dir_private_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("priv");
        create_dir_private(&path).unwrap();
        // Second call must not error even though the dir exists.
        create_dir_private(&path).unwrap();
    }
}
