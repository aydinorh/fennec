//! Bundled-skill sync.
//!
//! The Fennec binary embeds the repo's `skills/` directory at compile
//! time via [`include_dir`]. On every agent boot, [`sync_bundled`]
//! walks that embedded set and reconciles it with the user's
//! `<home>/skills/` directory:
//!
//!   - **new bundled skill, no on-disk file** → copy embedded → user dir,
//!     record `name:hash` in `.bundled_manifest`.
//!   - **new bundled skill, but a same-named file already exists on
//!     disk** → skip (don't overwrite the user's skill). Record a
//!     manifest baseline only when the on-disk content already matches
//!     the embedded version.
//!   - **bundled skill recorded in manifest, file deleted on disk** →
//!     skip. The user explicitly removed it; the next sync should not
//!     resurrect it. (Use `fennec skills reset` to clear the manifest
//!     entry and re-seed.)
//!   - **bundled skill recorded in manifest, on-disk content matches
//!     manifest hash** → if the embedded version's hash differs, copy
//!     the new embedded content over and update the manifest. This is
//!     how bundled-skill content updates ship to existing installs.
//!   - **bundled skill recorded in manifest, on-disk content does NOT
//!     match manifest hash** → user has customized; skip. (Their copy
//!     stays untouched until they reset.)
//!
//! Manifest format is the v2 `name:hash` line set documented in
//! [`crate::skills::manifest`]. The hash is the SHA-256 of the
//! embedded `.md` (or directory archive) content, hex-encoded — chosen
//! over MD5 because we already pull `sha2` for other use.
//!
//! Sync is best-effort: any per-skill failure is logged at warn level
//! and skipped. A single broken file must never block the boot.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use include_dir::{Dir, File, include_dir};
use sha2::{Digest, Sha256};

use super::format::{is_reserved_flat_file, validate_skill_name};
use super::manifest::BundledManifest;

/// The bundled-skill set, embedded from `skills/` at compile time. The
/// path is relative to the workspace root because `CARGO_MANIFEST_DIR`
/// is the package root.
pub static BUNDLED: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/skills");

/// Outcome of a single sync run, returned to the caller and useful
/// for logs / boot diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyncCounts {
    /// Number of bundled skills considered (after name validation).
    pub considered: usize,
    /// Newly seeded to disk this run.
    pub installed: usize,
    /// Updated to a new bundled version (manifest hash matched user
    /// content, embedded version had a newer hash).
    pub updated: usize,
    /// Skipped because the user has customized the skill.
    pub skipped_customized: usize,
    /// Skipped because the user previously deleted the skill.
    pub skipped_deleted: usize,
    /// Already present and identical to the manifest baseline.
    pub up_to_date: usize,
    /// Failed for any reason (logged at warn level).
    pub errors: usize,
}

/// Reconcile the embedded bundled set with `<skills_root>` and
/// `<skills_root>/.bundled_manifest`.
pub fn sync_bundled(skills_root: &Path) -> Result<SyncCounts> {
    sync_with_source(skills_root, &BUNDLED)
}

/// Sync from a custom embedded source. Used by tests so they don't
/// need the real bundled set.
pub fn sync_with_source(skills_root: &Path, source: &Dir<'_>) -> Result<SyncCounts> {
    std::fs::create_dir_all(skills_root)
        .with_context(|| format!("creating skills dir {}", skills_root.display()))?;

    let mut manifest = BundledManifest::load(skills_root);
    let mut counts = SyncCounts::default();

    for file in iter_bundled_md_files(source) {
        let name = match bundled_skill_name(&file) {
            Some(n) => n,
            None => {
                tracing::debug!(path = %file.path().display(), "skipping bundled file: not a top-level .md");
                continue;
            }
        };
        if let Err(e) = validate_skill_name(&name) {
            tracing::warn!(name = %name, error = %e, "skipping bundled skill: invalid name");
            counts.errors += 1;
            continue;
        }
        counts.considered += 1;

        let embedded_bytes = file.contents();
        let embedded_hash = hash_bytes(embedded_bytes);
        let on_disk_path = skills_root.join(format!("{}.md", name));
        let manifest_hash = manifest.origin_hash(&name).map(str::to_string);

        match (on_disk_path.exists(), manifest_hash) {
            // (a) Brand-new bundled skill: not in manifest, not on disk.
            (false, None) => match write_bundled(&on_disk_path, embedded_bytes) {
                Ok(()) => {
                    manifest.set(&name, &embedded_hash);
                    counts.installed += 1;
                }
                Err(e) => {
                    tracing::warn!(name = %name, error = %e, "failed to install bundled skill");
                    counts.errors += 1;
                }
            },

            // (b) New bundled skill but a user file already exists at the
            // target. Two cases: matches the embedded → record baseline;
            // doesn't match → skip and leave the user's file untouched.
            (true, None) => match std::fs::read(&on_disk_path) {
                Ok(disk_bytes) if hash_bytes(&disk_bytes) == embedded_hash => {
                    manifest.set(&name, &embedded_hash);
                    counts.up_to_date += 1;
                }
                Ok(_) => {
                    tracing::info!(
                        name = %name,
                        "bundled skill name collides with a pre-existing user skill; \
                         not overwriting"
                    );
                    counts.skipped_customized += 1;
                }
                Err(e) => {
                    tracing::warn!(name = %name, error = %e, "could not read on-disk skill for hash compare");
                    counts.errors += 1;
                }
            },

            // (c) Manifest knows the skill, but the file is gone — the
            // user deleted it. Don't resurrect.
            (false, Some(_)) => {
                counts.skipped_deleted += 1;
            }

            // (d) Manifest knows the skill and it's still on disk:
            // compare hashes to decide whether to update or leave alone.
            (true, Some(prev_hash)) => match std::fs::read(&on_disk_path) {
                Ok(disk_bytes) => {
                    let disk_hash = hash_bytes(&disk_bytes);
                    if disk_hash == embedded_hash {
                        // No change at all.
                        counts.up_to_date += 1;
                    } else if disk_hash == prev_hash {
                        // User hasn't touched it — safe to update.
                        match write_bundled(&on_disk_path, embedded_bytes) {
                            Ok(()) => {
                                manifest.set(&name, &embedded_hash);
                                counts.updated += 1;
                            }
                            Err(e) => {
                                tracing::warn!(name = %name, error = %e, "failed to update bundled skill");
                                counts.errors += 1;
                            }
                        }
                    } else {
                        // User customized — leave alone.
                        counts.skipped_customized += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!(name = %name, error = %e, "could not read on-disk skill for update compare");
                    counts.errors += 1;
                }
            },
        }
    }

    if let Err(e) = manifest.save() {
        tracing::warn!(error = %e, "failed to write .bundled_manifest");
    }

    Ok(counts)
}

/// Clear the manifest entry for `skill_name` so the next sync treats
/// it as new again. When `restore` is true, also delete the user's
/// on-disk copy so the embedded version is restored on the next sync.
pub fn reset_bundled_skill(
    skills_root: &Path,
    skill_name: &str,
    restore: bool,
) -> Result<()> {
    let mut manifest = BundledManifest::load(skills_root);
    manifest.remove(skill_name);
    manifest
        .save()
        .context("writing .bundled_manifest after reset")?;

    if restore {
        let p = skills_root.join(format!("{}.md", skill_name));
        if p.exists() {
            std::fs::remove_file(&p).with_context(|| {
                format!("removing user copy {} during restore", p.display())
            })?;
        }
        // Best-effort: also remove a directory-format copy if present.
        let d = skills_root.join(skill_name);
        if d.is_dir() {
            std::fs::remove_dir_all(&d).with_context(|| {
                format!("removing user directory {} during restore", d.display())
            })?;
        }
    }
    Ok(())
}

/// Iterator over every top-level `.md` file in the embedded source.
/// Bundled skills live at the top level today; nested directory-format
/// bundled skills are not part of the seed set.
fn iter_bundled_md_files<'a>(source: &'a Dir<'a>) -> impl Iterator<Item = &'a File<'a>> + 'a {
    source
        .files()
        .filter(|f| f.path().extension().and_then(|e| e.to_str()) == Some("md"))
        // Exclude documentation files (README.md, etc.) the same way the
        // on-disk loader does. `skills/README.md` is embedded by include_dir
        // alongside the real bundled skills; without this it was parsed as a
        // skill named "README", failed name validation, and bumped the boot
        // error count on every start. It's documentation, not a skill.
        .filter(|f| {
            !f.path()
                .file_name()
                .and_then(|n| n.to_str())
                .map(is_reserved_flat_file)
                .unwrap_or(false)
        })
}

/// Skill name = file stem of a `.md` at the top level. Returns `None`
/// for files that aren't directly under the source root.
fn bundled_skill_name(file: &File<'_>) -> Option<String> {
    let path = file.path();
    if path.parent().map(|p| p.as_os_str().is_empty()).unwrap_or(true) {
        path.file_stem().and_then(|s| s.to_str()).map(String::from)
    } else {
        None
    }
}

/// SHA-256 hex digest of a byte slice. Used as the bundled-manifest
/// origin hash.
fn hash_bytes(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    hex::encode(h.finalize())
}

/// Atomic write of bundled bytes (temp + rename).
fn write_bundled(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_name = format!(
        "{}.tmp-{}-{}",
        target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("skill"),
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    );
    let tmp = target
        .parent()
        .map(|p| p.join(&tmp_name))
        .unwrap_or_else(|| PathBuf::from(&tmp_name));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, target)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use include_dir::{DirEntry, include_dir};
    use tempfile::TempDir;

    /// A small synthetic embedded set used by the tests so they don't
    /// depend on the real `skills/` content (which may legitimately
    /// change during PR review without the test's expectations needing
    /// to update).
    static TEST_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/src/skills/test_fixtures");

    /// Convenience: count files in the test fixture directory so a
    /// test can assert against the actual size of the seed set.
    fn fixture_skill_count() -> usize {
        TEST_DIR
            .entries()
            .iter()
            .filter(|e| matches!(e, DirEntry::File(_)))
            .count()
    }

    /// The real embedded bundle must sync with **zero** errors. `skills/`
    /// ships a README.md alongside the skills; before the reserved-file
    /// exclusion it was parsed as a skill named "README", failed name
    /// validation, and bumped `errors` on every boot.
    #[test]
    fn real_bundled_set_syncs_without_errors() {
        let tmp = TempDir::new().unwrap();
        let counts = sync_with_source(tmp.path(), &BUNDLED).unwrap();
        assert_eq!(
            counts.errors, 0,
            "bundled sync reported {} error(s); a doc file is likely being treated as a skill",
            counts.errors
        );
        assert!(counts.considered > 0, "expected real bundled skills");
    }

    /// First-run seed: every bundled skill lands on disk.
    #[test]
    fn seeds_into_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let counts = sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        assert_eq!(counts.installed, fixture_skill_count());
        assert_eq!(counts.skipped_customized, 0);

        // Manifest persisted with hashes.
        let m = BundledManifest::load(tmp.path());
        assert_eq!(m.len(), fixture_skill_count());
        for f in TEST_DIR.files() {
            let name = bundled_skill_name(f).unwrap();
            let h = m.origin_hash(&name).unwrap();
            assert!(!h.is_empty());
        }
    }

    #[test]
    fn second_run_is_up_to_date() {
        let tmp = TempDir::new().unwrap();
        sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        let counts = sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        assert_eq!(counts.installed, 0);
        assert_eq!(counts.up_to_date, fixture_skill_count());
    }

    #[test]
    fn user_customized_skill_is_not_overwritten() {
        let tmp = TempDir::new().unwrap();
        sync_with_source(tmp.path(), &TEST_DIR).unwrap();

        // Pick the first skill and edit its file.
        let first = TEST_DIR.files().next().expect("fixture has at least one skill");
        let name = bundled_skill_name(first).unwrap();
        let path = tmp.path().join(format!("{}.md", name));
        std::fs::write(&path, "---\nname: x\ndescription: customized\n---\nbody\n").unwrap();

        let counts = sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        assert_eq!(counts.skipped_customized, 1);
        // File still has the user's content.
        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("customized"));
    }

    #[test]
    fn user_deleted_skill_is_not_resurrected() {
        let tmp = TempDir::new().unwrap();
        sync_with_source(tmp.path(), &TEST_DIR).unwrap();

        let first = TEST_DIR.files().next().unwrap();
        let name = bundled_skill_name(first).unwrap();
        std::fs::remove_file(tmp.path().join(format!("{}.md", name))).unwrap();

        let counts = sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        assert_eq!(counts.skipped_deleted, 1);
        assert!(!tmp.path().join(format!("{}.md", name)).exists());
    }

    #[test]
    fn pre_existing_matching_user_file_baselines_in_manifest() {
        let tmp = TempDir::new().unwrap();
        // Pre-place a bundled skill's content with NO manifest entry.
        let first = TEST_DIR.files().next().unwrap();
        let name = bundled_skill_name(first).unwrap();
        std::fs::write(
            tmp.path().join(format!("{}.md", name)),
            first.contents(),
        )
        .unwrap();

        let counts = sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        // The pre-existing file matched embedded → baseline recorded.
        assert!(counts.up_to_date >= 1);
        let m = BundledManifest::load(tmp.path());
        assert!(m.contains(&name));
    }

    #[test]
    fn reset_clears_manifest_entry() {
        let tmp = TempDir::new().unwrap();
        sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        let first = TEST_DIR.files().next().unwrap();
        let name = bundled_skill_name(first).unwrap();
        assert!(BundledManifest::load(tmp.path()).contains(&name));

        reset_bundled_skill(tmp.path(), &name, false).unwrap();
        assert!(!BundledManifest::load(tmp.path()).contains(&name));
        // User file is still there.
        assert!(tmp.path().join(format!("{}.md", name)).is_file());
    }

    #[test]
    fn reset_with_restore_removes_user_copy() {
        let tmp = TempDir::new().unwrap();
        sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        let first = TEST_DIR.files().next().unwrap();
        let name = bundled_skill_name(first).unwrap();
        let path = tmp.path().join(format!("{}.md", name));
        assert!(path.exists());

        reset_bundled_skill(tmp.path(), &name, true).unwrap();
        assert!(!path.exists());

        // Next sync re-seeds it.
        let counts = sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        assert!(counts.installed >= 1);
        assert!(path.exists());
    }

    /// When the on-disk content matches the manifest baseline AND the
    /// embedded set has a different (newer) version, we update.
    #[test]
    fn updates_when_user_unchanged_and_embedded_changed() {
        let tmp = TempDir::new().unwrap();
        // Round 1: seed.
        sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        // Round 2: simulate an upstream change by rewriting the manifest
        // entry's hash to something stale, then re-syncing — the disk
        // content will still match the embedded version (so it's a
        // no-op for content), but the manifest hash will get refreshed.
        // To exercise the actual update path, mutate the on-disk file
        // back to a known prior content matching a fake manifest hash.
        let first = TEST_DIR.files().next().unwrap();
        let name = bundled_skill_name(first).unwrap();
        let p = tmp.path().join(format!("{}.md", name));
        let prior = b"---\nname: prior\ndescription: prior\n---\nold body\n";
        std::fs::write(&p, prior).unwrap();
        let mut m = BundledManifest::load(tmp.path());
        let prior_hash = hash_bytes(prior);
        m.set(&name, prior_hash);
        m.save().unwrap();

        let counts = sync_with_source(tmp.path(), &TEST_DIR).unwrap();
        assert_eq!(counts.updated, 1, "embedded should overwrite unchanged user copy");
        // File now matches embedded.
        let body = std::fs::read(&p).unwrap();
        assert_eq!(hash_bytes(&body), hash_bytes(first.contents()));
    }
}
