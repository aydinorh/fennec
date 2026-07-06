//! Operations behind the `skill_manage` agent tool: create, edit,
//! patch, delete, write_file, remove_file.
//!
//! These functions are independent of the JSON tool wrapper so they
//! can be tested directly and reused by the curator (which calls
//! them from inside the agent loop).
//!
//! Storage rules:
//!
//!   - New skills are written in **directory format**:
//!     `<root>/[<category>/]<name>/SKILL.md`. Even when no supporting
//!     files exist, directory format keeps the option to add them
//!     later without a migration step.
//!   - Existing flat skills (`<root>/<name>.md`) are edited in place;
//!     they are not promoted to directory format until they need a
//!     supporting file (`write_file` triggers the migration).
//!   - Bundled and hub-installed skills can be edited and patched —
//!     the change stays local — but they cannot be deleted (they
//!     would re-appear on next sync). To remove a bundled skill,
//!     remove its entry from the skills manifest under the Fennec
//!     home directory by hand (a `fennec skills` management CLI is
//!     planned but does not exist yet — do not reference one).
//!
//! Validation:
//!
//!   - Skill content (the SKILL.md body + frontmatter) is capped at
//!     100,000 characters. Larger bodies waste agent context for
//!     vanishing benefit and are usually unintentional copy-paste.
//!   - Supporting files are capped at 1 MiB each.
//!   - File paths inside a skill must live under `references/`,
//!     `templates/`, `scripts/`, or `assets/`. No `..` traversal.
//!   - Frontmatter must be valid YAML, the `name:` field must match
//!     the action's `name` argument, and `description:` must be
//!     non-empty.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::archive::{self, ArchiveResult};
use super::format::{
    SUPPORTING_DIRS, SkillLayout, SkillProvenance, validate_category, validate_skill_name,
};
use super::fuzzy::{self, ReplaceError};
use super::loader::{Skill, SkillsLoader};
use super::usage::UsageStore;

/// Hard cap on `SKILL.md` size (frontmatter + body). Comfortably
/// above legitimate skills, but small enough that an LLM-generated
/// runaway response can't blow out a user's context budget.
pub const MAX_SKILL_CONTENT_BYTES: usize = 100_000;

/// Hard cap on a single supporting file (anything under
/// `references/`, `templates/`, `scripts/`, `assets/`). 1 MiB matches
/// the loader's `MAX_SKILL_FILE_BYTES`.
pub const MAX_SUPPORT_FILE_BYTES: usize = 1 * 1024 * 1024;

/// Top-level error type for `skill_manage` operations. Discriminated
/// so the JSON tool wrapper can report a stable error category to
/// the LLM.
#[derive(Debug)]
pub enum ManageError {
    /// Argument validation (bad name, bad category, bad path,
    /// oversize content, etc).
    InvalidArgument(String),
    /// A pre-condition wasn't met (skill doesn't exist when the
    /// action requires it; skill already exists when create was
    /// called; skill is bundled/hub-installed and can't be deleted;
    /// fuzzy patch couldn't find or unambiguously locate the search
    /// string).
    Conflict(String),
    /// I/O or filesystem error from std::fs.
    Io(std::io::Error),
    /// Anything else surfaced via anyhow.
    Other(anyhow::Error),
}

impl std::fmt::Display for ManageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManageError::InvalidArgument(s) => write!(f, "invalid argument: {}", s),
            ManageError::Conflict(s) => write!(f, "conflict: {}", s),
            ManageError::Io(e) => write!(f, "io error: {}", e),
            ManageError::Other(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ManageError {}

impl From<std::io::Error> for ManageError {
    fn from(e: std::io::Error) -> Self {
        ManageError::Io(e)
    }
}

impl From<anyhow::Error> for ManageError {
    fn from(e: anyhow::Error) -> Self {
        ManageError::Other(e)
    }
}

/// What a successful action returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManageOutcome {
    /// One-line human-readable summary the tool wrapper relays back
    /// to the agent.
    pub message: String,
    /// Path most directly affected by the action: the `SKILL.md` for
    /// create/edit/patch/delete, the supporting file for
    /// write_file/remove_file.
    pub primary_path: PathBuf,
    /// `true` when this action moved a flat skill to directory
    /// format (write_file on a flat skill triggers this). The
    /// caller may want to log it.
    pub migrated_to_directory: bool,
}

/// Create a brand-new skill with `content` at
/// `<root>/[<category>/]<name>/SKILL.md`.
///
/// Errors:
///
///   - `InvalidArgument` if name or category fail validation, content
///     is empty, content exceeds `MAX_SKILL_CONTENT_BYTES`, or the
///     frontmatter is invalid.
///   - `Conflict` if a skill of this name already exists anywhere in
///     the root (flat OR directory, any category).
pub fn create(
    skills_root: &Path,
    name: &str,
    content: &str,
    category: Option<&str>,
    existing: &[Skill],
) -> Result<ManageOutcome, ManageError> {
    validate_skill_name(name)
        .map_err(|e| ManageError::InvalidArgument(format!("name: {}", e)))?;
    if let Some(c) = category {
        validate_category(c)
            .map_err(|e| ManageError::InvalidArgument(format!("category: {}", e)))?;
    }
    validate_content(name, content)?;

    if existing.iter().any(|s| s.name == name) {
        return Err(ManageError::Conflict(format!(
            "a skill named {:?} already exists; use `edit` to update it",
            name
        )));
    }

    let dir = match category {
        Some(c) => skills_root.join(c).join(name),
        None => skills_root.join(name),
    };
    std::fs::create_dir_all(&dir).context("creating skill directory")?;
    let skill_md = dir.join("SKILL.md");
    write_atomic(&skill_md, content).context("writing SKILL.md")?;

    Ok(ManageOutcome {
        message: format!("created skill {:?} at {}", name, skill_md.display()),
        primary_path: skill_md,
        migrated_to_directory: false,
    })
}

/// Replace an existing skill's `SKILL.md` (or flat `.md`) wholesale.
pub fn edit(
    _skills_root: &Path,
    name: &str,
    new_content: &str,
    target: &Skill,
) -> Result<ManageOutcome, ManageError> {
    if target.name != name {
        return Err(ManageError::InvalidArgument(format!(
            "edit target {:?} does not match supplied name {:?}",
            target.name, name
        )));
    }
    validate_content(name, new_content)?;

    let layout = target
        .layout
        .as_ref()
        .ok_or_else(|| ManageError::Conflict(format!("skill {:?} has no on-disk layout", name)))?;
    let path = layout.skill_md_path();
    write_atomic(&path, new_content).context("writing edited SKILL.md")?;

    Ok(ManageOutcome {
        message: format!("edited skill {:?}", name),
        primary_path: path,
        migrated_to_directory: false,
    })
}

/// Fuzzy find-and-replace inside `SKILL.md` (default) or a supporting
/// file (when `file_path` is given). Uses the `fuzzy` matcher so
/// LLM-generated patches with whitespace drift still apply cleanly.
pub fn patch(
    skills_root: &Path,
    name: &str,
    old_string: &str,
    new_string: &str,
    file_path: Option<&str>,
    replace_all: bool,
    target: &Skill,
) -> Result<ManageOutcome, ManageError> {
    if target.name != name {
        return Err(ManageError::InvalidArgument(format!(
            "patch target {:?} does not match supplied name {:?}",
            target.name, name
        )));
    }
    if old_string.is_empty() {
        return Err(ManageError::InvalidArgument(
            "old_string must be non-empty".into(),
        ));
    }

    let layout = target.layout.as_ref().ok_or_else(|| {
        ManageError::Conflict(format!("skill {:?} has no on-disk layout", name))
    })?;
    let target_path = match file_path {
        Some(rel) => {
            let dir = match layout {
                SkillLayout::Directory { dir, .. } => dir.clone(),
                SkillLayout::Flat { .. } => {
                    return Err(ManageError::Conflict(format!(
                        "skill {:?} is flat; cannot patch supporting files until \
                         write_file migrates it to directory format",
                        name
                    )));
                }
            };
            resolve_supporting_path(&dir, rel)?
        }
        None => layout.skill_md_path(),
    };

    let original = std::fs::read_to_string(&target_path).context("reading patch target")?;
    let (updated, _strategy, _count) =
        fuzzy::replace(&original, old_string, new_string, replace_all).map_err(
            |e| match e {
                ReplaceError::NotFound => ManageError::Conflict(format!(
                    "old_string not found in {}",
                    target_path.display()
                )),
                ReplaceError::Ambiguous(n) => ManageError::Conflict(format!(
                    "old_string matches {} locations; set replace_all=true if intentional",
                    n
                )),
                ReplaceError::EmptyNeedle => {
                    ManageError::InvalidArgument("old_string must be non-empty".into())
                }
                ReplaceError::OverlappingMatches => {
                    ManageError::Other(anyhow::anyhow!("internal: overlapping matches"))
                }
            },
        )?;

    // For SKILL.md edits, validate the full updated content; for
    // supporting files we just check the size cap.
    if file_path.is_none() {
        validate_content(name, &updated)?;
    } else if updated.len() > MAX_SUPPORT_FILE_BYTES {
        return Err(ManageError::InvalidArgument(format!(
            "patched supporting file would be {} bytes, exceeding the {} cap",
            updated.len(),
            MAX_SUPPORT_FILE_BYTES
        )));
    }

    write_atomic(&target_path, &updated).context("writing patched file")?;

    let _ = skills_root; // currently unused by patch's body
    Ok(ManageOutcome {
        message: format!("patched {}", target_path.display()),
        primary_path: target_path,
        migrated_to_directory: false,
    })
}

/// Move a skill into `<root>/.archive/`. Refuses to delete bundled or
/// hub-installed skills (those would reappear on next sync).
pub fn delete(
    skills_root: &Path,
    name: &str,
    target: &Skill,
    usage: &UsageStore,
) -> Result<(ManageOutcome, ArchiveResult), ManageError> {
    if target.name != name {
        return Err(ManageError::InvalidArgument(format!(
            "delete target {:?} does not match supplied name {:?}",
            target.name, name
        )));
    }
    if target.provenance != SkillProvenance::AgentCreated {
        return Err(ManageError::Conflict(format!(
            "refusing to delete {} skill {:?}: edit it instead (deleting it \
             would only last until the next sync restores it)",
            match target.provenance {
                SkillProvenance::Bundled => "bundled",
                SkillProvenance::HubInstalled => "hub-installed",
                SkillProvenance::AgentCreated => unreachable!(),
            },
            name
        )));
    }

    let layout = target.layout.as_ref().ok_or_else(|| {
        ManageError::Conflict(format!("skill {:?} has no on-disk layout", name))
    })?;
    let result = archive::archive(skills_root, name, layout)
        .context("archiving deleted skill")?;
    usage.forget(name);

    let outcome = ManageOutcome {
        message: format!(
            "archived skill {:?} to {}",
            name,
            result.archived_to.display()
        ),
        primary_path: result.archived_to.clone(),
        migrated_to_directory: false,
    };
    Ok((outcome, result))
}

/// Write a supporting file under one of `references/`, `templates/`,
/// `scripts/`, `assets/`. If the target skill is currently flat, the
/// flat `.md` is migrated to directory format first
/// (`<root>/<name>.md` → `<root>/<name>/SKILL.md`) so the supporting
/// file can live alongside.
pub fn write_file(
    skills_root: &Path,
    name: &str,
    file_path: &str,
    content: &str,
    target: &Skill,
) -> Result<ManageOutcome, ManageError> {
    if target.name != name {
        return Err(ManageError::InvalidArgument(format!(
            "write_file target {:?} does not match supplied name {:?}",
            target.name, name
        )));
    }
    if content.len() > MAX_SUPPORT_FILE_BYTES {
        return Err(ManageError::InvalidArgument(format!(
            "supporting file is {} bytes, exceeding the {} cap",
            content.len(),
            MAX_SUPPORT_FILE_BYTES
        )));
    }

    let layout = target.layout.as_ref().ok_or_else(|| {
        ManageError::Conflict(format!("skill {:?} has no on-disk layout", name))
    })?;
    let mut migrated = false;
    let dir = match layout {
        SkillLayout::Directory { dir, .. } => dir.clone(),
        SkillLayout::Flat { file } => {
            let new_dir = skills_root.join(name);
            if new_dir.exists() {
                return Err(ManageError::Conflict(format!(
                    "cannot migrate flat skill {:?} to directory format: \
                     {} already exists",
                    name,
                    new_dir.display()
                )));
            }
            std::fs::create_dir(&new_dir).context("creating migrated skill directory")?;
            std::fs::rename(file, new_dir.join("SKILL.md"))
                .context("moving flat .md to SKILL.md")?;
            migrated = true;
            new_dir
        }
    };

    let dest = resolve_supporting_path(&dir, file_path)?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).context("creating supporting subdirectory")?;
    }
    write_atomic(&dest, content).context("writing supporting file")?;

    Ok(ManageOutcome {
        message: format!("wrote {}", dest.display()),
        primary_path: dest,
        migrated_to_directory: migrated,
    })
}

/// Remove a supporting file under one of `references/`, `templates/`,
/// `scripts/`, `assets/`.
pub fn remove_file(
    _skills_root: &Path,
    name: &str,
    file_path: &str,
    target: &Skill,
) -> Result<ManageOutcome, ManageError> {
    if target.name != name {
        return Err(ManageError::InvalidArgument(format!(
            "remove_file target {:?} does not match supplied name {:?}",
            target.name, name
        )));
    }
    let dir = match target.layout.as_ref() {
        Some(SkillLayout::Directory { dir, .. }) => dir.clone(),
        Some(SkillLayout::Flat { .. }) => {
            return Err(ManageError::Conflict(format!(
                "skill {:?} is flat; it has no supporting files to remove",
                name
            )));
        }
        None => {
            return Err(ManageError::Conflict(format!(
                "skill {:?} has no on-disk layout",
                name
            )));
        }
    };
    let dest = resolve_supporting_path(&dir, file_path)?;
    if !dest.exists() {
        return Err(ManageError::Conflict(format!(
            "supporting file {} does not exist",
            dest.display()
        )));
    }
    std::fs::remove_file(&dest).context("removing supporting file")?;

    Ok(ManageOutcome {
        message: format!("removed {}", dest.display()),
        primary_path: dest,
        migrated_to_directory: false,
    })
}

/// Validate `SKILL.md` content: parseable frontmatter, `name:` field
/// matches argument, `description:` non-empty, content within size
/// cap.
fn validate_content(name: &str, content: &str) -> Result<(), ManageError> {
    if content.is_empty() {
        return Err(ManageError::InvalidArgument("content must be non-empty".into()));
    }
    if content.len() > MAX_SKILL_CONTENT_BYTES {
        return Err(ManageError::InvalidArgument(format!(
            "content is {} bytes, exceeding the {} cap",
            content.len(),
            MAX_SKILL_CONTENT_BYTES
        )));
    }
    // Parse via the loader's frontmatter parser so we share the same
    // YAML tolerance and don't double-spec the format.
    let parsed = SkillsLoader::parse_skill(content)
        .map_err(|e| ManageError::InvalidArgument(format!("frontmatter: {}", e)))?;
    if parsed.name != name {
        return Err(ManageError::InvalidArgument(format!(
            "frontmatter name {:?} must match supplied name {:?}",
            parsed.name, name
        )));
    }
    if parsed.description.trim().is_empty() {
        return Err(ManageError::InvalidArgument(
            "frontmatter `description:` must be non-empty".into(),
        ));
    }
    Ok(())
}

/// Resolve a `file_path` argument to an absolute path inside one of
/// `references/`, `templates/`, `scripts/`, `assets/`. Rejects `..`
/// traversal and any path outside those subdirectories.
fn resolve_supporting_path(skill_dir: &Path, file_path: &str) -> Result<PathBuf, ManageError> {
    if file_path.is_empty() {
        return Err(ManageError::InvalidArgument("file_path must be non-empty".into()));
    }
    if file_path.contains("..") {
        return Err(ManageError::InvalidArgument(format!(
            "file_path {:?} contains traversal segments",
            file_path
        )));
    }
    let normalized = file_path.replace('\\', "/");
    let trimmed = normalized.trim_start_matches('/');
    let first_segment = trimmed
        .split('/')
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            ManageError::InvalidArgument(format!("file_path {:?} is not well-formed", file_path))
        })?;
    if !SUPPORTING_DIRS.contains(&first_segment) {
        return Err(ManageError::InvalidArgument(format!(
            "file_path must live under one of {:?} (got first segment {:?})",
            SUPPORTING_DIRS, first_segment
        )));
    }
    let dest = skill_dir.join(trimmed);
    // The textual `..` reject and the SUPPORTING_DIRS prefix check
    // together ensure `dest` cannot escape `skill_dir`. (We don't
    // canonicalize here because the leaf may not exist yet, and on
    // macOS canonicalize follows /private/var symlinks which would
    // produce false-positive escape errors.)
    Ok(dest)
}

/// Atomic write: temp file + rename. Used for SKILL.md and supporting
/// files so a crash mid-write can never produce a partial file.
fn write_atomic(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("mkdir {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    ));
    std::fs::write(&tmp, content)
        .with_context(|| format!("writing temp {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} → {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::format::SkillState;
    use tempfile::TempDir;

    fn skill_md(name: &str, body: &str) -> String {
        format!("---\nname: {}\ndescription: ok\n---\n{}\n", name, body)
    }

    fn flat_skill(root: &Path, name: &str) -> Skill {
        let p = root.join(format!("{}.md", name));
        std::fs::write(&p, skill_md(name, "body")).unwrap();
        Skill {
            name: name.to_string(),
            description: "ok".into(),
            content: "body".into(),
            always: false,
            requirements: vec![],
            layout: Some(SkillLayout::Flat { file: p }),
            provenance: SkillProvenance::AgentCreated,
            state: SkillState::Active,
            pinned: false,
        }
    }

    fn dir_skill(root: &Path, name: &str) -> Skill {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), skill_md(name, "body")).unwrap();
        Skill {
            name: name.to_string(),
            description: "ok".into(),
            content: "body".into(),
            always: false,
            requirements: vec![],
            layout: Some(SkillLayout::Directory {
                dir,
                category: None,
            }),
            provenance: SkillProvenance::AgentCreated,
            state: SkillState::Active,
            pinned: false,
        }
    }

    // -- create -----------------------------------------------------

    #[test]
    fn create_writes_directory_format_skill() {
        let tmp = TempDir::new().unwrap();
        let body = skill_md("foo", "Body");
        let r = create(tmp.path(), "foo", &body, None, &[]).unwrap();
        assert_eq!(r.primary_path, tmp.path().join("foo").join("SKILL.md"));
        let content = std::fs::read_to_string(&r.primary_path).unwrap();
        assert!(content.contains("Body"));
    }

    #[test]
    fn create_with_category() {
        let tmp = TempDir::new().unwrap();
        let body = skill_md("foo", "Body");
        let r = create(tmp.path(), "foo", &body, Some("productivity"), &[]).unwrap();
        assert_eq!(
            r.primary_path,
            tmp.path()
                .join("productivity")
                .join("foo")
                .join("SKILL.md")
        );
    }

    #[test]
    fn create_rejects_invalid_name() {
        let tmp = TempDir::new().unwrap();
        let body = skill_md("Foo", "Body");
        let err = create(tmp.path(), "Foo", &body, None, &[]).unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)), "got {:?}", err);
    }

    #[test]
    fn create_rejects_when_skill_exists() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let body = skill_md("foo", "Body");
        let err = create(tmp.path(), "foo", &body, None, std::slice::from_ref(&s))
            .unwrap_err();
        assert!(matches!(err, ManageError::Conflict(_)));
    }

    #[test]
    fn create_rejects_when_frontmatter_name_mismatches() {
        let tmp = TempDir::new().unwrap();
        let body = skill_md("not-foo", "Body");
        let err = create(tmp.path(), "foo", &body, None, &[]).unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)), "got {:?}", err);
    }

    #[test]
    fn create_rejects_oversize_content() {
        let tmp = TempDir::new().unwrap();
        let big = "a".repeat(MAX_SKILL_CONTENT_BYTES + 1);
        let body = skill_md("foo", &big);
        let err = create(tmp.path(), "foo", &body, None, &[]).unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)));
    }

    #[test]
    fn create_rejects_empty_description() {
        let tmp = TempDir::new().unwrap();
        let body = "---\nname: foo\ndescription: \n---\nbody\n";
        let err = create(tmp.path(), "foo", body, None, &[]).unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)), "got {:?}", err);
    }

    // -- edit -------------------------------------------------------

    #[test]
    fn edit_flat_skill_in_place() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let new_body = skill_md("foo", "New body");
        let r = edit(tmp.path(), "foo", &new_body, &s).unwrap();
        let content = std::fs::read_to_string(&r.primary_path).unwrap();
        assert!(content.contains("New body"));
        // Still flat — no directory promotion.
        assert!(tmp.path().join("foo.md").is_file());
        assert!(!tmp.path().join("foo").exists());
    }

    #[test]
    fn edit_directory_skill_writes_skill_md() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let new_body = skill_md("foo", "Updated");
        let r = edit(tmp.path(), "foo", &new_body, &s).unwrap();
        assert_eq!(r.primary_path, tmp.path().join("foo").join("SKILL.md"));
        let content = std::fs::read_to_string(&r.primary_path).unwrap();
        assert!(content.contains("Updated"));
    }

    #[test]
    fn edit_rejects_name_mismatch() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let body = skill_md("foo", "X");
        let err = edit(tmp.path(), "wrong-name", &body, &s).unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)));
    }

    // -- patch ------------------------------------------------------

    #[test]
    fn patch_skill_md_with_exact_match() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let r = patch(
            tmp.path(),
            "foo",
            "body",
            "patched",
            None,
            false,
            &s,
        )
        .unwrap();
        let content = std::fs::read_to_string(&r.primary_path).unwrap();
        assert!(content.contains("patched"));
        assert!(!content.contains("\nbody\n"));
    }

    #[test]
    fn patch_supporting_file() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let layout = match s.layout.as_ref().unwrap() {
            SkillLayout::Directory { dir, .. } => dir.clone(),
            _ => unreachable!(),
        };
        std::fs::create_dir_all(layout.join("references")).unwrap();
        std::fs::write(layout.join("references").join("api.md"), "old\nstuff\n").unwrap();

        let r = patch(
            tmp.path(),
            "foo",
            "old\n",
            "new\n",
            Some("references/api.md"),
            false,
            &s,
        )
        .unwrap();
        let content = std::fs::read_to_string(&r.primary_path).unwrap();
        assert_eq!(content, "new\nstuff\n");
    }

    #[test]
    fn patch_rejects_path_outside_supporting_dirs() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let err = patch(
            tmp.path(),
            "foo",
            "x",
            "y",
            Some("../../escape.txt"),
            false,
            &s,
        )
        .unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)), "got {:?}", err);
    }

    #[test]
    fn patch_rejects_bare_path_outside_supporting_dirs() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let err = patch(
            tmp.path(),
            "foo",
            "x",
            "y",
            Some("not-a-supporting-dir/foo.md"),
            false,
            &s,
        )
        .unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)));
    }

    #[test]
    fn patch_flat_skill_supporting_file_rejected() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let err = patch(
            tmp.path(),
            "foo",
            "x",
            "y",
            Some("references/api.md"),
            false,
            &s,
        )
        .unwrap_err();
        assert!(matches!(err, ManageError::Conflict(_)));
    }

    #[test]
    fn patch_propagates_not_found() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let err = patch(
            tmp.path(),
            "foo",
            "no-such-text",
            "x",
            None,
            false,
            &s,
        )
        .unwrap_err();
        assert!(matches!(err, ManageError::Conflict(_)), "got {:?}", err);
    }

    // -- delete -----------------------------------------------------

    #[test]
    fn delete_archives_agent_created_skill() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let usage = UsageStore::open(tmp.path());
        usage.bump_use("foo");

        let (outcome, archive_result) = delete(tmp.path(), "foo", &s, &usage).unwrap();
        assert!(archive_result.archived_to.starts_with(tmp.path().join(".archive")));
        assert!(outcome.primary_path.exists());
        // Original gone.
        assert!(!tmp.path().join("foo.md").exists());
        // Usage record forgotten.
        assert!(usage.get("foo").is_none());
    }

    #[test]
    fn delete_refuses_bundled_skill() {
        let tmp = TempDir::new().unwrap();
        let mut s = flat_skill(tmp.path(), "foo");
        s.provenance = SkillProvenance::Bundled;
        let usage = UsageStore::open(tmp.path());
        let err = delete(tmp.path(), "foo", &s, &usage).unwrap_err();
        assert!(matches!(err, ManageError::Conflict(_)));
        // Skill is still there.
        assert!(tmp.path().join("foo.md").is_file());
    }

    // -- write_file -------------------------------------------------

    #[test]
    fn write_file_on_directory_skill_creates_supporting_file() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let r = write_file(
            tmp.path(),
            "foo",
            "references/api.md",
            "REF\n",
            &s,
        )
        .unwrap();
        assert_eq!(
            r.primary_path,
            tmp.path().join("foo").join("references").join("api.md")
        );
        assert_eq!(std::fs::read_to_string(&r.primary_path).unwrap(), "REF\n");
        assert!(!r.migrated_to_directory);
    }

    #[test]
    fn write_file_migrates_flat_to_directory() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let r = write_file(
            tmp.path(),
            "foo",
            "references/note.md",
            "NOTE\n",
            &s,
        )
        .unwrap();
        assert!(r.migrated_to_directory);
        assert!(tmp.path().join("foo").join("SKILL.md").is_file());
        assert!(tmp.path().join("foo").join("references").join("note.md").is_file());
        // Old flat .md is gone.
        assert!(!tmp.path().join("foo.md").exists());
    }

    #[test]
    fn write_file_rejects_path_outside_supporting_dirs() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let err = write_file(
            tmp.path(),
            "foo",
            "etc/passwd",
            "X",
            &s,
        )
        .unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)));
    }

    #[test]
    fn write_file_rejects_oversize_content() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let big = "a".repeat(MAX_SUPPORT_FILE_BYTES + 1);
        let err = write_file(
            tmp.path(),
            "foo",
            "references/big.txt",
            &big,
            &s,
        )
        .unwrap_err();
        assert!(matches!(err, ManageError::InvalidArgument(_)));
    }

    // -- remove_file -----------------------------------------------

    #[test]
    fn remove_file_deletes_supporting_file() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let layout_dir = match s.layout.as_ref().unwrap() {
            SkillLayout::Directory { dir, .. } => dir.clone(),
            _ => unreachable!(),
        };
        std::fs::create_dir_all(layout_dir.join("references")).unwrap();
        std::fs::write(layout_dir.join("references").join("api.md"), "X").unwrap();

        remove_file(tmp.path(), "foo", "references/api.md", &s).unwrap();
        assert!(!layout_dir.join("references").join("api.md").exists());
    }

    #[test]
    fn remove_file_errors_when_missing() {
        let tmp = TempDir::new().unwrap();
        let s = dir_skill(tmp.path(), "foo");
        let err = remove_file(tmp.path(), "foo", "references/missing.md", &s)
            .unwrap_err();
        assert!(matches!(err, ManageError::Conflict(_)));
    }

    #[test]
    fn remove_file_rejects_flat_skill() {
        let tmp = TempDir::new().unwrap();
        let s = flat_skill(tmp.path(), "foo");
        let err = remove_file(tmp.path(), "foo", "references/api.md", &s)
            .unwrap_err();
        assert!(matches!(err, ManageError::Conflict(_)));
    }
}
