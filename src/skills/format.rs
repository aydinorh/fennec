//! Skill format types: state machine, provenance, and supporting-file
//! directory layout.
//!
//! The skill collection lives at `<home>/skills/`. Two layouts are supported:
//!
//!   - **flat**: a single `.md` file at the top level. Used by every bundled
//!     skill and any agent-created skill that needs no supporting files.
//!   - **directory**: a folder containing `SKILL.md` and optional
//!     `references/`, `templates/`, `scripts/`, `assets/` subdirectories.
//!     Used when a skill ships supporting content (the curator demotes
//!     narrow content into umbrella `references/` files).
//!
//! A directory skill may itself live under a single category folder
//! (`<home>/skills/<category>/<name>/SKILL.md`). Deeper nesting is rejected
//! at load time.
//!
//! The `.archive/` directory under `<home>/skills/` is reserved for skills
//! the curator has archived; it is never walked by the loader.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The four reserved supporting-file subdirectories inside a directory-format
/// skill. Anything written by `skill_manage` outside this set is rejected.
pub const SUPPORTING_DIRS: &[&str] = &["references", "templates", "scripts", "assets"];

/// Reserved names that cannot appear under `<home>/skills/` as either a flat
/// file or a directory. They are used for sidecar state (`.usage.json`,
/// `.bundled_manifest`, `.hub/`, `.archive/`, `.curator_state`).
pub const RESERVED_DIR_ENTRIES: &[&str] = &[
    ".archive",
    ".hub",
    ".usage.json",
    ".bundled_manifest",
    ".curator_state",
];

/// Documentation files that commonly live alongside skills but are NOT
/// skills themselves. Fennec supports a flat single-file skill format
/// (`<home>/skills/<name>.md`), so a `README.md` dropped into the skills
/// root would otherwise be parsed as a skill named "README" and rejected
/// with a confusing "name must start with lowercase" warning on every
/// load. We skip these well-known doc filenames up front instead. Matched
/// case-insensitively (`README.md`, `readme.md`, …). The upstream avoids
/// this entirely by only ever globbing files literally named `SKILL.md`,
/// but Fennec keeps its flat-skill convenience and excludes docs by name.
pub const RESERVED_FLAT_FILES: &[&str] = &[
    "readme.md",
    "description.md",
    "changelog.md",
    "license.md",
    "contributing.md",
    "agents.md",
];

/// Returns true if `file_name` is a documentation file that must not be
/// treated as a flat skill. Case-insensitive over [`RESERVED_FLAT_FILES`].
pub fn is_reserved_flat_file(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    RESERVED_FLAT_FILES.contains(&lower.as_str())
}

/// Lifecycle state of a skill. State lives in the usage sidecar
/// (`<home>/skills/.usage.json`), not in the skill's frontmatter — this
/// keeps skills' on-disk content stable across automated transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SkillState {
    /// In active use. Default for any skill the agent has touched recently
    /// or that has just been created.
    #[default]
    Active,
    /// Not used in `stale_after_days` (default 30). Still loaded into the
    /// agent context, but flagged for review by the curator.
    Stale,
    /// Not used in `archive_after_days` (default 90). Moved to
    /// `<home>/skills/.archive/<name>/` and no longer loaded. Recoverable
    /// via `fennec curator restore`.
    Archived,
}

impl SkillState {
    /// Stable string form for the usage-sidecar JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            SkillState::Active => "active",
            SkillState::Stale => "stale",
            SkillState::Archived => "archived",
        }
    }
}

/// Where a skill came from. Determines whether the curator and usage
/// tracker treat it as mutable.
///
/// The provenance filter is load-bearing for the usage sidecar: only
/// `AgentCreated` skills are tracked. Bundled and hub-installed skills
/// are excluded so user-installed third-party content does not pollute
/// the curator's signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SkillProvenance {
    /// Shipped with the Fennec binary. Tracked in `.bundled_manifest`
    /// with an origin hash so user customizations are detected (and
    /// preserved on update).
    Bundled,
    /// Installed via the skills hub from an external registry. Tracked
    /// in `.hub/lock.json`. (Hub installs land in a later phase; for now
    /// no skill ever reports this provenance.)
    HubInstalled,
    /// Authored by the user or by the agent via `skill_manage`. The
    /// curator and usage tracker only operate on these.
    #[default]
    AgentCreated,
}

impl SkillProvenance {
    /// Whether the curator and usage sidecar should track this skill.
    pub fn is_agent_created(self) -> bool {
        matches!(self, SkillProvenance::AgentCreated)
    }
}

/// Layout of a single skill on disk. Returned by the loader so callers
/// (the `skill_manage` tool, the archive routine, the curator) can find
/// the skill's storage without re-walking the directory tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillLayout {
    /// `<home>/skills/<name>.md` — a single markdown file. Cannot have
    /// supporting subfiles.
    Flat {
        /// Absolute path to the `.md` file.
        file: PathBuf,
    },
    /// `<home>/skills/[<category>/]<name>/SKILL.md` — a folder. May contain
    /// `references/`, `templates/`, `scripts/`, `assets/` subdirectories.
    Directory {
        /// Absolute path to the skill's directory (the parent of
        /// `SKILL.md`).
        dir: PathBuf,
        /// Optional single-segment category folder. `None` means the skill
        /// directory sits at the top level of `<home>/skills/`.
        category: Option<String>,
    },
}

impl SkillLayout {
    /// The path that should be opened to read the skill body.
    ///
    /// For a flat skill this is the `.md` file itself; for a directory
    /// skill it is `SKILL.md` inside the directory.
    pub fn skill_md_path(&self) -> PathBuf {
        match self {
            SkillLayout::Flat { file } => file.clone(),
            SkillLayout::Directory { dir, .. } => dir.join("SKILL.md"),
        }
    }

    /// The on-disk root for the skill. For a flat skill, the file itself
    /// (since there is nothing else); for a directory skill, the
    /// containing directory.
    ///
    /// Used by the archive routine: archiving a flat skill moves the
    /// `.md` file; archiving a directory skill moves the whole tree.
    pub fn root_path(&self) -> &Path {
        match self {
            SkillLayout::Flat { file } => file.as_path(),
            SkillLayout::Directory { dir, .. } => dir.as_path(),
        }
    }

    /// Whether this layout supports supporting files in
    /// `references/`, `templates/`, `scripts/`, `assets/`.
    pub fn supports_subfiles(&self) -> bool {
        matches!(self, SkillLayout::Directory { .. })
    }

    /// The category folder, if any. Always `None` for flat skills.
    pub fn category(&self) -> Option<&str> {
        match self {
            SkillLayout::Flat { .. } => None,
            SkillLayout::Directory { category, .. } => category.as_deref(),
        }
    }
}

/// Skill name validation: lowercase letters, digits, dot, underscore,
/// hyphen. Must start with a letter or digit. Length 1..=64.
///
/// Matches the upstream validation rules so skills authored under one
/// agent remain portable.
pub fn validate_skill_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("skill name must be non-empty".into());
    }
    if name.len() > 64 {
        return Err(format!("skill name too long ({} > 64 chars)", name.len()));
    }
    let mut chars = name.chars();
    let first = chars.next().expect("non-empty checked above");
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err(format!(
            "skill name must start with a lowercase letter or digit (got {:?})",
            first
        ));
    }
    for c in chars {
        let ok = c.is_ascii_lowercase()
            || c.is_ascii_digit()
            || c == '.'
            || c == '_'
            || c == '-';
        if !ok {
            return Err(format!(
                "skill name contains invalid character {:?} (allowed: a-z 0-9 . _ -)",
                c
            ));
        }
    }
    Ok(())
}

/// Category name validation: same charset and length as skill names, no
/// path separators. A single segment, never nested.
pub fn validate_category(category: &str) -> Result<(), String> {
    if category.contains('/') || category.contains('\\') {
        return Err("category must be a single directory segment, no slashes".into());
    }
    validate_skill_name(category).map_err(|e| format!("invalid category: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_state_as_str_round_trip() {
        for s in [SkillState::Active, SkillState::Stale, SkillState::Archived] {
            let raw = s.as_str();
            let back: SkillState = serde_json::from_str(&format!("\"{}\"", raw)).unwrap();
            assert_eq!(s, back);
        }
    }

    #[test]
    fn provenance_is_agent_created() {
        assert!(SkillProvenance::AgentCreated.is_agent_created());
        assert!(!SkillProvenance::Bundled.is_agent_created());
        assert!(!SkillProvenance::HubInstalled.is_agent_created());
    }

    #[test]
    fn flat_layout_paths() {
        let f = SkillLayout::Flat {
            file: PathBuf::from("/skills/foo.md"),
        };
        assert_eq!(f.skill_md_path(), PathBuf::from("/skills/foo.md"));
        assert_eq!(f.root_path(), Path::new("/skills/foo.md"));
        assert!(!f.supports_subfiles());
        assert_eq!(f.category(), None);
    }

    #[test]
    fn directory_layout_paths() {
        let d = SkillLayout::Directory {
            dir: PathBuf::from("/skills/cat/foo"),
            category: Some("cat".into()),
        };
        assert_eq!(d.skill_md_path(), PathBuf::from("/skills/cat/foo/SKILL.md"));
        assert_eq!(d.root_path(), Path::new("/skills/cat/foo"));
        assert!(d.supports_subfiles());
        assert_eq!(d.category(), Some("cat"));
    }

    #[test]
    fn validate_skill_name_accepts() {
        for n in ["foo", "foo-bar", "foo_bar", "f.b", "a", "0", "a1", "agent-self-improvement"] {
            assert!(validate_skill_name(n).is_ok(), "{} should be valid", n);
        }
    }

    #[test]
    fn validate_skill_name_rejects() {
        for n in [
            "",
            "Foo",          // uppercase
            "-foo",         // leading hyphen
            ".foo",         // leading dot
            "foo bar",      // space
            "foo/bar",      // slash
            "foo\\bar",     // backslash
            "foo!",         // punctuation
            "föö",          // non-ascii
        ] {
            assert!(
                validate_skill_name(n).is_err(),
                "{:?} should be invalid",
                n
            );
        }
        // length cap
        let long = "a".repeat(65);
        assert!(validate_skill_name(&long).is_err());
    }

    #[test]
    fn validate_category_rejects_slashes() {
        assert!(validate_category("foo/bar").is_err());
        assert!(validate_category("foo\\bar").is_err());
        assert!(validate_category("foo").is_ok());
    }

    #[test]
    fn supporting_dirs_are_well_formed() {
        // Cheap regression: each entry is lowercase, no slashes, distinct.
        let mut seen = std::collections::HashSet::new();
        for d in SUPPORTING_DIRS {
            assert!(d.chars().all(|c| c.is_ascii_lowercase()), "{} not lowercase", d);
            assert!(!d.contains('/'), "{} contains slash", d);
            assert!(seen.insert(*d), "{} duplicated", d);
        }
    }

    #[test]
    fn reserved_entries_distinct_from_supporting() {
        let supporting: std::collections::HashSet<_> = SUPPORTING_DIRS.iter().copied().collect();
        for r in RESERVED_DIR_ENTRIES {
            assert!(!supporting.contains(r), "{} overlaps SUPPORTING_DIRS", r);
        }
    }
}
