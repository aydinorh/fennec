use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::format::{
    is_reserved_flat_file, RESERVED_DIR_ENTRIES, SkillLayout, SkillProvenance, SkillState,
    validate_category, validate_skill_name,
};
use super::manifest::{BundledManifest, HubLock};

/// Max bytes we'll accept for a skill file. Skill markdown is short by
/// convention — 1 MiB is orders of magnitude above anything realistic and
/// keeps an attacker-planted huge file from OOMing the process at startup.
const MAX_SKILL_FILE_BYTES: u64 = 1 * 1024 * 1024;

/// A skill loaded from disk.
///
/// Skills come from one of two on-disk layouts (see `SkillLayout`):
/// flat `.md` files at the top level, or directory-format folders that
/// can carry `references/`, `templates/`, `scripts/`, `assets/` subfiles.
/// Both share the same in-memory shape so callers (the prompt builder,
/// the `load_skill` tool, the curator) treat them uniformly.
#[derive(Debug, Clone, Default)]
pub struct Skill {
    /// Machine-readable skill name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// The markdown content (body after frontmatter).
    pub content: String,
    /// Whether this skill is always injected into the agent context.
    pub always: bool,
    /// External commands that must be available in PATH.
    pub requirements: Vec<String>,
    /// On-disk layout (flat file or directory). Populated by the loader;
    /// `None` for `Skill` instances constructed in-memory by tests or
    /// callers that don't care about disk paths.
    #[doc(hidden)]
    pub layout: Option<SkillLayout>,
    /// Where this skill came from (bundled / hub-installed / agent-
    /// created). Defaults to `AgentCreated`.
    #[doc(hidden)]
    pub provenance: SkillProvenance,
    /// Lifecycle state pulled from the usage sidecar at load time.
    /// Defaults to `Active` when no record exists.
    #[doc(hidden)]
    pub state: SkillState,
    /// Whether the user has pinned this skill (exempt from automatic
    /// transitions). Pulled from the usage sidecar at load time.
    #[doc(hidden)]
    pub pinned: bool,
}

/// Wire shape of the YAML frontmatter block at the top of every skill file.
///
/// Using `serde_yaml_ng` here (instead of the hand-rolled line-by-line
/// scanner we used to have) fixes four classes of bug the old parser had:
///   - values containing `:` were silently truncated
///     (`description: "a: b"` → stored as `"\"a"`)
///   - YAML-valid boolean forms — `True`, `yes`, `"true"` — all evaluated
///     to false because only the literal unquoted `true` was accepted
///   - malformed entries under `requirements:` silently dropped
///   - YAML block-style vs inline-array styles for `requirements` weren't
///     both supported
#[derive(Deserialize, Default)]
struct FrontMatter {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    always: bool,
    #[serde(default)]
    requirements: Vec<String>,
}

/// Loads and manages skills from disk.
pub struct SkillsLoader;

impl SkillsLoader {
    /// Load every skill from the given directory, walking both layouts:
    ///
    ///   - **flat**: `<dir>/<name>.md` (current bundled-skill format).
    ///   - **directory**: `<dir>/<name>/SKILL.md` and
    ///     `<dir>/<category>/<name>/SKILL.md`, with optional supporting
    ///     subdirectories (`references/`, `templates/`, `scripts/`,
    ///     `assets/`).
    ///
    /// Hardening applied at every level:
    ///   - **symlink filter**: any symlink (file *or* directory) is
    ///     skipped. A symlink pointing at `~/.ssh/id_rsa` would
    ///     otherwise be read and have its first line logged as "invalid
    ///     skill" content — a data-exfiltration vector on multi-user
    ///     machines.
    ///   - **size cap**: files larger than `MAX_SKILL_FILE_BYTES` are
    ///     skipped. Prevents startup OOM on an attacker-planted huge file.
    ///   - **reserved-name skip**: `.archive/`, `.hub/`, `.usage.json`,
    ///     `.bundled_manifest`, `.curator_state` and any dotfile under
    ///     the root are ignored — these are sidecar state, not skills.
    ///
    /// Per-skill errors are logged and skipped — a single broken file
    /// must never block the whole load phase.
    ///
    /// All skills are returned with default provenance / state / pinned
    /// values. To populate provenance and lifecycle state from the
    /// bundled manifest, hub lock, and usage sidecar, call
    /// [`Self::load_with_provenance`].
    pub fn load_from_directory(path: &Path) -> Result<Vec<Skill>> {
        Self::load_with_provenance(path, None, None, None)
    }

    /// Same as [`Self::load_from_directory`] but populates each loaded
    /// skill's `provenance` (from the bundled manifest and hub lock) and
    /// `state` / `pinned` (from the usage sidecar) when those are
    /// available.
    ///
    /// Pass `None` for any input to skip that source — bundled skills
    /// will then default to `AgentCreated` provenance, and lifecycle
    /// fields default to `Active` / unpinned.
    pub fn load_with_provenance(
        path: &Path,
        bundled: Option<&BundledManifest>,
        hub: Option<&HubLock>,
        usage: Option<&super::usage::UsageStore>,
    ) -> Result<Vec<Skill>> {
        let mut skills = Vec::new();
        if !path.is_dir() {
            return Ok(skills);
        }

        let entries = std::fs::read_dir(path)
            .with_context(|| format!("reading skills directory: {}", path.display()))?;

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(
                        dir = %path.display(),
                        error = %e,
                        "skipping unreadable entry in skills dir"
                    );
                    continue;
                }
            };
            let entry_path = entry.path();
            let file_name = match entry_path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };

            // Skip dotfiles and reserved sidecar entries unconditionally.
            if file_name.starts_with('.')
                || RESERVED_DIR_ENTRIES.contains(&file_name)
            {
                continue;
            }

            let meta = match std::fs::symlink_metadata(&entry_path) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(
                        path = %entry_path.display(),
                        error = %e,
                        "skipping skills entry: stat failed"
                    );
                    continue;
                }
            };
            let ft = meta.file_type();
            if ft.is_symlink() {
                tracing::warn!(
                    path = %entry_path.display(),
                    "skipping symlink in skills directory"
                );
                continue;
            }

            if ft.is_file() {
                if entry_path.extension().and_then(|e| e.to_str()) != Some("md") {
                    continue;
                }
                // README.md and friends are documentation, not skills. In
                // Fennec's flat single-file skill format they would parse
                // as a skill named after the file and fail validation,
                // emitting a noisy warning on every load. Skip them.
                if is_reserved_flat_file(file_name) {
                    continue;
                }
                if meta.len() > MAX_SKILL_FILE_BYTES {
                    tracing::warn!(
                        path = %entry_path.display(),
                        bytes = meta.len(),
                        cap = MAX_SKILL_FILE_BYTES,
                        "skipping oversized flat skill file"
                    );
                    continue;
                }
                match Self::load_flat_skill(&entry_path) {
                    Ok(skill) => skills.push(skill),
                    Err(e) => tracing::warn!(
                        path = %entry_path.display(),
                        error = %e,
                        "skipping invalid flat skill"
                    ),
                }
            } else if ft.is_dir() {
                Self::walk_dir_entry(&entry_path, None, &mut skills);
            }
        }

        // Apply provenance and lifecycle state based on the on-disk
        // sidecars after every skill is loaded. Doing this in a second
        // pass keeps the walker self-contained.
        for s in &mut skills {
            if let Some(b) = bundled {
                if b.contains(&s.name) {
                    s.provenance = SkillProvenance::Bundled;
                }
            }
            if let Some(h) = hub {
                if h.contains(&s.name) {
                    // Hub wins over bundled: a hub install of a name
                    // that happens to overlap a bundled skill is
                    // attributable to the user's install action.
                    s.provenance = SkillProvenance::HubInstalled;
                }
            }
            if let Some(u) = usage {
                if let Some(rec) = u.get(&s.name) {
                    s.state = rec.state;
                    s.pinned = rec.pinned;
                }
            }
        }

        Ok(skills)
    }

    /// Walk a single subdirectory under `<root>/skills/`. Either it is
    /// itself a directory-format skill (`SKILL.md` at this level) or it
    /// is a category folder containing one or more directory-format
    /// skills (`<category>/<name>/SKILL.md`).
    ///
    /// `current_category` is `None` at the top level and `Some("…")`
    /// when we recurse into a category. Nesting beyond one level is
    /// flagged and skipped — skills don't live three levels deep.
    fn walk_dir_entry(
        dir: &Path,
        current_category: Option<&str>,
        skills: &mut Vec<Skill>,
    ) {
        let dir_name = match dir.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => return,
        };
        let skill_md = dir.join("SKILL.md");
        if skill_md.is_file() {
            // This subdirectory IS a skill. Validate its name.
            if let Err(e) = validate_skill_name(dir_name) {
                tracing::warn!(
                    path = %dir.display(),
                    error = %e,
                    "skipping directory skill: invalid name"
                );
                return;
            }
            match Self::load_directory_skill(
                dir,
                dir_name,
                current_category.map(str::to_string),
            ) {
                Ok(skill) => skills.push(skill),
                Err(e) => tracing::warn!(
                    path = %dir.display(),
                    error = %e,
                    "skipping invalid directory skill"
                ),
            }
            return;
        }

        // Not a skill itself. If we're already inside a category, give
        // up — categories don't nest.
        if current_category.is_some() {
            tracing::warn!(
                path = %dir.display(),
                "skipping deeply-nested directory: skills do not nest beyond one category level"
            );
            return;
        }

        // Treat as a category folder. Walk children one level deep.
        if let Err(e) = validate_category(dir_name) {
            tracing::warn!(
                path = %dir.display(),
                error = %e,
                "skipping category folder: invalid name"
            );
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    path = %dir.display(),
                    error = %e,
                    "skipping unreadable category folder"
                );
                return;
            }
        };
        for child in entries.flatten() {
            let child_path = child.path();
            let child_name = match child_path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if child_name.starts_with('.') {
                continue;
            }
            let cm = match std::fs::symlink_metadata(&child_path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if cm.file_type().is_symlink() {
                tracing::warn!(
                    path = %child_path.display(),
                    "skipping symlink under category folder"
                );
                continue;
            }
            if cm.file_type().is_dir() {
                Self::walk_dir_entry(&child_path, Some(dir_name), skills);
            }
        }
    }

    /// Read a flat `<name>.md` skill file from `path`.
    fn load_flat_skill(path: &Path) -> Result<Skill> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading flat skill {}", path.display()))?;
        let mut skill = Self::parse_skill(&raw)?;
        // Skill names from flat files derive from frontmatter, not the
        // filename, but we still validate them so directory-format
        // tools can rely on a consistent name shape.
        if let Err(e) = validate_skill_name(&skill.name) {
            anyhow::bail!("skill name {:?} invalid: {}", skill.name, e);
        }
        skill.layout = Some(SkillLayout::Flat {
            file: path.to_path_buf(),
        });
        Ok(skill)
    }

    /// Read a `<dir>/SKILL.md` directory-format skill.
    fn load_directory_skill(
        dir: &Path,
        dir_name: &str,
        category: Option<String>,
    ) -> Result<Skill> {
        let skill_md = dir.join("SKILL.md");
        let meta = std::fs::metadata(&skill_md)
            .with_context(|| format!("stat-ing {}", skill_md.display()))?;
        if meta.len() > MAX_SKILL_FILE_BYTES {
            anyhow::bail!(
                "SKILL.md too large ({} bytes, cap {})",
                meta.len(),
                MAX_SKILL_FILE_BYTES
            );
        }
        let raw = std::fs::read_to_string(&skill_md)
            .with_context(|| format!("reading {}", skill_md.display()))?;
        let mut skill = Self::parse_skill(&raw)?;
        // For directory skills, frontmatter `name` and the directory
        // name must agree — otherwise a rename outside the tool would
        // produce a phantom name mismatch the curator can't resolve.
        if skill.name != dir_name {
            anyhow::bail!(
                "frontmatter name {:?} does not match directory name {:?}",
                skill.name,
                dir_name
            );
        }
        skill.layout = Some(SkillLayout::Directory {
            dir: dir.to_path_buf(),
            category,
        });
        Ok(skill)
    }

    /// Parse a single skill from its raw markdown+frontmatter text.
    ///
    /// Frontmatter is delimited by two `---` lines (each on its own line,
    /// with optional `\r` before the newline). Everything in between is
    /// fed to a real YAML parser; everything after the closing fence is
    /// the markdown body.
    pub(crate) fn parse_skill(raw: &str) -> Result<Skill> {
        // Strip an optional UTF-8 BOM so editors that insert one don't
        // cause the opening fence check to fail.
        let raw = raw.trim_start_matches('\u{feff}');

        // Opening fence: the file must start with `---` followed by LF
        // or CRLF. Anything else — including a blank line — is rejected.
        let after_open = raw
            .strip_prefix("---\n")
            .or_else(|| raw.strip_prefix("---\r\n"))
            .context("skill file must start with YAML frontmatter (---)")?;

        // Closing fence: scan forward for `\n---` where the char after
        // `---` is a newline boundary (`\n`, `\r\n`, or end-of-file).
        // Requiring the newline boundary is what prevents the old
        // `find("\n---")` bug of matching inside the body on `\n---foo`.
        let mut i = 0usize;
        let close_pos = loop {
            let Some(rel) = after_open[i..].find("\n---") else {
                anyhow::bail!("missing closing '---' fence for frontmatter");
            };
            let abs = i + rel;
            let tail = &after_open[abs + 4..];
            if tail.is_empty() || tail.starts_with('\n') || tail.starts_with("\r\n") {
                break abs;
            }
            i = abs + 1;
        };

        let frontmatter = &after_open[..close_pos];
        // Step past the closing `\n---` and its trailing newline.
        let after_close = &after_open[close_pos + 4..];
        let body = after_close
            .strip_prefix('\n')
            .or_else(|| after_close.strip_prefix("\r\n"))
            .unwrap_or(after_close)
            .trim()
            .to_string();

        let fm: FrontMatter = serde_yaml_ng::from_str(frontmatter)
            .context("parsing skill frontmatter as YAML")?;

        anyhow::ensure!(
            !fm.name.trim().is_empty(),
            "skill must have a non-empty 'name' field"
        );

        Ok(Skill {
            name: fm.name.trim().to_string(),
            description: fm.description,
            content: body,
            always: fm.always,
            requirements: fm.requirements,
            ..Default::default()
        })
    }

    /// Filter skills to those whose requirements are all satisfied
    /// (commands exist in PATH).
    pub fn filter_available(skills: &[Skill]) -> Vec<&Skill> {
        skills
            .iter()
            .filter(|skill| skill.requirements.iter().all(|req| which_exists(req)))
            .collect()
    }

    /// Build a prompt string from skills.
    ///
    /// - `always: true` skills have their full content injected.
    /// - Other skills are listed with their descriptions so the agent knows
    ///   they can be loaded on demand.
    pub fn build_skills_prompt(skills: &[Skill]) -> String {
        let mut sections = Vec::new();

        // Always-on skills: inject full content.
        let always_skills: Vec<&Skill> = skills.iter().filter(|s| s.always).collect();
        if !always_skills.is_empty() {
            sections.push("## Active Skills\n".to_string());
            for skill in &always_skills {
                sections.push(format!("### {}\n{}\n", skill.name, skill.content));
            }
        }

        // On-demand skills: list with descriptions.
        let on_demand: Vec<&Skill> = skills.iter().filter(|s| !s.always).collect();
        if !on_demand.is_empty() {
            sections.push(
                "## Available Skills\nUse the `load_skill` tool to activate any of these:\n"
                    .to_string(),
            );
            for skill in &on_demand {
                sections.push(format!("- **{}**: {}\n", skill.name, skill.description));
            }
        }

        sections.join("")
    }
}

/// Check whether a command exists in PATH.
fn which_exists(command: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(command).is_file()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::UsageStore;

    #[test]
    fn test_parse_skill_basic() {
        let raw = r#"---
name: test-skill
description: A test skill
always: false
requirements:
  - git
---

This is the skill content.

It has multiple paragraphs.
"#;
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "A test skill");
        assert!(!skill.always);
        assert_eq!(skill.requirements, vec!["git"]);
        assert!(skill.content.contains("This is the skill content."));
        assert!(skill.content.contains("multiple paragraphs"));
    }

    #[test]
    fn test_parse_skill_always_true() {
        let raw = r#"---
name: always-on
description: Always active
always: true
---

Always injected content.
"#;
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(skill.name, "always-on");
        assert!(skill.always);
        assert!(skill.requirements.is_empty());
    }

    #[test]
    fn test_parse_skill_no_name_fails() {
        let raw = r#"---
description: No name
---

Content.
"#;
        assert!(SkillsLoader::parse_skill(raw).is_err());
    }

    #[test]
    fn test_parse_skill_no_frontmatter_fails() {
        let raw = "Just some text without frontmatter.";
        assert!(SkillsLoader::parse_skill(raw).is_err());
    }

    #[test]
    fn test_build_skills_prompt() {
        let skills = vec![
            Skill {
                name: "always-skill".to_string(),
                description: "Always on".to_string(),
                content: "Always content here.".to_string(),
                always: true,
                requirements: vec![],
                ..Default::default()
            },
            Skill {
                name: "optional-skill".to_string(),
                description: "Load on demand".to_string(),
                content: "Optional content.".to_string(),
                always: false,
                requirements: vec![],
                ..Default::default()
            },
        ];

        let prompt = SkillsLoader::build_skills_prompt(&skills);
        assert!(prompt.contains("Active Skills"));
        assert!(prompt.contains("Always content here."));
        assert!(prompt.contains("Available Skills"));
        assert!(prompt.contains("optional-skill"));
        assert!(prompt.contains("Load on demand"));
        // On-demand skill content should NOT be in the prompt.
        assert!(!prompt.contains("Optional content."));
    }

    #[test]
    fn test_filter_available_no_requirements() {
        let skills = vec![Skill {
            name: "no-req".to_string(),
            description: "No requirements".to_string(),
            content: "Content.".to_string(),
            always: false,
            requirements: vec![],
            ..Default::default()
        }];

        let available = SkillsLoader::filter_available(&skills);
        assert_eq!(available.len(), 1);
    }

    #[test]
    fn test_load_from_nonexistent_directory() {
        let result = SkillsLoader::load_from_directory(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    /// Regression: the old hand parser used split_once(':') and dropped
    /// the part after the first colon, so a description like
    /// `description: "Error 500: server fault"` was stored as `"\"Error 500"`.
    #[test]
    fn parse_skill_preserves_colons_in_values() {
        let raw = r#"---
name: http-errors
description: "Error 500: server fault vs 404: not found"
---

Body.
"#;
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(
            skill.description,
            "Error 500: server fault vs 404: not found"
        );
    }

    /// Regression: `always: True`, `always: yes`, `always: "true"` all
    /// evaluated to false because the old parser did literal `== "true"`
    /// on the raw value string. serde_yaml_ng understands YAML bool
    /// aliases; this test pins that.
    #[test]
    fn parse_skill_accepts_yaml_bool_variants() {
        for form in ["true", "True", "TRUE"] {
            let raw = format!(
                "---\nname: x\ndescription: y\nalways: {}\n---\nbody\n",
                form
            );
            let skill = SkillsLoader::parse_skill(&raw).unwrap();
            assert!(skill.always, "always: {} should parse as true", form);
        }
        for form in ["false", "False", "FALSE"] {
            let raw = format!(
                "---\nname: x\ndescription: y\nalways: {}\n---\nbody\n",
                form
            );
            let skill = SkillsLoader::parse_skill(&raw).unwrap();
            assert!(!skill.always, "always: {} should parse as false", form);
        }
    }

    /// Regression: `requirements` as an inline YAML array used to silently
    /// produce an empty list because the old parser only handled block style.
    #[test]
    fn parse_skill_accepts_inline_requirements_array() {
        let raw = r#"---
name: with-reqs
description: test
requirements: [git, curl, jq]
---
body
"#;
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(skill.requirements, vec!["git", "curl", "jq"]);
    }

    /// Regression: frontmatter containing `requirements:` with a list of
    /// items in block style must produce the full list.
    #[test]
    fn parse_skill_accepts_block_requirements_list() {
        let raw = r#"---
name: with-reqs
description: test
requirements:
  - git
  - curl
  - jq
---
body
"#;
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(skill.requirements, vec!["git", "curl", "jq"]);
    }

    /// Boundary: `\n---foo` inside the body must NOT be mistaken for the
    /// closing fence. (The audit flagged that the old `find("\n---")`
    /// match was permissive here.)
    #[test]
    fn parse_skill_closing_fence_requires_line_boundary() {
        let raw = r#"---
name: t
description: test
---
Before
---foo
After
"#;
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(skill.name, "t");
        // Body must contain the `---foo` line — it's a markdown content
        // line, not a frontmatter fence.
        assert!(skill.content.contains("---foo"), "body: {:?}", skill.content);
        assert!(skill.content.contains("After"));
    }

    /// Boundary: file with CRLF line endings must parse correctly.
    #[test]
    fn parse_skill_handles_crlf_line_endings() {
        let raw = "---\r\nname: crlf\r\ndescription: test\r\n---\r\nbody\r\n";
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(skill.name, "crlf");
        assert_eq!(skill.content, "body");
    }

    /// Boundary: file with an optional UTF-8 BOM (some editors add it)
    /// must parse correctly — the old parser required `raw.starts_with("---")`
    /// which failed on files with a BOM prefix.
    #[test]
    fn parse_skill_handles_utf8_bom() {
        let raw = "\u{feff}---\nname: bom\ndescription: test\n---\nbody\n";
        let skill = SkillsLoader::parse_skill(raw).unwrap();
        assert_eq!(skill.name, "bom");
    }

    /// load_from_directory must skip symlinks. A symlink pointing at a
    /// sensitive file would otherwise be read and its first lines logged
    /// as "invalid skill" content.
    #[cfg(unix)]
    #[test]
    fn load_from_directory_skips_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        // Create a real skill file outside the skills dir that we'll
        // link into the skills dir.
        let external = tmp.path().join("sensitive.md");
        std::fs::write(
            &external,
            "---\nname: sensitive\ndescription: leak\n---\nbody\n",
        )
        .unwrap();

        let skills_dir = tmp.path().join("skills");
        std::fs::create_dir(&skills_dir).unwrap();
        let link = skills_dir.join("linked.md");
        std::os::unix::fs::symlink(&external, &link).unwrap();

        let skills = SkillsLoader::load_from_directory(&skills_dir).unwrap();
        assert!(
            skills.is_empty(),
            "symlink should be skipped, got: {:?}",
            skills.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    /// A README.md (or other doc file) in the skills root must not be
    /// treated as a flat skill — it should be skipped silently, while
    /// genuine flat skills alongside it still load.
    #[test]
    fn load_from_directory_skips_readme_and_doc_files() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().to_path_buf();

        std::fs::write(
            skills_dir.join("README.md"),
            "# Skills\n\nThis directory holds skills.\n",
        )
        .unwrap();
        std::fs::write(
            skills_dir.join("CHANGELOG.md"),
            "# Changelog\n\n- stuff\n",
        )
        .unwrap();
        std::fs::write(
            skills_dir.join("real-skill.md"),
            "---\nname: real-skill\ndescription: ok\n---\nbody\n",
        )
        .unwrap();

        let skills = SkillsLoader::load_from_directory(&skills_dir).unwrap();
        let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["real-skill"],
            "only the genuine flat skill should load; doc files skipped"
        );
    }

    /// load_from_directory must skip files exceeding the size cap.
    #[test]
    fn load_from_directory_skips_oversized_files() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().to_path_buf();

        // Create a "skill" file that exceeds the cap. We don't actually
        // need a full MAX_SKILL_FILE_BYTES worth — we just need to make
        // sure oversized files are filtered. Use a custom override by
        // mocking via a ~1 MB + 1 file. That's slow; we use a smaller
        // cap via a helper-scoped test.
        let huge = skills_dir.join("huge.md");
        // 1 MB + 1 byte — exceeds the 1 MiB cap by 1025 bytes.
        let content = format!("---\nname: x\n---\n{}", "a".repeat(1_048_600));
        std::fs::write(&huge, content).unwrap();

        // Also add a normal skill to confirm the non-oversized one is loaded.
        let small = skills_dir.join("small.md");
        std::fs::write(
            &small,
            "---\nname: small\ndescription: ok\n---\nbody\n",
        )
        .unwrap();

        let skills = SkillsLoader::load_from_directory(&skills_dir).unwrap();
        let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"small"), "small skill missing");
        assert!(
            !names.contains(&"x"),
            "oversized skill should have been skipped"
        );
    }

    /// load_from_directory must return the successful skills even when
    /// one file is unparseable — a single bad file must not block the
    /// whole directory from loading.
    #[test]
    fn load_from_directory_skips_one_bad_file_but_keeps_others() {
        let tmp = tempfile::tempdir().unwrap();
        let skills_dir = tmp.path().to_path_buf();

        std::fs::write(
            skills_dir.join("good.md"),
            "---\nname: good\ndescription: ok\n---\nbody\n",
        )
        .unwrap();
        std::fs::write(
            skills_dir.join("broken.md"),
            "not a yaml file at all, no fences",
        )
        .unwrap();

        let skills = SkillsLoader::load_from_directory(&skills_dir).unwrap();
        let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["good"]);
    }

    // -- Directory-format walker ------------------------------------

    /// `<root>/foo/SKILL.md` should load as a directory skill.
    #[test]
    fn loads_directory_skill_at_top_level() {
        let tmp = tempfile::tempdir().unwrap();
        let foo = tmp.path().join("foo");
        std::fs::create_dir(&foo).unwrap();
        std::fs::write(
            foo.join("SKILL.md"),
            "---\nname: foo\ndescription: ok\n---\nbody\n",
        )
        .unwrap();

        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        assert_eq!(skills.len(), 1);
        let s = &skills[0];
        assert_eq!(s.name, "foo");
        match s.layout.as_ref().unwrap() {
            SkillLayout::Directory { dir, category } => {
                assert_eq!(dir, &foo);
                assert!(category.is_none());
            }
            _ => panic!("expected directory layout, got {:?}", s.layout),
        }
    }

    /// `<root>/<category>/<name>/SKILL.md` should load with category set.
    #[test]
    fn loads_directory_skill_in_category() {
        let tmp = tempfile::tempdir().unwrap();
        let cat = tmp.path().join("productivity");
        let foo = cat.join("foo");
        std::fs::create_dir_all(&foo).unwrap();
        std::fs::write(
            foo.join("SKILL.md"),
            "---\nname: foo\ndescription: ok\n---\nbody\n",
        )
        .unwrap();

        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        assert_eq!(skills.len(), 1);
        let s = &skills[0];
        assert_eq!(s.name, "foo");
        assert_eq!(s.layout.as_ref().unwrap().category(), Some("productivity"));
    }

    /// Directory skill with `references/` and `scripts/` should still
    /// load — only SKILL.md is read, supporting files don't break it.
    #[test]
    fn directory_skill_with_subfiles_loads() {
        let tmp = tempfile::tempdir().unwrap();
        let foo = tmp.path().join("foo");
        std::fs::create_dir_all(foo.join("references")).unwrap();
        std::fs::create_dir_all(foo.join("scripts")).unwrap();
        std::fs::write(
            foo.join("SKILL.md"),
            "---\nname: foo\ndescription: ok\n---\nbody\n",
        )
        .unwrap();
        std::fs::write(foo.join("references").join("api.md"), "ref body").unwrap();
        std::fs::write(foo.join("scripts").join("run.sh"), "#!/bin/sh\n").unwrap();

        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "foo");
    }

    /// Mixed flat + directory format in the same root must both load.
    #[test]
    fn mixed_flat_and_directory_load_together() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("flat-one.md"),
            "---\nname: flat-one\ndescription: ok\n---\nbody\n",
        )
        .unwrap();
        let dir = tmp.path().join("dir-one");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\nname: dir-one\ndescription: ok\n---\nbody\n",
        )
        .unwrap();

        let mut names: Vec<_> = SkillsLoader::load_from_directory(tmp.path())
            .unwrap()
            .into_iter()
            .map(|s| s.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["dir-one", "flat-one"]);
    }

    /// `.archive/`, `.hub/`, `.usage.json`, and `.bundled_manifest`
    /// must never be walked.
    #[test]
    fn reserved_dir_entries_are_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        // Plant a "skill" inside .archive/ — it should be ignored.
        let arch = tmp.path().join(".archive").join("ghost");
        std::fs::create_dir_all(&arch).unwrap();
        std::fs::write(
            arch.join("SKILL.md"),
            "---\nname: ghost\ndescription: ok\n---\nbody\n",
        )
        .unwrap();
        // Same for .hub/.
        let hub = tmp.path().join(".hub").join("alsoghost");
        std::fs::create_dir_all(&hub).unwrap();
        std::fs::write(
            hub.join("SKILL.md"),
            "---\nname: alsoghost\ndescription: ok\n---\nbody\n",
        )
        .unwrap();
        // .usage.json and .bundled_manifest are at the root.
        std::fs::write(tmp.path().join(".usage.json"), "{}").unwrap();
        std::fs::write(tmp.path().join(".bundled_manifest"), "").unwrap();

        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        let names: Vec<_> = skills.iter().map(|s| s.name.as_str()).collect();
        assert!(!names.contains(&"ghost"), "archived skill leaked");
        assert!(!names.contains(&"alsoghost"), "hub skill leaked");
    }

    /// A directory skill whose SKILL.md `name:` does not match the
    /// directory name must be rejected — silently renaming a directory
    /// outside the tool would otherwise produce a phantom mismatch.
    #[test]
    fn directory_skill_name_mismatch_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("expected-name");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\nname: different-name\ndescription: x\n---\nbody\n",
        )
        .unwrap();

        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        assert!(skills.is_empty(), "expected mismatch to be skipped");
    }

    /// Categories don't nest. `<root>/cat1/cat2/foo/SKILL.md` is
    /// rejected: the loader walks at most one category level.
    #[test]
    fn nested_categories_are_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("cat1").join("cat2").join("foo");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(
            nested.join("SKILL.md"),
            "---\nname: foo\ndescription: x\n---\nbody\n",
        )
        .unwrap();

        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        assert!(
            skills.is_empty(),
            "double-nested skill should be skipped, got {:?}",
            skills.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }

    /// A directory under <root> that has no SKILL.md is treated as a
    /// category folder. If it contains no skills either, the load is
    /// silent (no warning, no panic).
    #[test]
    fn empty_category_folder_is_silently_empty() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("empty-cat")).unwrap();
        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        assert!(skills.is_empty());
    }

    /// A directory-format skill whose dir name fails validate_skill_name
    /// (e.g. uppercase) must be skipped.
    #[test]
    fn directory_skill_invalid_dirname_is_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let bad = tmp.path().join("Foo"); // uppercase
        std::fs::create_dir(&bad).unwrap();
        std::fs::write(
            bad.join("SKILL.md"),
            "---\nname: Foo\ndescription: x\n---\nbody\n",
        )
        .unwrap();
        let skills = SkillsLoader::load_from_directory(tmp.path()).unwrap();
        assert!(skills.is_empty());
    }

    /// load_with_provenance assigns Bundled provenance to skills listed
    /// in the bundled manifest, and AgentCreated to others.
    #[test]
    fn provenance_populated_from_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("alpha.md"),
            "---\nname: alpha\ndescription: a\n---\nbody\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("beta.md"),
            "---\nname: beta\ndescription: b\n---\nbody\n",
        )
        .unwrap();

        let mut bundled = BundledManifest::default();
        bundled.set("alpha", "abc");
        let hub = HubLock::default();

        let skills =
            SkillsLoader::load_with_provenance(tmp.path(), Some(&bundled), Some(&hub), None)
                .unwrap();
        let by_name: std::collections::HashMap<_, _> =
            skills.iter().map(|s| (s.name.as_str(), s)).collect();
        assert_eq!(
            by_name.get("alpha").unwrap().provenance,
            SkillProvenance::Bundled
        );
        assert_eq!(
            by_name.get("beta").unwrap().provenance,
            SkillProvenance::AgentCreated
        );
    }

    /// Hub provenance overrides bundled when both manifests claim the
    /// same name (the user's install action takes precedence).
    #[test]
    fn hub_provenance_overrides_bundled() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("conflict.md"),
            "---\nname: conflict\ndescription: c\n---\nbody\n",
        )
        .unwrap();
        let mut bundled = BundledManifest::default();
        bundled.set("conflict", "abc");
        let mut hub = HubLock::default();
        hub.installed
            .insert("conflict".into(), serde_json::json!({"source": "github"}));

        let skills =
            SkillsLoader::load_with_provenance(tmp.path(), Some(&bundled), Some(&hub), None)
                .unwrap();
        assert_eq!(skills[0].provenance, SkillProvenance::HubInstalled);
    }

    /// load_with_provenance pulls state and pinned from the usage
    /// sidecar when an UsageStore is provided.
    #[test]
    fn state_and_pinned_pulled_from_usage_store() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("alpha.md"),
            "---\nname: alpha\ndescription: a\n---\nbody\n",
        )
        .unwrap();

        let store = UsageStore::open(tmp.path());
        store.bump_use("alpha");
        store.set_state("alpha", SkillState::Stale);
        store.set_pinned("alpha", true);

        let skills =
            SkillsLoader::load_with_provenance(tmp.path(), None, None, Some(&store)).unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].state, SkillState::Stale);
        assert!(skills[0].pinned);
    }

    /// Symlinked subdirectories must be skipped (data-exfiltration
    /// hardening at the directory level, not just the file level).
    #[cfg(unix)]
    #[test]
    fn symlinked_directory_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        // Real skill dir outside the skills root.
        let external = tmp.path().join("real-skill");
        std::fs::create_dir(&external).unwrap();
        std::fs::write(
            external.join("SKILL.md"),
            "---\nname: leak\ndescription: x\n---\nbody\n",
        )
        .unwrap();

        let skills_root = tmp.path().join("skills");
        std::fs::create_dir(&skills_root).unwrap();
        let link = skills_root.join("victim");
        std::os::unix::fs::symlink(&external, &link).unwrap();

        let skills = SkillsLoader::load_from_directory(&skills_root).unwrap();
        assert!(
            skills.is_empty(),
            "symlinked skill directory should not load, got {:?}",
            skills.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }
}
