use std::path::Path;

use anyhow::{Context, Result};

/// A skill loaded from a markdown file with YAML frontmatter.
#[derive(Debug, Clone)]
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
}

/// Loads and manages skills from disk.
pub struct SkillsLoader;

impl SkillsLoader {
    /// Load all `*.md` skill files from the given directory.
    ///
    /// Each file is expected to have YAML frontmatter between `---` fences
    /// followed by markdown content.
    pub fn load_from_directory(path: &Path) -> Result<Vec<Skill>> {
        let mut skills = Vec::new();

        if !path.is_dir() {
            return Ok(skills);
        }

        // A non-existent skills directory is the normal state for a
        // fresh install before the user has added any skills. Treat it
        // as "no skills loaded" rather than a hard error — otherwise
        // the agent fails to start with `~/.fennec/skills not found`,
        // which is a worse default than starting with an empty skill
        // catalog. We still propagate other I/O errors (permissions,
        // I/O failures) so a misconfigured environment is loud.
        let entries = match std::fs::read_dir(path) {
            Ok(e) => e,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!(
                    "Skills directory {} does not exist; starting with no skills",
                    path.display()
                );
                return Ok(skills);
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("reading skills directory: {}", path.display())
                });
            }
        };

        for entry in entries {
            let entry = entry?;
            let file_path = entry.path();

            if file_path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }

            let raw = std::fs::read_to_string(&file_path)
                .with_context(|| format!("reading skill file: {}", file_path.display()))?;

            match Self::parse_skill(&raw) {
                Ok(skill) => skills.push(skill),
                Err(e) => {
                    tracing::warn!(
                        "Skipping invalid skill file {}: {}",
                        file_path.display(),
                        e
                    );
                }
            }
        }

        Ok(skills)
    }

    /// Parse a single skill from its raw markdown+frontmatter text.
    fn parse_skill(raw: &str) -> Result<Skill> {
        let trimmed = raw.trim_start();

        anyhow::ensure!(
            trimmed.starts_with("---"),
            "skill file must start with YAML frontmatter (---)"
        );

        // Find the closing `---` after the opening one.
        let after_open = &trimmed[3..];
        let close_pos = after_open
            .find("\n---")
            .context("missing closing --- for frontmatter")?;

        let frontmatter = &after_open[..close_pos];
        // +4 to skip past the "\n---" itself, then skip the newline after it.
        let body_start = 3 + close_pos + 4; // 3 for opening ---, 4 for \n---
        let content = trimmed[body_start..].trim().to_string();

        // Parse frontmatter key-value pairs with simple line-by-line parsing.
        let mut name = String::new();
        let mut description = String::new();
        let mut always = false;
        let mut requirements = Vec::new();
        let mut in_requirements = false;

        for line in frontmatter.lines() {
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // Check for list items under requirements.
            if in_requirements {
                if let Some(item) = line.strip_prefix("- ") {
                    requirements.push(item.trim().to_string());
                    continue;
                } else if !line.contains(':') {
                    // Continuation of list without dash? Skip.
                    continue;
                } else {
                    in_requirements = false;
                }
            }

            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim();
                let value = value.trim();

                match key {
                    "name" => name = value.to_string(),
                    "description" => description = value.to_string(),
                    "always" => always = value == "true",
                    "requirements" => {
                        if value.is_empty() {
                            in_requirements = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        anyhow::ensure!(!name.is_empty(), "skill must have a name");

        Ok(Skill {
            name,
            description,
            content,
            always,
            requirements,
        })
    }

    /// Filter skills to those whose requirements are all satisfied
    /// (commands exist in PATH).
    pub fn filter_available(skills: &[Skill]) -> Vec<&Skill> {
        skills
            .iter()
            .filter(|skill| {
                skill.requirements.iter().all(|req| {
                    which_exists(req)
                })
            })
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
                sections.push(format!(
                    "### {}\n{}\n",
                    skill.name, skill.content
                ));
            }
        }

        // On-demand skills: list with descriptions.
        let on_demand: Vec<&Skill> = skills.iter().filter(|s| !s.always).collect();
        if !on_demand.is_empty() {
            sections.push("## Available Skills\nUse the `load_skill` tool to activate any of these:\n".to_string());
            for skill in &on_demand {
                sections.push(format!(
                    "- **{}**: {}\n",
                    skill.name, skill.description
                ));
            }
        }

        sections.join("")
    }
}

/// Check whether a command exists in PATH.
fn which_exists(command: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                dir.join(command).is_file()
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

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
            },
            Skill {
                name: "optional-skill".to_string(),
                description: "Load on demand".to_string(),
                content: "Optional content.".to_string(),
                always: false,
                requirements: vec![],
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
}
