//! Skill loading + prompt assembly for cron-fired jobs.
//!
//! Mirrors the upstream's `_build_job_prompt` skill block:
//! - Each name in the job's `skills` list resolves to a `Skill` under
//!   `<skills_dir>/`. Found content gets prepended to the prompt with the
//!   `[IMPORTANT: The user has invoked the "X" skill]` header. Missing
//!   skills surface a `'⚠️ Skill(s) not found and skipped'` notice the
//!   agent is told to repeat to the user.
//! - Successful loads bump the skill's usage counter so the curator
//!   sees the skill as actively used.
//! - The legacy single `skill` field is rolled into the canonical
//!   ordered list alongside `skills` via [`canonical_skills`].

use std::path::{Path, PathBuf};

use crate::skills::loader::SkillsLoader;
use crate::skills::usage::UsageStore;

/// Default skills directory derived from a `jobs.json` path. Sibling of
/// the jobs file (so a single Fennec home has a single skills tree),
/// matching the upstream's `~/.hermes/skills/` convention.
pub fn default_skills_dir_for(jobs_path: &Path) -> PathBuf {
    jobs_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("skills")
}

/// Normalise the legacy single `skill` field + the plural `skills`
/// list into a single ordered, deduplicated list. Empty / whitespace
/// entries are dropped. Mirrors the upstream's `_normalize_skill_list`
/// / `_canonical_skills`.
pub fn canonical_skills(legacy: Option<&str>, skills: Option<&[String]>) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    let push_unique = |bucket: &mut Vec<String>, raw: &str| {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && !bucket.iter().any(|n| n == trimmed) {
            bucket.push(trimmed.to_string());
        }
    };
    if let Some(s) = legacy {
        push_unique(&mut result, s);
    }
    if let Some(list) = skills {
        for s in list {
            push_unique(&mut result, s);
        }
    }
    result
}

/// Result of [`inject_skills`]: the assembled prompt + the lists of
/// loaded and skipped (not-found) skill names. The caller logs / surfaces
/// the skipped list; the assembled prompt already contains the
/// user-facing notice when anything was skipped.
#[derive(Debug, Clone, Default)]
pub struct InjectedPrompt {
    pub assembled: String,
    pub loaded: Vec<String>,
    pub skipped: Vec<String>,
}

/// Assemble the cron prompt by loading each named skill's content and
/// prepending it with the upstream's `[IMPORTANT: ...]` header. Missing
/// skills surface a `'⚠️ Skill(s) not found and skipped'` notice the
/// agent is told to repeat. Successful loads bump the skill's usage
/// counter via [`UsageStore::bump_use`].
///
/// On loader failure (skills dir unreadable), returns the original
/// prompt unchanged with an empty `loaded` list and every requested
/// name in `skipped` — best-effort, never blocks scheduling.
pub fn inject_skills(prompt: &str, skill_names: &[String], skills_dir: &Path) -> InjectedPrompt {
    if skill_names.is_empty() {
        return InjectedPrompt {
            assembled: prompt.to_string(),
            ..Default::default()
        };
    }

    let all_skills = match SkillsLoader::load_from_directory(skills_dir) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                "Cron skill injection: failed to load skills dir {}: {}",
                skills_dir.display(),
                e
            );
            return InjectedPrompt {
                assembled: prompt.to_string(),
                loaded: Vec::new(),
                skipped: skill_names.to_vec(),
            };
        }
    };
    let usage_store = UsageStore::open(skills_dir);

    let mut parts: Vec<String> = Vec::new();
    let mut loaded: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for skill_name in skill_names {
        match all_skills.iter().find(|s| s.name == *skill_name) {
            Some(skill) => {
                usage_store.bump_use(skill_name);
                if !parts.is_empty() {
                    parts.push(String::new());
                }
                parts.push(format!(
                    "[IMPORTANT: The user has invoked the \"{skill_name}\" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]"
                ));
                parts.push(String::new());
                parts.push(skill.content.trim().to_string());
                loaded.push(skill_name.clone());
            }
            None => {
                tracing::warn!(
                    "Cron skill injection: skill '{}' not found in {}, skipping",
                    skill_name,
                    skills_dir.display()
                );
                skipped.push(skill_name.clone());
            }
        }
    }

    if !skipped.is_empty() {
        let joined = skipped.join(", ");
        let notice = format!(
            "[IMPORTANT: The following skill(s) were listed for this job but could not be found and were skipped: {joined}. Start your response with a brief notice so the user is aware, e.g.: '⚠️ Skill(s) not found and skipped: {joined}']"
        );
        parts.insert(0, notice);
    }

    if !prompt.trim().is_empty() {
        parts.push(String::new());
        parts.push(format!(
            "The user has provided the following instruction alongside the skill invocation: {prompt}"
        ));
    }

    InjectedPrompt {
        assembled: parts.join("\n"),
        loaded,
        skipped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_dedupes_and_preserves_order() {
        // Legacy single + plural list with overlap + an empty entry.
        let out = canonical_skills(
            Some("alpha"),
            Some(&[
                "alpha".to_string(),
                "  beta  ".to_string(),
                String::new(),
                "gamma".to_string(),
                "beta".to_string(),
            ]),
        );
        assert_eq!(out, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn canonical_empty_inputs_yield_empty_list() {
        assert!(canonical_skills(None, None).is_empty());
        assert!(canonical_skills(Some(""), Some(&[String::new()])).is_empty());
    }

    #[test]
    fn default_skills_dir_is_sibling_of_jobs_path() {
        let jobs = Path::new("/home/me/.fennec/cron_jobs.json");
        assert_eq!(
            default_skills_dir_for(jobs),
            Path::new("/home/me/.fennec/skills")
        );
    }

    fn write_skill(dir: &Path, name: &str, content: &str) {
        let body = format!(
            "---\nname: {name}\ndescription: test\n---\n\n{content}\n"
        );
        std::fs::write(dir.join(format!("{name}.md")), body).unwrap();
    }

    #[test]
    fn inject_returns_prompt_unchanged_when_no_skills() {
        let tmp = tempfile::tempdir().unwrap();
        let out = inject_skills("hello world", &[], tmp.path());
        assert_eq!(out.assembled, "hello world");
        assert!(out.loaded.is_empty());
        assert!(out.skipped.is_empty());
    }

    #[test]
    fn inject_prepends_skill_content_and_appends_user_instruction() {
        let tmp = tempfile::tempdir().unwrap();
        let skills = tmp.path().join("skills");
        std::fs::create_dir_all(&skills).unwrap();
        write_skill(&skills, "playbook", "Do step 1, then step 2.");

        let out = inject_skills(
            "Investigate the alert.",
            &["playbook".to_string()],
            &skills,
        );
        assert_eq!(out.loaded, vec!["playbook"]);
        assert!(out.skipped.is_empty());
        assert!(
            out.assembled.contains("[IMPORTANT: The user has invoked the \"playbook\" skill"),
            "{}",
            out.assembled
        );
        assert!(out.assembled.contains("Do step 1, then step 2."));
        assert!(out.assembled.contains(
            "The user has provided the following instruction alongside the skill invocation: Investigate the alert."
        ));
    }

    #[test]
    fn inject_emits_notice_for_missing_skills() {
        let tmp = tempfile::tempdir().unwrap();
        let skills = tmp.path().join("skills");
        std::fs::create_dir_all(&skills).unwrap();
        write_skill(&skills, "real", "real body");

        let out = inject_skills(
            "prompt",
            &["real".to_string(), "ghost".to_string()],
            &skills,
        );
        assert_eq!(out.loaded, vec!["real"]);
        assert_eq!(out.skipped, vec!["ghost"]);
        assert!(out.assembled.contains(
            "Skill(s) not found and skipped: ghost"
        ));
        // Notice appears BEFORE the loaded skill content.
        let notice_at = out.assembled.find("skipped: ghost").unwrap();
        let body_at = out.assembled.find("real body").unwrap();
        assert!(notice_at < body_at);
    }

    #[test]
    fn inject_bumps_usage_for_loaded_skills() {
        let tmp = tempfile::tempdir().unwrap();
        let skills = tmp.path().join("skills");
        std::fs::create_dir_all(&skills).unwrap();
        write_skill(&skills, "tracked", "body");

        let before = UsageStore::open(&skills).get("tracked");
        let pre_count = before.map(|r| r.use_count).unwrap_or(0);

        let _ = inject_skills("p", &["tracked".to_string()], &skills);

        let after = UsageStore::open(&skills).get("tracked").unwrap();
        assert!(
            after.use_count > pre_count,
            "use_count should bump (was {pre_count}, now {})",
            after.use_count
        );
    }

    #[test]
    fn inject_reports_all_skipped_when_no_skills_match() {
        // A skills/ directory that exists but contains none of the
        // requested skills — every name lands in `skipped` and the
        // assembled prompt carries the standard not-found notice.
        let tmp = tempfile::tempdir().unwrap();
        let skills = tmp.path().join("skills");
        std::fs::create_dir_all(&skills).unwrap();
        let out = inject_skills("p", &["x".to_string(), "y".to_string()], &skills);
        assert!(out.loaded.is_empty());
        assert_eq!(out.skipped, vec!["x", "y"]);
        assert!(out.assembled.contains("Skill(s) not found and skipped: x, y"));
        assert!(out.assembled.contains(
            "The user has provided the following instruction alongside the skill invocation: p"
        ));
    }
}
