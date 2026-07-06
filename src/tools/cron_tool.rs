use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::json;

use crate::cron::jobs::{
    compute_next_run, parse_schedule_kind, schedule_display_for, CronJob, JobStore, RepeatConfig,
};
use crate::cron::output::{cleanup_job_output, default_output_dir_for};

use super::traits::{Tool, ToolResult};

/// Backwards-compatible alias for [`crate::bus::TurnOrigin`].
///
/// Originally defined here when only the cron tool needed this concept;
/// now `ask_user_tool` and `send_message_tool` also need to know which
/// `(channel, chat_id)` triggered the current turn, so the canonical
/// definition has moved to `bus::turn_context`. Re-exported here so
/// existing downstream callers keep compiling unchanged.
pub type CronOrigin = crate::bus::TurnOrigin;

/// LLM-callable tool for creating, listing, and removing scheduled tasks.
pub struct CronTool {
    store_path: PathBuf,
    default_origin: Arc<Mutex<Option<CronOrigin>>>,
}

impl CronTool {
    /// Create a new `CronTool`.
    ///
    /// * `store_path` - path to the JSON file backing the job store.
    /// * `default_origin` - shared origin that the gateway sets before each
    ///   agent turn so the tool knows where to route cron results.
    pub fn new(store_path: PathBuf, default_origin: Arc<Mutex<Option<CronOrigin>>>) -> Self {
        Self {
            store_path,
            default_origin,
        }
    }

    /// Get a clone of the shared origin arc (for the gateway to hold).
    pub fn origin_handle(&self) -> Arc<Mutex<Option<CronOrigin>>> {
        Arc::clone(&self.default_origin)
    }

    /// Load the job store from disk.
    fn load_store(&self) -> Result<JobStore> {
        let mut store = JobStore::new(self.store_path.clone());
        store.load()?;
        Ok(store)
    }

    fn execute_create(&self, prompt: Option<&str>, schedule: Option<&str>) -> Result<ToolResult> {
        let prompt = match prompt {
            Some(p) if !p.is_empty() => p,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: prompt".to_string()),
                });
            }
        };

        let schedule_raw = match schedule {
            Some(s) if !s.is_empty() => s,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: schedule".to_string()),
                });
            }
        };

        // No auto-`every`-prefix: bare durations are one-shot, `every X`
        // is recurring, `0 9 * * *` is a cron expression, and an ISO
        // timestamp is a one-shot at that moment. Matches the reference
        // agent's semantics so prompts written for either work the same.
        let schedule_str = schedule_raw.trim().to_string();
        if parse_schedule_kind(&schedule_str).is_none() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "invalid schedule '{}'. Use:\n  - Duration (one-shot):    '30m', '2h', '1d'\n  - Interval (recurring):   'every 30m', 'every 2h'\n  - Cron expression:        '0 9 * * 1-5', '*/15 * * * *'\n  - Timestamp (one-shot):   '2026-02-03T14:00:00'",
                    schedule_raw
                )),
            });
        }

        // Read origin from shared state.
        //
        // `unwrap_or_else(|p| p.into_inner())` recovers from a poisoned
        // mutex: the previous `.lock().unwrap()` would panic for every
        // future cron call after any single panic-while-locked. The
        // locked region here is just a `clone()` of an `Option<CronOrigin>`
        // so it can't itself panic — but a panic from an unrelated
        // holder (gateway agent loop sets the origin in main.rs:953)
        // would cascade. Recover the inner data and continue.
        let origin = self
            .default_origin
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();

        // Full UUID v4 — the previous 8-hex-char truncation produces
        // birthday collisions at ~64K jobs and shows up as
        // "job not found" failures for whichever colliding entry got
        // overwritten. Job IDs are LLM-facing identifiers, not
        // human-typed: full length is fine.
        let job_id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();

        // Compute the next-run timestamp via the shared helper so this
        // path matches the scheduler's own semantics exactly (cron
        // expressions ask croner; one-shots return their delay or
        // timestamp; recurring returns now + interval). `last_run=None`
        // because the job has never fired yet.
        let next_run_str = compute_next_run(&schedule_str, None).unwrap_or_else(|| {
            // Schedule validated above, so this branch is unreachable
            // in practice — keep a safe fallback so the tool never panics.
            now.to_rfc3339()
        });
        let next_run_dt =
            chrono::DateTime::parse_from_rfc3339(&next_run_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or(now);
        let job = CronJob {
            id: job_id.clone(),
            name: prompt.chars().take(60).collect::<String>(),
            schedule: schedule_str.clone(),
            command: prompt.to_string(),
            enabled: true,
            // No anchor — last_run is set by `mark_job_run` after the
            // first real fire. next_run_at carries the scheduling state.
            last_run: None,
            origin_channel: origin.as_ref().map(|o| o.channel.clone()),
            origin_chat_id: origin.as_ref().map(|o| o.chat_id.clone()),
            state: "scheduled".to_string(),
            created_at: Some(now.to_rfc3339()),
            next_run_at: Some(next_run_str.clone()),
            last_status: None,
            last_error: None,
            last_delivery_error: None,
            paused_at: None,
            paused_reason: None,
            repeat: RepeatConfig::default(),
            schedule_display: schedule_display_for(&schedule_str),
            // Per-job script / no_agent / context_from / model /
            // provider / base_url / enabled_toolsets / workdir /
            // profile are agent-facing parameters surfaced in the
            // cron_tool expansion PR; for now jobs created via this
            // path default to plain agent-prompt execution.
            // Hand-edited jobs.json that sets any of these loads
            // correctly because of the serde defaults on CronJob.
            script: None,
            no_agent: false,
            context_from: None,
            model: None,
            provider: None,
            base_url: None,
            enabled_toolsets: None,
            workdir: None,
            profile: None,
        };

        let mut store = self.load_store()?;
        store.add_job(job);
        store.save()?;

        let output = format!(
            "Scheduled job created.\n  ID: {}\n  Schedule: {}\n  Next run: {}\n  Prompt: {}",
            job_id,
            schedule_str,
            next_run_dt.format("%Y-%m-%d %H:%M:%S UTC"),
            prompt,
        );

        Ok(ToolResult {
            success: true,
            output,
            error: None,
        })
    }

    fn execute_list(&self) -> Result<ToolResult> {
        let store = self.load_store()?;
        let jobs = store.list_jobs();

        if jobs.is_empty() {
            return Ok(ToolResult {
                success: true,
                output: "No scheduled jobs.".to_string(),
                error: None,
            });
        }

        let mut lines = Vec::new();
        for job in jobs {
            let status = if job.enabled { "enabled" } else { "disabled" };
            let origin = match (&job.origin_channel, &job.origin_chat_id) {
                (Some(ch), Some(cid)) => format!(" -> {}:{}", ch, cid),
                _ => String::new(),
            };
            lines.push(format!(
                "- [{}] {} | {} | {}{}\n  Prompt: {}",
                job.id, status, job.schedule, job.name, origin, job.command,
            ));
        }

        Ok(ToolResult {
            success: true,
            output: lines.join("\n"),
            error: None,
        })
    }

    fn execute_remove(&self, job_id: Option<&str>) -> Result<ToolResult> {
        let job_id = match job_id {
            Some(id) if !id.is_empty() => id,
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: job_id".to_string()),
                });
            }
        };

        let mut store = self.load_store()?;
        // Accept either an ID or a name. The store handles the
        // ambiguity case; we surface a clear error if a name matches
        // multiple jobs so the operator can pick one by ID.
        let resolved_id = match store.resolve_job_ref(job_id) {
            Ok(Some(j)) => j.id.clone(),
            Ok(None) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Job '{}' not found.", job_id)),
                });
            }
            Err(ambiguity) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(ambiguity.to_string()),
                });
            }
        };

        if store.remove_job(&resolved_id) {
            store.save()?;
            // Orphaned output dir cleanup — matches the upstream's
            // `remove_job` shutil.rmtree on the per-job output dir.
            // Best-effort; cleanup errors are logged inside.
            let output_dir = default_output_dir_for(&self.store_path);
            cleanup_job_output(&output_dir, &resolved_id);
            Ok(ToolResult {
                success: true,
                output: format!("Job '{}' removed.", resolved_id),
                error: None,
            })
        } else {
            Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Job '{}' not found.", job_id)),
            })
        }
    }
}

#[async_trait]
impl Tool for CronTool {
    fn name(&self) -> &str {
        "cronjob"
    }

    fn description(&self) -> &str {
        "Create, list, or remove scheduled tasks. Use this when the user asks you to remind them, schedule something, or do something later."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "remove"],
                    "description": "The action to perform"
                },
                "prompt": {
                    "type": "string",
                    "description": "What to do when the job fires (for create)"
                },
                "schedule": {
                    "type": "string",
                    "description": "When to fire (for create). Four formats: (1) bare duration like '30m', '2h', '1d' — one-shot, fires once after the delay; (2) 'every <duration>' like 'every 30m', 'every 2h' — recurring; (3) cron expression like '0 9 * * 1-5' (weekdays at 9am) or '*/15 * * * *' (every 15 min) — standard 5- or 6-field syntax, recurring; (4) ISO 8601 timestamp like '2026-02-03T14:00:00' — one-shot at that specific moment."
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID to remove (for remove)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let action = match args.get("action").and_then(|v| v.as_str()) {
            Some(a) => a,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: action".to_string()),
                });
            }
        };

        match action {
            "create" => {
                let prompt = args.get("prompt").and_then(|v| v.as_str());
                let schedule = args.get("schedule").and_then(|v| v.as_str());
                self.execute_create(prompt, schedule)
            }
            "list" => self.execute_list(),
            "remove" => {
                let job_id = args.get("job_id").and_then(|v| v.as_str());
                self.execute_remove(job_id)
            }
            other => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "unknown action '{}'. Use 'create', 'list', or 'remove'.",
                    other
                )),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tool(dir: &TempDir) -> CronTool {
        let path = dir.path().join("cron_jobs.json");
        let origin = Arc::new(Mutex::new(Some(CronOrigin {
            channel: "telegram".to_string(),
            chat_id: "12345".to_string(),
        })));
        CronTool::new(path, origin)
    }

    #[tokio::test]
    async fn test_create_and_list() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "Remind me to drink water",
                "schedule": "every 30m"
            }))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("Scheduled job created"));

        let result = tool.execute(json!({"action": "list"})).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("Remind me to drink water"));
        assert!(result.output.contains("telegram:12345"));
    }

    #[tokio::test]
    async fn test_create_bare_duration_is_one_shot() {
        // Bare durations like "5m" are one-shot (fires once after the
        // delay) — matching the upstream's semantics. The tool no
        // longer auto-prepends "every", so a user writing "5m" gets a
        // one-shot, not a recurring job.
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "Check the oven",
                "schedule": "5m"
            }))
            .await
            .unwrap();
        assert!(result.success, "create failed: {:?}", result.error);
        assert!(
            result.output.contains("Schedule: 5m"),
            "expected verbatim '5m' in output, got: {}",
            result.output
        );
        assert!(
            !result.output.contains("every 5m"),
            "should not auto-prepend 'every': {}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_create_recurring_interval() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "Hourly status check",
                "schedule": "every 1h"
            }))
            .await
            .unwrap();
        assert!(result.success, "create failed: {:?}", result.error);
        assert!(result.output.contains("every 1h"));
    }

    #[tokio::test]
    async fn test_create_cron_expression() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "Weekday standup reminder",
                "schedule": "0 9 * * 1-5"
            }))
            .await
            .unwrap();
        assert!(result.success, "create failed: {:?}", result.error);
        assert!(
            result.output.contains("0 9 * * 1-5"),
            "expected cron expression in output, got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_create_iso_timestamp() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "Year-end review",
                "schedule": "2099-12-31T23:59:00Z"
            }))
            .await
            .unwrap();
        assert!(result.success, "create failed: {:?}", result.error);
        assert!(
            result.output.contains("2099-12-31T23:59:00Z"),
            "expected timestamp in output, got: {}",
            result.output
        );
    }

    #[tokio::test]
    async fn test_create_rejects_invalid_schedule() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "Bad schedule",
                "schedule": "not a real schedule"
            }))
            .await
            .unwrap();
        assert!(!result.success);
        let err = result.error.unwrap_or_default();
        assert!(err.contains("Cron expression"), "error should list all formats: {}", err);
        assert!(err.contains("Timestamp"), "error should list timestamp format: {}", err);
    }

    #[tokio::test]
    async fn test_create_and_remove() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "Do something",
                "schedule": "1h"
            }))
            .await
            .unwrap();
        assert!(result.success);

        // Extract job ID from output.
        let id_line = result
            .output
            .lines()
            .find(|l| l.contains("ID:"))
            .unwrap();
        let job_id = id_line.split("ID:").nth(1).unwrap().trim();

        let result = tool
            .execute(json!({"action": "remove", "job_id": job_id}))
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.output.contains("removed"));

        // List should be empty now.
        let result = tool.execute(json!({"action": "list"})).await.unwrap();
        assert!(result.output.contains("No scheduled jobs"));
    }

    #[tokio::test]
    async fn test_remove_nonexistent() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({"action": "remove", "job_id": "nope"}))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not found"));
    }

    #[tokio::test]
    async fn test_invalid_schedule() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool
            .execute(json!({
                "action": "create",
                "prompt": "test",
                "schedule": "whenever"
            }))
            .await
            .unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("invalid schedule"));
    }

    #[tokio::test]
    async fn test_missing_action() {
        let dir = TempDir::new().unwrap();
        let tool = make_tool(&dir);

        let result = tool.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("action"));
    }
}
