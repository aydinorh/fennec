use std::path::PathBuf;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use serde_json::json;

use crate::cron::jobs::{parse_schedule_kind, CronJob, JobStore, ScheduleKind};

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
        let kind = match parse_schedule_kind(&schedule_str) {
            Some(k) => k,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "invalid schedule '{}'. Use:\n  - Duration (one-shot):    '30m', '2h', '1d'\n  - Interval (recurring):   'every 30m', 'every 2h'\n  - Cron expression:        '0 9 * * 1-5', '*/15 * * * *'\n  - Timestamp (one-shot):   '2026-02-03T14:00:00'",
                        schedule_raw
                    )),
                });
            }
        };

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

        // Per-kind "next run" display. For Recurring/OneShot the value
        // is `now + delay`; for Cron we ask croner; for AtTimestamp we
        // already have it. Falls back to `now` only if croner can't
        // compute (shouldn't happen — parse_schedule_kind already
        // validated the expression).
        let next_run = match &kind {
            ScheduleKind::Recurring { interval_secs } => {
                now + chrono::Duration::seconds(*interval_secs as i64)
            }
            ScheduleKind::OneShot { delay_secs } => {
                now + chrono::Duration::seconds(*delay_secs as i64)
            }
            ScheduleKind::Cron(expr) => croner::Cron::from_str(expr)
                .ok()
                .and_then(|c| c.find_next_occurrence(&now, false).ok())
                .unwrap_or(now),
            ScheduleKind::AtTimestamp(ts) => *ts,
        };

        // AtTimestamp jobs must fire at the scheduled moment, so they
        // start with `last_run = None` — anchoring to `now` would mark
        // them as already-fired. Recurring / OneShot / Cron still anchor
        // here so the scheduler doesn't fire them on the immediately-next
        // tick after creation.
        let initial_last_run = match &kind {
            ScheduleKind::AtTimestamp(_) => None,
            _ => Some(now.to_rfc3339()),
        };

        let job = CronJob {
            id: job_id.clone(),
            name: prompt.chars().take(60).collect::<String>(),
            schedule: schedule_str.clone(),
            command: prompt.to_string(),
            enabled: true,
            last_run: initial_last_run,
            origin_channel: origin.as_ref().map(|o| o.channel.clone()),
            origin_chat_id: origin.as_ref().map(|o| o.chat_id.clone()),
        };

        let mut store = self.load_store()?;
        store.add_job(job);
        store.save()?;

        let output = format!(
            "Scheduled job created.\n  ID: {}\n  Schedule: {}\n  Next run: {}\n  Prompt: {}",
            job_id,
            schedule_str,
            next_run.format("%Y-%m-%d %H:%M:%S UTC"),
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
        if store.remove_job(job_id) {
            store.save()?;
            Ok(ToolResult {
                success: true,
                output: format!("Job '{}' removed.", job_id),
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
