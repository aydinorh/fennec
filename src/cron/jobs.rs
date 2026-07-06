use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    pub id: String,
    pub name: String,
    /// Schedule expression. Supported formats:
    /// - Recurring interval: `"every 30m"`, `"every 1h"`, `"every 24h"`, `"every 7d"`.
    /// - One-shot duration:  bare `"5m"`, `"1h"`, `"30s"` — fires once after the delay.
    /// - Cron expression:    standard 5- or 6-field syntax (`"0 9 * * 1-5"`,
    ///   `"*/15 * * * *"`). Names (`MON`/`JAN`) and the `L`/`W`/`#` extensions
    ///   are supported via the `croner` crate.
    /// - Timestamp one-shot: ISO 8601 (`"2026-02-03T14:00"`, `"2026-02-03T14:00:00Z"`).
    ///   Naive timestamps are interpreted as local time at parse time.
    pub schedule: String,
    /// Message to send to the agent when this job fires.
    pub command: String,
    pub enabled: bool,
    pub last_run: Option<String>,
    /// The channel this job originated from (e.g. "telegram", "discord").
    #[serde(default)]
    pub origin_channel: Option<String>,
    /// The chat ID within the origin channel to deliver results to.
    #[serde(default)]
    pub origin_chat_id: Option<String>,
    /// Lifecycle state: `"scheduled"` | `"paused"` | `"error"` | `"completed"`.
    /// Legacy jobs without this field are treated as `"scheduled"` when
    /// enabled and `"paused"` when disabled (see [`CronJob::effective_state`]).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub state: String,
    /// Creation timestamp (RFC3339). Populated at create time; legacy
    /// jobs without this stay `None` and consumers must tolerate the
    /// absence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    /// Next scheduled run time (RFC3339). Persisted so jobs don't drift
    /// across scheduler restarts and so missed windows can be fast-forwarded
    /// — matches the upstream's at-most-once + stale-fast-forward semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_run_at: Option<String>,
    /// Outcome of the most recent run: `"ok"` | `"error"`. None before
    /// the first run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_status: Option<String>,
    /// Error message from the most recent failed run. Cleared on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    /// Error message from the most recent failed delivery attempt
    /// (separate from `last_error` — a job can succeed but fail
    /// delivery, e.g. when the destination platform is down).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_delivery_error: Option<String>,
    /// When the job was paused (RFC3339). None when not paused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_at: Option<String>,
    /// Why the job was paused. None when not paused or no reason given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused_reason: Option<String>,
    /// Run-count gate. `times = None` means run forever; otherwise the
    /// job is auto-removed once `completed >= times` (mirrors the
    /// upstream's `repeat: {times, completed}` field).
    #[serde(default)]
    pub repeat: RepeatConfig,
    /// Human-readable schedule label for UI/list display (e.g.
    /// `"every 30m"`, `"once in 30m"`, `"once at 2026-02-03 14:00"`, or
    /// the cron expression itself). Computed at create time so a
    /// `/cron list` doesn't re-parse the schedule string.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub schedule_display: String,
}

impl CronJob {
    /// Resolve the lifecycle state, treating an empty stored `state` as
    /// `"scheduled"` when enabled and `"paused"` when disabled — matches
    /// the upstream's `_normalize_job_record` fallback so legacy jobs
    /// don't appear in an "unknown" limbo.
    pub fn effective_state(&self) -> &str {
        if !self.state.is_empty() {
            return &self.state;
        }
        if self.enabled {
            "scheduled"
        } else {
            "paused"
        }
    }
}

/// Repeat-count gate for a cron job. `times = None` ⇒ run forever;
/// otherwise the scheduler auto-removes the job once
/// `completed >= times` after its final run.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RepeatConfig {
    /// Requested total runs. `None` means run forever.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub times: Option<u32>,
    /// Completed runs so far.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub completed: u32,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// Subset of job fields that callers can update via [`JobStore::update_job`].
/// Each `Option` wraps a single update; outer `None` = leave unchanged.
/// For nullable fields, an outer `Some(None)` clears the field to `None`.
#[derive(Debug, Clone, Default)]
pub struct JobUpdates {
    pub name: Option<String>,
    pub schedule: Option<String>,
    pub command: Option<String>,
    pub enabled: Option<bool>,
    pub state: Option<String>,
    pub next_run_at: Option<Option<String>>,
    pub last_run: Option<Option<String>>,
    pub last_status: Option<Option<String>>,
    pub last_error: Option<Option<String>>,
    pub last_delivery_error: Option<Option<String>>,
    pub paused_at: Option<Option<String>>,
    pub paused_reason: Option<Option<String>>,
    pub repeat: Option<RepeatConfig>,
    pub schedule_display: Option<String>,
}

/// Error returned by [`JobStore::resolve_job_ref`] when a name matches
/// more than one job. Mirrors the upstream's `AmbiguousJobReference` —
/// callers should surface the matching IDs so the user can disambiguate.
#[derive(Debug)]
pub struct AmbiguousJobReference {
    pub reference: String,
    pub matches: Vec<String>,
}

impl std::fmt::Display for AmbiguousJobReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "job name '{}' is ambiguous — matches {} jobs: {}. Use the job ID instead.",
            self.reference,
            self.matches.len(),
            self.matches.join(", ")
        )
    }
}

impl std::error::Error for AmbiguousJobReference {}

pub struct JobStore {
    jobs: Vec<CronJob>,
    path: PathBuf,
}

impl JobStore {
    /// Create a new `JobStore` backed by the given JSON file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            jobs: Vec::new(),
            path: path.into(),
        }
    }

    /// Load jobs from the backing file. If the file doesn't exist, start
    /// empty. If the file exists but is corrupted (not valid JSON / wrong
    /// shape), rename it to `<path>.bad-<timestamp>` and start empty. The
    /// old behavior — propagating the parse error — would abort fennec
    /// startup on a single malformed jobs.json, which is a much worse
    /// failure mode than losing the one bad file.
    pub fn load(&mut self) -> Result<()> {
        if !self.path.exists() {
            return Ok(());
        }
        let data = std::fs::read_to_string(&self.path)
            .with_context(|| format!("reading job store from {}", self.path.display()))?;
        match serde_json::from_str::<Vec<CronJob>>(&data) {
            Ok(jobs) => {
                self.jobs = jobs;
                Ok(())
            }
            Err(e) => {
                let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
                let bad_path = self.path.with_extension(format!("bad-{}", ts));
                let _ = std::fs::rename(&self.path, &bad_path);
                tracing::warn!(
                    "Cron job store at {} is corrupted ({}); moved to {}, starting empty",
                    self.path.display(),
                    e,
                    bad_path.display()
                );
                self.jobs = Vec::new();
                Ok(())
            }
        }
    }

    /// Persist the current job list to the backing file atomically.
    ///
    /// Writes to a sibling tempfile and then `rename(2)`s into place, so a
    /// crash mid-write never leaves `jobs.json` truncated / empty. The old
    /// `std::fs::write` truncated the target first and wrote incrementally
    /// — a crash during that window lost every scheduled job.
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("creating parent dir {}", parent.display()))?;
                // Tighten the cron directory to owner-only access (0700) on
                // Unix. Matches the upstream's `_secure_dir` — cron jobs can
                // carry credentials in their command/prompt, so the store
                // should never be group/world-readable. No-op on Windows.
                secure_dir(parent);
            }
        }
        let data = serde_json::to_string_pretty(&self.jobs)
            .context("serializing job store")?;

        let file_name = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("jobs.json");
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        let tmp_path = parent.join(format!(".{}.tmp.{}", file_name, std::process::id()));

        let result = (|| -> Result<()> {
            std::fs::write(&tmp_path, &data)
                .with_context(|| format!("writing tempfile {}", tmp_path.display()))?;
            std::fs::rename(&tmp_path, &self.path)
                .with_context(|| format!("renaming {} to {}", tmp_path.display(), self.path.display()))
        })();

        if result.is_err() {
            let _ = std::fs::remove_file(&tmp_path);
        } else {
            // Lock the final file to 0600 — owner-only read/write — for the
            // same reason as the parent dir above. Matches the upstream's
            // `_secure_file`.
            secure_file(&self.path);
        }
        result
    }

    /// Add a job to the store.
    pub fn add_job(&mut self, job: CronJob) {
        self.jobs.push(job);
    }

    /// Remove a job by ID. Returns `true` if the job was found and removed.
    pub fn remove_job(&mut self, id: &str) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != id);
        self.jobs.len() < before
    }

    /// List all jobs.
    pub fn list_jobs(&self) -> &[CronJob] {
        &self.jobs
    }

    /// Get a mutable reference to a job by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut CronJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// The file path this store persists to.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl JobStore {
    /// Resolve a job reference (ID or name) to a job. Exact ID match
    /// wins first; otherwise case-insensitive name match. Returns
    /// [`AmbiguousJobReference`] when a name matches more than one job
    /// so the caller can surface the matching IDs rather than silently
    /// picking one — mirrors the upstream's `resolve_job_ref`.
    pub fn resolve_job_ref(
        &self,
        reference: &str,
    ) -> Result<Option<&CronJob>, AmbiguousJobReference> {
        let ref_trim = reference.trim();
        if ref_trim.is_empty() {
            return Ok(None);
        }
        if let Some(job) = self.jobs.iter().find(|j| j.id == ref_trim) {
            return Ok(Some(job));
        }
        let ref_lower = ref_trim.to_lowercase();
        let name_matches: Vec<&CronJob> = self
            .jobs
            .iter()
            .filter(|j| j.name.to_lowercase() == ref_lower)
            .collect();
        match name_matches.len() {
            0 => Ok(None),
            1 => Ok(Some(name_matches[0])),
            _ => Err(AmbiguousJobReference {
                reference: ref_trim.to_string(),
                matches: name_matches.iter().map(|j| j.id.clone()).collect(),
            }),
        }
    }

    /// Get a job by ID or name. Returns a clone for ergonomics; callers
    /// needing a mutable handle use [`JobStore::get_mut`] or
    /// [`JobStore::update_job`].
    pub fn get_job(&self, reference: &str) -> Result<Option<CronJob>, AmbiguousJobReference> {
        self.resolve_job_ref(reference).map(|opt| opt.cloned())
    }

    /// Remove a job by ID or name. Returns the boolean removal result
    /// inside `Result` so name ambiguity can be reported to the caller.
    pub fn remove_by_ref(&mut self, reference: &str) -> Result<bool, AmbiguousJobReference> {
        let id = match self.resolve_job_ref(reference)? {
            Some(job) => job.id.clone(),
            None => return Ok(false),
        };
        Ok(self.remove_job(&id))
    }

    /// Update fields on a job in place. The `id` field is immutable —
    /// it's a filesystem path component for output directories (see
    /// PR 3 / output persistence), so allowing renames leaks
    /// path-escape values into output writes/deletes (matches the
    /// upstream's `_IMMUTABLE_JOB_FIELDS` guard).
    ///
    /// Schedule changes auto-recompute `schedule_display` and
    /// `next_run_at`. Persists immediately.
    pub fn update_job(&mut self, id: &str, updates: JobUpdates) -> Result<Option<CronJob>> {
        let Some(idx) = self.jobs.iter().position(|j| j.id == id) else {
            return Ok(None);
        };
        let schedule_changed = updates.schedule.is_some();

        {
            let job = &mut self.jobs[idx];
            if let Some(name) = updates.name {
                job.name = name;
            }
            if let Some(schedule) = updates.schedule {
                job.schedule = schedule;
            }
            if let Some(command) = updates.command {
                job.command = command;
            }
            if let Some(enabled) = updates.enabled {
                job.enabled = enabled;
            }
            if let Some(state) = updates.state {
                job.state = state;
            }
            if let Some(next) = updates.next_run_at {
                job.next_run_at = next;
            }
            if let Some(last) = updates.last_run {
                job.last_run = last;
            }
            if let Some(s) = updates.last_status {
                job.last_status = s;
            }
            if let Some(e) = updates.last_error {
                job.last_error = e;
            }
            if let Some(d) = updates.last_delivery_error {
                job.last_delivery_error = d;
            }
            if let Some(p) = updates.paused_at {
                job.paused_at = p;
            }
            if let Some(r) = updates.paused_reason {
                job.paused_reason = r;
            }
            if let Some(repeat) = updates.repeat {
                job.repeat = repeat;
            }
            if let Some(disp) = updates.schedule_display {
                job.schedule_display = disp;
            }
        }

        if schedule_changed {
            let (display, next) = {
                let job = &self.jobs[idx];
                (
                    schedule_display_for(&job.schedule),
                    if job.state != "paused" {
                        compute_next_run(&job.schedule, job.last_run.as_deref())
                    } else {
                        job.next_run_at.clone()
                    },
                )
            };
            let job = &mut self.jobs[idx];
            job.schedule_display = display;
            if job.state != "paused" {
                job.next_run_at = next;
            }
        }

        // Re-arm next_run_at if the job ended up enabled, scheduled, and
        // missing it (e.g. resume_job from a state where it was cleared).
        let needs_rearm = {
            let job = &self.jobs[idx];
            job.enabled && job.state != "paused" && job.next_run_at.is_none()
        };
        if needs_rearm {
            let (schedule, last) = {
                let job = &self.jobs[idx];
                (job.schedule.clone(), job.last_run.clone())
            };
            self.jobs[idx].next_run_at = compute_next_run(&schedule, last.as_deref());
        }

        let updated = self.jobs[idx].clone();
        self.save()?;
        Ok(Some(updated))
    }

    /// Pause a job (by ID or name) without removing it. Sets
    /// `state="paused"`, `enabled=false`, records `paused_at` and
    /// `paused_reason`. Mirrors the upstream's `pause_job`.
    pub fn pause_job(&mut self, reference: &str, reason: Option<&str>) -> Result<Option<CronJob>> {
        let id = match self
            .resolve_job_ref(reference)
            .map_err(|e| anyhow::anyhow!(e))?
        {
            Some(job) => job.id.clone(),
            None => return Ok(None),
        };
        let now = chrono::Utc::now().to_rfc3339();
        let updates = JobUpdates {
            enabled: Some(false),
            state: Some("paused".to_string()),
            paused_at: Some(Some(now)),
            paused_reason: Some(reason.map(|s| s.to_string())),
            ..Default::default()
        };
        self.update_job(&id, updates)
    }

    /// Resume a paused job (by ID or name). Clears pause fields,
    /// re-enables, and recomputes the next run. Mirrors the upstream's
    /// `resume_job`.
    pub fn resume_job(&mut self, reference: &str) -> Result<Option<CronJob>> {
        let id = match self
            .resolve_job_ref(reference)
            .map_err(|e| anyhow::anyhow!(e))?
        {
            Some(job) => job.id.clone(),
            None => return Ok(None),
        };
        let next_run = {
            let job = self
                .jobs
                .iter()
                .find(|j| j.id == id)
                .expect("resolve_job_ref returned id that's no longer in store");
            compute_next_run(&job.schedule, job.last_run.as_deref())
        };
        let updates = JobUpdates {
            enabled: Some(true),
            state: Some("scheduled".to_string()),
            paused_at: Some(None),
            paused_reason: Some(None),
            next_run_at: Some(next_run),
            ..Default::default()
        };
        self.update_job(&id, updates)
    }

    /// Trigger a job to run on the next scheduler tick (by ID or name).
    /// Sets `next_run_at = now` so the next tick picks it up regardless
    /// of the schedule. Mirrors the upstream's `trigger_job`.
    pub fn trigger_job(&mut self, reference: &str) -> Result<Option<CronJob>> {
        let id = match self
            .resolve_job_ref(reference)
            .map_err(|e| anyhow::anyhow!(e))?
        {
            Some(job) => job.id.clone(),
            None => return Ok(None),
        };
        let now = chrono::Utc::now().to_rfc3339();
        let updates = JobUpdates {
            enabled: Some(true),
            state: Some("scheduled".to_string()),
            paused_at: Some(None),
            paused_reason: Some(None),
            next_run_at: Some(Some(now)),
            ..Default::default()
        };
        self.update_job(&id, updates)
    }

    /// Mark a job as having run. Updates `last_run`, `last_status`,
    /// `last_error`, `last_delivery_error`; increments `repeat.completed`
    /// and removes the job if the repeat limit is reached; recomputes
    /// `next_run_at` and sets the lifecycle state accordingly.
    ///
    /// Important parity detail: a recurring job whose `next_run_at`
    /// can't be computed is left enabled with `state="error"` rather
    /// than silently disabled — a missing croner / a runtime regression
    /// shouldn't quietly turn a daily reminder into "completed".
    /// Mirrors the upstream's `mark_job_run` (and its bug-fix history).
    pub fn mark_job_run(
        &mut self,
        id: &str,
        success: bool,
        error: Option<&str>,
        delivery_error: Option<&str>,
    ) -> Result<()> {
        let Some(idx) = self.jobs.iter().position(|j| j.id == id) else {
            tracing::warn!("mark_job_run: job id '{}' not found, skipping save", id);
            return Ok(());
        };
        let now = chrono::Utc::now().to_rfc3339();
        let schedule_str = self.jobs[idx].schedule.clone();

        {
            let job = &mut self.jobs[idx];
            job.last_run = Some(now.clone());
            job.last_status = Some(if success { "ok" } else { "error" }.to_string());
            job.last_error = if success {
                None
            } else {
                error.map(|s| s.to_string())
            };
            job.last_delivery_error = delivery_error.map(|s| s.to_string());
            job.repeat.completed = job.repeat.completed.saturating_add(1);

            if let Some(times) = job.repeat.times {
                if times > 0 && job.repeat.completed >= times {
                    self.jobs.remove(idx);
                    return self.save();
                }
            }
        }

        let next = compute_next_run(&schedule_str, Some(&now));
        let kind = parse_schedule_kind(&schedule_str);
        let is_recurring = matches!(
            kind,
            Some(ScheduleKind::Recurring { .. } | ScheduleKind::Cron(_))
        );

        let job = &mut self.jobs[idx];
        job.next_run_at = next.clone();

        if next.is_none() {
            if is_recurring {
                job.state = "error".to_string();
                if job.last_error.is_none() {
                    job.last_error = Some(
                        "Failed to compute next run for recurring schedule".to_string(),
                    );
                }
                tracing::error!(
                    "Cron job '{}' ({}): could not compute next_run_at — leaving enabled and marking state=error so the job is not silently disabled",
                    job.name, job.id
                );
            } else {
                job.enabled = false;
                job.state = "completed".to_string();
            }
        } else if job.state != "paused" {
            job.state = "scheduled".to_string();
        }

        self.save()
    }

    /// Preemptively advance `next_run_at` for a recurring job BEFORE it
    /// fires. Converts the scheduler from at-least-once to at-most-once
    /// for recurring jobs — a crash mid-execution loses one run rather
    /// than refiring on next restart. One-shots are left unchanged so
    /// they can still retry on restart. Returns true if advanced.
    /// Mirrors the upstream's `advance_next_run`.
    pub fn advance_next_run(&mut self, id: &str) -> Result<bool> {
        let Some(idx) = self.jobs.iter().position(|j| j.id == id) else {
            return Ok(false);
        };
        let schedule_str = self.jobs[idx].schedule.clone();
        let kind = parse_schedule_kind(&schedule_str);
        if !matches!(
            kind,
            Some(ScheduleKind::Recurring { .. } | ScheduleKind::Cron(_))
        ) {
            return Ok(false);
        }
        let now = chrono::Utc::now().to_rfc3339();
        let new_next = compute_next_run(&schedule_str, Some(&now));
        if let Some(new_next) = new_next {
            let job = &mut self.jobs[idx];
            if job.next_run_at.as_deref() != Some(new_next.as_str()) {
                job.next_run_at = Some(new_next);
                self.save()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Return all jobs that are due to run now.
    ///
    /// For recurring jobs whose `next_run_at` is more than the
    /// schedule-period-aware grace window in the past (e.g. the
    /// scheduler was offline overnight), fast-forwards to the next
    /// future occurrence and skips this run — preventing a burst of
    /// stale fires on restart. Mirrors the upstream's `get_due_jobs`
    /// + `_compute_grace_seconds` behaviour.
    ///
    /// May save the store as a side effect (recovered or fast-forwarded
    /// `next_run_at` values).
    pub fn get_due_jobs(&mut self) -> Vec<CronJob> {
        let now = chrono::Utc::now();
        let mut due: Vec<CronJob> = Vec::new();
        let mut dirty = false;

        for idx in 0..self.jobs.len() {
            if !self.jobs[idx].enabled {
                continue;
            }
            let id = self.jobs[idx].id.clone();
            let schedule = self.jobs[idx].schedule.clone();
            let last_run = self.jobs[idx].last_run.clone();
            let mut next_run_at = self.jobs[idx].next_run_at.clone();

            // Recovery: a missing next_run_at on an enabled job (legacy
            // import, hand-edited jobs.json, or a one-shot that never
            // anchored) — recompute from the schedule + last_run.
            if next_run_at.is_none() {
                if let Some(recovered) = compute_next_run(&schedule, last_run.as_deref()) {
                    next_run_at = Some(recovered.clone());
                    self.jobs[idx].next_run_at = Some(recovered);
                    dirty = true;
                } else {
                    continue;
                }
            }

            let next = next_run_at.expect("just set");
            let Ok(next_dt) = chrono::DateTime::parse_from_rfc3339(&next) else {
                tracing::warn!(
                    "Cron job '{}': invalid next_run_at '{}', skipping",
                    id,
                    next
                );
                continue;
            };
            let next_dt = next_dt.with_timezone(&chrono::Utc);

            if next_dt > now {
                continue; // not yet due
            }

            // Past due — for recurring jobs, check stale fast-forward.
            let kind = parse_schedule_kind(&schedule);
            let is_recurring = matches!(
                kind,
                Some(ScheduleKind::Recurring { .. } | ScheduleKind::Cron(_))
            );
            if is_recurring {
                let lateness = (now - next_dt).num_seconds();
                let grace = grace_seconds_for(&schedule);
                if lateness > grace {
                    if let Some(fast_fwd) = compute_next_run(&schedule, Some(&now.to_rfc3339())) {
                        tracing::info!(
                            "Cron job '{}': missed scheduled time {} (grace {}s, late {}s) — fast-forwarding to {}",
                            id, next, grace, lateness, fast_fwd
                        );
                        self.jobs[idx].next_run_at = Some(fast_fwd);
                        dirty = true;
                        continue;
                    }
                }
            }

            due.push(self.jobs[idx].clone());
        }

        if dirty {
            if let Err(e) = self.save() {
                tracing::error!("get_due_jobs: failed to save recovery updates: {}", e);
            }
        }

        due
    }
}

/// How a schedule string should be interpreted by the scheduler.
///
/// The old single `Option<u64>` interval lost the distinction between
/// "fire every N seconds forever" and "fire once N seconds from now",
/// so bare durations like `"5m"` re-fired on every scheduler tick after
/// the first — because `elapsed >= interval` stayed true forever. The
/// scheduler now uses this enum to enforce one-shot semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleKind {
    /// `"every Nu"` — fire every `interval_secs` seconds.
    Recurring { interval_secs: u64 },
    /// Bare `"Nu"` — fire once, `delay_secs` after the job is scheduled.
    OneShot { delay_secs: u64 },
    /// Standard cron expression (5 or 6 fields). The string is re-parsed
    /// each tick via `croner::Cron::from_str` to compute the next
    /// occurrence after `last_run`.
    Cron(String),
    /// Fire once at a specific UTC timestamp. Naive ISO timestamps from
    /// the source string are interpreted as local time at parse time and
    /// converted to UTC, so the stored value doesn't depend on the
    /// system timezone at fire-check time.
    AtTimestamp(chrono::DateTime<chrono::Utc>),
}

impl ScheduleKind {
    /// Interval-equivalent seconds. Returns 0 for non-interval kinds
    /// (cron, timestamp) — callers that need a "next fire in N seconds"
    /// value for those must branch on the variant and compute it
    /// directly (via croner or the stored timestamp).
    pub fn seconds(&self) -> u64 {
        match self {
            ScheduleKind::Recurring { interval_secs } => *interval_secs,
            ScheduleKind::OneShot { delay_secs } => *delay_secs,
            ScheduleKind::Cron(_) | ScheduleKind::AtTimestamp(_) => 0,
        }
    }

    /// True if this kind fires at most once.
    pub fn is_one_shot(&self) -> bool {
        matches!(
            self,
            ScheduleKind::OneShot { .. } | ScheduleKind::AtTimestamp(_)
        )
    }
}

/// Parse a schedule string into a [`ScheduleKind`].
///
/// Recurring: `"every 30m"`, `"every 1h"`, `"every 24h"`, `"every 7d"`.
/// One-shot: `"5m"`, `"1h"`, `"30s"` (no `every` prefix — delay-then-fire-once).
///
/// Supported units: `s`, `m`, `h`, `d`.
///
/// Returns `None` for:
/// - empty input
/// - a number with no unit
/// - zero (would produce a busy-loop in the scheduler)
/// - arithmetic overflow (very large numbers times large multipliers)
/// - unknown units
pub fn parse_schedule_kind(schedule: &str) -> Option<ScheduleKind> {
    let trimmed = schedule.trim();
    if trimmed.is_empty() {
        return None;
    }

    // 1. "every X" → Recurring interval.
    if let Some(rest) = trimmed.strip_prefix("every ") {
        let rest = rest.trim();
        if rest.is_empty() {
            return None;
        }
        let seconds = parse_duration_seconds(rest)?;
        return Some(ScheduleKind::Recurring {
            interval_secs: seconds,
        });
    }

    // 2. 5- or 6-field cron expression — validated via croner so the parse
    //    rejects malformed expressions up front instead of failing on a
    //    later scheduler tick.
    let field_count = trimmed.split_whitespace().count();
    if (field_count == 5 || field_count == 6) && croner::Cron::from_str(trimmed).is_ok() {
        return Some(ScheduleKind::Cron(trimmed.to_string()));
    }

    // 3. ISO 8601 timestamp ("2026-02-03T14:00", "2026-02-03 14:00:00",
    //    "2026-02-03T14:00:00Z"). Timezone-aware variants parse via
    //    RFC3339; naive variants are interpreted as local time then
    //    converted to UTC, matching the upstream's pin-at-parse-time
    //    behaviour.
    if trimmed.contains('T') || looks_like_iso_date_prefix(trimmed) {
        if let Some(ts) = parse_iso_timestamp(trimmed) {
            return Some(ScheduleKind::AtTimestamp(ts));
        }
    }

    // 4. Bare "Nu" → OneShot delay.
    let seconds = parse_duration_seconds(trimmed)?;
    Some(ScheduleKind::OneShot {
        delay_secs: seconds,
    })
}

/// Parse a duration token like `"30m"`, `"1h"`, `"7d"`, `"90s"` into
/// seconds. Rejects zero (would busy-loop the scheduler) and overflow.
fn parse_duration_seconds(s: &str) -> Option<u64> {
    let unit_pos = s.find(|c: char| c.is_alphabetic())?;
    let (num_str, unit) = s.split_at(unit_pos);
    let num: u64 = num_str.trim().parse().ok()?;
    if num == 0 {
        return None;
    }
    let multiplier: u64 = match unit.trim() {
        "s" => 1,
        "m" => 60,
        "h" => 3600,
        "d" => 86400,
        _ => return None,
    };
    num.checked_mul(multiplier)
}

/// Cheap pre-filter for the timestamp branch: looks like `YYYY-MM-DD…`.
/// The actual parse happens in `parse_iso_timestamp`.
fn looks_like_iso_date_prefix(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() >= 10
        && bytes[0..4].iter().all(u8::is_ascii_digit)
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[7] == b'-'
        && bytes[8..10].iter().all(u8::is_ascii_digit)
}

fn parse_iso_timestamp(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    // RFC3339 / ISO 8601 with explicit timezone.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&chrono::Utc));
    }
    // Naive formats — pin to local at parse time so the stored UTC value
    // doesn't drift if the system timezone changes before the fire check.
    use chrono::TimeZone;
    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return chrono::Local
                .from_local_datetime(&naive)
                .single()
                .map(|dt| dt.with_timezone(&chrono::Utc));
        }
    }
    None
}

/// Back-compat wrapper: returns just the seconds, discarding the recurring
/// vs one-shot distinction. Prefer [`parse_schedule_kind`] in new code.
pub fn parse_schedule(schedule: &str) -> Option<u64> {
    parse_schedule_kind(schedule).map(|k| k.seconds())
}

// =============================================================================
// Next-run computation + grace windows + display helpers
// =============================================================================

/// Grace window for one-shot timestamp jobs (seconds). A job scheduled
/// for HH:MM still fires if the scheduler tick happens within this many
/// seconds after the moment — matches the upstream's `ONESHOT_GRACE_SECONDS`.
pub const ONESHOT_GRACE_SECS: i64 = 120;

/// Floor for the recurring stale-fast-forward window (seconds). A daily
/// job missed by less than 2 minutes still fires; missed by 2h doesn't.
const MIN_RECURRING_GRACE_SECS: i64 = 120;
/// Ceiling for the recurring stale-fast-forward window (seconds, = 2h).
const MAX_RECURRING_GRACE_SECS: i64 = 7200;

/// Compute the next scheduled run time for a job, given its schedule
/// string and the last-run timestamp (if any). Returns an RFC3339 string
/// or `None` if the job has no more runs (e.g. a one-shot that already
/// fired).
///
/// Semantics, mirroring the upstream's `compute_next_run`:
/// - **Recurring interval**: `last_run + interval` (or `now + interval`
///   if never run). Each call returns a future timestamp.
/// - **One-shot duration** (`"30m"`): `now + delay` if never run; `None`
///   once fired.
/// - **Cron expression**: next occurrence after `last_run` (or `now` if
///   never run).
/// - **Timestamp one-shot**: the timestamp itself if never run; `None`
///   once fired.
pub fn compute_next_run(schedule: &str, last_run_at: Option<&str>) -> Option<String> {
    let kind = parse_schedule_kind(schedule)?;
    let now = chrono::Utc::now();
    let last = last_run_at
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));
    match kind {
        ScheduleKind::Recurring { interval_secs } => {
            let base = last.unwrap_or(now);
            Some((base + chrono::Duration::seconds(interval_secs as i64)).to_rfc3339())
        }
        ScheduleKind::OneShot { delay_secs } => {
            if last.is_some() {
                None
            } else {
                Some((now + chrono::Duration::seconds(delay_secs as i64)).to_rfc3339())
            }
        }
        ScheduleKind::Cron(expr) => {
            let cron = croner::Cron::from_str(&expr).ok()?;
            let base = last.unwrap_or(now);
            cron.find_next_occurrence(&base, false)
                .ok()
                .map(|dt| dt.to_rfc3339())
        }
        ScheduleKind::AtTimestamp(ts) => {
            if last.is_some() {
                None
            } else {
                Some(ts.to_rfc3339())
            }
        }
    }
}

/// Stale-fast-forward grace window (seconds) for a recurring job.
///
/// A recurring job missed by less than this window still fires on the
/// next tick (catch-up). Missed by more, the scheduler fast-forwards to
/// the next future occurrence instead of replaying a stale run.
///
/// Uses *half the schedule period*, clamped to [120s, 2h] — matches the
/// upstream's `_compute_grace_seconds`. One-shot kinds return the fixed
/// [`ONESHOT_GRACE_SECS`].
pub fn grace_seconds_for(schedule: &str) -> i64 {
    let Some(kind) = parse_schedule_kind(schedule) else {
        return MIN_RECURRING_GRACE_SECS;
    };
    match kind {
        ScheduleKind::OneShot { .. } | ScheduleKind::AtTimestamp(_) => ONESHOT_GRACE_SECS,
        ScheduleKind::Recurring { interval_secs } => {
            let period = interval_secs as i64;
            let half = period / 2;
            half.clamp(MIN_RECURRING_GRACE_SECS, MAX_RECURRING_GRACE_SECS)
        }
        ScheduleKind::Cron(expr) => {
            // Approximate the cron job's period via two successive occurrences
            // after now; fall back to MIN_RECURRING_GRACE_SECS on parse/iter
            // failure so the scheduler never blocks on a fancy expression.
            let now = chrono::Utc::now();
            (|| -> Option<i64> {
                let cron = croner::Cron::from_str(&expr).ok()?;
                let first = cron.find_next_occurrence(&now, false).ok()?;
                let second = cron.find_next_occurrence(&first, false).ok()?;
                let period = (second - first).num_seconds();
                let half = period / 2;
                Some(half.clamp(MIN_RECURRING_GRACE_SECS, MAX_RECURRING_GRACE_SECS))
            })()
            .unwrap_or(MIN_RECURRING_GRACE_SECS)
        }
    }
}

/// Compose a user-friendly schedule label for `cron list` and similar UI.
/// Mirrors the upstream's `display` field in the parsed schedule dict.
pub fn schedule_display_for(schedule: &str) -> String {
    let trimmed = schedule.trim();
    let Some(kind) = parse_schedule_kind(trimmed) else {
        return trimmed.to_string();
    };
    match kind {
        ScheduleKind::Recurring { interval_secs } => {
            format!("every {}", humanize_seconds(interval_secs))
        }
        ScheduleKind::OneShot { delay_secs } => {
            format!("once in {}", humanize_seconds(delay_secs))
        }
        ScheduleKind::Cron(expr) => expr,
        ScheduleKind::AtTimestamp(ts) => format!(
            "once at {}",
            ts.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M")
        ),
    }
}

// =============================================================================
// Secure filesystem helpers
// =============================================================================

/// Tighten a directory to owner-only (0700) on Unix. No-op on Windows
/// and on permission-denied (we don't own the parent — best-effort).
#[cfg(unix)]
fn secure_dir(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
}

#[cfg(not(unix))]
fn secure_dir(_path: &Path) {}

/// Tighten a regular file to owner-only read+write (0600) on Unix.
/// No-op on Windows.
#[cfg(unix)]
fn secure_file(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if path.exists() {
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

#[cfg(not(unix))]
fn secure_file(_path: &Path) {}

/// Pick the most-compact "Nu" representation for `secs`: "7d" over
/// "168h" when divisible, etc.
fn humanize_seconds(secs: u64) -> String {
    if secs == 0 {
        return "0s".to_string();
    }
    if secs.is_multiple_of(86400) {
        format!("{}d", secs / 86400)
    } else if secs.is_multiple_of(3600) {
        format!("{}h", secs / 3600)
    } else if secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{}s", secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schedule_minutes() {
        assert_eq!(parse_schedule("every 30m"), Some(1800));
    }

    #[test]
    fn test_parse_schedule_hours() {
        assert_eq!(parse_schedule("every 1h"), Some(3600));
        assert_eq!(parse_schedule("every 24h"), Some(86400));
    }

    #[test]
    fn test_parse_schedule_days() {
        assert_eq!(parse_schedule("every 7d"), Some(604800));
    }

    #[test]
    fn test_parse_schedule_seconds() {
        assert_eq!(parse_schedule("every 90s"), Some(90));
    }

    #[test]
    fn test_parse_schedule_bare_duration() {
        assert_eq!(parse_schedule("5m"), Some(300));
        assert_eq!(parse_schedule("1h"), Some(3600));
        assert_eq!(parse_schedule("30s"), Some(30));
    }

    #[test]
    fn test_parse_schedule_invalid() {
        assert_eq!(parse_schedule(""), None);
        assert_eq!(parse_schedule("every"), None);
        assert_eq!(parse_schedule("every abc"), None);
        assert_eq!(parse_schedule("every 30x"), None);
        assert_eq!(parse_schedule("abc"), None);
    }

    #[test]
    fn parse_schedule_kind_classifies_recurring_vs_oneshot() {
        assert_eq!(
            parse_schedule_kind("every 30m"),
            Some(ScheduleKind::Recurring { interval_secs: 1800 })
        );
        assert_eq!(
            parse_schedule_kind("30m"),
            Some(ScheduleKind::OneShot { delay_secs: 1800 })
        );
    }

    #[test]
    fn parse_schedule_kind_recognises_cron_expression() {
        // Standard 5-field cron — weekdays at 9am.
        assert_eq!(
            parse_schedule_kind("0 9 * * 1-5"),
            Some(ScheduleKind::Cron("0 9 * * 1-5".to_string()))
        );
        // Step + wildcard.
        assert_eq!(
            parse_schedule_kind("*/15 * * * *"),
            Some(ScheduleKind::Cron("*/15 * * * *".to_string()))
        );
        // 6-field cron (with seconds) — first of every month at midnight.
        assert!(matches!(
            parse_schedule_kind("0 0 0 1 * *"),
            Some(ScheduleKind::Cron(_))
        ));
    }

    #[test]
    fn parse_schedule_kind_recognises_cron_with_extensions() {
        // croner supports the Quartz/croniter `L` (last day of month) extension.
        // We want parity with that.
        assert!(matches!(
            parse_schedule_kind("0 0 L * *"),
            Some(ScheduleKind::Cron(_))
        ));
    }

    #[test]
    fn parse_schedule_kind_rejects_invalid_cron_falls_through_to_other_kinds() {
        // "99 99 * * *" has 5 fields but is invalid (out-of-range). Without a
        // duration/timestamp fallback match, it must return None — never
        // accept-as-cron silently.
        assert_eq!(parse_schedule_kind("99 99 * * *"), None);
    }

    #[test]
    fn parse_schedule_kind_recognises_iso_timestamp() {
        // RFC3339 with Z.
        match parse_schedule_kind("2026-02-03T14:00:00Z") {
            Some(ScheduleKind::AtTimestamp(dt)) => {
                assert_eq!(dt.to_rfc3339(), "2026-02-03T14:00:00+00:00");
            }
            other => panic!("expected AtTimestamp, got {:?}", other),
        }
        // Naive ISO — parses as local time. We only assert the variant +
        // round-trip, not a specific UTC value (depends on the test host TZ).
        assert!(matches!(
            parse_schedule_kind("2026-02-03T14:00"),
            Some(ScheduleKind::AtTimestamp(_))
        ));
        assert!(matches!(
            parse_schedule_kind("2026-02-03 14:00:00"),
            Some(ScheduleKind::AtTimestamp(_))
        ));
    }

    #[test]
    fn schedule_kind_is_one_shot_classification() {
        assert!(!ScheduleKind::Recurring { interval_secs: 60 }.is_one_shot());
        assert!(ScheduleKind::OneShot { delay_secs: 60 }.is_one_shot());
        assert!(!ScheduleKind::Cron("* * * * *".to_string()).is_one_shot());
        assert!(ScheduleKind::AtTimestamp(chrono::Utc::now()).is_one_shot());
    }

    #[test]
    fn seconds_returns_zero_for_non_interval_kinds() {
        assert_eq!(ScheduleKind::Cron("* * * * *".to_string()).seconds(), 0);
        assert_eq!(ScheduleKind::AtTimestamp(chrono::Utc::now()).seconds(), 0);
    }

    #[test]
    fn parse_schedule_rejects_zero() {
        // Without this, "every 0s" would return Some(0) and cause the
        // scheduler to fire the job on every tick in a tight loop.
        assert_eq!(parse_schedule("every 0s"), None);
        assert_eq!(parse_schedule("0m"), None);
        assert_eq!(parse_schedule("every 0h"), None);
    }

    #[test]
    fn parse_schedule_rejects_overflow() {
        // u64::MAX days would panic in debug / wrap in release without
        // checked_mul.
        assert_eq!(parse_schedule("every 9999999999999999d"), None);
        assert_eq!(parse_schedule("18446744073709551615d"), None);
    }

    #[test]
    fn parse_schedule_accepts_large_but_safe() {
        // Still accepts reasonable large values.
        assert_eq!(parse_schedule("every 100d"), Some(8_640_000));
    }

    #[test]
    fn save_is_atomic_no_tempfile_leaks() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("jobs.json");
        let mut store = JobStore::new(path.clone());
        store.add_job(CronJob {
            id: "a".into(),
            name: "test".into(),
            schedule: "every 1h".into(),
            command: "noop".into(),
            enabled: true,
            last_run: None,
            origin_channel: None,
            origin_chat_id: None,
            state: String::new(),
            created_at: None,
            next_run_at: None,
            last_status: None,
            last_error: None,
            last_delivery_error: None,
            paused_at: None,
            paused_reason: None,
            repeat: RepeatConfig::default(),
            schedule_display: String::new(),
        });
        store.save().unwrap();
        // After a clean save, the parent directory must contain exactly
        // the final file — no .tmp.* siblings.
        let leaks: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                s.contains(".tmp.")
            })
            .collect();
        assert!(leaks.is_empty(), "tempfiles left behind: {:?}", leaks);
        assert!(path.exists());
    }

    #[test]
    fn load_recovers_from_corrupt_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("jobs.json");
        std::fs::write(&path, b"this is not json at all").unwrap();

        let mut store = JobStore::new(path.clone());
        // The old implementation would return Err here, aborting fennec
        // startup. The new load() must succeed with an empty store.
        store.load().expect("corrupt jobs.json must not block startup");
        assert!(store.list_jobs().is_empty());

        // The bad file must have been moved aside (with a .bad- extension),
        // preserving it for post-mortem inspection.
        let bad_files: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .contains(".bad-")
            })
            .collect();
        assert_eq!(
            bad_files.len(),
            1,
            "expected exactly one .bad-<ts> file, got: {:?}",
            bad_files
        );
    }

    #[test]
    fn save_then_load_roundtrips() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("jobs.json");
        let mut store = JobStore::new(path.clone());
        store.add_job(CronJob {
            id: "x".into(),
            name: "test".into(),
            schedule: "every 30m".into(),
            command: "do a thing".into(),
            enabled: true,
            last_run: None,
            origin_channel: Some("telegram".into()),
            origin_chat_id: Some("123".into()),
            state: String::new(),
            created_at: None,
            next_run_at: None,
            last_status: None,
            last_error: None,
            last_delivery_error: None,
            paused_at: None,
            paused_reason: None,
            repeat: RepeatConfig::default(),
            schedule_display: String::new(),
        });
        store.save().unwrap();

        let mut reloaded = JobStore::new(path);
        reloaded.load().unwrap();
        assert_eq!(reloaded.list_jobs().len(), 1);
        assert_eq!(reloaded.list_jobs()[0].id, "x");
        assert_eq!(
            reloaded.list_jobs()[0].origin_channel.as_deref(),
            Some("telegram")
        );
    }
}
