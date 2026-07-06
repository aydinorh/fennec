pub mod jobs;
pub mod output;
pub mod script;
pub use jobs::{
    compute_next_run, grace_seconds_for, parse_schedule, parse_schedule_kind,
    schedule_display_for, AmbiguousJobReference, CronJob, JobStore, JobUpdates, RepeatConfig,
    ScheduleKind, ONESHOT_GRACE_SECS,
};

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;

use crate::bus::{InboundMessage, MessageBus};

/// Env var that lets ops cap the cron tick's parallelism without rebuilding.
/// Set to a positive integer; `0`, missing, or unparseable values mean
/// "unbounded" (every due job spawned at once). Matches the upstream's
/// `HERMES_CRON_MAX_PARALLEL` knob.
pub const MAX_PARALLEL_ENV: &str = "FENNEC_CRON_MAX_PARALLEL";

pub struct CronScheduler {
    store: JobStore,
    bus: MessageBus,
    check_interval_secs: u64,
    /// Optional concurrency cap for in-tick job publishes. `None` (the
    /// default) means unbounded — every due job is spawned at once.
    /// The env [`MAX_PARALLEL_ENV`] is read each tick and overrides
    /// this; the field is the programmatic fallback.
    max_parallel: Option<usize>,
    /// Optional override for the scripts dir (`<scripts_dir>/<script>`
    /// is where job scripts live). Defaults to
    /// `<jobs_dir>/scripts/`.
    scripts_dir: Option<std::path::PathBuf>,
    /// Optional override for the output dir
    /// (`<output_dir>/<job_id>/{ts}.md`). Defaults to
    /// `<jobs_dir>/cron_output/`.
    output_dir: Option<std::path::PathBuf>,
}

impl CronScheduler {
    /// Create a new scheduler.
    ///
    /// `check_interval_secs` controls how often the scheduler wakes up to check
    /// for due jobs (default: 30).
    pub fn new(store: JobStore, bus: MessageBus, check_interval_secs: Option<u64>) -> Self {
        Self {
            store,
            bus,
            check_interval_secs: check_interval_secs.unwrap_or(30),
            max_parallel: None,
            scripts_dir: None,
            output_dir: None,
        }
    }

    /// Set the in-tick concurrency cap. `Some(1)` mimics the old
    /// serial behavior; `None` (the default) is unbounded; `Some(n)`
    /// caps via a [`tokio::sync::Semaphore`] so a tick with hundreds
    /// of due jobs doesn't flood the bus or the agent consumers.
    pub fn with_max_parallel(mut self, n: Option<usize>) -> Self {
        self.max_parallel = n;
        self
    }

    /// Override the scripts directory. Default: `<jobs_dir>/scripts/`.
    pub fn with_scripts_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.scripts_dir = Some(dir);
        self
    }

    /// Override the cron output directory. Default:
    /// `<jobs_dir>/cron_output/`.
    pub fn with_output_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.output_dir = Some(dir);
        self
    }

    /// Resolve the scripts dir, deriving from the JobStore path when
    /// no override is configured.
    fn scripts_dir(&self) -> std::path::PathBuf {
        self.scripts_dir.clone().unwrap_or_else(|| {
            self.store
                .path()
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("scripts")
        })
    }

    /// Resolve the cron output dir, deriving from the JobStore path
    /// when no override is configured.
    fn output_dir(&self) -> std::path::PathBuf {
        self.output_dir
            .clone()
            .unwrap_or_else(|| output::default_output_dir_for(self.store.path()))
    }

    /// Run the scheduler loop. This blocks until the task is cancelled.
    ///
    /// The interval is configured with
    /// `MissedTickBehavior::Skip` so that a system suspend / resume (laptop
    /// sleep) doesn't produce a burst of back-to-back ticks catching up
    /// missed time — the scheduler just resumes on the next normal tick.
    /// With the default `Burst` behavior, every `"every 5m"` job whose
    /// `last_run` predates the sleep would refire rapidly on wake.
    pub async fn run(&mut self) {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(self.check_interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            self.tick().await;
        }
    }

    /// Path to the cross-process tick lock. Lives next to `jobs.json` so
    /// a single Fennec home means a single tick at a time, regardless of
    /// how many processes (gateway, CLI, manual `tick`) try to run one.
    fn tick_lock_path(&self) -> std::path::PathBuf {
        let parent = self
            .store
            .path()
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        parent.join(".tick.lock")
    }

    /// Resolve the in-tick parallelism cap from env first, then the
    /// programmatic [`with_max_parallel`] setting. Re-reading the env each
    /// tick lets ops change the cap without restarting Fennec — matches
    /// the upstream's behaviour for `HERMES_CRON_MAX_PARALLEL`.
    fn resolve_max_parallel(&self) -> Option<usize> {
        if let Ok(raw) = std::env::var(MAX_PARALLEL_ENV) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                match trimmed.parse::<usize>() {
                    Ok(0) => return None, // explicit 0 = unbounded
                    Ok(n) => return Some(n),
                    Err(_) => {
                        tracing::warn!(
                            "Invalid {} value '{}'; falling back to the configured cap",
                            MAX_PARALLEL_ENV,
                            raw
                        );
                    }
                }
            }
        }
        self.max_parallel
    }

    /// Execute a single scheduler tick.
    ///
    /// Flow:
    /// 1. Acquire a cross-process exclusive file lock on `.tick.lock`.
    ///    If another process / in-process call already holds it, skip
    ///    this tick — duplicate ticks would re-fire jobs and clobber
    ///    each other's `mark_job_run` writes.
    /// 2. Ask the store for all due jobs (handles recovery + stale
    ///    fast-forward).
    /// 3. Pre-advance each recurring job's `next_run_at` so a crash
    ///    between publish and mark loses one run rather than refiring
    ///    on every restart — at-most-once semantics.
    /// 4. Publish each job's command to the bus, in parallel (bounded
    ///    by [`MAX_PARALLEL_ENV`] / [`with_max_parallel`] if set).
    /// 5. Collect the per-job publish results and `mark_job_run`
    ///    sequentially — keeps the store's writes serialized so
    ///    concurrent `mark_job_run` calls can't clobber each other.
    pub async fn tick(&mut self) {
        // Acquire the cross-process file lock. The lock is held for the
        // duration of this tick (the `_lock_guard` keeps the File alive;
        // drop releases the lock on Unix and Windows).
        let _lock_guard = match self.acquire_tick_lock() {
            Some(g) => g,
            None => {
                tracing::debug!("Cron tick skipped — another tick holds the lock");
                return;
            }
        };

        let due_jobs = self.store.get_due_jobs();
        if due_jobs.is_empty() {
            return;
        }

        // Pre-advance recurring jobs' next_run_at BEFORE we publish so a
        // mid-publish crash doesn't refire them on restart. Sequential +
        // cheap; runs under the file lock.
        for job in &due_jobs {
            if let Err(e) = self.store.advance_next_run(&job.id) {
                tracing::error!("Cron job '{}': advance_next_run failed: {}", job.id, e);
            }
        }

        // Spawn one task per due job: each runs its pre-script (if any),
        // applies the wake-gate, prepends script output + context_from
        // into the prompt, and either publishes to the bus (agent jobs)
        // or saves the script output (no_agent jobs). Bounded by the
        // optional concurrency cap so a hundred-job tick doesn't flood
        // downstream consumers.
        let cap = self.resolve_max_parallel();
        let sem = cap.map(|n| Arc::new(tokio::sync::Semaphore::new(n)));
        let bus = self.bus.clone();
        let scripts_dir = self.scripts_dir();
        let output_dir = self.output_dir();
        let script_timeout = script::resolve_script_timeout();

        let mut handles = Vec::with_capacity(due_jobs.len());
        for job in due_jobs {
            let bus = bus.clone();
            let sem = sem.clone();
            let scripts_dir = scripts_dir.clone();
            let output_dir = output_dir.clone();
            handles.push(tokio::spawn(async move {
                let permit = match sem {
                    Some(s) => s.acquire_owned().await.ok(),
                    None => None,
                };
                process_due_job(
                    job,
                    bus,
                    scripts_dir,
                    output_dir,
                    script_timeout,
                    permit,
                )
                .await
            }));
        }

        // Collect results, then mark each run SEQUENTIALLY so concurrent
        // `mark_job_run` writes can't clobber each other's JobStore
        // mutations (it saves the whole store JSON per call). After
        // mark_job_run, if a job was auto-removed (repeat limit) the
        // output dir is cleaned up too so orphaned output trees don't
        // accumulate.
        let output_dir = self.output_dir();
        for h in handles {
            match h.await {
                Ok((id, success, err_msg)) => {
                    if let Some(ref err) = err_msg {
                        tracing::error!("Cron job '{}': run error: {}", id, err);
                    }
                    let was_present = self.store.list_jobs().iter().any(|j| j.id == id);
                    if let Err(e) =
                        self.store
                            .mark_job_run(&id, success, err_msg.as_deref(), None)
                    {
                        tracing::error!("Cron job '{}': mark_job_run failed: {}", id, e);
                    }
                    let still_present = self.store.list_jobs().iter().any(|j| j.id == id);
                    if was_present && !still_present {
                        output::cleanup_job_output(&output_dir, &id);
                    }
                }
                Err(join_err) => {
                    tracing::error!("Cron tick: publish task panicked: {}", join_err);
                }
            }
        }
    }

    /// Open and exclusively lock `.tick.lock` non-blocking. Returns a
    /// `LockGuard` whose Drop releases the lock; `None` when another
    /// process / call already holds it.
    fn acquire_tick_lock(&self) -> Option<LockGuard> {
        use fs2::FileExt;

        let lock_path = self.tick_lock_path();
        if let Some(parent) = lock_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                tracing::warn!(
                    "Cron tick: failed to create lock dir {}: {}",
                    parent.display(),
                    e
                );
                return None;
            }
        }
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
        {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(
                    "Cron tick: failed to open lock file {}: {}",
                    lock_path.display(),
                    e
                );
                return None;
            }
        };
        match file.try_lock_exclusive() {
            Ok(()) => Some(LockGuard { file: Some(file) }),
            Err(_) => None,
        }
    }
}

/// RAII guard for the tick file lock. Drop releases the OS-level lock
/// (`fs2` calls `unlock_file` on Windows / `fcntl LOCK_UN` on Unix
/// when the file is closed). Stays in scope for the lifetime of one
/// `tick()` call.
struct LockGuard {
    file: Option<std::fs::File>,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            // Explicit fs2 trait call: `File::unlock` is also a stdlib
            // method since Rust 1.89, but the MSRV here is 1.87. Going
            // through `fs2::FileExt::unlock` keeps the behaviour stable
            // across both compilers. File closes on drop too, which
            // also releases the lock — the explicit unlock is a
            // belt-and-suspenders for the narrow window between unlock
            // and close on platforms where Drop ordering matters.
            let _ = fs2::FileExt::unlock(&file);
        }
    }
}

/// Per-job tick worker. Runs the pre-script (if configured), applies
/// the wake-gate, builds the composite prompt (script output +
/// `context_from`), and either publishes to the bus (agent jobs) or
/// saves the script output (no_agent jobs).
///
/// Returns `(job_id, success, err_msg)` so the main loop can call
/// `mark_job_run` sequentially. Mirrors the upstream's `run_job` flow
/// — script-first, wake-gate-second, context-from-third — adapted for
/// Fennec's bus-based architecture (cron publishes; downstream
/// consumers run the agent).
async fn process_due_job(
    job: CronJob,
    bus: MessageBus,
    scripts_dir: std::path::PathBuf,
    output_dir: std::path::PathBuf,
    script_timeout: std::time::Duration,
    _permit: Option<tokio::sync::OwnedSemaphorePermit>,
) -> (String, bool, Option<String>) {
    let job_id = job.id.clone();
    let now = Utc::now();
    let now_str = now.format("%Y-%m-%d %H:%M:%S").to_string();
    let mut prompt = job.command.clone();
    let mut script_stdout: Option<String> = None;

    // 1. Pre-script execution.
    if let Some(script_path) = &job.script {
        let (ok, out) = script::run_job_script(script_path, &scripts_dir, script_timeout).await;
        if !ok {
            // Script failed. For no_agent jobs this is the terminal
            // outcome — save an error doc and report failure. For
            // agent jobs, prepend the error into the prompt so the
            // agent can surface it.
            if job.no_agent {
                let doc = format!(
                    "# Cron Job: {}\n\n**Job ID:** {}\n**Run Time:** {}\n**Mode:** no_agent (script)\n**Status:** script failed\n\n{}\n",
                    job.name, job_id, now_str, out
                );
                if let Err(e) = output::save_job_output(&output_dir, &job_id, &doc) {
                    tracing::warn!("Cron job '{}': save_job_output failed: {}", job_id, e);
                }
                return (job_id, false, Some(out));
            }
            prompt = format!(
                "## Script Error\nThe data-collection script failed. Report this to the user.\n\n```\n{out}\n```\n\n{prompt}"
            );
        } else {
            script_stdout = Some(out.clone());

            // 2. Wake-gate: `{"wakeAgent": false}` short-circuits the
            // whole job (no publish, silent success). For no_agent
            // jobs we still save a silent-marker doc so chained
            // `context_from` consumers can tell the job ran but had
            // nothing to report.
            if !script::parse_wake_gate(&out) {
                if job.no_agent {
                    let doc = format!(
                        "# Cron Job: {}\n\n**Job ID:** {}\n**Run Time:** {}\n**Mode:** no_agent (script)\n**Status:** silent (wakeAgent=false)\n",
                        job.name, job_id, now_str
                    );
                    if let Err(e) = output::save_job_output(&output_dir, &job_id, &doc) {
                        tracing::warn!(
                            "Cron job '{}': save_job_output failed: {}",
                            job_id,
                            e
                        );
                    }
                }
                tracing::info!(
                    "Cron job '{}': wakeAgent=false — silent run, no publish",
                    job_id
                );
                return (job_id, true, None);
            }

            // 3. no_agent + empty stdout = silent (same as wakeAgent=false).
            if job.no_agent && out.trim().is_empty() {
                let doc = format!(
                    "# Cron Job: {}\n\n**Job ID:** {}\n**Run Time:** {}\n**Mode:** no_agent (script)\n**Status:** silent (empty output)\n",
                    job.name, job_id, now_str
                );
                if let Err(e) = output::save_job_output(&output_dir, &job_id, &doc) {
                    tracing::warn!("Cron job '{}': save_job_output failed: {}", job_id, e);
                }
                tracing::info!(
                    "Cron job '{}': empty script stdout — silent run, no publish",
                    job_id
                );
                return (job_id, true, None);
            }

            // 4. Agent job with successful script: inject the output as
            // context (the data-collection pattern).
            if !job.no_agent {
                prompt = format!(
                    "## Script Output\nThe following data was collected by a pre-run script. Use it as context for your analysis.\n\n```\n{out}\n```\n\n{prompt}"
                );
            }
        }
    }

    // 5. context_from: inject preceding jobs' latest outputs as context.
    if let Some(refs) = &job.context_from {
        for ref_id in refs {
            if let Some(latest) = output::latest_job_output(&output_dir, ref_id) {
                prompt = format!(
                    "## Output from job '{ref_id}'\nThe following is the most recent output from a preceding cron job. Use it as context for your analysis.\n\n```\n{latest}\n```\n\n{prompt}"
                );
            }
        }
    }

    // 6a. no_agent path: persist the script output as the deliverable.
    //     Real channel delivery is wired up in the delivery PR; for now
    //     the saved doc is what downstream callers (or the operator)
    //     pick up. context_from for the next chained job reads it.
    if job.no_agent {
        let out = script_stdout.unwrap_or_default();
        let doc = format!(
            "# Cron Job: {}\n\n**Job ID:** {}\n**Run Time:** {}\n**Mode:** no_agent (script)\n\n---\n\n{}\n",
            job.name, job_id, now_str, out
        );
        if let Err(e) = output::save_job_output(&output_dir, &job_id, &doc) {
            tracing::warn!("Cron job '{}': save_job_output failed: {}", job_id, e);
        }
        return (job_id, true, None);
    }

    // 6b. Agent path: publish the composite prompt to the bus.
    let msg = InboundMessage {
        id: uuid::Uuid::new_v4().to_string(),
        sender: format!("cron:{}", job_id),
        content: prompt,
        channel: job
            .origin_channel
            .clone()
            .unwrap_or_else(|| "cron".to_string()),
        chat_id: job
            .origin_chat_id
            .clone()
            .unwrap_or_else(|| format!("cron:{}", job_id)),
        timestamp: now.timestamp() as u64,
        reply_to: None,
        metadata: {
            let mut m = HashMap::new();
            m.insert("source".to_string(), "cron".to_string());
            m.insert("cron_job_id".to_string(), job_id.clone());
            m
        },
    };
    match bus.publish_inbound(msg).await {
        Ok(_) => (job_id, true, None),
        Err(e) => (job_id, false, Some(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::MessageBus;

    /// Build a test job with explicit `last_run` / `next_run_at` so each
    /// case can pin the scheduler's view of the world without timing
    /// flakiness. All other fields take sensible defaults.
    fn job(
        id: &str,
        schedule: &str,
        last_run: Option<&str>,
        next_run_at: Option<&str>,
    ) -> CronJob {
        CronJob {
            id: id.into(),
            name: id.into(),
            schedule: schedule.into(),
            command: "do a thing".into(),
            enabled: true,
            last_run: last_run.map(|s| s.to_string()),
            origin_channel: None,
            origin_chat_id: None,
            state: String::new(),
            created_at: None,
            next_run_at: next_run_at.map(|s| s.to_string()),
            last_status: None,
            last_error: None,
            last_delivery_error: None,
            paused_at: None,
            paused_reason: None,
            repeat: RepeatConfig::default(),
            schedule_display: String::new(),
            script: None,
            no_agent: false,
            context_from: None,
        }
    }

    #[tokio::test]
    async fn newly_added_recurring_job_does_not_fire_immediately() {
        // A new "every 24h" job has no next_run_at; get_due_jobs recovery
        // computes it as `now + 24h` and the job is correctly NOT due.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        store.add_job(job("j1", "every 24h", None, None));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        assert!(
            rx.inbound_rx.try_recv().is_err(),
            "newly added job fired on first tick — regression"
        );
        let j = scheduler
            .store
            .list_jobs()
            .iter()
            .find(|j| j.id == "j1")
            .unwrap();
        assert!(
            j.next_run_at.is_some(),
            "next_run_at must be populated by get_due_jobs recovery"
        );
    }

    #[tokio::test]
    async fn recurring_job_fires_when_next_run_at_is_past() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        store.add_job(job("j2", "every 1h", None, Some(&past)));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        let msg = rx.inbound_rx.try_recv().expect("due job should fire");
        assert_eq!(msg.sender, "cron:j2");
    }

    #[tokio::test]
    async fn stale_recurring_job_is_fast_forwarded_not_fired() {
        // "every 1h" missed by 2h — more than the half-period grace
        // (30min, clamped to [120s, 7200s]). Must be fast-forwarded
        // instead of firing a stale run.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let two_hours_ago = (Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        store.add_job(job("stale", "every 1h", None, Some(&two_hours_ago)));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        assert!(
            rx.inbound_rx.try_recv().is_err(),
            "stale recurring job must be fast-forwarded, not fired"
        );
        let j = scheduler
            .store
            .list_jobs()
            .iter()
            .find(|j| j.id == "stale")
            .unwrap();
        let new_next = chrono::DateTime::parse_from_rfc3339(j.next_run_at.as_ref().unwrap())
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(
            new_next > Utc::now(),
            "fast-forward must move next_run_at into the future"
        );
    }

    #[tokio::test]
    async fn one_shot_completes_after_firing() {
        // A one-shot whose next_run_at is in the past fires once,
        // mark_job_run sets state=completed + enabled=false, and the
        // next tick does NOT re-fire.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        store.add_job(job("once", "5m", None, Some(&past)));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        let msg = rx.inbound_rx.try_recv().expect("one-shot fires once");
        assert_eq!(msg.sender, "cron:once");

        let j = scheduler
            .store
            .list_jobs()
            .iter()
            .find(|j| j.id == "once")
            .unwrap();
        assert!(!j.enabled, "one-shot must be disabled after firing");
        assert_eq!(j.state, "completed");
        assert_eq!(j.repeat.completed, 1, "repeat counter must increment");

        scheduler.tick().await;
        assert!(
            rx.inbound_rx.try_recv().is_err(),
            "one-shot re-fired — regression"
        );
    }

    #[tokio::test]
    async fn invalid_schedule_does_not_crash_scheduler() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        store.add_job(job("bad", "every 0s", None, None));
        store.add_job(job("good", "every 1h", None, None));

        let (bus, _rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;
    }

    #[tokio::test]
    async fn repeat_limit_auto_removes_job() {
        // A job with repeat.times=2 fires twice, then disappears.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        let mut j = job("limit", "every 1m", None, Some(&past));
        j.repeat.times = Some(2);
        store.add_job(j);

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));

        // Fire 1.
        scheduler.tick().await;
        let _ = rx.inbound_rx.try_recv().expect("fire 1");
        // Re-arm next_run_at to past for fire 2.
        if let Some(j) = scheduler.store.get_mut("limit") {
            j.next_run_at = Some((Utc::now() - chrono::Duration::seconds(1)).to_rfc3339());
        }
        scheduler.tick().await;
        let _ = rx.inbound_rx.try_recv().expect("fire 2");

        assert!(
            scheduler
                .store
                .list_jobs()
                .iter()
                .all(|j| j.id != "limit"),
            "job must be auto-removed after repeat limit"
        );
    }

    #[tokio::test]
    async fn at_most_once_advance_before_publish() {
        // advance_next_run runs BEFORE publish. After tick, next_run_at
        // is in the future even though the job just fired.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        store.add_job(job("a1", "every 1h", None, Some(&past)));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;
        let _ = rx.inbound_rx.try_recv().expect("fires");

        let j = scheduler
            .store
            .list_jobs()
            .iter()
            .find(|j| j.id == "a1")
            .unwrap();
        let next = chrono::DateTime::parse_from_rfc3339(j.next_run_at.as_ref().unwrap())
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert!(
            next > Utc::now(),
            "next_run_at must be advanced into the future after fire"
        );
        assert_eq!(j.last_status.as_deref(), Some("ok"));
    }

    #[test]
    fn pause_resume_trigger_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        store.add_job(job("p1", "every 1h", None, None));

        // Pause by ID.
        let paused = store
            .pause_job("p1", Some("for maintenance"))
            .unwrap()
            .unwrap();
        assert_eq!(paused.state, "paused");
        assert!(!paused.enabled);
        assert_eq!(paused.paused_reason.as_deref(), Some("for maintenance"));
        assert!(paused.paused_at.is_some());

        // Resume by ID — re-enables, clears pause, recomputes next_run_at.
        let resumed = store.resume_job("p1").unwrap().unwrap();
        assert_eq!(resumed.state, "scheduled");
        assert!(resumed.enabled);
        assert!(resumed.paused_at.is_none());
        assert!(resumed.next_run_at.is_some());

        // Trigger — sets next_run_at = now (immediately due).
        let triggered = store.trigger_job("p1").unwrap().unwrap();
        let triggered_at =
            chrono::DateTime::parse_from_rfc3339(triggered.next_run_at.as_ref().unwrap())
                .unwrap()
                .with_timezone(&chrono::Utc);
        let drift = (Utc::now() - triggered_at).num_seconds().abs();
        assert!(drift < 5, "trigger should set next_run_at to ~now");
    }

    #[test]
    fn resolve_job_ref_by_name_and_ambiguity() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let mut j1 = job("id1", "every 1h", None, None);
        j1.name = "Daily Review".to_string();
        let mut j2 = job("id2", "every 24h", None, None);
        j2.name = "Daily Review".to_string();
        let mut j3 = job("id3", "every 30m", None, None);
        j3.name = "Unique".to_string();
        store.add_job(j1);
        store.add_job(j2);
        store.add_job(j3);

        // By exact ID — wins over any name match.
        let by_id = store.resolve_job_ref("id1").unwrap().unwrap();
        assert_eq!(by_id.id, "id1");

        // By unique name (case-insensitive).
        let by_name = store.resolve_job_ref("unique").unwrap().unwrap();
        assert_eq!(by_name.id, "id3");

        // By ambiguous name — error with both matching IDs.
        let err = store.resolve_job_ref("daily review").unwrap_err();
        assert_eq!(err.matches.len(), 2);
        assert!(err.matches.contains(&"id1".to_string()));
        assert!(err.matches.contains(&"id2".to_string()));
    }

    #[test]
    fn update_job_immutable_id_and_schedule_recompute() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        store.add_job(job("j1", "every 1h", None, None));

        // Schedule change recomputes display + next_run_at.
        let updates = JobUpdates {
            schedule: Some("every 2h".to_string()),
            ..Default::default()
        };
        let updated = store.update_job("j1", updates).unwrap().unwrap();
        assert_eq!(updated.schedule, "every 2h");
        assert_eq!(updated.schedule_display, "every 2h");
        assert!(updated.next_run_at.is_some());
    }

    #[test]
    fn remove_by_ref_supports_id_and_name() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let mut j = job("id1", "every 1h", None, None);
        j.name = "MyReminder".to_string();
        store.add_job(j);
        store.add_job(job("id2", "every 30m", None, None));

        // Remove by name.
        assert!(store.remove_by_ref("MyReminder").unwrap());
        assert_eq!(store.list_jobs().len(), 1);

        // Remove by ID.
        assert!(store.remove_by_ref("id2").unwrap());
        assert_eq!(store.list_jobs().len(), 0);

        // Non-existent ref: Ok(false).
        assert!(!store.remove_by_ref("nope").unwrap());
    }

    #[tokio::test]
    async fn parallel_tick_fires_all_due_jobs() {
        // Five due jobs in one tick: every one should produce an
        // inbound message regardless of publish ordering.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        for i in 0..5 {
            store.add_job(job(
                &format!("p{i}"),
                "every 1m",
                None,
                Some(&past),
            ));
        }

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        let mut got = std::collections::HashSet::new();
        for _ in 0..5 {
            let m = rx
                .inbound_rx
                .try_recv()
                .expect("expected 5 inbound messages from parallel tick");
            got.insert(m.sender);
        }
        assert_eq!(got.len(), 5, "all 5 jobs should publish exactly once");
        for i in 0..5 {
            assert!(
                got.contains(&format!("cron:p{i}")),
                "missing fire for p{i}"
            );
        }
    }

    #[tokio::test]
    async fn max_parallel_cap_does_not_drop_jobs() {
        // Cap = 1 (serial publishes); 3 due jobs must all still fire,
        // just one at a time via the semaphore.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        for i in 0..3 {
            store.add_job(job(
                &format!("c{i}"),
                "every 1m",
                None,
                Some(&past),
            ));
        }

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler =
            CronScheduler::new(store, bus, Some(30)).with_max_parallel(Some(1));
        scheduler.tick().await;

        for _ in 0..3 {
            let _ = rx
                .inbound_rx
                .try_recv()
                .expect("max_parallel cap must not drop jobs");
        }
    }

    #[tokio::test]
    async fn tick_creates_lock_file() {
        // The cross-process file lock lives at <store_dir>/.tick.lock.
        // Verify it appears after the first tick so ops can see a single
        // canonical lock location for diagnostics.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        store.add_job(job("j1", "every 1h", None, None));

        let (bus, _rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        assert!(
            tmp.path().join(".tick.lock").exists(),
            ".tick.lock should be created at tick time"
        );
    }

    #[test]
    fn resolve_max_parallel_reads_env_first() {
        // Env override wins over the programmatic setting.
        let tmp = tempfile::tempdir().unwrap();
        let store = JobStore::new(tmp.path().join("jobs.json"));
        let (bus, _rx) = MessageBus::new(1);
        let scheduler = CronScheduler::new(store, bus, Some(30)).with_max_parallel(Some(2));

        // SAFETY: env mutation in tests is process-global; this test is
        // intentionally narrow in scope and restores the env on exit.
        let prev = std::env::var(MAX_PARALLEL_ENV).ok();
        // SAFETY: single-threaded scope around env mutation; the
        // restoration block below pairs every set with the original.
        unsafe {
            std::env::set_var(MAX_PARALLEL_ENV, "7");
        }
        assert_eq!(scheduler.resolve_max_parallel(), Some(7));

        unsafe {
            std::env::set_var(MAX_PARALLEL_ENV, "0");
        }
        assert_eq!(
            scheduler.resolve_max_parallel(),
            None,
            "0 in env should mean unbounded"
        );

        unsafe {
            std::env::remove_var(MAX_PARALLEL_ENV);
        }
        assert_eq!(
            scheduler.resolve_max_parallel(),
            Some(2),
            "no env should fall back to programmatic setting"
        );

        // Restore original env.
        unsafe {
            match prev {
                Some(v) => std::env::set_var(MAX_PARALLEL_ENV, v),
                None => std::env::remove_var(MAX_PARALLEL_ENV),
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn agent_job_with_script_prepends_output_to_prompt() {
        // A pre-script's stdout is injected into the agent's prompt as
        // context before the bus publish. Verifies the data-collection
        // pattern works end-to-end through tick().
        let tmp = tempfile::tempdir().unwrap();
        let scripts = tmp.path().join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        std::fs::write(scripts.join("data.sh"), "#!/bin/bash\necho VALUE=42\n").unwrap();

        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        let mut j = job("agent_with_script", "every 1m", None, Some(&past));
        j.command = "Summarize the data.".to_string();
        j.script = Some("data.sh".to_string());
        store.add_job(j);

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30))
            .with_scripts_dir(scripts.clone())
            .with_output_dir(tmp.path().join("cron_output"));
        scheduler.tick().await;

        let msg = rx
            .inbound_rx
            .try_recv()
            .expect("agent job with script should publish");
        assert!(
            msg.content.contains("VALUE=42"),
            "prompt should include script output: {}",
            msg.content
        );
        assert!(
            msg.content.contains("Summarize the data."),
            "prompt should include the original command: {}",
            msg.content
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn no_agent_job_saves_output_and_does_not_publish() {
        // no_agent jobs run the script but don't publish — their script
        // stdout IS the deliverable. Output saved to <output_dir>/<job_id>/.
        let tmp = tempfile::tempdir().unwrap();
        let scripts = tmp.path().join("scripts");
        let output = tmp.path().join("cron_output");
        std::fs::create_dir_all(&scripts).unwrap();
        std::fs::write(scripts.join("alert.sh"), "#!/bin/bash\necho disk 92% full\n").unwrap();

        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        let mut j = job("watchdog", "every 1m", None, Some(&past));
        j.script = Some("alert.sh".to_string());
        j.no_agent = true;
        store.add_job(j);

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30))
            .with_scripts_dir(scripts)
            .with_output_dir(output.clone());
        scheduler.tick().await;

        assert!(
            rx.inbound_rx.try_recv().is_err(),
            "no_agent jobs must not publish to the bus"
        );
        let saved = output::latest_job_output(&output, "watchdog")
            .expect("no_agent job must save its script output");
        assert!(
            saved.contains("disk 92% full"),
            "saved output must include script stdout: {saved}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn wake_gate_false_skips_publish_silently() {
        // {"wakeAgent": false} on the last line of script stdout means
        // "nothing to report, skip the agent". For agent jobs that's a
        // skipped publish + success mark.
        let tmp = tempfile::tempdir().unwrap();
        let scripts = tmp.path().join("scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        std::fs::write(
            scripts.join("check.sh"),
            "#!/bin/bash\necho \"all clear\"\necho '{\"wakeAgent\": false}'\n",
        )
        .unwrap();

        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        let mut j = job("checker", "every 1m", None, Some(&past));
        j.command = "Investigate the check.".to_string();
        j.script = Some("check.sh".to_string());
        store.add_job(j);

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30))
            .with_scripts_dir(scripts)
            .with_output_dir(tmp.path().join("cron_output"));
        scheduler.tick().await;

        assert!(
            rx.inbound_rx.try_recv().is_err(),
            "wakeAgent=false must suppress publish"
        );
        let j = scheduler
            .store
            .list_jobs()
            .iter()
            .find(|j| j.id == "checker")
            .unwrap();
        assert_eq!(j.last_status.as_deref(), Some("ok"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn context_from_injects_preceding_job_output() {
        // Plant a saved output for job "feeder", then run "consumer"
        // with context_from=["feeder"]. The consumer's published prompt
        // must include the feeder's saved content.
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("cron_output");
        output::save_job_output(&output, "feeder", "produced: 7 results").unwrap();

        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        let past = (Utc::now() - chrono::Duration::seconds(1)).to_rfc3339();
        let mut j = job("consumer", "every 1m", None, Some(&past));
        j.command = "Process the feeder results.".to_string();
        j.context_from = Some(vec!["feeder".to_string()]);
        store.add_job(j);

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30))
            .with_output_dir(output);
        scheduler.tick().await;

        let msg = rx
            .inbound_rx
            .try_recv()
            .expect("consumer should publish");
        assert!(
            msg.content.contains("produced: 7 results"),
            "context_from must inject upstream output: {}",
            msg.content
        );
        assert!(
            msg.content.contains("Process the feeder results."),
            "original command must remain in the prompt: {}",
            msg.content
        );
    }

    #[cfg(unix)]
    #[test]
    fn save_secures_jobs_json_to_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("jobs.json");
        let mut store = JobStore::new(path.clone());
        store.add_job(job("j1", "every 1h", None, None));
        store.save().unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "jobs.json must be 0600 on Unix");
    }
}
