pub mod jobs;
pub use jobs::{
    compute_next_run, grace_seconds_for, parse_schedule, parse_schedule_kind,
    schedule_display_for, AmbiguousJobReference, CronJob, JobStore, JobUpdates, RepeatConfig,
    ScheduleKind, ONESHOT_GRACE_SECS,
};

use std::collections::HashMap;

use chrono::Utc;

use crate::bus::{InboundMessage, MessageBus};

pub struct CronScheduler {
    store: JobStore,
    bus: MessageBus,
    check_interval_secs: u64,
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
        }
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

    /// Execute a single scheduler tick: ask the store for all due jobs,
    /// advance their `next_run_at` (at-most-once for recurring jobs),
    /// publish each job's command to the bus, and mark the run.
    ///
    /// The heavy lifting — recovery of missing `next_run_at`, stale
    /// fast-forward, repeat counting, lifecycle state — lives in
    /// [`JobStore`] so this loop stays thin and the same semantics work
    /// for any future tick entry point (CLI, gateway, tests).
    pub async fn tick(&mut self) {
        let due_jobs = self.store.get_due_jobs();
        if due_jobs.is_empty() {
            return;
        }

        for job in due_jobs {
            // Pre-advance recurring jobs' next_run_at BEFORE we publish
            // anything. If we crash between publish and mark_job_run, we
            // lose one run instead of refiring on every restart —
            // at-most-once semantics for recurring schedules.
            if let Err(e) = self.store.advance_next_run(&job.id) {
                tracing::error!("Cron job '{}': advance_next_run failed: {}", job.id, e);
            }

            let now = Utc::now();
            let msg = InboundMessage {
                id: uuid::Uuid::new_v4().to_string(),
                sender: format!("cron:{}", job.id),
                content: job.command.clone(),
                channel: job
                    .origin_channel
                    .clone()
                    .unwrap_or_else(|| "cron".to_string()),
                chat_id: job
                    .origin_chat_id
                    .clone()
                    .unwrap_or_else(|| format!("cron:{}", job.id)),
                timestamp: now.timestamp() as u64,
                reply_to: None,
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), "cron".to_string());
                    m.insert("cron_job_id".to_string(), job.id.clone());
                    m
                },
            };

            let (success, err_msg) = match self.bus.publish_inbound(msg).await {
                Ok(_) => (true, None),
                Err(e) => {
                    tracing::error!("Cron job '{}': failed to publish message: {}", job.id, e);
                    (false, Some(e.to_string()))
                }
            };

            // mark_job_run handles last_run / last_status / last_error /
            // last_delivery_error, repeat counting (+ auto-remove), and
            // recomputes next_run_at + state. delivery_error is None for
            // the bus-publish path — true remote-delivery error tracking
            // arrives with the delivery system (later PR).
            if let Err(e) =
                self.store
                    .mark_job_run(&job.id, success, err_msg.as_deref(), None)
            {
                tracing::error!("Cron job '{}': mark_job_run failed: {}", job.id, e);
            }
        }
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
