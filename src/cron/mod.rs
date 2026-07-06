pub mod jobs;
pub use jobs::{parse_schedule, parse_schedule_kind, CronJob, JobStore, ScheduleKind};

use std::collections::HashMap;
use std::str::FromStr;

use chrono::Utc;

use crate::bus::{InboundMessage, MessageBus};

/// Grace window for one-shot timestamp jobs: a job scheduled for HH:MM
/// still fires if the scheduler tick happens within this many seconds
/// after the scheduled moment. Past that, the job is skipped as stale
/// and disabled so the scheduler stops re-evaluating it every tick.
const ONESHOT_GRACE_SECS: i64 = 120;

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

    /// Execute a single scheduler tick: check all enabled jobs and fire those
    /// that are due.
    pub async fn tick(&mut self) {
        let now = Utc::now();
        let mut dirty = false;

        // Snapshot the state we need before we mutate. Holding an
        // iterator over store.list_jobs() while taking a &mut to the
        // store would require gymnastics; a small clone is fine.
        let jobs: Vec<(String, String, String, Option<String>, Option<String>, Option<String>)> =
            self.store
                .list_jobs()
                .iter()
                .filter(|j| j.enabled)
                .map(|j| {
                    (
                        j.id.clone(),
                        j.schedule.clone(),
                        j.command.clone(),
                        j.last_run.clone(),
                        j.origin_channel.clone(),
                        j.origin_chat_id.clone(),
                    )
                })
                .collect();

        for (id, schedule, command, last_run, origin_channel, origin_chat_id) in jobs {
            let kind = match parse_schedule_kind(&schedule) {
                Some(k) => k,
                None => {
                    tracing::warn!("Cron job '{}': invalid schedule '{}'", id, schedule);
                    continue;
                }
            };

            // Resolve the baseline timestamp to measure "elapsed since …"
            // against. If the job has never run (`last_run == None`), anchor
            // the baseline to NOW on first observation — recording it as the
            // synthetic last_run so the first real fire is an interval later.
            // The old code treated `last_run == None` as "fire immediately,"
            // which meant adding a new `"every 24h"` job caused it to fire
            // within 30 seconds instead of 24 hours later.
            let last_dt_opt: Option<chrono::DateTime<chrono::Utc>> = last_run
                .as_deref()
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc));

            // Per-kind due-check + first-tick behavior.
            //
            // Recurring / OneShot / Cron all anchor-and-skip on first
            // observation so a newly-created job doesn't fire on the very
            // next tick ("every 24h" added at noon should fire ~24h later,
            // not 30s later).
            //
            // AtTimestamp is different: it must fire at the scheduled
            // wall-clock time regardless of when the job was created, and
            // is allowed a small grace window to catch up if the scheduler
            // tick lands slightly after the moment.
            let (is_due, is_one_shot) = match &kind {
                ScheduleKind::Recurring { interval_secs } => {
                    let Some(last_dt) = last_dt_opt else {
                        if let Some(job) = self.store.get_mut(&id) {
                            job.last_run = Some(now.to_rfc3339());
                        }
                        dirty = true;
                        continue;
                    };
                    let elapsed = now
                        .signed_duration_since(last_dt)
                        .num_seconds()
                        .max(0) as u64;
                    (elapsed >= *interval_secs, false)
                }
                ScheduleKind::OneShot { delay_secs } => {
                    let Some(last_dt) = last_dt_opt else {
                        if let Some(job) = self.store.get_mut(&id) {
                            job.last_run = Some(now.to_rfc3339());
                        }
                        dirty = true;
                        continue;
                    };
                    let elapsed = now
                        .signed_duration_since(last_dt)
                        .num_seconds()
                        .max(0) as u64;
                    (elapsed >= *delay_secs, true)
                }
                ScheduleKind::Cron(expr) => {
                    let Some(last_dt) = last_dt_opt else {
                        if let Some(job) = self.store.get_mut(&id) {
                            job.last_run = Some(now.to_rfc3339());
                        }
                        dirty = true;
                        continue;
                    };
                    let cron = match croner::Cron::from_str(expr) {
                        Ok(c) => c,
                        Err(e) => {
                            tracing::warn!(
                                "Cron job '{}': invalid cron expression '{}': {}",
                                id, expr, e
                            );
                            continue;
                        }
                    };
                    match cron.find_next_occurrence(&last_dt, false) {
                        Ok(next) => (now >= next, false),
                        Err(e) => {
                            tracing::warn!(
                                "Cron job '{}': failed to compute next occurrence for '{}': {}",
                                id, expr, e
                            );
                            continue;
                        }
                    }
                }
                ScheduleKind::AtTimestamp(ts) => {
                    // Already fired (last_run is set) — done.
                    if last_dt_opt.is_some() {
                        continue;
                    }
                    if *ts > now {
                        // Not yet due — wait without anchoring so we
                        // re-check on the next tick.
                        continue;
                    }
                    if now.signed_duration_since(*ts).num_seconds() > ONESHOT_GRACE_SECS {
                        // Stale (past the grace window) — mark handled +
                        // disable so we stop re-evaluating it.
                        tracing::warn!(
                            "Cron job '{}': skipping stale one-shot scheduled for {} (more than {}s past due)",
                            id, ts, ONESHOT_GRACE_SECS
                        );
                        if let Some(job) = self.store.get_mut(&id) {
                            job.last_run = Some(now.to_rfc3339());
                            job.enabled = false;
                        }
                        dirty = true;
                        continue;
                    }
                    // Within grace window — fire.
                    (true, true)
                }
            };

            if !is_due {
                continue;
            }

            let msg = InboundMessage {
                id: uuid::Uuid::new_v4().to_string(),
                sender: format!("cron:{}", id),
                content: command,
                channel: origin_channel.clone().unwrap_or_else(|| "cron".to_string()),
                chat_id: origin_chat_id.clone().unwrap_or_else(|| format!("cron:{}", id)),
                timestamp: now.timestamp() as u64,
                reply_to: None,
                metadata: {
                    let mut m = HashMap::new();
                    m.insert("source".to_string(), "cron".to_string());
                    m.insert("cron_job_id".to_string(), id.clone());
                    m
                },
            };

            if let Err(e) = self.bus.publish_inbound(msg).await {
                tracing::error!("Cron job '{}': failed to publish message: {}", id, e);
                continue;
            }

            // Update last_run; for one-shot jobs, also disable so the next
            // tick skips them entirely instead of re-firing every tick
            // (the headline bug this branch fixes).
            if let Some(job) = self.store.get_mut(&id) {
                job.last_run = Some(now.to_rfc3339());
                if is_one_shot {
                    job.enabled = false;
                    tracing::debug!("Cron job '{}': one-shot fired, disabling", id);
                }
            }
            dirty = true;
        }

        if dirty {
            if let Err(e) = self.store.save() {
                tracing::error!("Failed to save job store after cron tick: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::MessageBus;

    fn job(id: &str, schedule: &str, last_run: Option<&str>) -> CronJob {
        CronJob {
            id: id.into(),
            name: id.into(),
            schedule: schedule.into(),
            command: "do a thing".into(),
            enabled: true,
            last_run: last_run.map(|s| s.to_string()),
            origin_channel: None,
            origin_chat_id: None,
        }
    }

    #[tokio::test]
    async fn newly_added_recurring_job_does_not_fire_immediately() {
        // Regression: the old tick() treated last_run=None as "fire now"
        // which meant a newly-added `"every 24h"` job fired within the
        // first 30-second scheduler tick.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        store.add_job(job("j1", "every 24h", None));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        // No inbound message should have been published.
        assert!(
            rx.inbound_rx.try_recv().is_err(),
            "newly added job fired on first tick — regression"
        );

        // last_run must have been anchored so future ticks can compute
        // elapsed correctly.
        let anchored = scheduler
            .store
            .list_jobs()
            .iter()
            .find(|j| j.id == "j1")
            .unwrap()
            .last_run
            .clone();
        assert!(anchored.is_some(), "last_run must be anchored");
    }

    #[tokio::test]
    async fn recurring_job_fires_after_interval_elapsed() {
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        // last_run 2 hours ago — every 1h schedule should fire.
        let old = (Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
        store.add_job(job("j2", "every 1h", Some(&old)));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        let msg = rx
            .inbound_rx
            .try_recv()
            .expect("recurring job past its interval should fire");
        assert_eq!(msg.sender, "cron:j2");
    }

    #[tokio::test]
    async fn one_shot_disables_after_firing() {
        // Regression for the headline T2-B bug: `"5m"` used to fire on
        // every scheduler tick after the first because `elapsed >= 300`
        // stayed true forever. The scheduler must now set enabled=false
        // after the first fire.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        // last_run 1 hour ago — one-shot "5m" should have fired long ago.
        let old = (Utc::now() - chrono::Duration::hours(1)).to_rfc3339();
        store.add_job(job("once", "5m", Some(&old)));

        let (bus, mut rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        scheduler.tick().await;

        // Fires once.
        let msg = rx.inbound_rx.try_recv().expect("one-shot fires once");
        assert_eq!(msg.sender, "cron:once");

        // After firing, the job must be disabled.
        let disabled = scheduler
            .store
            .list_jobs()
            .iter()
            .find(|j| j.id == "once")
            .unwrap();
        assert!(!disabled.enabled, "one-shot job must be disabled after firing");

        // Second tick: must NOT fire again.
        scheduler.tick().await;
        assert!(
            rx.inbound_rx.try_recv().is_err(),
            "one-shot re-fired — regression"
        );
    }

    #[tokio::test]
    async fn invalid_schedule_does_not_crash_scheduler() {
        // A job with an unparseable schedule should be logged and skipped,
        // not crash the whole tick.
        let tmp = tempfile::tempdir().unwrap();
        let mut store = JobStore::new(tmp.path().join("jobs.json"));
        store.add_job(job("bad", "every 0s", None));
        store.add_job(job("good", "every 1h", None));

        let (bus, _rx) = MessageBus::new(16);
        let mut scheduler = CronScheduler::new(store, bus, Some(30));
        // Must not panic. No assertion needed beyond "this returns".
        scheduler.tick().await;
    }
}
