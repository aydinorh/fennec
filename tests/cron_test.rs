use fennec::bus::MessageBus;
use fennec::cron::{parse_schedule, CronJob, CronScheduler, JobStore, RepeatConfig};

// ---------------------------------------------------------------------------
// Schedule parsing
// ---------------------------------------------------------------------------

#[test]
fn test_parse_schedule_minutes() {
    assert_eq!(parse_schedule("every 30m"), Some(1800));
    assert_eq!(parse_schedule("every 5m"), Some(300));
}

#[test]
fn test_parse_schedule_hours() {
    assert_eq!(parse_schedule("every 1h"), Some(3600));
    assert_eq!(parse_schedule("every 24h"), Some(86400));
}

#[test]
fn test_parse_schedule_days() {
    assert_eq!(parse_schedule("every 7d"), Some(604800));
    assert_eq!(parse_schedule("every 1d"), Some(86400));
}

#[test]
fn test_parse_schedule_seconds() {
    assert_eq!(parse_schedule("every 90s"), Some(90));
}

#[test]
fn test_parse_schedule_bare_duration() {
    // Bare durations (without "every " prefix) are now accepted.
    assert_eq!(parse_schedule("30m"), Some(1800));
    assert_eq!(parse_schedule("1h"), Some(3600));
    assert_eq!(parse_schedule("10s"), Some(10));
}

#[test]
fn test_parse_schedule_invalid() {
    assert_eq!(parse_schedule(""), None);
    assert_eq!(parse_schedule("every"), None);
    assert_eq!(parse_schedule("every abc"), None);
    assert_eq!(parse_schedule("every 30x"), None);
    assert_eq!(parse_schedule("abc"), None);
}

// ---------------------------------------------------------------------------
// JobStore add / remove / list
// ---------------------------------------------------------------------------

fn sample_job(id: &str, name: &str, schedule: &str) -> CronJob {
    CronJob {
        id: id.to_string(),
        name: name.to_string(),
        schedule: schedule.to_string(),
        command: format!("run {name}"),
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
        script: None,
        no_agent: false,
        context_from: None,
    }
}

#[test]
fn test_job_store_add_and_list() {
    // Each scheduler test gets its own tempdir so the cross-process
    // `.tick.lock` file (sibling of jobs.json) doesn't collide when
    // cargo runs multiple integration tests in parallel.
    let tmp = tempfile::tempdir().unwrap();
    let mut store = JobStore::new(tmp.path().join("jobs.json"));

    assert!(store.list_jobs().is_empty());

    store.add_job(sample_job("1", "backup", "every 1h"));
    store.add_job(sample_job("2", "summary", "every 24h"));

    assert_eq!(store.list_jobs().len(), 2);
    assert_eq!(store.list_jobs()[0].id, "1");
    assert_eq!(store.list_jobs()[1].id, "2");
}

#[test]
fn test_job_store_remove() {
    // Each scheduler test gets its own tempdir so the cross-process
    // `.tick.lock` file (sibling of jobs.json) doesn't collide when
    // cargo runs multiple integration tests in parallel.
    let tmp = tempfile::tempdir().unwrap();
    let mut store = JobStore::new(tmp.path().join("jobs.json"));

    store.add_job(sample_job("1", "a", "every 1h"));
    store.add_job(sample_job("2", "b", "every 1h"));

    assert!(store.remove_job("1"));
    assert_eq!(store.list_jobs().len(), 1);
    assert_eq!(store.list_jobs()[0].id, "2");

    // Removing non-existent returns false.
    assert!(!store.remove_job("99"));
}

#[test]
fn test_job_store_get_mut() {
    // Each scheduler test gets its own tempdir so the cross-process
    // `.tick.lock` file (sibling of jobs.json) doesn't collide when
    // cargo runs multiple integration tests in parallel.
    let tmp = tempfile::tempdir().unwrap();
    let mut store = JobStore::new(tmp.path().join("jobs.json"));

    store.add_job(sample_job("1", "a", "every 1h"));
    {
        let job = store.get_mut("1").unwrap();
        job.enabled = false;
    }
    assert!(!store.list_jobs()[0].enabled);

    assert!(store.get_mut("nonexistent").is_none());
}

#[test]
fn test_job_store_persist_and_reload() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_path_buf();

    // Save.
    {
        let mut store = JobStore::new(&path);
        store.add_job(sample_job("1", "persist_test", "every 30m"));
        store.save().unwrap();
    }

    // Reload.
    {
        let mut store = JobStore::new(&path);
        store.load().unwrap();
        assert_eq!(store.list_jobs().len(), 1);
        assert_eq!(store.list_jobs()[0].name, "persist_test");
        assert_eq!(store.list_jobs()[0].schedule, "every 30m");
    }
}

#[test]
fn test_job_store_load_missing_file() {
    let mut store = JobStore::new("/tmp/fennec_test_nonexistent_jobs.json");
    // Should succeed (no file = empty).
    store.load().unwrap();
    assert!(store.list_jobs().is_empty());
}

// ---------------------------------------------------------------------------
// CronScheduler tick -- job due detection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_scheduler_tick_fires_due_job() {
    // Each scheduler test gets its own tempdir so the cross-process
    // `.tick.lock` file (sibling of jobs.json) doesn't collide when
    // cargo runs multiple integration tests in parallel.
    let tmp = tempfile::tempdir().unwrap();
    let mut store = JobStore::new(tmp.path().join("jobs.json"));

    // The scheduler now reads stored `next_run_at` (post-refactor). Plant
    // next_run_at a few seconds in the past — within "every 1m"'s 120s
    // grace window — so the tick fires it instead of fast-forwarding.
    let mut job = sample_job("fire_me", "test_fire", "every 1m");
    let past = chrono::Utc::now() - chrono::Duration::seconds(5);
    job.next_run_at = Some(past.to_rfc3339());
    store.add_job(job);

    let (bus, mut receiver) = MessageBus::new(16);
    let mut scheduler = CronScheduler::new(store, bus, Some(60));

    scheduler.tick().await;

    let msg = receiver
        .inbound_rx
        .try_recv()
        .expect("expected an inbound message from cron");
    assert_eq!(msg.channel, "cron");
    assert!(msg.sender.contains("fire_me"));
    assert_eq!(msg.content, "run test_fire");
}

#[tokio::test]
async fn test_scheduler_tick_skips_recently_run_job() {
    // Each scheduler test gets its own tempdir so the cross-process
    // `.tick.lock` file (sibling of jobs.json) doesn't collide when
    // cargo runs multiple integration tests in parallel.
    let tmp = tempfile::tempdir().unwrap();
    let mut store = JobStore::new(tmp.path().join("jobs.json"));

    // Job that was just run -- should not fire again within the interval.
    let mut job = sample_job("skip_me", "test_skip", "every 1h");
    job.last_run = Some(chrono::Utc::now().to_rfc3339());
    store.add_job(job);

    let (bus, mut receiver) = MessageBus::new(16);
    let mut scheduler = CronScheduler::new(store, bus, Some(60));

    scheduler.tick().await;

    // No message should be published.
    assert!(receiver.inbound_rx.try_recv().is_err());
}

#[tokio::test]
async fn test_scheduler_tick_skips_disabled_job() {
    // Each scheduler test gets its own tempdir so the cross-process
    // `.tick.lock` file (sibling of jobs.json) doesn't collide when
    // cargo runs multiple integration tests in parallel.
    let tmp = tempfile::tempdir().unwrap();
    let mut store = JobStore::new(tmp.path().join("jobs.json"));

    let mut job = sample_job("disabled", "test_disabled", "every 1m");
    job.enabled = false;
    store.add_job(job);

    let (bus, mut receiver) = MessageBus::new(16);
    let mut scheduler = CronScheduler::new(store, bus, Some(60));

    scheduler.tick().await;

    assert!(receiver.inbound_rx.try_recv().is_err());
}

#[tokio::test]
async fn test_scheduler_tick_updates_last_run() {
    // After the refactor, `last_run` is written by `mark_job_run` after a
    // real fire (not anchored on first sight). Set up a job whose
    // `next_run_at` is past so the tick fires it; then verify the
    // post-run bookkeeping (last_run + last_status) is persisted.
    // Each scheduler test gets its own tempdir so the cross-process
    // `.tick.lock` file (sibling of jobs.json) doesn't collide when
    // cargo runs multiple integration tests in parallel.
    let tmp = tempfile::tempdir().unwrap();
    let mut store = JobStore::new(tmp.path().join("jobs.json"));
    let mut job = sample_job("updater", "test_update", "every 1m");
    let past = chrono::Utc::now() - chrono::Duration::seconds(5);
    job.next_run_at = Some(past.to_rfc3339());
    store.add_job(job);

    let (bus, _receiver) = MessageBus::new(16);
    let mut scheduler = CronScheduler::new(store, bus, Some(60));

    scheduler.tick().await;

    let mut reloaded = JobStore::new(tmp.path().join("jobs.json"));
    reloaded.load().unwrap();
    let job = &reloaded.list_jobs()[0];
    assert!(
        job.last_run.is_some(),
        "last_run should be set after firing"
    );
    assert_eq!(job.last_status.as_deref(), Some("ok"));
}
