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
}

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
