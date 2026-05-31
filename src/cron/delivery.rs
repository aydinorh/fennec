//! Cron job delivery routing.
//!
//! Parses a job's `deliver` string into concrete delivery targets and
//! wraps responses with a job-name header so messages on busy channels
//! are recognisable. Mirrors the upstream's `_resolve_delivery_targets`
//! / `_deliver_result` semantics adapted for Fennec's bus architecture.
//!
//! Supported `deliver` tokens (single or comma-separated):
//! - `"local"` — no delivery; output saved locally only.
//! - `"origin"` — deliver to where the job was created.
//! - `"<platform>"` — deliver to the platform's configured home channel
//!   (`FENNEC_<PLATFORM>_HOME_CHANNEL` env, with the per-platform
//!   override map below for non-`HOME_CHANNEL`-suffixed names).
//! - `"<platform>:<chat_id>"` (optionally `:<thread_id>`) — explicit
//!   target.
//! - `"all"` — fan-out to every platform with a configured home channel.

/// Sentinel a cron job's agent (or `no_agent` script) can emit to
/// suppress delivery for this tick. Output is still saved locally.
/// Matches the upstream's `SILENT_MARKER`.
pub const SILENT_MARKER: &str = "[SILENT]";

/// One concrete delivery target resolved from a `deliver` token at
/// fire time. `thread_id` is optional and platform-specific (Matrix
/// rooms don't use it; Telegram + some others do).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeliveryTarget {
    pub platform: String,
    pub chat_id: String,
    pub thread_id: Option<String>,
}

/// Origin info captured at create time (the channel + chat the job
/// was created from). Used to resolve the `"origin"` token.
#[derive(Debug, Clone)]
pub struct JobOrigin<'a> {
    pub channel: &'a str,
    pub chat_id: &'a str,
}

/// Default `deliver` value at create time. Matches the upstream's
/// `deliver = "origin" if origin else "local"`.
pub fn default_deliver_for(origin: Option<&JobOrigin<'_>>) -> &'static str {
    if origin.is_some() {
        "origin"
    } else {
        "local"
    }
}

/// Test whether a response should suppress delivery. The agent (or a
/// `no_agent` script) signals "nothing to report" by emitting the
/// [`SILENT_MARKER`] anywhere in the trimmed response — matches the
/// upstream's `SILENT_MARKER in deliver_content.strip().upper()`
/// check (case-insensitive containment).
pub fn is_silent_response(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return false;
    }
    trimmed
        .to_ascii_uppercase()
        .contains(SILENT_MARKER)
}

/// Wrap a cron job's response with a header naming the job + ID so
/// users on busy channels can tell which scheduled task spoke. Matches
/// the upstream's `wrap_response` block in `_deliver_result`.
pub fn wrap_response(content: &str, job_name: &str, job_id: &str) -> String {
    let label = if job_name.is_empty() { "cron job" } else { job_name };
    format!(
        "Cron Response: {label}\n(job_id: {job_id})\n-------------\n\n{content}\n\n\
         To stop or manage this job, send me a new message (e.g. \"stop reminder {label}\")."
    )
}

/// Resolve `deliver` into concrete targets. Returns an empty vec when
/// `deliver == "local"` (no delivery) or when no token resolves.
/// `origin` provides the channel/chat for the `"origin"` token.
pub fn parse_deliver(deliver: &str, origin: Option<&JobOrigin<'_>>) -> Vec<DeliveryTarget> {
    let trimmed = deliver.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("local") {
        return Vec::new();
    }

    let raw_tokens: Vec<String> = trimmed
        .split(',')
        .map(|p| p.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Expand routing-intent tokens (currently just `all`) into the
    // concrete platform set.
    let mut expanded: Vec<String> = Vec::new();
    for token in raw_tokens {
        let lower = token.to_ascii_lowercase();
        if lower == "all" {
            for platform in iter_home_target_platforms() {
                if home_channel_for(platform).is_some() {
                    expanded.push(platform.to_string());
                }
            }
        } else if lower == "local" {
            // Mixed `local,telegram` doesn't make sense — silently
            // drop the local component when explicit targets follow.
        } else {
            expanded.push(token);
        }
    }

    let mut seen: std::collections::HashSet<(String, String, Option<String>)> =
        std::collections::HashSet::new();
    let mut targets = Vec::new();
    for token in expanded {
        if let Some(target) = resolve_single_token(&token, origin) {
            let key = (
                target.platform.to_ascii_lowercase(),
                target.chat_id.clone(),
                target.thread_id.clone(),
            );
            if seen.insert(key) {
                targets.push(target);
            }
        }
    }
    targets
}

/// Resolve a single token (`"origin"`, `"telegram"`, or
/// `"telegram:-1001:17"`) to a concrete [`DeliveryTarget`]. Returns
/// `None` when the token names an unknown platform or a platform with
/// no configured home channel.
fn resolve_single_token(token: &str, origin: Option<&JobOrigin<'_>>) -> Option<DeliveryTarget> {
    if token.eq_ignore_ascii_case("origin") {
        let o = origin?;
        return Some(DeliveryTarget {
            platform: o.channel.to_string(),
            chat_id: o.chat_id.to_string(),
            thread_id: None,
        });
    }

    // `<platform>:<chat_id>` or `<platform>:<chat_id>:<thread_id>`.
    if let Some((platform, rest)) = token.split_once(':') {
        let (chat_id, thread_id) = match rest.split_once(':') {
            Some((c, t)) => (c.to_string(), Some(t.to_string())),
            None => (rest.to_string(), None),
        };
        if platform.trim().is_empty() || chat_id.trim().is_empty() {
            return None;
        }
        return Some(DeliveryTarget {
            platform: platform.to_string(),
            chat_id,
            thread_id,
        });
    }

    // Bare platform name: look up its home channel from env.
    let platform = token.trim();
    if platform.is_empty() {
        return None;
    }
    let chat_id = home_channel_for(platform)?;
    let thread_id = home_thread_for(platform);
    Some(DeliveryTarget {
        platform: platform.to_string(),
        chat_id,
        thread_id,
    })
}

/// Built-in platforms whose home channel is read from a
/// `FENNEC_<PLATFORM>_HOME_CHANNEL` env var. The list mirrors the
/// channels Fennec ships with — plugins that register new channels
/// can be added here (or via a registry in a future PR).
pub fn iter_home_target_platforms() -> impl Iterator<Item = &'static str> {
    [
        "telegram",
        "discord",
        "slack",
        "signal",
        "matrix",
        "email",
        "webhook",
        "whatsapp",
    ]
    .into_iter()
}

/// Resolve a platform's home-channel env var name. A few platforms
/// follow upstream conventions (Matrix uses `HOME_ROOM`, Email uses
/// `HOME_ADDRESS`); everything else uses `HOME_CHANNEL`.
fn home_channel_env_var(platform: &str) -> String {
    let lower = platform.to_ascii_lowercase();
    let suffix = match lower.as_str() {
        "matrix" => "HOME_ROOM",
        "email" => "HOME_ADDRESS",
        _ => "HOME_CHANNEL",
    };
    format!("FENNEC_{}_{}", lower.to_ascii_uppercase(), suffix)
}

/// Look up the configured home channel ID for `platform`. Returns
/// `None` when the env var is unset or empty.
pub fn home_channel_for(platform: &str) -> Option<String> {
    let var = home_channel_env_var(platform);
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Look up the optional home-channel thread / topic ID for
/// `platform` — read from `FENNEC_<PLATFORM>_HOME_CHANNEL_THREAD_ID`
/// (or `_HOME_ROOM_THREAD_ID` for Matrix). Used by platforms that
/// route into forum-style topics so cron deliveries don't land in a
/// system-only lobby.
pub fn home_thread_for(platform: &str) -> Option<String> {
    let base = home_channel_env_var(platform);
    let var = format!("{base}_THREAD_ID");
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Serialise a list of targets to the comma-separated form used in
/// `cron_deliver_targets` metadata: `platform:chat_id[:thread_id]`
/// per target.
pub fn encode_targets(targets: &[DeliveryTarget]) -> String {
    targets
        .iter()
        .map(|t| match &t.thread_id {
            Some(thread) => format!("{}:{}:{}", t.platform, t.chat_id, thread),
            None => format!("{}:{}", t.platform, t.chat_id),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse the `cron_deliver_targets` metadata back into a list of
/// targets. Inverse of [`encode_targets`]. Malformed entries are
/// silently dropped so a corrupted metadata value can't crash the
/// consumer.
pub fn decode_targets(encoded: &str) -> Vec<DeliveryTarget> {
    encoded
        .split(',')
        .filter_map(|raw| {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return None;
            }
            let mut parts = trimmed.splitn(3, ':');
            let platform = parts.next()?.trim();
            let chat_id = parts.next()?.trim();
            let thread_id = parts.next().map(|s| s.trim().to_string());
            if platform.is_empty() || chat_id.is_empty() {
                return None;
            }
            Some(DeliveryTarget {
                platform: platform.to_string(),
                chat_id: chat_id.to_string(),
                thread_id,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises tests that mutate `FENNEC_*_HOME_CHANNEL` env vars.
    /// Cargo runs lib tests in parallel by default, so a concurrent
    /// `set_var("FENNEC_DISCORD_HOME_CHANNEL", "X")` from one test and
    /// `..="Y"` from another would flap reads in a third. Acquire this
    /// guard for the lifetime of any test that touches env state.
    static ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn origin(channel: &str, chat: &str) -> JobOrigin<'static> {
        // SAFETY: the test holds the &str source for the lifetime of
        // the resulting JobOrigin via 'static refs — `Box::leak` here
        // is the simplest way to get a 'static lifetime in a test.
        let channel: &'static str = Box::leak(channel.to_string().into_boxed_str());
        let chat_id: &'static str = Box::leak(chat.to_string().into_boxed_str());
        JobOrigin { channel, chat_id }
    }

    #[test]
    fn default_deliver_origin_when_present() {
        let o = origin("telegram", "12345");
        assert_eq!(default_deliver_for(Some(&o)), "origin");
        assert_eq!(default_deliver_for(None), "local");
    }

    #[test]
    fn is_silent_detects_marker_and_trims() {
        assert!(is_silent_response("[SILENT]"));
        assert!(is_silent_response("  [SILENT]  "));
        assert!(is_silent_response("[silent]"));
        assert!(is_silent_response("nothing to report [SILENT]"));
        assert!(!is_silent_response(""));
        assert!(!is_silent_response("status nominal"));
    }

    #[test]
    fn wrap_response_includes_job_name_and_id() {
        let wrapped = wrap_response("hello world", "Daily Status", "abc123");
        assert!(wrapped.starts_with("Cron Response: Daily Status\n"));
        assert!(wrapped.contains("(job_id: abc123)"));
        assert!(wrapped.contains("hello world"));
        // Empty job name falls back to "cron job".
        let wrapped = wrap_response("hello", "", "id");
        assert!(wrapped.contains("Cron Response: cron job"));
    }

    #[test]
    fn parse_deliver_local_returns_empty() {
        let o = origin("telegram", "12345");
        assert!(parse_deliver("local", Some(&o)).is_empty());
        assert!(parse_deliver("", Some(&o)).is_empty());
        assert!(parse_deliver("   ", None).is_empty());
        assert!(parse_deliver("LOCAL", None).is_empty());
    }

    #[test]
    fn parse_deliver_origin_uses_job_origin() {
        let o = origin("signal", "+15551234567");
        let targets = parse_deliver("origin", Some(&o));
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].platform, "signal");
        assert_eq!(targets[0].chat_id, "+15551234567");
        assert!(targets[0].thread_id.is_none());
    }

    #[test]
    fn parse_deliver_origin_no_origin_yields_nothing() {
        assert!(parse_deliver("origin", None).is_empty());
    }

    #[test]
    fn parse_deliver_explicit_platform_chat_thread() {
        let targets = parse_deliver("telegram:-1001:17", None);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].platform, "telegram");
        assert_eq!(targets[0].chat_id, "-1001");
        assert_eq!(targets[0].thread_id.as_deref(), Some("17"));
    }

    #[test]
    fn parse_deliver_explicit_platform_chat_only() {
        let targets = parse_deliver("discord:9000", None);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].chat_id, "9000");
        assert!(targets[0].thread_id.is_none());
    }

    #[test]
    fn parse_deliver_comma_list_deduplicates() {
        // Same explicit target listed twice collapses.
        let targets = parse_deliver("discord:9000,discord:9000", None);
        assert_eq!(targets.len(), 1);
        // Different chat IDs stay separate.
        let targets = parse_deliver("discord:9000,discord:9001", None);
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn parse_deliver_bare_platform_uses_env() {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let prev = std::env::var("FENNEC_DISCORD_HOME_CHANNEL").ok();
        unsafe {
            std::env::set_var("FENNEC_DISCORD_HOME_CHANNEL", "1234567890");
        }
        let targets = parse_deliver("discord", None);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].platform, "discord");
        assert_eq!(targets[0].chat_id, "1234567890");
        unsafe {
            match prev {
                Some(v) => std::env::set_var("FENNEC_DISCORD_HOME_CHANNEL", v),
                None => std::env::remove_var("FENNEC_DISCORD_HOME_CHANNEL"),
            }
        }
    }

    #[test]
    fn parse_deliver_all_expands_to_configured_platforms() {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let prev_t = std::env::var("FENNEC_TELEGRAM_HOME_CHANNEL").ok();
        let prev_d = std::env::var("FENNEC_DISCORD_HOME_CHANNEL").ok();
        unsafe {
            std::env::set_var("FENNEC_TELEGRAM_HOME_CHANNEL", "tchat");
            std::env::set_var("FENNEC_DISCORD_HOME_CHANNEL", "dchat");
        }
        let targets = parse_deliver("all", None);
        let platforms: Vec<_> = targets.iter().map(|t| t.platform.as_str()).collect();
        assert!(platforms.contains(&"telegram"));
        assert!(platforms.contains(&"discord"));
        unsafe {
            match prev_t {
                Some(v) => std::env::set_var("FENNEC_TELEGRAM_HOME_CHANNEL", v),
                None => std::env::remove_var("FENNEC_TELEGRAM_HOME_CHANNEL"),
            }
            match prev_d {
                Some(v) => std::env::set_var("FENNEC_DISCORD_HOME_CHANNEL", v),
                None => std::env::remove_var("FENNEC_DISCORD_HOME_CHANNEL"),
            }
        }
    }

    #[test]
    fn matrix_uses_home_room_env_naming() {
        let _guard = ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let prev = std::env::var("FENNEC_MATRIX_HOME_ROOM").ok();
        unsafe {
            std::env::set_var("FENNEC_MATRIX_HOME_ROOM", "!room:example.org");
        }
        let targets = parse_deliver("matrix", None);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].chat_id, "!room:example.org");
        unsafe {
            match prev {
                Some(v) => std::env::set_var("FENNEC_MATRIX_HOME_ROOM", v),
                None => std::env::remove_var("FENNEC_MATRIX_HOME_ROOM"),
            }
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        let targets = vec![
            DeliveryTarget {
                platform: "telegram".to_string(),
                chat_id: "-1001".to_string(),
                thread_id: Some("17".to_string()),
            },
            DeliveryTarget {
                platform: "discord".to_string(),
                chat_id: "9000".to_string(),
                thread_id: None,
            },
        ];
        let encoded = encode_targets(&targets);
        assert_eq!(encoded, "telegram:-1001:17,discord:9000");
        let decoded = decode_targets(&encoded);
        assert_eq!(decoded, targets);
    }

    #[test]
    fn decode_drops_malformed_entries() {
        let decoded = decode_targets("telegram:9000,broken,discord:1");
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].platform, "telegram");
        assert_eq!(decoded[1].platform, "discord");
    }
}
