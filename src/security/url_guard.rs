//! SSRF guard for outbound HTTP from agent tools.
//!
//! Agent tools (`http_request`, `web_fetch`, `pdf_read`, `vision`, `image_info`,
//! …) let the LLM supply URLs. Without restrictions this is an SSRF primitive:
//! the agent can hit `http://169.254.169.254/` (cloud instance metadata),
//! loopback services, internal RFC1918 addresses, or link-local hosts.
//!
//! This module provides three primitives every URL-accepting tool should use:
//!
//! - [`validate_url`] — parse + reject non-http(s), loopback, private,
//!   link-local, multicast, broadcast, IMDS, unique-local, documentation ranges.
//! - [`build_guarded_client`] — a `reqwest::Client` with a custom redirect
//!   policy that re-validates every hop (so an allowed first URL can't 302
//!   into an internal host) and a sane connect/read timeout.
//! - [`read_body_capped`] — stream a response body with a hard byte cap. The
//!   old `.bytes().await` pattern buffered everything before any size check,
//!   so a hostile server returning 10 GB OOM'd the process before the cap
//!   ever ran.
//!
//! **Known limitation**: this does *not* resolve DNS before the request, so a
//! hostile domain whose A record points to a private IP is not caught at the
//! static-host level (reqwest will connect, though the custom redirect policy
//! still re-validates redirected URLs). A resolver-level guard is a
//! follow-up — see the Tier-2 notes.
//!
//! **Opt-out**: users who need the agent to reach loopback / private services
//! (e.g. a local Ollama or an internal API on their LAN) can set
//! `FENNEC_ALLOW_PRIVATE_URLS=1` in the env. The override is intentionally a
//! process-global switch rather than per-tool config because it's a blunt
//! security posture decision.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use futures::StreamExt;
use reqwest::Url;

/// Max redirect hops we follow (each re-validated).
const MAX_REDIRECT_HOPS: usize = 5;

/// Env var that opts out of private-address rejection.
const OVERRIDE_ENV: &str = "FENNEC_ALLOW_PRIVATE_URLS";

/// Validate a URL string for outbound use. Parses and runs the safety
/// checks. Returns the parsed `Url` for callers that want to reuse it.
pub fn validate_url_str(url: &str) -> Result<Url> {
    let parsed = Url::parse(url).with_context(|| format!("invalid URL: {}", url))?;
    validate_url(&parsed)?;
    Ok(parsed)
}

/// Validate a parsed URL. Rejects non-http(s) schemes and hosts in private
/// or reserved ranges (unless `FENNEC_ALLOW_PRIVATE_URLS=1`).
pub fn validate_url(url: &Url) -> Result<()> {
    // Scheme allowlist. No file://, ftp://, gopher://, dict:// etc.
    match url.scheme() {
        "http" | "https" => {}
        other => bail!(
            "URL scheme '{}' not allowed (only http/https)",
            other
        ),
    }

    let host = url
        .host()
        .ok_or_else(|| anyhow!("URL missing host: {}", url))?;

    if private_urls_allowed() {
        return Ok(());
    }

    match host {
        url::Host::Ipv4(ip) => check_ipv4(ip)?,
        url::Host::Ipv6(ip) => check_ipv6(ip)?,
        url::Host::Domain(name) => check_domain(name)?,
    }
    Ok(())
}

fn private_urls_allowed() -> bool {
    matches!(
        std::env::var(OVERRIDE_ENV).as_deref(),
        Ok("1" | "true" | "yes" | "TRUE")
    )
}

/// Build a rejection error that tells the caller (LLM or human) why we
/// refused AND how to opt out. Without this hint a user trying to point
/// the agent at e.g. their local Ollama or Pi-hole sees a flat error and
/// has no path forward — the agent can't learn `FENNEC_ALLOW_PRIVATE_URLS`
/// from a stack trace.
fn blocked(reason: &str) -> anyhow::Error {
    anyhow!(
        "{} (set {}=1 in the agent's env to allow private/internal addresses)",
        reason,
        OVERRIDE_ENV
    )
}

fn check_ipv4(ip: Ipv4Addr) -> Result<()> {
    if ip.is_loopback() {
        return Err(blocked(&format!("URL targets loopback: {}", ip)));
    }
    if ip.is_private() {
        return Err(blocked(&format!("URL targets RFC1918 private range: {}", ip)));
    }
    if ip.is_link_local() {
        return Err(blocked(&format!("URL targets link-local range: {}", ip)));
    }
    if ip.is_multicast() {
        return Err(blocked(&format!("URL targets multicast range: {}", ip)));
    }
    if ip.is_broadcast() {
        return Err(blocked(&format!("URL targets broadcast: {}", ip)));
    }
    if ip.is_unspecified() {
        return Err(blocked(&format!("URL targets unspecified address: {}", ip)));
    }
    if ip.is_documentation() {
        return Err(blocked(&format!("URL targets documentation range: {}", ip)));
    }
    // 100.64.0.0/10 — CGNAT. Not covered by `is_private`.
    let [a, b, _, _] = ip.octets();
    if a == 100 && (64..=127).contains(&b) {
        return Err(blocked(&format!("URL targets CGNAT range: {}", ip)));
    }
    Ok(())
}

fn check_ipv6(ip: Ipv6Addr) -> Result<()> {
    if ip.is_loopback() {
        return Err(blocked(&format!("URL targets IPv6 loopback: {}", ip)));
    }
    if ip.is_multicast() {
        return Err(blocked(&format!("URL targets IPv6 multicast: {}", ip)));
    }
    if ip.is_unspecified() {
        return Err(blocked(&format!("URL targets IPv6 unspecified: {}", ip)));
    }
    // IPv4-mapped IPv6 (::ffff:a.b.c.d) — re-run IPv4 checks on the mapped addr.
    if let Some(v4) = ip.to_ipv4_mapped() {
        check_ipv4(v4)?;
    }
    let segs = ip.segments();
    // fc00::/7 — unique local.
    if (segs[0] & 0xfe00) == 0xfc00 {
        return Err(blocked(&format!("URL targets IPv6 unique-local range: {}", ip)));
    }
    // fe80::/10 — link-local.
    if (segs[0] & 0xffc0) == 0xfe80 {
        return Err(blocked(&format!("URL targets IPv6 link-local range: {}", ip)));
    }
    Ok(())
}

fn check_domain(name: &str) -> Result<()> {
    let lower = name.to_lowercase();
    const BLOCKED_EXACT: &[&str] = &[
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
        "broadcasthost",
        // Cloud instance metadata aliases.
        "metadata.google.internal",
        "metadata.goog",
        "metadata",
    ];
    for bad in BLOCKED_EXACT {
        if lower == *bad {
            return Err(blocked(&format!("URL targets blocked host: {}", name)));
        }
    }
    // .localhost and .internal TLDs.
    if lower.ends_with(".localhost") || lower.ends_with(".internal") {
        return Err(blocked(&format!("URL targets blocked TLD: {}", name)));
    }
    Ok(())
}

/// Default `User-Agent` for the guarded HTTP client.
///
/// Many common targets — notably the GitHub API and Wikipedia (both REST
/// and rendered pages) — reject requests that send **no** `User-Agent` with
/// `403 Forbidden`. Without a default here, `web_fetch` / `http_request`
/// (and the pdf/vision/image helpers that share this client) silently
/// failed on those sites while the `browser` tool — which sets its own UA —
/// succeeded on the same URLs.
///
/// We use the `Mozilla/5.0 (compatible; …)` form that already works for the
/// `browser` tool: it carries a contact URL (satisfying Wikipedia's
/// user-agent policy, which requires a *meaningful* UA) and is honest about
/// being Fennec rather than impersonating a real browser. A `compatible`
/// token is the conventional way for a well-behaved bot to identify itself.
/// Callers that need a different UA (e.g. `http_request` with a
/// user-supplied header) still override this per request, since a
/// request-level header takes precedence over the client default.
pub const GUARDED_CLIENT_USER_AGENT: &str = concat!(
    "Mozilla/5.0 (compatible; Fennec/",
    env!("CARGO_PKG_VERSION"),
    "; +https://fennec.dev)"
);

/// Build a `reqwest::Client` with a redirect policy that re-validates every
/// hop through [`validate_url`]. Caps redirects at `MAX_REDIRECT_HOPS`.
///
/// The client carries a default [`GUARDED_CLIENT_USER_AGENT`] so requests
/// don't 403 on UA-strict sites; see that constant for the rationale.
pub fn build_guarded_client(timeout: Duration) -> reqwest::Client {
    let policy = reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= MAX_REDIRECT_HOPS {
            return attempt.error(TooManyRedirects);
        }
        match validate_url(attempt.url()) {
            Ok(()) => attempt.follow(),
            Err(e) => attempt.error(RedirectBlocked(e.to_string())),
        }
    });
    reqwest::Client::builder()
        .timeout(timeout)
        .user_agent(GUARDED_CLIENT_USER_AGENT)
        .redirect(policy)
        .build()
        .expect("build reqwest client")
}

#[derive(Debug)]
struct TooManyRedirects;
impl std::fmt::Display for TooManyRedirects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "too many redirects")
    }
}
impl std::error::Error for TooManyRedirects {}

#[derive(Debug)]
struct RedirectBlocked(String);
impl std::fmt::Display for RedirectBlocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "redirect blocked: {}", self.0)
    }
}
impl std::error::Error for RedirectBlocked {}

/// Stream a response body into a `Vec<u8>` capped at `max_bytes`. Returns
/// the captured bytes plus a `truncated` flag. Beyond the cap, chunks are
/// discarded but still read so the server isn't left hanging on a
/// backpressured stream.
pub async fn read_body_capped(
    resp: reqwest::Response,
    max_bytes: usize,
) -> Result<(Vec<u8>, bool)> {
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading response body chunk")?;
        if buf.len() >= max_bytes {
            truncated = true;
            // Drain remaining bytes into the void so the peer can finish.
            continue;
        }
        let remaining = max_bytes - buf.len();
        if chunk.len() <= remaining {
            buf.extend_from_slice(&chunk);
        } else {
            buf.extend_from_slice(&chunk[..remaining]);
            truncated = true;
        }
    }
    Ok((buf, truncated))
}

/// Return `IpAddr` if the URL's host is a literal IP; used by tests /
/// diagnostic code paths. Callers should prefer [`validate_url`] over
/// manually inspecting IP literals.
pub fn url_literal_ip(url: &Url) -> Option<IpAddr> {
    match url.host() {
        Some(url::Host::Ipv4(ip)) => Some(IpAddr::V4(ip)),
        Some(url::Host::Ipv6(ip)) => Some(IpAddr::V6(ip)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests touching env vars must serialize because `cargo test` runs in
    /// parallel by default — concurrent set/unset of OVERRIDE_ENV across
    /// tests races and produces flake. This mutex makes those tests
    /// effectively serial.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn with_override<F: FnOnce() -> R, R>(value: &str, f: F) -> R {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::set_var(OVERRIDE_ENV, value);
        }
        let r = f();
        unsafe {
            std::env::remove_var(OVERRIDE_ENV);
        }
        r
    }

    /// Mirror of `with_override` for tests that READ env state but
    /// don't want to set it. Acquires `ENV_MUTEX` (so this test
    /// can't race with a `with_override` caller) and defensively
    /// clears `OVERRIDE_ENV` before running the body — guarding
    /// against the case where a sibling test panicked while
    /// holding the lock, leaving the var set and the mutex
    /// poisoned (the `unwrap_or_else(e.into_inner)` above recovers
    /// the mutex but a stale var would still flip the test result).
    fn with_default_env<F: FnOnce() -> R, R>(f: F) -> R {
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var(OVERRIDE_ENV);
        }
        f()
    }

    #[test]
    fn guarded_client_user_agent_is_meaningful() {
        // Regression for the 403-on-UA-strict-sites bug: the guarded
        // client must carry a non-empty, meaningful User-Agent. It needs
        // a version (so operators can correlate traffic) and a contact URL
        // (Wikipedia's UA policy requires a meaningful agent), and must not
        // be an empty string (the exact condition that 403'd on GitHub /
        // Wikipedia).
        assert!(GUARDED_CLIENT_USER_AGENT.starts_with("Mozilla/5.0 (compatible; Fennec/"));
        assert!(GUARDED_CLIENT_USER_AGENT.contains(env!("CARGO_PKG_VERSION")));
        assert!(GUARDED_CLIENT_USER_AGENT.contains("+https://fennec.dev"));
        assert!(!GUARDED_CLIENT_USER_AGENT.trim().is_empty());
    }

    #[test]
    fn guarded_client_builds() {
        // Building must not panic and the UA must be accepted by reqwest
        // as a valid header value (panics at build otherwise).
        let _ = build_guarded_client(Duration::from_secs(5));
    }

    #[test]
    fn accepts_public_https() {
        validate_url_str("https://example.com/path?q=1").unwrap();
    }

    #[test]
    fn accepts_public_http() {
        validate_url_str("http://example.com").unwrap();
    }

    #[test]
    fn rejects_non_http_scheme() {
        assert!(validate_url_str("file:///etc/passwd").is_err());
        assert!(validate_url_str("ftp://example.com").is_err());
        assert!(validate_url_str("gopher://example.com").is_err());
        assert!(validate_url_str("dict://example.com").is_err());
    }

    #[test]
    fn rejects_loopback_ipv4() {
        with_default_env(|| {
            assert!(validate_url_str("http://127.0.0.1/").is_err());
            assert!(validate_url_str("http://127.0.0.1:8080/admin").is_err());
        });
    }

    #[test]
    fn rejects_imds_literal() {
        with_default_env(|| {
            assert!(validate_url_str("http://169.254.169.254/latest/meta-data/").is_err());
        });
    }

    #[test]
    fn rejects_private_rfc1918() {
        with_default_env(|| {
            assert!(validate_url_str("http://10.0.0.1/").is_err());
            assert!(validate_url_str("http://192.168.1.1/").is_err());
            assert!(validate_url_str("http://172.16.0.1/").is_err());
            // Boundary: 172.32.0.1 is NOT private.
            validate_url_str("http://172.32.0.1/").unwrap();
        });
    }

    #[test]
    fn rejects_cgnat() {
        with_default_env(|| {
            assert!(validate_url_str("http://100.64.0.1/").is_err());
            assert!(validate_url_str("http://100.127.255.254/").is_err());
            // 100.63.x and 100.128.x are NOT CGNAT.
            validate_url_str("http://100.63.0.1/").unwrap();
            validate_url_str("http://100.128.0.1/").unwrap();
        });
    }

    #[test]
    fn rejects_link_local() {
        with_default_env(|| {
            assert!(validate_url_str("http://169.254.0.1/").is_err());
        });
    }

    #[test]
    fn rejects_broadcast_and_unspec() {
        with_default_env(|| {
            assert!(validate_url_str("http://255.255.255.255/").is_err());
            assert!(validate_url_str("http://0.0.0.0/").is_err());
        });
    }

    #[test]
    fn rejects_ipv6_loopback_and_ula() {
        with_default_env(|| {
            assert!(validate_url_str("http://[::1]/").is_err());
            assert!(validate_url_str("http://[fc00::1]/").is_err());
            assert!(validate_url_str("http://[fe80::1]/").is_err());
        });
    }

    #[test]
    fn rejects_ipv4_mapped_ipv6_loopback() {
        with_default_env(|| {
            // ::ffff:127.0.0.1
            assert!(validate_url_str("http://[::ffff:7f00:1]/").is_err());
        });
    }

    #[test]
    fn rejects_literal_localhost_and_metadata() {
        with_default_env(|| {
            assert!(validate_url_str("http://localhost/").is_err());
            assert!(validate_url_str("http://localhost:8080/").is_err());
            assert!(validate_url_str("http://metadata.google.internal/").is_err());
            assert!(validate_url_str("http://foo.internal/").is_err());
            assert!(validate_url_str("http://api.localhost/").is_err());
        });
    }

    /// Reject errors must mention the override env var so an agent (or a
    /// human reading the agent's tool output) can learn how to opt in.
    /// Without the hint, a user pointing the agent at e.g. their local
    /// Ollama or Pi-hole sees a flat error and the agent stalls.
    ///
    /// Acquires `ENV_MUTEX` even though it doesn't write to the env —
    /// it READS `FENNEC_ALLOW_PRIVATE_URLS` transitively through
    /// `validate_url_str`. Without the mutex this test races with
    /// any `with_override` caller: if the sibling test has just set
    /// the var to `"1"`, validate_url_str returns `Ok` for the
    /// loopback / RFC1918 cases below and the `unwrap_err()` panics.
    #[test]
    fn rejection_messages_mention_override_env() {
        with_default_env(|| {
            let cases = [
                "http://127.0.0.1/",
                "http://192.168.1.1/",
                "http://localhost/",
                "http://[fc00::1]/",
                "http://169.254.169.254/",
            ];
            for url in cases {
                let err = validate_url_str(url).unwrap_err().to_string();
                assert!(
                    err.contains("FENNEC_ALLOW_PRIVATE_URLS"),
                    "{} → error missing override hint: {}",
                    url,
                    err
                );
            }
        });
    }

    #[test]
    fn env_override_allows_loopback() {
        with_override("1", || {
            validate_url_str("http://127.0.0.1:8080/").unwrap();
            validate_url_str("http://localhost/").unwrap();
        });
    }

    #[test]
    fn env_override_accepts_bool_variants() {
        with_override("true", || {
            validate_url_str("http://localhost/").unwrap();
        });
        with_override("yes", || {
            validate_url_str("http://localhost/").unwrap();
        });
    }

    #[test]
    fn env_override_other_values_dont_allow() {
        with_override("0", || {
            assert!(validate_url_str("http://localhost/").is_err());
        });
        with_override("false", || {
            assert!(validate_url_str("http://localhost/").is_err());
        });
    }

    #[test]
    fn url_literal_ip_detects_v4_and_v6() {
        let u = Url::parse("http://10.0.0.1/").unwrap();
        assert!(matches!(url_literal_ip(&u), Some(IpAddr::V4(_))));
        let u = Url::parse("http://[::1]/").unwrap();
        assert!(matches!(url_literal_ip(&u), Some(IpAddr::V6(_))));
        let u = Url::parse("http://example.com/").unwrap();
        assert_eq!(url_literal_ip(&u), None);
    }
}
