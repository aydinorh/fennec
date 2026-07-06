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
//! - [`validate_url_str_resolved`] — everything `validate_url` does PLUS a
//!   DNS resolution check: domain hosts are resolved and every returned
//!   address is re-validated, so a hostile domain whose A record points at
//!   a private IP is caught before any connection is made. URL-accepting
//!   tools should prefer this (they're all async).
//!
//! **Known limitation**: the resolve-then-connect sequence is not atomic — a
//! DNS rebinding attacker who flips the record between our lookup and
//! reqwest's own lookup can still slip through. Closing that fully requires
//! pinning the resolved address into the connector; follow-up noted in the
//! plan. The pre-resolve check still removes the entire static-DNS attack
//! class.
//!
//! **Opt-out**: users who need the agent to reach loopback / private services
//! (e.g. a local Ollama or an internal API on their LAN) can set
//! `FENNEC_ALLOW_PRIVATE_URLS=1` in the env. The override is intentionally a
//! process-global switch rather than per-tool config because it's a blunt
//! security posture decision. The override does NOT unblock cloud-metadata
//! endpoints (IMDS, metadata.google.internal): leaking instance credentials
//! is never what "let me reach my LAN" means, so those stay on an
//! always-blocked floor.

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

    // Cloud-metadata floor: ALWAYS blocked, even with the private-URL
    // override. The override exists for "reach my LAN/loopback", never
    // for "hand instance credentials to the model".
    match &host {
        url::Host::Ipv4(ip) => check_metadata_floor_ip(IpAddr::V4(*ip))?,
        url::Host::Ipv6(ip) => check_metadata_floor_ip(IpAddr::V6(*ip))?,
        url::Host::Domain(name) => check_metadata_floor_domain(name)?,
    }

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

/// Validate a URL string INCLUDING a DNS-resolution check for domain
/// hosts: every address the name resolves to is run through the same
/// IP checks (and the metadata floor), so a domain whose A/AAAA record
/// points at a private or metadata address is rejected before any
/// connection. IP-literal hosts skip the lookup.
pub async fn validate_url_str_resolved(url: &str) -> Result<Url> {
    let parsed = validate_url_str(url)?;
    validate_resolved_addrs(&parsed).await?;
    Ok(parsed)
}

/// DNS half of [`validate_url_str_resolved`]. Domain hosts are resolved
/// via the system resolver; resolution failure is an error (the request
/// would fail anyway, and silently skipping the check would let a
/// flaky-then-poisoned resolver bypass it).
async fn validate_resolved_addrs(url: &Url) -> Result<()> {
    let Some(url::Host::Domain(name)) = url.host() else {
        return Ok(()); // IP literal — already fully checked statically.
    };
    let port = url.port_or_known_default().unwrap_or(443);
    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((name, port))
        .await
        .with_context(|| format!("could not resolve host '{}'", name))?
        .collect();
    if addrs.is_empty() {
        bail!("host '{}' resolved to no addresses", name);
    }
    let allow_private = private_urls_allowed();
    for addr in addrs {
        let ip = addr.ip();
        // Metadata floor applies even with the private-URL override.
        check_metadata_floor_ip(ip)
            .with_context(|| format!("host '{}' resolves to {}", name, ip))?;
        if allow_private {
            continue;
        }
        let check = match ip {
            IpAddr::V4(v4) => check_ipv4(v4),
            IpAddr::V6(v6) => check_ipv6(v6),
        };
        check.with_context(|| format!("host '{}' resolves to {}", name, ip))?;
    }
    Ok(())
}

/// Cloud-metadata addresses that stay blocked regardless of the
/// private-URL override: AWS/GCP/Azure IMDS (169.254.169.254), the AWS
/// IPv6 IMDS endpoint, and their IPv4-mapped forms.
fn check_metadata_floor_ip(ip: IpAddr) -> Result<()> {
    const IMDS_V4: Ipv4Addr = Ipv4Addr::new(169, 254, 169, 254);
    // fd00:ec2::254 — AWS IMDS over IPv6.
    const IMDS_V6: Ipv6Addr = Ipv6Addr::new(0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x254);
    let blocked = match ip {
        IpAddr::V4(v4) => v4 == IMDS_V4,
        IpAddr::V6(v6) => v6 == IMDS_V6 || v6.to_ipv4_mapped() == Some(IMDS_V4),
    };
    if blocked {
        bail!(
            "URL targets a cloud instance-metadata endpoint ({}) — always blocked, the private-URL override does not apply",
            ip
        );
    }
    Ok(())
}

/// Hostname aliases for cloud metadata services — same always-blocked
/// floor as [`check_metadata_floor_ip`].
fn check_metadata_floor_domain(name: &str) -> Result<()> {
    let lower = name.to_lowercase();
    const METADATA_HOSTS: &[&str] = &["metadata.google.internal", "metadata.goog", "metadata"];
    if METADATA_HOSTS.iter().any(|h| lower == *h) {
        bail!(
            "URL targets a cloud instance-metadata host ({}) — always blocked, the private-URL override does not apply",
            name
        );
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

/// Build a `reqwest::Client` with a redirect policy that re-validates every
/// hop through [`validate_url`]. Caps redirects at `MAX_REDIRECT_HOPS`.
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
            // NB: 169.254.169.254 is NOT in this list — IMDS now gets the
            // metadata-floor message, which deliberately does NOT suggest
            // the override (see metadata_floor_error_does_not_suggest_override).
            let cases = [
                "http://127.0.0.1/",
                "http://192.168.1.1/",
                "http://localhost/",
                "http://[fc00::1]/",
                "http://169.254.0.1/",
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

    /// The private-URL override exists for "reach my LAN/loopback" — it
    /// must NOT unblock cloud instance-metadata endpoints, which leak
    /// instance credentials.
    #[test]
    fn env_override_does_not_unblock_metadata_floor() {
        with_override("1", || {
            // IMDS IPv4, AWS IMDS IPv6, IPv4-mapped form, and hostname aliases.
            assert!(validate_url_str("http://169.254.169.254/latest/meta-data/").is_err());
            assert!(validate_url_str("http://[fd00:ec2::254]/").is_err());
            assert!(validate_url_str("http://[::ffff:169.254.169.254]/").is_err());
            assert!(validate_url_str("http://metadata.google.internal/computeMetadata/").is_err());
            assert!(validate_url_str("http://metadata.goog/").is_err());
            assert!(validate_url_str("http://metadata/").is_err());
        });
    }

    #[test]
    fn metadata_floor_error_does_not_suggest_override() {
        with_default_env(|| {
            let err = validate_url_str("http://169.254.169.254/")
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("always blocked"),
                "floor error must say the override doesn't apply: {err}"
            );
        });
    }

    #[tokio::test]
    async fn resolved_validation_passes_ip_literals_without_dns() {
        with_default_env(|| {});
        // Public IP literal: no DNS lookup involved, passes statically.
        validate_url_str_resolved("http://93.184.216.34/").await.unwrap();
    }

    #[tokio::test]
    async fn resolved_validation_blocks_domains_resolving_private() {
        // localtest.me publicly resolves to 127.0.0.1 — the classic
        // DNS-based SSRF vector the static check can't see. Skip
        // gracefully when offline (resolution failure is an error too,
        // but a DIFFERENT one; only assert when resolution succeeded).
        let _guard = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var(OVERRIDE_ENV);
        }
        match validate_url_str_resolved("http://localtest.me/").await {
            Err(e) => {
                let msg = e.to_string();
                // Either blocked (resolved to loopback) or unresolvable
                // (offline) — both are rejections; only the blocked case
                // proves the DNS check, but offline must not fail CI.
                assert!(
                    msg.contains("resolves to") || msg.contains("could not resolve"),
                    "unexpected error shape: {msg}"
                );
            }
            Ok(_) => panic!("localtest.me must not validate (resolves to 127.0.0.1)"),
        }
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
