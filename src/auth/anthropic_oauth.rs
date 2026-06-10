use std::path::Path;

use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const REDIRECT_URI: &str = "https://console.anthropic.com/oauth/code/callback";
const SCOPE: &str = "org:create_api_key user:profile user:inference";
const TOKEN_ENDPOINT: &str = "https://console.anthropic.com/v1/oauth/token";
const AUTH_ENDPOINT: &str = "https://claude.ai/oauth/authorize";
const TOKEN_FILE: &str = ".anthropic_oauth.json";

/// Persisted OAuth credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthCredentials {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64, // unix timestamp in seconds
}

/// Generate a cryptographically random PKCE verifier (43-128 chars, base64url).
fn generate_verifier() -> String {
    let mut buf = [0u8; 32];
    rand::rng().fill(&mut buf);
    base64url_encode(&buf)
}

/// Compute the S256 code challenge from the verifier.
fn compute_challenge(verifier: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let hash = hasher.finalize();
    base64url_encode(&hash)
}

/// Base64url-encode without padding (RFC 7636 §4.2).
///
/// Was a hand-rolled implementation that had two issues:
///   1. Subtle padding bug at certain input lengths (worked at the
///      32-byte verifier length we use today, but a foot-gun if we
///      ever change input sizes).
///   2. The `replace('+', "-").replace('/', "_")` post-pass walks the
///      string twice, allocating each time.
///
/// Now delegated to the `base64` crate's `URL_SAFE_NO_PAD` engine —
/// same RFC 7636 output, well-tested, no allocator sins.
fn base64url_encode(input: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(input)
}

/// Open a URL in the user's default browser.
fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let cmd = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let cmd = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let cmd = std::process::Command::new("cmd")
        .args(["/C", "start", url])
        .spawn();

    if let Err(e) = cmd {
        tracing::warn!("Failed to open browser: {e}");
    }
}

/// Run the full OAuth PKCE login flow interactively.
///
/// Opens the browser, waits for the user to paste the authorization code,
/// exchanges it for tokens, and saves them to `fennec_home/.anthropic_oauth.json`.
pub fn run_oauth_login(fennec_home: &Path) -> Result<OAuthCredentials> {
    // 1. Generate PKCE pair.
    let verifier = generate_verifier();
    let challenge = compute_challenge(&verifier);

    // 2. Build authorization URL.
    // Build authorization URL — matches the upstream's exact parameter set.
    // The "code=true" param and "state=verifier" are required by Anthropic.
    let auth_url = format!(
        "{}?code=true&client_id={}&response_type=code&redirect_uri={}&scope={}&code_challenge={}&code_challenge_method=S256&state={}",
        AUTH_ENDPOINT,
        CLIENT_ID,
        urlencoded(REDIRECT_URI),
        urlencoded(SCOPE),
        challenge,
        urlencoded(&verifier),
    );

    // 3. Show URL and prompt user.
    println!();
    println!("  Open this link in your browser:");
    println!();
    println!("  {}", auth_url);
    println!();
    open_browser(&auth_url);
    println!("After authorizing, you'll see a code. Paste it below.");
    println!();

    print!("Authorization code: ");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let mut raw_input = String::new();
    #[cfg(unix)]
    {
        use std::io::BufRead;
        if let Ok(tty) = std::fs::File::open("/dev/tty") {
            let mut reader = std::io::BufReader::new(tty);
            reader.read_line(&mut raw_input).context("reading authorization code")?;
        } else {
            std::io::stdin().read_line(&mut raw_input).context("reading authorization code")?;
        }
    }
    #[cfg(not(unix))]
    {
        std::io::stdin().read_line(&mut raw_input).context("reading authorization code")?;
    }
    let raw_input = raw_input.trim();
    if raw_input.is_empty() {
        anyhow::bail!("Empty authorization code");
    }

    // The callback page may return "code#state" — split on '#'.
    let parts: Vec<&str> = raw_input.splitn(2, '#').collect();
    let code = parts[0];
    let state = if parts.len() > 1 { parts[1] } else { "" };

    // CSRF protection: the `state` we sent in the auth URL is the PKCE
    // verifier (Anthropic's required form). The callback hands us back
    // `code#state` — we must reject if the state doesn't match what we
    // sent, otherwise an attacker who tricked the user into pasting a
    // `code#state` from a different OAuth flow could redeem the code
    // through our running `run_oauth_login`. PKCE alone protects token
    // redemption (the verifier is needed for the token exchange) but
    // state-binding is the standard CSRF anchor and is cheap to add.
    if state.is_empty() {
        anyhow::bail!(
            "Missing state in pasted authorization code. Make sure you \
             pasted the FULL string from the Anthropic callback page — it \
             should look like 'CODE#STATE' (the part after '#' is required \
             for CSRF protection)."
        );
    }
    if state != verifier {
        anyhow::bail!(
            "OAuth state mismatch: the callback's state did not match the \
             value we sent. This usually means the code was pasted from a \
             different login attempt. Restart the login flow."
        );
    }

    // 4. Exchange code for tokens — must include User-Agent matching Claude CLI.
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(TOKEN_ENDPOINT)
        .header("Content-Type", "application/json")
        .header("User-Agent", "claude-cli/1.0 (external, cli)")
        .json(&serde_json::json!({
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "state": state,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        }))
        .send()
        .context("sending token exchange request")?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().context("parsing token response")?;

    if !status.is_success() {
        let error_msg = body
            .get("error_description")
            .or_else(|| body.get("error"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        anyhow::bail!("Token exchange failed ({}): {}", status, error_msg);
    }

    let access_token = body
        .get("access_token")
        .and_then(|v| v.as_str())
        .context("missing access_token in response")?
        .to_string();

    let refresh_token = body
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let expires_in = body
        .get("expires_in")
        .and_then(|v| v.as_u64())
        .unwrap_or(3600);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let expires_at = now + expires_in;

    let creds = OAuthCredentials {
        access_token,
        refresh_token,
        expires_at,
    };

    // 5. Save to file.
    save_credentials(fennec_home, &creds)?;
    println!("Authentication successful! Credentials saved.");

    Ok(creds)
}

/// Load an OAuth access token from disk, refreshing if expired.
///
/// Returns `Ok(Some(token))` if valid credentials exist, `Ok(None)` if no
/// credentials are stored, or an error if refresh fails.
pub fn load_oauth_token(fennec_home: &Path) -> Result<Option<String>> {
    let path = fennec_home.join(TOKEN_FILE);
    if !path.exists() {
        return Ok(None);
    }

    let data = std::fs::read_to_string(&path).context("reading oauth token file")?;
    let creds: OAuthCredentials =
        serde_json::from_str(&data).context("parsing oauth credentials")?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // If token has more than 60 seconds remaining, use it as-is.
    if creds.expires_at > now + 60 {
        return Ok(Some(creds.access_token));
    }

    // Try to refresh.
    if creds.refresh_token.is_empty() {
        // No refresh token — credentials are expired and unrecoverable.
        tracing::warn!("OAuth token expired and no refresh token available");
        return Ok(None);
    }

    match refresh_oauth_token(fennec_home, &creds.refresh_token) {
        Ok(new_creds) => Ok(Some(new_creds.access_token)),
        Err(e) => {
            tracing::warn!("Failed to refresh OAuth token: {e}");
            Ok(None)
        }
    }
}

/// Refresh an OAuth token using the refresh_token grant.
///
/// Wraps the read-check-refresh-write cycle in an advisory file lock
/// (`<fennec_home>/.anthropic_oauth.lock`) so two Fennec processes
/// (e.g. CLI + gateway daemon) sharing the same `fennec_home` don't
/// race on refresh. Anthropic rotates the refresh token on each use,
/// so without the lock the second-to-finish process saves a stale
/// refresh_token and the next refresh from that process fails.
pub fn refresh_oauth_token(
    fennec_home: &Path,
    refresh_token: &str,
) -> Result<OAuthCredentials> {
    use fs2::FileExt;

    std::fs::create_dir_all(fennec_home)
        .context("creating fennec home for oauth lock")?;
    let lock_path = fennec_home.join(".anthropic_oauth.lock");
    let lock_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening oauth lock at {}", lock_path.display()))?;

    // Block until we can acquire the exclusive lock. Concurrent processes
    // wait here; we hold the lock for the duration of the refresh round
    // trip and the credentials write, then release on file drop at the
    // end of this function.
    lock_file
        .lock_exclusive()
        .context("acquiring exclusive oauth refresh lock")?;

    // Re-read the credentials after acquiring the lock — another process
    // may have refreshed while we were waiting, in which case OUR
    // refresh_token is now stale and we should hand back the freshly-
    // refreshed credentials instead of trying to redeem a stale token.
    let path = fennec_home.join(TOKEN_FILE);
    if path.exists() {
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(creds) = serde_json::from_str::<OAuthCredentials>(&data) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                if creds.expires_at > now + 60 && creds.refresh_token != refresh_token {
                    // Another process refreshed; use their result.
                    tracing::debug!(
                        "OAuth credentials were refreshed by another process \
                         while we waited for the lock; using their result"
                    );
                    return Ok(creds);
                }
            }
        }
    }

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(TOKEN_ENDPOINT)
        .json(&serde_json::json!({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        }))
        .send()
        .context("sending refresh token request")?;

    let status = resp.status();
    let body: serde_json::Value = resp.json().context("parsing refresh response")?;

    if !status.is_success() {
        let error_msg = body
            .get("error_description")
            .or_else(|| body.get("error"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown error");
        anyhow::bail!("Token refresh failed ({}): {}", status, error_msg);
    }

    let access_token = body
        .get("access_token")
        .and_then(|v| v.as_str())
        .context("missing access_token in refresh response")?
        .to_string();

    let new_refresh = body
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or(refresh_token)
        .to_string();

    let expires_in = body
        .get("expires_in")
        .and_then(|v| v.as_u64())
        .unwrap_or(3600);

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let expires_at = now + expires_in;

    let creds = OAuthCredentials {
        access_token,
        refresh_token: new_refresh,
        expires_at,
    };

    save_credentials(fennec_home, &creds)?;
    Ok(creds)
}

/// Persist credentials to the token file.
///
/// Uses `security::fs::write_secure` so the file is created with 0600 from
/// the start — the old write-then-chmod sequence had a race where the file
/// existed with umask-default (typically 0644) perms for a brief window.
fn save_credentials(fennec_home: &Path, creds: &OAuthCredentials) -> Result<()> {
    std::fs::create_dir_all(fennec_home)
        .context("creating fennec home directory")?;
    let path = fennec_home.join(TOKEN_FILE);
    let json = serde_json::to_string_pretty(creds).context("serializing credentials")?;
    crate::security::fs::write_secure(&path, json.as_bytes())
        .context("writing credentials file")?;
    Ok(())
}

/// Minimal percent-encoding for URL query parameters.
fn urlencoded(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", b));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64url_encode() {
        // Test with known values.
        let input = b"hello";
        let encoded = base64url_encode(input);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));
    }

    #[test]
    fn test_pkce_challenge_is_deterministic() {
        let verifier = "test-verifier-value";
        let c1 = compute_challenge(verifier);
        let c2 = compute_challenge(verifier);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_verifier_length() {
        let v = generate_verifier();
        // 32 bytes base64url-encoded should be 43 characters.
        assert!(v.len() >= 40);
    }

    #[test]
    fn test_urlencoded() {
        assert_eq!(urlencoded("hello world"), "hello%20world");
        assert_eq!(urlencoded("a+b"), "a%2Bb");
    }

    #[test]
    fn test_save_and_load_credentials() {
        let dir = tempfile::tempdir().unwrap();
        let creds = OAuthCredentials {
            access_token: "test-access".to_string(),
            refresh_token: "test-refresh".to_string(),
            expires_at: u64::MAX,
        };
        save_credentials(dir.path(), &creds).unwrap();

        let loaded = load_oauth_token(dir.path()).unwrap();
        assert_eq!(loaded, Some("test-access".to_string()));
    }

    #[test]
    fn test_load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load_oauth_token(dir.path()).unwrap();
        assert_eq!(loaded, None);
    }
}
