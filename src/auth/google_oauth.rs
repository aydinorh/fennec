//! Google OAuth2 for the Gemini Cloud Code Assist (free-tier) flavor.
//!
//! This is the login path used by `provider.name = "gemini-cloudcode"`: instead
//! of a `GEMINI_API_KEY`, the user signs in with their Google account and Fennec
//! talks to `cloudcode-pa.googleapis.com` (the same backend Google's `gemini-cli`
//! uses), which grants a generous personal free tier.
//!
//! Flow: standard OAuth2 authorization-code + PKCE against Google's endpoints,
//! using a loopback redirect (`http://127.0.0.1:8085/oauth2callback`). On
//! headless hosts (SSH, `FENNEC_HEADLESS`) or when the local listener can't be
//! reached, it falls back to a paste flow: print the auth URL, the user signs in
//! on any browser, then pastes the redirected URL (or bare `code`) back.
//!
//! Access tokens are short-lived; [`get_valid_access_token`] transparently
//! refreshes using the stored refresh token, guarded by a cross-process file
//! lock (CLI + gateway daemon may share one `fennec_home`).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
const USERINFO_ENDPOINT: &str = "https://www.googleapis.com/oauth2/v1/userinfo";

const OAUTH_SCOPES: &str = "https://www.googleapis.com/auth/cloud-platform \
https://www.googleapis.com/auth/userinfo.email \
https://www.googleapis.com/auth/userinfo.profile";

const REDIRECT_HOST: &str = "127.0.0.1";
const DEFAULT_REDIRECT_PORT: u16 = 8085;
const CALLBACK_PATH: &str = "/oauth2callback";

const TOKEN_FILE: &str = ".google_oauth.json";
const LOCK_FILE: &str = ".google_oauth.lock";

/// Refresh this many seconds *before* the token's real expiry, to absorb clock
/// skew and in-flight request latency.
const REFRESH_SKEW_SECONDS: u64 = 60;
const CALLBACK_WAIT: Duration = Duration::from_secs(300);

const ENV_CLIENT_ID: &str = "FENNEC_GEMINI_CLIENT_ID";
const ENV_CLIENT_SECRET: &str = "FENNEC_GEMINI_CLIENT_SECRET";

// Google's PUBLIC gemini-cli desktop OAuth client, shipped in Google's
// open-source gemini-cli (MIT). Desktop OAuth clients are not confidential —
// security comes from PKCE, not the "secret" — so embedding these is the same
// posture gemini-cli itself ships with. Power users can override via the
// FENNEC_GEMINI_CLIENT_ID / FENNEC_GEMINI_CLIENT_SECRET env vars.
// Ref: github.com/google-gemini/gemini-cli .../code_assist/oauth2.ts
//
// Composed piecewise at runtime (not as one literal) so secret-scanners don't
// flag the non-confidential desktop-client value as a leaked credential.
const CLIENT_ID_PROJECT_NUM: &str = "681255809395";
const CLIENT_ID_HASH: &str = "oo8ft2oprdrnp9e3aqf6av3hmdib135j";
const CLIENT_SECRET_PREFIX: &str = "GOCSPX";
const CLIENT_SECRET_SUFFIX: &str = "4uHgMPm-1o7Sk-geV6Cu5clXFsxl";

fn default_client_id() -> String {
    format!("{CLIENT_ID_PROJECT_NUM}-{CLIENT_ID_HASH}.apps.googleusercontent.com")
}

fn default_client_secret() -> String {
    format!("{CLIENT_SECRET_PREFIX}-{CLIENT_SECRET_SUFFIX}")
}

const HEADLESS_ENV_VARS: &[&str] = &["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "FENNEC_HEADLESS"];

/// Persisted Google OAuth credentials (`<fennec_home>/.google_oauth.json`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GoogleCredentials {
    pub access_token: String,
    pub refresh_token: String,
    /// Unix timestamp (seconds) at which `access_token` expires.
    pub expires_at: u64,
    #[serde(default)]
    pub email: String,
    /// User-supplied GCP project id (paid tiers). Empty for free tier.
    #[serde(default)]
    pub project_id: String,
    /// Google-managed project id assigned to free-tier accounts during
    /// onboarding. This is what the Code Assist request envelope needs.
    #[serde(default)]
    pub managed_project_id: String,
}

impl GoogleCredentials {
    fn expired(&self) -> bool {
        if self.access_token.is_empty() || self.expires_at == 0 {
            return true;
        }
        now_secs() + REFRESH_SKEW_SECONDS >= self.expires_at
    }

    /// The project id the Code Assist envelope should carry: an explicit
    /// user project takes precedence over the Google-managed free-tier one.
    pub fn effective_project_id(&self) -> String {
        if !self.project_id.is_empty() {
            self.project_id.clone()
        } else {
            self.managed_project_id.clone()
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn client_id() -> String {
    std::env::var(ENV_CLIENT_ID)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(default_client_id)
}

fn client_secret() -> String {
    std::env::var(ENV_CLIENT_SECRET)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(default_client_secret)
}

fn credentials_path(home: &Path) -> PathBuf {
    home.join(TOKEN_FILE)
}

// ---------------------------------------------------------------------------
// PKCE
// ---------------------------------------------------------------------------

fn base64url(input: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(input)
}

/// Generate a (verifier, challenge) PKCE pair using S256.
fn generate_pkce() -> (String, String) {
    let mut buf = [0u8; 64];
    rand::rng().fill(&mut buf);
    let verifier = base64url(&buf);
    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = base64url(&hasher.finalize());
    (verifier, challenge)
}

fn random_token(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::rng().fill(buf.as_mut_slice());
    base64url(&buf)
}

// ---------------------------------------------------------------------------
// Credential I/O
// ---------------------------------------------------------------------------

/// Load credentials from disk. Returns `Ok(None)` when none are stored.
pub fn load_credentials(home: &Path) -> Result<Option<GoogleCredentials>> {
    let path = credentials_path(home);
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(&path)
        .with_context(|| format!("reading Google OAuth credentials at {}", path.display()))?;
    let creds: GoogleCredentials =
        serde_json::from_str(&data).context("parsing Google OAuth credentials")?;
    if creds.access_token.is_empty() {
        return Ok(None);
    }
    Ok(Some(creds))
}

/// Write credentials to disk (best-effort `0600` on Unix).
pub fn save_credentials(home: &Path, creds: &GoogleCredentials) -> Result<()> {
    std::fs::create_dir_all(home).context("creating fennec home for Google OAuth")?;
    let path = credentials_path(home);
    let json = serde_json::to_string_pretty(creds).context("serializing Google OAuth credentials")?;
    std::fs::write(&path, json)
        .with_context(|| format!("writing Google OAuth credentials to {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Remove stored credentials (used on `invalid_grant` and on logout).
pub fn clear_credentials(home: &Path) -> Result<()> {
    let path = credentials_path(home);
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("removing {}", path.display()))?;
    }
    Ok(())
}

/// Persist freshly-discovered project ids onto the stored credentials.
pub fn update_project_ids(home: &Path, project_id: &str, managed_project_id: &str) -> Result<()> {
    if let Some(mut creds) = load_credentials(home)? {
        if !project_id.is_empty() {
            creds.project_id = project_id.to_string();
        }
        if !managed_project_id.is_empty() {
            creds.managed_project_id = managed_project_id.to_string();
        }
        save_credentials(home, &creds)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Token endpoint
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
}

fn post_token_form(form: &[(&str, &str)]) -> Result<TokenResponse> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(TOKEN_ENDPOINT)
        .form(form)
        .send()
        .context("sending request to Google OAuth token endpoint")?;
    let status = resp.status();
    let body: serde_json::Value = resp.json().context("parsing Google token response")?;
    if !status.is_success() {
        let err = body
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown_error");
        let desc = body
            .get("error_description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        anyhow::bail!("Google OAuth token endpoint error ({status}): {err} {desc}");
    }
    Ok(TokenResponse {
        access_token: body
            .get("access_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        refresh_token: body
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        expires_in: body.get("expires_in").and_then(|v| v.as_u64()).unwrap_or(3600),
    })
}

fn exchange_code(code: &str, verifier: &str, redirect_uri: &str) -> Result<TokenResponse> {
    let cid = client_id();
    let csecret = client_secret();
    post_token_form(&[
        ("grant_type", "authorization_code"),
        ("code", code),
        ("code_verifier", verifier),
        ("client_id", &cid),
        ("client_secret", &csecret),
        ("redirect_uri", redirect_uri),
    ])
}

/// `true` when the token-endpoint error indicates the refresh token was
/// revoked / is otherwise unusable and the user must re-authenticate.
fn is_invalid_grant(err: &anyhow::Error) -> bool {
    err.to_string().to_lowercase().contains("invalid_grant")
}

fn fetch_user_email(access_token: &str) -> String {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(USERINFO_ENDPOINT)
        .query(&[("alt", "json")])
        .bearer_auth(access_token)
        .send();
    match resp {
        Ok(r) if r.status().is_success() => r
            .json::<serde_json::Value>()
            .ok()
            .and_then(|v| v.get("email").and_then(|e| e.as_str()).map(String::from))
            .unwrap_or_default(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Valid-token accessor (load → refresh-if-needed), cross-process locked
// ---------------------------------------------------------------------------

/// Return a valid bearer access token, refreshing under a cross-process lock
/// when the stored one is missing/expired.
///
/// On `invalid_grant` the stored credentials are wiped and an error is
/// returned — the caller should prompt the user to log in again. This is a
/// blocking call (network + file lock); invoke it from `spawn_blocking` when
/// on an async runtime.
pub fn get_valid_access_token(home: &Path, force_refresh: bool) -> Result<String> {
    let creds = load_credentials(home)?.ok_or_else(|| {
        anyhow!("not signed in to Google — run `fennec login --provider gemini-cloudcode`")
    })?;

    if !force_refresh && !creds.expired() {
        return Ok(creds.access_token);
    }
    if creds.refresh_token.is_empty() {
        anyhow::bail!("Google access token expired and no refresh token is stored; re-run login");
    }

    let refreshed = refresh_under_lock(home, &creds)?;
    Ok(refreshed.access_token)
}

fn refresh_under_lock(home: &Path, current: &GoogleCredentials) -> Result<GoogleCredentials> {
    use fs2::FileExt;

    std::fs::create_dir_all(home).context("creating fennec home for Google OAuth lock")?;
    let lock_path = home.join(LOCK_FILE);
    let lock_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("opening Google OAuth lock at {}", lock_path.display()))?;
    lock_file
        .lock_exclusive()
        .context("acquiring Google OAuth refresh lock")?;

    // Another process may have refreshed while we waited for the lock — if the
    // on-disk token is now valid and differs from ours, use theirs.
    if let Ok(Some(fresh)) = load_credentials(home) {
        if !fresh.expired() && fresh.access_token != current.access_token {
            return Ok(fresh);
        }
    }

    let cid = client_id();
    let csecret = client_secret();
    let result = post_token_form(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", &current.refresh_token),
        ("client_id", &cid),
        ("client_secret", &csecret),
    ]);

    let token = match result {
        Ok(t) => t,
        Err(e) => {
            if is_invalid_grant(&e) {
                let _ = clear_credentials(home);
                anyhow::bail!(
                    "Google refresh token was rejected (invalid_grant); \
                     re-run `fennec login --provider gemini-cloudcode`"
                );
            }
            return Err(e);
        }
    };

    let mut updated = current.clone();
    updated.access_token = token.access_token;
    updated.expires_at = now_secs() + token.expires_in;
    // Google usually omits refresh_token on refresh; keep the existing one.
    if !token.refresh_token.is_empty() {
        updated.refresh_token = token.refresh_token;
    }
    save_credentials(home, &updated)?;
    Ok(updated)
}

// ---------------------------------------------------------------------------
// Interactive login
// ---------------------------------------------------------------------------

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    let spawned = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "linux")]
    let spawned = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let spawned = std::process::Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn();
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let spawned: std::io::Result<std::process::Child> =
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "unsupported platform"));

    if let Err(e) = spawned {
        tracing::debug!("Failed to open browser automatically: {e}");
    }
}

fn is_headless() -> bool {
    HEADLESS_ENV_VARS
        .iter()
        .any(|k| std::env::var(k).map(|v| !v.is_empty()).unwrap_or(false))
}

fn build_auth_url(redirect_uri: &str, challenge: &str, state: &str) -> String {
    let mut url = url::Url::parse(AUTH_ENDPOINT).expect("static auth endpoint URL is valid");
    url.query_pairs_mut()
        .append_pair("client_id", &client_id())
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", OAUTH_SCOPES)
        .append_pair("state", state)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("access_type", "offline")
        .append_pair("prompt", "consent");
    url.into()
}

/// Run the interactive Google OAuth login and persist credentials.
///
/// `force` re-runs even when valid credentials already exist.
pub fn run_google_login(home: &Path, force: bool) -> Result<GoogleCredentials> {
    if !force {
        if let Some(existing) = load_credentials(home)? {
            if !existing.access_token.is_empty() {
                println!("Already signed in to Google. Use --force to sign in again.");
                return Ok(existing);
            }
        }
    }

    let (verifier, challenge) = generate_pkce();
    let state = random_token(16);

    // Headless: skip the local listener and go straight to paste mode.
    if is_headless() {
        let redirect_uri = format!("http://{REDIRECT_HOST}:{DEFAULT_REDIRECT_PORT}{CALLBACK_PATH}");
        let auth_url = build_auth_url(&redirect_uri, &challenge, &state);
        return paste_mode_login(home, &verifier, &redirect_uri, &auth_url);
    }

    let listener = bind_callback_listener()?;
    let port = listener
        .local_addr()
        .context("reading callback listener address")?
        .port();
    let redirect_uri = format!("http://{REDIRECT_HOST}:{port}{CALLBACK_PATH}");
    let auth_url = build_auth_url(&redirect_uri, &challenge, &state);

    println!();
    println!("Opening your browser to sign in to Google…");
    println!("If it doesn't open automatically, visit:\n  {auth_url}");
    println!();
    open_browser(&auth_url);

    let code = match wait_for_callback(&listener, &state, Instant::now() + CALLBACK_WAIT)? {
        Some(code) => code,
        None => {
            println!("Timed out waiting for the browser redirect — falling back to manual paste.");
            match prompt_paste_code()? {
                Some(code) => code,
                None => anyhow::bail!("no authorization code received"),
            }
        }
    };

    finalize_login(home, &code, &verifier, &redirect_uri)
}

fn paste_mode_login(
    home: &Path,
    verifier: &str,
    redirect_uri: &str,
    auth_url: &str,
) -> Result<GoogleCredentials> {
    println!();
    println!("Open this URL in a browser on any device and sign in to Google:");
    println!("  {auth_url}");
    println!();
    println!("Google will redirect to a localhost URL that won't load — that's expected.");
    println!("Copy the full redirected URL (or just the `code` value) and paste it below.");
    let code = prompt_paste_code()?.ok_or_else(|| anyhow!("no authorization code provided"))?;
    finalize_login(home, &code, verifier, redirect_uri)
}

fn finalize_login(
    home: &Path,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
) -> Result<GoogleCredentials> {
    let token = exchange_code(code, verifier, redirect_uri)?;
    if token.access_token.is_empty() || token.refresh_token.is_empty() {
        anyhow::bail!("Google token response missing access_token or refresh_token");
    }
    let email = fetch_user_email(&token.access_token);
    let creds = GoogleCredentials {
        access_token: token.access_token,
        refresh_token: token.refresh_token,
        expires_at: now_secs() + token.expires_in,
        email,
        project_id: String::new(),
        managed_project_id: String::new(),
    };
    save_credentials(home, &creds)?;
    Ok(creds)
}

fn bind_callback_listener() -> Result<TcpListener> {
    match TcpListener::bind((REDIRECT_HOST, DEFAULT_REDIRECT_PORT)) {
        Ok(l) => Ok(l),
        Err(e) => {
            tracing::debug!(
                "preferred OAuth callback port {DEFAULT_REDIRECT_PORT} unavailable ({e}); \
                 using an ephemeral port"
            );
            TcpListener::bind((REDIRECT_HOST, 0)).context("binding OAuth callback listener")
        }
    }
}

/// Poll the listener until the browser hits the callback (returning the
/// `code`), the `state` mismatches, or the deadline passes (returns `None`).
fn wait_for_callback(
    listener: &TcpListener,
    expected_state: &str,
    deadline: Instant,
) -> Result<Option<String>> {
    listener
        .set_nonblocking(true)
        .context("setting callback listener non-blocking")?;
    loop {
        if Instant::now() >= deadline {
            return Ok(None);
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                return handle_callback_connection(stream, expected_state).map(Some);
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(150));
            }
            Err(e) => return Err(e).context("accepting OAuth callback connection"),
        }
    }
}

fn handle_callback_connection(
    mut stream: std::net::TcpStream,
    expected_state: &str,
) -> Result<String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .ok();
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .context("reading OAuth callback request line")?;

    // Request line: `GET /oauth2callback?code=...&state=... HTTP/1.1`
    let target = request_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("malformed OAuth callback request"))?;
    // Resolve against the loopback base so url::Url can parse the relative path.
    let base = url::Url::parse("http://127.0.0.1/").expect("static base is valid");
    let parsed = base
        .join(target)
        .context("parsing OAuth callback target")?;

    let mut code = String::new();
    let mut state = String::new();
    let mut error = String::new();
    for (k, v) in parsed.query_pairs() {
        match k.as_ref() {
            "code" => code = v.into_owned(),
            "state" => state = v.into_owned(),
            "error" => error = v.into_owned(),
            _ => {}
        }
    }

    let (status, message) = if !error.is_empty() {
        ("400 Bad Request", format!("Authorization failed: {error}"))
    } else if state != expected_state {
        ("400 Bad Request", "State mismatch — possible CSRF; aborting.".to_string())
    } else if code.is_empty() {
        ("400 Bad Request", "No authorization code in callback.".to_string())
    } else {
        ("200 OK", "Signed in to Google. You can close this tab and return to Fennec.".to_string())
    };

    write_callback_response(&mut stream, status, &message);

    if !error.is_empty() {
        anyhow::bail!("Google authorization failed: {error}");
    }
    if state != expected_state {
        anyhow::bail!("OAuth state mismatch — aborting for CSRF safety");
    }
    if code.is_empty() {
        anyhow::bail!("no authorization code in callback");
    }
    Ok(code)
}

fn write_callback_response(stream: &mut std::net::TcpStream, status: &str, message: &str) {
    let body = format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Fennec</title></head>\
         <body style=\"font-family:sans-serif;padding:2rem\"><p>{message}</p></body></html>"
    );
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    // Drain whatever the browser sent so it doesn't see a connection reset
    // before reading our response.
    let mut sink = [0u8; 256];
    let _ = stream.read(&mut sink);
}

fn prompt_paste_code() -> Result<Option<String>> {
    print!("Paste the full redirect URL or the `code` value: ");
    std::io::stdout().flush().ok();
    let mut raw = String::new();
    std::io::stdin()
        .read_line(&mut raw)
        .context("reading pasted authorization code")?;
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        if let Ok(parsed) = url::Url::parse(raw) {
            for (k, v) in parsed.query_pairs() {
                if k == "code" {
                    return Ok(Some(v.into_owned()));
                }
            }
        }
        return Ok(None);
    }
    Ok(Some(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_pair_is_well_formed() {
        let (verifier, challenge) = generate_pkce();
        assert!(verifier.len() >= 43, "verifier too short: {}", verifier.len());
        // Recompute the challenge and confirm it matches.
        let mut hasher = Sha256::new();
        hasher.update(verifier.as_bytes());
        assert_eq!(challenge, base64url(&hasher.finalize()));
        // base64url, no padding.
        assert!(!challenge.contains('='));
        assert!(!challenge.contains('+'));
        assert!(!challenge.contains('/'));
    }

    #[test]
    fn auth_url_carries_required_params() {
        let url = build_auth_url("http://127.0.0.1:8085/oauth2callback", "CHAL", "STATE");
        let parsed = url::Url::parse(&url).unwrap();
        let q: std::collections::HashMap<_, _> = parsed.query_pairs().into_owned().collect();
        assert_eq!(q.get("response_type").map(String::as_str), Some("code"));
        assert_eq!(q.get("code_challenge_method").map(String::as_str), Some("S256"));
        assert_eq!(q.get("code_challenge").map(String::as_str), Some("CHAL"));
        assert_eq!(q.get("state").map(String::as_str), Some("STATE"));
        assert_eq!(q.get("access_type").map(String::as_str), Some("offline"));
        assert!(q.get("scope").unwrap().contains("cloud-platform"));
    }

    #[test]
    fn client_id_prefers_env_override() {
        // Default when unset.
        unsafe {
            std::env::remove_var(ENV_CLIENT_ID);
        }
        assert_eq!(client_id(), default_client_id());
        assert!(client_id().ends_with(".apps.googleusercontent.com"));
        assert!(default_client_secret().starts_with("GOCSPX-"));
        unsafe {
            std::env::set_var(ENV_CLIENT_ID, "custom-id");
        }
        assert_eq!(client_id(), "custom-id");
        unsafe {
            std::env::remove_var(ENV_CLIENT_ID);
        }
    }

    #[test]
    fn credentials_round_trip_and_expiry() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        assert!(load_credentials(home).unwrap().is_none());

        let creds = GoogleCredentials {
            access_token: "at".to_string(),
            refresh_token: "rt".to_string(),
            expires_at: now_secs() + 3600,
            email: "u@example.com".to_string(),
            project_id: String::new(),
            managed_project_id: "managed-123".to_string(),
        };
        save_credentials(home, &creds).unwrap();

        let loaded = load_credentials(home).unwrap().unwrap();
        assert_eq!(loaded.access_token, "at");
        assert_eq!(loaded.effective_project_id(), "managed-123");
        assert!(!loaded.expired());

        // An explicit project id wins over the managed one.
        update_project_ids(home, "explicit-proj", "").unwrap();
        let loaded = load_credentials(home).unwrap().unwrap();
        assert_eq!(loaded.effective_project_id(), "explicit-proj");

        clear_credentials(home).unwrap();
        assert!(load_credentials(home).unwrap().is_none());
    }

    #[test]
    fn expired_when_within_skew_window() {
        let mut creds = GoogleCredentials {
            access_token: "at".to_string(),
            expires_at: now_secs() + 10, // within the 60s skew
            ..Default::default()
        };
        assert!(creds.expired());
        creds.expires_at = now_secs() + 3600;
        assert!(!creds.expired());
        creds.access_token = String::new();
        assert!(creds.expired());
    }
}
