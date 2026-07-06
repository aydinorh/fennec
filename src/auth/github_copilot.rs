//! GitHub Copilot authentication for the `copilot` provider.
//!
//! Copilot's chat endpoint (`api.githubcopilot.com`) is OpenAI-compatible but
//! needs a short-lived **Copilot API token**, obtained by exchanging a GitHub
//! OAuth token at `api.github.com/copilot_internal/v2/token`. This module
//! resolves the underlying GitHub token (env vars → `gh auth token` → a token
//! saved by the device-code login), runs the device-code OAuth flow, performs
//! the exchange (cached + refreshed), and builds the Copilot request headers.
//!
//! GitHub tokens must be OAuth (`gho_`) or app (`ghu_`) tokens — classic PATs
//! (`ghp_`) are rejected by the Copilot API.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::schema::FennecConfig;

/// Public OAuth client id used by the Copilot CLI / opencode for the device
/// flow (not confidential — device flow doesn't use a client secret).
const OAUTH_CLIENT_ID: &str = "Ov23li8tweQw6odWQebz";
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const TOKEN_EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const EDITOR_VERSION: &str = "vscode/1.104.1";
const EXCHANGE_USER_AGENT: &str = "GitHubCopilotChat/0.26.7";
const TOKEN_FILE: &str = ".github_copilot.json";

const ENV_TOKEN_VARS: &[&str] = &["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"];

/// Stored GitHub OAuth token (from the device-code login).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredToken {
    github_token: String,
}

fn token_path() -> PathBuf {
    FennecConfig::resolve_home(None).join(TOKEN_FILE)
}

fn load_stored_token() -> Option<String> {
    let data = std::fs::read_to_string(token_path()).ok()?;
    let stored: StoredToken = serde_json::from_str(&data).ok()?;
    Some(stored.github_token).filter(|t| !t.is_empty())
}

fn save_stored_token(token: &str) -> Result<()> {
    let path = token_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).context("creating fennec home for Copilot token")?;
    }
    let json = serde_json::to_string_pretty(&StoredToken {
        github_token: token.to_string(),
    })?;
    std::fs::write(&path, json)
        .with_context(|| format!("writing Copilot token to {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Try `gh auth token` from the GitHub CLI. Strips `GITHUB_TOKEN`/`GH_TOKEN`
/// from the child env so `gh` returns its stored (device-flow) token rather
/// than echoing an env var back.
fn gh_cli_token() -> Option<String> {
    let output = std::process::Command::new("gh")
        .args(["auth", "token"])
        .env_remove("GITHUB_TOKEN")
        .env_remove("GH_TOKEN")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// Resolve a GitHub token for Copilot: env vars → `gh auth token` → the
/// device-code login's stored token. Blocking (`gh` subprocess + file read);
/// call from `spawn_blocking` on an async runtime.
pub fn resolve_github_token() -> Option<String> {
    for var in ENV_TOKEN_VARS {
        if let Ok(token) = std::env::var(var) {
            if !token.is_empty() {
                return Some(token);
            }
        }
    }
    if let Some(token) = gh_cli_token() {
        return Some(token);
    }
    load_stored_token()
}

/// Headers required by the Copilot chat API (mirrors the Copilot CLI / VS Code
/// client). `Authorization` is added separately by the caller.
pub fn copilot_request_headers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Editor-Version", EDITOR_VERSION),
        ("Copilot-Integration-Id", "vscode-chat"),
        ("Openai-Intent", "conversation-edits"),
        ("x-initiator", "agent"),
        ("User-Agent", EXCHANGE_USER_AGENT),
    ]
}

/// Exchange a GitHub token for a short-lived Copilot API token. Returns
/// `(copilot_token, expires_at_unix_secs)`.
pub async fn exchange_copilot_token(
    client: &reqwest::Client,
    github_token: &str,
) -> Result<(String, u64)> {
    let resp = client
        .get(TOKEN_EXCHANGE_URL)
        .header("Authorization", format!("token {github_token}"))
        .header("User-Agent", EXCHANGE_USER_AGENT)
        .header("Accept", "application/json")
        .header("Editor-Version", EDITOR_VERSION)
        .send()
        .await
        .context("exchanging GitHub token for a Copilot token")?;
    let status = resp.status();
    let body: Value = resp
        .json()
        .await
        .context("parsing Copilot token-exchange response")?;
    if !status.is_success() {
        let msg = body
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("token exchange rejected");
        anyhow::bail!(
            "Copilot token exchange failed ({status}): {msg} — \
             the GitHub token must be a Copilot-enabled OAuth token (gho_/ghu_), not a classic PAT"
        );
    }
    let token = body
        .get("token")
        .and_then(|t| t.as_str())
        .filter(|t| !t.is_empty())
        .context("Copilot token exchange returned an empty token")?
        .to_string();
    let expires_at = body
        .get("expires_at")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| now_secs() + 1800);
    Ok((token, expires_at))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Run the GitHub device-code OAuth flow and persist the resulting token.
/// Blocking (prints a code, polls until the user authorizes).
pub fn run_device_login() -> Result<String> {
    let client = reqwest::blocking::Client::new();

    let device: Value = client
        .post(DEVICE_CODE_URL)
        .header("Accept", "application/json")
        .header("User-Agent", EXCHANGE_USER_AGENT)
        .form(&[("client_id", OAUTH_CLIENT_ID), ("scope", "read:user")])
        .send()
        .context("requesting GitHub device code")?
        .json()
        .context("parsing GitHub device-code response")?;

    let device_code = device["device_code"]
        .as_str()
        .context("GitHub did not return a device_code")?
        .to_string();
    let user_code = device["user_code"].as_str().unwrap_or("");
    let verification_uri = device["verification_uri"]
        .as_str()
        .unwrap_or("https://github.com/login/device");
    let mut interval = device["interval"].as_u64().unwrap_or(5).max(1);

    println!();
    println!("  Open this URL in your browser: {verification_uri}");
    println!("  Enter this code: {user_code}");
    println!();
    print!("  Waiting for authorization");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        if Instant::now() >= deadline {
            anyhow::bail!("device-code login timed out");
        }
        std::thread::sleep(Duration::from_secs(interval + 1));

        let result: Value = match client
            .post(ACCESS_TOKEN_URL)
            .header("Accept", "application/json")
            .header("User-Agent", EXCHANGE_USER_AGENT)
            .form(&[
                ("client_id", OAUTH_CLIENT_ID),
                ("device_code", device_code.as_str()),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ])
            .send()
            .and_then(|r| r.json())
        {
            Ok(v) => v,
            Err(_) => {
                print!(".");
                std::io::stdout().flush().ok();
                continue;
            }
        };

        if let Some(token) = result["access_token"].as_str().filter(|t| !t.is_empty()) {
            println!(" done");
            save_stored_token(token)?;
            return Ok(token.to_string());
        }
        match result["error"].as_str() {
            Some("authorization_pending") => {
                print!(".");
                std::io::stdout().flush().ok();
            }
            Some("slow_down") => {
                interval += 5;
            }
            Some(other) => anyhow::bail!("GitHub device login failed: {other}"),
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_include_copilot_integration_id() {
        let headers = copilot_request_headers();
        assert!(headers.iter().any(|(k, v)| *k == "Copilot-Integration-Id" && *v == "vscode-chat"));
        assert!(headers.iter().any(|(k, _)| *k == "Editor-Version"));
        assert!(headers.iter().any(|(k, v)| *k == "x-initiator" && *v == "agent"));
    }

    #[test]
    fn stored_token_round_trips() {
        // Exercise (de)serialization without touching the real home dir.
        let json = serde_json::to_string(&StoredToken {
            github_token: "gho_abc".to_string(),
        })
        .unwrap();
        let back: StoredToken = serde_json::from_str(&json).unwrap();
        assert_eq!(back.github_token, "gho_abc");
    }
}
