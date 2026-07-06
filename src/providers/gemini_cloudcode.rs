//! Google Gemini via Cloud Code Assist (OAuth free-tier).
//!
//! This is the OAuth-authenticated sibling of [`super::gemini::GeminiProvider`].
//! Instead of a `GEMINI_API_KEY`, the user signs in with their Google account
//! (see [`crate::auth::google_oauth`]) and requests go to
//! `cloudcode-pa.googleapis.com/v1internal` — the same backend Google's
//! `gemini-cli` uses, which grants a generous personal free tier.
//!
//! The inner request/response bodies are *identical* to the native Gemini API,
//! so this module reuses the translation in [`super::gemini`] wholesale and
//! only adds three things: an OAuth bearer token (transparently refreshed), a
//! wrapping `{project, model, request:{…}}` envelope, and the matching
//! `{response:{…}}` unwrap on the way back.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use crate::auth::google_oauth;

use super::gemini::{
    build_gemini_request_body, dispatch_gemini_event, extract_error_message, finish_stream,
    parse_gemini_response,
};
use super::sse::SseBuffer;
use super::traits::{ChatRequest, ChatResponse, Provider, StreamEvent, UsageInfo};

const CODE_ASSIST_ENDPOINT: &str = "https://cloudcode-pa.googleapis.com";
// Match Google's gemini-cli client identity — cloudcode-pa may reject requests
// whose User-Agent / api-client headers it doesn't recognize.
const GEMINI_CLI_USER_AGENT: &str = "google-api-nodejs-client/9.15.1 (gzip)";
const X_GOOG_API_CLIENT: &str = "gl-node/24.0.0";

const FREE_TIER_ID: &str = "free-tier";

const DEFAULT_CLOUDCODE_MODEL: &str = "gemini-2.5-flash";
const DEFAULT_CONTEXT_WINDOW: usize = 1_048_576;

const PROJECT_ENV_VARS: &[&str] = &["FENNEC_GEMINI_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"];

// Onboarding can be a long-running operation; poll its completion.
const ONBOARD_POLL_ATTEMPTS: usize = 12;
const ONBOARD_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Metadata block gemini-cli sends on Code Assist calls. Unknown values may be
/// rejected, so mirror it exactly.
fn client_metadata() -> Value {
    json!({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    })
}

fn project_id_from_env() -> Option<String> {
    PROJECT_ENV_VARS
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
}

/// Peel the Code Assist `{response:{…}}` envelope, returning the inner Gemini
/// body. Falls back to the value itself when no envelope is present.
fn unwrap_response(body: &Value) -> &Value {
    match body.get("response") {
        Some(inner) if inner.is_object() => inner,
        _ => body,
    }
}

// ---------------------------------------------------------------------------
// Onboarding (login-time, blocking)
// ---------------------------------------------------------------------------

fn code_assist_post_blocking(url: &str, body: &Value, token: &str) -> Result<Value> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(url)
        .bearer_auth(token)
        .header("Content-Type", "application/json")
        .header("User-Agent", GEMINI_CLI_USER_AGENT)
        .header("X-Goog-Api-Client", X_GOOG_API_CLIENT)
        .header("x-activity-request-id", uuid::Uuid::new_v4().to_string())
        .json(body)
        .send()
        .with_context(|| format!("POST {url}"))?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "Code Assist API error ({status}): {}",
            extract_error_message(text.as_bytes())
        );
    }
    if text.trim().is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&text).context("parsing Code Assist response as JSON")
}

/// `POST /v1internal:loadCodeAssist` — discover the account's tier and any
/// already-assigned Google-managed project. Returns `(tier_id, project_id)`.
fn load_code_assist(token: &str, project_id: &str) -> Result<(String, String)> {
    let mut metadata = client_metadata();
    metadata["duetProject"] = json!(project_id);
    let mut body = json!({ "metadata": metadata });
    if !project_id.is_empty() {
        body["cloudaicompanionProject"] = json!(project_id);
    }
    let url = format!("{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist");
    let resp = code_assist_post_blocking(&url, &body, token)?;
    let tier = resp
        .get("currentTier")
        .and_then(|t| t.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let project = resp
        .get("cloudaicompanionProject")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok((tier, project))
}

/// `POST /v1internal:onboardUser` — provision the account on `tier_id`,
/// polling the long-running operation to completion. Returns the
/// Google-managed project id when one is assigned.
fn onboard_user(token: &str, tier_id: &str, project_id: &str) -> Result<String> {
    let mut body = json!({ "tierId": tier_id, "metadata": client_metadata() });
    if !project_id.is_empty() {
        body["cloudaicompanionProject"] = json!(project_id);
    }
    let url = format!("{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser");
    let mut resp = code_assist_post_blocking(&url, &body, token)?;

    if resp.get("done").and_then(|v| v.as_bool()) != Some(true) {
        if let Some(op_name) = resp.get("name").and_then(|v| v.as_str()).map(String::from) {
            for _ in 0..ONBOARD_POLL_ATTEMPTS {
                std::thread::sleep(ONBOARD_POLL_INTERVAL);
                let poll_url = format!("{CODE_ASSIST_ENDPOINT}/v1internal/{op_name}");
                if let Ok(polled) = code_assist_post_blocking(&poll_url, &json!({}), token) {
                    if polled.get("done").and_then(|v| v.as_bool()) == Some(true) {
                        resp = polled;
                        break;
                    }
                }
            }
        }
    }

    let managed = resp
        .get("response")
        .and_then(|r| r.get("cloudaicompanionProject"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(managed)
}

/// Resolve and persist the project id for the signed-in account.
///
/// Called at login time (after OAuth). An explicit project (env var) short-
/// circuits discovery; otherwise `loadCodeAssist` reports the tier, and a not-
/// yet-onboarded account is provisioned on the free tier. The resolved project
/// is stored on the credentials so the provider can build request envelopes.
/// Blocking — run from `spawn_blocking` on an async runtime.
pub fn ensure_project_context(home: &Path) -> Result<String> {
    let token = google_oauth::get_valid_access_token(home, false)?;

    if let Some(env_project) = project_id_from_env() {
        google_oauth::update_project_ids(home, &env_project, "")?;
        return Ok(env_project);
    }

    let (tier, mut managed) = load_code_assist(&token, "")?;
    if tier.is_empty() {
        // Not onboarded yet — provision on the free tier.
        let onboarded = onboard_user(&token, FREE_TIER_ID, "")?;
        if !onboarded.is_empty() {
            managed = onboarded;
        }
    }

    google_oauth::update_project_ids(home, "", &managed)?;
    Ok(managed)
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Gemini provider authenticated via Google OAuth + Cloud Code Assist.
pub struct GeminiCloudCodeProvider {
    home: PathBuf,
    client: reqwest::Client,
    model: String,
    project_id: String,
    ctx_window: usize,
}

impl GeminiCloudCodeProvider {
    /// Create a Cloud Code provider. The project id is resolved from (in
    /// order) the explicit override, the project env vars, then the stored
    /// credentials' effective project. Token resolution/refresh happens
    /// per-request against `home`.
    pub fn new(
        home: PathBuf,
        model: Option<String>,
        project_override: Option<String>,
        context_window: Option<usize>,
    ) -> Self {
        let project_id = project_override
            .filter(|s| !s.is_empty())
            .or_else(project_id_from_env)
            .or_else(|| {
                google_oauth::load_credentials(&home)
                    .ok()
                    .flatten()
                    .map(|c| c.effective_project_id())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_default();

        Self {
            home,
            client: reqwest::Client::new(),
            model: model.unwrap_or_else(|| DEFAULT_CLOUDCODE_MODEL.to_string()),
            project_id,
            ctx_window: context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
        }
    }

    /// Fetch a valid bearer token, refreshing if needed. Runs the blocking
    /// token machinery off the async executor.
    async fn access_token(&self, force_refresh: bool) -> Result<String> {
        let home = self.home.clone();
        tokio::task::spawn_blocking(move || {
            google_oauth::get_valid_access_token(&home, force_refresh)
        })
        .await
        .context("joining Google token refresh task")?
    }

    fn wrap_envelope(&self, inner: Value) -> Value {
        json!({
            "project": self.project_id,
            "model": self.model,
            "user_prompt_id": uuid::Uuid::new_v4().to_string(),
            "request": inner,
        })
    }
}

#[async_trait]
impl Provider for GeminiCloudCodeProvider {
    fn name(&self) -> &str {
        "gemini-cloudcode"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let envelope = self.wrap_envelope(build_gemini_request_body(&request));
        let url = format!("{CODE_ASSIST_ENDPOINT}/v1internal:generateContent");

        let mut force_refresh = false;
        loop {
            let token = self.access_token(force_refresh).await?;
            let response = self
                .client
                .post(&url)
                .bearer_auth(&token)
                .header("Content-Type", "application/json")
                .header("User-Agent", GEMINI_CLI_USER_AGENT)
                .header("X-Goog-Api-Client", X_GOOG_API_CLIENT)
                .json(&envelope)
                .send()
                .await
                .context("sending request to Code Assist API")?;

            let status = response.status();
            // A 401/403 mid-session usually means a stale token — force one
            // refresh and retry before surfacing the error.
            if (status.as_u16() == 401 || status.as_u16() == 403) && !force_refresh {
                force_refresh = true;
                continue;
            }

            let raw_body = response
                .bytes()
                .await
                .context("reading Code Assist response body")?;

            if !status.is_success() {
                anyhow::bail!(
                    "Code Assist API error ({}): {}",
                    status,
                    extract_error_message(&raw_body)
                );
            }

            let body: Value = serde_json::from_slice(&raw_body)
                .context("parsing Code Assist response as JSON")?;
            return parse_gemini_response(unwrap_response(&body));
        }
    }

    fn supports_tool_calling(&self) -> bool {
        true
    }

    fn context_window(&self) -> usize {
        self.ctx_window
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat_stream(
        &self,
        request: ChatRequest<'_>,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        let envelope = self.wrap_envelope(build_gemini_request_body(&request));
        let url =
            format!("{CODE_ASSIST_ENDPOINT}/v1internal:streamGenerateContent?alt=sse");

        let mut force_refresh = false;
        let response = loop {
            let token = self.access_token(force_refresh).await?;
            let response = self
                .client
                .post(&url)
                .bearer_auth(&token)
                .header("Content-Type", "application/json")
                .header("User-Agent", GEMINI_CLI_USER_AGENT)
                .header("X-Goog-Api-Client", X_GOOG_API_CLIENT)
                .header("Accept", "text/event-stream")
                .json(&envelope)
                .send()
                .await
                .context("sending streaming request to Code Assist API")?;

            let status = response.status();
            if (status.as_u16() == 401 || status.as_u16() == 403) && !force_refresh {
                force_refresh = true;
                continue;
            }
            if !status.is_success() {
                let raw_body = response
                    .bytes()
                    .await
                    .context("reading Code Assist streaming error body")?;
                anyhow::bail!(
                    "Code Assist API error ({}): {}",
                    status,
                    extract_error_message(&raw_body)
                );
            }
            break response;
        };

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let mut byte_stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut sse = SseBuffer::new();
            let mut usage_acc = UsageInfo::default();

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                        return;
                    }
                };
                sse.extend(&chunk);

                while let Some(line_bytes) = sse.next_line() {
                    let line = match std::str::from_utf8(&line_bytes) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let data_str = match line.strip_prefix("data: ") {
                        Some(s) => s,
                        None => continue,
                    };
                    if data_str == "[DONE]" {
                        finish_stream(&tx, &usage_acc).await;
                        return;
                    }
                    let data: Value = match serde_json::from_str(data_str) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    // Each event carries the same `{response:{…}}` envelope.
                    let inner = data
                        .get("response")
                        .filter(|v| v.is_object())
                        .cloned()
                        .unwrap_or(data);
                    if dispatch_gemini_event(&inner, &tx, &mut usage_acc).await {
                        finish_stream(&tx, &usage_acc).await;
                        return;
                    }
                }
            }

            finish_stream(&tx, &usage_acc).await;
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> GeminiCloudCodeProvider {
        GeminiCloudCodeProvider {
            home: PathBuf::from("/tmp/does-not-matter"),
            client: reqwest::Client::new(),
            model: "gemini-2.5-flash".to_string(),
            project_id: "proj-123".to_string(),
            ctx_window: DEFAULT_CONTEXT_WINDOW,
        }
    }

    #[test]
    fn wrap_envelope_has_project_model_and_request() {
        let p = provider();
        let inner = json!({ "contents": [] });
        let env = p.wrap_envelope(inner.clone());
        assert_eq!(env["project"], "proj-123");
        assert_eq!(env["model"], "gemini-2.5-flash");
        assert_eq!(env["request"], inner);
        assert!(env["user_prompt_id"].as_str().is_some());
    }

    #[test]
    fn unwrap_response_peels_envelope() {
        let wrapped = json!({ "response": { "candidates": [{ "x": 1 }] } });
        assert_eq!(unwrap_response(&wrapped), &json!({ "candidates": [{ "x": 1 }] }));
        // No envelope → identity.
        let bare = json!({ "candidates": [] });
        assert_eq!(unwrap_response(&bare), &bare);
        // `response` present but not an object → identity (don't peel).
        let odd = json!({ "response": "oops", "candidates": [] });
        assert_eq!(unwrap_response(&odd), &odd);
    }

    #[test]
    fn client_metadata_matches_gemini_cli_shape() {
        let m = client_metadata();
        assert_eq!(m["ideType"], "IDE_UNSPECIFIED");
        assert_eq!(m["platform"], "PLATFORM_UNSPECIFIED");
        assert_eq!(m["pluginType"], "GEMINI");
    }

    #[test]
    fn provider_identity() {
        let p = provider();
        assert_eq!(p.name(), "gemini-cloudcode");
        assert_eq!(p.model(), "gemini-2.5-flash");
        assert!(p.supports_tool_calling());
        assert!(p.supports_streaming());
        assert_eq!(p.context_window(), DEFAULT_CONTEXT_WINDOW);
    }
}
