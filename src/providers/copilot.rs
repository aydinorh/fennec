//! GitHub Copilot provider.
//!
//! Copilot's chat endpoint (`api.githubcopilot.com/chat/completions`) is
//! OpenAI-compatible, so this reuses the OpenAI request builder, response
//! parser, and streaming loop (see [`super::openai`]) and layers on Copilot's
//! auth: a short-lived Copilot API token (obtained by exchanging a GitHub
//! OAuth token — see [`crate::auth::github_copilot`]) plus the Copilot client
//! headers. The token is cached, refreshed before expiry, and force-refreshed
//! once on a 401/403.
//!
//! Only the current chat-API path is implemented; the older `gh copilot` ACP
//! subprocess path is deprecated upstream and intentionally not ported.

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;

use crate::auth::github_copilot;

use super::openai::{build_openai_request_body, spawn_openai_stream, OpenAIProvider};
use super::traits::{ChatRequest, ChatResponse, Provider, StreamEvent};

const COPILOT_BASE_URL: &str = "https://api.githubcopilot.com";
const DEFAULT_MODEL: &str = "gpt-4o";
const DEFAULT_CONTEXT_WINDOW: usize = 128_000;
const TOKEN_REFRESH_MARGIN_SECS: u64 = 120;

/// GitHub Copilot chat provider.
pub struct CopilotProvider {
    client: reqwest::Client,
    model: String,
    base_url: String,
    ctx_window: usize,
    /// Cached Copilot API token + its unix expiry.
    cached_token: Mutex<Option<(String, u64)>>,
}

impl CopilotProvider {
    pub fn new(model: Option<String>, base_url: Option<String>, context_window: Option<usize>) -> Self {
        let mut base = base_url.unwrap_or_else(|| COPILOT_BASE_URL.to_string());
        while base.ends_with('/') {
            base.pop();
        }
        Self {
            client: reqwest::Client::new(),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            base_url: base,
            ctx_window: context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
            cached_token: Mutex::new(None),
        }
    }

    /// Return a valid Copilot API token, exchanging a resolved GitHub token
    /// when the cache is empty/expired (or `force`).
    async fn bearer(&self, force: bool) -> Result<String> {
        if !force {
            if let Some((token, expires_at)) = self.cached_token.lock().as_ref() {
                if *expires_at > now_secs() + TOKEN_REFRESH_MARGIN_SECS {
                    return Ok(token.clone());
                }
            }
        }

        let github_token = tokio::task::spawn_blocking(github_copilot::resolve_github_token)
            .await
            .context("joining GitHub token resolution")?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no GitHub token for Copilot — run `fennec login --provider copilot`, \
                     set GH_TOKEN, or `gh auth login`"
                )
            })?;

        let (token, expires_at) =
            github_copilot::exchange_copilot_token(&self.client, &github_token).await?;
        *self.cached_token.lock() = Some((token.clone(), expires_at));
        Ok(token)
    }

    fn apply_headers(&self, req: reqwest::RequestBuilder, token: &str) -> reqwest::RequestBuilder {
        let mut req = req
            .header("Authorization", format!("Bearer {token}"))
            .header("Content-Type", "application/json");
        for (name, value) in github_copilot::copilot_request_headers() {
            req = req.header(name, value);
        }
        req
    }

    fn url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn extract_error(raw: &[u8]) -> String {
    serde_json::from_slice::<Value>(raw)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message").and_then(|m| m.as_str()))
                .or_else(|| v.get("message").and_then(|m| m.as_str()))
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| String::from_utf8_lossy(raw).chars().take(200).collect())
}

#[async_trait]
impl Provider for CopilotProvider {
    fn name(&self) -> &str {
        "copilot"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let body = build_openai_request_body(&self.model, &request, false);
        let url = self.url();

        let mut force_refresh = false;
        loop {
            let token = self.bearer(force_refresh).await?;
            let response = self
                .apply_headers(self.client.post(&url), &token)
                .json(&body)
                .send()
                .await
                .context("sending request to Copilot API")?;

            let status = response.status();
            if (status.as_u16() == 401 || status.as_u16() == 403) && !force_refresh {
                force_refresh = true;
                continue;
            }

            let raw_body = response.bytes().await.context("reading Copilot response body")?;
            if !status.is_success() {
                anyhow::bail!("Copilot API error ({}): {}", status, extract_error(&raw_body));
            }
            let body: Value = serde_json::from_slice(&raw_body)
                .context("parsing Copilot response as JSON")?;
            return OpenAIProvider::parse_response(&body);
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
        let body = build_openai_request_body(&self.model, &request, true);
        let url = self.url();

        let mut force_refresh = false;
        let response = loop {
            let token = self.bearer(force_refresh).await?;
            let response = self
                .apply_headers(self.client.post(&url), &token)
                .json(&body)
                .send()
                .await
                .context("sending streaming request to Copilot API")?;

            let status = response.status();
            if (status.as_u16() == 401 || status.as_u16() == 403) && !force_refresh {
                force_refresh = true;
                continue;
            }
            if !status.is_success() {
                let raw_body = response
                    .bytes()
                    .await
                    .context("reading Copilot streaming error body")?;
                anyhow::bail!("Copilot API error ({}): {}", status, extract_error(&raw_body));
            }
            break response;
        };

        Ok(spawn_openai_stream(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_targets_chat_completions() {
        let p = CopilotProvider::new(None, None, None);
        assert_eq!(p.url(), "https://api.githubcopilot.com/chat/completions");
        assert_eq!(p.model(), "gpt-4o");
        assert_eq!(p.name(), "copilot");
        assert!(p.supports_streaming());
        assert!(p.supports_tool_calling());
    }

    #[test]
    fn base_url_override_trims_slash() {
        let p = CopilotProvider::new(
            Some("o1".to_string()),
            Some("https://proxy.example.com/".to_string()),
            None,
        );
        assert_eq!(p.url(), "https://proxy.example.com/chat/completions");
        assert_eq!(p.model(), "o1");
    }

    #[test]
    fn extract_error_reads_message() {
        assert_eq!(
            extract_error(br#"{"error":{"message":"bad model"}}"#),
            "bad model"
        );
        assert_eq!(extract_error(br#"{"message":"nope"}"#), "nope");
    }

    #[test]
    fn body_reuses_openai_shape() {
        let p = CopilotProvider::new(Some("gpt-4o".to_string()), None, None);
        let messages = vec![super::super::traits::ChatMessage::user("hi")];
        let request = ChatRequest {
            system: None,
            messages: &messages,
            tools: None,
            max_tokens: 256,
            temperature: 0.4,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };
        let body = build_openai_request_body(&p.model, &request, false);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["messages"][0]["role"], "user");
    }
}
