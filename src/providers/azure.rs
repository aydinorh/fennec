//! Azure OpenAI / Azure AI Foundry provider.
//!
//! Azure exposes the **same Chat Completions request/response shape as OpenAI**
//! — it differs only in routing and auth — so this provider reuses the OpenAI
//! request builder, response parser, and streaming loop wholesale (see
//! [`super::openai`]) and layers Azure specifics on top:
//!
//!   - **Routing:** requests go to
//!     `{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=…`
//!     where `provider.base_url` is the resource endpoint
//!     (`https://<resource>.openai.azure.com`) and `provider.model` is the
//!     *deployment* name.
//!   - **Auth (auto-detected):** an API key (`api-key` header), or keyless
//!     Microsoft Entra ID — either by shelling out to the Azure CLI
//!     (`az account get-access-token`) or via a service-principal
//!     client-credentials flow (`AZURE_TENANT_ID` / `AZURE_CLIENT_ID` /
//!     `AZURE_CLIENT_SECRET`). Entra bearer tokens are cached and refreshed.
//!
//! Note: reasoning-model detection (which decides `max_completion_tokens` vs
//! `max_tokens` and whether to send `temperature`) keys off the deployment
//! name, so name reasoning-model deployments after their model family
//! (`o3-mini`, `gpt-5`, …) for it to apply.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;

use super::openai::{build_openai_request_body, spawn_openai_stream, OpenAIProvider};
use super::traits::{ChatRequest, ChatResponse, Provider, StreamEvent};

const DEFAULT_API_VERSION: &str = "2024-10-21";
const DEFAULT_CONTEXT_WINDOW: usize = 128_000;
/// Token audience for Azure OpenAI (`*.openai.azure.com`) resources. Foundry
/// users on `ai.azure.com` shapes can override via `AZURE_OPENAI_SCOPE`.
const DEFAULT_ENTRA_SCOPE: &str = "https://cognitiveservices.azure.com/.default";
/// Refresh Entra tokens this long before their real expiry.
const TOKEN_SKEW: Duration = Duration::from_secs(60);

/// How the provider authenticates to Azure.
enum AzureAuth {
    /// `api-key: <key>` header.
    ApiKey(String),
    /// Keyless Microsoft Entra ID (bearer token).
    Entra(EntraSource),
}

/// Where an Entra bearer token comes from.
enum EntraSource {
    /// Shell out to `az account get-access-token`.
    AzureCli,
    /// Service-principal client-credentials grant.
    ServicePrincipal {
        tenant_id: String,
        client_id: String,
        client_secret: String,
    },
}

/// Azure OpenAI / Foundry provider.
pub struct AzureProvider {
    client: reqwest::Client,
    /// Resource endpoint, e.g. `https://my-resource.openai.azure.com`.
    base_url: String,
    /// Azure *deployment* name (used both in the URL and as the body `model`).
    deployment: String,
    api_version: String,
    scope: String,
    ctx_window: usize,
    auth: AzureAuth,
    /// Cached Entra bearer token + its expiry (unused in API-key mode).
    cached_token: Mutex<Option<(String, Instant)>>,
}

impl AzureProvider {
    /// Create an Azure provider. The auth mode is auto-detected: a non-empty
    /// `api_key` selects API-key auth; otherwise a complete set of
    /// `AZURE_TENANT_ID`/`AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET` env vars
    /// selects the service-principal flow; otherwise the Azure CLI is used.
    pub fn new(
        api_key: String,
        deployment: Option<String>,
        base_url: Option<String>,
        context_window: Option<usize>,
    ) -> Self {
        let mut base = base_url.unwrap_or_default();
        while base.ends_with('/') {
            base.pop();
        }

        let auth = if !api_key.is_empty() {
            AzureAuth::ApiKey(api_key)
        } else if let (Ok(tenant_id), Ok(client_id), Ok(client_secret)) = (
            std::env::var("AZURE_TENANT_ID"),
            std::env::var("AZURE_CLIENT_ID"),
            std::env::var("AZURE_CLIENT_SECRET"),
        ) {
            if !tenant_id.is_empty() && !client_id.is_empty() && !client_secret.is_empty() {
                AzureAuth::Entra(EntraSource::ServicePrincipal {
                    tenant_id,
                    client_id,
                    client_secret,
                })
            } else {
                AzureAuth::Entra(EntraSource::AzureCli)
            }
        } else {
            AzureAuth::Entra(EntraSource::AzureCli)
        };

        Self {
            client: reqwest::Client::new(),
            base_url: base,
            deployment: deployment.unwrap_or_default(),
            api_version: std::env::var("AZURE_OPENAI_API_VERSION")
                .ok()
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_API_VERSION.to_string()),
            scope: std::env::var("AZURE_OPENAI_SCOPE")
                .ok()
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_ENTRA_SCOPE.to_string()),
            ctx_window: context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
            auth,
            cached_token: Mutex::new(None),
        }
    }

    fn chat_completions_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.base_url, self.deployment, self.api_version
        )
    }

    /// Apply the right auth header to a request builder.
    async fn apply_auth(&self, req: reqwest::RequestBuilder) -> Result<reqwest::RequestBuilder> {
        match &self.auth {
            AzureAuth::ApiKey(key) => Ok(req.header("api-key", key)),
            AzureAuth::Entra(_) => {
                let token = self.entra_token().await?;
                Ok(req.header("Authorization", format!("Bearer {token}")))
            }
        }
    }

    /// Return a valid Entra bearer token, refreshing (and caching) as needed.
    async fn entra_token(&self) -> Result<String> {
        if let Some((token, expiry)) = self.cached_token.lock().as_ref() {
            if *expiry > Instant::now() {
                return Ok(token.clone());
            }
        }

        let (token, ttl) = match &self.auth {
            AzureAuth::ApiKey(_) => anyhow::bail!("entra_token called in API-key mode"),
            AzureAuth::Entra(EntraSource::AzureCli) => self.fetch_token_via_cli().await?,
            AzureAuth::Entra(EntraSource::ServicePrincipal {
                tenant_id,
                client_id,
                client_secret,
            }) => {
                self.fetch_token_via_service_principal(tenant_id, client_id, client_secret)
                    .await?
            }
        };

        let expiry = Instant::now() + ttl.saturating_sub(TOKEN_SKEW);
        *self.cached_token.lock() = Some((token.clone(), expiry));
        Ok(token)
    }

    /// Mint a token by shelling out to the Azure CLI. Runs off the async
    /// executor since it spawns a subprocess.
    async fn fetch_token_via_cli(&self) -> Result<(String, Duration)> {
        let scope = self.scope.clone();
        let output = tokio::task::spawn_blocking(move || {
            std::process::Command::new("az")
                .args([
                    "account",
                    "get-access-token",
                    "--scope",
                    &scope,
                    "--output",
                    "json",
                ])
                .output()
        })
        .await
        .context("joining az CLI task")?
        .context(
            "running `az account get-access-token` — is the Azure CLI installed and logged in?",
        )?;

        if !output.status.success() {
            anyhow::bail!(
                "az CLI token request failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let parsed: Value = serde_json::from_slice(&output.stdout)
            .context("parsing az CLI token JSON")?;
        let token = parsed
            .get("accessToken")
            .and_then(|t| t.as_str())
            .context("az CLI response missing accessToken")?
            .to_string();
        // `az` reports `expiresOn` as a local timestamp that's fiddly to parse
        // portably; Azure access tokens last ~60 min, so refresh well inside
        // that window rather than trusting a parsed expiry.
        Ok((token, Duration::from_secs(50 * 60)))
    }

    /// Mint a token via the service-principal client-credentials grant.
    async fn fetch_token_via_service_principal(
        &self,
        tenant_id: &str,
        client_id: &str,
        client_secret: &str,
    ) -> Result<(String, Duration)> {
        let url =
            format!("https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token");
        let resp = self
            .client
            .post(&url)
            .form(&[
                ("grant_type", "client_credentials"),
                ("client_id", client_id),
                ("client_secret", client_secret),
                ("scope", self.scope.as_str()),
            ])
            .send()
            .await
            .context("requesting Entra token (service principal)")?;
        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .context("parsing Entra token response")?;
        if !status.is_success() {
            let err = body
                .get("error_description")
                .or_else(|| body.get("error"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            anyhow::bail!("Entra token request failed ({status}): {err}");
        }
        let token = body
            .get("access_token")
            .and_then(|t| t.as_str())
            .context("Entra response missing access_token")?
            .to_string();
        let ttl = body
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .unwrap_or(3600);
        Ok((token, Duration::from_secs(ttl)))
    }

    fn extract_error(raw: &[u8]) -> String {
        serde_json::from_slice::<Value>(raw)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| String::from_utf8_lossy(raw).chars().take(200).collect())
    }
}

#[async_trait]
impl Provider for AzureProvider {
    fn name(&self) -> &str {
        "azure"
    }

    fn model(&self) -> &str {
        &self.deployment
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let body = build_openai_request_body(&self.deployment, &request, false);
        let req = self
            .client
            .post(self.chat_completions_url())
            .header("Content-Type", "application/json");
        let req = self.apply_auth(req).await?;

        let response = req
            .json(&body)
            .send()
            .await
            .context("sending request to Azure OpenAI")?;

        let status = response.status();
        let raw_body = response
            .bytes()
            .await
            .context("reading Azure OpenAI response body")?;
        if !status.is_success() {
            anyhow::bail!("Azure OpenAI error ({}): {}", status, Self::extract_error(&raw_body));
        }

        let response_body: Value = serde_json::from_slice(&raw_body)
            .context("parsing Azure OpenAI response as JSON")?;
        OpenAIProvider::parse_response(&response_body)
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
        let body = build_openai_request_body(&self.deployment, &request, true);
        let req = self
            .client
            .post(self.chat_completions_url())
            .header("Content-Type", "application/json");
        let req = self.apply_auth(req).await?;

        let response = req
            .json(&body)
            .send()
            .await
            .context("sending streaming request to Azure OpenAI")?;

        let status = response.status();
        if !status.is_success() {
            let raw_body = response
                .bytes()
                .await
                .context("reading Azure OpenAI streaming error body")?;
            anyhow::bail!("Azure OpenAI error ({}): {}", status, Self::extract_error(&raw_body));
        }

        Ok(spawn_openai_stream(response))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_routes_to_deployment_with_api_version() {
        let p = AzureProvider::new(
            "k".to_string(),
            Some("gpt-4o-deploy".to_string()),
            Some("https://my-res.openai.azure.com/".to_string()),
            None,
        );
        assert_eq!(
            p.chat_completions_url(),
            "https://my-res.openai.azure.com/openai/deployments/gpt-4o-deploy/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn api_key_present_selects_api_key_auth() {
        let p = AzureProvider::new("secret".to_string(), None, None, None);
        assert!(matches!(p.auth, AzureAuth::ApiKey(_)));
    }

    #[test]
    fn empty_key_falls_back_to_entra() {
        // With no SP env vars set, an empty key resolves to the CLI flow.
        // (We avoid asserting the exact variant when AZURE_* env vars may be
        // present in the runner; just confirm it's the keyless branch.)
        let p = AzureProvider::new(String::new(), Some("dep".to_string()), None, None);
        assert!(matches!(p.auth, AzureAuth::Entra(_)));
    }

    #[test]
    fn model_accessor_returns_deployment() {
        let p = AzureProvider::new("k".to_string(), Some("my-deploy".to_string()), None, None);
        assert_eq!(p.model(), "my-deploy");
        assert_eq!(p.name(), "azure");
        assert!(p.supports_streaming());
        assert!(p.supports_tool_calling());
    }

    #[test]
    fn body_reuses_openai_shape_with_deployment_as_model() {
        // The body is built by the shared OpenAI builder with the deployment
        // name as the `model` field.
        let p = AzureProvider::new("k".to_string(), Some("gpt-4o".to_string()), None, None);
        let messages = vec![super::super::traits::ChatMessage::user("hi")];
        let request = ChatRequest {
            system: None,
            messages: &messages,
            tools: None,
            max_tokens: 256,
            temperature: 0.3,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };
        let body = build_openai_request_body(&p.deployment, &request, false);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["max_tokens"], 256);
        assert_eq!(body["messages"][0]["role"], "user");
    }
}
