use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde_json::Value;

use crate::bus::InboundMessage;
use crate::security::ct::ct_eq_bytes;

use super::traits::{Channel, SendMessage};

type HmacSha256 = Hmac<Sha256>;

/// WhatsApp channel using Meta's WhatsApp Cloud API.
///
/// Unlike Telegram (long-polling) or Discord/Slack (WebSocket), WhatsApp uses
/// webhooks. The `listen` method starts a mini HTTP server (axum) that receives
/// webhook POST requests from Meta.
///
/// Webhook verification: GET request with
///   `hub.mode=subscribe&hub.verify_token=...&hub.challenge=...`
/// returns the challenge value.
///
/// Message webhook: POST with messages array. When `app_secret` is set, the
/// body is first verified against the `X-Hub-Signature-256` HMAC Meta sends,
/// before parsing or acting on any field.
///
/// Sending: POST to
///   `https://graph.facebook.com/v21.0/{phone_number_id}/messages`
pub struct WhatsAppChannel {
    phone_number_id: String,
    access_token: String,
    verify_token: String,
    /// Meta App Secret for HMAC-SHA256 verification of inbound webhooks.
    /// None means signature verification is disabled (dev only).
    app_secret: Option<String>,
    webhook_port: u16,
    client: reqwest::Client,
    allowed_users: Vec<String>,
}

impl WhatsAppChannel {
    pub fn new(
        phone_number_id: String,
        access_token: String,
        verify_token: String,
        webhook_port: u16,
        allowed_users: Vec<String>,
        app_secret: String,
    ) -> Self {
        let app_secret = if app_secret.is_empty() {
            None
        } else {
            Some(app_secret)
        };
        Self {
            phone_number_id,
            access_token,
            verify_token,
            app_secret,
            webhook_port,
            client: reqwest::Client::new(),
            allowed_users,
        }
    }

    fn api_url(&self, path: &str) -> String {
        format!(
            "https://graph.facebook.com/v21.0/{}{}",
            self.phone_number_id, path
        )
    }

    /// Parse a WhatsApp webhook payload into a list of (sender_phone, message_text, message_id).
    pub fn parse_webhook_messages(body: &Value) -> Vec<(String, String, String)> {
        let mut results = Vec::new();

        let entries = match body.get("entry").and_then(|v| v.as_array()) {
            Some(arr) => arr,
            None => return results,
        };

        for entry in entries {
            let changes = match entry.get("changes").and_then(|v| v.as_array()) {
                Some(arr) => arr,
                None => continue,
            };

            for change in changes {
                let value = match change.get("value") {
                    Some(v) => v,
                    None => continue,
                };

                let messages = match value.get("messages").and_then(|v| v.as_array()) {
                    Some(arr) => arr,
                    None => continue,
                };

                for message in messages {
                    let msg_type = message
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if msg_type != "text" {
                        continue;
                    }

                    let from = message
                        .get("from")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let text = message
                        .get("text")
                        .and_then(|t| t.get("body"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let msg_id = message
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    if !from.is_empty() && !text.is_empty() {
                        results.push((from, text, msg_id));
                    }
                }
            }
        }

        results
    }
}

#[async_trait]
impl Channel for WhatsAppChannel {
    fn name(&self) -> &str {
        "whatsapp"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        let body = serde_json::json!({
            "messaging_product": "whatsapp",
            "to": message.recipient,
            "type": "text",
            "text": {
                "body": message.content
            }
        });

        let resp = self
            .client
            .post(self.api_url("/messages"))
            .header("Authorization", format!("Bearer {}", self.access_token))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("WhatsApp send message request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("WhatsApp send message returned {}: {}", status, text);
        }
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<InboundMessage>) -> Result<()> {
        use axum::body::Bytes;
        use axum::extract::{Query, State};
        use axum::http::HeaderMap;
        use axum::routing::{get, post};
        use axum::Router;

        #[derive(Clone)]
        struct WebhookState {
            verify_token: String,
            app_secret: Option<String>,
            allowed_users: Vec<String>,
            tx: tokio::sync::mpsc::Sender<InboundMessage>,
        }

        #[derive(serde::Deserialize)]
        struct VerifyParams {
            #[serde(rename = "hub.mode")]
            hub_mode: Option<String>,
            #[serde(rename = "hub.verify_token")]
            hub_verify_token: Option<String>,
            #[serde(rename = "hub.challenge")]
            hub_challenge: Option<String>,
        }

        // Webhook verification handler (GET).
        async fn verify_webhook(
            State(state): State<WebhookState>,
            Query(params): Query<VerifyParams>,
        ) -> axum::response::Response {
            use axum::http::StatusCode;
            use axum::response::IntoResponse;

            let mode_ok = params.hub_mode.as_deref() == Some("subscribe");
            let token_ok = params
                .hub_verify_token
                .as_deref()
                .map(|provided| {
                    ct_eq_bytes(provided.as_bytes(), state.verify_token.as_bytes())
                })
                .unwrap_or(false);

            if mode_ok && token_ok {
                if let Some(challenge) = params.hub_challenge {
                    return (StatusCode::OK, challenge).into_response();
                }
            }
            (StatusCode::FORBIDDEN, "Verification failed").into_response()
        }

        // Message webhook handler (POST). Takes the raw body so we can verify
        // the HMAC over the exact bytes Meta signed before parsing.
        async fn receive_webhook(
            State(state): State<WebhookState>,
            headers: HeaderMap,
            body_bytes: Bytes,
        ) -> axum::http::StatusCode {
            if let Some(secret) = state.app_secret.as_deref() {
                let sig_header = headers
                    .get("x-hub-signature-256")
                    .and_then(|v| v.to_str().ok());
                if let Err(e) = verify_meta_signature(secret, &body_bytes, sig_header) {
                    tracing::warn!("WhatsApp webhook signature rejected: {}", e);
                    return axum::http::StatusCode::UNAUTHORIZED;
                }
            }

            let body: Value = match serde_json::from_slice(&body_bytes) {
                Ok(v) => v,
                Err(e) => {
                    tracing::debug!("WhatsApp webhook body not valid JSON: {}", e);
                    return axum::http::StatusCode::BAD_REQUEST;
                }
            };

            let messages = WhatsAppChannel::parse_webhook_messages(&body);

            for (sender, text, _msg_id) in messages {
                // Check allowed users — default-DENY on an empty list,
                // mirroring `allows_sender` ("*" opts into everyone).
                let sender_allowed = !state.allowed_users.is_empty()
                    && (state.allowed_users.iter().any(|u| u == "*")
                        || state.allowed_users.iter().any(|u| u == &sender));
                if !sender_allowed {
                    if state.allowed_users.is_empty() {
                        tracing::warn!(
                            "WhatsApp: refusing message from '{sender}' — no allowed_users \
                             configured. Add your number to [channels.whatsapp].allowed_users \
                             (or \"*\" to allow everyone)."
                        );
                    } else {
                        tracing::debug!(
                            "WhatsApp: ignoring message from disallowed sender {}",
                            sender
                        );
                    }
                    continue;
                }

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                let msg = InboundMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    sender: sender.clone(),
                    content: text,
                    channel: "whatsapp".to_string(),
                    chat_id: sender,
                    timestamp: now,
                    reply_to: None,
                    metadata: HashMap::new(),
                };

                if state.tx.send(msg).await.is_err() {
                    tracing::info!("WhatsApp: inbound channel closed");
                    break;
                }
            }

            axum::http::StatusCode::OK
        }

        if self.app_secret.is_none() {
            tracing::warn!(
                "WhatsApp: app_secret is not configured — webhook HMAC \
                 verification is DISABLED. Anyone able to reach the webhook \
                 port can inject messages. Set channels.whatsapp.app_secret \
                 before exposing this port to the internet."
            );
        }

        let state = WebhookState {
            verify_token: self.verify_token.clone(),
            app_secret: self.app_secret.clone(),
            allowed_users: self.allowed_users.clone(),
            tx,
        };

        let app = Router::new()
            .route("/webhook/whatsapp", get(verify_webhook))
            .route("/webhook/whatsapp", post(receive_webhook))
            .with_state(state);

        let addr = format!("0.0.0.0:{}", self.webhook_port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .context("WhatsApp webhook server bind failed")?;

        tracing::info!("WhatsApp webhook listening on {}", addr);
        axum::serve(listener, app)
            .await
            .context("WhatsApp webhook server error")?;

        Ok(())
    }

    fn allows_sender(&self, sender_id: &str) -> bool {
        // Default-DENY: an empty allowlist refuses everyone ("*" opts
        // into allow-everyone). See the telegram channel for rationale.
        if self.allowed_users.is_empty() {
            tracing::warn!(
                "WhatsApp: refusing message from '{sender_id}' — no allowed_users configured. \
                 Add your number to [channels.whatsapp].allowed_users (or \"*\" to allow everyone)."
            );
            return false;
        }
        if self.allowed_users.iter().any(|u| u == "*") {
            return true;
        }
        self.allowed_users.iter().any(|u| u == sender_id)
    }
}

/// Verify Meta's `X-Hub-Signature-256` HMAC over the raw webhook body.
///
/// Meta sends `X-Hub-Signature-256: sha256=<hex>` where `<hex>` is the
/// HMAC-SHA256 of the raw request body keyed with the app secret. See
/// https://developers.facebook.com/docs/graph-api/webhooks/getting-started
fn verify_meta_signature(
    app_secret: &str,
    body: &[u8],
    signature_header: Option<&str>,
) -> Result<()> {
    let sig = signature_header
        .ok_or_else(|| anyhow::anyhow!("missing X-Hub-Signature-256 header"))?;
    let hex_sig = sig
        .strip_prefix("sha256=")
        .ok_or_else(|| anyhow::anyhow!("signature header missing sha256= prefix"))?;
    let received = hex::decode(hex_sig.trim())
        .map_err(|_| anyhow::anyhow!("signature is not valid hex"))?;

    let mut mac = HmacSha256::new_from_slice(app_secret.as_bytes())
        .map_err(|_| anyhow::anyhow!("failed to init HMAC key"))?;
    mac.update(body);
    let expected = mac.finalize().into_bytes();

    if !ct_eq_bytes(&received, expected.as_slice()) {
        anyhow::bail!("signature mismatch");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_webhook_messages_valid() {
        let payload = serde_json::json!({
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "123456",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "15551234567",
                            "phone_number_id": "123456789"
                        },
                        "contacts": [{
                            "profile": {"name": "Alice"},
                            "wa_id": "15559876543"
                        }],
                        "messages": [{
                            "from": "15559876543",
                            "id": "wamid.abc123",
                            "timestamp": "1700000000",
                            "text": {"body": "Hello Fennec!"},
                            "type": "text"
                        }]
                    },
                    "field": "messages"
                }]
            }]
        });

        let messages = WhatsAppChannel::parse_webhook_messages(&payload);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].0, "15559876543");
        assert_eq!(messages[0].1, "Hello Fennec!");
        assert_eq!(messages[0].2, "wamid.abc123");
    }

    #[test]
    fn test_parse_webhook_messages_empty() {
        let payload = serde_json::json!({
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "123456",
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp",
                        "statuses": []
                    },
                    "field": "messages"
                }]
            }]
        });

        let messages = WhatsAppChannel::parse_webhook_messages(&payload);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_parse_webhook_messages_non_text_skipped() {
        let payload = serde_json::json!({
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": "15559876543",
                            "id": "wamid.img123",
                            "type": "image",
                            "image": {"id": "img-id"}
                        }]
                    }
                }]
            }]
        });

        let messages = WhatsAppChannel::parse_webhook_messages(&payload);
        assert!(messages.is_empty());
    }

    #[test]
    fn test_allows_sender() {
        let ch = WhatsAppChannel::new(
            "123".to_string(),
            "token".to_string(),
            "verify".to_string(),
            9090,
            vec!["15551234567".to_string()],
            String::new(),
        );
        assert!(ch.allows_sender("15551234567"));
        assert!(!ch.allows_sender("15559999999"));

        // Empty list denies all (default-deny; "*" is the explicit opt-in)
        let ch2 = WhatsAppChannel::new(
            "123".to_string(),
            "token".to_string(),
            "verify".to_string(),
            9090,
            vec![],
            String::new(),
        );
        assert!(!ch2.allows_sender("anyone"));

        // Wildcard allows all
        let ch3 = WhatsAppChannel::new(
            "123".to_string(),
            "token".to_string(),
            "verify".to_string(),
            9090,
            vec!["*".to_string()],
            String::new(),
        );
        assert!(ch3.allows_sender("anyone"));
    }

    #[test]
    fn empty_app_secret_means_verification_disabled() {
        let ch = WhatsAppChannel::new(
            "123".to_string(),
            "token".to_string(),
            "verify".to_string(),
            9090,
            vec![],
            String::new(),
        );
        assert!(ch.app_secret.is_none());
    }

    #[test]
    fn nonempty_app_secret_is_retained() {
        let ch = WhatsAppChannel::new(
            "123".to_string(),
            "token".to_string(),
            "verify".to_string(),
            9090,
            vec![],
            "the-app-secret".to_string(),
        );
        assert_eq!(ch.app_secret.as_deref(), Some("the-app-secret"));
    }

    /// Compute the valid signature for a (secret, body) pair so the test can
    /// present the "correct" header to `verify_meta_signature`.
    fn valid_signature_for(secret: &str, body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        format!("sha256={}", hex::encode(mac.finalize().into_bytes()))
    }

    #[test]
    fn verify_signature_accepts_valid() {
        let secret = "topsecret";
        let body = br#"{"entry":[]}"#;
        let sig = valid_signature_for(secret, body);
        assert!(verify_meta_signature(secret, body, Some(&sig)).is_ok());
    }

    #[test]
    fn verify_signature_rejects_tampered_body() {
        let secret = "topsecret";
        let body = br#"{"entry":[]}"#;
        let sig = valid_signature_for(secret, body);
        let tampered = br#"{"entry":[{"hi":1}]}"#;
        assert!(verify_meta_signature(secret, tampered, Some(&sig)).is_err());
    }

    #[test]
    fn verify_signature_rejects_wrong_secret() {
        let body = br#"{"entry":[]}"#;
        let sig = valid_signature_for("the-real-secret", body);
        assert!(verify_meta_signature("different-secret", body, Some(&sig)).is_err());
    }

    #[test]
    fn verify_signature_rejects_missing_header() {
        assert!(verify_meta_signature("s", b"{}", None).is_err());
    }

    #[test]
    fn verify_signature_rejects_wrong_prefix() {
        assert!(
            verify_meta_signature("s", b"{}", Some("sha1=deadbeef")).is_err()
        );
    }

    #[test]
    fn verify_signature_rejects_non_hex_body() {
        assert!(
            verify_meta_signature("s", b"{}", Some("sha256=zzz-not-hex")).is_err()
        );
    }
}
