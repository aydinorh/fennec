use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::SinkExt;
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message as WsMessage;

use crate::bus::InboundMessage;

use super::traits::{Channel, SendMessage};

/// Hard cap on the number of per-channel entries in
/// [`SlackChannel::last_edit`] / [`SlackChannel::chat_threads`]. When
/// exceeded, the oldest entry is evicted on insert.
const MAX_TRACKED_CHANNELS: usize = 1024;

/// Maximum number of times a request will retry on 429 (Too Many Requests)
/// before bailing.
const MAX_RETRY_AFTER_ATTEMPTS: u32 = 3;

/// Cap on `Retry-After` honored from a 429 response.
const RETRY_AFTER_CAP_SECS: u64 = 60;

/// Slack channel using Socket Mode (WebSocket) for events and Web API for
/// sending messages. Hand-rolled with `tokio-tungstenite` and `reqwest`.
pub struct SlackChannel {
    bot_token: String,
    app_token: String,
    client: reqwest::Client,
    allowed_users: Vec<String>,
    /// Per-channel timestamp of the last edit, used for rate-limiting streaming deltas.
    last_edit: Arc<Mutex<HashMap<String, Instant>>>,
    /// Per-channel most-recent inbound `thread_ts`. When the user messages
    /// inside a thread, we cache it so outbound replies (regular and
    /// streaming) land in the same thread instead of at top level.
    chat_threads: Arc<Mutex<HashMap<String, String>>>,
}

impl SlackChannel {
    pub fn new(bot_token: String, app_token: String, allowed_users: Vec<String>) -> Self {
        Self {
            bot_token,
            app_token,
            client: reqwest::Client::new(),
            allowed_users,
            last_edit: Arc::new(Mutex::new(HashMap::new())),
            chat_threads: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Insert into `last_edit` with LRU-style eviction at capacity.
    fn last_edit_insert(&self, chat_id: String, when: Instant) {
        let mut map = self.last_edit.lock();
        if map.len() >= MAX_TRACKED_CHANNELS && !map.contains_key(&chat_id) {
            if let Some(oldest_key) = map
                .iter()
                .min_by_key(|(_, v)| *v)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest_key);
            }
        }
        map.insert(chat_id, when);
    }

    /// Record the most recent thread_ts seen on `chat_id`. Bounded by
    /// `MAX_TRACKED_CHANNELS`; oldest entry by insertion order is dropped
    /// when at capacity. Insertion order is approximated by iteration order
    /// of `HashMap`, which is non-deterministic — fine for our use, since
    /// stale thread mappings just mean a reply lands at top level instead
    /// of a thread.
    fn remember_thread(&self, chat_id: String, thread_ts: String) {
        let mut map = self.chat_threads.lock();
        if map.len() >= MAX_TRACKED_CHANNELS && !map.contains_key(&chat_id) {
            if let Some(any_key) = map.keys().next().cloned() {
                map.remove(&any_key);
            }
        }
        map.insert(chat_id, thread_ts);
    }

    fn thread_for(&self, chat_id: &str) -> Option<String> {
        self.chat_threads.lock().get(chat_id).cloned()
    }

    /// POST to the Slack Web API honoring 429 `Retry-After`. Returns the
    /// parsed JSON body. Bails after `MAX_RETRY_AFTER_ATTEMPTS` retries or
    /// on any non-success status that isn't 429.
    async fn post_web_api(&self, method: &str, body: &Value, token: &str) -> Result<Value> {
        let url = format!("https://slack.com/api/{}", method);
        let mut attempt: u32 = 0;
        loop {
            let resp = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token))
                .json(body)
                .send()
                .await
                .with_context(|| format!("Slack {} request failed", method))?;
            let status = resp.status();

            if status.as_u16() == 429 && attempt < MAX_RETRY_AFTER_ATTEMPTS {
                let retry_after = parse_retry_after_header(&resp);
                tracing::warn!(
                    "Slack 429 on {}: sleeping {}s (attempt {}/{})",
                    method,
                    retry_after,
                    attempt + 1,
                    MAX_RETRY_AFTER_ATTEMPTS
                );
                tokio::time::sleep(Duration::from_secs(retry_after)).await;
                attempt += 1;
                continue;
            }

            if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                anyhow::bail!("Slack {} returned {}: {}", method, status, text);
            }

            // Slack also returns rate-limit errors as HTTP 200 with
            // `{ok: false, error: "ratelimited"}` in the body, sometimes
            // with `Retry-After` in headers. Without checking the body
            // we'd treat this as a hard failure and drop the request.
            // Capture the Retry-After header BEFORE moving the response
            // into `.json()`, since `resp.json()` consumes `resp`.
            let retry_after_header = parse_retry_after_header(&resp);
            let data: Value = resp
                .json()
                .await
                .with_context(|| format!("Slack {} parse failed", method))?;
            if data.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                let err = data.get("error").and_then(|v| v.as_str()).unwrap_or("unknown");
                if err == "ratelimited" && attempt < MAX_RETRY_AFTER_ATTEMPTS {
                    tracing::warn!(
                        "Slack ratelimited (200 body) on {}: sleeping {}s (attempt {}/{})",
                        method,
                        retry_after_header,
                        attempt + 1,
                        MAX_RETRY_AFTER_ATTEMPTS
                    );
                    tokio::time::sleep(Duration::from_secs(retry_after_header)).await;
                    attempt += 1;
                    continue;
                }
                anyhow::bail!("Slack {} error: {}", method, err);
            }
            return Ok(data);
        }
    }

    /// Parse a Slack Socket Mode events_api envelope. Returns
    /// (envelope_id, user, text, channel, thread_ts) if the envelope is a
    /// valid non-bot message event. `thread_ts` is `Some` when the message
    /// was posted inside a thread.
    pub fn parse_event_envelope(
        payload: &Value,
    ) -> Option<(String, String, String, String, Option<String>)> {
        let envelope_type = payload.get("type")?.as_str()?;
        if envelope_type != "events_api" {
            return None;
        }
        let envelope_id = payload.get("envelope_id")?.as_str()?.to_string();

        let event = payload.get("payload")?.get("event")?;
        let event_type = event.get("type")?.as_str()?;
        if event_type != "message" {
            return None;
        }

        // Skip bot messages (they have a bot_id field).
        if event.get("bot_id").is_some() {
            return None;
        }
        // Also skip message subtypes (edits, joins, etc.) — we only want plain messages.
        if event.get("subtype").is_some() {
            return None;
        }

        let user = event.get("user")?.as_str()?.to_string();
        let text = event.get("text")?.as_str()?.to_string();
        let channel = event.get("channel")?.as_str()?.to_string();
        let thread_ts = event
            .get("thread_ts")
            .and_then(|v| v.as_str())
            .map(String::from);

        Some((envelope_id, user, text, channel, thread_ts))
    }

    /// Build the acknowledgement JSON for a Socket Mode envelope.
    pub fn ack_envelope(envelope_id: &str) -> String {
        serde_json::json!({ "envelope_id": envelope_id }).to_string()
    }
}

/// Read `Retry-After` from a Slack response and clamp it. Used in two
/// retry paths: HTTP 429 and HTTP 200 with `error: "ratelimited"`.
/// Defaults to 1s if missing or unparseable; caps at
/// `RETRY_AFTER_CAP_SECS` so a hostile/bug-y header can't pin us in
/// a multi-minute sleep.
fn parse_retry_after_header(resp: &reqwest::Response) -> u64 {
    resp.headers()
        .get("Retry-After")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1)
        .min(RETRY_AFTER_CAP_SECS)
}

#[async_trait]
impl Channel for SlackChannel {
    fn name(&self) -> &str {
        "slack"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        let mut body = serde_json::json!({
            "channel": message.recipient,
            "text": message.content,
        });
        if let Some(thread_ts) = self.thread_for(&message.recipient) {
            body["thread_ts"] = Value::String(thread_ts);
        }
        self.post_web_api("chat.postMessage", &body, &self.bot_token).await?;
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<InboundMessage>) -> Result<()> {
        // 1. Open a Socket Mode connection
        let open_data = self
            .post_web_api(
                "apps.connections.open",
                &serde_json::json!({}),
                &self.app_token,
            )
            .await?;
        let ws_url = open_data
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Slack connection response missing 'url'"))?;

        // 2. Connect WebSocket
        let (ws_stream, _) = tokio_tungstenite::connect_async(ws_url)
            .await
            .context("Slack WebSocket connect failed")?;
        let (mut ws_tx, mut ws_rx) = ws_stream.split();

        // 3. Message loop
        while let Some(msg) = ws_rx.next().await {
            let msg = match msg {
                Ok(m) => m,
                Err(e) => {
                    anyhow::bail!("Slack WebSocket error: {}", e);
                }
            };

            if msg.is_close() {
                anyhow::bail!("Slack WebSocket received close frame");
            }

            let text = match msg.to_text() {
                Ok(t) => t,
                Err(_) => continue,
            };

            let payload: Value = match serde_json::from_str(text) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Check for disconnect envelope
            if payload.get("type").and_then(|v| v.as_str()) == Some("disconnect") {
                tracing::info!("Slack: received disconnect envelope, will reconnect");
                anyhow::bail!("Slack Socket Mode disconnect received");
            }

            // Try to parse as an events_api message
            if let Some((envelope_id, user, content, channel, thread_ts)) =
                Self::parse_event_envelope(&payload)
            {
                // Send acknowledgement immediately
                let ack = Self::ack_envelope(&envelope_id);
                let _ = ws_tx.send(WsMessage::Text(ack.into())).await;

                if !self.allows_sender(&user) {
                    tracing::debug!("Slack: ignoring message from disallowed sender {}", user);
                    continue;
                }

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Cache thread_ts so outbound replies (regular and streaming)
                // land in the same thread the user posted in.
                let mut metadata = HashMap::new();
                if let Some(ts) = &thread_ts {
                    metadata.insert("thread_ts".to_string(), ts.clone());
                    self.remember_thread(channel.clone(), ts.clone());
                } else {
                    // User posted at top level — clear any stale thread mapping
                    // so we don't accidentally reply into an old thread.
                    self.chat_threads.lock().remove(&channel);
                }

                let msg = InboundMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    sender: user,
                    content,
                    channel: "slack".to_string(),
                    chat_id: channel,
                    timestamp: now,
                    reply_to: None,
                    metadata,
                };

                if tx.send(msg).await.is_err() {
                    tracing::info!("Slack: inbound channel closed, stopping listener");
                    return Ok(());
                }
            }
        }

        // The `while let Some(msg) = ws_rx.next().await` loop exits to
        // here when the WebSocket stream returns `None` — the server
        // closed the underlying TCP without sending a Close frame
        // (network blip, server hiccup, Socket Mode 60-min rotation).
        // Returning `Ok(())` here would tell the supervisor "channel
        // exited cleanly, do not restart" — Slack would then go quiet
        // until the user notices and bounces the process. Surfacing
        // this as an error lets the supervisor's retry-with-backoff
        // path take over.
        anyhow::bail!("Slack WebSocket closed unexpectedly (no close frame); restarting")
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn send_streaming_start(&self, chat_id: &str) -> Result<Option<String>> {
        let mut body = serde_json::json!({
            "channel": chat_id,
            "text": "...",
        });
        if let Some(thread_ts) = self.thread_for(chat_id) {
            body["thread_ts"] = Value::String(thread_ts);
        }
        let data = self.post_web_api("chat.postMessage", &body, &self.bot_token).await?;
        // Slack returns the message timestamp as `ts` which serves as the message ID.
        let ts = data.get("ts").and_then(|v| v.as_str()).map(String::from);
        Ok(ts)
    }

    async fn send_streaming_delta(
        &self,
        chat_id: &str,
        message_id: &str,
        full_text: &str,
    ) -> Result<()> {
        // Rate-limit: skip if last edit was <300ms ago.
        {
            let map = self.last_edit.lock();
            if let Some(last) = map.get(chat_id) {
                if last.elapsed().as_millis() < 300 {
                    return Ok(());
                }
            }
        }

        let body = serde_json::json!({
            "channel": chat_id,
            "ts": message_id,
            "text": full_text,
        });
        self.post_web_api("chat.update", &body, &self.bot_token).await?;

        self.last_edit_insert(chat_id.to_string(), Instant::now());

        Ok(())
    }

    async fn send_streaming_end(
        &self,
        chat_id: &str,
        message_id: &str,
        full_text: &str,
    ) -> Result<()> {
        let body = serde_json::json!({
            "channel": chat_id,
            "ts": message_id,
            "text": full_text,
        });
        self.post_web_api("chat.update", &body, &self.bot_token).await?;

        {
            let mut map = self.last_edit.lock();
            map.remove(chat_id);
        }

        Ok(())
    }

    fn allows_sender(&self, sender_id: &str) -> bool {
        // Default-DENY: an empty allowlist refuses everyone ("*" opts
        // into allow-everyone). See the telegram channel for rationale.
        if self.allowed_users.is_empty() {
            tracing::warn!(
                "Slack: refusing message from '{sender_id}' — no allowed_users configured. \
                 Add your user ID to [channels.slack].allowed_users (or \"*\" to allow everyone)."
            );
            return false;
        }
        if self.allowed_users.iter().any(|u| u == "*") {
            return true;
        }
        self.allowed_users.iter().any(|u| u == sender_id)
    }
}
