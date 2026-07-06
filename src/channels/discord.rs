use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::SinkExt;
use futures::StreamExt;
use futures::stream::SplitSink;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::bus::InboundMessage;

use super::telegram::split_message;
use super::traits::{Channel, SendMessage};

/// Discord Gateway intents:
///   GUILDS          = 1 << 0  = 1
///   GUILD_MESSAGES  = 1 << 9  = 512
///   DIRECT_MESSAGES = 1 << 12 = 4096
///   MESSAGE_CONTENT = 1 << 15 = 32768
pub const DISCORD_INTENTS: u64 = 1 + 512 + 4096 + 32768; // 37377

/// Maximum length of a single message's `content` on Discord's REST API.
/// Bots without Nitro are capped at 2000 characters.
const DISCORD_MAX_CONTENT: usize = 2000;

/// Hard cap on the number of per-channel entries in
/// [`DiscordChannel::last_edit`]. When exceeded, the oldest entry is evicted
/// on insert.
const MAX_LAST_EDIT_ENTRIES: usize = 1024;

/// State carried across reconnects to support Gateway RESUME (op:6) per the
/// Discord docs: `session_id` and `resume_gateway_url` come from the READY
/// dispatch; `seq` is the most recent sequence number observed.
#[derive(Clone)]
struct ResumeState {
    session_id: String,
    resume_gateway_url: String,
    seq: Option<u64>,
}

/// Disposition returned from one connection's event loop, used by the outer
/// reconnect loop in [`DiscordChannel::listen`].
enum LoopDisposition {
    /// The inbound bus was closed; exit cleanly.
    InboundClosed,
    /// Reconnect using the cached `ResumeState` (op:7, zombie heartbeat,
    /// or op:9 with `d:true`).
    Reconnect,
    /// Drop the resume state and IDENTIFY fresh against the original gateway
    /// URL (op:9 with `d:false`).
    Reidentify,
}

type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>;

/// Discord channel using the Gateway (WebSocket) for events and REST API for
/// sending messages. Hand-rolled with `tokio-tungstenite` and `reqwest`.
pub struct DiscordChannel {
    bot_token: String,
    client: reqwest::Client,
    allowed_users: Vec<String>,
    /// Per-channel timestamp of the last edit, used for rate-limiting streaming deltas.
    last_edit: Arc<Mutex<HashMap<String, Instant>>>,
}

impl DiscordChannel {
    pub fn new(bot_token: String, allowed_users: Vec<String>) -> Self {
        Self {
            bot_token,
            client: reqwest::Client::new(),
            allowed_users,
            last_edit: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Parse a Discord MESSAGE_CREATE dispatch payload into
    /// (author_id, channel_id, content, message_id, is_bot).
    pub fn parse_message_create(d: &Value) -> Option<(String, String, String, String, bool)> {
        let author = d.get("author")?;
        let author_id = author.get("id")?.as_str()?.to_string();
        let is_bot = author.get("bot").and_then(|v| v.as_bool()).unwrap_or(false);
        let channel_id = d.get("channel_id")?.as_str()?.to_string();
        let content = d.get("content")?.as_str()?.to_string();
        let message_id = d.get("id")?.as_str()?.to_string();
        Some((author_id, channel_id, content, message_id, is_bot))
    }

    /// Compute the intent value (exposed for testing).
    pub fn intents() -> u64 {
        DISCORD_INTENTS
    }

    fn rest_url(&self, path: &str) -> String {
        format!("https://discord.com/api/v10{}", path)
    }

    /// Insert into `last_edit` with LRU-style eviction at capacity.
    fn last_edit_insert(&self, chat_id: String, when: Instant) {
        let mut map = self.last_edit.lock();
        if map.len() >= MAX_LAST_EDIT_ENTRIES && !map.contains_key(&chat_id) {
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

    /// POST a single message (no splitting). Used by both the public
    /// [`Channel::send`] flow and the streaming end's continuation messages.
    async fn post_message(&self, channel_id: &str, content: &str) -> Result<Value> {
        let url = self.rest_url(&format!("/channels/{}/messages", channel_id));
        let body = serde_json::json!({ "content": content });
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .json(&body)
            .send()
            .await
            .context("Discord post_message request failed")?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Discord post_message returned {}: {}", status, text);
        }
        resp.json().await.context("Discord post_message parse failed")
    }

    /// PATCH an existing message's content.
    async fn patch_message(&self, channel_id: &str, message_id: &str, content: &str) -> Result<()> {
        let url = self.rest_url(&format!(
            "/channels/{}/messages/{}",
            channel_id, message_id
        ));
        let body = serde_json::json!({ "content": content });
        let resp = self
            .client
            .patch(&url)
            .header("Authorization", format!("Bot {}", self.bot_token))
            .json(&body)
            .send()
            .await
            .context("Discord patch_message request failed")?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Discord patch_message returned {}: {}", status, text);
        }
        Ok(())
    }

    /// Fetch a fresh gateway URL via REST. Used on initial connect and on
    /// op:9 Invalid Session with `d:false`.
    async fn fetch_gateway_url(&self) -> Result<String> {
        let resp = self
            .client
            .get(self.rest_url("/gateway/bot"))
            .header("Authorization", format!("Bot {}", self.bot_token))
            .send()
            .await
            .context("Discord gateway/bot request failed")?;
        let data: Value = resp
            .json()
            .await
            .context("Discord gateway/bot parse failed")?;
        let url = data
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Discord gateway response missing 'url'"))?
            .to_string();
        Ok(format!("{}?v=10&encoding=json", url))
    }

    fn identify_payload(&self) -> Value {
        serde_json::json!({
            "op": 2,
            "d": {
                "token": self.bot_token,
                "intents": DISCORD_INTENTS,
                "properties": {
                    "os": "linux",
                    "browser": "fennec",
                    "device": "fennec"
                }
            }
        })
    }

    fn resume_payload(&self, state: &ResumeState) -> Value {
        serde_json::json!({
            "op": 6,
            "d": {
                "token": self.bot_token,
                "session_id": state.session_id,
                "seq": state.seq,
            }
        })
    }

    /// Run one connection's event loop. Heartbeat-ACK enforcement: if a
    /// heartbeat tick arrives before the previous heartbeat has been ACKed
    /// (op:11), the connection is "zombied" per the Discord docs — close
    /// with a non-`1000`/`1001` code and signal a Reconnect so the outer
    /// loop RESUMEs.
    async fn run_event_loop(
        &self,
        ws_tx: &mut WsSink,
        ws_rx: &mut futures::stream::SplitStream<
            WebSocketStream<MaybeTlsStream<TcpStream>>,
        >,
        resume: &mut Option<ResumeState>,
        tx: &tokio::sync::mpsc::Sender<InboundMessage>,
        heartbeat_interval_ms: u64,
    ) -> Result<LoopDisposition> {
        let mut heartbeat_interval =
            tokio::time::interval(Duration::from_millis(heartbeat_interval_ms));
        // Skip the initial immediate tick (interval fires once on creation).
        heartbeat_interval.tick().await;
        let mut ack_received = true;

        loop {
            tokio::select! {
                _ = heartbeat_interval.tick() => {
                    if !ack_received {
                        // Zombied connection: close with non-1000/1001 code
                        // and signal RESUME.
                        let _ = ws_tx
                            .send(WsMessage::Close(Some(CloseFrame {
                                code: 4000_u16.into(),
                                reason: "missed heartbeat ack".into(),
                            })))
                            .await;
                        tracing::warn!("Discord heartbeat ACK missed; reconnecting with RESUME");
                        return Ok(LoopDisposition::Reconnect);
                    }
                    let seq = resume.as_ref().and_then(|r| r.seq);
                    let hb = serde_json::json!({ "op": 1, "d": seq });
                    if ws_tx.send(WsMessage::Text(hb.to_string().into())).await.is_err() {
                        anyhow::bail!("Discord WebSocket closed while sending heartbeat");
                    }
                    ack_received = false;
                }
                msg = ws_rx.next() => {
                    let msg = match msg {
                        Some(Ok(m)) => m,
                        Some(Err(e)) => {
                            anyhow::bail!("Discord WebSocket error: {}", e);
                        }
                        None => {
                            anyhow::bail!("Discord WebSocket closed unexpectedly");
                        }
                    };

                    if msg.is_close() {
                        tracing::warn!("Discord WebSocket received close frame; reconnecting with RESUME");
                        return Ok(LoopDisposition::Reconnect);
                    }

                    let text = match msg.to_text() {
                        Ok(t) => t,
                        Err(_) => continue,
                    };

                    let payload: Value = match serde_json::from_str(text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let op = payload.get("op").and_then(|v| v.as_u64()).unwrap_or(99);

                    // Track sequence number for both heartbeats and resumes.
                    if let Some(s) = payload.get("s").and_then(|v| v.as_u64()) {
                        if let Some(r) = resume.as_mut() {
                            r.seq = Some(s);
                        }
                    }

                    match op {
                        0 => {
                            // Dispatch
                            let event_type =
                                payload.get("t").and_then(|v| v.as_str()).unwrap_or("");
                            if event_type == "READY" {
                                if let Some(d) = payload.get("d") {
                                    let session_id = d
                                        .get("session_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let resume_gateway_url = d
                                        .get("resume_gateway_url")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    if !session_id.is_empty() && !resume_gateway_url.is_empty() {
                                        let seq = payload
                                            .get("s")
                                            .and_then(|v| v.as_u64());
                                        *resume = Some(ResumeState {
                                            session_id,
                                            resume_gateway_url,
                                            seq,
                                        });
                                        tracing::debug!("Discord READY received; resume state cached");
                                    }
                                }
                            } else if event_type == "RESUMED" {
                                tracing::debug!("Discord RESUMED");
                            } else if event_type == "MESSAGE_CREATE" {
                                if let Some(d) = payload.get("d") {
                                    if let Some((author_id, channel_id, content, _msg_id, is_bot)) =
                                        Self::parse_message_create(d)
                                    {
                                        if is_bot {
                                            continue;
                                        }
                                        if !self.allows_sender(&author_id) {
                                            tracing::debug!(
                                                "Discord: ignoring message from disallowed sender {}",
                                                author_id
                                            );
                                            continue;
                                        }

                                        let now = std::time::SystemTime::now()
                                            .duration_since(std::time::UNIX_EPOCH)
                                            .unwrap_or_default()
                                            .as_secs();

                                        let msg = InboundMessage {
                                            id: uuid::Uuid::new_v4().to_string(),
                                            sender: author_id,
                                            content,
                                            channel: "discord".to_string(),
                                            chat_id: channel_id,
                                            timestamp: now,
                                            reply_to: None,
                                            metadata: HashMap::new(),
                                        };

                                        if tx.send(msg).await.is_err() {
                                            tracing::info!(
                                                "Discord: inbound channel closed, stopping listener"
                                            );
                                            return Ok(LoopDisposition::InboundClosed);
                                        }
                                    }
                                }
                            }
                        }
                        1 => {
                            // Heartbeat request: send heartbeat immediately
                            let seq = resume.as_ref().and_then(|r| r.seq);
                            let hb = serde_json::json!({ "op": 1, "d": seq });
                            let _ = ws_tx.send(WsMessage::Text(hb.to_string().into())).await;
                        }
                        7 => {
                            // Reconnect requested by Discord — connect to
                            // resume_gateway_url and RESUME.
                            tracing::info!("Discord op:7 Reconnect; will RESUME");
                            return Ok(LoopDisposition::Reconnect);
                        }
                        9 => {
                            // Invalid Session: `d` is a boolean indicating
                            // whether the session is resumable.
                            let resumable = payload
                                .get("d")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            if resumable {
                                tracing::info!("Discord op:9 Invalid Session (resumable); will RESUME");
                                return Ok(LoopDisposition::Reconnect);
                            } else {
                                tracing::info!(
                                    "Discord op:9 Invalid Session (not resumable); will IDENTIFY fresh"
                                );
                                return Ok(LoopDisposition::Reidentify);
                            }
                        }
                        11 => {
                            // Heartbeat ACK
                            ack_received = true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

#[async_trait]
impl Channel for DiscordChannel {
    fn name(&self) -> &str {
        "discord"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        // Discord caps per-message content at 2000 chars; split longer
        // messages preserving code-block boundaries.
        let parts = split_message(&message.content, DISCORD_MAX_CONTENT);
        for part in &parts {
            self.post_message(&message.recipient, part).await?;
        }
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<InboundMessage>) -> Result<()> {
        let mut resume: Option<ResumeState> = None;

        loop {
            // Pick connect URL: use resume_gateway_url if we have a session,
            // otherwise fetch a fresh one from /gateway/bot.
            let connect_url = match &resume {
                Some(r) => format!("{}?v=10&encoding=json", r.resume_gateway_url),
                None => self.fetch_gateway_url().await?,
            };

            // Connect WebSocket
            let (ws_stream, _) = tokio_tungstenite::connect_async(&connect_url)
                .await
                .context("Discord WebSocket connect failed")?;
            let (mut ws_tx, mut ws_rx) = ws_stream.split();

            // Receive Hello (op:10) to learn heartbeat_interval
            let hello_msg = ws_rx
                .next()
                .await
                .ok_or_else(|| anyhow::anyhow!("Discord WebSocket closed before Hello"))?
                .context("Discord WebSocket read error")?;
            let hello: Value = serde_json::from_str(
                hello_msg.to_text().context("Discord Hello was not text")?,
            )
            .context("Discord Hello parse failed")?;
            let heartbeat_interval_ms = hello
                .get("d")
                .and_then(|d| d.get("heartbeat_interval"))
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("Discord Hello missing heartbeat_interval"))?;

            // Either RESUME or IDENTIFY
            if let Some(state) = &resume {
                let resume_payload = self.resume_payload(state);
                ws_tx
                    .send(WsMessage::Text(resume_payload.to_string().into()))
                    .await
                    .context("Discord send Resume failed")?;
            } else {
                ws_tx
                    .send(WsMessage::Text(self.identify_payload().to_string().into()))
                    .await
                    .context("Discord send Identify failed")?;
            }

            // Run the per-connection event loop.
            let disposition = self
                .run_event_loop(&mut ws_tx, &mut ws_rx, &mut resume, &tx, heartbeat_interval_ms)
                .await?;

            match disposition {
                LoopDisposition::InboundClosed => return Ok(()),
                LoopDisposition::Reconnect => continue,
                LoopDisposition::Reidentify => {
                    resume = None;
                    continue;
                }
            }
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn send_streaming_start(&self, chat_id: &str) -> Result<Option<String>> {
        let data = self.post_message(chat_id, "...").await?;
        let message_id = data.get("id").and_then(|v| v.as_str()).map(String::from);
        Ok(message_id)
    }

    async fn send_streaming_delta(
        &self,
        chat_id: &str,
        message_id: &str,
        full_text: &str,
    ) -> Result<()> {
        // Empty content → Discord rejects PATCH /messages with 400
        // (code 50006: "cannot send empty message"). Skip until there's
        // something to render. The placeholder from `send_streaming_start`
        // remains visible to the user until the next non-empty delta.
        if full_text.is_empty() {
            return Ok(());
        }

        // Rate-limit: skip if last edit was <300ms ago.
        {
            let map = self.last_edit.lock();
            if let Some(last) = map.get(chat_id) {
                if last.elapsed().as_millis() < 300 {
                    return Ok(());
                }
            }
        }

        // Discord caps per-message content at 2000 chars. During the live
        // preview, truncate (with an ellipsis) so the PATCH stays valid;
        // the final `send_streaming_end` is responsible for spilling
        // overflow into continuation messages.
        let preview = if full_text.len() > DISCORD_MAX_CONTENT {
            let mut cut = DISCORD_MAX_CONTENT.saturating_sub(1);
            // Avoid splitting a UTF-8 codepoint.
            while cut > 0 && !full_text.is_char_boundary(cut) {
                cut -= 1;
            }
            format!("{}…", &full_text[..cut])
        } else {
            full_text.to_string()
        };

        self.patch_message(chat_id, message_id, &preview).await?;

        self.last_edit_insert(chat_id.to_string(), Instant::now());

        Ok(())
    }

    async fn send_streaming_end(
        &self,
        chat_id: &str,
        message_id: &str,
        full_text: &str,
    ) -> Result<()> {
        // Empty final text → PATCH /messages with empty content is a
        // 400 (code 50006). Happens when a turn produced only a tool
        // call and no prose. Leave the start-of-stream placeholder
        // ("...") in place rather than erroring; the user sees an
        // unfilled bubble, which is unusual but not broken.
        if full_text.is_empty() {
            let mut map = self.last_edit.lock();
            map.remove(chat_id);
            return Ok(());
        }

        // Split honoring 2000-char limit and code-block boundaries.
        let parts = split_message(full_text, DISCORD_MAX_CONTENT);
        // `split_message` of a non-empty string always returns ≥1
        // element, so `first` will be Some — but defensively, skip a
        // zero-length first chunk so we don't reintroduce the empty-
        // PATCH crash via an unforeseen splitter edge case.
        if let Some(first) = parts.first() {
            if !first.is_empty() {
                self.patch_message(chat_id, message_id, first).await?;
            }
        }
        for part in parts.iter().skip(1) {
            if !part.is_empty() {
                self.post_message(chat_id, part).await?;
            }
        }

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
                "Discord: refusing message from '{sender_id}' — no allowed_users configured. \
                 Add your user ID to [channels.discord].allowed_users (or \"*\" to allow everyone)."
            );
            return false;
        }
        if self.allowed_users.iter().any(|u| u == "*") {
            return true;
        }
        self.allowed_users.iter().any(|u| u == sender_id)
    }
}
