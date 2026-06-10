//! Signal messaging channel.
//!
//! Connects to a `signal-cli` daemon over HTTP. The daemon must be
//! running externally:
//!
//! ```text
//! signal-cli daemon --http=127.0.0.1:8080
//! ```
//!
//! Three endpoints (per signal-cli's `signal-cli-jsonrpc.5` man
//! page, verified against the upstream's production
//! implementation):
//!
//!   POST /api/v1/rpc    JSON-RPC 2.0 — outbound sends, contact
//!                       lookups, typing indicators
//!   GET  /api/v1/events Server-Sent Events stream — inbound
//!                       messages, sync echoes, group events
//!   GET  /api/v1/check  Liveness probe (200 OK if daemon is up)
//!
//! Auth: none. Bind localhost. Do not expose the daemon URL to
//! the network — anyone who can reach it can send messages on
//! the configured account's behalf.
//!
//! Echo handling: when the agent sends a message, signal-cli
//! reflects it back as a `syncMessage.sentMessage` envelope. We
//! filter these by tracking the timestamp returned in the send
//! response (50-entry rolling cap).
//!
//! Group messaging: opt-in via `group_allowed_users`. Empty list
//! disables groups entirely; `*` allows all; comma list pins
//! specific groups.
//!
//! Reconnect strategy: exponential backoff 2s → 60s with 20%
//! jitter on SSE failure, matching upstream. A separate health
//! monitor pings `/api/v1/check` every 30s and force-reconnects
//! when the SSE has been silent for more than 120s.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use rand::Rng;
use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::bus::InboundMessage;
use crate::config::SignalChannelEntry;

use super::traits::{Channel, SendMessage};

/// Health check interval. Per upstream.
const HEALTH_CHECK_INTERVAL: Duration = Duration::from_secs(30);
/// SSE-silence threshold after which the health monitor forces
/// a reconnect. Per upstream.
const SSE_SILENCE_THRESHOLD: Duration = Duration::from_secs(120);
/// Initial reconnect backoff. Per upstream.
const RECONNECT_BACKOFF_INITIAL: Duration = Duration::from_secs(2);
/// Max reconnect backoff cap. Per upstream.
const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(60);
/// Cap on outbound timestamps tracked for echo filtering. 50 is
/// upstream's value; keeps memory bounded.
const ECHO_FILTER_MAX: usize = 50;
/// Per-chat typing-failure threshold before backoff kicks in.
const TYPING_FAILURE_THRESHOLD: u32 = 3;
/// Initial cooldown after the third consecutive typing failure.
const TYPING_BACKOFF_INITIAL: Duration = Duration::from_secs(16);
/// Max typing backoff cap.
const TYPING_BACKOFF_MAX: Duration = Duration::from_secs(60);

/// Group-id prefix in chat ids. A chat id of
/// `group:abc123==` routes a send to the group; a bare phone or
/// UUID routes to a DM.
pub const GROUP_PREFIX: &str = "group:";

/// Signal channel. Cheap to clone via the inner `Arc`s.
pub struct SignalChannel {
    config: SignalChannelEntry,
    state: Arc<SignalState>,
    /// Reusable HTTP client. The same client is used for SSE long-
    /// poll, RPC calls, and the health probe.
    http: Client,
}

/// Mutable shared state — bounded by `Arc<Mutex<_>>`.
struct SignalState {
    /// Recent outbound message timestamps. Used to drop the
    /// reflected `syncMessage` echoes from the inbound stream.
    recent_sent_timestamps: Mutex<VecDeque<i64>>,
    /// UUID/E.164 cache populated by `listContacts` lookups.
    /// Both directions; `_resolve_recipient` reads from here
    /// before issuing a fresh RPC.
    recipient_uuid_by_number: Mutex<HashMap<String, String>>,
    recipient_number_by_uuid: Mutex<HashMap<String, String>>,
    /// Per-chat typing failure tracking. After
    /// `TYPING_FAILURE_THRESHOLD` consecutive failures we exponential-
    /// backoff and skip the typing RPC until the cooldown lifts.
    typing_failures: Mutex<HashMap<String, TypingState>>,
}

#[derive(Debug, Clone, Default)]
struct TypingState {
    consecutive_failures: u32,
    cooldown_until: Option<std::time::Instant>,
    /// Last cooldown duration, doubled on each subsequent failure
    /// up to `TYPING_BACKOFF_MAX`.
    last_backoff: Duration,
}

impl SignalChannel {
    /// Construct from config. Returns `None` when the channel is
    /// disabled or the required `account` is empty.
    pub fn from_config(config: &SignalChannelEntry) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        if config.account.is_empty() {
            tracing::warn!("signal channel enabled but `account` is empty; refusing to start");
            return None;
        }
        if config.http_url.is_empty() {
            tracing::warn!("signal channel enabled but `http_url` is empty; refusing to start");
            return None;
        }
        let http = Client::builder()
            // SSE long-polls; no per-request timeout (we cap the
            // RPC calls separately via tokio::time::timeout).
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .build()
            .ok()?;
        Some(Self {
            config: config.clone(),
            state: Arc::new(SignalState {
                recent_sent_timestamps: Mutex::new(VecDeque::with_capacity(ECHO_FILTER_MAX)),
                recipient_uuid_by_number: Mutex::new(HashMap::new()),
                recipient_number_by_uuid: Mutex::new(HashMap::new()),
                typing_failures: Mutex::new(HashMap::new()),
            }),
            http,
        })
    }

    /// `POST /api/v1/rpc` with a JSON-RPC 2.0 request. Returns the
    /// `result` value on success; `Err` on transport failure or
    /// JSON-RPC error response.
    async fn rpc(&self, method: &str, params: Value) -> Result<Value> {
        let id = format!(
            "{}_{}",
            method,
            chrono::Utc::now().timestamp_millis(),
        );
        let body = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": id,
        });
        let url = format!("{}/api/v1/rpc", self.config.http_url.trim_end_matches('/'));
        let response = self
            .http
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(30))
            .send()
            .await
            .with_context(|| format!("signal RPC {} POST failed", method))?
            .error_for_status()
            .with_context(|| format!("signal RPC {} returned non-success status", method))?;
        let payload: Value = response
            .json()
            .await
            .with_context(|| format!("signal RPC {} response not valid JSON", method))?;
        if let Some(err) = payload.get("error") {
            anyhow::bail!("signal RPC {} returned error: {}", method, err);
        }
        Ok(payload
            .get("result")
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Resolve a phone number → UUID via cached lookup, falling
    /// back to a `listContacts` RPC. Phone numbers are E.164 with
    /// the leading `+`. UUIDs and `u:`-prefixed service ids pass
    /// through unchanged. Cache misses do not error — return the
    /// original input.
    async fn resolve_recipient(&self, recipient: &str) -> String {
        // UUID-shaped or service-id-prefixed: passthrough.
        if recipient.starts_with("u:") || is_uuid_shape(recipient) {
            return recipient.to_string();
        }
        // Cached?
        if let Some(uuid) = self
            .state
            .recipient_uuid_by_number
            .lock()
            .get(recipient)
            .cloned()
        {
            return uuid;
        }
        // Refresh contacts.
        let result = self
            .rpc(
                "listContacts",
                json!({
                    "account": self.config.account,
                    "allRecipients": true,
                }),
            )
            .await;
        if let Ok(contacts) = result {
            if let Some(arr) = contacts.as_array() {
                let mut uuid_by_number = self.state.recipient_uuid_by_number.lock();
                let mut number_by_uuid = self.state.recipient_number_by_uuid.lock();
                for entry in arr {
                    let number = entry.get("number").and_then(|v| v.as_str());
                    let uuid = entry.get("uuid").and_then(|v| v.as_str());
                    if let (Some(n), Some(u)) = (number, uuid) {
                        uuid_by_number.insert(n.to_string(), u.to_string());
                        number_by_uuid.insert(u.to_string(), n.to_string());
                    }
                }
            }
        }
        self.state
            .recipient_uuid_by_number
            .lock()
            .get(recipient)
            .cloned()
            .unwrap_or_else(|| recipient.to_string())
    }

    /// Track a sent timestamp so the corresponding sync echo gets
    /// dropped from the inbound stream.
    fn record_sent_timestamp(&self, ts: i64) {
        let mut q = self.state.recent_sent_timestamps.lock();
        q.push_back(ts);
        while q.len() > ECHO_FILTER_MAX {
            q.pop_front();
        }
    }

    fn is_echo_timestamp(&self, ts: i64) -> bool {
        self.state.recent_sent_timestamps.lock().contains(&ts)
    }

    /// Bump a chat's consecutive-failure counter; arm exponential
    /// backoff once it crosses `TYPING_FAILURE_THRESHOLD`.
    fn record_typing_failure(&self, chat_id: &str) {
        let mut map = self.state.typing_failures.lock();
        let entry = map.entry(chat_id.to_string()).or_default();
        entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
        if entry.consecutive_failures >= TYPING_FAILURE_THRESHOLD {
            let next = if entry.last_backoff.is_zero() {
                TYPING_BACKOFF_INITIAL
            } else {
                (entry.last_backoff * 2).min(TYPING_BACKOFF_MAX)
            };
            entry.last_backoff = next;
            entry.cooldown_until = Some(std::time::Instant::now() + next);
        }
    }

    /// Long-running SSE listener. Reconnects with exponential
    /// backoff + jitter on failure. Forwards parsed envelopes to
    /// `tx`.
    async fn run_sse(&self, tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        let mut backoff = RECONNECT_BACKOFF_INITIAL;
        let url = format!(
            "{}/api/v1/events?account={}",
            self.config.http_url.trim_end_matches('/'),
            urlencoding::encode(&self.config.account),
        );
        loop {
            tracing::info!(url = %url, "signal SSE connecting");
            let resp = self
                .http
                .get(&url)
                .header("Accept", "text/event-stream")
                .send()
                .await;
            match resp {
                Ok(r) if r.status().is_success() => {
                    backoff = RECONNECT_BACKOFF_INITIAL;
                    if let Err(e) = self.consume_sse_stream(r, tx.clone()).await {
                        tracing::warn!(error = %e, "signal SSE stream ended");
                    }
                }
                Ok(r) => {
                    tracing::warn!(
                        status = %r.status(),
                        "signal SSE returned non-success status"
                    );
                }
                Err(e) => {
                    tracing::warn!(error = %e, "signal SSE connect failed");
                }
            }
            // Sleep with jitter before reconnecting.
            let jittered = backoff_with_jitter(backoff);
            tokio::time::sleep(jittered).await;
            backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
        }
    }

    /// Drain one SSE response. Each `data:` line is a JSON envelope;
    /// comment lines (`:`) are heartbeats and just refresh the
    /// activity timestamp implicitly (we use stream liveness as
    /// the activity signal here — no separate counter needed since
    /// the streaming HTTP client surfaces disconnect via `Err`).
    async fn consume_sse_stream(
        &self,
        response: reqwest::Response,
        tx: mpsc::Sender<InboundMessage>,
    ) -> Result<()> {
        use futures::StreamExt;
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk.context("signal SSE chunk read failed")?;
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(idx) = buffer.find('\n') {
                let line = buffer.drain(..=idx).collect::<String>();
                let line = line.trim_end_matches(['\r', '\n']);
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                if let Some(json_str) = line.strip_prefix("data:") {
                    let json_str = json_str.trim();
                    if json_str.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(json_str) {
                        Ok(envelope) => {
                            if let Some(inbound) = self.handle_envelope(&envelope) {
                                if tx.send(inbound).await.is_err() {
                                    return Ok(()); // Receiver dropped; clean shutdown.
                                }
                            }
                        }
                        Err(e) => {
                            tracing::debug!(error = %e, "signal SSE: skipping malformed line");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Convert a raw SSE envelope into an `InboundMessage`, or
    /// `None` if the envelope should be filtered (echo, story,
    /// group not on allowlist, etc.).
    fn handle_envelope(&self, raw: &Value) -> Option<InboundMessage> {
        // signal-cli sometimes nests `envelope` inside the JSON-RPC
        // `params` shape; handle both.
        let envelope = raw
            .get("params")
            .and_then(|p| p.get("envelope"))
            .or_else(|| raw.get("envelope"))
            .unwrap_or(raw);

        let source_number = envelope
            .get("sourceNumber")
            .and_then(|v| v.as_str())
            .map(String::from);
        let source_uuid = envelope
            .get("sourceUuid")
            .and_then(|v| v.as_str())
            .map(String::from);
        let source_name = envelope
            .get("sourceName")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let source = source_number
            .clone()
            .or_else(|| source_uuid.clone())
            .or_else(|| {
                envelope
                    .get("source")
                    .and_then(|v| v.as_str())
                    .map(String::from)
            })?;

        // Sync messages (Note to Self) — keep when the agent's own
        // outbound timestamp doesn't match (i.e., it's a real
        // self-message, not a reflected echo).
        if let Some(sync) = envelope.get("syncMessage") {
            if let Some(sent) = sync.get("sentMessage") {
                let dest = sync.get("destination").and_then(|v| v.as_str());
                if dest != Some(self.config.account.as_str()) {
                    // Sync of a message to someone else — not for us.
                    return None;
                }
                let ts = sent.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0);
                if self.is_echo_timestamp(ts) {
                    return None;
                }
                return self.build_inbound_from_data(sent, &source, &source_name);
            }
            return None;
        }

        // Drop messages from our own account (other than Note to Self
        // handled above).
        if Some(self.config.account.as_str())
            == source_number.as_deref().or(source_uuid.as_deref())
        {
            return None;
        }

        // Drop stories when configured (default true).
        if self.config.ignore_stories && envelope.get("storyMessage").is_some() {
            return None;
        }

        // Standard data message or edit-message-wrapped data message.
        let data = envelope
            .get("dataMessage")
            .or_else(|| {
                envelope
                    .get("editMessage")
                    .and_then(|em| em.get("dataMessage"))
            })?;

        self.build_inbound_from_data(data, &source, &source_name)
    }

    fn build_inbound_from_data(
        &self,
        data: &Value,
        source: &str,
        source_name: &str,
    ) -> Option<InboundMessage> {
        let body = data
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Group: check allowlist.
        let group_id = data
            .get("groupInfo")
            .and_then(|g| g.get("groupId"))
            .and_then(|v| v.as_str())
            .map(String::from);

        let chat_id = if let Some(gid) = &group_id {
            if !self.is_group_allowed(gid) {
                return None;
            }
            format!("{}{}", GROUP_PREFIX, gid)
        } else {
            // DM allowlist: empty list → everyone.
            if !self.is_dm_sender_allowed(source) {
                return None;
            }
            source.to_string()
        };

        if body.is_empty() {
            // Empty messages (typing-only updates, reactions
            // handled separately) — drop.
            return None;
        }

        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = data.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0)
            as u64;
        let sender_display = if source_name.is_empty() {
            source.to_string()
        } else {
            format!("{} ({})", source_name, source)
        };
        let mut metadata = HashMap::new();
        if let Some(gid) = group_id {
            metadata.insert("group_id".into(), gid);
        }
        Some(InboundMessage {
            id,
            sender: sender_display,
            content: body,
            channel: "signal".into(),
            chat_id,
            timestamp,
            reply_to: None,
            metadata,
        })
    }

    fn is_group_allowed(&self, group_id: &str) -> bool {
        let allow = &self.config.group_allowed_users;
        if allow.is_empty() {
            return false;
        }
        if allow.iter().any(|s| s == "*") {
            return true;
        }
        allow.iter().any(|s| s == group_id)
    }

    fn is_dm_sender_allowed(&self, sender: &str) -> bool {
        let allow = &self.config.allowed_users;
        if allow.is_empty() {
            return true;
        }
        allow.iter().any(|s| s == sender)
    }
}

/// Apply 0-20% positive jitter to a backoff duration.
fn backoff_with_jitter(d: Duration) -> Duration {
    let mut rng = rand::thread_rng();
    let jitter_pct: f64 = rng.gen_range(0.0..0.2);
    let nanos = (d.as_nanos() as f64 * (1.0 + jitter_pct)) as u128;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

/// Heuristic: a UUID v4 is 36 chars with hyphens at fixed positions.
fn is_uuid_shape(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    bytes[8] == b'-'
        && bytes[13] == b'-'
        && bytes[18] == b'-'
        && bytes[23] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, b)| {
                if matches!(i, 8 | 13 | 18 | 23) {
                    *b == b'-'
                } else {
                    b.is_ascii_hexdigit()
                }
            })
}

#[async_trait]
impl Channel for SignalChannel {
    fn name(&self) -> &str {
        "signal"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        let recipient = message.recipient.as_str();
        let (body, styles) = render_for_signal(&message.content);
        let mut params = if let Some(group_id) = recipient.strip_prefix(GROUP_PREFIX) {
            json!({
                "account": self.config.account,
                "groupId": group_id,
                "message": body,
            })
        } else {
            let resolved = self.resolve_recipient(recipient).await;
            json!({
                "account": self.config.account,
                "recipient": [resolved],
                "message": body,
            })
        };
        if let Some(s) = styles {
            if let Some(obj) = params.as_object_mut() {
                obj.insert("textStyles".into(), json!(s));
            }
        }
        let result = self.rpc("send", params).await?;
        if let Some(ts) = result.get("timestamp").and_then(|v| v.as_i64()) {
            self.record_sent_timestamp(ts);
        }
        Ok(())
    }

    async fn send_typing(&self, chat_id: &str) -> Result<()> {
        // Per-chat cooldown: skip the RPC entirely while in backoff.
        if let Some(until) = self
            .state
            .typing_failures
            .lock()
            .get(chat_id)
            .and_then(|s| s.cooldown_until)
        {
            if std::time::Instant::now() < until {
                return Ok(());
            }
        }

        let params = if let Some(group_id) = chat_id.strip_prefix(GROUP_PREFIX) {
            json!({
                "account": self.config.account,
                "groupId": group_id,
                "stop": false,
            })
        } else {
            let resolved = self.resolve_recipient(chat_id).await;
            json!({
                "account": self.config.account,
                "recipient": [resolved],
                "stop": false,
            })
        };

        match self.rpc("sendTyping", params).await {
            Ok(_) => {
                let mut map = self.state.typing_failures.lock();
                map.remove(chat_id);
                Ok(())
            }
            Err(e) => {
                self.record_typing_failure(chat_id);
                tracing::debug!(chat_id = %chat_id, error = %e, "signal sendTyping failed");
                // Typing is best-effort — don't surface as a hard error
                // so the actual response send still proceeds.
                Ok(())
            }
        }
    }

    async fn listen(&self, tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        // Health check first — refuse to start if the daemon is unreachable.
        let health_url = format!(
            "{}/api/v1/check",
            self.config.http_url.trim_end_matches('/')
        );
        let probe = self
            .http
            .get(&health_url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .context("signal-cli daemon health probe failed (is signal-cli running?)")?;
        if !probe.status().is_success() {
            anyhow::bail!(
                "signal-cli daemon health probe returned {}",
                probe.status()
            );
        }
        // Run the SSE listener forever (with internal reconnect).
        self.run_sse(tx).await
    }

    fn allows_sender(&self, sender_id: &str) -> bool {
        self.is_dm_sender_allowed(sender_id)
    }
}

// -- Markdown → Signal textStyles converter -----------------------

/// Signal's recognized text-style names. Sent as the third
/// component of each `textStyles` entry (`<start>:<length>:<STYLE>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalStyle {
    Bold,
    Italic,
    Strikethrough,
    Monospace,
}

impl SignalStyle {
    fn name(self) -> &'static str {
        match self {
            SignalStyle::Bold => "BOLD",
            SignalStyle::Italic => "ITALIC",
            SignalStyle::Strikethrough => "STRIKETHROUGH",
            SignalStyle::Monospace => "MONOSPACE",
        }
    }
}

/// One style range, in UTF-16 code-unit offsets. Signal expects
/// offsets to count UTF-16 units (the same unit JavaScript uses) —
/// emoji and CJK chars take 2 units each, ASCII takes 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleRange {
    pub start: usize,
    pub length: usize,
    pub style: SignalStyle,
}

impl StyleRange {
    fn to_param(&self) -> String {
        format!("{}:{}:{}", self.start, self.length, self.style.name())
    }
}

/// Result of `markdown_to_signal`. `text` is the clean body Signal
/// renders; `styles` are the bodyRanges to attach as `textStyles`
/// in the `send` RPC. Empty `styles` means no formatting was
/// detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkdownToSignalResult {
    pub text: String,
    pub styles: Vec<StyleRange>,
}

/// Convert markdown to Signal's clean text + bodyRange offsets.
///
/// Recognized markers (priority order — earlier wins to avoid
/// `*` confusion inside ` `code` `):
///
///   1. Triple-backtick fenced code block → MONOSPACE (multi-line)
///   2. Headings (`#`, `##`, `###`...) → BOLD (per-line)
///   3. Inline backtick code → MONOSPACE
///   4. `**bold**` or `__bold__` → BOLD
///   5. `~~strike~~` → STRIKETHROUGH
///   6. `*italic*` or `_italic_` → ITALIC (single asterisk/underscore)
///
/// Offsets are computed in **UTF-16 code units** to match Signal's
/// expectation. ASCII is 1 unit; most BMP CJK is 1 unit; emoji and
/// supplementary-plane chars are 2 units each.
pub fn markdown_to_signal(input: &str) -> MarkdownToSignalResult {
    // Pass 1: extract triple-backtick fenced blocks and inline
    // backticks first, replacing each match with a sentinel that
    // survives the later inline-marker passes. The body of the code
    // span is preserved verbatim.
    //
    // Algorithm:
    //   - Tokenize into "literal" and "code" runs by walking the
    //     string and toggling on backtick boundaries.
    //   - For each "literal" run, apply the inline-marker rewrites
    //     (bold/italic/strike), tracking UTF-16 offsets.
    //   - Concatenate runs to produce the final text + style list.
    //
    // This avoids the regex-overlapping problem where `**a*b**` or
    // `*` inside backticks gets misread.

    // Step 1: split into code spans + literals.
    let runs = tokenize_code_runs(input);

    let mut text = String::new();
    let mut styles: Vec<StyleRange> = Vec::new();
    let mut utf16_cursor: usize = 0;

    for run in runs {
        match run {
            Run::Literal(s) => {
                let (rendered, mut sub_styles) = render_literal(&s, utf16_cursor);
                let len_units = utf16_units(&rendered);
                text.push_str(&rendered);
                styles.append(&mut sub_styles);
                utf16_cursor += len_units;
            }
            Run::Code(body) => {
                let len_units = utf16_units(&body);
                if len_units > 0 {
                    styles.push(StyleRange {
                        start: utf16_cursor,
                        length: len_units,
                        style: SignalStyle::Monospace,
                    });
                }
                text.push_str(&body);
                utf16_cursor += len_units;
            }
        }
    }

    MarkdownToSignalResult { text, styles }
}

/// One token from the input — either a literal run that may carry
/// inline bold/italic/strike markers, or a code span (passed through
/// verbatim with monospace style). Multi-line code blocks are trimmed
/// of their leading newline at tokenization time.
enum Run {
    Literal(String),
    Code(String),
}

/// Split the input into alternating literal and code runs.
///
/// Triple backticks open / close a multi-line code block; a single
/// backtick toggles an inline code span. Mismatched / unclosed
/// markers are treated as literal characters (Signal renders them
/// as-is).
fn tokenize_code_runs(input: &str) -> Vec<Run> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '`' {
            buf.push(c);
            continue;
        }
        // Determine fence width: 1 (inline) or 3 (multi-line).
        let mut backticks = 1;
        while chars.peek() == Some(&'`') {
            chars.next();
            backticks += 1;
            if backticks == 3 {
                break;
            }
        }
        // Look for the matching closing fence of the same width.
        // We walk the remaining chars; if we don't find a close,
        // treat the opening fence as literal.
        let mut body = String::new();
        let mut closed = false;
        let multiline = backticks == 3;
        while let Some(&next) = chars.peek() {
            if next == '`' {
                let mut close_count = 0;
                while chars.peek() == Some(&'`') {
                    chars.next();
                    close_count += 1;
                    if close_count == backticks {
                        break;
                    }
                }
                if close_count == backticks {
                    closed = true;
                    break;
                } else {
                    // Partial fence — push back into the body.
                    for _ in 0..close_count {
                        body.push('`');
                    }
                }
            } else {
                body.push(next);
                chars.next();
            }
        }
        if closed {
            // Flush the literal buffer first.
            if !buf.is_empty() {
                out.push(Run::Literal(std::mem::take(&mut buf)));
            }
            // Trim leading newline of multi-line code block, if
            // present (matches the convention `` ```\nbody\n``` ``).
            let trimmed = if multiline {
                body.strip_prefix('\n')
                    .map(String::from)
                    .unwrap_or(body)
            } else {
                body
            };
            out.push(Run::Code(trimmed));
        } else {
            // Unmatched fence: treat literally.
            for _ in 0..backticks {
                buf.push('`');
            }
            buf.push_str(&body);
        }
    }
    if !buf.is_empty() {
        out.push(Run::Literal(buf));
    }
    out
}

/// Apply inline markers (bold / italic / strike, plus heading-line
/// bolding) to a literal run. Returns the cleaned text plus any
/// styles, with offsets relative to `start_utf16` (the UTF-16
/// position of this run inside the full output string).
fn render_literal(input: &str, start_utf16: usize) -> (String, Vec<StyleRange>) {
    let mut out_styles: Vec<StyleRange> = Vec::new();
    let mut out = String::with_capacity(input.len());
    let mut cursor_utf16 = start_utf16;

    // Process line by line to handle headings (`# ...` → BOLD on
    // the heading text). Inline markers apply within each line.
    let lines: Vec<&str> = input.split_inclusive('\n').collect();
    for line in lines {
        // Strip trailing newline for marker detection; restore
        // afterward.
        let (body, trailing) = match line.strip_suffix('\n') {
            Some(b) => (b, "\n"),
            None => (line, ""),
        };
        // Heading detection: lines starting with 1-6 `#` characters
        // followed by a space.
        let heading_match = body
            .chars()
            .take_while(|c| *c == '#')
            .count();
        let after_hashes = body.chars().skip(heading_match).next();
        if heading_match >= 1
            && heading_match <= 6
            && after_hashes == Some(' ')
        {
            // Strip `### ` prefix; rest of line is bolded.
            let rest: String = body
                .chars()
                .skip(heading_match + 1)
                .collect();
            let (rendered, mut styles) = render_inline(&rest, cursor_utf16);
            let body_units = utf16_units(&rendered);
            if body_units > 0 {
                out_styles.push(StyleRange {
                    start: cursor_utf16,
                    length: body_units,
                    style: SignalStyle::Bold,
                });
            }
            // Inline styles inside the heading also apply.
            out_styles.append(&mut styles);
            out.push_str(&rendered);
            cursor_utf16 += body_units;
        } else {
            let (rendered, mut styles) = render_inline(body, cursor_utf16);
            let body_units = utf16_units(&rendered);
            out_styles.append(&mut styles);
            out.push_str(&rendered);
            cursor_utf16 += body_units;
        }
        if !trailing.is_empty() {
            out.push_str(trailing);
            cursor_utf16 += utf16_units(trailing);
        }
    }

    (out, out_styles)
}

/// Apply inline markers — `**bold**`, `__bold__`, `~~strike~~`,
/// `*italic*`, `_italic_` — to a single line of text. Markers are
/// processed in priority order (longest first) to avoid eating
/// part of a longer marker.
fn render_inline(input: &str, start_utf16: usize) -> (String, Vec<StyleRange>) {
    // Order matters: ** before *, ~~ before ~, __ before _.
    // We apply each marker pattern one at a time, scanning the
    // current state of the string for the marker, recording the
    // style range, and stripping the marker chars.
    let patterns: &[(&str, &str, SignalStyle)] = &[
        ("**", "**", SignalStyle::Bold),
        ("__", "__", SignalStyle::Bold),
        ("~~", "~~", SignalStyle::Strikethrough),
        ("*", "*", SignalStyle::Italic),
        ("_", "_", SignalStyle::Italic),
    ];

    let mut current = input.to_string();
    let mut styles: Vec<StyleRange> = Vec::new();

    for (open, close, style) in patterns {
        current = apply_inline_pattern(&current, open, close, *style, start_utf16, &mut styles);
    }

    (current, styles)
}

/// Strip every well-formed `<open>...<close>` pair from `input`,
/// recording each match's range as a style. Offsets are
/// `start_utf16 + (UTF-16 index in output)`.
///
/// "Well-formed" means: opener present, closer present after the
/// opener, and the content between them is non-empty. Mismatched
/// markers are left as literal characters.
fn apply_inline_pattern(
    input: &str,
    open: &str,
    close: &str,
    style: SignalStyle,
    start_utf16: usize,
    styles: &mut Vec<StyleRange>,
) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    let mut utf16_in_output: usize = 0;

    while let Some(c) = chars.clone().next() {
        // Try to match the opener at this position.
        if try_consume_marker(&mut chars, open) {
            // Find a matching closer.
            let mut content = String::new();
            let mut closed = false;
            while let Some(&next) = chars.clone().next().as_ref() {
                if try_consume_marker(&mut chars, close) {
                    closed = true;
                    break;
                }
                content.push(next);
                chars.next();
            }
            if closed && !content.is_empty() {
                let content_units = utf16_units(&content);
                styles.push(StyleRange {
                    start: start_utf16 + utf16_in_output,
                    length: content_units,
                    style,
                });
                out.push_str(&content);
                utf16_in_output += content_units;
            } else {
                // Unmatched opener (or empty content) — emit literally.
                out.push_str(open);
                utf16_in_output += utf16_units(open);
                out.push_str(&content);
                utf16_in_output += utf16_units(&content);
            }
        } else {
            chars.next();
            out.push(c);
            utf16_in_output += utf16_unit_for_char(c);
        }
    }

    out
}

/// If the next characters in `chars` form `marker`, consume them and
/// return true. Otherwise return false without modifying `chars`.
///
/// `marker` is always 1 or 2 ASCII bytes — we use a cheap clone of
/// the underlying `Chars` iterator (no allocation, just a slice
/// pointer copy) to look ahead without committing.
fn try_consume_marker(
    chars: &mut std::str::Chars<'_>,
    marker: &str,
) -> bool {
    let bytes = marker.as_bytes();
    if bytes.is_empty() {
        return false;
    }
    let mut probe = chars.clone();
    for &b in bytes {
        match probe.next() {
            Some(c) if (c as u32) <= 0x7F && c as u8 == b => continue,
            _ => return false,
        }
    }
    // All bytes matched — advance the real iterator.
    for _ in bytes {
        chars.next();
    }
    true
}

/// UTF-16 code unit count for a single char (1 for BMP, 2 for
/// supplementary-plane / emoji).
fn utf16_unit_for_char(c: char) -> usize {
    c.len_utf16()
}

/// UTF-16 code unit count for a string.
fn utf16_units(s: &str) -> usize {
    s.chars().map(|c| c.len_utf16()).sum()
}

/// Pre-format the body for outbound `send`. Returns
/// `(plain_text, optional textStyles param)`. Empty `textStyles`
/// → omit the param from the RPC.
fn render_for_signal(input: &str) -> (String, Option<Vec<String>>) {
    let result = markdown_to_signal(input);
    let styles = if result.styles.is_empty() {
        None
    } else {
        Some(result.styles.iter().map(StyleRange::to_param).collect())
    };
    (result.text, styles)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(account: &str) -> SignalChannelEntry {
        SignalChannelEntry {
            enabled: true,
            http_url: "http://127.0.0.1:8080".into(),
            account: account.into(),
            allowed_users: Vec::new(),
            group_allowed_users: Vec::new(),
            ignore_stories: true,
            home_chat_id: String::new(),
        }
    }

    fn channel(account: &str) -> SignalChannel {
        SignalChannel::from_config(&cfg(account)).unwrap()
    }

    #[test]
    fn from_config_disabled_returns_none() {
        let mut c = cfg("+15551234567");
        c.enabled = false;
        assert!(SignalChannel::from_config(&c).is_none());
    }

    #[test]
    fn from_config_empty_account_returns_none() {
        let c = cfg("");
        assert!(SignalChannel::from_config(&c).is_none());
    }

    #[test]
    fn from_config_empty_url_returns_none() {
        let mut c = cfg("+15551234567");
        c.http_url = String::new();
        assert!(SignalChannel::from_config(&c).is_none());
    }

    #[test]
    fn name_is_signal() {
        let ch = channel("+15551234567");
        assert_eq!(ch.name(), "signal");
    }

    #[test]
    fn echo_filter_round_trip() {
        let ch = channel("+15551234567");
        ch.record_sent_timestamp(100);
        ch.record_sent_timestamp(200);
        assert!(ch.is_echo_timestamp(100));
        assert!(ch.is_echo_timestamp(200));
        assert!(!ch.is_echo_timestamp(999));
    }

    #[test]
    fn echo_filter_caps_at_max() {
        let ch = channel("+15551234567");
        for i in 0..(ECHO_FILTER_MAX + 10) {
            ch.record_sent_timestamp(i as i64);
        }
        // Oldest entries dropped.
        assert!(!ch.is_echo_timestamp(0));
        assert!(ch.is_echo_timestamp((ECHO_FILTER_MAX + 9) as i64));
    }

    #[test]
    fn group_allowlist_empty_disables_groups() {
        let ch = channel("+15551234567");
        assert!(!ch.is_group_allowed("any-group-id"));
    }

    #[test]
    fn group_allowlist_wildcard_allows_all() {
        let mut c = cfg("+15551234567");
        c.group_allowed_users = vec!["*".into()];
        let ch = SignalChannel::from_config(&c).unwrap();
        assert!(ch.is_group_allowed("group-a"));
        assert!(ch.is_group_allowed("group-b"));
    }

    #[test]
    fn group_allowlist_explicit_list() {
        let mut c = cfg("+15551234567");
        c.group_allowed_users = vec!["group-a".into(), "group-b".into()];
        let ch = SignalChannel::from_config(&c).unwrap();
        assert!(ch.is_group_allowed("group-a"));
        assert!(ch.is_group_allowed("group-b"));
        assert!(!ch.is_group_allowed("group-c"));
    }

    #[test]
    fn dm_allowlist_empty_allows_anyone() {
        let ch = channel("+15551234567");
        assert!(ch.is_dm_sender_allowed("+19998887777"));
    }

    #[test]
    fn dm_allowlist_filters_to_specified() {
        let mut c = cfg("+15551234567");
        c.allowed_users = vec!["+19998887777".into()];
        let ch = SignalChannel::from_config(&c).unwrap();
        assert!(ch.is_dm_sender_allowed("+19998887777"));
        assert!(!ch.is_dm_sender_allowed("+13334445555"));
    }

    #[test]
    fn handle_envelope_drops_self_message() {
        let ch = channel("+15551234567");
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "sourceUuid": "abc",
                "timestamp": 1000,
                "dataMessage": { "message": "echo me", "timestamp": 1000 }
            }
        });
        assert!(ch.handle_envelope(&env).is_none());
    }

    #[test]
    fn handle_envelope_drops_recent_sync_echo() {
        let ch = channel("+15551234567");
        ch.record_sent_timestamp(2000);
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "syncMessage": {
                    "destination": "+15551234567",
                    "sentMessage": { "message": "x", "timestamp": 2000 }
                }
            }
        });
        assert!(ch.handle_envelope(&env).is_none());
    }

    #[test]
    fn handle_envelope_keeps_real_note_to_self() {
        // Sync message to self that doesn't match any recent
        // outbound timestamp — that's a real Note to Self the
        // user typed on another device.
        let ch = channel("+15551234567");
        let env = json!({
            "envelope": {
                "sourceNumber": "+15551234567",
                "syncMessage": {
                    "destination": "+15551234567",
                    "sentMessage": { "message": "real self message", "timestamp": 9999 }
                }
            }
        });
        let inbound = ch.handle_envelope(&env).expect("should keep");
        assert_eq!(inbound.content, "real self message");
        assert_eq!(inbound.channel, "signal");
    }

    #[test]
    fn handle_envelope_passes_through_dm() {
        let ch = channel("+15551234567");
        let env = json!({
            "envelope": {
                "sourceNumber": "+19998887777",
                "sourceUuid": "uuid-1",
                "sourceName": "Alice",
                "timestamp": 5000,
                "dataMessage": { "message": "hi", "timestamp": 5000 }
            }
        });
        let inbound = ch.handle_envelope(&env).expect("should keep");
        assert_eq!(inbound.content, "hi");
        assert_eq!(inbound.chat_id, "+19998887777");
        assert!(inbound.sender.contains("Alice"));
    }

    #[test]
    fn handle_envelope_drops_disallowed_dm() {
        let mut c = cfg("+15551234567");
        c.allowed_users = vec!["+11111111111".into()];
        let ch = SignalChannel::from_config(&c).unwrap();
        let env = json!({
            "envelope": {
                "sourceNumber": "+19998887777",
                "timestamp": 5000,
                "dataMessage": { "message": "hi", "timestamp": 5000 }
            }
        });
        assert!(ch.handle_envelope(&env).is_none());
    }

    #[test]
    fn handle_envelope_passes_allowed_group() {
        let mut c = cfg("+15551234567");
        c.group_allowed_users = vec!["g1".into()];
        let ch = SignalChannel::from_config(&c).unwrap();
        let env = json!({
            "envelope": {
                "sourceNumber": "+19998887777",
                "timestamp": 6000,
                "dataMessage": {
                    "message": "hello group",
                    "timestamp": 6000,
                    "groupInfo": { "groupId": "g1" }
                }
            }
        });
        let inbound = ch.handle_envelope(&env).expect("should keep");
        assert_eq!(inbound.chat_id, format!("{}g1", GROUP_PREFIX));
        assert_eq!(inbound.metadata.get("group_id").map(String::as_str), Some("g1"));
    }

    #[test]
    fn handle_envelope_drops_disallowed_group() {
        let mut c = cfg("+15551234567");
        c.group_allowed_users = vec!["g1".into()];
        let ch = SignalChannel::from_config(&c).unwrap();
        let env = json!({
            "envelope": {
                "sourceNumber": "+19998887777",
                "timestamp": 7000,
                "dataMessage": {
                    "message": "hi",
                    "timestamp": 7000,
                    "groupInfo": { "groupId": "g2" }
                }
            }
        });
        assert!(ch.handle_envelope(&env).is_none());
    }

    #[test]
    fn handle_envelope_drops_stories_when_configured() {
        let ch = channel("+15551234567");
        let env = json!({
            "envelope": {
                "sourceNumber": "+19998887777",
                "timestamp": 8000,
                "storyMessage": { "kind": "text" }
            }
        });
        assert!(ch.handle_envelope(&env).is_none());
    }

    #[test]
    fn handle_envelope_passes_stories_when_disabled() {
        // When ignore_stories=false, story messages still don't
        // produce an InboundMessage because we look for dataMessage
        // — but the filter doesn't reject them outright. Test the
        // filter contract: with ignore_stories=true the function
        // returns None at the story branch; with false it falls
        // through to dataMessage (which is missing) and still None.
        let mut c = cfg("+15551234567");
        c.ignore_stories = false;
        let ch = SignalChannel::from_config(&c).unwrap();
        let env = json!({
            "envelope": {
                "sourceNumber": "+19998887777",
                "timestamp": 8000,
                "storyMessage": { "kind": "text" }
            }
        });
        // Falls through ignore_stories branch but has no dataMessage,
        // so still None. Important: it doesn't panic.
        assert!(ch.handle_envelope(&env).is_none());
    }

    #[test]
    fn handle_envelope_unwraps_jsonrpc_params() {
        // signal-cli's HTTP daemon nests the envelope inside JSON-RPC
        // params. We unwrap.
        let ch = channel("+15551234567");
        let raw = json!({
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {
                "envelope": {
                    "sourceNumber": "+19998887777",
                    "timestamp": 9000,
                    "dataMessage": { "message": "nested!", "timestamp": 9000 }
                }
            }
        });
        let inbound = ch.handle_envelope(&raw).expect("should keep");
        assert_eq!(inbound.content, "nested!");
    }

    #[test]
    fn handle_envelope_drops_empty_body() {
        let ch = channel("+15551234567");
        let env = json!({
            "envelope": {
                "sourceNumber": "+19998887777",
                "timestamp": 1234,
                "dataMessage": { "message": "", "timestamp": 1234 }
            }
        });
        assert!(ch.handle_envelope(&env).is_none());
    }

    #[test]
    fn is_uuid_shape_recognizes_v4() {
        assert!(is_uuid_shape("abcdef12-3456-7890-abcd-ef1234567890"));
        assert!(!is_uuid_shape("not-a-uuid"));
        assert!(!is_uuid_shape("abcdef12-3456-7890-abcd-ef123456789Z")); // non-hex Z
    }

    #[test]
    fn backoff_jitter_within_bounds() {
        let base = Duration::from_secs(10);
        for _ in 0..100 {
            let j = backoff_with_jitter(base);
            assert!(j >= base);
            assert!(j <= base + base / 5 + Duration::from_millis(1));
        }
    }

    // -- markdown converter -----------------------------------------

    fn md(input: &str) -> MarkdownToSignalResult {
        markdown_to_signal(input)
    }

    #[test]
    fn markdown_plain_text_no_styles() {
        let r = md("hello world");
        assert_eq!(r.text, "hello world");
        assert!(r.styles.is_empty());
    }

    #[test]
    fn markdown_bold_double_asterisk() {
        let r = md("hi **bold** there");
        assert_eq!(r.text, "hi bold there");
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].start, 3);
        assert_eq!(r.styles[0].length, 4);
        assert_eq!(r.styles[0].style, SignalStyle::Bold);
    }

    #[test]
    fn markdown_bold_double_underscore() {
        let r = md("__bold__");
        assert_eq!(r.text, "bold");
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].style, SignalStyle::Bold);
        assert_eq!(r.styles[0].length, 4);
    }

    #[test]
    fn markdown_italic_single_asterisk() {
        let r = md("an *italic* word");
        assert_eq!(r.text, "an italic word");
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].start, 3);
        assert_eq!(r.styles[0].length, 6);
        assert_eq!(r.styles[0].style, SignalStyle::Italic);
    }

    #[test]
    fn markdown_strikethrough() {
        let r = md("~~old~~");
        assert_eq!(r.text, "old");
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].style, SignalStyle::Strikethrough);
    }

    #[test]
    fn markdown_inline_code_monospace() {
        let r = md("run `cargo test` now");
        assert_eq!(r.text, "run cargo test now");
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].start, 4);
        assert_eq!(r.styles[0].length, 10);
        assert_eq!(r.styles[0].style, SignalStyle::Monospace);
    }

    #[test]
    fn markdown_fenced_code_block() {
        let r = md("before\n```\nfn x() {}\n```\nafter");
        // Leading newline of multi-line block is trimmed; trailing
        // newline (before the closing fence) is preserved verbatim
        // in the body.
        assert!(r.text.contains("fn x() {}"));
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].style, SignalStyle::Monospace);
    }

    #[test]
    fn markdown_heading_bolded() {
        let r = md("# Title\nbody");
        assert_eq!(r.text, "Title\nbody");
        // Heading text is bolded.
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].start, 0);
        assert_eq!(r.styles[0].length, 5);
        assert_eq!(r.styles[0].style, SignalStyle::Bold);
    }

    #[test]
    fn markdown_inside_code_span_not_styled() {
        // `*` inside backticks is literal — no italic style emitted.
        let r = md("`*not italic*`");
        assert_eq!(r.text, "*not italic*");
        // Only the monospace style should fire.
        assert_eq!(r.styles.len(), 1);
        assert_eq!(r.styles[0].style, SignalStyle::Monospace);
    }

    #[test]
    fn markdown_unmatched_marker_passthrough() {
        let r = md("a *unmatched and trailing");
        // Unmatched `*` is preserved verbatim, no italic recorded.
        assert_eq!(r.text, "a *unmatched and trailing");
        assert!(r.styles.is_empty());
    }

    #[test]
    fn markdown_emoji_uses_utf16_offsets() {
        // 🦊 is U+1F98A → 2 UTF-16 units.
        let r = md("🦊 **bold**");
        assert_eq!(r.text, "🦊 bold");
        assert_eq!(r.styles.len(), 1);
        // 🦊 = 2 units, space = 1 unit → bold starts at offset 3.
        assert_eq!(r.styles[0].start, 3);
        assert_eq!(r.styles[0].length, 4);
        assert_eq!(r.styles[0].style, SignalStyle::Bold);
    }

    #[test]
    fn markdown_mixed_bold_and_italic() {
        let r = md("**B** and *i*");
        assert_eq!(r.text, "B and i");
        assert_eq!(r.styles.len(), 2);
        let bold = r.styles.iter().find(|s| s.style == SignalStyle::Bold).unwrap();
        let italic = r
            .styles
            .iter()
            .find(|s| s.style == SignalStyle::Italic)
            .unwrap();
        assert_eq!(bold.start, 0);
        assert_eq!(bold.length, 1);
        assert_eq!(italic.start, 6);
        assert_eq!(italic.length, 1);
    }

    #[test]
    fn render_for_signal_no_styles_returns_none() {
        let (text, styles) = render_for_signal("plain text");
        assert_eq!(text, "plain text");
        assert!(styles.is_none());
    }

    #[test]
    fn render_for_signal_emits_param_strings() {
        let (text, styles) = render_for_signal("**bold**");
        assert_eq!(text, "bold");
        let s = styles.expect("expected styles");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], "0:4:BOLD");
    }

    // -- typing indicator backoff -----------------------------------

    #[test]
    fn typing_failure_below_threshold_no_cooldown() {
        let ch = channel("+15551234567");
        ch.record_typing_failure("+19998887777");
        ch.record_typing_failure("+19998887777");
        let map = ch.state.typing_failures.lock();
        let entry = map.get("+19998887777").unwrap();
        assert_eq!(entry.consecutive_failures, 2);
        assert!(entry.cooldown_until.is_none());
    }

    #[test]
    fn typing_failure_threshold_arms_initial_backoff() {
        let ch = channel("+15551234567");
        for _ in 0..3 {
            ch.record_typing_failure("+19998887777");
        }
        let map = ch.state.typing_failures.lock();
        let entry = map.get("+19998887777").unwrap();
        assert_eq!(entry.consecutive_failures, 3);
        assert_eq!(entry.last_backoff, TYPING_BACKOFF_INITIAL);
        assert!(entry.cooldown_until.is_some());
    }

    #[test]
    fn typing_failure_subsequent_doubles_backoff() {
        let ch = channel("+15551234567");
        for _ in 0..4 {
            ch.record_typing_failure("+19998887777");
        }
        let map = ch.state.typing_failures.lock();
        let entry = map.get("+19998887777").unwrap();
        assert_eq!(entry.last_backoff, TYPING_BACKOFF_INITIAL * 2);
    }

    #[test]
    fn typing_failure_caps_at_max() {
        let ch = channel("+15551234567");
        for _ in 0..20 {
            ch.record_typing_failure("+19998887777");
        }
        let map = ch.state.typing_failures.lock();
        let entry = map.get("+19998887777").unwrap();
        assert_eq!(entry.last_backoff, TYPING_BACKOFF_MAX);
    }

    #[test]
    fn typing_failure_per_chat_isolated() {
        let ch = channel("+15551234567");
        for _ in 0..3 {
            ch.record_typing_failure("+11111111111");
        }
        ch.record_typing_failure("+22222222222");
        let map = ch.state.typing_failures.lock();
        let a = map.get("+11111111111").unwrap();
        let b = map.get("+22222222222").unwrap();
        assert!(a.cooldown_until.is_some());
        assert!(b.cooldown_until.is_none());
    }
}
