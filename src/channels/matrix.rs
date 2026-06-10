//! Matrix messaging channel.
//!
//! Talks the Matrix Client-Server API (v3) directly over HTTPS. No
//! SDK dependency — `reqwest` + `serde_json` is enough since the
//! protocol is plain JSON over REST. E2EE is **not** in scope for
//! this implementation; encrypted rooms are joinable and visible
//! but messages in them are sent as plaintext (encrypted-room peers
//! will see "could not decrypt"). E2EE lands in a follow-up PR
//! behind a Cargo feature, which can layer on top of this code
//! without modifying any of the types here.
//!
//! Endpoints used (per Matrix spec v1.13):
//!
//!   POST  /_matrix/client/v3/login                         — exchange password for token
//!   GET   /_matrix/client/v3/account/whoami                — verify token
//!   GET   /_matrix/client/v3/sync                          — long-poll
//!   POST  /_matrix/client/v3/join/{roomIdOrAlias}          — auto-accept invites
//!   PUT   /_matrix/client/v3/rooms/{roomId}/send/m.room.message/{txnId}
//!   PUT   /_matrix/client/v3/rooms/{roomId}/typing/{userId}
//!
//! Auth: Bearer token in the `Authorization` header. The token is
//! either provided via config (`access_token`) or obtained at startup
//! by exchanging `user_id` + `password`.
//!
//! Sync loop: 30-second long-poll with a 45-second outer
//! `tokio::time::timeout` guard (in case TCP hangs), exponential
//! backoff 2s → 60s + 0–20% jitter on transport failures, immediate
//! abort on permanent auth errors (`M_UNKNOWN_TOKEN`, 401, 403).
//! Event dedup via a 1000-entry rolling cache keyed by `event_id`.
//!
//! Outbound: per-room text via `m.room.message` with `m.text`. Long
//! bodies are sliced at 4000 characters and sent as separate events
//! (matches what Matrix clients themselves do — exceeding the
//! ~64KiB event-size limit is not the concern, but client/server
//! truncation behavior at ~4KiB is). When `markdown_to_html` is on,
//! the agent's output is rendered to `formatted_body` HTML in the
//! same event.
//!
//! Threading: `auto_thread` (default) wraps replies in a thread
//! anchored on the inbound mention. The inbound `metadata` carries
//! `thread_id` when an event already belongs to a thread; the
//! outbound side honors it. DM-specific gating
//! (`dm_auto_thread`, `dm_mention_threads`) refines this for direct
//! messages.

use std::collections::{HashMap, HashSet, VecDeque};
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
use crate::config::MatrixChannelEntry;

use super::traits::{Channel, SendMessage};

// -- Constants ---------------------------------------------------

/// Long-poll timeout passed to `/sync`. The server holds the
/// connection open up to this long waiting for events.
const SYNC_LONGPOLL_MS: u64 = 30_000;
/// Outer timeout wrapping the entire sync request, defending
/// against TCP-level hangs the long-poll itself can't catch.
const SYNC_OUTER_TIMEOUT: Duration = Duration::from_secs(45);
/// Initial backoff after a sync failure.
const RECONNECT_BACKOFF_INITIAL: Duration = Duration::from_secs(2);
/// Cap on sync-failure backoff.
const RECONNECT_BACKOFF_MAX: Duration = Duration::from_secs(60);
/// Cap on the inbound event-id dedup cache. Trimmed on insert.
const DEDUP_CAP: usize = 1000;
/// Max plain-text body length per outbound message. Practical
/// client-side split point; longer bodies are chunked.
const MAX_MESSAGE_LENGTH: usize = 4000;
/// Typing-indicator lifetime. The server stops showing the
/// indicator after this without a fresh request. 30s matches what
/// the canonical Matrix clients send.
const TYPING_TIMEOUT_MS: u64 = 30_000;
/// Per-chat typing-failure threshold before backoff arms.
const TYPING_FAILURE_THRESHOLD: u32 = 3;
const TYPING_BACKOFF_INITIAL: Duration = Duration::from_secs(16);
const TYPING_BACKOFF_MAX: Duration = Duration::from_secs(60);
/// Grace window after startup. Events older than this (relative
/// to channel-start time) are dropped to avoid replaying history
/// the homeserver returns on initial sync.
const STARTUP_GRACE_MS: u64 = 5_000;
/// HTTP per-request timeout for non-sync calls.
const RPC_TIMEOUT: Duration = Duration::from_secs(30);
/// Max attempts for an inbound media download before giving up.
const MEDIA_DOWNLOAD_ATTEMPTS: u32 = 3;
/// Character threshold above which a buffered message is treated
/// as "near the chunk limit" — its presence in a batching window
/// doubles the flush delay since a continuation is likely coming.
const BATCH_LONG_THRESHOLD: usize = 3900;

// -- Types -------------------------------------------------------

/// Matrix channel handle. Cheap to clone via `Arc`s inside.
#[derive(Clone)]
pub struct MatrixChannel {
    config: MatrixChannelEntry,
    state: Arc<MatrixState>,
    http: Client,
}

/// Mutable shared state.
struct MatrixState {
    /// Resolved access token (config-supplied or login-fetched).
    access_token: Mutex<String>,
    /// Resolved user id (config-supplied or whoami-fetched).
    user_id: Mutex<String>,
    /// Last sync token; persists across reconnects within a
    /// single channel lifetime so we don't replay events the
    /// homeserver already showed us.
    next_batch: Mutex<Option<String>>,
    /// Event-id dedup. `seen` is the order-preserving set,
    /// `order` keeps insertion order so we can evict the oldest.
    seen_events: Mutex<DedupCache>,
    /// Cached "is DM" answers per room id.
    dm_rooms: Mutex<HashSet<String>>,
    /// Joined-rooms snapshot from the last sync.
    joined_rooms: Mutex<HashSet<String>>,
    /// Threads we ourselves opened or participated in. Used to
    /// bypass the mention-required gate when a follow-up message
    /// lands in a thread we're already in.
    bot_threads: Mutex<HashSet<String>>,
    /// Per-chat typing-failure tracking with exponential backoff.
    typing_failures: Mutex<HashMap<String, TypingState>>,
    /// Channel start time (ms since epoch) for the startup grace.
    started_at_ms: u64,
    /// Most recent event id this channel sent, per chat. Used by
    /// approval-prompt flows that want to react to "the message we
    /// just sent" without threading the event id through the
    /// caller. Capped implicitly by HashMap key set size (one
    /// entry per active chat).
    last_sent_by_chat: Mutex<HashMap<String, String>>,
    /// Per-chat text-message buffer for the
    /// `text_batch_delay_ms` feature. When the delay is non-zero,
    /// `send` pushes here instead of dispatching immediately and a
    /// flush task drains the buffer after the delay window.
    text_batch_buffers: Mutex<HashMap<String, Vec<BufferedText>>>,
    /// In-flight flush JoinHandles, keyed by chat. Tracked so we
    /// can abort and re-arm with a longer delay when a near-limit
    /// chunk shows up mid-window (the upstream's split-detection
    /// behavior — a continuation is likely coming).
    flush_handles: Mutex<HashMap<String, tokio::task::JoinHandle<()>>>,
    /// Tracked joined members per room. Populated from
    /// `m.room.member` state events during sync; consumed by the
    /// E2EE layer's `share_room_key` to know which devices need
    /// the Megolm key. Maintained even when E2EE is off so we
    /// don't have to retrofit member tracking later.
    room_members: Mutex<HashMap<String, HashSet<String>>>,
    /// E2EE handle. Populated during `listen()` when the
    /// `matrix-e2ee` feature is on AND `crypto_store_dir` is
    /// configured. None otherwise (channel falls back to the
    /// unencrypted path for everything).
    #[cfg(feature = "matrix-e2ee")]
    crypto: parking_lot::RwLock<Option<crate::channels::matrix_e2ee::MatrixCrypto>>,
}

#[derive(Debug, Clone)]
struct BufferedText {
    content: String,
    metadata: HashMap<String, String>,
    reply_to: Option<String>,
}

#[derive(Debug, Default)]
struct DedupCache {
    set: HashSet<String>,
    order: VecDeque<String>,
}

impl DedupCache {
    fn insert(&mut self, id: &str) -> bool {
        if !self.set.insert(id.to_string()) {
            return false;
        }
        self.order.push_back(id.to_string());
        while self.order.len() > DEDUP_CAP {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
        true
    }
}

#[derive(Debug, Clone, Default)]
struct TypingState {
    consecutive_failures: u32,
    cooldown_until: Option<std::time::Instant>,
    last_backoff: Duration,
}

// -- Construction ------------------------------------------------

impl MatrixChannel {
    /// Construct from config. Returns `None` when disabled or
    /// when required fields are missing. Auth is verified lazily
    /// at the start of `listen()` so config errors don't crash
    /// the whole gateway at startup.
    pub fn from_config(config: &MatrixChannelEntry) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        if config.homeserver.is_empty() {
            tracing::warn!("matrix channel enabled but `homeserver` is empty; refusing to start");
            return None;
        }
        if config.user_id.is_empty() {
            tracing::warn!("matrix channel enabled but `user_id` is empty; refusing to start");
            return None;
        }
        if config.access_token.is_empty() && config.password.is_empty() {
            tracing::warn!(
                "matrix channel enabled but neither `access_token` nor `password` is set; refusing to start"
            );
            return None;
        }
        let http = Client::builder()
            .pool_idle_timeout(Some(Duration::from_secs(90)))
            .build()
            .ok()?;
        let started_at_ms = chrono::Utc::now().timestamp_millis().max(0) as u64;
        Some(Self {
            config: config.clone(),
            state: Arc::new(MatrixState {
                access_token: Mutex::new(config.access_token.clone()),
                user_id: Mutex::new(config.user_id.clone()),
                next_batch: Mutex::new(None),
                seen_events: Mutex::new(DedupCache::default()),
                dm_rooms: Mutex::new(HashSet::new()),
                joined_rooms: Mutex::new(HashSet::new()),
                bot_threads: Mutex::new(HashSet::new()),
                typing_failures: Mutex::new(HashMap::new()),
                started_at_ms,
                last_sent_by_chat: Mutex::new(HashMap::new()),
                text_batch_buffers: Mutex::new(HashMap::new()),
                flush_handles: Mutex::new(HashMap::new()),
                room_members: Mutex::new(HashMap::new()),
                #[cfg(feature = "matrix-e2ee")]
                crypto: parking_lot::RwLock::new(None),
            }),
            http,
        })
    }

    fn base_url(&self) -> String {
        self.config.homeserver.trim_end_matches('/').to_string()
    }

    fn token(&self) -> String {
        self.state.access_token.lock().clone()
    }

    fn current_user_id(&self) -> String {
        self.state.user_id.lock().clone()
    }

    // -- Auth ----------------------------------------------------

    /// Resolve the access token. If config supplied one, verify it
    /// via `/account/whoami`. Otherwise exchange the password.
    async fn ensure_authenticated(&self) -> Result<()> {
        if !self.config.access_token.is_empty() {
            self.whoami().await
        } else {
            self.password_login().await
        }
    }

    async fn whoami(&self) -> Result<()> {
        let url = format!("{}/_matrix/client/v3/account/whoami", self.base_url());
        let resp = self
            .http
            .get(&url)
            .bearer_auth(self.token())
            .timeout(RPC_TIMEOUT)
            .send()
            .await
            .context("matrix whoami request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("matrix whoami returned {}", resp.status());
        }
        let body: Value = resp
            .json()
            .await
            .context("matrix whoami response not JSON")?;
        if let Some(uid) = body.get("user_id").and_then(|v| v.as_str()) {
            *self.state.user_id.lock() = uid.to_string();
        }
        Ok(())
    }

    async fn password_login(&self) -> Result<()> {
        let url = format!("{}/_matrix/client/v3/login", self.base_url());
        // user_id may be `@bot:server` — split for the m.id.user identifier.
        let identifier_user = self
            .config
            .user_id
            .strip_prefix('@')
            .map(|rest| rest.split(':').next().unwrap_or(rest))
            .unwrap_or(&self.config.user_id);
        let mut body = json!({
            "type": "m.login.password",
            "identifier": { "type": "m.id.user", "user": identifier_user },
            "password": self.config.password,
            "initial_device_display_name": "fennec",
        });
        if !self.config.device_id.is_empty() {
            body.as_object_mut()
                .unwrap()
                .insert("device_id".into(), json!(self.config.device_id));
        }
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(RPC_TIMEOUT)
            .send()
            .await
            .context("matrix login request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("matrix login returned {}", resp.status());
        }
        let value: Value = resp
            .json()
            .await
            .context("matrix login response not JSON")?;
        let token = value
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("matrix login response missing access_token"))?
            .to_string();
        *self.state.access_token.lock() = token;
        if let Some(uid) = value.get("user_id").and_then(|v| v.as_str()) {
            *self.state.user_id.lock() = uid.to_string();
        }
        Ok(())
    }

    // -- Sync loop ----------------------------------------------

    /// Long-running sync listener. Reconnects with exponential
    /// backoff + jitter on failure; aborts immediately on
    /// permanent auth errors.
    async fn run_sync(&self, tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        let mut backoff = RECONNECT_BACKOFF_INITIAL;
        loop {
            match self.sync_once(&tx).await {
                Ok(()) => {
                    backoff = RECONNECT_BACKOFF_INITIAL;
                }
                Err(SyncError::Permanent(e)) => {
                    tracing::error!(error = %e, "matrix sync permanent error; stopping");
                    return Err(e);
                }
                Err(SyncError::Transient(e)) => {
                    tracing::warn!(error = %e, "matrix sync transient error; backing off");
                    let jittered = backoff_with_jitter(backoff);
                    tokio::time::sleep(jittered).await;
                    backoff = (backoff * 2).min(RECONNECT_BACKOFF_MAX);
                }
            }
        }
    }

    async fn sync_once(&self, tx: &mpsc::Sender<InboundMessage>) -> Result<(), SyncError> {
        let mut url = format!("{}/_matrix/client/v3/sync", self.base_url());
        let since = self.state.next_batch.lock().clone();
        let mut sep = '?';
        if let Some(s) = &since {
            url.push_str(&format!("{}since={}", sep, urlencoding::encode(s)));
            sep = '&';
        }
        // Long-poll only after the first sync — initial sync uses
        // a short timeout so we don't hang for 30s while the user
        // is waiting for "channel ready" logs.
        let timeout_ms = if since.is_some() { SYNC_LONGPOLL_MS } else { 0 };
        url.push_str(&format!("{}timeout={}", sep, timeout_ms));

        let req = self
            .http
            .get(&url)
            .bearer_auth(self.token())
            .timeout(SYNC_OUTER_TIMEOUT);
        let resp = match tokio::time::timeout(SYNC_OUTER_TIMEOUT, req.send()).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return Err(SyncError::Transient(anyhow::anyhow!(e))),
            Err(_) => return Err(SyncError::Transient(anyhow::anyhow!("matrix sync wall-clock timeout"))),
        };
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(SyncError::Permanent(anyhow::anyhow!(
                "matrix sync auth failure: {}",
                status
            )));
        }
        if !status.is_success() {
            return Err(SyncError::Transient(anyhow::anyhow!(
                "matrix sync returned {}",
                status
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| SyncError::Transient(anyhow::anyhow!(e)))?;
        if let Some(errcode) = body.get("errcode").and_then(|v| v.as_str()) {
            if errcode == "M_UNKNOWN_TOKEN" {
                return Err(SyncError::Permanent(anyhow::anyhow!(
                    "matrix sync M_UNKNOWN_TOKEN"
                )));
            }
        }
        // Feed encryption-relevant fields (to_device, device_lists,
        // OTK counts, fallback keys) into the OlmMachine BEFORE we
        // persist next_batch — to-device events are ephemeral and
        // the homeserver won't re-deliver them.
        #[cfg(feature = "matrix-e2ee")]
        if let Some(crypto) = self.crypto_handle() {
            if let Err(e) = crypto.process_sync_changes(&body).await {
                tracing::warn!(error = %e, "matrix crypto process_sync_changes failed");
            }
        }
        if let Some(nb) = body.get("next_batch").and_then(|v| v.as_str()) {
            *self.state.next_batch.lock() = Some(nb.to_string());
            // Best-effort persist so a restart resumes here. Logged
            // but not failed-on if the disk write trips.
            if let Err(e) = self.persist_state().await {
                tracing::debug!(error = %e, "matrix state persist failed");
            }
        }
        // Refresh DM cache from account_data.
        if let Some(account) = body.get("account_data") {
            self.refresh_dm_rooms(account);
        }
        // Auto-join invites.
        if let Some(invites) = body
            .pointer("/rooms/invite")
            .and_then(|v| v.as_object())
        {
            for room_id in invites.keys() {
                if !self.is_room_allowed(room_id) {
                    continue;
                }
                if let Err(e) = self.join_room(room_id).await {
                    tracing::warn!(room = %room_id, error = %e, "matrix room join failed");
                }
            }
        }
        // Process joined rooms' timelines.
        if let Some(rooms) = body
            .pointer("/rooms/join")
            .and_then(|v| v.as_object())
        {
            {
                let mut joined = self.state.joined_rooms.lock();
                for k in rooms.keys() {
                    joined.insert(k.clone());
                }
            }
            for (room_id, room_data) in rooms {
                // Process state events first so encrypted-room
                // detection and member tracking are up-to-date
                // before timeline events are dispatched.
                if let Some(state_events) = room_data
                    .pointer("/state/events")
                    .and_then(|v| v.as_array())
                {
                    self.process_state_events(room_id, state_events);
                }
                let events = room_data
                    .pointer("/timeline/events")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                for event in events {
                    // State events can also arrive in the timeline
                    // (e.g. m.room.encryption when a room first
                    // turns on encryption). Handle them too.
                    if let Some(t) = event.get("type").and_then(|v| v.as_str()) {
                        if t == "m.room.encryption" || t == "m.room.member" {
                            self.process_state_events(room_id, std::slice::from_ref(&event));
                        }
                    }
                    // Decrypt m.room.encrypted events to recover
                    // the inner plaintext before passing through
                    // the regular handler. Failures pass the
                    // event through as-is so the agent at least
                    // sees a placeholder InboundMessage.
                    let event_for_handler = self.maybe_decrypt(room_id, event).await;
                    if let Some(mut inbound) =
                        self.handle_event(room_id, &event_for_handler)
                    {
                        // For media events with a configured cache
                        // dir, download the binary inline before
                        // surfacing the InboundMessage so consumers
                        // (vision tool, transcription tool) can
                        // read it from `matrix_media_path`.
                        if let Err(e) = self.enrich_inbound_with_media(&mut inbound).await {
                            tracing::debug!(
                                event_id = %inbound.id,
                                error = %e,
                                "matrix: media download failed; surfacing event without local path"
                            );
                        }
                        if tx.send(inbound).await.is_err() {
                            return Ok(()); // receiver dropped
                        }
                    }
                }
            }
        }
        // After all room events are processed, drain any
        // outgoing crypto requests (key shares, etc.) the
        // OlmMachine queued. Best-effort.
        #[cfg(feature = "matrix-e2ee")]
        if let Some(crypto) = self.crypto_handle() {
            if let Err(e) = crypto.drain_outgoing_requests().await {
                tracing::debug!(error = %e, "matrix crypto drain failed");
            }
        }
        Ok(())
    }

    fn refresh_dm_rooms(&self, account_data: &Value) {
        let events = match account_data.get("events").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => return,
        };
        for ev in events {
            if ev.get("type").and_then(|v| v.as_str()) != Some("m.direct") {
                continue;
            }
            let content = match ev.get("content").and_then(|v| v.as_object()) {
                Some(c) => c,
                None => continue,
            };
            let mut dm = self.state.dm_rooms.lock();
            dm.clear();
            for (_user, rooms) in content {
                if let Some(arr) = rooms.as_array() {
                    for r in arr {
                        if let Some(rid) = r.as_str() {
                            dm.insert(rid.to_string());
                        }
                    }
                }
            }
        }
    }

    async fn join_room(&self, room_id_or_alias: &str) -> Result<()> {
        let url = format!(
            "{}/_matrix/client/v3/join/{}",
            self.base_url(),
            urlencoding::encode(room_id_or_alias)
        );
        let resp = self
            .http
            .post(&url)
            .bearer_auth(self.token())
            .json(&json!({}))
            .timeout(RPC_TIMEOUT)
            .send()
            .await
            .context("matrix join request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("matrix join {} returned {}", room_id_or_alias, resp.status());
        }
        Ok(())
    }

    // -- Event handling -----------------------------------------

    fn handle_event(&self, room_id: &str, event: &Value) -> Option<InboundMessage> {
        let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match event_type {
            "m.room.message" => {}
            "m.room.encryption" => {
                // Track encryption status of the room. Without
                // E-4-2's E2EE feature on, we can't decrypt, but
                // we record the state for the (forthcoming) crypto
                // layer. Surfacing this via a dedicated metadata
                // event would require a new InboundMessage shape,
                // which is the encrypted-channel concern; here we
                // just no-op.
                return None;
            }
            "m.reaction" => {
                return self.handle_reaction_event(room_id, event);
            }
            _ => return None,
        }
        let event_id = event
            .get("event_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if event_id.is_empty() {
            return None;
        }
        if !self.state.seen_events.lock().insert(event_id) {
            return None; // dedup
        }
        // Startup grace: skip events older than (started_at - GRACE).
        if let Some(ts) = event.get("origin_server_ts").and_then(|v| v.as_i64()) {
            let cutoff = self.state.started_at_ms.saturating_sub(STARTUP_GRACE_MS);
            if (ts as u64) < cutoff {
                return None;
            }
        }
        let sender = event
            .get("sender")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if sender.is_empty() {
            return None;
        }
        // Drop our own messages.
        if sender.eq_ignore_ascii_case(&self.current_user_id()) {
            return None;
        }
        // Drop appservice/bridge bot senders (`@_…:server`).
        if is_bridge_sender(&sender) {
            return None;
        }
        let content = event.get("content")?;
        let msgtype = content
            .get("msgtype")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        // Drop notices to avoid bot-loops.
        if msgtype == "m.notice" {
            return None;
        }
        // Edits: skip; the original event already produced the
        // user's message. (Honoring edits would update the
        // existing inbound; we don't have a hook for that.)
        if let Some(rel) = content
            .pointer("/m.relates_to/rel_type")
            .and_then(|v| v.as_str())
        {
            if rel == "m.replace" {
                return None;
            }
        }
        // Body: prefer plain `body`. Strip reply-fallback `> ` if
        // present.
        let raw_body = content
            .get("body")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let body = strip_reply_fallback(&raw_body);

        // Reply target.
        let reply_to = content
            .pointer("/m.relates_to/m.in_reply_to/event_id")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Thread id (from m.thread relation).
        let thread_id = if content
            .pointer("/m.relates_to/rel_type")
            .and_then(|v| v.as_str())
            == Some("m.thread")
        {
            content
                .pointer("/m.relates_to/event_id")
                .and_then(|v| v.as_str())
                .map(String::from)
        } else {
            None
        };

        // Mention detection.
        let mentions_us = self.event_mentions_us(content, &body);
        // If we already participate in this thread, we're in
        // regardless of mention.
        let in_bot_thread = thread_id
            .as_deref()
            .map(|t| self.state.bot_threads.lock().contains(t))
            .unwrap_or(false);

        // Visibility filtering.
        let is_dm = self.state.dm_rooms.lock().contains(room_id);
        if is_dm {
            if !self.is_dm_sender_allowed(&sender) {
                return None;
            }
        } else {
            if !self.is_room_allowed(room_id) {
                return None;
            }
            // Mention-required gating in non-free rooms.
            let free = self
                .config
                .free_response_rooms
                .iter()
                .any(|r| r == room_id);
            if self.config.require_mention && !free && !mentions_us && !in_bot_thread {
                tracing::debug!(
                    room = %room_id,
                    sender = %sender,
                    "matrix: dropping event — mention required and not present"
                );
                return None;
            }
        }
        let is_media_msg = matches!(
            msgtype,
            "m.image" | "m.file" | "m.audio" | "m.video"
        );
        // Suppress transport-style filenames (`IMG_*.jpg`,
        // `signal-*.png`, etc.) from the visible body; the agent
        // should respond to the media itself, not the auto-name.
        // The full filename is still recoverable via the
        // `matrix_media_filename` metadata field below.
        let mut body = body;
        let media_filename = if is_media_msg && looks_like_transport_filename(&body) {
            let captured = body.clone();
            body = String::new();
            Some(captured)
        } else {
            None
        };
        if body.is_empty() && !is_media_msg {
            return None;
        }
        let timestamp = event
            .get("origin_server_ts")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            .max(0) as u64;
        let mut metadata = HashMap::new();
        metadata.insert("matrix_event_id".into(), event_id.to_string());
        metadata.insert("matrix_room_id".into(), room_id.to_string());
        metadata.insert("matrix_msgtype".into(), msgtype.to_string());
        if is_dm {
            metadata.insert("matrix_dm".into(), "true".into());
        }
        if let Some(t) = &thread_id {
            metadata.insert("matrix_thread_id".into(), t.clone());
            // We've now seen activity in this thread — track it
            // so future events without explicit mention still pass
            // the gate.
            self.state.bot_threads.lock().insert(t.clone());
        }
        if mentions_us {
            metadata.insert("matrix_mentioned".into(), "true".into());
        }
        // Carry m.image / m.file / m.audio / m.video metadata
        // through; downstream consumers (e.g. vision tools) can
        // resolve mxc URLs via `mxc_to_http_url`, or read the
        // local cache path from `matrix_media_path` when
        // `media_cache_dir` is configured (the sync loop runs the
        // download asynchronously via `enrich_inbound_with_media`).
        let is_media = matches!(
            msgtype,
            "m.image" | "m.file" | "m.audio" | "m.video"
        );
        // Plaintext media has `content.url` (an mxc URI). Encrypted
        // media has `content.file` instead — an EncryptedFile
        // object whose own `url` field is the (encrypted) ciphertext
        // mxc URI plus the AES key/IV/hashes needed to decrypt
        // after download.
        if let Some(url) = content.get("url").and_then(|v| v.as_str()) {
            metadata.insert("matrix_media_mxc".into(), url.to_string());
        } else if let Some(file_obj) = content.get("file") {
            if let Some(url) = file_obj.get("url").and_then(|v| v.as_str()) {
                metadata.insert("matrix_media_mxc".into(), url.to_string());
            }
            metadata.insert("matrix_media_encrypted".into(), "true".into());
            // The full EncryptedFile JSON travels through metadata
            // so the async downloader (`enrich_inbound_with_media`)
            // can hand it to the crypto layer's
            // `decrypt_attachment`.
            if let Ok(s) = serde_json::to_string(file_obj) {
                metadata.insert("matrix_media_file_json".into(), s);
            }
        }
        if is_media {
            metadata.insert(
                "matrix_media_kind".into(),
                msgtype.trim_start_matches("m.").to_string(),
            );
        }
        if let Some(mime) = content
            .pointer("/info/mimetype")
            .and_then(|v| v.as_str())
        {
            metadata.insert("matrix_media_mime".into(), mime.to_string());
        }
        if let Some(size) = content
            .pointer("/info/size")
            .and_then(|v| v.as_u64())
        {
            metadata.insert("matrix_media_size".into(), size.to_string());
        }
        if let Some(w) = content.pointer("/info/w").and_then(|v| v.as_u64()) {
            metadata.insert("matrix_media_width".into(), w.to_string());
        }
        if let Some(h) = content.pointer("/info/h").and_then(|v| v.as_u64()) {
            metadata.insert("matrix_media_height".into(), h.to_string());
        }
        if let Some(d) = content
            .pointer("/info/duration")
            .and_then(|v| v.as_u64())
        {
            metadata.insert("matrix_media_duration_ms".into(), d.to_string());
        }
        if let Some(name) = media_filename {
            metadata.insert("matrix_media_filename".into(), name);
        }

        Some(InboundMessage {
            id: event_id.to_string(),
            sender,
            content: body,
            channel: "matrix".into(),
            chat_id: room_id.to_string(),
            timestamp,
            reply_to,
            metadata,
        })
    }

    /// Download the media for an inbound `m.image`/`m.file`/
    /// `m.audio`/`m.video` event into the configured local cache,
    /// and surface the local path in `matrix_media_path`. No-op
    /// when `media_cache_dir` is empty or the event isn't media.
    async fn enrich_inbound_with_media(&self, inbound: &mut InboundMessage) -> Result<()> {
        if self.config.media_cache_dir.is_empty() {
            return Ok(());
        }
        let mxc = match inbound.metadata.get("matrix_media_mxc") {
            Some(s) if !s.is_empty() => s.clone(),
            _ => return Ok(()),
        };
        let mime = inbound.metadata.get("matrix_media_mime").cloned();
        let is_encrypted = inbound
            .metadata
            .get("matrix_media_encrypted")
            .map(|s| s == "true")
            .unwrap_or(false);

        if !is_encrypted {
            // Plaintext media — straight download to cache.
            let path = self
                .download_media(&mxc, &inbound.id, mime.as_deref())
                .await?;
            if let Some(p) = path {
                inbound
                    .metadata
                    .insert("matrix_media_path".into(), p.to_string_lossy().into_owned());
            }
            return Ok(());
        }

        // Encrypted media path: fetch the ciphertext, hand it to
        // the crypto layer to decrypt, then write the plaintext to
        // the cache directory.
        #[cfg(feature = "matrix-e2ee")]
        {
            let crypto = match self.crypto_handle() {
                Some(c) => c,
                None => {
                    // Encrypted media but the crypto layer isn't
                    // available — surface the mxc URL but skip
                    // download (we can't decrypt anyway).
                    tracing::debug!(
                        event_id = %inbound.id,
                        "matrix: encrypted media but no crypto handle; skipping decrypt"
                    );
                    return Ok(());
                }
            };
            let file_json_str = match inbound.metadata.get("matrix_media_file_json") {
                Some(s) if !s.is_empty() => s.clone(),
                _ => return Ok(()),
            };
            let file_json: serde_json::Value = serde_json::from_str(&file_json_str)
                .context("matrix encrypted file metadata not parseable")?;
            let ciphertext = match self.fetch_media_bytes(&mxc).await? {
                Some(b) => b,
                None => return Ok(()),
            };
            let plaintext = crypto.decrypt_attachment(&file_json, ciphertext)?;
            let path = self
                .write_media_bytes(&plaintext, &inbound.id, mime.as_deref())
                .await?;
            inbound
                .metadata
                .insert("matrix_media_path".into(), path.to_string_lossy().into_owned());
        }
        #[cfg(not(feature = "matrix-e2ee"))]
        {
            tracing::debug!(
                event_id = %inbound.id,
                "matrix: encrypted media event but matrix-e2ee feature is disabled; skipping decrypt"
            );
        }
        Ok(())
    }

    /// Build an `InboundMessage` from an `m.reaction` event.
    /// Reactions are surfaced through the same channel as text
    /// messages — content is `"[reaction] {key}"` and the
    /// `matrix_reaction_target` / `matrix_reaction_key` metadata
    /// fields carry the structured payload. Higher-level flows
    /// (e.g. `ask_user` approval prompts) match on these.
    fn handle_reaction_event(
        &self,
        room_id: &str,
        event: &Value,
    ) -> Option<InboundMessage> {
        let event_id = event
            .get("event_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if event_id.is_empty() {
            return None;
        }
        if !self.state.seen_events.lock().insert(event_id) {
            return None;
        }
        if let Some(ts) = event.get("origin_server_ts").and_then(|v| v.as_i64()) {
            let cutoff = self.state.started_at_ms.saturating_sub(STARTUP_GRACE_MS);
            if (ts as u64) < cutoff {
                return None;
            }
        }
        let sender = event
            .get("sender")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if sender.is_empty()
            || sender.eq_ignore_ascii_case(&self.current_user_id())
            || is_bridge_sender(&sender)
        {
            return None;
        }
        let target_event_id = event
            .pointer("/content/m.relates_to/event_id")
            .and_then(|v| v.as_str())?
            .to_string();
        let key = event
            .pointer("/content/m.relates_to/key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let rel_type = event
            .pointer("/content/m.relates_to/rel_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if rel_type != "m.annotation" {
            return None;
        }
        let is_dm = self.state.dm_rooms.lock().contains(room_id);
        if is_dm {
            if !self.is_dm_sender_allowed(&sender) {
                return None;
            }
        } else if !self.is_room_allowed(room_id) {
            return None;
        }
        let timestamp = event
            .get("origin_server_ts")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            .max(0) as u64;
        let mut metadata = HashMap::new();
        metadata.insert("matrix_event_id".into(), event_id.to_string());
        metadata.insert("matrix_room_id".into(), room_id.to_string());
        metadata.insert("matrix_msgtype".into(), "m.reaction".into());
        metadata.insert("matrix_reaction_target".into(), target_event_id);
        metadata.insert("matrix_reaction_key".into(), key.clone());
        if is_dm {
            metadata.insert("matrix_dm".into(), "true".into());
        }
        Some(InboundMessage {
            id: event_id.to_string(),
            sender,
            content: format!("[reaction] {}", key),
            channel: "matrix".into(),
            chat_id: room_id.to_string(),
            timestamp,
            reply_to: None,
            metadata,
        })
    }

    /// Whether `content`/`body` mentions our bot. Prefers MSC3952
    /// `m.mentions.user_ids`; falls back to a body substring scan
    /// when the metadata is absent (older clients).
    fn event_mentions_us(&self, content: &Value, body: &str) -> bool {
        let me = self.current_user_id();
        if me.is_empty() {
            return false;
        }
        if let Some(arr) = content
            .pointer("/m.mentions/user_ids")
            .and_then(|v| v.as_array())
        {
            for item in arr {
                if let Some(s) = item.as_str() {
                    if s.eq_ignore_ascii_case(&me) {
                        return true;
                    }
                }
            }
            // If the metadata exists at all, trust it as
            // authoritative — don't fall through to body scan.
            return false;
        }
        body.to_ascii_lowercase()
            .contains(&me.to_ascii_lowercase())
    }

    fn is_dm_sender_allowed(&self, sender: &str) -> bool {
        // Default-DENY: an empty allowlist refuses everyone ("*" opts
        // into allow-everyone). See the telegram channel for rationale.
        let allow = &self.config.allowed_users;
        if allow.is_empty() {
            tracing::warn!(
                "Matrix: refusing DM from '{sender}' — no allowed_users configured. \
                 Add your user id to [channels.matrix].allowed_users (or \"*\" to allow everyone)."
            );
            return false;
        }
        if allow.iter().any(|s| s == "*") {
            return true;
        }
        allow.iter().any(|s| s.eq_ignore_ascii_case(sender))
    }

    fn is_room_allowed(&self, room_id: &str) -> bool {
        let allow = &self.config.allowed_rooms;
        if allow.is_empty() {
            return false;
        }
        if allow.iter().any(|s| s == "*") {
            return true;
        }
        allow.iter().any(|s| s == room_id)
    }

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

    fn typing_in_cooldown(&self, chat_id: &str) -> bool {
        self.state
            .typing_failures
            .lock()
            .get(chat_id)
            .and_then(|s| s.cooldown_until)
            .map(|t| std::time::Instant::now() < t)
            .unwrap_or(false)
    }

    fn typing_clear(&self, chat_id: &str) {
        self.state.typing_failures.lock().remove(chat_id);
    }
}

#[derive(Debug)]
enum SyncError {
    Permanent(anyhow::Error),
    Transient(anyhow::Error),
}

/// State for an in-flight Matrix approval prompt: a message the
/// bot sent, decorated with seed reactions (e.g. ✅ / ❎) that the
/// human is expected to click on. Higher-level flows
/// (`ask_user_tool`-style approvals) construct one of these,
/// `send` it, then on inbound `m.reaction` events with
/// `matrix_reaction_target` matching `prompt_event_id` resolve to
/// the user's choice. `cleanup` redacts the bot's seed reactions
/// once the prompt is resolved.
///
/// The struct is the *capability layer*; wiring it to Fennec's
/// generic `ask_user_tool` flow is a cross-cutting concern that
/// requires hooking reaction-as-reply into `PendingReplies`. Until
/// that's done, this type is available for direct use by
/// Matrix-specific tools or tests.
#[derive(Debug, Clone)]
pub struct MatrixApprovalPrompt {
    pub room_id: String,
    pub prompt_event_id: String,
    /// Seed reactions the bot added (key → reaction event_id).
    /// Used by `cleanup` to redact each one.
    pub seed_reactions: Vec<(String, String)>,
}

impl MatrixApprovalPrompt {
    /// Send a fresh prompt: posts `prompt_text` as `m.text`,
    /// then adds each `seed_key` as an `m.reaction` so the user
    /// can resolve the prompt with one click.
    pub async fn send(
        channel: &MatrixChannel,
        room_id: &str,
        prompt_text: &str,
        seed_keys: &[&str],
    ) -> Result<Self> {
        let mut content = json!({
            "msgtype": "m.text",
            "body": prompt_text,
        });
        if channel.config.markdown_to_html && chunk_has_markdown(prompt_text) {
            let html = markdown_to_html(prompt_text);
            let obj = content.as_object_mut().unwrap();
            obj.insert("format".into(), json!("org.matrix.custom.html"));
            obj.insert("formatted_body".into(), json!(html));
        }
        let prompt_event_id = channel
            .send_event(room_id, "m.room.message", content)
            .await?;
        let mut seed_reactions = Vec::with_capacity(seed_keys.len());
        for key in seed_keys {
            match channel.react(room_id, &prompt_event_id, key).await {
                Ok(rx_id) => seed_reactions.push((key.to_string(), rx_id)),
                Err(e) => {
                    tracing::debug!(
                        room = %room_id,
                        key = %key,
                        error = %e,
                        "matrix approval prompt: seed reaction failed"
                    );
                }
            }
        }
        Ok(Self {
            room_id: room_id.to_string(),
            prompt_event_id,
            seed_reactions,
        })
    }

    /// Whether an inbound `m.reaction` event resolves this prompt.
    /// Matches by `matrix_reaction_target`.
    pub fn matches_reaction(&self, inbound_metadata: &HashMap<String, String>) -> bool {
        inbound_metadata
            .get("matrix_reaction_target")
            .map(|s| s.as_str())
            == Some(self.prompt_event_id.as_str())
    }

    /// Redact the bot's seed reactions. Best-effort — failures
    /// log but don't propagate, since cleanup is purely
    /// cosmetic.
    pub async fn cleanup(self, channel: &MatrixChannel) {
        for (key, rx_id) in self.seed_reactions {
            if let Err(e) = channel.redact(&self.room_id, &rx_id, None).await {
                tracing::debug!(
                    room = %self.room_id,
                    key = %key,
                    error = %e,
                    "matrix approval prompt: seed reaction redact failed"
                );
            }
        }
    }
}

// -- Public API beyond Channel trait -----------------------------

impl MatrixChannel {
    /// Generic event-send helper. Returns the homeserver-assigned
    /// `event_id`. Used internally by `send`, `react`, `edit`,
    /// `redact`; also available to higher-level integration code
    /// (e.g. an approval-prompt flow that wants the event id of
    /// the prompt to attach reactions to).
    pub async fn send_event(
        &self,
        room_id: &str,
        event_type: &str,
        content: Value,
    ) -> Result<String> {
        let txn_id = format!("fennec.{}", uuid::Uuid::new_v4());
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/send/{}/{}",
            self.base_url(),
            urlencoding::encode(room_id),
            urlencoding::encode(event_type),
            urlencoding::encode(&txn_id),
        );
        let resp = self
            .http
            .put(&url)
            .bearer_auth(self.token())
            .json(&content)
            .timeout(RPC_TIMEOUT)
            .send()
            .await
            .context("matrix send_event request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("matrix send_event returned {}", resp.status());
        }
        let body: Value = resp
            .json()
            .await
            .context("matrix send_event response not JSON")?;
        let event_id = body
            .get("event_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !event_id.is_empty() && event_type == "m.room.message" {
            self.state
                .last_sent_by_chat
                .lock()
                .insert(room_id.to_string(), event_id.clone());
        }
        Ok(event_id)
    }

    /// Add a reaction (`m.annotation`) to an existing event. Used by
    /// approval-prompt flows to mark a sent message with ✅/❎ seed
    /// reactions, and to react to inbound events for acknowledgement.
    pub async fn react(
        &self,
        room_id: &str,
        target_event_id: &str,
        key: &str,
    ) -> Result<String> {
        let content = json!({
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": target_event_id,
                "key": key,
            }
        });
        self.send_event(room_id, "m.reaction", content).await
    }

    /// Redact (delete) an event. Used to clean up seed reactions
    /// after an approval prompt resolves, or to retract bot
    /// messages.
    pub async fn redact(
        &self,
        room_id: &str,
        event_id: &str,
        reason: Option<&str>,
    ) -> Result<String> {
        let txn_id = format!("fennec.{}", uuid::Uuid::new_v4());
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/redact/{}/{}",
            self.base_url(),
            urlencoding::encode(room_id),
            urlencoding::encode(event_id),
            urlencoding::encode(&txn_id),
        );
        let body = match reason {
            Some(r) => json!({ "reason": r }),
            None => json!({}),
        };
        let resp = self
            .http
            .put(&url)
            .bearer_auth(self.token())
            .json(&body)
            .timeout(RPC_TIMEOUT)
            .send()
            .await
            .context("matrix redact request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("matrix redact returned {}", resp.status());
        }
        let parsed: Value = resp
            .json()
            .await
            .context("matrix redact response not JSON")?;
        Ok(parsed
            .get("event_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string())
    }

    /// Edit a previously-sent message via the `m.replace` relation.
    /// `original_event_id` is the message being edited; `new_body`
    /// replaces its plain-text content (an HTML `formatted_body`
    /// is included when `markdown_to_html` is on and the new body
    /// looks like markdown). The fallback `body` follows the
    /// upstream convention of prefixing `* ` so non-edit-aware
    /// clients show "* edited body".
    pub async fn edit(
        &self,
        room_id: &str,
        original_event_id: &str,
        new_body: &str,
    ) -> Result<String> {
        let mut new_content = json!({
            "msgtype": "m.text",
            "body": new_body,
        });
        if self.config.markdown_to_html && chunk_has_markdown(new_body) {
            let html = markdown_to_html(new_body);
            let obj = new_content.as_object_mut().unwrap();
            obj.insert("format".into(), json!("org.matrix.custom.html"));
            obj.insert("formatted_body".into(), json!(html));
        }
        let content = json!({
            "msgtype": "m.text",
            "body": format!("* {}", new_body),
            "m.new_content": new_content,
            "m.relates_to": {
                "rel_type": "m.replace",
                "event_id": original_event_id,
            }
        });
        self.send_event(room_id, "m.room.message", content).await
    }

    /// Upload binary media to the homeserver's media repository.
    /// Returns the `mxc://` URI on success. The caller then
    /// references that URI in an `m.image`/`m.file`/`m.audio`/
    /// `m.video` event to surface the media inline.
    pub async fn upload_media(
        &self,
        bytes: &[u8],
        mime: &str,
        filename: Option<&str>,
    ) -> Result<String> {
        let mut url = format!("{}/_matrix/media/v3/upload", self.base_url());
        if let Some(name) = filename {
            url.push_str(&format!("?filename={}", urlencoding::encode(name)));
        }
        let resp = self
            .http
            .post(&url)
            .bearer_auth(self.token())
            .header("Content-Type", mime)
            .body(bytes.to_vec())
            .timeout(Duration::from_secs(120))
            .send()
            .await
            .context("matrix media upload failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("matrix media upload returned {}", resp.status());
        }
        let body: Value = resp
            .json()
            .await
            .context("matrix media upload response not JSON")?;
        body.get("content_uri")
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| anyhow::anyhow!("matrix media upload missing content_uri"))
    }

    /// Most recent `event_id` this channel sent into the given
    /// chat, if any. Useful for approval-prompt flows that want
    /// to react to the message they just sent without threading
    /// the id through the caller.
    pub fn last_sent_event_id(&self, chat_id: &str) -> Option<String> {
        self.state.last_sent_by_chat.lock().get(chat_id).cloned()
    }

    /// Decide whether to wrap an outbound message in a thread,
    /// and if so on which event. Honors `auto_thread` (rooms),
    /// `dm_auto_thread` (DMs), and `dm_mention_threads` (DMs only
    /// when the inbound was an @mention). Caller may pin an
    /// explicit thread anchor via `metadata.matrix_thread_id`,
    /// otherwise we anchor on `reply_to`.
    fn pick_thread_anchor(&self, room_id: &str, msg: &SendMessage) -> Option<String> {
        if let Some(t) = msg.metadata.get("matrix_thread_id") {
            if !t.is_empty() {
                return Some(t.clone());
            }
        }
        let is_dm = self.state.dm_rooms.lock().contains(room_id);
        let want_thread = if is_dm {
            if self.config.dm_auto_thread {
                true
            } else if self.config.dm_mention_threads {
                msg.metadata.get("matrix_mentioned").map(String::as_str) == Some("true")
            } else {
                false
            }
        } else {
            self.config.auto_thread
        };
        if !want_thread {
            return None;
        }
        msg.reply_to.clone()
    }

    /// Attach reply / thread relation metadata to an outbound
    /// content object. Called by `send` for both text bodies and
    /// media events so attachments and their text companion stay
    /// in the same thread.
    fn attach_relation(&self, content: &mut Value, room_id: &str, msg: &SendMessage) {
        if let Some(thread_id) = self.pick_thread_anchor(room_id, msg) {
            let in_reply_to = msg
                .reply_to
                .clone()
                .unwrap_or_else(|| thread_id.clone());
            content.as_object_mut().unwrap().insert(
                "m.relates_to".into(),
                json!({
                    "rel_type": "m.thread",
                    "event_id": thread_id,
                    "is_falling_back": true,
                    "m.in_reply_to": { "event_id": in_reply_to },
                }),
            );
            self.state.bot_threads.lock().insert(thread_id);
        } else if let Some(reply_to) = &msg.reply_to {
            content.as_object_mut().unwrap().insert(
                "m.relates_to".into(),
                json!({
                    "m.in_reply_to": { "event_id": reply_to },
                }),
            );
        }
    }

    /// Build the JSON content for an *encrypted* media event.
    /// Same shape as `media_event_content` but with `file: { v,
    /// key, iv, hashes, url }` instead of a top-level `url`.
    /// `file_info` is the EncryptedFile JSON returned by
    /// `encrypt_attachment` with the uploaded ciphertext's mxc
    /// URL added in.
    #[cfg(feature = "matrix-e2ee")]
    fn media_event_content_encrypted(
        &self,
        att: &crate::bus::MediaAttachment,
        file_info: Value,
    ) -> Value {
        let (msgtype, default_name, voice) = match att.kind {
            crate::bus::MediaKind::Image => ("m.image", "image", false),
            crate::bus::MediaKind::Audio => ("m.audio", "audio", false),
            crate::bus::MediaKind::Voice => ("m.audio", "voice", true),
            crate::bus::MediaKind::Video => ("m.video", "video", false),
            crate::bus::MediaKind::File => ("m.file", "file", false),
        };
        let body = att
            .filename
            .clone()
            .unwrap_or_else(|| default_name.to_string());
        let mut info = serde_json::Map::new();
        info.insert("mimetype".into(), json!(att.mime));
        info.insert("size".into(), json!(att.bytes.len()));
        if let Some(w) = att.width {
            info.insert("w".into(), json!(w));
        }
        if let Some(h) = att.height {
            info.insert("h".into(), json!(h));
        }
        if let Some(d) = att.duration_ms {
            info.insert("duration".into(), json!(d));
        }
        let mut content = serde_json::Map::new();
        content.insert("msgtype".into(), json!(msgtype));
        content.insert("body".into(), json!(body));
        // Encrypted media replaces top-level `url` with `file`.
        content.insert("file".into(), file_info);
        content.insert("info".into(), Value::Object(info));
        if voice {
            content.insert(
                "org.matrix.msc3245.voice".into(),
                Value::Object(serde_json::Map::new()),
            );
            content.insert(
                "org.matrix.msc1767.audio".into(),
                Value::Object(serde_json::Map::new()),
            );
        }
        Value::Object(content)
    }

    /// Build the JSON content for an `m.image`/`m.file`/`m.audio`/
    /// `m.video` event, given the uploaded media's mxc URI. For
    /// `MediaKind::Voice` we emit `m.audio` plus the MSC3245 voice
    /// marker so clients render it as a voice note rather than a
    /// generic audio attachment.
    fn media_event_content(
        &self,
        att: &crate::bus::MediaAttachment,
        mxc: &str,
    ) -> Value {
        let (msgtype, default_name, voice) = match att.kind {
            crate::bus::MediaKind::Image => ("m.image", "image", false),
            crate::bus::MediaKind::Audio => ("m.audio", "audio", false),
            crate::bus::MediaKind::Voice => ("m.audio", "voice", true),
            crate::bus::MediaKind::Video => ("m.video", "video", false),
            crate::bus::MediaKind::File => ("m.file", "file", false),
        };
        let body = att
            .filename
            .clone()
            .unwrap_or_else(|| default_name.to_string());
        let mut info = serde_json::Map::new();
        info.insert("mimetype".into(), json!(att.mime));
        info.insert("size".into(), json!(att.bytes.len()));
        if let Some(w) = att.width {
            info.insert("w".into(), json!(w));
        }
        if let Some(h) = att.height {
            info.insert("h".into(), json!(h));
        }
        if let Some(d) = att.duration_ms {
            info.insert("duration".into(), json!(d));
        }
        let mut content = serde_json::Map::new();
        content.insert("msgtype".into(), json!(msgtype));
        content.insert("body".into(), json!(body));
        content.insert("url".into(), json!(mxc));
        content.insert("info".into(), Value::Object(info));
        if voice {
            content.insert(
                "org.matrix.msc3245.voice".into(),
                Value::Object(serde_json::Map::new()),
            );
            // MSC1767 audio extension — empty audio object signals
            // a voice message to clients that don't understand the
            // top-level voice marker.
            content.insert(
                "org.matrix.msc1767.audio".into(),
                Value::Object(serde_json::Map::new()),
            );
        }
        Value::Object(content)
    }

    /// Fetch raw media bytes from the homeserver. Resolves the
    /// mxc URI to the authenticated download endpoint, retries up
    /// to `MEDIA_DOWNLOAD_ATTEMPTS` on transient failure with
    /// exponential backoff, aborts on permanent 4xx (except 429).
    /// Used by both the plaintext download and the encrypted
    /// download path.
    pub async fn fetch_media_bytes(&self, mxc: &str) -> Result<Option<Vec<u8>>> {
        let url = match mxc_to_http_url(&self.base_url(), mxc) {
            Some(u) => u,
            None => return Ok(None),
        };
        let mut attempt: u32 = 0;
        let mut last_err: Option<anyhow::Error> = None;
        loop {
            attempt += 1;
            match self
                .http
                .get(&url)
                .bearer_auth(self.token())
                .timeout(Duration::from_secs(120))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => match resp.bytes().await {
                    Ok(b) => return Ok(Some(b.to_vec())),
                    Err(e) => last_err = Some(anyhow::anyhow!(e)),
                },
                Ok(resp) => {
                    let status = resp.status();
                    last_err = Some(anyhow::anyhow!(
                        "matrix media download returned {}",
                        status
                    ));
                    if status.is_client_error()
                        && status != reqwest::StatusCode::TOO_MANY_REQUESTS
                    {
                        return Err(last_err.unwrap());
                    }
                }
                Err(e) => last_err = Some(anyhow::anyhow!(e)),
            }
            if attempt >= MEDIA_DOWNLOAD_ATTEMPTS {
                return Err(
                    last_err.unwrap_or_else(|| anyhow::anyhow!("matrix media download failed")),
                );
            }
            let delay = Duration::from_millis(200u64 * (1u64 << (attempt - 1)).min(10));
            tokio::time::sleep(delay).await;
        }
    }

    /// Download plaintext media to the configured cache directory
    /// and return the local path. No-op when `media_cache_dir` is
    /// empty.
    pub async fn download_media(
        &self,
        mxc: &str,
        event_id: &str,
        mime: Option<&str>,
    ) -> Result<Option<std::path::PathBuf>> {
        if self.config.media_cache_dir.is_empty() {
            return Ok(None);
        }
        let bytes = match self.fetch_media_bytes(mxc).await? {
            Some(b) => b,
            None => return Ok(None),
        };
        self.write_media_bytes(&bytes, event_id, mime).await.map(Some)
    }

    /// Persist media bytes to the cache directory. Shared between
    /// the plaintext and encrypted download paths.
    async fn write_media_bytes(
        &self,
        bytes: &[u8],
        event_id: &str,
        mime: Option<&str>,
    ) -> Result<std::path::PathBuf> {
        let dir = std::path::PathBuf::from(&self.config.media_cache_dir);
        tokio::fs::create_dir_all(&dir)
            .await
            .with_context(|| format!("matrix media cache dir create failed: {}", dir.display()))?;
        let ext = mime.and_then(extension_for_mime).unwrap_or("bin");
        let safe_id = event_id.replace(['/', '\\', ':'], "_");
        let path = dir.join(format!("{}.{}", safe_id, ext));
        tokio::fs::write(&path, bytes)
            .await
            .with_context(|| format!("matrix media write failed: {}", path.display()))?;
        Ok(path)
    }

    /// Persist the channel's `next_batch` sync token to the
    /// configured `state_file`, so the next start resumes from
    /// the same point rather than re-running the initial sync.
    /// No-op when `state_file` is empty.
    async fn persist_state(&self) -> Result<()> {
        if self.config.state_file.is_empty() {
            return Ok(());
        }
        let nb = self.state.next_batch.lock().clone();
        let body = serde_json::json!({ "next_batch": nb });
        let path = std::path::PathBuf::from(&self.config.state_file);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        tokio::fs::write(&path, body.to_string())
            .await
            .with_context(|| format!("matrix state persist failed: {}", path.display()))?;
        Ok(())
    }

    /// Read the persisted `next_batch` token from `state_file` and
    /// load it into channel state. No-op when `state_file` is
    /// empty or the file is missing/malformed.
    async fn load_state(&self) {
        if self.config.state_file.is_empty() {
            return;
        }
        let path = std::path::PathBuf::from(&self.config.state_file);
        let bytes = match tokio::fs::read(&path).await {
            Ok(b) => b,
            Err(_) => return,
        };
        let parsed: Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(_) => return,
        };
        if let Some(s) = parsed.get("next_batch").and_then(|v| v.as_str()) {
            *self.state.next_batch.lock() = Some(s.to_string());
        }
    }

    /// Core send implementation, bypassing any batching layer.
    /// Called directly by the trait's `send` when batching is off,
    /// and indirectly by the flush task once a batched buffer is
    /// drained.
    async fn send_immediate(&self, message: &SendMessage) -> Result<()> {
        let room_id = message.recipient.as_str();
        if room_id.is_empty() {
            anyhow::bail!("matrix: empty recipient (room_id) in send");
        }

        // Attachments first — each becomes its own m.image / m.file
        // / m.audio / m.video event so clients render them inline.
        // They share the reply / thread anchor with the text body
        // so threaded conversations stay grouped.
        for att in &message.attachments {
            // Encrypted-room branch: encrypt bytes via the crypto
            // layer's AttachmentEncryptor, upload the ciphertext,
            // and emit `file` (EncryptedFile) instead of `url` so
            // recipient clients know to decrypt. Falls back to
            // plaintext upload + send when the room isn't
            // encrypted or the crypto handle isn't active.
            #[cfg(feature = "matrix-e2ee")]
            {
                let encrypted_path = match self.crypto_handle() {
                    Some(c) if c.is_encrypted(room_id) => Some(c),
                    _ => None,
                };
                if let Some(crypto) = encrypted_path {
                    match crypto.encrypt_attachment(&att.bytes) {
                        Ok((ciphertext, mut file_info)) => {
                            let mxc = self
                                .upload_media(&ciphertext, "application/octet-stream", None)
                                .await?;
                            // The encrypted-file JSON wants `url`
                            // alongside the crypto fields.
                            if let Some(obj) = file_info.as_object_mut() {
                                obj.insert("url".into(), json!(mxc));
                            }
                            let mut content =
                                self.media_event_content_encrypted(att, file_info);
                            self.attach_relation(&mut content, room_id, message);
                            // Wrap the whole `m.room.message`
                            // through the same encrypt path text
                            // takes — the *content* is encrypted,
                            // not just the file.
                            let recipients = self.room_member_recipients(room_id);
                            match crypto
                                .encrypt_room_message(room_id, &content, &recipients)
                                .await
                            {
                                Ok(encrypted) => {
                                    let _ = self
                                        .send_event(
                                            room_id,
                                            "m.room.encrypted",
                                            encrypted,
                                        )
                                        .await?;
                                    continue;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        room = %room_id,
                                        error = %e,
                                        "matrix media: room-event encrypt failed; \
                                         falling back to plaintext send"
                                    );
                                    // fall through to plaintext send below
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                room = %room_id,
                                error = %e,
                                "matrix media: attachment encrypt failed; \
                                 falling back to plaintext send"
                            );
                            // fall through to plaintext send below
                        }
                    }
                }
            }
            let mxc = self
                .upload_media(&att.bytes, &att.mime, att.filename.as_deref())
                .await?;
            let mut content = self.media_event_content(att, &mxc);
            self.attach_relation(&mut content, room_id, message);
            let _ = self
                .send_event(room_id, "m.room.message", content)
                .await?;
        }

        // Empty content with attachments — no text companion needed.
        if message.content.is_empty() && !message.attachments.is_empty() {
            return Ok(());
        }

        let msgtype = message
            .metadata
            .get("matrix_msgtype")
            .map(|s| s.as_str())
            .filter(|s| matches!(*s, "m.text" | "m.emote" | "m.notice"))
            .unwrap_or("m.text");

        let chunks = chunk_text(&message.content, MAX_MESSAGE_LENGTH);
        for chunk in chunks {
            // Wrap @user:server mentions in matrix.to markdown
            // links so HTML rendering produces clickable anchors.
            // Plain-text clients fall back to the raw form, which
            // is still readable.
            let body_for_send = wrap_mention_links(&chunk);
            let mut content = json!({
                "msgtype": msgtype,
                "body": body_for_send,
            });
            if self.config.markdown_to_html && chunk_has_markdown(&body_for_send) {
                let html = markdown_to_html(&body_for_send);
                let obj = content.as_object_mut().unwrap();
                obj.insert("format".into(), json!("org.matrix.custom.html"));
                obj.insert("formatted_body".into(), json!(html));
            }
            // m.mentions for @user references in the body. Run
            // against the original (un-wrapped) chunk so we don't
            // double-count when wrap_mention_links added linkifies.
            let mentions = extract_mentioned_user_ids(&chunk);
            if !mentions.is_empty() {
                content.as_object_mut().unwrap().insert(
                    "m.mentions".into(),
                    json!({ "user_ids": mentions }),
                );
            }
            self.attach_relation(&mut content, room_id, message);
            // Encrypted-room branch: wrap the content via the
            // OlmMachine and emit `m.room.encrypted` instead of
            // `m.room.message`. Falls back to plaintext send if
            // the crypto handle isn't active or the room hasn't
            // been seen as encrypted.
            #[cfg(feature = "matrix-e2ee")]
            if let Some(crypto) = self.crypto_handle() {
                if crypto.is_encrypted(room_id) {
                    let recipients = self.room_member_recipients(room_id);
                    match crypto
                        .encrypt_room_message(room_id, &content, &recipients)
                        .await
                    {
                        Ok(encrypted) => {
                            let _ = self
                                .send_event(room_id, "m.room.encrypted", encrypted)
                                .await?;
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(
                                room = %room_id,
                                error = %e,
                                "matrix encrypt failed; falling back to plaintext send"
                            );
                        }
                    }
                }
            }
            let _ = self.send_event(room_id, "m.room.message", content).await?;
        }
        Ok(())
    }

    /// Push a text-only send into the per-chat buffer and arm a
    /// flush task. Subsequent sends to the same chat within the
    /// `text_batch_delay_ms` window land in the same buffer and
    /// the flush task coalesces them into a single send.
    ///
    /// **Split detection**: when any message in the buffer is
    /// `>=` `BATCH_LONG_THRESHOLD` characters, the flush delay
    /// is doubled. Mirrors the upstream's heuristic — a near-
    /// limit chunk usually means more chunks are coming, so we
    /// wait longer before flushing.
    async fn enqueue_batched(&self, message: &SendMessage) -> Result<()> {
        let chat_id = message.recipient.clone();
        let long_buffered = {
            let mut buffers = self.state.text_batch_buffers.lock();
            let buf = buffers.entry(chat_id.clone()).or_default();
            buf.push(BufferedText {
                content: message.content.clone(),
                metadata: message.metadata.clone(),
                reply_to: message.reply_to.clone(),
            });
            // Evaluate against the full buffer so a long message
            // arriving after a short one still triggers the
            // extended delay.
            buf.iter()
                .any(|b| b.content.chars().count() >= BATCH_LONG_THRESHOLD)
        };
        // Always (re)arm the flush task. When a long message
        // shows up mid-window we abort the in-flight flush and
        // spawn a new one with the doubled delay — that's what
        // extends the deadline.
        let base_delay = self.config.text_batch_delay_ms;
        let actual_delay = if long_buffered {
            base_delay.saturating_mul(2)
        } else {
            base_delay
        };
        if let Some(prev) = self.state.flush_handles.lock().remove(&chat_id) {
            prev.abort();
        }
        let me = self.clone();
        let chat_for_task = chat_id.clone();
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(actual_delay)).await;
            me.flush_chat_buffer(&chat_for_task).await;
        });
        self.state.flush_handles.lock().insert(chat_id, handle);
        Ok(())
    }

    /// Drain the buffer for one chat and dispatch a single
    /// coalesced send. Joining is by `\n\n`; the first buffered
    /// message's reply / thread metadata is reused (subsequent
    /// messages in the same window typically share the same
    /// reply target). Errors during the actual send are logged
    /// and discarded since the caller has already returned.
    async fn flush_chat_buffer(&self, chat_id: &str) {
        let buffered = {
            let mut buffers = self.state.text_batch_buffers.lock();
            buffers.remove(chat_id).unwrap_or_default()
        };
        // Drop the now-completed handle so the next send starts
        // a fresh batching window cleanly.
        self.state.flush_handles.lock().remove(chat_id);
        if buffered.is_empty() {
            return;
        }
        let combined = buffered
            .iter()
            .map(|b| b.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let first = &buffered[0];
        let merged = SendMessage::new(combined, chat_id)
            .with_reply_to(first.reply_to.clone())
            .with_metadata(first.metadata.clone());
        if let Err(e) = self.send_immediate(&merged).await {
            tracing::warn!(
                chat_id = %chat_id,
                error = %e,
                count = buffered.len(),
                "matrix batched send failed"
            );
        }
    }

    // -- E2EE plumbing -----------------------------------------
    //
    // All crypto access goes through `crypto_handle()` which
    // returns `None` when the feature is off OR when the user
    // didn't configure a `crypto_store_dir`. Callers branch on
    // that and fall through to the unencrypted path.

    /// Initialize the crypto layer if both the feature is built
    /// and the user configured a crypto store directory. No-op
    /// otherwise — channel falls back to unencrypted operation.
    #[cfg(feature = "matrix-e2ee")]
    async fn init_crypto(&self) -> Result<()> {
        if self.config.crypto_store_dir.is_empty() {
            return Ok(());
        }
        use crate::channels::matrix_e2ee::{MatrixCrypto, MatrixCryptoConfig};
        let state = self.state.clone();
        let token_provider: Arc<dyn Fn() -> String + Send + Sync> =
            Arc::new(move || state.access_token.lock().clone());
        let cfg = MatrixCryptoConfig {
            homeserver: self.config.homeserver.clone(),
            user_id: self.current_user_id(),
            device_id: if self.config.device_id.is_empty() {
                None
            } else {
                Some(self.config.device_id.clone())
            },
            crypto_store_dir: std::path::PathBuf::from(&self.config.crypto_store_dir),
            crypto_store_passphrase: if self.config.crypto_store_passphrase.is_empty() {
                None
            } else {
                Some(self.config.crypto_store_passphrase.clone())
            },
            http: self.http.clone(),
            token_provider,
        };
        let crypto = MatrixCrypto::new(cfg).await?;
        // Drain whatever the OlmMachine wants to upload at
        // startup (initial keys/upload, cross-signing bootstrap,
        // etc). Best-effort — log on failure but proceed; the
        // sync loop's per-tick drain will retry.
        if let Err(e) = crypto.drain_outgoing_requests().await {
            tracing::warn!(error = %e, "matrix crypto initial drain failed");
        }
        *self.state.crypto.write() = Some(crypto);
        tracing::info!("matrix E2EE initialized");
        Ok(())
    }

    /// Default-build stub — when the feature is off, init is a
    /// no-op so `listen()` can call it unconditionally.
    #[cfg(not(feature = "matrix-e2ee"))]
    async fn init_crypto(&self) -> Result<()> {
        if !self.config.crypto_store_dir.is_empty() {
            tracing::warn!(
                "matrix `crypto_store_dir` is set but the `matrix-e2ee` Cargo feature is disabled; \
                 falling back to unencrypted operation"
            );
        }
        Ok(())
    }

    /// Returns the active crypto handle, if any. None when the
    /// feature is off or the user didn't configure E2EE. Callers
    /// branch on this to decide between encrypted and plaintext
    /// paths.
    #[cfg(feature = "matrix-e2ee")]
    fn crypto_handle(&self) -> Option<crate::channels::matrix_e2ee::MatrixCrypto> {
        self.state.crypto.read().clone()
    }

    /// Stub for default builds — returns a placeholder type that
    /// can never be matched, so `if let Some(_) = crypto_handle()`
    /// is always false and the encrypted path compiles out.
    #[cfg(not(feature = "matrix-e2ee"))]
    fn crypto_handle(&self) -> Option<std::convert::Infallible> {
        None
    }

    /// Called for each room's `state.events` array from a sync
    /// response. Tracks `m.room.encryption` (marks the room as
    /// encrypted) and `m.room.member` (maintains the joined-
    /// members set used by `share_room_key`).
    fn process_state_events(&self, room_id: &str, state_events: &[Value]) {
        for ev in state_events {
            let event_type = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match event_type {
                "m.room.encryption" => {
                    #[cfg(feature = "matrix-e2ee")]
                    if let Some(crypto) = self.crypto_handle() {
                        crypto.mark_encrypted(room_id);
                    }
                }
                "m.room.member" => {
                    let user = ev
                        .get("state_key")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let membership = ev
                        .pointer("/content/membership")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if user.is_empty() {
                        continue;
                    }
                    let mut members = self.state.room_members.lock();
                    let entry = members.entry(room_id.to_string()).or_default();
                    match membership {
                        "join" | "invite" => {
                            entry.insert(user.to_string());
                        }
                        "leave" | "ban" => {
                            entry.remove(user);
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    /// If `event` is `m.room.encrypted` and we have an active
    /// crypto handle, decrypt and return the recovered plaintext
    /// event JSON. Otherwise pass the event through unchanged.
    /// Decrypt failures log and return the original event so
    /// downstream filtering can still drop it cleanly.
    async fn maybe_decrypt(&self, room_id: &str, event: Value) -> Value {
        #[cfg(feature = "matrix-e2ee")]
        {
            let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if event_type != "m.room.encrypted" {
                return event;
            }
            let crypto = match self.crypto_handle() {
                Some(c) => c,
                None => return event,
            };
            match crypto.decrypt(room_id, &event).await {
                Ok(plain) => plain,
                Err(e) => {
                    tracing::debug!(
                        room = %room_id,
                        error = %e,
                        "matrix: decrypt failed; passing encrypted event through"
                    );
                    event
                }
            }
        }
        #[cfg(not(feature = "matrix-e2ee"))]
        {
            let _ = room_id;
            event
        }
    }

    /// Snapshot the joined members of a room. Used as the
    /// recipient list for `share_room_key`. The bot's own user-id
    /// is filtered out — sharing keys with ourselves is wasted
    /// work and the OlmMachine handles same-user device sharing
    /// internally.
    fn room_member_recipients(&self, room_id: &str) -> Vec<String> {
        let me = self.current_user_id();
        self.state
            .room_members
            .lock()
            .get(room_id)
            .map(|s| {
                s.iter()
                    .filter(|u| !u.eq_ignore_ascii_case(&me))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
}

// -- Channel impl ------------------------------------------------

#[async_trait]
impl Channel for MatrixChannel {
    fn name(&self) -> &str {
        "matrix"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        // Batching path: when text_batch_delay_ms is configured
        // and the message is text-only, push into the per-chat
        // buffer and rely on the flush task to dispatch a
        // coalesced send. Attachments always send immediately
        // (binary uploads are too costly to batch).
        if self.config.text_batch_delay_ms > 0
            && message.attachments.is_empty()
            && !message.content.is_empty()
        {
            return self.enqueue_batched(message).await;
        }
        self.send_immediate(message).await
    }

    async fn listen(&self, tx: mpsc::Sender<InboundMessage>) -> Result<()> {
        self.ensure_authenticated().await?;
        self.load_state().await;
        self.init_crypto().await?;
        self.run_sync(tx).await
    }

    async fn send_typing(&self, chat_id: &str) -> Result<()> {
        if self.typing_in_cooldown(chat_id) {
            return Ok(());
        }
        let user_id = self.current_user_id();
        if user_id.is_empty() {
            return Ok(());
        }
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/typing/{}",
            self.base_url(),
            urlencoding::encode(chat_id),
            urlencoding::encode(&user_id),
        );
        let body = json!({ "typing": true, "timeout": TYPING_TIMEOUT_MS });
        match self
            .http
            .put(&url)
            .bearer_auth(self.token())
            .json(&body)
            .timeout(RPC_TIMEOUT)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                self.typing_clear(chat_id);
                Ok(())
            }
            Ok(r) => {
                tracing::debug!(chat_id = %chat_id, status = %r.status(), "matrix typing returned non-2xx");
                self.record_typing_failure(chat_id);
                Ok(())
            }
            Err(e) => {
                tracing::debug!(chat_id = %chat_id, error = %e, "matrix typing request failed");
                self.record_typing_failure(chat_id);
                Ok(())
            }
        }
    }

    fn allows_sender(&self, sender_id: &str) -> bool {
        self.is_dm_sender_allowed(sender_id)
    }
}

// -- Free helpers ------------------------------------------------

/// Apply 0–20% positive jitter to a backoff duration.
fn backoff_with_jitter(d: Duration) -> Duration {
    let mut rng = rand::thread_rng();
    let jitter_pct: f64 = rng.gen_range(0.0..0.2);
    let nanos = (d.as_nanos() as f64 * (1.0 + jitter_pct)) as u128;
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

/// Bridge / appservice senders use the localpart prefix `_` by
/// convention. Filtering them prevents bot-on-bot traffic loops.
fn is_bridge_sender(sender: &str) -> bool {
    let localpart = sender
        .strip_prefix('@')
        .and_then(|s| s.split(':').next())
        .unwrap_or("");
    localpart.starts_with('_')
}

/// Strip the rich-reply fallback Matrix clients prepend to a body
/// (`> <@user:server> original\n\n` lines). Multiple `> ` lines
/// may stack; we drop them until the first non-`> ` line.
fn strip_reply_fallback(body: &str) -> String {
    let mut iter = body.lines();
    let mut consumed = 0usize;
    while let Some(line) = iter.clone().next() {
        if line.starts_with("> ") || line.starts_with(">> ") {
            iter.next();
            consumed += line.len() + 1;
        } else {
            break;
        }
    }
    let rest = body[consumed.min(body.len())..].trim_start_matches('\n').to_string();
    if rest.is_empty() && consumed == 0 {
        body.to_string()
    } else if rest.is_empty() {
        // Body was entirely fallback; keep the original (probably
        // never useful, but more honest than a blank).
        body.to_string()
    } else {
        rest
    }
}

/// Extract `@user:server` mentions from a plain-text body. Used
/// to populate `m.mentions.user_ids` on outbound messages. Valid
/// Matrix user-ids may contain `.` (DNS server names), so we walk
/// to whitespace or `,;!?()` and only then strip a trailing `.`
/// (sentence-ending punctuation that's never part of a server
/// name since RFC 1035 forbids trailing dots).
fn extract_mentioned_user_ids(body: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    let bytes = body.as_bytes();
    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 1;
        while j < bytes.len()
            && !bytes[j].is_ascii_whitespace()
            && !matches!(bytes[j], b',' | b';' | b'!' | b'?' | b'(' | b')')
        {
            j += 1;
        }
        let mut candidate = &body[start..j];
        while candidate.ends_with('.') {
            candidate = &candidate[..candidate.len() - 1];
        }
        if candidate.contains(':') && candidate.len() > 3 {
            if !out.iter().any(|s| s == candidate) {
                out.push(candidate.to_string());
            }
        }
        i = j.max(i + 1);
    }
    out
}

/// Wrap each `@user:server` mention in a body with the canonical
/// matrix.to markdown link form so HTML rendering produces a
/// proper anchor. Skips mentions already inside a markdown link
/// (`[…](url)` constructs) and inside backtick code spans.
///
/// Example:
///   `Hi @alice:example.org` →
///   `Hi [@alice:example.org](https://matrix.to/#/@alice:example.org)`
pub(crate) fn wrap_mention_links(body: &str) -> String {
    if !body.contains('@') {
        return body.to_string();
    }
    let mut out = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0;
    let mut in_code = false;
    // True between `](` and the matching `)` — i.e. inside the
    // URL portion of an existing markdown link. Mentions there
    // are part of the URL itself and must not be re-wrapped.
    let mut in_link_url = false;
    let mut link_url_depth = 0i32;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '`' {
            in_code = !in_code;
            out.push(c);
            i += 1;
            continue;
        }
        if !in_code && !in_link_url && c == ']' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            // Entering an existing markdown link's URL portion.
            in_link_url = true;
            link_url_depth = 0;
            out.push(c);
            out.push('(');
            i += 2;
            continue;
        }
        if in_link_url {
            if c == '(' {
                link_url_depth += 1;
            } else if c == ')' {
                if link_url_depth == 0 {
                    in_link_url = false;
                } else {
                    link_url_depth -= 1;
                }
            }
            out.push(c);
            i += 1;
            continue;
        }
        if in_code || c != '@' {
            out.push(c);
            i += 1;
            continue;
        }
        // Possible mention. Skip if preceded by `[` (already in
        // a markdown link's display text) or by an alphanumeric /
        // `.` (looks like an email local-part such as
        // `alice@example.org`).
        let prev = out.chars().last();
        let skip = match prev {
            Some('[') => true,
            Some(c2) if c2.is_ascii_alphanumeric() || c2 == '.' => true,
            _ => false,
        };
        if skip {
            out.push(c);
            i += 1;
            continue;
        }
        // Walk to token end.
        let start = i;
        let mut j = i + 1;
        while j < bytes.len()
            && !bytes[j].is_ascii_whitespace()
            && !matches!(bytes[j], b',' | b';' | b'!' | b'?' | b'(' | b')' | b'[' | b']')
        {
            j += 1;
        }
        let mut candidate_end = j;
        while candidate_end > start + 1 && bytes[candidate_end - 1] == b'.' {
            candidate_end -= 1;
        }
        let candidate = &body[start..candidate_end];
        if candidate.contains(':') && candidate.len() > 3 {
            // Detect "already inside a markdown link" by checking
            // for `](` immediately after the candidate.
            let after = &body[candidate_end..];
            if after.starts_with("](") {
                out.push_str(candidate);
                i = candidate_end;
                continue;
            }
            out.push('[');
            out.push_str(candidate);
            out.push_str("](https://matrix.to/#/");
            out.push_str(candidate);
            out.push(')');
            i = candidate_end;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Heuristic that detects "transport-style" media filenames —
/// names a Matrix client auto-generates (`IMG_*.jpg`,
/// `VID_*.mp4`, `Screenshot_*.png`, `signal-*.png`, ...) — that
/// shouldn't be surfaced as the message body. The audit's note
/// references the upstream's `_looks_like_matrix_image_filename`.
pub(crate) fn looks_like_transport_filename(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let trimmed = lower.trim();
    if trimmed.is_empty() || trimmed.contains(' ') {
        return false;
    }
    // Common camera / OS auto-naming patterns.
    let prefixes = [
        "img_", "img-", "vid_", "vid-", "video_", "video-",
        "screenshot_", "screenshot-", "screen shot ", "photo_",
        "photo-", "signal-", "whatsapp ", "telegram ", "image_",
        "image-", "audio_", "audio-", "voice_", "voice-",
        "pxl_", "dsc_", "dscn",
    ];
    for p in prefixes {
        if trimmed.starts_with(p) {
            return true;
        }
    }
    // Names that are JUST a UUID + extension, or a long hex string,
    // or "matrix-<…>".
    if trimmed.starts_with("matrix-") {
        return true;
    }
    // Any plain filename with no spaces and one of the common media
    // extensions, that's at least 8 chars before the ext, qualifies.
    let common_exts = [
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov",
        ".m4a", ".mp3", ".ogg", ".wav", ".webm",
    ];
    for ext in common_exts {
        if let Some(stem) = trimmed.strip_suffix(ext) {
            if stem.len() >= 8 && stem.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                return true;
            }
        }
    }
    false
}

/// Map a MIME type to a sensible file extension for the local
/// media cache. Conservative — falls through to `None` when we
/// don't have a clear winner, and the caller picks `bin`.
fn extension_for_mime(mime: &str) -> Option<&'static str> {
    let m = mime.split(';').next().unwrap_or(mime).trim();
    Some(match m {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/heic" | "image/heif" => "heic",
        "image/svg+xml" => "svg",
        "audio/mp4" | "audio/aac" => "m4a",
        "audio/mpeg" => "mp3",
        "audio/ogg" => "ogg",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/webm" => "weba",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        "application/pdf" => "pdf",
        "application/json" => "json",
        "application/zip" => "zip",
        "text/plain" => "txt",
        "text/markdown" => "md",
        _ => return None,
    })
}

/// Slice a body into chunks no larger than `max_chars` characters.
/// Splits on whitespace boundaries when possible; falls back to a
/// hard cut at the limit.
fn chunk_text(input: &str, max_chars: usize) -> Vec<String> {
    if input.is_empty() {
        return vec![String::new()];
    }
    if input.chars().count() <= max_chars {
        return vec![input.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for word in input.split_inclusive(char::is_whitespace) {
        if buf.chars().count() + word.chars().count() > max_chars {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            // Word itself longer than max_chars — hard split.
            if word.chars().count() > max_chars {
                let mut chunk = String::new();
                for c in word.chars() {
                    if chunk.chars().count() == max_chars {
                        out.push(std::mem::take(&mut chunk));
                    }
                    chunk.push(c);
                }
                if !chunk.is_empty() {
                    buf.push_str(&chunk);
                }
            } else {
                buf.push_str(word);
            }
        } else {
            buf.push_str(word);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Cheap heuristic: does this chunk contain markdown that's
/// worth converting? Skips the HTML pass for purely plain text
/// to keep small messages cheap.
fn chunk_has_markdown(chunk: &str) -> bool {
    chunk.contains("**")
        || chunk.contains("__")
        || chunk.contains('*')
        || chunk.contains('_')
        || chunk.contains('`')
        || chunk.contains("~~")
        || chunk.contains('#')
        || chunk.contains('[')
}

/// Convert markdown to Matrix-compatible HTML
/// (`org.matrix.custom.html`). Handles fenced code blocks, inline
/// code, bold, italic, strikethrough, links, headings, and
/// paragraphs. Output is stable, predictable HTML — clients
/// sanitize anyway, so we keep the tag set conservative.
fn markdown_to_html(input: &str) -> String {
    // Step 1: pull out fenced code blocks; they're rendered as
    // <pre><code>…</code></pre> verbatim and shouldn't have
    // inline markers reinterpreted.
    let runs = tokenize_md_blocks(input);
    let mut out = String::new();
    for run in runs {
        match run {
            MdBlock::Fence(body) => {
                out.push_str("<pre><code>");
                out.push_str(&html_escape(&body));
                out.push_str("</code></pre>");
            }
            MdBlock::Text(body) => {
                out.push_str(&render_md_text(&body));
            }
        }
    }
    out
}

enum MdBlock {
    Fence(String),
    Text(String),
}

fn tokenize_md_blocks(input: &str) -> Vec<MdBlock> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '`' && chars.peek() == Some(&'`') {
            // Possibly triple-backtick.
            chars.next();
            if chars.peek() == Some(&'`') {
                chars.next();
                // Consume optional language tag up to newline.
                while let Some(&n) = chars.peek() {
                    chars.next();
                    if n == '\n' {
                        break;
                    }
                }
                let mut body = String::new();
                let mut closed = false;
                while let Some(&n) = chars.peek() {
                    if n == '`' {
                        let mut count = 0;
                        let mut tmp = String::new();
                        while chars.peek() == Some(&'`') {
                            chars.next();
                            count += 1;
                            tmp.push('`');
                            if count == 3 {
                                break;
                            }
                        }
                        if count == 3 {
                            closed = true;
                            break;
                        } else {
                            body.push_str(&tmp);
                        }
                    } else {
                        body.push(n);
                        chars.next();
                    }
                }
                if closed {
                    if !buf.is_empty() {
                        out.push(MdBlock::Text(std::mem::take(&mut buf)));
                    }
                    out.push(MdBlock::Fence(
                        body.strip_prefix('\n').map(String::from).unwrap_or(body),
                    ));
                    continue;
                } else {
                    // Unclosed fence — treat literally.
                    buf.push_str("```");
                    buf.push_str(&body);
                }
            } else {
                buf.push_str("``");
            }
        } else {
            buf.push(c);
        }
    }
    if !buf.is_empty() {
        out.push(MdBlock::Text(buf));
    }
    out
}

fn render_md_text(input: &str) -> String {
    let mut out = String::new();
    for line in input.split_inclusive('\n') {
        let (body, trailing) = match line.strip_suffix('\n') {
            Some(b) => (b, "\n"),
            None => (line, ""),
        };
        // Headings: `# `..`###### ` at line start.
        let hash_count = body.chars().take_while(|c| *c == '#').count();
        let after_hashes = body.chars().nth(hash_count);
        if (1..=6).contains(&hash_count) && after_hashes == Some(' ') {
            let rest: String = body.chars().skip(hash_count + 1).collect();
            out.push_str(&format!("<h{0}>", hash_count));
            out.push_str(&render_md_inline(&rest));
            out.push_str(&format!("</h{0}>", hash_count));
        } else {
            out.push_str(&render_md_inline(body));
        }
        out.push_str(trailing);
    }
    out
}

fn render_md_inline(input: &str) -> String {
    // Inline code first (so other markers don't apply inside).
    let runs = tokenize_inline_code(input);
    let mut out = String::new();
    for run in runs {
        match run {
            InlineRun::Code(body) => {
                out.push_str("<code>");
                out.push_str(&html_escape(&body));
                out.push_str("</code>");
            }
            InlineRun::Plain(body) => {
                out.push_str(&apply_md_marks(&body));
            }
        }
    }
    out
}

enum InlineRun {
    Code(String),
    Plain(String),
}

fn tokenize_inline_code(input: &str) -> Vec<InlineRun> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '`' {
            // Find matching backtick.
            let mut body = String::new();
            let mut closed = false;
            while let Some(&n) = chars.peek() {
                if n == '`' {
                    chars.next();
                    closed = true;
                    break;
                }
                body.push(n);
                chars.next();
            }
            if closed && !body.is_empty() {
                if !buf.is_empty() {
                    out.push(InlineRun::Plain(std::mem::take(&mut buf)));
                }
                out.push(InlineRun::Code(body));
            } else {
                buf.push('`');
                buf.push_str(&body);
            }
        } else {
            buf.push(c);
        }
    }
    if !buf.is_empty() {
        out.push(InlineRun::Plain(buf));
    }
    out
}

fn apply_md_marks(input: &str) -> String {
    // HTML-escape first; markers are ASCII so escaping doesn't
    // hide them. Then apply each marker pattern.
    let escaped = html_escape(input);
    let mut s = escaped;
    s = wrap_marker(&s, "**", "<strong>", "</strong>");
    s = wrap_marker(&s, "__", "<strong>", "</strong>");
    s = wrap_marker(&s, "~~", "<del>", "</del>");
    s = wrap_marker(&s, "*", "<em>", "</em>");
    s = wrap_marker(&s, "_", "<em>", "</em>");
    s = render_md_links(&s);
    s
}

/// Replace each `<open>...<close>` pair (well-formed, non-empty
/// content) with `<htag>...</htag>`. Mismatched markers are left
/// alone.
fn wrap_marker(input: &str, marker: &str, htag_open: &str, htag_close: &str) -> String {
    let m = marker;
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        let Some(start) = rest.find(m) else {
            out.push_str(rest);
            break;
        };
        let after_open = &rest[start + m.len()..];
        let Some(end_rel) = after_open.find(m) else {
            out.push_str(&rest[..start + m.len()]);
            rest = after_open;
            continue;
        };
        let inner = &after_open[..end_rel];
        if inner.is_empty() {
            out.push_str(&rest[..start + m.len()]);
            rest = after_open;
            continue;
        }
        out.push_str(&rest[..start]);
        out.push_str(htag_open);
        out.push_str(inner);
        out.push_str(htag_close);
        rest = &after_open[end_rel + m.len()..];
    }
    out
}

fn render_md_links(input: &str) -> String {
    // [text](url) → <a href="url">text</a>
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        let Some(open) = rest.find('[') else {
            out.push_str(rest);
            break;
        };
        let after_open = &rest[open + 1..];
        let Some(close) = after_open.find(']') else {
            out.push_str(&rest[..open + 1]);
            rest = after_open;
            continue;
        };
        let after_close = &after_open[close + 1..];
        if !after_close.starts_with('(') {
            out.push_str(&rest[..open + 1]);
            rest = after_open;
            continue;
        }
        let url_rest = &after_close[1..];
        let Some(paren_close) = url_rest.find(')') else {
            out.push_str(&rest[..open + 1]);
            rest = after_open;
            continue;
        };
        let text = &after_open[..close];
        let url = &url_rest[..paren_close];
        out.push_str(&rest[..open]);
        out.push_str(&format!(
            "<a href=\"{}\">{}</a>",
            html_escape_attr(url),
            text
        ));
        rest = &url_rest[paren_close + 1..];
    }
    out
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

fn html_escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Convert a Matrix `mxc://server/mediaId` URI to an authenticated
/// HTTP download URL (Matrix v1.11+ moved unauthenticated media
/// endpoints under the client-server tree). Returns `None` if the
/// input isn't a well-formed mxc URI.
pub fn mxc_to_http_url(homeserver: &str, mxc: &str) -> Option<String> {
    let after = mxc.strip_prefix("mxc://")?;
    let (server, media_id) = after.split_once('/')?;
    if server.is_empty() || media_id.is_empty() {
        return None;
    }
    Some(format!(
        "{}/_matrix/client/v1/media/download/{}/{}",
        homeserver.trim_end_matches('/'),
        urlencoding::encode(server),
        urlencoding::encode(media_id),
    ))
}

// -- Tests -------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MatrixChannelEntry {
        MatrixChannelEntry {
            enabled: true,
            homeserver: "https://matrix.example.org".into(),
            access_token: "tok".into(),
            user_id: "@bot:example.org".into(),
            password: String::new(),
            device_id: String::new(),
            allowed_users: Vec::new(),
            allowed_rooms: Vec::new(),
            free_response_rooms: Vec::new(),
            require_mention: true,
            auto_thread: true,
            dm_auto_thread: false,
            dm_mention_threads: false,
            markdown_to_html: true,
            media_cache_dir: String::new(),
            state_file: String::new(),
            text_batch_delay_ms: 0,
            crypto_store_dir: String::new(),
            crypto_store_passphrase: String::new(),
            home_chat_id: String::new(),
        }
    }

    fn channel() -> MatrixChannel {
        MatrixChannel::from_config(&cfg()).unwrap()
    }

    // -- from_config ---------------------------------------------

    #[test]
    fn from_config_disabled_returns_none() {
        let mut c = cfg();
        c.enabled = false;
        assert!(MatrixChannel::from_config(&c).is_none());
    }

    #[test]
    fn from_config_missing_homeserver_returns_none() {
        let mut c = cfg();
        c.homeserver = String::new();
        assert!(MatrixChannel::from_config(&c).is_none());
    }

    #[test]
    fn from_config_missing_user_id_returns_none() {
        let mut c = cfg();
        c.user_id = String::new();
        assert!(MatrixChannel::from_config(&c).is_none());
    }

    #[test]
    fn from_config_missing_token_and_password_returns_none() {
        let mut c = cfg();
        c.access_token = String::new();
        c.password = String::new();
        assert!(MatrixChannel::from_config(&c).is_none());
    }

    #[test]
    fn name_is_matrix() {
        let ch = channel();
        assert_eq!(ch.name(), "matrix");
    }

    // -- Allowlists ----------------------------------------------

    #[test]
    fn dm_allowlist_empty_denies_all() {
        // Default-deny: an unconfigured allowlist must refuse everyone;
        // "*" is the explicit allow-everyone opt-in.
        let ch = channel();
        assert!(!ch.is_dm_sender_allowed("@alice:example.org"));
    }

    #[test]
    fn dm_allowlist_wildcard_allows_all() {
        let mut c = cfg();
        c.allowed_users = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        assert!(ch.is_dm_sender_allowed("@anyone:example.org"));
    }

    #[test]
    fn dm_allowlist_filters() {
        let mut c = cfg();
        c.allowed_users = vec!["@alice:example.org".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        assert!(ch.is_dm_sender_allowed("@alice:example.org"));
        assert!(ch.is_dm_sender_allowed("@ALICE:EXAMPLE.ORG"));
        assert!(!ch.is_dm_sender_allowed("@bob:example.org"));
    }

    #[test]
    fn room_allowlist_empty_disables_rooms() {
        let ch = channel();
        assert!(!ch.is_room_allowed("!any:example.org"));
    }

    #[test]
    fn room_allowlist_wildcard() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        assert!(ch.is_room_allowed("!a:example.org"));
        assert!(ch.is_room_allowed("!b:example.org"));
    }

    #[test]
    fn room_allowlist_explicit() {
        let mut c = cfg();
        c.allowed_rooms = vec!["!a:example.org".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        assert!(ch.is_room_allowed("!a:example.org"));
        assert!(!ch.is_room_allowed("!b:example.org"));
    }

    // -- Bridge sender filter -----------------------------------

    #[test]
    fn bridge_sender_with_underscore_prefix() {
        assert!(is_bridge_sender("@_irc_user:example.org"));
        assert!(!is_bridge_sender("@alice:example.org"));
        assert!(!is_bridge_sender("@_:example.org") == false); // explicit edge: localpart "_"
    }

    // -- Reply-fallback strip -----------------------------------

    #[test]
    fn strip_reply_fallback_removes_quote() {
        let body = "> <@alice:example.org> hi\n\nactual reply";
        let stripped = strip_reply_fallback(body);
        assert_eq!(stripped, "actual reply");
    }

    #[test]
    fn strip_reply_fallback_passthrough_when_no_quote() {
        let body = "plain message";
        assert_eq!(strip_reply_fallback(body), "plain message");
    }

    // -- Mention extraction -------------------------------------

    #[test]
    fn extract_mentions_finds_user_ids() {
        let body = "hi @alice:example.org and @bob:server.io check this";
        let m = extract_mentioned_user_ids(body);
        assert_eq!(m, vec!["@alice:example.org", "@bob:server.io"]);
    }

    #[test]
    fn extract_mentions_skips_emails_and_bare_at() {
        let body = "email me @ alice@example.org";
        let m = extract_mentioned_user_ids(body);
        // `alice@example.org` doesn't start with `@` so isn't a Matrix mention.
        // The bare `@` followed by space is filtered (no colon).
        assert!(m.is_empty(), "expected empty, got {:?}", m);
    }

    #[test]
    fn extract_mentions_dedups_repeats() {
        let body = "@alice:example.org said something to @alice:example.org";
        let m = extract_mentioned_user_ids(body);
        assert_eq!(m.len(), 1);
    }

    // -- Chunking -----------------------------------------------

    #[test]
    fn chunk_text_short_passthrough() {
        assert_eq!(chunk_text("hello", 100), vec!["hello"]);
    }

    #[test]
    fn chunk_text_splits_long_at_word_boundary() {
        let body = "aaa bbb ccc ddd eee fff";
        let chunks = chunk_text(body, 7);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= 7 + 4, "chunk too long: {:?}", c);
        }
        assert_eq!(chunks.concat().replace(' ', "").replace('\n', ""), "aaabbbcccdddeeefff");
    }

    #[test]
    fn chunk_text_handles_overlong_word() {
        let huge = "a".repeat(50);
        let chunks = chunk_text(&huge, 10);
        assert_eq!(chunks.len(), 5);
        for c in &chunks {
            assert_eq!(c.chars().count(), 10);
        }
    }

    // -- Dedup ---------------------------------------------------

    #[test]
    fn dedup_cache_drops_repeats() {
        let mut c = DedupCache::default();
        assert!(c.insert("a"));
        assert!(!c.insert("a"));
        assert!(c.insert("b"));
    }

    #[test]
    fn dedup_cache_evicts_oldest() {
        let mut c = DedupCache::default();
        for i in 0..(DEDUP_CAP + 5) {
            c.insert(&format!("e{}", i));
        }
        assert!(!c.set.contains("e0"));
        assert!(c.set.contains(&format!("e{}", DEDUP_CAP + 4)));
    }

    // -- Markdown → HTML ----------------------------------------

    #[test]
    fn markdown_plain_returns_plain() {
        let html = markdown_to_html("hello world");
        assert_eq!(html, "hello world");
    }

    #[test]
    fn markdown_bold_double_asterisk() {
        assert_eq!(markdown_to_html("hi **bold** there"), "hi <strong>bold</strong> there");
    }

    #[test]
    fn markdown_italic_single_asterisk() {
        assert_eq!(markdown_to_html("an *italic* word"), "an <em>italic</em> word");
    }

    #[test]
    fn markdown_strikethrough() {
        assert_eq!(markdown_to_html("~~old~~"), "<del>old</del>");
    }

    #[test]
    fn markdown_inline_code_escapes() {
        let html = markdown_to_html("run `<x>`");
        assert_eq!(html, "run <code>&lt;x&gt;</code>");
    }

    #[test]
    fn markdown_fenced_code_block() {
        let html = markdown_to_html("```\nfn x() {}\n```");
        assert_eq!(html, "<pre><code>fn x() {}\n</code></pre>");
    }

    #[test]
    fn markdown_heading() {
        let html = markdown_to_html("# Title\nbody");
        assert!(html.contains("<h1>Title</h1>"));
    }

    #[test]
    fn markdown_link() {
        let html = markdown_to_html("see [docs](https://example.com)");
        assert_eq!(html, "see <a href=\"https://example.com\">docs</a>");
    }

    #[test]
    fn markdown_escapes_html_chars() {
        let html = markdown_to_html("a < b & c > d");
        assert_eq!(html, "a &lt; b &amp; c &gt; d");
    }

    #[test]
    fn markdown_inside_code_not_styled() {
        let html = markdown_to_html("`*not bold*`");
        assert_eq!(html, "<code>*not bold*</code>");
    }

    // -- mxc URL parsing ----------------------------------------

    #[test]
    fn mxc_to_http_url_round_trip() {
        let url = mxc_to_http_url("https://matrix.org", "mxc://matrix.org/abc123").unwrap();
        assert_eq!(
            url,
            "https://matrix.org/_matrix/client/v1/media/download/matrix.org/abc123"
        );
    }

    #[test]
    fn mxc_to_http_url_strips_trailing_slash() {
        let url = mxc_to_http_url("https://matrix.org/", "mxc://matrix.org/abc").unwrap();
        assert!(url.starts_with("https://matrix.org/_matrix"));
    }

    #[test]
    fn mxc_to_http_url_rejects_garbage() {
        assert!(mxc_to_http_url("https://matrix.org", "not-a-mxc").is_none());
        assert!(mxc_to_http_url("https://matrix.org", "mxc://").is_none());
        assert!(mxc_to_http_url("https://matrix.org", "mxc://server").is_none());
    }

    // -- Event handling -----------------------------------------

    fn evt(json: Value) -> Value {
        json
    }

    #[test]
    fn handle_event_drops_self_message() {
        let ch = channel();
        // Manually advance started_at so the test event passes the
        // grace window.
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$1",
            "sender": "@bot:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.text", "body": "hi" }
        }));
        assert!(ch.handle_event("!room:example.org", &event).is_none());
    }

    #[test]
    fn handle_event_drops_dup_event_id() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$dup",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.text", "body": "first" }
        }));
        assert!(ch.handle_event("!room:example.org", &event).is_some());
        assert!(ch.handle_event("!room:example.org", &event).is_none());
    }

    #[test]
    fn handle_event_drops_bridge_sender() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$bridge",
            "sender": "@_irc_relay:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.text", "body": "from bridge" }
        }));
        assert!(ch.handle_event("!room:example.org", &event).is_none());
    }

    #[test]
    fn handle_event_drops_notice() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$n",
            "sender": "@otherbot:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.notice", "body": "automated" }
        }));
        assert!(ch.handle_event("!room:example.org", &event).is_none());
    }

    #[test]
    fn handle_event_drops_edit() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$e",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.text",
                "body": "edited body",
                "m.relates_to": {
                    "rel_type": "m.replace",
                    "event_id": "$orig"
                }
            }
        }));
        assert!(ch.handle_event("!room:example.org", &event).is_none());
    }

    #[test]
    fn handle_event_drops_room_when_not_allowed() {
        let ch = channel();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$r",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.text", "body": "hi" }
        }));
        // No allowed_rooms set → not allowed.
        assert!(ch.handle_event("!room:example.org", &event).is_none());
    }

    #[test]
    fn handle_event_passes_dm_with_wildcard_allowlist() {
        let mut c = cfg();
        c.allowed_users = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        // Mark room as DM.
        ch.state.dm_rooms.lock().insert("!dm:example.org".into());
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$dm",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.text", "body": "hi" }
        }));
        let inbound = ch.handle_event("!dm:example.org", &event).expect("kept");
        assert_eq!(inbound.content, "hi");
        assert_eq!(inbound.chat_id, "!dm:example.org");
        assert_eq!(inbound.metadata.get("matrix_dm").map(String::as_str), Some("true"));
    }

    #[test]
    fn handle_event_drops_room_message_without_mention() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = true;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$m",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.text", "body": "no mention here" }
        }));
        assert!(ch.handle_event("!room:example.org", &event).is_none());
    }

    #[test]
    fn handle_event_passes_room_with_mentions_metadata() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$mm",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.text",
                "body": "hi bot",
                "m.mentions": { "user_ids": ["@bot:example.org"] }
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &event).expect("kept");
        assert_eq!(inbound.metadata.get("matrix_mentioned").map(String::as_str), Some("true"));
    }

    #[test]
    fn handle_event_passes_free_response_room_without_mention() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.free_response_rooms = vec!["!free:example.org".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$f",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": { "msgtype": "m.text", "body": "no mention" }
        }));
        let inbound = ch.handle_event("!free:example.org", &event).expect("kept");
        assert_eq!(inbound.content, "no mention");
    }

    #[test]
    fn handle_event_extracts_thread_id() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$t",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.text",
                "body": "in thread",
                "m.relates_to": {
                    "rel_type": "m.thread",
                    "event_id": "$root"
                }
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &event).expect("kept");
        assert_eq!(inbound.metadata.get("matrix_thread_id").map(String::as_str), Some("$root"));
    }

    #[test]
    fn handle_event_extracts_reply_to() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$r2",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.text",
                "body": "reply body",
                "m.relates_to": {
                    "m.in_reply_to": { "event_id": "$orig" }
                }
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &event).expect("kept");
        assert_eq!(inbound.reply_to.as_deref(), Some("$orig"));
    }

    #[test]
    fn handle_event_carries_image_metadata() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let event = evt(json!({
            "type": "m.room.message",
            "event_id": "$img",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.image",
                "body": "photo.jpg",
                "url": "mxc://example.org/abc",
                "info": { "mimetype": "image/jpeg" }
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &event).expect("kept");
        assert_eq!(
            inbound.metadata.get("matrix_media_mxc").map(String::as_str),
            Some("mxc://example.org/abc")
        );
        assert_eq!(
            inbound.metadata.get("matrix_media_mime").map(String::as_str),
            Some("image/jpeg")
        );
    }

    // -- Typing indicator backoff -------------------------------

    #[test]
    fn typing_failure_below_threshold_no_cooldown() {
        let ch = channel();
        ch.record_typing_failure("!a:example.org");
        ch.record_typing_failure("!a:example.org");
        let map = ch.state.typing_failures.lock();
        let entry = map.get("!a:example.org").unwrap();
        assert_eq!(entry.consecutive_failures, 2);
        assert!(entry.cooldown_until.is_none());
    }

    #[test]
    fn typing_failure_threshold_arms_initial_backoff() {
        let ch = channel();
        for _ in 0..3 {
            ch.record_typing_failure("!a:example.org");
        }
        let map = ch.state.typing_failures.lock();
        let entry = map.get("!a:example.org").unwrap();
        assert_eq!(entry.last_backoff, TYPING_BACKOFF_INITIAL);
        assert!(entry.cooldown_until.is_some());
    }

    #[test]
    fn typing_failure_doubles_then_caps() {
        let ch = channel();
        for _ in 0..20 {
            ch.record_typing_failure("!a:example.org");
        }
        let map = ch.state.typing_failures.lock();
        let entry = map.get("!a:example.org").unwrap();
        assert_eq!(entry.last_backoff, TYPING_BACKOFF_MAX);
    }

    #[test]
    fn typing_clear_resets_state() {
        let ch = channel();
        for _ in 0..3 {
            ch.record_typing_failure("!a:example.org");
        }
        assert!(ch.typing_in_cooldown("!a:example.org"));
        ch.typing_clear("!a:example.org");
        assert!(!ch.typing_in_cooldown("!a:example.org"));
    }

    // -- DM cache from account_data -----------------------------

    #[test]
    fn refresh_dm_rooms_populates_cache() {
        let ch = channel();
        let account = json!({
            "events": [{
                "type": "m.direct",
                "content": {
                    "@alice:example.org": ["!dm1:example.org", "!dm2:example.org"]
                }
            }]
        });
        ch.refresh_dm_rooms(&account);
        let dm = ch.state.dm_rooms.lock();
        assert!(dm.contains("!dm1:example.org"));
        assert!(dm.contains("!dm2:example.org"));
    }

    // -- Backoff jitter -----------------------------------------

    #[test]
    fn backoff_jitter_within_bounds() {
        let base = Duration::from_secs(10);
        for _ in 0..100 {
            let j = backoff_with_jitter(base);
            assert!(j >= base);
            assert!(j <= base + base / 5 + Duration::from_millis(1));
        }
    }

    // -- Reaction events ----------------------------------------

    #[test]
    fn handle_event_reaction_surfaces() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        let ev = evt(json!({
            "type": "m.reaction",
            "event_id": "$rx",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": "$prompt",
                    "key": "✅"
                }
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &ev).expect("kept");
        assert_eq!(inbound.content, "[reaction] ✅");
        assert_eq!(
            inbound.metadata.get("matrix_reaction_target").map(String::as_str),
            Some("$prompt")
        );
        assert_eq!(
            inbound.metadata.get("matrix_reaction_key").map(String::as_str),
            Some("✅")
        );
        assert_eq!(
            inbound.metadata.get("matrix_msgtype").map(String::as_str),
            Some("m.reaction")
        );
    }

    #[test]
    fn handle_event_reaction_drops_self() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        let ev = evt(json!({
            "type": "m.reaction",
            "event_id": "$rx2",
            "sender": "@bot:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": "$prompt",
                    "key": "✅"
                }
            }
        }));
        assert!(ch.handle_event("!room:example.org", &ev).is_none());
    }

    #[test]
    fn handle_event_reaction_drops_disallowed_room() {
        let ch = channel(); // empty allowed_rooms
        let ev = evt(json!({
            "type": "m.reaction",
            "event_id": "$rx3",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "m.relates_to": {
                    "rel_type": "m.annotation",
                    "event_id": "$prompt",
                    "key": "✅"
                }
            }
        }));
        assert!(ch.handle_event("!room:example.org", &ev).is_none());
    }

    #[test]
    fn handle_event_reaction_requires_annotation_rel_type() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        let ev = evt(json!({
            "type": "m.reaction",
            "event_id": "$rx4",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "m.relates_to": {
                    "rel_type": "m.thread",
                    "event_id": "$x"
                }
            }
        }));
        assert!(ch.handle_event("!room:example.org", &ev).is_none());
    }

    // -- Media inbound expansion --------------------------------

    #[test]
    fn handle_event_image_with_empty_body_still_surfaces() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let ev = evt(json!({
            "type": "m.room.message",
            "event_id": "$img",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.image",
                "body": "",
                "url": "mxc://example.org/abc"
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &ev).expect("kept");
        assert_eq!(
            inbound.metadata.get("matrix_media_kind").map(String::as_str),
            Some("image")
        );
    }

    #[test]
    fn handle_event_media_full_metadata() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let ev = evt(json!({
            "type": "m.room.message",
            "event_id": "$vid",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.video",
                "body": "clip.mp4",
                "url": "mxc://example.org/vid",
                "info": {
                    "mimetype": "video/mp4",
                    "size": 4096,
                    "w": 1280,
                    "h": 720,
                    "duration": 12000
                }
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &ev).expect("kept");
        assert_eq!(inbound.metadata.get("matrix_media_kind").map(String::as_str), Some("video"));
        assert_eq!(inbound.metadata.get("matrix_media_size").map(String::as_str), Some("4096"));
        assert_eq!(inbound.metadata.get("matrix_media_width").map(String::as_str), Some("1280"));
        assert_eq!(inbound.metadata.get("matrix_media_height").map(String::as_str), Some("720"));
        assert_eq!(inbound.metadata.get("matrix_media_duration_ms").map(String::as_str), Some("12000"));
    }

    // -- pick_thread_anchor -------------------------------------

    #[test]
    fn thread_anchor_room_with_auto_thread_uses_reply_to() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        let ch = MatrixChannel::from_config(&c).unwrap();
        let msg = SendMessage::new("hi", "!room:example.org")
            .with_reply_to(Some("$inbound".into()));
        assert_eq!(
            ch.pick_thread_anchor("!room:example.org", &msg),
            Some("$inbound".into())
        );
    }

    #[test]
    fn thread_anchor_room_off_when_auto_thread_disabled() {
        let mut c = cfg();
        c.auto_thread = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let msg = SendMessage::new("hi", "!room:example.org")
            .with_reply_to(Some("$x".into()));
        assert_eq!(ch.pick_thread_anchor("!room:example.org", &msg), None);
    }

    #[test]
    fn thread_anchor_dm_off_when_dm_auto_thread_disabled() {
        let mut c = cfg();
        c.dm_auto_thread = false;
        c.dm_mention_threads = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        ch.state.dm_rooms.lock().insert("!dm:example.org".into());
        let msg = SendMessage::new("hi", "!dm:example.org")
            .with_reply_to(Some("$x".into()));
        assert_eq!(ch.pick_thread_anchor("!dm:example.org", &msg), None);
    }

    #[test]
    fn thread_anchor_dm_on_with_dm_auto_thread() {
        let mut c = cfg();
        c.dm_auto_thread = true;
        let ch = MatrixChannel::from_config(&c).unwrap();
        ch.state.dm_rooms.lock().insert("!dm:example.org".into());
        let msg = SendMessage::new("hi", "!dm:example.org")
            .with_reply_to(Some("$x".into()));
        assert_eq!(
            ch.pick_thread_anchor("!dm:example.org", &msg),
            Some("$x".into())
        );
    }

    #[test]
    fn thread_anchor_dm_mention_threads_only_when_mentioned() {
        let mut c = cfg();
        c.dm_auto_thread = false;
        c.dm_mention_threads = true;
        let ch = MatrixChannel::from_config(&c).unwrap();
        ch.state.dm_rooms.lock().insert("!dm:example.org".into());
        // Without mention metadata: no thread.
        let msg_no_mention = SendMessage::new("hi", "!dm:example.org")
            .with_reply_to(Some("$x".into()));
        assert_eq!(
            ch.pick_thread_anchor("!dm:example.org", &msg_no_mention),
            None
        );
        // With mention metadata: thread.
        let mut meta = std::collections::HashMap::new();
        meta.insert("matrix_mentioned".into(), "true".into());
        let msg_mentioned = SendMessage::new("hi", "!dm:example.org")
            .with_reply_to(Some("$x".into()))
            .with_metadata(meta);
        assert_eq!(
            ch.pick_thread_anchor("!dm:example.org", &msg_mentioned),
            Some("$x".into())
        );
    }

    #[test]
    fn thread_anchor_explicit_metadata_wins() {
        let mut c = cfg();
        c.auto_thread = false;
        c.dm_auto_thread = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let mut meta = std::collections::HashMap::new();
        meta.insert("matrix_thread_id".into(), "$pinned".into());
        let msg = SendMessage::new("hi", "!room:example.org").with_metadata(meta);
        // Even with both auto_thread flags off, explicit metadata
        // forces a thread anchor.
        assert_eq!(
            ch.pick_thread_anchor("!room:example.org", &msg),
            Some("$pinned".into())
        );
    }

    // -- media_event_content shape ------------------------------

    #[test]
    fn media_event_content_image_shape() {
        let ch = channel();
        let att = crate::bus::MediaAttachment {
            kind: crate::bus::MediaKind::Image,
            bytes: vec![0; 64],
            mime: "image/png".into(),
            filename: Some("shot.png".into()),
            width: Some(800),
            height: Some(600),
            duration_ms: None,
        };
        let v = ch.media_event_content(&att, "mxc://example.org/abc");
        assert_eq!(v["msgtype"], "m.image");
        assert_eq!(v["body"], "shot.png");
        assert_eq!(v["url"], "mxc://example.org/abc");
        assert_eq!(v["info"]["mimetype"], "image/png");
        assert_eq!(v["info"]["size"], 64);
        assert_eq!(v["info"]["w"], 800);
        assert_eq!(v["info"]["h"], 600);
        assert!(v["info"].get("duration").is_none());
    }

    #[test]
    fn media_event_content_audio_carries_duration() {
        let ch = channel();
        let att = crate::bus::MediaAttachment {
            kind: crate::bus::MediaKind::Audio,
            bytes: vec![0; 100],
            mime: "audio/mp4".into(),
            filename: Some("voice.m4a".into()),
            width: None,
            height: None,
            duration_ms: Some(5000),
        };
        let v = ch.media_event_content(&att, "mxc://example.org/aa");
        assert_eq!(v["msgtype"], "m.audio");
        assert_eq!(v["info"]["duration"], 5000);
    }

    #[test]
    fn media_event_content_file_default_body() {
        let ch = channel();
        let att = crate::bus::MediaAttachment {
            kind: crate::bus::MediaKind::File,
            bytes: vec![0; 8],
            mime: "application/pdf".into(),
            filename: None,
            width: None,
            height: None,
            duration_ms: None,
        };
        let v = ch.media_event_content(&att, "mxc://example.org/f");
        assert_eq!(v["msgtype"], "m.file");
        assert_eq!(v["body"], "file");
    }

    // -- extension_for_mime -------------------------------------

    #[test]
    fn extension_for_mime_known_types() {
        assert_eq!(extension_for_mime("image/png"), Some("png"));
        assert_eq!(extension_for_mime("image/jpeg"), Some("jpg"));
        assert_eq!(extension_for_mime("audio/mp4"), Some("m4a"));
        assert_eq!(extension_for_mime("video/mp4"), Some("mp4"));
        assert_eq!(extension_for_mime("application/pdf"), Some("pdf"));
    }

    #[test]
    fn extension_for_mime_handles_charset_suffix() {
        assert_eq!(
            extension_for_mime("text/plain; charset=utf-8"),
            Some("txt")
        );
    }

    #[test]
    fn extension_for_mime_unknown_returns_none() {
        assert_eq!(extension_for_mime("application/x-weirdo"), None);
    }

    // -- last_sent_event_id -------------------------------------

    #[test]
    fn last_sent_event_id_initially_none() {
        let ch = channel();
        assert!(ch.last_sent_event_id("!any:example.org").is_none());
    }

    // -- wrap_mention_links -------------------------------------

    #[test]
    fn wrap_mention_links_basic() {
        let s = wrap_mention_links("hi @alice:example.org how are you");
        assert_eq!(
            s,
            "hi [@alice:example.org](https://matrix.to/#/@alice:example.org) how are you"
        );
    }

    #[test]
    fn wrap_mention_links_strips_trailing_dot() {
        let s = wrap_mention_links("ping @alice:example.org.");
        assert!(s.contains("[@alice:example.org](https://matrix.to/#/@alice:example.org)"));
        assert!(s.ends_with("."));
    }

    #[test]
    fn wrap_mention_links_skips_email() {
        // "alice@example.org" — `@` preceded by alphanumeric, looks
        // like an email local-part, not a Matrix mention.
        let s = wrap_mention_links("contact alice@example.org for details");
        assert_eq!(s, "contact alice@example.org for details");
    }

    #[test]
    fn wrap_mention_links_skips_already_in_link() {
        let s = wrap_mention_links(
            "[@bob:server](https://matrix.to/#/@bob:server) said hi",
        );
        assert_eq!(
            s,
            "[@bob:server](https://matrix.to/#/@bob:server) said hi"
        );
    }

    #[test]
    fn wrap_mention_links_skips_inside_code_span() {
        let s = wrap_mention_links("see `@notamention:server` token");
        assert_eq!(s, "see `@notamention:server` token");
    }

    #[test]
    fn wrap_mention_links_no_mentions_passthrough() {
        let s = wrap_mention_links("plain text with no mentions");
        assert_eq!(s, "plain text with no mentions");
    }

    // -- looks_like_transport_filename --------------------------

    #[test]
    fn transport_filename_recognizes_camera_patterns() {
        assert!(looks_like_transport_filename("IMG_1234.jpg"));
        assert!(looks_like_transport_filename("VID_20260101.mp4"));
        assert!(looks_like_transport_filename("Screenshot_2026-05-04.png"));
        assert!(looks_like_transport_filename("signal-2026-05-04.png"));
        assert!(looks_like_transport_filename("PXL_20260101_123456.jpg"));
    }

    #[test]
    fn transport_filename_rejects_real_text() {
        assert!(!looks_like_transport_filename("Hello there"));
        assert!(!looks_like_transport_filename("look at this image"));
        assert!(!looks_like_transport_filename(""));
    }

    #[test]
    fn transport_filename_recognizes_uuid_style() {
        assert!(looks_like_transport_filename("8a3f9b21-c1d4-4e2a.jpg"));
    }

    // -- Voice MSC3245 -----------------------------------------

    #[test]
    fn media_event_content_voice_carries_msc3245_marker() {
        let ch = channel();
        let att = crate::bus::MediaAttachment {
            kind: crate::bus::MediaKind::Voice,
            bytes: vec![0; 100],
            mime: "audio/ogg".into(),
            filename: None,
            width: None,
            height: None,
            duration_ms: Some(2500),
        };
        let v = ch.media_event_content(&att, "mxc://example.org/v");
        assert_eq!(v["msgtype"], "m.audio");
        assert!(v.get("org.matrix.msc3245.voice").is_some());
        assert!(v.get("org.matrix.msc1767.audio").is_some());
        assert_eq!(v["info"]["duration"], 2500);
    }

    // -- Filename heuristic in inbound media --------------------

    #[test]
    fn handle_event_image_with_transport_filename_strips_body() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let ev = evt(json!({
            "type": "m.room.message",
            "event_id": "$tx",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.image",
                "body": "IMG_4567.jpg",
                "url": "mxc://example.org/abc"
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &ev).expect("kept");
        assert_eq!(inbound.content, "");
        assert_eq!(
            inbound.metadata.get("matrix_media_filename").map(String::as_str),
            Some("IMG_4567.jpg")
        );
    }

    #[test]
    fn handle_event_image_with_real_caption_keeps_body() {
        let mut c = cfg();
        c.allowed_rooms = vec!["*".into()];
        c.require_mention = false;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let ev = evt(json!({
            "type": "m.room.message",
            "event_id": "$cap",
            "sender": "@alice:example.org",
            "origin_server_ts": chrono::Utc::now().timestamp_millis(),
            "content": {
                "msgtype": "m.image",
                "body": "look at this fennec",
                "url": "mxc://example.org/abc"
            }
        }));
        let inbound = ch.handle_event("!room:example.org", &ev).expect("kept");
        assert_eq!(inbound.content, "look at this fennec");
        assert!(inbound.metadata.get("matrix_media_filename").is_none());
    }

    // -- Approval prompt ----------------------------------------

    #[test]
    fn approval_prompt_matches_reaction() {
        let prompt = MatrixApprovalPrompt {
            room_id: "!r:example.org".into(),
            prompt_event_id: "$prompt".into(),
            seed_reactions: Vec::new(),
        };
        let mut meta = std::collections::HashMap::new();
        meta.insert("matrix_reaction_target".into(), "$prompt".into());
        assert!(prompt.matches_reaction(&meta));
        meta.insert("matrix_reaction_target".into(), "$other".into());
        assert!(!prompt.matches_reaction(&meta));
    }

    // -- Session restore ----------------------------------------

    #[tokio::test]
    async fn state_file_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let state_path = dir.path().join("matrix-state.json");
        let mut c = cfg();
        c.state_file = state_path.to_string_lossy().into_owned();
        let ch = MatrixChannel::from_config(&c).unwrap();
        // Simulate a sync that produced a next_batch token.
        *ch.state.next_batch.lock() = Some("s12_456".into());
        ch.persist_state().await.expect("persist");
        // New channel, same state_file → load_state restores it.
        let ch2 = MatrixChannel::from_config(&c).unwrap();
        assert!(ch2.state.next_batch.lock().is_none());
        ch2.load_state().await;
        assert_eq!(
            ch2.state.next_batch.lock().clone(),
            Some("s12_456".into())
        );
    }

    #[tokio::test]
    async fn state_file_empty_is_noop() {
        let ch = channel(); // state_file = ""
        // Both should silently no-op.
        ch.persist_state().await.expect("persist no-op");
        ch.load_state().await;
        assert!(ch.state.next_batch.lock().is_none());
    }

    // -- Text batching ------------------------------------------

    #[tokio::test]
    async fn batched_send_buffers_when_delay_set() {
        let mut c = cfg();
        c.text_batch_delay_ms = 50;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let msg = SendMessage::new("first", "!room:example.org");
        let _ = ch.enqueue_batched(&msg).await;
        let msg2 = SendMessage::new("second", "!room:example.org");
        let _ = ch.enqueue_batched(&msg2).await;
        // Buffer should hold both before the flush task runs.
        let buf = ch.state.text_batch_buffers.lock();
        let chat = buf.get("!room:example.org").expect("buffer");
        assert_eq!(chat.len(), 2);
        assert_eq!(chat[0].content, "first");
        assert_eq!(chat[1].content, "second");
    }

    #[tokio::test]
    async fn batched_send_separate_chats_isolated() {
        let mut c = cfg();
        c.text_batch_delay_ms = 50;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let _ = ch.enqueue_batched(&SendMessage::new("a", "!r1:example.org")).await;
        let _ = ch.enqueue_batched(&SendMessage::new("b", "!r2:example.org")).await;
        let buf = ch.state.text_batch_buffers.lock();
        assert_eq!(buf.get("!r1:example.org").map(|v| v.len()), Some(1));
        assert_eq!(buf.get("!r2:example.org").map(|v| v.len()), Some(1));
    }

    #[tokio::test]
    async fn batched_send_arms_flush_handle() {
        let mut c = cfg();
        c.text_batch_delay_ms = 10_000; // long enough we never see the flush
        let ch = MatrixChannel::from_config(&c).unwrap();
        let _ = ch
            .enqueue_batched(&SendMessage::new("hi", "!r:example.org"))
            .await;
        assert!(ch
            .state
            .flush_handles
            .lock()
            .contains_key("!r:example.org"));
    }

    #[tokio::test]
    async fn batched_send_long_message_aborts_and_re_arms() {
        // First arm with a short message — picks up the base
        // delay. Second enqueue is a "near-limit" chunk; the
        // implementation must abort the current flush handle and
        // spawn a fresh one with the doubled delay. We verify by
        // observing that the handle in state changes.
        let mut c = cfg();
        c.text_batch_delay_ms = 10_000;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let _ = ch
            .enqueue_batched(&SendMessage::new("short", "!r:example.org"))
            .await;
        let handle1_id = {
            let h = ch.state.flush_handles.lock();
            // JoinHandle isn't directly comparable; use its raw
            // task id (stable for the lifetime of the handle).
            h.get("!r:example.org").map(|h| h.id())
        };
        // Now a long message — should trigger abort + re-arm.
        let long = "x".repeat(BATCH_LONG_THRESHOLD + 10);
        let _ = ch
            .enqueue_batched(&SendMessage::new(long, "!r:example.org"))
            .await;
        let handle2_id = {
            let h = ch.state.flush_handles.lock();
            h.get("!r:example.org").map(|h| h.id())
        };
        assert!(handle1_id.is_some());
        assert!(handle2_id.is_some());
        assert_ne!(handle1_id, handle2_id);
    }

    #[tokio::test]
    async fn batched_send_short_messages_dont_re_arm_with_extended_delay() {
        // Sanity: when no message in the buffer crosses the
        // threshold, the flush should still arm (every enqueue
        // re-arms by design now), but the delay is the base.
        // We can't directly observe the configured delay from
        // outside, so we just verify the buffer accumulates and
        // a flush handle exists.
        let mut c = cfg();
        c.text_batch_delay_ms = 10_000;
        let ch = MatrixChannel::from_config(&c).unwrap();
        let _ = ch.enqueue_batched(&SendMessage::new("a", "!r:example.org")).await;
        let _ = ch.enqueue_batched(&SendMessage::new("b", "!r:example.org")).await;
        let _ = ch.enqueue_batched(&SendMessage::new("c", "!r:example.org")).await;
        assert_eq!(
            ch.state
                .text_batch_buffers
                .lock()
                .get("!r:example.org")
                .map(|v| v.len()),
            Some(3)
        );
        assert!(ch
            .state
            .flush_handles
            .lock()
            .contains_key("!r:example.org"));
    }
}
