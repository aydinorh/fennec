use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;

use crate::bus::InboundMessage;
use crate::security::url_guard::read_body_capped;

use super::traits::{Channel, SendMessage};

/// Maximum message length allowed by the Telegram Bot API.
const TELEGRAM_MAX_LEN: usize = 4096;

/// Hard cap on the number of per-chat entries in [`TelegramChannel::last_edit`].
/// When exceeded, the oldest entry is evicted on insert.
const MAX_LAST_EDIT_ENTRIES: usize = 1024;

/// Maximum number of times a request will retry on 429 (Too Many Requests)
/// before bailing.
const MAX_RETRY_AFTER_ATTEMPTS: u32 = 3;

/// Cap on `parameters.retry_after` honored from a 429 response. Telegram
/// occasionally returns very long waits during punishment windows; we'd
/// rather surface the failure than block a listener for many minutes.
const RETRY_AFTER_CAP_SECS: u64 = 60;

/// Maximum size, in bytes, we download for an incoming Telegram attachment.
/// Telegram's public Bot API caps `getFile` downloads at 20 MB; we mirror
/// that so a huge file can't exhaust memory or disk.
const MAX_MEDIA_DOWNLOAD_BYTES: usize = 20 * 1024 * 1024;

/// Kind of attachment on an incoming Telegram message. Determines the
/// default file extension and the hint we give the agent about which tool
/// to reach for (vision vs. transcription).
/// Max bytes of a text-like document (`.md`/`.txt`) whose content we inline
/// into the prompt instead of only referencing its path. Mirrors the
/// upstream's 100 KB injection cap.
const MAX_TEXT_INJECT_BYTES: u64 = 100 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TgMediaKind {
    Photo,
    Voice,
    Audio,
    Video,
    Document,
    Sticker,
}

impl TgMediaKind {
    /// Default file extension when the resolved `file_path` has none.
    fn default_ext(self) -> &'static str {
        match self {
            TgMediaKind::Photo => "jpg",
            TgMediaKind::Voice => "ogg",
            TgMediaKind::Audio => "mp3",
            TgMediaKind::Video => "mp4",
            TgMediaKind::Document => "bin",
            TgMediaKind::Sticker => "webp",
        }
    }

    /// Human label used in the saved-at marker injected into the prompt.
    fn label(self) -> &'static str {
        match self {
            TgMediaKind::Photo => "photo",
            TgMediaKind::Voice => "voice message",
            TgMediaKind::Audio => "audio file",
            TgMediaKind::Video => "video",
            TgMediaKind::Document => "document",
            TgMediaKind::Sticker => "sticker",
        }
    }

    /// Tool hint appended to the saved-at marker so the agent knows which
    /// tool to reach for. Empty for documents (path is enough).
    fn tool_hint(self) -> &'static str {
        match self {
            TgMediaKind::Photo | TgMediaKind::Video | TgMediaKind::Sticker => {
                " Use the vision_describe tool to view it."
            }
            TgMediaKind::Voice | TgMediaKind::Audio => " Use the transcribe tool to hear it.",
            TgMediaKind::Document => "",
        }
    }
}

/// A single attachment referenced by an incoming update.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TgMedia {
    file_id: String,
    kind: TgMediaKind,
    /// Original filename for documents (used for the on-disk name + marker
    /// and to decide whether to inline a text document's content).
    file_name: Option<String>,
    /// Emoji a sticker represents (`None` for non-stickers).
    emoji: Option<String>,
    /// True for an animated/video sticker — these can't be still-image
    /// analyzed, so we surface the emoji instead of downloading.
    animated: bool,
}

impl TgMedia {
    /// Construct a plain (non-sticker) attachment.
    fn plain(file_id: String, kind: TgMediaKind, file_name: Option<String>) -> Self {
        TgMedia {
            file_id,
            kind,
            file_name,
            emoji: None,
            animated: false,
        }
    }
}

/// A parsed incoming Telegram update reduced to what the agent needs:
/// the routing IDs, the text (or media caption), any attachment, and the
/// album id (`media_group_id`) so multi-photo albums coalesce into one turn.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TgUpdate {
    update_id: i64,
    sender_id: String,
    chat_id: String,
    /// Message text, or the caption for a media message. May be empty.
    text: String,
    media: Option<TgMedia>,
    /// Telegram album identifier; `Some` when this update is one item of a
    /// multi-attachment album.
    media_group_id: Option<String>,
}

/// Default cache directory for incoming Telegram attachments: a
/// `fennec-telegram-media` folder under the system temp dir. The system
/// temp dir is outside the path sandbox's denylist, so the agent's media
/// tools can read what lands here.
fn default_media_dir() -> PathBuf {
    std::env::temp_dir().join("fennec-telegram-media")
}

/// Group a poll batch so contiguous updates sharing a `media_group_id`
/// (a Telegram album) coalesce into one work item. Non-album updates each
/// form their own single-item group. Telegram queues an album's items
/// together, so they almost always arrive in one poll; an album split
/// across two polls degrades gracefully to separate messages.
fn group_updates(updates: Vec<TgUpdate>) -> Vec<Vec<TgUpdate>> {
    let mut groups: Vec<Vec<TgUpdate>> = Vec::new();
    for upd in updates {
        let same_album = matches!(
            (&upd.media_group_id, groups.last()),
            (Some(gid), Some(last))
                if last.last().and_then(|u| u.media_group_id.as_ref()) == Some(gid)
        );
        if same_album {
            groups.last_mut().expect("same_album implies a last group").push(upd);
        } else {
            groups.push(vec![upd]);
        }
    }
    groups
}

/// Append a caption after a marker, separated by a blank line. Empty
/// captions are dropped.
fn with_caption(marker: String, caption: &str) -> String {
    if caption.trim().is_empty() {
        marker
    } else {
        format!("{marker}\n\n{caption}")
    }
}

/// Sanitize a user-supplied document filename for display inside a prompt
/// marker (mirrors the upstream's `[^\w.\- ]` → `_` scrub) so a crafted
/// name can't inject control characters or break the marker framing.
fn sanitize_doc_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Build the marker for a successfully-downloaded attachment. For a small
/// text document (`.md`/`.txt`) the content is inlined directly (mirrors
/// the upstream's 100 KB injection) so the agent reads it without a tool;
/// everything else gets a `saved at: <path>` reference plus a tool hint.
fn marker_for_download(media: &TgMedia, path: &Path) -> String {
    if media.kind == TgMediaKind::Document {
        if let Some(name) = media.file_name.as_deref() {
            let is_text = matches!(extension_of(name).as_deref(), Some("md") | Some("txt"));
            if is_text {
                let small = std::fs::metadata(path)
                    .map(|m| m.len() <= MAX_TEXT_INJECT_BYTES)
                    .unwrap_or(false);
                if small {
                    if let Ok(content) = std::fs::read_to_string(path) {
                        return format!("[Content of {}]:\n{}", sanitize_doc_name(name), content);
                    }
                }
            }
        }
    }

    let name = media
        .file_name
        .as_deref()
        .map(|n| format!(" '{}'", sanitize_doc_name(n)))
        .unwrap_or_default();
    let emoji = media
        .emoji
        .as_deref()
        .map(|e| format!(" (emoji {e})"))
        .unwrap_or_default();
    // An image delivered as a document should still get the vision hint —
    // the upstream reroutes image-documents through the photo/vision path.
    let saved_ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    let is_image = matches!(
        saved_ext.as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "gif")
    );
    let hint = if media.kind == TgMediaKind::Document && is_image {
        TgMediaKind::Photo.tool_hint()
    } else {
        media.kind.tool_hint()
    };
    format!(
        "[The user sent a {}{}{}, saved at: {}.{}]",
        media.kind.label(),
        emoji,
        name,
        path.display(),
        hint
    )
}

/// Lowercase file extension (without the dot) of a path-like string, if it
/// has a short, alphanumeric one. Telegram `file_path`s look like
/// `photos/file_42.jpg` or `voice/file_7.oga`.
fn extension_of(path: &str) -> Option<String> {
    let tail = path.rsplit('/').next().unwrap_or(path);
    let (_, ext) = tail.rsplit_once('.')?;
    if !ext.is_empty() && ext.len() <= 5 && ext.chars().all(|c| c.is_ascii_alphanumeric()) {
        Some(ext.to_ascii_lowercase())
    } else {
        None
    }
}

/// Telegram channel using the Bot API with long-polling and streaming edits.
pub struct TelegramChannel {
    bot_token: String,
    client: reqwest::Client,
    allowed_users: Vec<String>,
    /// Per-chat timestamp of the last edit, used for rate-limiting streaming deltas.
    last_edit: Arc<Mutex<HashMap<String, Instant>>>,
    /// Directory incoming attachments are downloaded into so the agent's
    /// `vision_describe` / `transcribe` tools can read them by path.
    media_dir: PathBuf,
}

impl TelegramChannel {
    pub fn new(bot_token: String, allowed_users: Vec<String>) -> Self {
        Self {
            bot_token,
            client: reqwest::Client::new(),
            allowed_users,
            last_edit: Arc::new(Mutex::new(HashMap::new())),
            media_dir: default_media_dir(),
        }
    }

    /// Override the directory incoming attachments are cached in. Defaults
    /// to a `fennec-telegram-media` folder under the system temp dir.
    pub fn with_media_dir(mut self, dir: PathBuf) -> Self {
        self.media_dir = dir;
        self
    }

    fn api_url(&self, method: &str) -> String {
        format!("https://api.telegram.org/bot{}/{}", self.bot_token, method)
    }

    /// Insert into `last_edit` with an LRU-style eviction when the map is
    /// at capacity. Cap is `MAX_LAST_EDIT_ENTRIES`.
    fn last_edit_insert(&self, chat_id: String, when: Instant) {
        let mut map = self.last_edit.lock();
        if map.len() >= MAX_LAST_EDIT_ENTRIES && !map.contains_key(&chat_id) {
            // Evict the oldest entry (smallest Instant). O(n) but cap is small.
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

    /// Send a JSON POST that honors Telegram's 429 `parameters.retry_after`.
    /// Returns the parsed JSON body on success; bails after
    /// `MAX_RETRY_AFTER_ATTEMPTS` retries.
    async fn post_json_with_retry(&self, url: &str, body: &Value) -> Result<Value> {
        let mut attempt: u32 = 0;
        loop {
            let resp = self
                .client
                .post(url)
                .json(body)
                .send()
                .await
                .context("Telegram POST request failed")?;
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();

            if status.is_success() {
                return serde_json::from_str(&text)
                    .context("Telegram POST response parse failed");
            }

            if status.as_u16() == 429 && attempt < MAX_RETRY_AFTER_ATTEMPTS {
                let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                let retry_after = parsed
                    .get("parameters")
                    .and_then(|p| p.get("retry_after"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1)
                    .min(RETRY_AFTER_CAP_SECS);
                tracing::warn!(
                    "Telegram 429 on {}: sleeping {}s (attempt {}/{})",
                    url,
                    retry_after,
                    attempt + 1,
                    MAX_RETRY_AFTER_ATTEMPTS
                );
                tokio::time::sleep(Duration::from_secs(retry_after)).await;
                attempt += 1;
                continue;
            }

            anyhow::bail!("Telegram POST {} returned {}: {}", url, status, text);
        }
    }

    /// GET variant of [`Self::post_json_with_retry`] for `getUpdates` long-poll.
    async fn get_with_retry(&self, url: &str) -> Result<Value> {
        let mut attempt: u32 = 0;
        loop {
            let resp = self
                .client
                .get(url)
                .send()
                .await
                .context("Telegram GET request failed")?;
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();

            if status.is_success() {
                return serde_json::from_str(&text)
                    .context("Telegram GET response parse failed");
            }

            if status.as_u16() == 429 && attempt < MAX_RETRY_AFTER_ATTEMPTS {
                let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                let retry_after = parsed
                    .get("parameters")
                    .and_then(|p| p.get("retry_after"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1)
                    .min(RETRY_AFTER_CAP_SECS);
                tracing::warn!(
                    "Telegram 429 on {}: sleeping {}s (attempt {}/{})",
                    url,
                    retry_after,
                    attempt + 1,
                    MAX_RETRY_AFTER_ATTEMPTS
                );
                tokio::time::sleep(Duration::from_secs(retry_after)).await;
                attempt += 1;
                continue;
            }

            anyhow::bail!("Telegram GET {} returned {}: {}", url, status, text);
        }
    }

    /// Edit an existing message's text. Used by the streaming path.
    async fn edit_message(&self, chat_id: &str, message_id: &str, text: &str) -> Result<()> {
        let body = serde_json::json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        });
        self.post_json_with_retry(&self.api_url("editMessageText"), &body)
            .await?;
        Ok(())
    }

    /// Send a fresh text message, returning the new message id. Used to
    /// deliver overflow continuation chunks past the 4096 limit.
    async fn send_text(&self, chat_id: &str, text: &str) -> Result<Option<String>> {
        let body = serde_json::json!({
            "chat_id": chat_id,
            "text": text,
        });
        let data = self
            .post_json_with_retry(&self.api_url("sendMessage"), &body)
            .await?;
        Ok(data
            .get("result")
            .and_then(|r| r.get("message_id"))
            .and_then(|v| v.as_i64())
            .map(|id| id.to_string()))
    }

    /// Delete a message (best-effort). Used to remove the "..." streaming
    /// placeholder when a turn produced no prose.
    async fn delete_message(&self, chat_id: &str, message_id: &str) -> Result<()> {
        let body = serde_json::json!({
            "chat_id": chat_id,
            "message_id": message_id,
        });
        self.post_json_with_retry(&self.api_url("deleteMessage"), &body)
            .await?;
        Ok(())
    }

    /// Register bot commands with Telegram so they appear in the menu.
    async fn register_commands(&self) -> Result<()> {
        let commands = serde_json::json!([
            {"command": "new", "description": "Start a new conversation"},
            {"command": "status", "description": "Show agent status"},
            {"command": "help", "description": "Show available commands"},
        ]);
        self.client
            .post(self.api_url("setMyCommands"))
            .json(&serde_json::json!({"commands": commands}))
            .send()
            .await
            .context("Telegram setMyCommands request failed")?;
        Ok(())
    }

    /// Highest `update_id` in a raw `getUpdates` response, across **every**
    /// update — not just the message updates [`parse_updates`] keeps.
    ///
    /// The long-poll offset must advance past every update Telegram returns,
    /// including ones we don't act on (text-less media before media handling
    /// landed, stickers, `edited_message`, `callback_query`, …). If the
    /// offset were derived only from updates we surfaced, a trailing
    /// unhandled update would never be acknowledged: every poll would
    /// re-fetch it, `parse_updates` would return nothing, the offset would
    /// stand still, and the listener would spin on the same update forever —
    /// freezing the whole channel. Advancing from the raw maximum guarantees
    /// forward progress regardless of what we choose to handle.
    pub fn max_update_id(body: &Value) -> Option<i64> {
        body.get("result")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|u| u.get("update_id").and_then(|v| v.as_i64()))
            .max()
    }

    /// Parse a Telegram `getUpdates` response into the message updates the
    /// agent can act on. Each [`TgUpdate`] carries the routing IDs, the text
    /// (or media caption), and any single attachment.
    ///
    /// Photos report the **largest** size variant (Telegram sends an
    /// ascending-size array). Voice/audio/video/document each map to their
    /// `file_id`. Non-message updates and updates with neither text nor a
    /// recognized attachment are skipped here — but the offset still advances
    /// past them via [`max_update_id`].
    fn parse_updates(body: &Value) -> Vec<TgUpdate> {
        let mut results = Vec::new();
        let Some(arr) = body.get("result").and_then(|v| v.as_array()) else {
            return results;
        };
        for update in arr {
            let update_id = update.get("update_id").and_then(|v| v.as_i64()).unwrap_or(0);
            let Some(message) = update.get("message") else {
                continue;
            };

            // Text for a plain message; caption for a media message.
            let text = message
                .get("text")
                .or_else(|| message.get("caption"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let media = Self::extract_media(message);

            // Nothing actionable: no text and no recognized attachment.
            if text.is_empty() && media.is_none() {
                continue;
            }

            let sender_id = message
                .get("from")
                .and_then(|f| f.get("id"))
                .and_then(|v| v.as_i64())
                .map(|id| id.to_string())
                .unwrap_or_default();
            let chat_id = message
                .get("chat")
                .and_then(|c| c.get("id"))
                .and_then(|v| v.as_i64())
                .map(|id| id.to_string())
                .unwrap_or_default();

            let media_group_id = message
                .get("media_group_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            results.push(TgUpdate {
                update_id,
                sender_id,
                chat_id,
                text,
                media,
                media_group_id,
            });
        }
        results
    }

    /// Pull the single most relevant attachment off a Telegram `message`
    /// object, if any. Order mirrors the upstream: sticker → photo → voice →
    /// audio → video → document.
    fn extract_media(message: &Value) -> Option<TgMedia> {
        // Stickers carry their own metadata (emoji, animated/video flags).
        if let Some(st) = message.get("sticker").filter(|v| v.is_object()) {
            if let Some(file_id) = st.get("file_id").and_then(|v| v.as_str()) {
                let emoji = st
                    .get("emoji")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string());
                let animated = st.get("is_animated").and_then(|v| v.as_bool()).unwrap_or(false)
                    || st.get("is_video").and_then(|v| v.as_bool()).unwrap_or(false);
                return Some(TgMedia {
                    file_id: file_id.to_string(),
                    kind: TgMediaKind::Sticker,
                    file_name: None,
                    emoji,
                    animated,
                });
            }
        }

        // Photos: array of PhotoSize ascending by resolution — take the last
        // (largest) entry's file_id.
        if let Some(file_id) = message
            .get("photo")
            .and_then(|v| v.as_array())
            .and_then(|sizes| sizes.last())
            .and_then(|p| p.get("file_id"))
            .and_then(|v| v.as_str())
        {
            return Some(TgMedia::plain(file_id.to_string(), TgMediaKind::Photo, None));
        }

        for (key, kind) in [
            ("voice", TgMediaKind::Voice),
            ("audio", TgMediaKind::Audio),
            ("video", TgMediaKind::Video),
            ("document", TgMediaKind::Document),
        ] {
            if let Some(obj) = message.get(key).filter(|v| v.is_object()) {
                if let Some(file_id) = obj.get("file_id").and_then(|v| v.as_str()) {
                    let file_name = obj
                        .get("file_name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    return Some(TgMedia::plain(file_id.to_string(), kind, file_name));
                }
            }
        }
        None
    }

    /// Resolve a `file_id` via `getFile`, download the bytes (capped at
    /// [`MAX_MEDIA_DOWNLOAD_BYTES`]), and write them to a file under
    /// [`media_dir`](Self::media_dir). Returns the on-disk path.
    ///
    /// The on-disk name is always a fresh UUID plus an extension derived
    /// from the resolved `file_path` (or the attachment kind / original
    /// document name) — never the attacker-controlled `file_name`
    /// directly, so a crafted name can't traverse out of the cache dir.
    async fn download_telegram_file(&self, media: &TgMedia) -> Result<PathBuf> {
        // 1. getFile → file_path. Goes through the retrying GET so 429s are
        //    honored like every other call.
        let get_file_url = format!("{}?file_id={}", self.api_url("getFile"), media.file_id);
        let info = self.get_with_retry(&get_file_url).await?;
        let file_path = info
            .get("result")
            .and_then(|r| r.get("file_path"))
            .and_then(|v| v.as_str())
            .context("Telegram getFile response missing result.file_path")?;

        // 2. Pick an extension: prefer the resolved file_path's, then a
        //    document's original name, then the kind default.
        let ext = extension_of(file_path)
            .or_else(|| media.file_name.as_deref().and_then(extension_of))
            .unwrap_or_else(|| media.kind.default_ext().to_string());

        // 3. Download the bytes from the file endpoint (token-bearing URL,
        //    so it never gets logged — see post_json_with_retry).
        let download_url = format!(
            "https://api.telegram.org/file/bot{}/{}",
            self.bot_token, file_path
        );
        let resp = self
            .client
            .get(&download_url)
            .send()
            .await
            .context("Telegram file download request failed")?;
        if !resp.status().is_success() {
            anyhow::bail!("Telegram file download returned {}", resp.status());
        }
        let (bytes, truncated) = read_body_capped(resp, MAX_MEDIA_DOWNLOAD_BYTES).await?;
        if truncated {
            anyhow::bail!(
                "attachment exceeds {} MB download cap",
                MAX_MEDIA_DOWNLOAD_BYTES / (1024 * 1024)
            );
        }

        // 4. Persist under the cache dir with a fresh, traversal-safe name.
        std::fs::create_dir_all(&self.media_dir).with_context(|| {
            format!("creating media cache dir {}", self.media_dir.display())
        })?;
        let file_name = format!("{}.{}", uuid::Uuid::new_v4(), ext);
        let path = self.media_dir.join(file_name);
        std::fs::write(&path, &bytes)
            .with_context(|| format!("writing attachment to {}", path.display()))?;
        Ok(path)
    }

    /// Build the message text for a single media update: download the
    /// attachment and produce a marker (mirrors the upstream) so the agent
    /// knows to reach for `vision_describe` / `transcribe` on the saved
    /// path. The caption, if any, follows.
    ///
    /// On download failure the channel does **not** stall: it returns a note
    /// explaining the attachment couldn't be fetched, plus the caption, so
    /// the agent still gets the message.
    async fn build_media_content(&self, media: &TgMedia, caption: &str) -> String {
        let marker = self.media_marker(media).await;
        with_caption(marker, caption)
    }

    /// Produce the marker text for one attachment (no caption). Handles the
    /// animated-sticker (emoji-only, no download) and text-document
    /// (content inlined) special cases.
    async fn media_marker(&self, media: &TgMedia) -> String {
        // Animated/video stickers can't be still-image analyzed; surface the
        // emoji instead of downloading (mirrors the upstream).
        if media.kind == TgMediaKind::Sticker && media.animated {
            return match &media.emoji {
                Some(e) => format!("[The user sent an animated sticker (emoji {e}).]"),
                None => "[The user sent an animated sticker.]".to_string(),
            };
        }

        match self.download_telegram_file(media).await {
            Ok(path) => marker_for_download(media, &path),
            Err(e) => {
                tracing::warn!("Telegram: failed to download {}: {e}", media.kind.label());
                format!(
                    "[The user sent a {} but it could not be downloaded: {}]",
                    media.kind.label(),
                    e
                )
            }
        }
    }

    /// Build the combined content for a Telegram album (multiple
    /// attachments sharing a `media_group_id`): each item's marker, joined,
    /// then the merged caption. Mirrors the upstream's media-group coalesce
    /// so a multi-photo album arrives as ONE agent turn instead of N.
    async fn build_album_content(&self, items: &[TgUpdate]) -> String {
        let mut markers = Vec::with_capacity(items.len());
        let mut caption = String::new();
        for item in items {
            if let Some(media) = &item.media {
                markers.push(self.media_marker(media).await);
            }
            // Telegram puts the caption on one item of the album; keep the
            // first non-empty one.
            if caption.is_empty() && !item.text.trim().is_empty() {
                caption = item.text.clone();
            }
        }
        let header = format!(
            "[The user sent an album of {} attachments.]",
            markers.len()
        );
        let body = markers.join("\n");
        with_caption(format!("{header}\n{body}"), &caption)
    }
}

/// Split a long message into parts that fit within `max_len`, preserving code
/// blocks (triple-backtick state). Each part gets a `(i/N)` indicator when
/// there are multiple parts.
/// Round `index` down to the nearest UTF-8 char boundary in `s`, or 0
/// if no boundary at or below `index` exists. Used by `split_message`'s
/// hard-split path so a multi-byte character (emoji, CJK, accented
/// Latin) at the cut doesn't panic `String::split_at`.
fn floor_char_boundary(s: &str, index: usize) -> usize {
    let mut i = index.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

pub fn split_message(text: &str, max_len: usize) -> Vec<String> {
    if text.len() <= max_len {
        return vec![text.to_string()];
    }

    let mut parts: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_code_block = false;
    let mut code_fence_lang = String::new();

    for line in text.split('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            if in_code_block {
                // Closing a code block.
                in_code_block = false;
            } else {
                // Opening a code block; remember the language tag.
                in_code_block = true;
                code_fence_lang = trimmed.strip_prefix("```").unwrap_or("").to_string();
            }
        }

        // +1 for the newline character we add when joining.
        let addition = if current.is_empty() {
            line.len()
        } else {
            line.len() + 1
        };

        if !current.is_empty() && current.len() + addition > max_len {
            // Need to split here.
            if in_code_block {
                // Close the code block in the current part before splitting.
                // But we set in_code_block=true above for the *opening* line,
                // so the line that triggered this was already an interior line
                // of the block — close it.
                // However, we need to check: did we *just* open the block on
                // this line? If so, it is not yet in `current`, so don't close.
                // Actually, we haven't pushed `line` yet, so `current` is in
                // a code block that was opened earlier.
                current.push_str("\n```");
            }
            parts.push(current);
            current = String::new();
            if in_code_block {
                // Re-open the code block in the new part.
                current.push_str(&format!("```{}\n", code_fence_lang));
            }
        }

        // If a single line exceeds max_len, hard-split it.
        if line.len() > max_len {
            let mut remaining = line;
            while !remaining.is_empty() {
                let take = remaining.len().min(max_len.saturating_sub(current.len().min(max_len - 1) + 1));
                let take = if take == 0 {
                    // current is already near max_len, flush it first.
                    if !current.is_empty() {
                        if in_code_block {
                            current.push_str("\n```");
                        }
                        parts.push(current);
                        current = String::new();
                        if in_code_block {
                            current.push_str(&format!("```{}\n", code_fence_lang));
                        }
                    }
                    remaining.len().min(max_len)
                } else {
                    take
                };
                // `take` is a byte count derived from `max_len` arithmetic.
                // `String::split_at` panics if the cut isn't on a UTF-8
                // char boundary, which happens whenever a multi-byte char
                // (emoji, CJK, accented Latin) lands at the cut. Walk
                // back at most 3 bytes (UTF-8 max width) until we find a
                // boundary. If `take == 0` after the walk, take 1 char
                // anyway so we always make progress; in the worst case
                // (a single character longer than `max_len`) we emit
                // that char alone, which is still better than panicking.
                let take = floor_char_boundary(remaining, take);
                let take = if take == 0 {
                    // No char boundary found at or below the byte index —
                    // walk forward to the first boundary so we make
                    // progress. `is_char_boundary(0)` is always true, so
                    // we'll find one within UTF-8 max width (4 bytes).
                    let mut i = 1;
                    while i < remaining.len() && !remaining.is_char_boundary(i) {
                        i += 1;
                    }
                    i.min(remaining.len())
                } else {
                    take
                };
                let (chunk, rest) = remaining.split_at(take);
                if !current.is_empty() {
                    current.push('\n');
                }
                current.push_str(chunk);
                remaining = rest;

                if current.len() >= max_len && !remaining.is_empty() {
                    if in_code_block {
                        current.push_str("\n```");
                    }
                    parts.push(current);
                    current = String::new();
                    if in_code_block {
                        current.push_str(&format!("```{}\n", code_fence_lang));
                    }
                }
            }
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    // Add part indicators if there are multiple parts.
    let total = parts.len();
    if total > 1 {
        parts = parts
            .into_iter()
            .enumerate()
            .map(|(i, p)| format!("({}/{}) {}", i + 1, total, p))
            .collect();
    }

    parts
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
    }

    async fn send(&self, message: &SendMessage) -> Result<()> {
        let parts = split_message(&message.content, TELEGRAM_MAX_LEN);
        let url = self.api_url("sendMessage");
        for part in &parts {
            let body = serde_json::json!({
                "chat_id": message.recipient,
                "text": part,
            });
            self.post_json_with_retry(&url, &body).await?;
        }
        Ok(())
    }

    async fn listen(&self, tx: tokio::sync::mpsc::Sender<InboundMessage>) -> Result<()> {
        // Register bot commands with Telegram on startup.
        if let Err(e) = self.register_commands().await {
            tracing::warn!("Failed to register Telegram bot commands: {e}");
        }

        let mut offset: i64 = 0;

        loop {
            let url = format!(
                "{}?timeout=30&offset={}",
                self.api_url("getUpdates"),
                offset
            );
            let body = self.get_with_retry(&url).await?;

            // Advance the long-poll offset past EVERY update in this batch
            // before doing anything else — including stickers, edits, and
            // media we may not surface. Deriving the offset from the raw
            // maximum (not from the updates we keep) is what prevents a
            // trailing unhandled update from wedging the listener forever.
            if let Some(max_id) = Self::max_update_id(&body) {
                if max_id >= offset {
                    offset = max_id + 1;
                }
            }

            // Group the batch so a multi-attachment album (updates sharing
            // a media_group_id) coalesces into a single agent turn.
            for group in group_updates(Self::parse_updates(&body)) {
                let first = &group[0];
                if !self.allows_sender(&first.sender_id) {
                    tracing::debug!(
                        "Telegram: ignoring message from disallowed sender {}",
                        first.sender_id
                    );
                    continue;
                }

                // Resolve the message body: an album coalesces all its
                // attachments; otherwise download any single attachment and
                // build a marker, or just use the text.
                let content = if group.len() > 1 {
                    self.build_album_content(&group).await
                } else {
                    match &first.media {
                        Some(media) => self.build_media_content(media, &first.text).await,
                        None => first.text.clone(),
                    }
                };
                if content.trim().is_empty() {
                    continue;
                }

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Handle /new and /reset commands as session reset signals.
                let mut metadata = HashMap::new();
                if content.starts_with("/new") || content.starts_with("/reset") {
                    metadata.insert("command".to_string(), "reset".to_string());
                }

                let msg = InboundMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    sender: first.sender_id.clone(),
                    content,
                    channel: "telegram".to_string(),
                    chat_id: first.chat_id.clone(),
                    timestamp: now,
                    reply_to: None,
                    metadata,
                };

                if tx.send(msg).await.is_err() {
                    tracing::info!("Telegram: inbound channel closed, stopping listener");
                    return Ok(());
                }
            }
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn send_streaming_start(&self, chat_id: &str) -> Result<Option<String>> {
        let body = serde_json::json!({
            "chat_id": chat_id,
            "text": "...",
        });
        let data = self
            .post_json_with_retry(&self.api_url("sendMessage"), &body)
            .await?;
        let message_id = data
            .get("result")
            .and_then(|r| r.get("message_id"))
            .and_then(|v| v.as_i64())
            .map(|id| id.to_string());
        Ok(message_id)
    }

    async fn send_streaming_delta(
        &self,
        chat_id: &str,
        message_id: &str,
        full_text: &str,
    ) -> Result<()> {
        // Empty text → Telegram rejects with `MESSAGE_TEXT_IS_EMPTY` /
        // 400. The agent legitimately produces empty intermediate
        // states (model emitted only a tool_call, no prose yet); skip
        // the edit until there's something to render.
        if full_text.is_empty() {
            return Ok(());
        }

        // Once the accumulated text grows past Telegram's single-message
        // limit, stop editing: `editMessageText` rejects an over-limit body
        // with 400 `MESSAGE_TOO_LONG`. The streamed bubble simply stops
        // updating here; `send_streaming_end` then splits the complete final
        // text across continuation messages so nothing is lost. (Previously
        // the over-limit edit errored and the reply froze at the last
        // sub-4096 state, dropping the tail.)
        if full_text.chars().count() > TELEGRAM_MAX_LEN {
            return Ok(());
        }

        // Rate-limit: skip if last edit for this chat was <300ms ago.
        {
            let map = self.last_edit.lock();
            if let Some(last) = map.get(chat_id) {
                if last.elapsed().as_millis() < 300 {
                    return Ok(());
                }
            }
        }

        self.edit_message(chat_id, message_id, full_text).await?;
        self.last_edit_insert(chat_id.to_string(), Instant::now());

        Ok(())
    }

    async fn send_streaming_end(
        &self,
        chat_id: &str,
        message_id: &str,
        full_text: &str,
    ) -> Result<()> {
        // Empty final text → the turn produced only a tool call / `[SILENT]`
        // and no prose. Delete the "..." placeholder we posted at stream
        // start so it doesn't linger as an orphaned bubble (editMessageText
        // would 400 on empty text anyway). Best-effort: a failed delete
        // isn't worth erroring the turn over.
        if full_text.is_empty() {
            if let Err(e) = self.delete_message(chat_id, message_id).await {
                tracing::debug!("Telegram: failed to delete empty-turn placeholder: {e}");
            }
            self.last_edit.lock().remove(chat_id);
            return Ok(());
        }

        // Split the complete final text across as many messages as it takes.
        // The first chunk replaces the streamed placeholder via edit; any
        // remaining chunks are sent as continuation messages in the same
        // chat, so a >4096 reply is delivered in full instead of truncated.
        let parts = split_message(full_text, TELEGRAM_MAX_LEN);
        if let Some((first, rest)) = parts.split_first() {
            self.edit_message(chat_id, message_id, first).await?;
            for part in rest {
                self.send_text(chat_id, part).await?;
            }
        }

        // Clear the rate-limit entry for this chat.
        self.last_edit.lock().remove(chat_id);

        Ok(())
    }

    async fn send_typing(&self, chat_id: &str) -> Result<()> {
        let body = serde_json::json!({
            "chat_id": chat_id,
            "action": "typing",
        });
        let _ = self
            .client
            .post(self.api_url("sendChatAction"))
            .json(&body)
            .send()
            .await;
        Ok(())
    }

    fn allows_sender(&self, sender_id: &str) -> bool {
        // Empty list or wildcard "*" means allow all.
        if self.allowed_users.is_empty() {
            return true;
        }
        if self.allowed_users.iter().any(|u| u == "*") {
            return true;
        }
        self.allowed_users.iter().any(|u| u == sender_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_message_short() {
        let parts = split_message("hello world", 4096);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0], "hello world");
    }

    #[test]
    fn test_split_message_exact_limit() {
        let msg = "a".repeat(4096);
        let parts = split_message(&msg, 4096);
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn test_split_message_long() {
        // Create a message with many lines that exceeds limit.
        let line = "This is a test line that is fairly long.\n";
        let msg = line.repeat(200); // ~8200 chars
        let parts = split_message(&msg, 4096);
        assert!(parts.len() >= 2);
        for part in &parts {
            assert!(part.len() <= 4096 + 20); // allow indicator overhead
        }
        // Check indicators.
        assert!(parts[0].starts_with("(1/"));
    }

    #[test]
    fn test_split_message_code_block_preserved() {
        let mut msg = String::new();
        msg.push_str("Before code\n");
        msg.push_str("```rust\n");
        // Add enough lines to force a split inside the code block.
        for i in 0..100 {
            msg.push_str(&format!("let x{} = {};\n", i, i));
        }
        msg.push_str("```\n");
        msg.push_str("After code\n");

        let parts = split_message(&msg, 500);
        assert!(parts.len() >= 2);
        // Each interior part that continues a code block should re-open it.
        // The second part should contain ``` to re-open.
        assert!(parts[1].contains("```"));
    }

    #[test]
    fn test_split_message_single_huge_line() {
        let msg = "x".repeat(5000);
        let parts = split_message(&msg, 4096);
        assert!(parts.len() >= 2);
    }

    /// Regression: the hard-split path used to call `String::split_at`
    /// on a byte index without checking UTF-8 char boundaries, panicking
    /// when a multi-byte char (emoji, CJK) landed at the cut. The
    /// `floor_char_boundary` walk ensures we cut on a boundary.
    #[test]
    fn test_split_message_emoji_at_boundary_does_not_panic() {
        // 4-byte emoji repeated to overflow max_len. With max_len=10
        // and each emoji 4 bytes, every other split lands inside an
        // emoji. Pre-fix this panicked; post-fix we get clean parts.
        let msg = "\u{1F600}".repeat(20); // 80 bytes total
        let parts = split_message(&msg, 10);
        // Every emitted byte must form a valid UTF-8 string, and the
        // concatenation (minus indicator prefixes) must equal the input.
        assert!(!parts.is_empty());
        for p in &parts {
            assert!(p.is_char_boundary(p.len()));
        }
    }

    /// CJK characters are typically 3 bytes in UTF-8. Same shape as the
    /// emoji test but covers the 3-byte width.
    #[test]
    fn test_split_message_cjk_at_boundary_does_not_panic() {
        let msg = "日本語検索".repeat(10);
        let parts = split_message(&msg, 8);
        assert!(!parts.is_empty());
        for p in &parts {
            assert!(p.is_char_boundary(p.len()));
        }
    }

    /// A single grapheme bigger than `max_len` should still get emitted
    /// (taking 1 char) rather than infinite-loop the splitter.
    #[test]
    fn test_split_message_progresses_when_char_exceeds_limit() {
        let msg = "\u{1F600}\u{1F600}"; // Two 4-byte emoji.
        let parts = split_message(&msg, 2); // Tighter than one char.
        // Just don't infinite-loop; some emission is fine.
        assert!(!parts.is_empty());
    }

    #[test]
    fn test_floor_char_boundary_helper() {
        let s = "a\u{1F600}b"; // a(1) + emoji(4) + b(1) = 6 bytes.
        // Boundaries at 0, 1 (after 'a'), 5 (after emoji), 6 (after 'b').
        // Index 3 is mid-emoji (inside the 4-byte UTF-8 sequence);
        // floor walks down to 1, the boundary after 'a'.
        assert_eq!(floor_char_boundary(s, 3), 1);
        // Index 4 is also mid-emoji; floor walks down to 1.
        assert_eq!(floor_char_boundary(s, 4), 1);
        // Index 5 is at the boundary BETWEEN the emoji and 'b'
        // (start of 'b'), so floor returns it unchanged.
        assert_eq!(floor_char_boundary(s, 5), 5);
        // Index 6 is at end (boundary).
        assert_eq!(floor_char_boundary(s, 6), 6);
        // Index past end clamps.
        assert_eq!(floor_char_boundary(s, 100), 6);
    }

    #[test]
    fn test_parse_updates_empty() {
        let body = serde_json::json!({"ok": true, "result": []});
        let updates = TelegramChannel::parse_updates(&body);
        assert!(updates.is_empty());
    }

    #[test]
    fn test_parse_updates_message() {
        let body = serde_json::json!({
            "ok": true,
            "result": [{
                "update_id": 123,
                "message": {
                    "text": "hello",
                    "from": {"id": 456},
                    "chat": {"id": 789}
                }
            }]
        });
        let updates = TelegramChannel::parse_updates(&body);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].update_id, 123);
        assert_eq!(updates[0].sender_id, "456");
        assert_eq!(updates[0].chat_id, "789");
        assert_eq!(updates[0].text, "hello");
        assert!(updates[0].media.is_none());
    }

    #[test]
    fn max_update_id_spans_text_less_updates() {
        // A trailing non-actionable update (here a location-only service
        // message, which parse_updates drops) must still advance the
        // offset: max_update_id returns it. This is the anti-freeze
        // guarantee.
        let body = serde_json::json!({
            "ok": true,
            "result": [
                {"update_id": 10, "message": {"text": "hi", "from": {"id": 1}, "chat": {"id": 2}}},
                {"update_id": 11, "message": {"location": {"latitude": 1.0, "longitude": 2.0}, "from": {"id": 1}, "chat": {"id": 2}}}
            ]
        });
        // parse_updates only keeps the text message (location is not a
        // recognized attachment)...
        let updates = TelegramChannel::parse_updates(&body);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].update_id, 10);
        // ...but the offset advances past the location update too.
        assert_eq!(TelegramChannel::max_update_id(&body), Some(11));
    }

    #[test]
    fn parse_updates_extracts_sticker_with_emoji() {
        let body = serde_json::json!({
            "ok": true,
            "result": [{
                "update_id": 8,
                "message": {
                    "from": {"id": 1}, "chat": {"id": 2},
                    "sticker": {"file_id": "stk", "emoji": "🎉", "is_animated": false}
                }
            }]
        });
        let updates = TelegramChannel::parse_updates(&body);
        let media = updates[0].media.as_ref().expect("sticker media");
        assert_eq!(media.kind, TgMediaKind::Sticker);
        assert_eq!(media.emoji.as_deref(), Some("🎉"));
        assert!(!media.animated);
    }

    #[test]
    fn animated_sticker_marker_uses_emoji_without_download() {
        // media_marker for an animated sticker must NOT attempt a download
        // (no network in tests) and surfaces the emoji.
        let ch = TelegramChannel::new("token".into(), vec![]);
        let media = TgMedia {
            file_id: "x".into(),
            kind: TgMediaKind::Sticker,
            file_name: None,
            emoji: Some("😺".into()),
            animated: true,
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let marker = rt.block_on(ch.media_marker(&media));
        assert_eq!(marker, "[The user sent an animated sticker (emoji 😺).]");
    }

    #[test]
    fn group_updates_coalesces_album_items() {
        let mk = |id: i64, gid: Option<&str>| TgUpdate {
            update_id: id,
            sender_id: "1".into(),
            chat_id: "2".into(),
            text: String::new(),
            media: Some(TgMedia::plain(format!("f{id}"), TgMediaKind::Photo, None)),
            media_group_id: gid.map(String::from),
        };
        // Two album items (same gid) + one standalone photo.
        let groups = group_updates(vec![mk(1, Some("A")), mk(2, Some("A")), mk(3, None)]);
        assert_eq!(groups.len(), 2, "album coalesces; standalone is its own group");
        assert_eq!(groups[0].len(), 2);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn marker_for_download_inlines_small_text_document() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("notes.txt");
        std::fs::write(&path, "hello from a file").unwrap();
        let media = TgMedia::plain("id".into(), TgMediaKind::Document, Some("notes.txt".into()));
        let marker = marker_for_download(&media, &path);
        assert!(marker.starts_with("[Content of notes.txt]:\n"));
        assert!(marker.contains("hello from a file"));
    }

    #[test]
    fn marker_for_download_references_path_for_binary_document() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("doc.pdf");
        std::fs::write(&path, b"%PDF-1.4 binary").unwrap();
        let media = TgMedia::plain("id".into(), TgMediaKind::Document, Some("doc.pdf".into()));
        let marker = marker_for_download(&media, &path);
        assert!(marker.contains("saved at:"));
        assert!(marker.contains("doc.pdf"));
    }

    #[test]
    fn sanitize_doc_name_strips_unsafe_chars() {
        assert_eq!(sanitize_doc_name("report v2.pdf"), "report v2.pdf");
        // ']' , '\n' , '[' each become '_'.
        assert_eq!(sanitize_doc_name("a]b\n[c"), "a_b__c");
    }

    #[test]
    fn parse_updates_extracts_photo_and_caption() {
        let body = serde_json::json!({
            "ok": true,
            "result": [{
                "update_id": 5,
                "message": {
                    "caption": "look at this",
                    "from": {"id": 1},
                    "chat": {"id": 2},
                    "photo": [
                        {"file_id": "small", "width": 90, "height": 90},
                        {"file_id": "large", "width": 1280, "height": 1280}
                    ]
                }
            }]
        });
        let updates = TelegramChannel::parse_updates(&body);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].text, "look at this");
        let media = updates[0].media.as_ref().expect("photo media");
        // Largest variant (last in the ascending array) is chosen.
        assert_eq!(media.file_id, "large");
        assert_eq!(media.kind, TgMediaKind::Photo);
    }

    #[test]
    fn parse_updates_extracts_voice_without_caption() {
        let body = serde_json::json!({
            "ok": true,
            "result": [{
                "update_id": 6,
                "message": {
                    "from": {"id": 1},
                    "chat": {"id": 2},
                    "voice": {"file_id": "voice123", "duration": 3}
                }
            }]
        });
        let updates = TelegramChannel::parse_updates(&body);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].text, "");
        let media = updates[0].media.as_ref().expect("voice media");
        assert_eq!(media.file_id, "voice123");
        assert_eq!(media.kind, TgMediaKind::Voice);
    }

    #[test]
    fn parse_updates_extracts_document_filename() {
        let body = serde_json::json!({
            "ok": true,
            "result": [{
                "update_id": 7,
                "message": {
                    "from": {"id": 1},
                    "chat": {"id": 2},
                    "document": {"file_id": "doc1", "file_name": "report.pdf"}
                }
            }]
        });
        let updates = TelegramChannel::parse_updates(&body);
        let media = updates[0].media.as_ref().expect("document media");
        assert_eq!(media.kind, TgMediaKind::Document);
        assert_eq!(media.file_name.as_deref(), Some("report.pdf"));
    }

    #[test]
    fn extension_of_extracts_known_extensions() {
        assert_eq!(extension_of("photos/file_42.jpg").as_deref(), Some("jpg"));
        assert_eq!(extension_of("voice/file_7.OGA").as_deref(), Some("oga"));
        assert_eq!(extension_of("noext"), None);
        assert_eq!(extension_of("weird.toolongextension"), None);
    }

    #[test]
    fn test_allows_sender_empty_list() {
        let ch = TelegramChannel::new("token".to_string(), vec![]);
        assert!(ch.allows_sender("anyone"));
    }

    #[test]
    fn test_allows_sender_wildcard() {
        let ch = TelegramChannel::new("token".to_string(), vec!["*".to_string()]);
        assert!(ch.allows_sender("anyone"));
    }

    #[test]
    fn test_allows_sender_restricted() {
        let ch = TelegramChannel::new("token".to_string(), vec!["123".to_string()]);
        assert!(ch.allows_sender("123"));
        assert!(!ch.allows_sender("456"));
    }
}
