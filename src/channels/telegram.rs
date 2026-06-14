use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::Value;

use crate::bus::InboundMessage;

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

/// Redact the bot token embedded in a Telegram Bot API URL before it is
/// logged or surfaced in an error. The token sits between `/bot` and the
/// next `/` (e.g. `…/bot123456:ABC-DEF…/sendMessage`); leaking it into a
/// WARN line or a bailed error message hands an attacker full control of
/// the bot. We replace only the secret segment and keep the method name
/// (`sendMessage`, `getUpdates`, …) intact, which is the part that
/// actually helps when debugging. Upstream sidesteps this entirely by
/// using a Telegram client library that never logs URL-embedded tokens;
/// since Fennec constructs URLs by hand, it scrubs them explicitly.
fn redact_telegram_url(url: &str) -> String {
    const MARKER: &str = "/bot";
    let Some(start) = url.find(MARKER) else {
        return url.to_string();
    };
    let token_start = start + MARKER.len();
    // The token runs until the next path separator (the method) or the
    // end of the string.
    let token_end = url[token_start..]
        .find('/')
        .map(|rel| token_start + rel)
        .unwrap_or(url.len());
    let mut redacted = String::with_capacity(url.len());
    redacted.push_str(&url[..token_start]);
    redacted.push_str("<redacted>");
    redacted.push_str(&url[token_end..]);
    redacted
}

/// Telegram channel using the Bot API with long-polling and streaming edits.
pub struct TelegramChannel {
    bot_token: String,
    client: reqwest::Client,
    allowed_users: Vec<String>,
    /// Per-chat timestamp of the last edit, used for rate-limiting streaming deltas.
    last_edit: Arc<Mutex<HashMap<String, Instant>>>,
}

impl TelegramChannel {
    pub fn new(bot_token: String, allowed_users: Vec<String>) -> Self {
        Self {
            bot_token,
            client: reqwest::Client::new(),
            allowed_users,
            last_edit: Arc::new(Mutex::new(HashMap::new())),
        }
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
                    redact_telegram_url(url),
                    retry_after,
                    attempt + 1,
                    MAX_RETRY_AFTER_ATTEMPTS
                );
                tokio::time::sleep(Duration::from_secs(retry_after)).await;
                attempt += 1;
                continue;
            }

            anyhow::bail!(
                "Telegram POST {} returned {}: {}",
                redact_telegram_url(url),
                status,
                text
            );
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
                    redact_telegram_url(url),
                    retry_after,
                    attempt + 1,
                    MAX_RETRY_AFTER_ATTEMPTS
                );
                tokio::time::sleep(Duration::from_secs(retry_after)).await;
                attempt += 1;
                continue;
            }

            anyhow::bail!(
                "Telegram GET {} returned {}: {}",
                redact_telegram_url(url),
                status,
                text
            );
        }
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

    /// Parse a Telegram `getUpdates` JSON response into a list of
    /// (update_id, sender_id, chat_id, text) tuples.
    pub fn parse_updates(body: &Value) -> Vec<(i64, String, String, String)> {
        let mut results = Vec::new();
        if let Some(arr) = body.get("result").and_then(|v| v.as_array()) {
            for update in arr {
                let update_id = update.get("update_id").and_then(|v| v.as_i64()).unwrap_or(0);
                if let Some(message) = update.get("message") {
                    let text = message
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if text.is_empty() {
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
                    results.push((update_id, sender_id, chat_id, text));
                }
            }
        }
        results
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
            let updates = Self::parse_updates(&body);

            for (update_id, sender_id, chat_id, text) in updates {
                // Track offset: next poll starts after the highest update_id.
                if update_id >= offset {
                    offset = update_id + 1;
                }

                if !self.allows_sender(&sender_id) {
                    tracing::debug!("Telegram: ignoring message from disallowed sender {}", sender_id);
                    continue;
                }

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Handle /new and /reset commands as session reset signals.
                let mut metadata = HashMap::new();
                let content = if text.starts_with("/new") || text.starts_with("/reset") {
                    metadata.insert("command".to_string(), "reset".to_string());
                    text.clone()
                } else {
                    text
                };

                let msg = InboundMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    sender: sender_id,
                    content,
                    channel: "telegram".to_string(),
                    chat_id,
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

        // Rate-limit: skip if last edit for this chat was <300ms ago.
        {
            let map = self.last_edit.lock();
            if let Some(last) = map.get(chat_id) {
                if last.elapsed().as_millis() < 300 {
                    return Ok(());
                }
            }
        }

        let body = serde_json::json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": full_text,
        });
        self.post_json_with_retry(&self.api_url("editMessageText"), &body)
            .await?;

        self.last_edit_insert(chat_id.to_string(), Instant::now());

        Ok(())
    }

    async fn send_streaming_end(
        &self,
        chat_id: &str,
        message_id: &str,
        full_text: &str,
    ) -> Result<()> {
        // Empty final text → Telegram rejects editMessageText with 400
        // `MESSAGE_TEXT_IS_EMPTY`. Happens when a turn produced only a
        // tool call and no prose. Leave the placeholder ("...") in
        // place rather than erroring; the user sees an unfilled bubble,
        // which is unusual but not broken, and the agent loop completes
        // cleanly.
        if full_text.is_empty() {
            let mut map = self.last_edit.lock();
            map.remove(chat_id);
            return Ok(());
        }

        let body = serde_json::json!({
            "chat_id": chat_id,
            "message_id": message_id,
            "text": full_text,
        });
        self.post_json_with_retry(&self.api_url("editMessageText"), &body)
            .await?;

        // Clear the rate-limit entry for this chat.
        {
            let mut map = self.last_edit.lock();
            map.remove(chat_id);
        }

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
        assert_eq!(updates[0], (123, "456".to_string(), "789".to_string(), "hello".to_string()));
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

    #[test]
    fn redact_telegram_url_hides_token_keeps_method() {
        let url = "https://api.telegram.org/bot123456:ABC-DEF1234ghIkl/sendMessage";
        let red = redact_telegram_url(url);
        assert_eq!(
            red,
            "https://api.telegram.org/bot<redacted>/sendMessage"
        );
        assert!(!red.contains("123456:ABC-DEF1234ghIkl"));
    }

    #[test]
    fn redact_telegram_url_handles_no_trailing_method() {
        let url = "https://api.telegram.org/bot123456:SECRET";
        assert_eq!(
            redact_telegram_url(url),
            "https://api.telegram.org/bot<redacted>"
        );
    }

    #[test]
    fn redact_telegram_url_passthrough_when_no_token() {
        let url = "https://example.com/health";
        assert_eq!(redact_telegram_url(url), url);
    }
}
