//! Tool: agent-initiated outbound messaging across configured channels.
//!
//! Three actions are supported:
//!
//!   - `send` (default): deliver a message to a target. The target is a
//!     string of the form `"<channel>"` (use the channel's configured
//!     home chat), `"<channel>:<chat_id>"` (explicit), or just a
//!     `<chat_id>` if the call also includes `channel`.
//!   - `list`: return the directory of `(channel, chat_id)` pairs the
//!     bot has seen recently, plus the configured home chat for each
//!     enabled channel. The LLM should call this first when it isn't
//!     sure where a particular contact is.
//!   - `home`: send to the home chat of the named channel without
//!     having to format a `<channel>:<chat_id>` target.
//!
//! ## Why a directory + home rather than a free-form `chat_id` arg
//!
//! The original tool took `(channel, chat_id, message)` and trusted the
//! LLM with whatever `chat_id` it produced. That made the agent
//! prompt-injection-vulnerable: an attacker who could place text in
//! something the agent read (a webpage, a PDF, a tool output) could
//! drive the agent to message arbitrary chats the bot was a member of.
//!
//! Layering a home-chat default + a directory-of-seen-chats on top
//! gives the LLM a curated menu to pick from. Numeric chat_id still
//! passes through (matches the analogous upstream-agent design and
//! covers legitimate "I learned this id from a webhook" cases) but the
//! tool's *typical* path is "ask, then send to a known chat."
//!
//! No new code was copied from any other agent project — the
//! mechanism is described in the comparison notes the project owner
//! reviewed; the implementation here is an independent Rust design
//! around the existing `ChatDirectory` and `home_chat_id` config.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::bus::ChatDirectory;
use crate::channels::traits::{Channel, SendMessage};
use crate::channels::ChannelMapHandle;

use super::traits::{Tool, ToolResult};

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// How far back the directory lookup is allowed to reach when the LLM
/// asks for a `list`. Anything older is omitted from the menu (but
/// `contains` still considers it a known chat for security gating).
const DIRECTORY_LOOKBACK: Duration = Duration::from_secs(30 * 24 * 3600); // 30d

/// Resolved (channel, chat_id) destination plus a description of how it
/// got resolved. The description goes into the success message so the
/// agent and the user can see whether it landed at the home chat, the
/// most-recent chat, or an explicit one.
#[derive(Debug, Clone)]
struct Destination {
    channel: String,
    chat_id: String,
    resolved_via: ResolvedVia,
}

#[derive(Debug, Clone, Copy)]
enum ResolvedVia {
    Explicit,
    HomeChannel,
    MostRecent,
}

impl ResolvedVia {
    fn label(self) -> &'static str {
        match self {
            ResolvedVia::Explicit => "explicit",
            ResolvedVia::HomeChannel => "home channel",
            ResolvedVia::MostRecent => "most recent inbound",
        }
    }
}

pub struct SendMessageTool {
    channels: ChannelMapHandle,
    directory: ChatDirectory,
    /// Map of channel name → home chat_id from config. Empty string
    /// means "no home configured for this channel"; the entry is still
    /// present so the `list` action can show "home: not configured."
    home_chats: HashMap<String, String>,
}

impl SendMessageTool {
    pub fn new(
        channels: ChannelMapHandle,
        directory: ChatDirectory,
        home_chats: HashMap<String, String>,
    ) -> Self {
        Self {
            channels,
            directory,
            home_chats,
        }
    }

    /// Resolve the LLM-supplied target into a concrete destination.
    ///
    /// Inputs the LLM can supply (in order of how the tool tries them):
    ///
    ///   - `target = "telegram:1234567890"` — explicit, parsed.
    ///   - `target = "telegram"` (no colon) — fall back to home_chat for
    ///     telegram, or most-recent inbound for telegram.
    ///   - `channel = "telegram"`, `chat_id = "1234567890"` — legacy
    ///     two-arg form; treated as explicit.
    ///   - `channel = "telegram"`, no chat_id — fall back to home_chat
    ///     or most-recent.
    fn resolve_target(&self, args: &Value) -> Result<Destination, String> {
        // Two-arg form for back-compat.
        if let (Some(channel), Some(chat_id)) = (
            args.get("channel").and_then(|v| v.as_str()),
            args.get("chat_id").and_then(|v| v.as_str()),
        ) {
            if !chat_id.is_empty() {
                return Ok(Destination {
                    channel: channel.to_string(),
                    chat_id: chat_id.to_string(),
                    resolved_via: ResolvedVia::Explicit,
                });
            }
        }

        // Single-arg form: parse `target`.
        let target = args
            .get("target")
            .and_then(|v| v.as_str())
            .or_else(|| args.get("channel").and_then(|v| v.as_str()))
            .ok_or_else(|| "missing 'target' (or 'channel') parameter".to_string())?;

        let (channel_name, explicit_chat_id) = match target.split_once(':') {
            Some((c, id)) if !c.is_empty() && !id.is_empty() => {
                (c.to_string(), Some(id.to_string()))
            }
            // `:foo`, `bar:`, or empty `:` — treat the whole thing as the
            // channel name and let the home/recent fallback take over.
            _ => (target.to_string(), None),
        };

        if let Some(chat_id) = explicit_chat_id {
            return Ok(Destination {
                channel: channel_name,
                chat_id,
                resolved_via: ResolvedVia::Explicit,
            });
        }

        // Try home_chat for this channel.
        if let Some(home) = self
            .home_chats
            .get(&channel_name)
            .filter(|s| !s.is_empty())
        {
            return Ok(Destination {
                channel: channel_name,
                chat_id: home.clone(),
                resolved_via: ResolvedVia::HomeChannel,
            });
        }

        // Try most-recent inbound for this channel.
        if let Some(recent) = self.directory.most_recent_for(&channel_name) {
            return Ok(Destination {
                channel: channel_name,
                chat_id: recent,
                resolved_via: ResolvedVia::MostRecent,
            });
        }

        Err(format!(
            "no destination available for channel '{}': no home_chat_id is \
             configured and no inbound message has been received from this \
             channel yet. Set channels.{}.home_chat_id in config, or use an \
             explicit '{}:CHAT_ID' target.",
            channel_name, channel_name, channel_name
        ))
    }

    fn lookup_channel(&self, name: &str) -> Option<Arc<dyn Channel>> {
        let map = self.channels.read();
        map.get(name).cloned()
    }

    fn build_directory_listing(&self) -> Value {
        let mut by_channel: HashMap<String, Vec<Value>> = HashMap::new();
        for entry in self.directory.list_recent(DIRECTORY_LOOKBACK) {
            by_channel
                .entry(entry.channel.clone())
                .or_default()
                .push(json!({
                    "chat_id": entry.chat_id,
                    "seconds_since_seen": entry.last_seen.elapsed().as_secs(),
                }));
        }
        let mut channels = serde_json::Map::new();
        // Include every configured channel, every channel with a home_chat
        // pin, AND every channel that has a recorded entry. The third
        // chain matters: a recording for a channel name that isn't in the
        // configured map (e.g. test fixture, late-bound channel) would
        // otherwise be silently dropped from the listing.
        let known_channels: std::collections::BTreeSet<String> = self
            .channels
            .read()
            .keys()
            .cloned()
            .chain(self.home_chats.keys().cloned())
            .chain(by_channel.keys().cloned())
            .collect();
        for name in known_channels {
            let recent = by_channel.remove(&name).unwrap_or_default();
            let home = self
                .home_chats
                .get(&name)
                .filter(|s| !s.is_empty())
                .map(|s| Value::String(s.clone()))
                .unwrap_or(Value::Null);
            channels.insert(
                name,
                json!({
                    "home_chat_id": home,
                    "recent_chats": recent,
                }),
            );
        }
        Value::Object(channels)
    }
}

#[async_trait]
impl Tool for SendMessageTool {
    fn name(&self) -> &str {
        "send_message"
    }

    fn description(&self) -> &str {
        "Send a message to a chat the bot is in, or list available targets. \
         Prefer to call action='list' first when you're not sure where a \
         particular contact lives. The default 'send' action accepts a \
         target like 'telegram' (sends to the configured home chat or the \
         most recent inbound), 'telegram:CHAT_ID' (explicit), or the legacy \
         two-arg form (channel + chat_id)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["send", "list"],
                    "description": "Default 'send'. 'list' returns the directory of channels with their home chats and recently-seen chats."
                },
                "target": {
                    "type": "string",
                    "description": "Destination. Either '<channel>' (use home/most-recent) or '<channel>:<chat_id>' (explicit). Required for 'send' unless 'channel' is provided."
                },
                "channel": {
                    "type": "string",
                    "description": "Legacy alternative to 'target'. Channel name only."
                },
                "chat_id": {
                    "type": "string",
                    "description": "Legacy alternative to 'target'. Explicit chat_id; combine with 'channel'."
                },
                "message": {
                    "type": "string",
                    "description": "Message text. Required for 'send'."
                }
            }
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolResult> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("send");

        if action == "list" {
            let listing = self.build_directory_listing();
            return Ok(ToolResult {
                success: true,
                output: serde_json::to_string_pretty(&listing).unwrap_or_default(),
                error: None,
            });
        }

        if action != "send" {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "unknown action '{}', expected 'send' or 'list'",
                    action
                )),
            });
        }

        let message = match args.get("message").and_then(|v| v.as_str()) {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: message".to_string()),
                });
            }
        };

        let dest = match self.resolve_target(&args) {
            Ok(d) => d,
            Err(msg) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(msg),
                });
            }
        };

        let channel = match self.lookup_channel(&dest.channel) {
            Some(ch) => ch,
            None => {
                let available: Vec<String> = self.channels.read().keys().cloned().collect();
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "channel '{}' not registered. Available: [{}]",
                        dest.channel,
                        available.join(", ")
                    )),
                });
            }
        };

        let send_msg = SendMessage::new(&message, &dest.chat_id);
        match channel.send(&send_msg).await {
            Ok(()) => Ok(ToolResult {
                success: true,
                output: format!(
                    "Sent to {}:{} (resolved via {})",
                    dest.channel,
                    dest.chat_id,
                    dest.resolved_via.label(),
                ),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("failed to send: {e}")),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::new_channel_map;

    fn empty_homes() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn list_action_includes_known_channels() {
        let channels = new_channel_map();
        let dir = ChatDirectory::new();
        dir.record("telegram", "100");
        let mut homes = HashMap::new();
        homes.insert("telegram".to_string(), "100".to_string());
        let tool = SendMessageTool::new(channels, dir, homes);
        let listing = tool.build_directory_listing();
        let obj = listing.as_object().unwrap();
        assert!(obj.contains_key("telegram"));
        assert_eq!(obj["telegram"]["home_chat_id"], json!("100"));
    }

    #[test]
    fn resolve_explicit_colon_target() {
        let tool = SendMessageTool::new(new_channel_map(), ChatDirectory::new(), empty_homes());
        let d = tool
            .resolve_target(&json!({"target": "telegram:42"}))
            .unwrap();
        assert_eq!(d.channel, "telegram");
        assert_eq!(d.chat_id, "42");
        assert!(matches!(d.resolved_via, ResolvedVia::Explicit));
    }

    #[test]
    fn resolve_uses_home_when_no_chat_id() {
        let mut homes = HashMap::new();
        homes.insert("telegram".to_string(), "9999".to_string());
        let tool = SendMessageTool::new(new_channel_map(), ChatDirectory::new(), homes);
        let d = tool.resolve_target(&json!({"target": "telegram"})).unwrap();
        assert_eq!(d.chat_id, "9999");
        assert!(matches!(d.resolved_via, ResolvedVia::HomeChannel));
    }

    #[test]
    fn resolve_falls_back_to_most_recent() {
        let dir = ChatDirectory::new();
        dir.record("telegram", "111");
        std::thread::sleep(Duration::from_millis(2));
        dir.record("telegram", "222");
        let tool = SendMessageTool::new(new_channel_map(), dir, empty_homes());
        let d = tool.resolve_target(&json!({"target": "telegram"})).unwrap();
        assert_eq!(d.chat_id, "222");
        assert!(matches!(d.resolved_via, ResolvedVia::MostRecent));
    }

    #[test]
    fn resolve_errors_when_no_chat_known_and_no_home() {
        let tool = SendMessageTool::new(new_channel_map(), ChatDirectory::new(), empty_homes());
        let r = tool.resolve_target(&json!({"target": "telegram"}));
        let err = r.unwrap_err();
        assert!(err.contains("home_chat_id"), "got: {err}");
        assert!(err.contains("explicit"), "got: {err}");
    }

    #[test]
    fn resolve_legacy_two_arg_form() {
        let tool = SendMessageTool::new(new_channel_map(), ChatDirectory::new(), empty_homes());
        let d = tool
            .resolve_target(&json!({"channel": "discord", "chat_id": "5"}))
            .unwrap();
        assert_eq!(d.channel, "discord");
        assert_eq!(d.chat_id, "5");
        assert!(matches!(d.resolved_via, ResolvedVia::Explicit));
    }

    #[tokio::test]
    async fn execute_action_list_returns_directory() {
        let channels = new_channel_map();
        let dir = ChatDirectory::new();
        dir.record("telegram", "1");
        let tool = SendMessageTool::new(channels, dir, empty_homes());
        let r = tool.execute(json!({"action": "list"})).await.unwrap();
        assert!(r.success);
        assert!(r.output.contains("telegram"));
    }

    #[tokio::test]
    async fn execute_send_without_message_fails() {
        let tool = SendMessageTool::new(new_channel_map(), ChatDirectory::new(), empty_homes());
        let r = tool
            .execute(json!({"target": "telegram:1"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("message"));
    }

    #[tokio::test]
    async fn execute_send_with_unknown_action_fails() {
        let tool = SendMessageTool::new(new_channel_map(), ChatDirectory::new(), empty_homes());
        let r = tool
            .execute(json!({"action": "delete", "target": "telegram:1"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("unknown action"));
    }
}
