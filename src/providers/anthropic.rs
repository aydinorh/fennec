use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::sse::SseBuffer;
use super::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, StreamEvent, ToolCall, UsageInfo,
};

/// How the Anthropic provider authenticates its requests.
#[derive(Debug, Clone)]
pub enum AnthropicAuthMode {
    /// Traditional API key sent as `x-api-key` header.
    ApiKey(String),
    /// OAuth Bearer token sent as `Authorization: Bearer <token>`.
    OAuthBearer(String),
}

/// Anthropic Claude API provider.
pub struct AnthropicProvider {
    auth: AnthropicAuthMode,
    client: reqwest::Client,
    default_model: String,
}

/// What kind of content block occupies a particular `index` in the
/// streaming response. We remember this so `content_block_stop` can tell
/// which block types deserve a `ToolCallEnd` event.
#[derive(Debug, Clone)]
enum ToolBlockInfo {
    ToolUse(String),
    Other,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider using an API key.
    ///
    /// - `api_key`: Anthropic API key.
    /// - `model`: Override the default model. Defaults to `claude-sonnet-4-6`
    ///   (Sonnet 4.6 alias). The previous default `claude-sonnet-4-20250514`
    ///   is deprecated by Anthropic and retires June 15, 2026.
    pub fn new(api_key: String, model: Option<String>) -> Self {
        Self {
            auth: AnthropicAuthMode::ApiKey(api_key),
            client: reqwest::Client::new(),
            default_model: model.unwrap_or_else(|| "claude-sonnet-4-6".to_string()),
        }
    }

    /// Create a new Anthropic provider using an OAuth Bearer token.
    pub fn new_with_oauth(token: String, model: Option<String>) -> Self {
        Self {
            auth: AnthropicAuthMode::OAuthBearer(token),
            client: reqwest::Client::new(),
            default_model: model.unwrap_or_else(|| "claude-sonnet-4-6".to_string()),
        }
    }

    /// Apply the appropriate authentication header to a request builder.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth {
            AnthropicAuthMode::ApiKey(key) => builder.header("x-api-key", key),
            AnthropicAuthMode::OAuthBearer(token) => {
                builder
                    .header("Authorization", format!("Bearer {}", token))
                    // OAuth requires beta headers to work with the Messages API
                    .header("anthropic-beta", "claude-code-20250219,oauth-2025-04-20")
                    .header("User-Agent", "claude-cli/1.0 (external, cli)")
            }
        }
    }

    /// Convert our ChatMessages to the Anthropic API message format.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<Value> {
        let mut api_messages = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    // System messages are handled separately via the top-level `system` field.
                    // Skip them here.
                }
                "assistant" => {
                    if let Some(ref tool_calls) = msg.tool_calls {
                        // Assistant message with tool use blocks.
                        let mut content_blocks = Vec::new();

                        // Include text content if present.
                        if let Some(ref text) = msg.content {
                            if !text.is_empty() {
                                content_blocks.push(json!({
                                    "type": "text",
                                    "text": text
                                }));
                            }
                        }

                        // Add tool_use blocks.
                        for tc in tool_calls {
                            content_blocks.push(json!({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments
                            }));
                        }

                        api_messages.push(json!({
                            "role": "assistant",
                            "content": content_blocks
                        }));
                    } else {
                        api_messages.push(json!({
                            "role": "assistant",
                            "content": msg.content.as_deref().unwrap_or("")
                        }));
                    }
                }
                "tool" => {
                    // Tool results become user messages with tool_result content blocks.
                    let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
                    let content = msg.content.as_deref().unwrap_or("");

                    api_messages.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content
                        }]
                    }));
                }
                _ => {
                    // "user" and anything else. If there are
                    // image attachments queued for this turn,
                    // serialise them as image blocks alongside
                    // the text. Anthropic expects each image as
                    // {type: image, source: {type: base64,
                    // media_type, data}}.
                    let text = msg.content.as_deref().unwrap_or("");
                    if let Some(ref atts) = msg.attachments {
                        let mut blocks = Vec::with_capacity(atts.len() + 1);
                        if !text.is_empty() {
                            blocks.push(json!({"type": "text", "text": text}));
                        }
                        for a in atts {
                            blocks.push(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": a.mime_type,
                                    "data": a.base64_data,
                                }
                            }));
                        }
                        api_messages.push(json!({
                            "role": msg.role,
                            "content": blocks,
                        }));
                    } else {
                        api_messages.push(json!({
                            "role": msg.role,
                            "content": text,
                        }));
                    }
                }
            }
        }

        api_messages
    }

    /// Convert our ToolSpec list to Anthropic's tools format.
    fn convert_tools(tools: &[crate::tools::traits::ToolSpec]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                })
            })
            .collect()
    }

    /// Build the request body for the Anthropic API (shared between chat and chat_stream).
    fn build_request_body(
        &self,
        request: &ChatRequest<'_>,
        stream: bool,
    ) -> Value {
        let messages = Self::convert_messages(request.messages);

        let mut body = json!({
            "model": self.default_model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
        });

        if stream {
            body["stream"] = json!(true);
        }

        if let Some(system_text) = request.system {
            body["system"] = json!([{
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"}
            }]);
        }

        if let Some(tools) = request.tools {
            if !tools.is_empty() {
                body["tools"] = json!(Self::convert_tools(tools));
            }
        }

        // Apply extended thinking parameters if the agent selected a level.
        crate::agent::thinking::apply_thinking_params(
            &mut body,
            request.thinking_level,
            "anthropic",
        );

        Self::add_cache_breakpoints(&mut body);

        body
    }

    /// Add `cache_control` breakpoints so Anthropic caches the static request
    /// prefix. The system prompt is already marked above; this also caches the
    /// tools block (stable for the whole session) and the conversation prefix
    /// (a rolling breakpoint on the last message). On multi-turn tool loops
    /// this cuts re-billed input tokens sharply — and a stale/short prefix
    /// simply misses the cache rather than erroring, so it's always safe.
    /// Stays within Anthropic's 4-breakpoint limit (system + tools + prefix).
    fn add_cache_breakpoints(body: &mut Value) {
        let ephemeral = || json!({ "type": "ephemeral" });

        // Cache the tools block (mark the last tool).
        if let Some(last_tool) = body
            .get_mut("tools")
            .and_then(|t| t.as_array_mut())
            .and_then(|tools| tools.last_mut())
        {
            last_tool["cache_control"] = ephemeral();
        }

        // Cache the conversation prefix (mark the last content block of the
        // last message). String content is promoted to a text block so the
        // marker has somewhere to live.
        if let Some(last_msg) = body
            .get_mut("messages")
            .and_then(|m| m.as_array_mut())
            .and_then(|msgs| msgs.last_mut())
        {
            match last_msg.get_mut("content") {
                Some(Value::Array(blocks)) => {
                    if let Some(last_block) = blocks.last_mut() {
                        last_block["cache_control"] = ephemeral();
                    }
                }
                Some(content @ Value::String(_)) => {
                    let text = content.as_str().unwrap_or("").to_string();
                    *content = json!([{
                        "type": "text",
                        "text": text,
                        "cache_control": ephemeral(),
                    }]);
                }
                _ => {}
            }
        }
    }

    /// Parse a single SSE event from Anthropic's streaming API and emit
    /// [`StreamEvent`]s on the provided sender.
    ///
    /// `index_to_tool` maps each `content_block_*` event's `index` to the
    /// `tool_use_id` advertised at `content_block_start`. Without this
    /// mapping, `input_json_delta` events (which carry only an index) and
    /// `content_block_stop` events can't be correlated back to the right
    /// tool call. Text blocks are tracked with a sentinel so we know NOT
    /// to emit a spurious `ToolCallEnd` for them.
    /// Returns `true` if this event ended the stream (i.e. emitted
    /// `StreamEvent::Done`). The caller uses this flag to suppress the
    /// post-loop "Done anyway" fallback, which previously ran
    /// unconditionally and produced duplicate `Done` events on every
    /// successful stream.
    async fn handle_sse_event(
        event_type: &str,
        data: &Value,
        tx: &tokio::sync::mpsc::Sender<StreamEvent>,
        index_to_tool: &mut HashMap<u64, ToolBlockInfo>,
        usage_acc: &mut UsageInfo,
    ) -> bool {
        match event_type {
            "message_start" => {
                // Anthropic emits initial usage (input_tokens, cache_read,
                // cache_creation) on the message_start event. Output tokens
                // accumulate via message_delta as the model emits text.
                if let Some(u) = data.get("message").and_then(|m| m.get("usage")) {
                    if let Some(v) = u.get("input_tokens").and_then(|v| v.as_u64()) {
                        usage_acc.input_tokens = v;
                    }
                    if let Some(v) = u.get("cache_read_input_tokens").and_then(|v| v.as_u64()) {
                        usage_acc.cache_read_tokens = Some(v);
                    }
                    if let Some(v) = u
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        usage_acc.cache_write_tokens = Some(v);
                    }
                }
            }
            "message_delta" => {
                // Output tokens land here, cumulative.
                if let Some(u) = data.get("usage") {
                    if let Some(v) = u.get("output_tokens").and_then(|v| v.as_u64()) {
                        usage_acc.output_tokens = v;
                    }
                }
            }
            "content_block_start" => {
                let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let Some(block) = data.get("content_block") else { return false };
                let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match block_type {
                    "tool_use" => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        index_to_tool.insert(idx, ToolBlockInfo::ToolUse(id.clone()));
                        let _ = tx.send(StreamEvent::ToolCallStart { id, name }).await;
                    }
                    _ => {
                        // Text (or future block types) — remember it's NOT a
                        // tool_use so the matching stop doesn't emit ToolCallEnd.
                        index_to_tool.insert(idx, ToolBlockInfo::Other);
                    }
                }
            }
            "content_block_delta" => {
                let Some(delta) = data.get("delta") else { return false };
                match delta.get("type").and_then(|t| t.as_str()) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                            let _ = tx.send(StreamEvent::Delta(text.to_string())).await;
                        }
                    }
                    Some("thinking_delta") => {
                        // Extended-thinking text. Anthropic emits
                        // these on content_block_delta frames whose
                        // owning content_block was a `type: thinking`
                        // block — same shape as text_delta but with
                        // the `thinking` field instead of `text`.
                        if let Some(text) =
                            delta.get("thinking").and_then(|t| t.as_str())
                        {
                            let _ = tx
                                .send(StreamEvent::Reasoning(text.to_string()))
                                .await;
                        }
                    }
                    Some("input_json_delta") => {
                        if let Some(partial) =
                            delta.get("partial_json").and_then(|t| t.as_str())
                        {
                            let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                            // Resolve index → real tool_use_id.
                            let Some(ToolBlockInfo::ToolUse(id)) = index_to_tool.get(&idx)
                            else {
                                // Delta for a block we never saw a tool_use start
                                // for — drop it rather than emit with a wrong id.
                                return false;
                            };
                            let _ = tx
                                .send(StreamEvent::ToolCallDelta {
                                    id: id.clone(),
                                    arguments_delta: partial.to_string(),
                                })
                                .await;
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let idx = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                // Only emit ToolCallEnd for tool_use blocks. Text blocks
                // have their own completion signaled by the absence of
                // further text_deltas; surfacing a phantom ToolCallEnd for
                // them confuses consumers tracking tool state by id.
                if let Some(ToolBlockInfo::ToolUse(id)) = index_to_tool.remove(&idx) {
                    let _ = tx.send(StreamEvent::ToolCallEnd { id }).await;
                }
            }
            "message_stop" => {
                // Emit accumulated usage just before terminating the
                // stream so the agent's turn_streaming() can update its
                // session totals on the same path the non-streaming
                // turn() reads `ChatResponse.usage`.
                let _ = tx.send(StreamEvent::Usage(usage_acc.clone())).await;
                let _ = tx.send(StreamEvent::Done).await;
                return true;
            }
            "error" => {
                let msg = data
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown streaming error");
                let _ = tx.send(StreamEvent::Error(msg.to_string())).await;
                // Surface the error as a terminal event too so consumers
                // that wait for `Done` don't hang when the server emits
                // `error` instead. Treat error as the terminator.
                let _ = tx.send(StreamEvent::Done).await;
                return true;
            }
            _ => { /* message_start, ping, etc. — ignore */ }
        }
        false
    }

    /// Parse the Anthropic response JSON into our ChatResponse.
    fn parse_response(body: &Value) -> Result<ChatResponse> {
        let mut text_parts = Vec::new();
        let mut reasoning_parts = Vec::new();
        let mut tool_calls = Vec::new();

        if let Some(content) = body.get("content").and_then(|c| c.as_array()) {
            for block in content {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    Some("thinking") => {
                        // Extended-thinking blocks (Claude 4.x). The
                        // text lives in `thinking` (not `text`).
                        // Ignore signature / cache fields — those are
                        // for the model's internal verification.
                        if let Some(text) =
                            block.get("thinking").and_then(|t| t.as_str())
                        {
                            reasoning_parts.push(text.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments = block
                            .get("input")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);

                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    _ => {}
                }
            }
        }

        let content = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };
        let reasoning = if reasoning_parts.is_empty() {
            None
        } else {
            Some(reasoning_parts.join("\n"))
        };

        // Parse usage info.
        let usage = body.get("usage").map(|u| UsageInfo {
            input_tokens: u
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: u
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_read_tokens: u
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_u64()),
            cache_write_tokens: u
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_u64()),
        });

        Ok(ChatResponse {
            content,
            tool_calls,
            usage,
            reasoning,
        })
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.default_model
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let body = self.build_request_body(&request, false);

        let req = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body);
        let req = self.apply_auth(req);

        let response = req
            .send()
            .await
            .context("sending request to Anthropic API")?;

        // Read raw bytes first so a non-JSON error body (e.g. a proxy/gateway
        // 502 HTML page) doesn't lose the real HTTP status context. The old
        // order — .json() first, then check status — produced a cryptic
        // "parsing Anthropic API response" failure with no status/body info.
        let status = response.status();
        let raw_body = response
            .bytes()
            .await
            .context("reading Anthropic API response body")?;

        if !status.is_success() {
            let error_msg = serde_json::from_slice::<Value>(&raw_body)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| {
                    let preview = String::from_utf8_lossy(&raw_body);
                    preview.chars().take(200).collect::<String>()
                });
            anyhow::bail!("Anthropic API error ({}): {}", status, error_msg);
        }

        let response_body: Value = serde_json::from_slice(&raw_body)
            .context("parsing Anthropic API response as JSON")?;
        Self::parse_response(&response_body)
    }

    fn supports_tool_calling(&self) -> bool {
        true
    }

    fn context_window(&self) -> usize {
        200_000
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat_stream(
        &self,
        request: ChatRequest<'_>,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        let body = self.build_request_body(&request, true);

        let req = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body);
        let req = self.apply_auth(req);

        let response = req
            .send()
            .await
            .context("sending streaming request to Anthropic API")?;

        let status = response.status();
        if !status.is_success() {
            // Defensive body-read: non-streaming 4xx/5xx may not be JSON.
            let raw_body = response
                .bytes()
                .await
                .context("reading Anthropic streaming error body")?;
            let error_msg = serde_json::from_slice::<Value>(&raw_body)
                .ok()
                .and_then(|v| {
                    v.get("error")
                        .and_then(|e| e.get("message"))
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| {
                    String::from_utf8_lossy(&raw_body)
                        .chars()
                        .take(200)
                        .collect()
                });
            anyhow::bail!("Anthropic API error ({}): {}", status, error_msg);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let mut byte_stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut sse = SseBuffer::new();
            let mut current_event_type = String::new();
            let mut index_to_tool: HashMap<u64, ToolBlockInfo> = HashMap::new();
            let mut usage_acc = UsageInfo::default();
            // Tracks whether `handle_sse_event` already emitted Done (via
            // `message_stop` or `error`). Without this, the post-loop
            // "Done anyway" fallback fires on every successful stream too,
            // and consumers receive `Done` twice.
            let mut done_sent = false;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                        // Surface a terminator after the error so consumers
                        // waiting for `Done` don't hang.
                        if !done_sent {
                            let _ = tx.send(StreamEvent::Done).await;
                        }
                        return;
                    }
                };
                sse.extend(&chunk);

                while let Some(line_bytes) = sse.next_line() {
                    // Decode once we have a complete line — safe from the
                    // UTF-8-split-across-chunks bug.
                    let line = match std::str::from_utf8(&line_bytes) {
                        Ok(s) => s,
                        Err(_) => continue,
                    };

                    if line.is_empty() {
                        current_event_type.clear();
                        continue;
                    }

                    if let Some(evt) = line.strip_prefix("event: ") {
                        current_event_type = evt.to_string();
                    } else if let Some(data_str) = line.strip_prefix("data: ") {
                        if let Ok(data) = serde_json::from_str::<Value>(data_str) {
                            let terminated = Self::handle_sse_event(
                                &current_event_type,
                                &data,
                                &tx,
                                &mut index_to_tool,
                                &mut usage_acc,
                            )
                            .await;
                            if terminated {
                                done_sent = true;
                            }
                        }
                    }
                }
            }

            // Fallback: if we drained the byte stream without ever seeing a
            // `message_stop` or `error` SSE event (e.g. server closed the
            // connection mid-stream), surface a single Done so consumers
            // don't hang. If `done_sent` is already true, do nothing —
            // consumers have already received their terminator.
            if !done_sent {
                let _ = tx.send(StreamEvent::Done).await;
            }
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// Drive `handle_sse_event` against a fixed sequence of events and
    /// return every emitted StreamEvent. Verifies the index→id stitching.
    async fn replay(events: Vec<(&str, Value)>) -> Vec<StreamEvent> {
        let (tx, mut rx) = mpsc::channel(64);
        let mut map: HashMap<u64, ToolBlockInfo> = HashMap::new();
        let mut usage = UsageInfo::default();
        for (ty, data) in events {
            AnthropicProvider::handle_sse_event(ty, &data, &tx, &mut map, &mut usage).await;
        }
        drop(tx);
        let mut out = Vec::new();
        while let Some(e) = rx.recv().await {
            out.push(e);
        }
        out
    }

    fn is_tool_call_delta_for(e: &StreamEvent, expected_id: &str) -> bool {
        matches!(
            e,
            StreamEvent::ToolCallDelta { id, .. } if id == expected_id
        )
    }

    #[test]
    fn cache_breakpoints_on_system_tools_and_prefix() {
        let p = AnthropicProvider::new("k".to_string(), Some("claude-sonnet-4-6".to_string()));
        let messages = vec![ChatMessage::user("hello there")];
        let specs = vec![
            crate::tools::traits::ToolSpec {
                name: "a".to_string(),
                description: "first".to_string(),
                parameters: json!({ "type": "object", "properties": {} }),
            },
            crate::tools::traits::ToolSpec {
                name: "b".to_string(),
                description: "last".to_string(),
                parameters: json!({ "type": "object", "properties": {} }),
            },
        ];
        let request = ChatRequest {
            system: Some("be terse"),
            messages: &messages,
            tools: Some(&specs),
            max_tokens: 1024,
            temperature: 0.7,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };
        let body = p.build_request_body(&request, false);

        // System block is cached.
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        // The LAST tool is cached (and only the last).
        let tools = body["tools"].as_array().unwrap();
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
        // The last message's content (string) was promoted to a cached text block.
        let last_content = &body["messages"][0]["content"];
        assert_eq!(last_content[0]["type"], "text");
        assert_eq!(last_content[0]["text"], "hello there");
        assert_eq!(last_content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn cache_breakpoint_marks_last_block_of_array_content() {
        // A tool-result turn (array content) gets the marker on its last block.
        let p = AnthropicProvider::new("k".to_string(), Some("claude-sonnet-4-6".to_string()));
        let messages = vec![ChatMessage::tool_result("tc_1", "the result")];
        let request = ChatRequest {
            system: None,
            messages: &messages,
            tools: None,
            max_tokens: 512,
            temperature: 0.5,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };
        let body = p.build_request_body(&request, false);
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.last().unwrap()["type"], "tool_result");
        assert_eq!(blocks.last().unwrap()["cache_control"]["type"], "ephemeral");
    }

    #[tokio::test]
    async fn tool_call_delta_uses_real_id_not_index() {
        // Event shape from Anthropic:
        //   content_block_start  index=1, content_block={type: tool_use, id: "toolu_abc", name: "read_file"}
        //   content_block_delta  index=1, delta={type: input_json_delta, partial_json: "{\"pat"}
        //   content_block_delta  index=1, delta={type: input_json_delta, partial_json: "h\":\"/tmp\"}"}
        //   content_block_stop   index=1
        let events = vec![
            (
                "content_block_start",
                json!({
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "read_file"
                    }
                }),
            ),
            (
                "content_block_delta",
                json!({
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": "{\"pat"
                    }
                }),
            ),
            (
                "content_block_delta",
                json!({
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": "h\":\"/tmp\"}"
                    }
                }),
            ),
            ("content_block_stop", json!({ "index": 1 })),
        ];

        let out = replay(events).await;

        // Should emit: Start{id=toolu_abc}, Delta{id=toolu_abc}, Delta{id=toolu_abc}, End{id=toolu_abc}
        assert_eq!(out.len(), 4, "events: {:?}", out);
        assert!(matches!(&out[0], StreamEvent::ToolCallStart { id, name }
            if id == "toolu_abc" && name == "read_file"));
        assert!(is_tool_call_delta_for(&out[1], "toolu_abc"));
        assert!(is_tool_call_delta_for(&out[2], "toolu_abc"));
        assert!(matches!(&out[3], StreamEvent::ToolCallEnd { id } if id == "toolu_abc"));
    }

    #[tokio::test]
    async fn text_block_stop_does_not_emit_tool_call_end() {
        // Text block at index 0 should NOT produce a ToolCallEnd at stop.
        // Previous implementation emitted ToolCallEnd { id: "0" } which
        // confused consumers tracking tool state.
        let events = vec![
            (
                "content_block_start",
                json!({
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                }),
            ),
            (
                "content_block_delta",
                json!({
                    "index": 0,
                    "delta": { "type": "text_delta", "text": "Hello" }
                }),
            ),
            ("content_block_stop", json!({ "index": 0 })),
        ];

        let out = replay(events).await;

        // Only one emission expected: the Delta.
        assert_eq!(out.len(), 1, "events: {:?}", out);
        assert!(matches!(&out[0], StreamEvent::Delta(t) if t == "Hello"));
    }

    #[tokio::test]
    async fn mixed_text_and_tool_blocks_stitch_correctly() {
        // Realistic: text at index 0, tool_use at index 1 (with different id),
        // interleaved deltas. Ids must never get confused.
        let events = vec![
            (
                "content_block_start",
                json!({
                    "index": 0,
                    "content_block": { "type": "text", "text": "" }
                }),
            ),
            (
                "content_block_delta",
                json!({ "index": 0, "delta": { "type": "text_delta", "text": "Let me " } }),
            ),
            (
                "content_block_start",
                json!({
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_xyz",
                        "name": "search"
                    }
                }),
            ),
            (
                "content_block_delta",
                json!({ "index": 0, "delta": { "type": "text_delta", "text": "search." } }),
            ),
            (
                "content_block_delta",
                json!({
                    "index": 1,
                    "delta": { "type": "input_json_delta", "partial_json": "{\"q\":\"x\"}" }
                }),
            ),
            ("content_block_stop", json!({ "index": 0 })),
            ("content_block_stop", json!({ "index": 1 })),
            ("message_stop", json!({})),
        ];

        let out = replay(events).await;

        // Expect: tool Start, text Delta, text Delta, tool Delta, tool End, Done.
        // The text block stop must NOT produce a ToolCallEnd.
        let tool_ends: Vec<&StreamEvent> = out
            .iter()
            .filter(|e| matches!(e, StreamEvent::ToolCallEnd { .. }))
            .collect();
        assert_eq!(tool_ends.len(), 1, "expected exactly 1 ToolCallEnd, got {:?}", tool_ends);
        let StreamEvent::ToolCallEnd { id } = tool_ends[0] else { unreachable!() };
        assert_eq!(id, "toolu_xyz");

        // And every ToolCallDelta uses toolu_xyz.
        for e in &out {
            if let StreamEvent::ToolCallDelta { id, .. } = e {
                assert_eq!(id, "toolu_xyz", "unexpected tool delta id");
            }
        }
    }

    #[tokio::test]
    async fn delta_without_prior_start_is_dropped() {
        // If we somehow see a delta for an index we never saw start, we
        // must NOT emit with an incorrect id (the old code would emit
        // with id=index_as_string). Dropping is the safe choice.
        let events = vec![(
            "content_block_delta",
            json!({
                "index": 42,
                "delta": { "type": "input_json_delta", "partial_json": "{}" }
            }),
        )];
        let out = replay(events).await;
        assert!(out.is_empty(), "orphan delta should be dropped: {:?}", out);
    }

    /// Regression: previously the SSE handler emitted `Done` on
    /// `message_stop` AND the post-loop fallback also unconditionally
    /// emitted `Done`, so consumers saw Done twice on every successful
    /// stream. With `handle_sse_event` returning `bool` and the post-
    /// loop checking the flag, exactly one Done should be emitted from
    /// the handler when `message_stop` arrives.
    #[tokio::test]
    async fn message_stop_emits_done_exactly_once() {
        let out = replay(vec![("message_stop", json!({}))]).await;
        let dones = out
            .iter()
            .filter(|e| matches!(e, StreamEvent::Done))
            .count();
        assert_eq!(dones, 1, "expected exactly one Done, got {:?}", out);
    }

    /// `error` SSE event must surface both `Error` (so the consumer
    /// knows what went wrong) and `Done` (as a terminator so any
    /// `await`-on-Done loop terminates). It must NOT depend on the
    /// post-loop fallback for the terminator.
    /// `message_start` carries initial usage counts (input,
    /// cache_read, cache_creation); `message_delta` carries the
    /// running output_tokens; `message_stop` must surface those
    /// as a single `StreamEvent::Usage` immediately before
    /// `Done`. Without this the agent's streaming-turn path
    /// would never see token counts and `/usage` would silently
    /// drop streaming-mode usage.
    #[tokio::test]
    async fn message_stop_emits_usage_event_with_aggregated_counts() {
        let out = replay(vec![
            (
                "message_start",
                json!({
                    "message": {
                        "usage": {
                            "input_tokens": 100,
                            "cache_read_input_tokens": 50,
                            "cache_creation_input_tokens": 10,
                        }
                    }
                }),
            ),
            (
                "message_delta",
                json!({"usage": {"output_tokens": 42}}),
            ),
            ("message_stop", json!({})),
        ])
        .await;
        // Find the Usage event.
        let usage = out
            .iter()
            .find_map(|e| match e {
                StreamEvent::Usage(u) => Some(u.clone()),
                _ => None,
            })
            .expect("expected a Usage event before Done");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 42);
        assert_eq!(usage.cache_read_tokens, Some(50));
        assert_eq!(usage.cache_write_tokens, Some(10));
        // Usage must come before Done.
        let usage_idx = out.iter().position(|e| matches!(e, StreamEvent::Usage(_)));
        let done_idx = out.iter().position(|e| matches!(e, StreamEvent::Done));
        assert!(
            usage_idx < done_idx,
            "Usage must be emitted before Done; got {out:?}"
        );
    }

    #[tokio::test]
    async fn thinking_delta_emits_reasoning_event() {
        // Anthropic streams extended-thinking text as
        // `content_block_delta` frames whose `delta.type` is
        // `thinking_delta`. We must surface those as
        // `StreamEvent::Reasoning(text)` so the agent's
        // streaming path can forward them to `on_reasoning_delta`.
        let out = replay(vec![
            (
                "content_block_start",
                json!({
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""}
                }),
            ),
            (
                "content_block_delta",
                json!({
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "let me think... "}
                }),
            ),
            (
                "content_block_delta",
                json!({
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "step 1: parse"}
                }),
            ),
        ])
        .await;
        let reasonings: Vec<String> = out
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Reasoning(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(reasonings.len(), 2, "got: {out:?}");
        assert_eq!(reasonings[0], "let me think... ");
        assert_eq!(reasonings[1], "step 1: parse");
    }

    #[test]
    fn parse_response_extracts_thinking_blocks() {
        // Non-streaming response with mixed thinking + text +
        // tool_use blocks. parse_response must populate the
        // `reasoning` field with the thinking text and `content`
        // with the text-only portion.
        let body = json!({
            "content": [
                {"type": "thinking", "thinking": "first I should check X"},
                {"type": "text", "text": "Sure! "},
                {"type": "thinking", "thinking": "now Y"},
                {"type": "text", "text": "Here's my answer."},
            ]
        });
        let resp = AnthropicProvider::parse_response(&body).unwrap();
        assert_eq!(
            resp.reasoning.as_deref(),
            Some("first I should check X\nnow Y")
        );
        assert_eq!(resp.content.as_deref(), Some("Sure! Here's my answer."));
    }

    #[tokio::test]
    async fn error_event_emits_error_then_done() {
        let out = replay(vec![(
            "error",
            json!({"error": {"message": "boom"}}),
        )])
        .await;
        let mut iter = out.iter();
        match iter.next() {
            Some(StreamEvent::Error(m)) => assert_eq!(m, "boom"),
            other => panic!("expected Error first, got {:?}", other),
        }
        assert!(matches!(iter.next(), Some(StreamEvent::Done)));
        assert!(iter.next().is_none(), "no events should follow Done");
    }
}
