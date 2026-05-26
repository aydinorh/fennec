use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::sse::SseBuffer;
use super::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, StreamEvent, ToolCall, UsageInfo,
};

/// OpenAI-compatible API provider.
///
/// Works with OpenAI, Azure OpenAI, and any API that follows the
/// OpenAI chat completions format.
pub struct OpenAIProvider {
    api_key: String,
    client: reqwest::Client,
    model: String,
    base_url: String,
    ctx_window: usize,
    extra_headers: Vec<(String, String)>,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider.
    ///
    /// Defaults: model `"gpt-4o"`, base_url `"https://api.openai.com/v1"`,
    /// context_window `128_000`.
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        context_window: Option<usize>,
    ) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
            model: model.unwrap_or_else(|| "gpt-4o".to_string()),
            base_url: base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
            ctx_window: context_window.unwrap_or(128_000),
            extra_headers: Vec::new(),
        }
    }

    /// Add extra headers to be sent with every request.
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    /// Convert our ChatMessages to OpenAI API message format.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<Value> {
        let mut api_messages = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "assistant" => {
                    if let Some(ref tool_calls) = msg.tool_calls {
                        let tc_array: Vec<Value> = tool_calls
                            .iter()
                            .map(|tc| {
                                json!({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": tc.arguments.to_string()
                                    }
                                })
                            })
                            .collect();

                        let mut m = json!({
                            "role": "assistant",
                            "tool_calls": tc_array
                        });
                        if let Some(ref content) = msg.content {
                            if !content.is_empty() {
                                m["content"] = json!(content);
                            }
                        }
                        api_messages.push(m);
                    } else {
                        api_messages.push(json!({
                            "role": "assistant",
                            "content": msg.content.as_deref().unwrap_or("")
                        }));
                    }
                }
                "tool" => {
                    api_messages.push(json!({
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id.as_deref().unwrap_or(""),
                        "content": msg.content.as_deref().unwrap_or("")
                    }));
                }
                _ => {
                    // "system", "user", etc. If image attachments
                    // are queued for this turn, serialise as the
                    // multi-part content form: array of
                    // {type: text, text} + {type: image_url,
                    // image_url: {url: data:<mime>;base64,<data>}}.
                    let text = msg.content.as_deref().unwrap_or("");
                    if let Some(ref atts) = msg.attachments {
                        let mut parts = Vec::with_capacity(atts.len() + 1);
                        if !text.is_empty() {
                            parts.push(json!({"type": "text", "text": text}));
                        }
                        for a in atts {
                            let url = format!(
                                "data:{};base64,{}",
                                a.mime_type, a.base64_data
                            );
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": {"url": url},
                            }));
                        }
                        api_messages.push(json!({
                            "role": msg.role,
                            "content": parts,
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

    /// Convert our ToolSpec list to OpenAI's function calling format.
    fn convert_tools(tools: &[crate::tools::traits::ToolSpec]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect()
    }

    /// Build the common request body (shared by chat and chat_stream).
    fn build_request_body(&self, request: &ChatRequest<'_>, stream: bool) -> Value {
        build_openai_request_body(&self.model, request, stream)
    }

    /// Parse the OpenAI response JSON into our ChatResponse. Shared with the
    /// Azure provider, which receives the identical Chat Completions shape.
    pub(crate) fn parse_response(body: &Value) -> Result<ChatResponse> {
        let choice = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .context("missing choices[0] in response")?;

        let message = choice
            .get("message")
            .context("missing message in choice")?;

        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());

        // Reasoning text — two field shapes in the wild:
        //   - `reasoning` for native OpenAI o1/o3/o4 + OpenRouter
        //     unified shape
        //   - `reasoning_content` for DeepSeek / Moonshot
        // Read both, prefer the first non-empty.
        let reasoning = message
            .get("reasoning")
            .and_then(|r| r.as_str())
            .or_else(|| message.get("reasoning_content").and_then(|r| r.as_str()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());

        let mut tool_calls = Vec::new();
        if let Some(tcs) = message.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let function = tc.get("function").unwrap_or(&Value::Null);
                let name = function
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments_str = function
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let arguments: Value =
                    serde_json::from_str(arguments_str).unwrap_or(Value::Null);

                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
        }

        let usage = body.get("usage").map(|u| UsageInfo {
            input_tokens: u
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cache_read_tokens: u
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_u64()),
            // OpenAI doesn't return a cache-write counter — Anthropic
            // is the only provider exposing it today.
            cache_write_tokens: None,
        });

        Ok(ChatResponse {
            content,
            tool_calls,
            usage,
            reasoning,
        })
    }
}

/// Build an OpenAI Chat Completions request body. Shared by [`OpenAIProvider`]
/// and the Azure provider (which sends the identical body to Azure's
/// deployment-routed endpoint) — see [`super::azure`].
pub(crate) fn build_openai_request_body(
    model: &str,
    request: &ChatRequest<'_>,
    stream: bool,
) -> Value {
    let mut messages = Vec::new();

    if let Some(system_text) = request.system {
        messages.push(json!({ "role": "system", "content": system_text }));
    }
    messages.extend(OpenAIProvider::convert_messages(request.messages));

    // Reasoning-family models (o1, o3, o4-*, gpt-5*) REJECT `max_tokens` at the
    // Chat Completions endpoint and require `max_completion_tokens` instead.
    // They also reject `temperature`.
    let token_field = if is_reasoning_model(model) {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };

    let mut body = json!({
        "model": model,
        token_field: request.max_tokens,
        "messages": messages,
    });

    if !is_reasoning_model(model) {
        body["temperature"] = json!(request.temperature);
    }

    if stream {
        body["stream"] = json!(true);
        // Request a usage payload in the final streaming chunk (per OpenAI's
        // stream_options spec); servers that don't recognise it ignore it.
        body["stream_options"] = json!({ "include_usage": true });
    }

    if let Some(tools) = request.tools {
        if !tools.is_empty() {
            body["tools"] = json!(OpenAIProvider::convert_tools(tools));
        }
    }

    crate::agent::thinking::apply_thinking_params(&mut body, request.thinking_level, "openai");

    body
}

/// Return `true` for OpenAI model names that belong to the "reasoning"
/// family — these models reject `max_tokens` (and `temperature`) at the
/// Chat Completions endpoint and require `max_completion_tokens` instead.
///
/// Covered families (per Azure / OpenAI reasoning docs):
///   - `o1`, `o1-mini`, `o1-preview`, `o1-pro`
///   - `o3`, `o3-mini`, `o3-pro`
///   - `o4-mini`
///   - `gpt-5*` (gpt-5, gpt-5-mini, gpt-5.1, gpt-5.1-codex, gpt-5.2, ...)
///
/// Matching is on lowercase prefix, so dated snapshots
/// (e.g. `o3-mini-2025-01-31`) and future variants are caught
/// automatically. Non-reasoning models (`gpt-4o`, `gpt-4.1`, `gpt-3.5-*`,
/// `chatgpt-4o-*`) return `false`.
pub(crate) fn is_reasoning_model(model: &str) -> bool {
    let m = model.to_lowercase();
    // gpt-5* is always reasoning. (gpt-4* and gpt-3.5* are not.)
    if m.starts_with("gpt-5") {
        return true;
    }
    // o<N> families: match `oN` where the next char (if any) is `-` or a
    // digit. This catches `o1`, `o1-mini`, `o3`, `o3-mini`, `o3-pro`,
    // `o4-mini`, `o1-2024-12-17`, etc. — without false-matching unrelated
    // model names that happen to start with `o` (e.g. some OpenRouter
    // aliases).
    for prefix in ["o1", "o3", "o4"] {
        if let Some(rest) = m.strip_prefix(prefix) {
            if rest.is_empty() || rest.starts_with('-') || rest.starts_with(char::is_numeric) {
                return true;
            }
        }
    }
    false
}

/// Dispatch one OpenAI streaming chunk payload.
///
/// Emits `Delta` / `ToolCallStart` / `ToolCallDelta` events for anything in
/// `choice.delta`, then reports `true` if `choice.finish_reason` indicates
/// the stream should terminate.
///
/// **Order matters**: the old implementation checked `finish_reason` first
/// and `return`ed on `"stop"` / `"tool_calls"`, which dropped any tool-call
/// arguments delta present in the *same* chunk. Processing the delta
/// unconditionally before the terminator check means we never lose
/// a tail.
async fn dispatch_openai_chunk(
    data: &Value,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    usage_acc: &mut UsageInfo,
) -> bool {
    // OpenAI's final streaming chunk (when `stream_options.include_usage`
    // is set) carries an empty `choices` array and a populated `usage`
    // object. Capture it into the accumulator so the stream loop can
    // emit `StreamEvent::Usage` once the terminator arrives.
    if let Some(u) = data.get("usage") {
        if let Some(v) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
            usage_acc.input_tokens = v;
        }
        if let Some(v) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
            usage_acc.output_tokens = v;
        }
        if let Some(v) = u
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
        {
            usage_acc.cache_read_tokens = Some(v);
        }
    }

    let Some(choice) = data
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
    else {
        return false;
    };

    if let Some(delta) = choice.get("delta") {
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                let _ = tx.send(StreamEvent::Delta(content.to_string())).await;
            }
        }

        if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
            for tc in tcs {
                let id = tc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let func = tc.get("function").unwrap_or(&Value::Null);

                if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                    let _ = tx
                        .send(StreamEvent::ToolCallStart {
                            id: id.clone(),
                            name: name.to_string(),
                        })
                        .await;
                }

                if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                    if !args.is_empty() {
                        let _ = tx
                            .send(StreamEvent::ToolCallDelta {
                                id: id.clone(),
                                arguments_delta: args.to_string(),
                            })
                            .await;
                    }
                }
            }
        }
    }

    if let Some(reason) = choice.get("finish_reason").and_then(|r| r.as_str()) {
        if matches!(reason, "stop" | "tool_calls" | "length" | "content_filter") {
            return true;
        }
    }
    false
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let body = self.build_request_body(&request, false);

        let mut req = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        for (key, value) in &self.extra_headers {
            req = req.header(key.as_str(), value.as_str());
        }

        let response = req
            .json(&body)
            .send()
            .await
            .context("sending request to OpenAI-compatible API")?;

        // Read raw bytes first so a non-JSON error body (proxy/gateway HTML
        // 502, rate-limit plaintext, etc.) preserves HTTP status context.
        let status = response.status();
        let raw_body = response
            .bytes()
            .await
            .context("reading OpenAI API response body")?;

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
                    String::from_utf8_lossy(&raw_body)
                        .chars()
                        .take(200)
                        .collect()
                });
            anyhow::bail!("OpenAI API error ({}): {}", status, error_msg);
        }

        let response_body: Value = serde_json::from_slice(&raw_body)
            .context("parsing OpenAI API response as JSON")?;
        Self::parse_response(&response_body)
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
        let body = self.build_request_body(&request, true);

        let mut req = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");

        for (key, value) in &self.extra_headers {
            req = req.header(key.as_str(), value.as_str());
        }

        let response = req
            .json(&body)
            .send()
            .await
            .context("sending streaming request to OpenAI-compatible API")?;

        let status = response.status();
        if !status.is_success() {
            let raw_body = response
                .bytes()
                .await
                .context("reading OpenAI streaming error body")?;
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
            anyhow::bail!("OpenAI API error ({}): {}", status, error_msg);
        }

        Ok(spawn_openai_stream(response))
    }
}

/// Spawn the SSE-processing task for an OpenAI-style streaming response and
/// return the event receiver. Shared by [`OpenAIProvider`] and the Azure
/// provider, whose streaming response is the identical Chat Completions SSE
/// shape.
pub(crate) fn spawn_openai_stream(
    response: reqwest::Response,
) -> tokio::sync::mpsc::Receiver<StreamEvent> {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    let mut byte_stream = response.bytes_stream();

    tokio::spawn(async move {
        let mut sse = SseBuffer::new();
        let mut usage_acc = UsageInfo::default();
        let mut usage_seen = false;
        // Emit `StreamEvent::Usage` exactly once when we have data
        // for it. The accumulator may be populated either by an
        // intermediate chunk that already carried `usage` or by
        // the final chunk that arrives with empty `choices`.
        let emit_usage = |usage: &UsageInfo,
                          tx: &tokio::sync::mpsc::Sender<StreamEvent>,
                          seen: &mut bool|
         -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
            let usage = usage.clone();
            let tx = tx.clone();
            let seen_now = !*seen
                && (usage.input_tokens > 0
                    || usage.output_tokens > 0
                    || usage.cache_read_tokens.is_some());
            if seen_now {
                *seen = true;
            }
            Box::pin(async move {
                if seen_now {
                    let _ = tx.send(StreamEvent::Usage(usage)).await;
                }
            })
        };

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                    return;
                }
            };
            sse.extend(&chunk);

            while let Some(line_bytes) = sse.next_line() {
                let line = match std::str::from_utf8(&line_bytes) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                let data_str = match line.strip_prefix("data: ") {
                    Some(s) => s,
                    None => continue,
                };

                if data_str == "[DONE]" {
                    emit_usage(&usage_acc, &tx, &mut usage_seen).await;
                    let _ = tx.send(StreamEvent::Done).await;
                    return;
                }

                let data: Value = match serde_json::from_str(data_str) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if dispatch_openai_chunk(&data, &tx, &mut usage_acc).await {
                    // finish_reason observed — still prefer [DONE] as
                    // the authoritative terminator when the server
                    // sends it, but close now so we don't hang if it
                    // doesn't.
                    emit_usage(&usage_acc, &tx, &mut usage_seen).await;
                    let _ = tx.send(StreamEvent::Done).await;
                    return;
                }
            }
        }

        emit_usage(&usage_acc, &tx, &mut usage_seen).await;
        let _ = tx.send(StreamEvent::Done).await;
    });

    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages_user() {
        let msgs = vec![ChatMessage::user("hello")];
        let converted = OpenAIProvider::convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "user");
        assert_eq!(converted[0]["content"], "hello");
    }

    #[test]
    fn test_convert_messages_with_tool_calls() {
        let mut msg = ChatMessage::assistant("thinking...");
        msg.tool_calls = Some(vec![ToolCall {
            id: "tc_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "/tmp/test.txt"}),
        }]);

        let converted = OpenAIProvider::convert_messages(&[msg]);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "assistant");
        assert!(converted[0]["tool_calls"].is_array());
        let tc = &converted[0]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "read_file");
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let msg = ChatMessage::tool_result("tc_1", "file contents");
        let converted = OpenAIProvider::convert_messages(&[msg]);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["role"], "tool");
        assert_eq!(converted[0]["tool_call_id"], "tc_1");
        assert_eq!(converted[0]["content"], "file contents");
    }

    #[test]
    fn test_parse_response_text() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        });

        let response = OpenAIProvider::parse_response(&body).unwrap();
        assert_eq!(response.content.as_deref(), Some("Hello!"));
        assert!(response.tool_calls.is_empty());
        let usage = response.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_response_with_tool_calls() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"SF\"}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10
            }
        });

        let response = OpenAIProvider::parse_response(&body).unwrap();
        assert!(response.content.is_none());
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "call_abc");
        assert_eq!(response.tool_calls[0].name, "get_weather");
        assert_eq!(response.tool_calls[0].arguments["location"], "SF");
    }

    #[test]
    fn test_convert_tools() {
        let specs = vec![crate::tools::traits::ToolSpec {
            name: "my_tool".to_string(),
            description: "Does stuff".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                }
            }),
        }];
        let converted = OpenAIProvider::convert_tools(&specs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0]["type"], "function");
        assert_eq!(converted[0]["function"]["name"], "my_tool");
        assert_eq!(converted[0]["function"]["description"], "Does stuff");
    }

    #[test]
    fn is_reasoning_model_o_series() {
        assert!(is_reasoning_model("o1"));
        assert!(is_reasoning_model("o1-mini"));
        assert!(is_reasoning_model("o1-preview"));
        assert!(is_reasoning_model("o1-pro"));
        assert!(is_reasoning_model("o1-2024-12-17"));
        assert!(is_reasoning_model("o3"));
        assert!(is_reasoning_model("o3-mini"));
        assert!(is_reasoning_model("o3-mini-2025-01-31"));
        assert!(is_reasoning_model("o3-pro"));
        assert!(is_reasoning_model("o4-mini"));
    }

    #[test]
    fn is_reasoning_model_gpt5_family() {
        assert!(is_reasoning_model("gpt-5"));
        assert!(is_reasoning_model("gpt-5-mini"));
        assert!(is_reasoning_model("gpt-5.1"));
        assert!(is_reasoning_model("gpt-5.1-codex"));
        assert!(is_reasoning_model("gpt-5.1-codex-max"));
        assert!(is_reasoning_model("gpt-5.2"));
    }

    #[test]
    fn is_reasoning_model_case_insensitive() {
        assert!(is_reasoning_model("O1"));
        assert!(is_reasoning_model("O3-Mini"));
        assert!(is_reasoning_model("GPT-5.1"));
    }

    #[test]
    fn is_reasoning_model_rejects_non_reasoning() {
        // Chat / non-reasoning OpenAI models.
        assert!(!is_reasoning_model("gpt-4o"));
        assert!(!is_reasoning_model("gpt-4o-mini"));
        assert!(!is_reasoning_model("gpt-4-turbo"));
        assert!(!is_reasoning_model("gpt-4.1"));
        assert!(!is_reasoning_model("gpt-3.5-turbo"));
        assert!(!is_reasoning_model("chatgpt-4o-latest"));
        // Unrelated OpenRouter / vendor aliases that happen to start with o.
        assert!(!is_reasoning_model("ollama"));
        assert!(!is_reasoning_model("openrouter/foo"));
        assert!(!is_reasoning_model("oss-model"));
        // Empty / garbage.
        assert!(!is_reasoning_model(""));
        assert!(!is_reasoning_model("claude-sonnet-4"));
        assert!(!is_reasoning_model("llama3.1"));
    }

    #[test]
    fn build_request_body_non_reasoning_uses_max_tokens() {
        let p = OpenAIProvider::new("k".to_string(), Some("gpt-4o".to_string()), None, None);
        let req = ChatRequest {
            messages: &[],
            system: None,
            tools: None,
            max_tokens: 1234,
            temperature: 0.42,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };
        let body = p.build_request_body(&req, false);
        assert_eq!(body["max_tokens"], 1234);
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body["temperature"], 0.42);
    }

    #[test]
    fn build_request_body_reasoning_uses_max_completion_tokens_and_drops_temperature() {
        let p = OpenAIProvider::new("k".to_string(), Some("o3-mini".to_string()), None, None);
        let req = ChatRequest {
            messages: &[],
            system: None,
            tools: None,
            max_tokens: 1234,
            temperature: 0.42,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };
        let body = p.build_request_body(&req, false);
        assert_eq!(body["max_completion_tokens"], 1234);
        assert!(body.get("max_tokens").is_none());
        // Reasoning models reject `temperature`; we must not send it.
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn build_request_body_gpt5_treated_as_reasoning() {
        let p = OpenAIProvider::new("k".to_string(), Some("gpt-5.1".to_string()), None, None);
        let req = ChatRequest {
            messages: &[],
            system: None,
            tools: None,
            max_tokens: 100,
            temperature: 0.7,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };
        let body = p.build_request_body(&req, false);
        assert_eq!(body["max_completion_tokens"], 100);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    /// Drain a receiver after a dispatch call into a Vec<StreamEvent>.
    async fn drain(
        rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>,
    ) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        out
    }

    #[tokio::test]
    async fn dispatch_emits_delta_then_reports_termination() {
        // Realistic "final" chunk: contains tool-call args AND finish_reason.
        // Previous impl would return early on finish_reason and lose the
        // arguments delta.
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let data = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"SF\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let mut usage = UsageInfo::default();
        let terminate = dispatch_openai_chunk(&data, &tx, &mut usage).await;
        let events = drain(&mut rx).await;
        assert!(terminate, "finish_reason=tool_calls should signal termination");
        // We must have seen both Start and Delta before termination.
        assert!(events.iter().any(|e| matches!(e,
            StreamEvent::ToolCallStart { id, name } if id == "call_1" && name == "get_weather"
        )), "missing ToolCallStart: {:?}", events);
        assert!(events.iter().any(|e| matches!(e,
            StreamEvent::ToolCallDelta { id, arguments_delta }
                if id == "call_1" && arguments_delta.contains("SF")
        )), "missing ToolCallDelta — regression of the early-return bug: {:?}", events);
    }

    #[tokio::test]
    async fn dispatch_content_delta_then_stop() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let data = json!({
            "choices": [{
                "delta": { "content": "Hello!" },
                "finish_reason": "stop"
            }]
        });
        let mut usage = UsageInfo::default();
        let terminate = dispatch_openai_chunk(&data, &tx, &mut usage).await;
        let events = drain(&mut rx).await;
        assert!(terminate);
        assert!(matches!(&events[0], StreamEvent::Delta(s) if s == "Hello!"));
    }

    #[tokio::test]
    async fn dispatch_no_finish_reason_does_not_terminate() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let data = json!({
            "choices": [{
                "delta": { "content": "partial..." }
            }]
        });
        let mut usage = UsageInfo::default();
        let terminate = dispatch_openai_chunk(&data, &tx, &mut usage).await;
        let events = drain(&mut rx).await;
        assert!(!terminate);
        assert!(matches!(&events[0], StreamEvent::Delta(s) if s == "partial..."));
    }

    #[tokio::test]
    async fn dispatch_handles_length_and_content_filter_as_terminators() {
        // These reasons were NOT recognized by the old code; some models
        // (especially safety-gated paths) report them.
        for reason in ["length", "content_filter"] {
            let (tx, mut _rx) = tokio::sync::mpsc::channel::<StreamEvent>(4);
            let data = json!({ "choices": [{ "delta": {}, "finish_reason": reason }] });
            let mut usage = UsageInfo::default();
        let terminate = dispatch_openai_chunk(&data, &tx, &mut usage).await;
            assert!(terminate, "finish_reason={} should terminate", reason);
        }
    }
}
