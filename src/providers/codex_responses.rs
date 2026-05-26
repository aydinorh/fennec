//! OpenAI Responses API provider (`POST /v1/responses`).
//!
//! The Responses API is OpenAI's newer surface (used by the Codex / gpt-5
//! models) and is shaped quite differently from Chat Completions: the
//! conversation is an `input` array of *items* (`message`, `function_call`,
//! `function_call_output`, `reasoning`) rather than `messages`; the system
//! prompt is a separate `instructions` field; tools use a flat
//! `{type:"function", name, description, parameters}` shape; reasoning effort
//! is a structured `reasoning.effort` field; and the response comes back as an
//! `output` array of items. This module translates Fennec's provider-agnostic
//! [`ChatMessage`] / [`ToolCall`] types to and from that format for both the
//! non-streaming and streaming (`stream:true`) endpoints.
//!
//! Multi-turn reasoning replay (threading the opaque `encrypted_content`
//! reasoning items back on later turns) is intentionally out of scope here —
//! it requires carrying reasoning items on [`ChatMessage`] across turns, a
//! cross-cutting change. The provider works fully without it; the model simply
//! re-reasons each turn instead of resuming a prior chain.

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use super::openai::is_reasoning_model;
use super::sse::SseBuffer;
use super::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, StreamEvent, ToolCall, UsageInfo,
};
use crate::agent::thinking::ThinkingLevel;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "gpt-5-codex";
const DEFAULT_CONTEXT_WINDOW: usize = 400_000;

/// OpenAI Responses API provider.
pub struct CodexResponsesProvider {
    api_key: String,
    client: reqwest::Client,
    model: String,
    base_url: String,
    ctx_window: usize,
    extra_headers: Vec<(String, String)>,
}

impl CodexResponsesProvider {
    /// Create a new Responses provider.
    ///
    /// Defaults: model `"gpt-5-codex"`, base_url `"https://api.openai.com/v1"`,
    /// context window `400_000`.
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        context_window: Option<usize>,
    ) -> Self {
        let mut base = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        while base.ends_with('/') {
            base.pop();
        }
        Self {
            api_key,
            client: reqwest::Client::new(),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            base_url: base,
            ctx_window: context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
            extra_headers: Vec::new(),
        }
    }

    /// Add extra headers sent with every request (e.g. for compatible gateways).
    pub fn with_extra_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    fn build_request_body(&self, request: &ChatRequest<'_>, stream: bool) -> Value {
        let (input, instructions) = build_input_items(request.system, request.messages);

        let mut body = json!({
            "model": self.model,
            "input": input,
            // We manage history ourselves and send it in full each turn, so the
            // server must not persist responses for stateful chaining.
            "store": false,
        });
        if !instructions.is_empty() {
            body["instructions"] = json!(instructions);
        }
        if stream {
            body["stream"] = json!(true);
        }

        if let Some(tools) = request.tools {
            let converted = build_tools(tools);
            if !converted.is_empty() {
                body["tools"] = json!(converted);
            }
        }

        body["max_output_tokens"] = json!(request.max_tokens);

        // Reasoning models reject `temperature`; only send it for the rest.
        if !is_reasoning_model(&self.model) {
            body["temperature"] = json!(request.temperature);
        }

        // Structured reasoning effort, valid only for reasoning models.
        if is_reasoning_model(&self.model) {
            if let Some(effort) = reasoning_effort(request.thinking_level) {
                body["reasoning"] = json!({ "effort": effort });
            }
        }

        body
    }
}

/// Map Fennec's reasoning level onto the Responses `reasoning.effort` enum.
fn reasoning_effort(level: ThinkingLevel) -> Option<&'static str> {
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High | ThinkingLevel::Max => Some("high"),
    }
}

/// Translate the conversation into Responses `input` items plus the
/// `instructions` string (system text lives outside `input`).
fn build_input_items(system: Option<&str>, messages: &[ChatMessage]) -> (Vec<Value>, String) {
    let mut instructions: Vec<String> = Vec::new();
    if let Some(s) = system {
        if !s.is_empty() {
            instructions.push(s.to_string());
        }
    }

    let mut input: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(c) = &msg.content {
                    if !c.is_empty() {
                        instructions.push(c.clone());
                    }
                }
            }
            "tool" | "function" => {
                let call_id = msg.tool_call_id.clone().unwrap_or_default();
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": msg.content.clone().unwrap_or_default(),
                }));
            }
            role => {
                let is_assistant = role == "assistant";
                let text_type = if is_assistant { "output_text" } else { "input_text" };
                let api_role = if is_assistant { "assistant" } else { "user" };

                let mut content: Vec<Value> = Vec::new();
                if let Some(c) = &msg.content {
                    if !c.is_empty() {
                        content.push(json!({ "type": text_type, "text": c }));
                    }
                }
                // Image attachments — only meaningful on user turns.
                if !is_assistant {
                    if let Some(attachments) = &msg.attachments {
                        for a in attachments {
                            let data_url =
                                format!("data:{};base64,{}", a.mime_type, a.base64_data);
                            content.push(json!({
                                "type": "input_image",
                                "image_url": data_url,
                            }));
                        }
                    }
                }
                if !content.is_empty() {
                    input.push(json!({
                        "type": "message",
                        "role": api_role,
                        "content": content,
                    }));
                }

                // Assistant tool calls become standalone function_call items.
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        input.push(json!({
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments.to_string(),
                        }));
                    }
                }
            }
        }
    }

    (input, instructions.join("\n"))
}

/// Convert Fennec tool specs to the Responses flat function-tool shape.
fn build_tools(tools: &[crate::tools::traits::ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        })
        .collect()
}

/// Parse the Responses `usage` object into [`UsageInfo`].
fn parse_usage(usage: &Value) -> UsageInfo {
    UsageInfo {
        input_tokens: usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        output_tokens: usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_read_tokens: usage
            .get("input_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64()),
        cache_write_tokens: None,
    }
}

/// Parse a non-streaming Responses body into a [`ChatResponse`].
fn parse_response(body: &Value) -> Result<ChatResponse> {
    let output = body
        .get("output")
        .and_then(|o| o.as_array())
        .context("missing output array in Responses reply")?;

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();

    for item in output {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("message") => {
                if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                    for part in parts {
                        if matches!(
                            part.get("type").and_then(|t| t.as_str()),
                            Some("output_text") | Some("text")
                        ) {
                            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                text.push_str(t);
                            }
                        }
                    }
                }
            }
            Some("reasoning") => {
                // Some backends expose a plain `text`; others nest the
                // human-readable trace under `summary[].text`.
                if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                    reasoning.push_str(t);
                } else if let Some(summary) = item.get("summary").and_then(|s| s.as_array()) {
                    for part in summary {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            reasoning.push_str(t);
                        }
                    }
                }
            }
            Some("function_call") => {
                let name = item
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let call_id = item
                    .get("call_id")
                    .and_then(|c| c.as_str())
                    .or_else(|| item.get("id").and_then(|c| c.as_str()))
                    .unwrap_or("")
                    .to_string();
                let arguments_str = item
                    .get("arguments")
                    .and_then(|a| a.as_str())
                    .unwrap_or("{}");
                let arguments: Value =
                    serde_json::from_str(arguments_str).unwrap_or(Value::Null);
                tool_calls.push(ToolCall {
                    id: call_id,
                    name,
                    arguments,
                });
            }
            _ => {}
        }
    }

    let usage = body.get("usage").map(parse_usage);

    Ok(ChatResponse {
        content: if text.is_empty() { None } else { Some(text) },
        tool_calls,
        usage,
        reasoning: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
    })
}

fn extract_error_message(raw: &[u8]) -> String {
    serde_json::from_slice::<Value>(raw)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| String::from_utf8_lossy(raw).chars().take(200).collect())
}

/// State threaded across streaming events. Responses emits output items
/// sequentially, so a single "current function call id" is enough to pair the
/// `function_call_arguments.delta` events with the call that `output_item.added`
/// announced.
#[derive(Default)]
struct StreamState {
    current_call_id: Option<String>,
}

/// Dispatch one Responses streaming event (the parsed `data:` JSON). Returns
/// `true` once the stream should terminate (`response.completed`/`failed`).
async fn dispatch_responses_event(
    data: &Value,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    state: &mut StreamState,
) -> bool {
    match data.get("type").and_then(|t| t.as_str()) {
        Some("response.output_text.delta") => {
            if let Some(delta) = data.get("delta").and_then(|d| d.as_str()) {
                if !delta.is_empty() {
                    let _ = tx.send(StreamEvent::Delta(delta.to_string())).await;
                }
            }
            false
        }
        Some("response.reasoning_summary_text.delta") => {
            if let Some(delta) = data.get("delta").and_then(|d| d.as_str()) {
                if !delta.is_empty() {
                    let _ = tx.send(StreamEvent::Reasoning(delta.to_string())).await;
                }
            }
            false
        }
        Some("response.output_item.added") => {
            let item = data.get("item");
            if item.and_then(|i| i.get("type")).and_then(|t| t.as_str()) == Some("function_call") {
                let item = item.unwrap();
                let call_id = item
                    .get("call_id")
                    .and_then(|c| c.as_str())
                    .or_else(|| item.get("id").and_then(|c| c.as_str()))
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                state.current_call_id = Some(call_id.clone());
                let _ = tx
                    .send(StreamEvent::ToolCallStart { id: call_id, name })
                    .await;
            }
            false
        }
        Some("response.function_call_arguments.delta") => {
            if let (Some(id), Some(delta)) = (
                state.current_call_id.clone(),
                data.get("delta").and_then(|d| d.as_str()),
            ) {
                if !delta.is_empty() {
                    let _ = tx
                        .send(StreamEvent::ToolCallDelta {
                            id,
                            arguments_delta: delta.to_string(),
                        })
                        .await;
                }
            }
            false
        }
        Some("response.function_call_arguments.done") => {
            if let Some(id) = state.current_call_id.take() {
                let _ = tx.send(StreamEvent::ToolCallEnd { id }).await;
            }
            false
        }
        Some("response.completed") => {
            if let Some(usage) = data
                .get("response")
                .and_then(|r| r.get("usage"))
                .filter(|u| u.is_object())
            {
                let _ = tx.send(StreamEvent::Usage(parse_usage(usage))).await;
            }
            let _ = tx.send(StreamEvent::Done).await;
            true
        }
        Some("response.failed") | Some("response.error") | Some("error") => {
            let msg = data
                .get("response")
                .and_then(|r| r.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .or_else(|| data.get("message").and_then(|m| m.as_str()))
                .unwrap_or("Responses stream error")
                .to_string();
            let _ = tx.send(StreamEvent::Error(msg)).await;
            true
        }
        _ => false,
    }
}

#[async_trait]
impl Provider for CodexResponsesProvider {
    fn name(&self) -> &str {
        "codex"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let body = self.build_request_body(&request, false);

        let mut req = self
            .client
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json");
        for (k, v) in &self.extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let response = req
            .json(&body)
            .send()
            .await
            .context("sending request to Responses API")?;

        let status = response.status();
        let raw_body = response
            .bytes()
            .await
            .context("reading Responses API response body")?;

        if !status.is_success() {
            anyhow::bail!(
                "Responses API error ({}): {}",
                status,
                extract_error_message(&raw_body)
            );
        }

        let response_body: Value = serde_json::from_slice(&raw_body)
            .context("parsing Responses API response as JSON")?;
        parse_response(&response_body)
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
            .post(format!("{}/responses", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");
        for (k, v) in &self.extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }

        let response = req
            .json(&body)
            .send()
            .await
            .context("sending streaming request to Responses API")?;

        let status = response.status();
        if !status.is_success() {
            let raw_body = response
                .bytes()
                .await
                .context("reading Responses streaming error body")?;
            anyhow::bail!(
                "Responses API error ({}): {}",
                status,
                extract_error_message(&raw_body)
            );
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let mut byte_stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut sse = SseBuffer::new();
            let mut state = StreamState::default();

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
                    // Responses streams named SSE events (`event:` + `data:`);
                    // the `data` JSON also carries a `type` field, so dispatch
                    // on that and ignore the `event:` lines.
                    let data_str = match line.strip_prefix("data: ") {
                        Some(s) => s,
                        None => continue,
                    };
                    if data_str == "[DONE]" {
                        let _ = tx.send(StreamEvent::Done).await;
                        return;
                    }
                    let data: Value = match serde_json::from_str(data_str) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if dispatch_responses_event(&data, &tx, &mut state).await {
                        return;
                    }
                }
            }

            // Stream ended without an explicit terminator — close cleanly.
            let _ = tx.send(StreamEvent::Done).await;
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::traits::ImageAttachmentRef;

    fn spec(name: &str, description: &str, parameters: Value) -> crate::tools::traits::ToolSpec {
        crate::tools::traits::ToolSpec {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }

    #[test]
    fn build_input_lifts_system_to_instructions() {
        let messages = vec![ChatMessage::user("hello")];
        let (input, instructions) = build_input_items(Some("be terse"), &messages);
        assert_eq!(instructions, "be terse");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn build_input_folds_system_role_messages() {
        let messages = vec![ChatMessage::system("rule one"), ChatMessage::user("hi")];
        let (input, instructions) = build_input_items(Some("rule zero"), &messages);
        assert_eq!(instructions, "rule zero\nrule one");
        assert_eq!(input.len(), 1); // system message must not leak into input
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn build_input_assistant_uses_output_text() {
        let messages = vec![ChatMessage::assistant("sure thing")];
        let (input, _) = build_input_items(None, &messages);
        assert_eq!(input[0]["role"], "assistant");
        assert_eq!(input[0]["content"][0]["type"], "output_text");
    }

    #[test]
    fn build_input_tool_call_and_result() {
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "/tmp/x" }),
        }]);
        let tool = ChatMessage::tool_result("call_1", "file body");

        let (input, _) = build_input_items(None, &[assistant, tool]);
        assert_eq!(input.len(), 2);
        // function_call item carries call_id + name + stringified arguments.
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["name"], "read_file");
        let args: Value = serde_json::from_str(input[0]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["path"], "/tmp/x");
        // function_call_output references the same call_id.
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["output"], "file body");
    }

    #[test]
    fn build_input_emits_image_for_user_only() {
        let mut user = ChatMessage::user("what is this");
        user.attachments = Some(vec![ImageAttachmentRef {
            mime_type: "image/png".to_string(),
            base64_data: "AAAA".to_string(),
            display_name: None,
        }]);
        let (input, _) = build_input_items(None, &[user]);
        let content = input[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[1]["type"], "input_image");
        assert!(content[1]["image_url"].as_str().unwrap().starts_with("data:image/png;base64,"));
    }

    #[test]
    fn build_tools_uses_flat_function_shape() {
        let specs = vec![spec(
            "my_tool",
            "Does stuff",
            json!({ "type": "object", "properties": { "arg": { "type": "string" } } }),
        )];
        let tools = build_tools(&specs);
        assert_eq!(tools.len(), 1);
        // Flat — name/description/parameters are siblings of `type`, not nested.
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "my_tool");
        assert_eq!(tools[0]["description"], "Does stuff");
        assert_eq!(tools[0]["parameters"]["properties"]["arg"]["type"], "string");
    }

    #[test]
    fn request_body_reasoning_model_drops_temperature_adds_reasoning() {
        let p = CodexResponsesProvider::new("k".to_string(), Some("gpt-5-codex".to_string()), None, None);
        let messages = vec![ChatMessage::user("hi")];
        let request = ChatRequest {
            system: Some("sys"),
            messages: &messages,
            tools: None,
            max_tokens: 2048,
            temperature: 0.7,
            thinking_level: ThinkingLevel::High,
        };
        let body = p.build_request_body(&request, false);
        assert_eq!(body["model"], "gpt-5-codex");
        assert_eq!(body["instructions"], "sys");
        assert_eq!(body["store"], false);
        assert_eq!(body["max_output_tokens"], 2048);
        assert!(body.get("temperature").is_none(), "reasoning model must not send temperature");
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn request_body_non_reasoning_model_keeps_temperature() {
        let p = CodexResponsesProvider::new("k".to_string(), Some("gpt-4o".to_string()), None, None);
        let messages = vec![ChatMessage::user("hi")];
        let request = ChatRequest {
            system: None,
            messages: &messages,
            tools: None,
            max_tokens: 100,
            temperature: 0.42,
            thinking_level: ThinkingLevel::High,
        };
        let body = p.build_request_body(&request, false);
        assert_eq!(body["temperature"], 0.42);
        assert!(body.get("reasoning").is_none(), "non-reasoning model gets no reasoning block");
    }

    #[test]
    fn parse_response_text_reasoning_and_function_call() {
        let body = json!({
            "output": [
                { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "thinking" }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Hello!" }] },
                { "type": "function_call", "call_id": "call_9", "name": "get_weather", "arguments": "{\"city\":\"SF\"}" }
            ],
            "usage": { "input_tokens": 11, "output_tokens": 4, "total_tokens": 15, "input_tokens_details": { "cached_tokens": 2 } }
        });
        let resp = parse_response(&body).unwrap();
        assert_eq!(resp.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.reasoning.as_deref(), Some("thinking"));
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_9");
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].arguments["city"], "SF");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.cache_read_tokens, Some(2));
    }

    async fn drain(rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        out
    }

    #[tokio::test]
    async fn stream_text_delta_then_completed_terminates() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut state = StreamState::default();
        assert!(
            !dispatch_responses_event(
                &json!({ "type": "response.output_text.delta", "delta": "Hel" }),
                &tx,
                &mut state
            )
            .await
        );
        let terminate = dispatch_responses_event(
            &json!({ "type": "response.completed", "response": { "usage": { "input_tokens": 3, "output_tokens": 1 } } }),
            &tx,
            &mut state,
        )
        .await;
        assert!(terminate);
        let events = drain(&mut rx).await;
        assert!(matches!(&events[0], StreamEvent::Delta(s) if s == "Hel"));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Usage(_))));
        assert!(matches!(events.last(), Some(StreamEvent::Done)));
    }

    #[tokio::test]
    async fn stream_function_call_emits_start_delta_end_in_order() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut state = StreamState::default();
        dispatch_responses_event(
            &json!({
                "type": "response.output_item.added",
                "item": { "type": "function_call", "call_id": "call_7", "name": "run", "arguments": "" }
            }),
            &tx,
            &mut state,
        )
        .await;
        dispatch_responses_event(
            &json!({ "type": "response.function_call_arguments.delta", "delta": "{\"x\":1}" }),
            &tx,
            &mut state,
        )
        .await;
        dispatch_responses_event(
            &json!({ "type": "response.function_call_arguments.done", "arguments": "{\"x\":1}" }),
            &tx,
            &mut state,
        )
        .await;
        let events = drain(&mut rx).await;
        match &events[0] {
            StreamEvent::ToolCallStart { id, name } => {
                assert_eq!(id, "call_7");
                assert_eq!(name, "run");
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }
        match &events[1] {
            StreamEvent::ToolCallDelta { id, arguments_delta } => {
                assert_eq!(id, "call_7");
                assert!(arguments_delta.contains("\"x\""));
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
        match &events[2] {
            StreamEvent::ToolCallEnd { id } => assert_eq!(id, "call_7"),
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
        assert!(state.current_call_id.is_none());
    }

    #[tokio::test]
    async fn stream_failed_emits_error_and_terminates() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut state = StreamState::default();
        let terminate = dispatch_responses_event(
            &json!({ "type": "response.failed", "response": { "error": { "message": "boom" } } }),
            &tx,
            &mut state,
        )
        .await;
        assert!(terminate);
        let events = drain(&mut rx).await;
        assert!(matches!(&events[0], StreamEvent::Error(s) if s == "boom"));
    }
}
