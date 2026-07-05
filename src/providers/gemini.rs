//! Google Gemini provider (native `generativelanguage.googleapis.com` API).
//!
//! Unlike OpenRouter / Kimi (which are OpenAI-compatible and reuse
//! [`super::openai::OpenAIProvider`]), Gemini's REST API has its own shape:
//! messages live under a `contents` array of `{role, parts}` objects where
//! `role` is `user` or `model`; the system prompt is a separate
//! `systemInstruction` field; tools are `functionDeclarations` whose
//! parameter schema is only a *subset* of JSON Schema; tool calls and
//! results are `functionCall` / `functionResponse` parts; and reasoning
//! comes back as parts flagged `thought: true`. This module translates
//! between Fennec's provider-agnostic [`ChatMessage`] / [`ToolCall`] types
//! and that wire format in both directions, for the non-streaming
//! (`:generateContent`) and streaming (`:streamGenerateContent?alt=sse`)
//! endpoints.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Map, Value};

use crate::agent::thinking::ThinkingLevel;

use super::sse::SseBuffer;
use super::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, StreamEvent, ToolCall, UsageInfo,
};

const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_GEMINI_MODEL: &str = "gemini-2.5-flash";
/// Gemini 2.5 Flash and Pro both expose a ~1,048,576-token input window.
const DEFAULT_GEMINI_CONTEXT_WINDOW: usize = 1_048_576;

/// Native Google Gemini API provider.
pub struct GeminiProvider {
    api_key: String,
    client: reqwest::Client,
    model: String,
    base_url: String,
    ctx_window: usize,
}

impl GeminiProvider {
    /// Create a new Gemini provider.
    ///
    /// Defaults: model `"gemini-2.5-flash"`, base_url
    /// `"https://generativelanguage.googleapis.com/v1beta"`, context window
    /// `1_048_576`.
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        context_window: Option<usize>,
    ) -> Self {
        // Normalize the base URL: drop trailing slashes and a trailing
        // `/openai` segment (some setups point at Gemini's OpenAI-compat
        // shim URL; the native path lives one level up).
        let mut base = base_url.unwrap_or_else(|| DEFAULT_GEMINI_BASE_URL.to_string());
        while base.ends_with('/') {
            base.pop();
        }
        if let Some(stripped) = base.strip_suffix("/openai") {
            base = stripped.to_string();
        }

        Self {
            api_key,
            client: reqwest::Client::new(),
            model: model.unwrap_or_else(|| DEFAULT_GEMINI_MODEL.to_string()),
            base_url: base,
            ctx_window: context_window.unwrap_or(DEFAULT_GEMINI_CONTEXT_WINDOW),
        }
    }

    /// Build the Gemini request body shared by `chat` and `chat_stream`.
    fn build_request_body(&self, request: &ChatRequest<'_>) -> Value {
        let (contents, system_instruction) =
            build_contents(request.system, request.messages);

        let mut body = json!({ "contents": contents });
        if let Some(si) = system_instruction {
            body["systemInstruction"] = si;
        }

        if let Some(tools) = request.tools {
            let declarations = convert_tools(tools);
            if !declarations.is_empty() {
                body["tools"] = json!(declarations);
            }
        }

        let mut generation_config = json!({
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens,
        });
        if let Some(thinking) = thinking_config_for(request.thinking_level) {
            generation_config["thinkingConfig"] = thinking;
        }
        body["generationConfig"] = generation_config;

        body
    }

    /// Parse a non-streaming Gemini response into a [`ChatResponse`].
    fn parse_response(body: &Value) -> Result<ChatResponse> {
        let usage = parse_usage(body);

        let candidate = body
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first());

        let Some(candidate) = candidate else {
            // No candidates (e.g. a fully safety-blocked prompt). Surface an
            // empty assistant turn rather than erroring — the agent loop
            // treats it as a final, contentless response.
            return Ok(ChatResponse {
                content: None,
                tool_calls: Vec::new(),
                usage,
                reasoning: None,
            });
        };

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();

        if let Some(parts) = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
        {
            for part in parts {
                if part.get("thought").and_then(|t| t.as_bool()) == Some(true) {
                    if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                        reasoning.push_str(t);
                    }
                    continue;
                }
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    text.push_str(t);
                    continue;
                }
                if let Some(fc) = part.get("functionCall") {
                    if let Some(name) = fc
                        .get("name")
                        .and_then(|n| n.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        let arguments = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                        tool_calls.push(ToolCall {
                            id: gen_tool_call_id(),
                            name: name.to_string(),
                            arguments,
                        });
                    }
                }
            }
        }

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
}

/// Allowed keys for Gemini's `FunctionDeclaration.parameters` schema. Gemini
/// accepts only this subset of OpenAPI 3.0 / JSON Schema; keys outside it
/// (e.g. `$schema`, `additionalProperties`) are rejected, so strip them.
const GEMINI_SCHEMA_ALLOWED_KEYS: &[&str] = &[
    "type",
    "format",
    "title",
    "description",
    "nullable",
    "enum",
    "maxItems",
    "minItems",
    "properties",
    "required",
    "minProperties",
    "maxProperties",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "anyOf",
    "propertyOrdering",
    "default",
    "items",
    "minimum",
    "maximum",
];

/// Return a Gemini-compatible copy of a tool-parameter schema.
///
/// Fennec tool schemas are OpenAI-flavored JSON Schema and may carry keys
/// such as `$schema` or `additionalProperties` that Gemini's `Schema` object
/// rejects. Preserve the documented subset and recurse into nested
/// `properties` / `items` / `anyOf` definitions.
fn sanitize_gemini_schema(schema: &Value) -> Value {
    let Some(obj) = schema.as_object() else {
        return json!({});
    };

    let mut cleaned = Map::new();
    for (key, value) in obj {
        if !GEMINI_SCHEMA_ALLOWED_KEYS.contains(&key.as_str()) {
            continue;
        }
        match key.as_str() {
            "properties" => {
                let Some(props) = value.as_object() else {
                    continue;
                };
                let mut out = Map::new();
                for (prop_name, prop_schema) in props {
                    out.insert(prop_name.clone(), sanitize_gemini_schema(prop_schema));
                }
                cleaned.insert(key.clone(), Value::Object(out));
            }
            "items" => {
                cleaned.insert(key.clone(), sanitize_gemini_schema(value));
            }
            "anyOf" => {
                let Some(arr) = value.as_array() else {
                    continue;
                };
                let out: Vec<Value> = arr
                    .iter()
                    .filter(|item| item.is_object())
                    .map(sanitize_gemini_schema)
                    .collect();
                cleaned.insert(key.clone(), Value::Array(out));
            }
            _ => {
                cleaned.insert(key.clone(), value.clone());
            }
        }
    }

    // Gemini's Schema validator requires every `enum` entry to be a string,
    // even when the parent `type` is `integer` / `number` / `boolean`.
    // OpenAI / Anthropic accept typed enums, so only drop the `enum` when it
    // would collide with Gemini's rule — the `type` plus description still
    // guide the model, and the tool handler validates the value anyway.
    let type_is_numeric_or_bool = matches!(
        cleaned.get("type").and_then(|t| t.as_str()),
        Some("integer") | Some("number") | Some("boolean")
    );
    if type_is_numeric_or_bool {
        if let Some(Value::Array(enum_vals)) = cleaned.get("enum") {
            if enum_vals.iter().any(|v| !v.is_string()) {
                cleaned.remove("enum");
            }
        }
    }

    Value::Object(cleaned)
}

/// Normalize tool parameters to a valid Gemini object schema.
fn sanitize_gemini_tool_parameters(parameters: &Value) -> Value {
    let cleaned = sanitize_gemini_schema(parameters);
    if cleaned.as_object().map(|o| o.is_empty()).unwrap_or(true) {
        json!({ "type": "object", "properties": {} })
    } else {
        cleaned
    }
}

/// Convert Fennec's tool specs to Gemini's `tools` array shape:
/// `[{ "functionDeclarations": [ {name, description, parameters}, ... ] }]`.
fn convert_tools(tools: &[crate::tools::traits::ToolSpec]) -> Vec<Value> {
    let declarations: Vec<Value> = tools
        .iter()
        .map(|t| {
            let mut decl = json!({
                "name": t.name,
                "parameters": sanitize_gemini_tool_parameters(&t.parameters),
            });
            if !t.description.is_empty() {
                decl["description"] = json!(t.description);
            }
            decl
        })
        .collect();

    if declarations.is_empty() {
        Vec::new()
    } else {
        vec![json!({ "functionDeclarations": declarations })]
    }
}

/// Translate the conversation into Gemini `contents` plus an optional
/// `systemInstruction`. System text (from `request.system` and any
/// `system`-role messages) is collected separately, since Gemini carries it
/// outside `contents`.
fn build_contents(
    system: Option<&str>,
    messages: &[ChatMessage],
) -> (Vec<Value>, Option<Value>) {
    let mut system_parts: Vec<String> = Vec::new();
    if let Some(s) = system {
        if !s.is_empty() {
            system_parts.push(s.to_string());
        }
    }

    let mut contents: Vec<Value> = Vec::new();
    // Gemini's `functionResponse` part needs the tool's *name*, but Fennec's
    // tool-result messages only carry the `tool_call_id`. Rebuild the
    // id→name map from the assistant tool calls we pass on the way through.
    let mut tool_name_by_id: HashMap<String, String> = HashMap::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(c) = &msg.content {
                    if !c.is_empty() {
                        system_parts.push(c.clone());
                    }
                }
            }
            "tool" | "function" => {
                let call_id = msg.tool_call_id.clone().unwrap_or_default();
                let name = tool_name_by_id
                    .get(&call_id)
                    .cloned()
                    .unwrap_or_else(|| {
                        if call_id.is_empty() {
                            "tool".to_string()
                        } else {
                            call_id.clone()
                        }
                    });
                let content = msg.content.clone().unwrap_or_default();
                contents.push(json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": name,
                            "response": tool_response_value(&content),
                        }
                    }]
                }));
            }
            role => {
                let gemini_role = if role == "assistant" { "model" } else { "user" };
                let mut parts: Vec<Value> = Vec::new();

                if let Some(c) = &msg.content {
                    if !c.is_empty() {
                        parts.push(json!({ "text": c }));
                    }
                }

                if let Some(attachments) = &msg.attachments {
                    for a in attachments {
                        parts.push(json!({
                            "inlineData": {
                                "mimeType": a.mime_type,
                                "data": a.base64_data,
                            }
                        }));
                    }
                }

                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        if !tc.id.is_empty() && !tc.name.is_empty() {
                            tool_name_by_id.insert(tc.id.clone(), tc.name.clone());
                        }
                        let args = if tc.arguments.is_object() {
                            tc.arguments.clone()
                        } else {
                            json!({})
                        };
                        parts.push(json!({
                            "functionCall": { "name": tc.name, "args": args }
                        }));
                    }
                }

                if !parts.is_empty() {
                    contents.push(json!({ "role": gemini_role, "parts": parts }));
                }
            }
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(json!({ "parts": [{ "text": system_parts.join("\n") }] }))
    };

    (contents, system_instruction)
}

/// Build the Gemini `functionResponse.response` object from a tool result.
/// Gemini expects an object; if the tool emitted a JSON object we pass it
/// through, otherwise wrap the text under an `output` key.
fn tool_response_value(content: &str) -> Value {
    let trimmed = content.trim_start();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(parsed) = serde_json::from_str::<Value>(content) {
            if parsed.is_object() {
                return parsed;
            }
        }
    }
    json!({ "output": content })
}

/// Map Fennec's reasoning effort onto Gemini's `thinkingConfig`.
///
/// Budgets stay within both Gemini 2.5 Flash (0–24576) and Pro (128–32768)
/// ranges so a single mapping is safe across models. `Off` omits
/// `thinkingConfig` entirely: sending `thinkingBudget: 0` disables thinking
/// on Flash but is rejected by Pro (whose minimum is 128), so omitting it
/// lets each model fall back to its own default instead of erroring.
fn thinking_config_for(level: ThinkingLevel) -> Option<Value> {
    let budget = match level {
        ThinkingLevel::Off => return None,
        ThinkingLevel::Low => 2048,
        ThinkingLevel::Medium => 8192,
        ThinkingLevel::High => 16384,
        ThinkingLevel::Max => 24576,
    };
    Some(json!({ "thinkingBudget": budget, "includeThoughts": true }))
}

/// Parse Gemini's `usageMetadata` into [`UsageInfo`].
fn parse_usage(body: &Value) -> Option<UsageInfo> {
    body.get("usageMetadata").map(|u| UsageInfo {
        input_tokens: u
            .get("promptTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        output_tokens: u
            .get("candidatesTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        cache_read_tokens: u.get("cachedContentTokenCount").and_then(|v| v.as_u64()),
        // Gemini doesn't report a cache-write counter.
        cache_write_tokens: None,
    })
}

/// Generate a synthetic tool-call id. Gemini's `functionCall` parts carry no
/// id of their own, but Fennec's agent loop pairs each tool call with its
/// result by id, so we mint a unique one per call.
fn gen_tool_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

/// Extract a human-readable message from a Gemini error body, falling back to
/// the first 200 bytes of the raw payload.
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

/// Dispatch one Gemini streaming event payload.
///
/// Emits `Reasoning` / `Delta` for text parts and a complete
/// `ToolCallStart` → `ToolCallDelta` → `ToolCallEnd` group per `functionCall`
/// part (Gemini sends each call whole in a single event, so no cross-chunk
/// accumulation is needed — and the agent's stream consumer tracks only one
/// in-flight tool call at a time, so each group must be self-contained).
/// Captures `usageMetadata` into `usage_acc` and returns `true` once a
/// `finishReason` is observed.
async fn dispatch_gemini_event(
    data: &Value,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    usage_acc: &mut UsageInfo,
) -> bool {
    if let Some(u) = data.get("usageMetadata") {
        if let Some(v) = u.get("promptTokenCount").and_then(|v| v.as_u64()) {
            usage_acc.input_tokens = v;
        }
        if let Some(v) = u.get("candidatesTokenCount").and_then(|v| v.as_u64()) {
            usage_acc.output_tokens = v;
        }
        if let Some(v) = u.get("cachedContentTokenCount").and_then(|v| v.as_u64()) {
            usage_acc.cache_read_tokens = Some(v);
        }
    }

    let Some(candidate) = data
        .get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
    else {
        return false;
    };

    if let Some(parts) = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
    {
        for part in parts {
            if part.get("thought").and_then(|t| t.as_bool()) == Some(true) {
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    if !t.is_empty() {
                        let _ = tx.send(StreamEvent::Reasoning(t.to_string())).await;
                    }
                }
                continue;
            }
            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                if !t.is_empty() {
                    let _ = tx.send(StreamEvent::Delta(t.to_string())).await;
                }
                continue;
            }
            if let Some(fc) = part.get("functionCall") {
                if let Some(name) = fc
                    .get("name")
                    .and_then(|n| n.as_str())
                    .filter(|s| !s.is_empty())
                {
                    let id = gen_tool_call_id();
                    let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                    let _ = tx
                        .send(StreamEvent::ToolCallStart {
                            id: id.clone(),
                            name: name.to_string(),
                        })
                        .await;
                    let _ = tx
                        .send(StreamEvent::ToolCallDelta {
                            id: id.clone(),
                            arguments_delta: args.to_string(),
                        })
                        .await;
                    let _ = tx.send(StreamEvent::ToolCallEnd { id }).await;
                }
            }
        }
    }

    candidate
        .get("finishReason")
        .and_then(|r| r.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false)
}

/// Emit a final `Usage` (when token counts were seen) followed by `Done`.
async fn finish_stream(tx: &tokio::sync::mpsc::Sender<StreamEvent>, usage_acc: &UsageInfo) {
    if usage_acc.input_tokens > 0
        || usage_acc.output_tokens > 0
        || usage_acc.cache_read_tokens.is_some()
    {
        let _ = tx.send(StreamEvent::Usage(usage_acc.clone())).await;
    }
    let _ = tx.send(StreamEvent::Done).await;
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let body = self.build_request_body(&request);
        let url = format!("{}/models/{}:generateContent", self.base_url, self.model);

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .context("sending request to Gemini API")?;

        let status = response.status();
        let raw_body = response
            .bytes()
            .await
            .context("reading Gemini API response body")?;

        if !status.is_success() {
            anyhow::bail!(
                "Gemini API error ({}): {}",
                status,
                extract_error_message(&raw_body)
            );
        }

        let response_body: Value = serde_json::from_slice(&raw_body)
            .context("parsing Gemini API response as JSON")?;
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
        let body = self.build_request_body(&request);
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, self.model
        );

        let response = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .context("sending streaming request to Gemini API")?;

        let status = response.status();
        if !status.is_success() {
            let raw_body = response
                .bytes()
                .await
                .context("reading Gemini streaming error body")?;
            anyhow::bail!(
                "Gemini API error ({}): {}",
                status,
                extract_error_message(&raw_body)
            );
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let mut byte_stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut sse = SseBuffer::new();
            let mut usage_acc = UsageInfo::default();

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

                    // Gemini's SSE stream ends by closing the body rather than
                    // sending a sentinel, but tolerate `[DONE]` defensively.
                    if data_str == "[DONE]" {
                        finish_stream(&tx, &usage_acc).await;
                        return;
                    }

                    let data: Value = match serde_json::from_str(data_str) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    if dispatch_gemini_event(&data, &tx, &mut usage_acc).await {
                        finish_stream(&tx, &usage_acc).await;
                        return;
                    }
                }
            }

            finish_stream(&tx, &usage_acc).await;
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
    fn base_url_normalization_strips_slash_and_openai_suffix() {
        let p = GeminiProvider::new(
            "k".to_string(),
            None,
            Some("https://example.com/v1beta/openai/".to_string()),
            None,
        );
        assert_eq!(p.base_url, "https://example.com/v1beta");
        assert_eq!(p.model, DEFAULT_GEMINI_MODEL);
        assert_eq!(p.ctx_window, DEFAULT_GEMINI_CONTEXT_WINDOW);
    }

    #[test]
    fn build_contents_splits_system_instruction() {
        let messages = vec![ChatMessage::user("hello")];
        let (contents, system) = build_contents(Some("be terse"), &messages);
        assert_eq!(system.unwrap()["parts"][0]["text"], "be terse");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn build_contents_folds_system_role_messages() {
        let messages = vec![
            ChatMessage::system("rule one"),
            ChatMessage::user("hi"),
        ];
        let (contents, system) = build_contents(Some("rule zero"), &messages);
        // Both the explicit system arg and the system-role message are joined.
        assert_eq!(system.unwrap()["parts"][0]["text"], "rule zero\nrule one");
        // System-role message must NOT leak into contents.
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn build_contents_maps_assistant_role_to_model() {
        let messages = vec![ChatMessage::assistant("sure")];
        let (contents, _) = build_contents(None, &messages);
        assert_eq!(contents[0]["role"], "model");
        assert_eq!(contents[0]["parts"][0]["text"], "sure");
    }

    #[test]
    fn build_contents_emits_function_call_and_response_with_resolved_name() {
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "tc_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "/tmp/x" }),
        }]);
        let tool = ChatMessage::tool_result("tc_1", "file body");

        let (contents, _) = build_contents(None, &[assistant, tool]);
        assert_eq!(contents.len(), 2);

        // Assistant tool call → functionCall part in a `model` turn.
        assert_eq!(contents[0]["role"], "model");
        let fc = &contents[0]["parts"][0]["functionCall"];
        assert_eq!(fc["name"], "read_file");
        assert_eq!(fc["args"]["path"], "/tmp/x");

        // Tool result → functionResponse with the name resolved from the id.
        assert_eq!(contents[1]["role"], "user");
        let fr = &contents[1]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "read_file");
        // Non-JSON output is wrapped under `output`.
        assert_eq!(fr["response"]["output"], "file body");
    }

    #[test]
    fn tool_response_passes_through_json_object() {
        assert_eq!(
            tool_response_value("{\"ok\":true}"),
            json!({ "ok": true })
        );
        // Arrays are not valid Gemini response objects → wrapped.
        assert_eq!(
            tool_response_value("[1,2,3]"),
            json!({ "output": "[1,2,3]" })
        );
        assert_eq!(
            tool_response_value("plain text"),
            json!({ "output": "plain text" })
        );
    }

    #[test]
    fn build_contents_emits_inline_image_data() {
        let mut msg = ChatMessage::user("what is this");
        msg.attachments = Some(vec![ImageAttachmentRef {
            mime_type: "image/png".to_string(),
            base64_data: "AAAA".to_string(),
            display_name: None,
        }]);
        let (contents, _) = build_contents(None, &[msg]);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "what is this");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "AAAA");
    }

    #[test]
    fn convert_tools_wraps_in_function_declarations() {
        let specs = vec![spec(
            "my_tool",
            "Does stuff",
            json!({
                "type": "object",
                "properties": { "arg": { "type": "string" } }
            }),
        )];
        let tools = convert_tools(&specs);
        assert_eq!(tools.len(), 1);
        let decls = tools[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(decls[0]["name"], "my_tool");
        assert_eq!(decls[0]["description"], "Does stuff");
        assert_eq!(decls[0]["parameters"]["properties"]["arg"]["type"], "string");
    }

    #[test]
    fn convert_tools_empty_yields_empty() {
        assert!(convert_tools(&[]).is_empty());
    }

    #[test]
    fn sanitize_schema_strips_unsupported_keys_recursively() {
        let schema = json!({
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "additionalProperties": false,
            "properties": {
                "name": {
                    "type": "string",
                    "additionalProperties": false
                },
                "items": {
                    "type": "array",
                    "items": { "type": "string", "$comment": "drop me" }
                }
            },
            "required": ["name"]
        });
        let cleaned = sanitize_gemini_schema(&schema);
        assert!(cleaned.get("$schema").is_none());
        assert!(cleaned.get("additionalProperties").is_none());
        assert_eq!(cleaned["type"], "object");
        assert_eq!(cleaned["required"][0], "name");
        // Nested objects are sanitized too.
        assert!(cleaned["properties"]["name"]
            .get("additionalProperties")
            .is_none());
        assert!(cleaned["properties"]["items"]["items"]
            .get("$comment")
            .is_none());
        assert_eq!(
            cleaned["properties"]["items"]["items"]["type"],
            "string"
        );
    }

    #[test]
    fn sanitize_schema_drops_numeric_enum_but_keeps_string_enum() {
        let numeric = json!({ "type": "integer", "enum": [60, 1440] });
        let cleaned = sanitize_gemini_schema(&numeric);
        assert!(cleaned.get("enum").is_none(), "typed numeric enum must be dropped");
        assert_eq!(cleaned["type"], "integer");

        let string_enum = json!({ "type": "string", "enum": ["a", "b"] });
        let cleaned = sanitize_gemini_schema(&string_enum);
        assert_eq!(cleaned["enum"][0], "a");
    }

    #[test]
    fn sanitize_tool_parameters_defaults_empty_to_object() {
        let cleaned = sanitize_gemini_tool_parameters(&json!({ "$schema": "x" }));
        assert_eq!(cleaned, json!({ "type": "object", "properties": {} }));
    }

    #[test]
    fn thinking_config_maps_levels_within_caps() {
        assert!(thinking_config_for(ThinkingLevel::Off).is_none());
        for level in [
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::Max,
        ] {
            let cfg = thinking_config_for(level).unwrap();
            let budget = cfg["thinkingBudget"].as_u64().unwrap();
            // Within both Flash (0–24576) and Pro (128–32768) ranges.
            assert!((128..=24576).contains(&budget), "budget {budget} out of range");
            assert_eq!(cfg["includeThoughts"], true);
        }
    }

    #[test]
    fn build_request_body_assembles_expected_shape() {
        let p = GeminiProvider::new("k".to_string(), Some("gemini-2.5-pro".to_string()), None, None);
        let messages = vec![ChatMessage::user("hi")];
        let specs = vec![spec("t", "d", json!({ "type": "object", "properties": {} }))];
        let request = ChatRequest {
            system: Some("sys"),
            messages: &messages,
            tools: Some(&specs),
            max_tokens: 4096,
            temperature: 0.5,
            thinking_level: ThinkingLevel::High,
        };
        let body = p.build_request_body(&request);
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "sys");
        assert_eq!(body["contents"][0]["role"], "user");
        assert!(body["tools"][0]["functionDeclarations"].is_array());
        assert_eq!(body["generationConfig"]["temperature"], 0.5);
        assert_eq!(body["generationConfig"]["maxOutputTokens"], 4096);
        assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingBudget"], 16384);
    }

    #[test]
    fn parse_response_extracts_text_reasoning_and_usage() {
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "thought": true, "text": "thinking..." },
                        { "text": "Hello!" }
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 7,
                "totalTokenCount": 19,
                "cachedContentTokenCount": 3
            }
        });
        let resp = GeminiProvider::parse_response(&body).unwrap();
        assert_eq!(resp.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.reasoning.as_deref(), Some("thinking..."));
        assert!(resp.tool_calls.is_empty());
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.cache_read_tokens, Some(3));
    }

    #[test]
    fn parse_response_extracts_function_call() {
        let body = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "functionCall": { "name": "get_weather", "args": { "city": "SF" } } }
                    ]
                },
                "finishReason": "STOP"
            }]
        });
        let resp = GeminiProvider::parse_response(&body).unwrap();
        assert!(resp.content.is_none());
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].arguments["city"], "SF");
        assert!(resp.tool_calls[0].id.starts_with("call_"));
    }

    #[test]
    fn parse_response_empty_candidates_is_contentless() {
        let body = json!({ "candidates": [] });
        let resp = GeminiProvider::parse_response(&body).unwrap();
        assert!(resp.content.is_none());
        assert!(resp.tool_calls.is_empty());
    }

    /// Drain a receiver into a Vec<StreamEvent>.
    async fn drain(rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        out
    }

    #[tokio::test]
    async fn dispatch_event_emits_text_and_terminates_on_finish_reason() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let data = json!({
            "candidates": [{
                "content": { "parts": [{ "text": "partial" }] },
                "finishReason": "STOP"
            }]
        });
        let mut usage = UsageInfo::default();
        let terminate = dispatch_gemini_event(&data, &tx, &mut usage).await;
        let events = drain(&mut rx).await;
        assert!(terminate, "finishReason should terminate the stream");
        assert!(matches!(&events[0], StreamEvent::Delta(s) if s == "partial"));
    }

    #[tokio::test]
    async fn dispatch_event_emits_complete_tool_call_group() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let data = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "functionCall": { "name": "run", "args": { "x": 1 } } }
                    ]
                }
            }]
        });
        let mut usage = UsageInfo::default();
        let terminate = dispatch_gemini_event(&data, &tx, &mut usage).await;
        let events = drain(&mut rx).await;
        assert!(!terminate, "no finishReason → not terminal");
        // The consumer needs Start → Delta → End, in that order, all present.
        let id = match &events[0] {
            StreamEvent::ToolCallStart { id, name } => {
                assert_eq!(name, "run");
                id.clone()
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        };
        match &events[1] {
            StreamEvent::ToolCallDelta { id: did, arguments_delta } => {
                assert_eq!(did, &id);
                assert!(arguments_delta.contains("\"x\""));
            }
            other => panic!("expected ToolCallDelta, got {other:?}"),
        }
        match &events[2] {
            StreamEvent::ToolCallEnd { id: eid } => assert_eq!(eid, &id),
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn dispatch_event_emits_reasoning_for_thought_parts() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let data = json!({
            "candidates": [{
                "content": { "parts": [{ "thought": true, "text": "hmm" }] }
            }]
        });
        let mut usage = UsageInfo::default();
        dispatch_gemini_event(&data, &tx, &mut usage).await;
        let events = drain(&mut rx).await;
        assert!(matches!(&events[0], StreamEvent::Reasoning(s) if s == "hmm"));
    }

    #[tokio::test]
    async fn dispatch_event_accumulates_usage_metadata() {
        let (tx, mut _rx) = tokio::sync::mpsc::channel(16);
        let data = json!({
            "candidates": [{ "content": { "parts": [] }, "finishReason": "STOP" }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 9,
                "cachedContentTokenCount": 2
            }
        });
        let mut usage = UsageInfo::default();
        dispatch_gemini_event(&data, &tx, &mut usage).await;
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 9);
        assert_eq!(usage.cache_read_tokens, Some(2));
    }
}
