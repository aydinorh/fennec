//! AWS Bedrock provider (Converse API), hand-rolled — no AWS SDK.
//!
//! Talks to `bedrock-runtime.{region}.amazonaws.com` using the unified Converse
//! API (`/model/{modelId}/converse` and `/converse-stream`), signing each
//! request with SigV4 (see [`super::aws_sigv4`]). Streaming decodes the AWS
//! binary event-stream protocol (`application/vnd.amazon.eventstream`).
//!
//! Credentials resolve from env vars (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
//! / optional `AWS_SESSION_TOKEN`); when absent, from the EC2/EKS instance role
//! via IMDSv2. SSO / named-profile resolution is a follow-up. Region comes from
//! `AWS_REGION` / `AWS_DEFAULT_REGION` (default `us-east-1`).
//!
//! `provider.model` is the Bedrock model id or inference-profile id, e.g.
//! `anthropic.claude-3-5-sonnet-20241022-v2:0` or `us.anthropic.claude-…`.
//!
//! Follow-up: enabling native extended thinking is model-specific (Claude on
//! Bedrock uses `additionalModelRequestFields.reasoning_config`); we surface
//! `reasoningContent` if the model emits it but don't yet send the enable flag.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::{json, Value};

use super::aws_sigv4::{sign, AwsCredentials};
use super::traits::{
    ChatMessage, ChatRequest, ChatResponse, Provider, StreamEvent, ToolCall, UsageInfo,
};
use crate::agent::thinking::ThinkingLevel;

const SERVICE: &str = "bedrock";
const DEFAULT_REGION: &str = "us-east-1";
const DEFAULT_CONTEXT_WINDOW: usize = 200_000;
const IMDS_TIMEOUT: Duration = Duration::from_secs(2);
const CRED_SKEW: Duration = Duration::from_secs(120);

/// Where the provider's AWS credentials come from.
enum CredSource {
    /// Static credentials from the environment.
    Static(AwsCredentials),
    /// EC2/EKS instance role via IMDSv2 (fetched + cached lazily).
    Imds,
}

/// AWS Bedrock provider over the Converse API.
pub struct BedrockProvider {
    client: reqwest::Client,
    model: String,
    region: String,
    host: String,
    ctx_window: usize,
    cred_source: CredSource,
    /// Cached IMDS credentials + their expiry (unused for static creds).
    cached_imds: Mutex<Option<(AwsCredentials, Instant)>>,
}

impl BedrockProvider {
    /// Create a Bedrock provider. `region` defaults to `AWS_REGION` /
    /// `AWS_DEFAULT_REGION` then `us-east-1`; `base_url`, when set, overrides
    /// the derived `bedrock-runtime.{region}.amazonaws.com` host.
    pub fn new(model: Option<String>, base_url: Option<String>, context_window: Option<usize>) -> Self {
        let region = std::env::var("AWS_REGION")
            .ok()
            .filter(|v| !v.is_empty())
            .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok().filter(|v| !v.is_empty()))
            .unwrap_or_else(|| DEFAULT_REGION.to_string());

        let host = match base_url.filter(|b| !b.is_empty()) {
            Some(b) => b
                .trim_start_matches("https://")
                .trim_start_matches("http://")
                .trim_end_matches('/')
                .to_string(),
            None => format!("bedrock-runtime.{region}.amazonaws.com"),
        };

        let cred_source = match (
            std::env::var("AWS_ACCESS_KEY_ID").ok().filter(|v| !v.is_empty()),
            std::env::var("AWS_SECRET_ACCESS_KEY").ok().filter(|v| !v.is_empty()),
        ) {
            (Some(access_key_id), Some(secret_access_key)) => {
                CredSource::Static(AwsCredentials {
                    access_key_id,
                    secret_access_key,
                    session_token: std::env::var("AWS_SESSION_TOKEN").ok().filter(|v| !v.is_empty()),
                })
            }
            _ => CredSource::Imds,
        };

        Self {
            client: reqwest::Client::new(),
            model: model.unwrap_or_default(),
            region,
            host,
            ctx_window: context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
            cred_source,
            cached_imds: Mutex::new(None),
        }
    }

    async fn credentials(&self) -> Result<AwsCredentials> {
        match &self.cred_source {
            CredSource::Static(c) => Ok(c.clone()),
            CredSource::Imds => self.imds_credentials().await,
        }
    }

    /// Fetch (and cache) instance-role credentials via IMDSv2.
    async fn imds_credentials(&self) -> Result<AwsCredentials> {
        if let Some((creds, expiry)) = self.cached_imds.lock().as_ref() {
            if *expiry > Instant::now() {
                return Ok(creds.clone());
            }
        }

        let imds = reqwest::Client::builder()
            .timeout(IMDS_TIMEOUT)
            .build()
            .context("building IMDS client")?;

        // IMDSv2: obtain a session token first.
        let token = imds
            .put("http://169.254.169.254/latest/api/token")
            .header("X-aws-ec2-metadata-token-ttl-seconds", "21600")
            .send()
            .await
            .context("requesting IMDSv2 token (no AWS credentials in env, and instance metadata unreachable)")?
            .error_for_status()
            .context("IMDSv2 token request rejected")?
            .text()
            .await
            .context("reading IMDSv2 token")?;

        let role = imds
            .get("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
            .header("X-aws-ec2-metadata-token", &token)
            .send()
            .await
            .context("listing IMDS instance roles")?
            .error_for_status()
            .context("no IAM role attached to this instance")?
            .text()
            .await
            .context("reading IMDS role name")?;
        let role = role.lines().next().unwrap_or("").trim().to_string();

        let creds_json: Value = imds
            .get(format!(
                "http://169.254.169.254/latest/meta-data/iam/security-credentials/{role}"
            ))
            .header("X-aws-ec2-metadata-token", &token)
            .send()
            .await
            .context("fetching IMDS role credentials")?
            .error_for_status()
            .context("IMDS role credentials request rejected")?
            .json()
            .await
            .context("parsing IMDS role credentials")?;

        let creds = AwsCredentials {
            access_key_id: creds_json["AccessKeyId"].as_str().unwrap_or("").to_string(),
            secret_access_key: creds_json["SecretAccessKey"].as_str().unwrap_or("").to_string(),
            session_token: creds_json["Token"].as_str().map(String::from),
        };
        if creds.access_key_id.is_empty() {
            anyhow::bail!("IMDS returned empty credentials");
        }

        // IMDS credentials are temporary; cache until shortly before expiry.
        // Fall back to a conservative 5-minute TTL if the field is unparseable.
        let ttl = creds_json["Expiration"]
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|exp| {
                let secs = (exp.timestamp() - chrono::Utc::now().timestamp()).max(0) as u64;
                Duration::from_secs(secs)
            })
            .unwrap_or(Duration::from_secs(300));
        let expiry = Instant::now() + ttl.saturating_sub(CRED_SKEW);
        *self.cached_imds.lock() = Some((creds.clone(), expiry));
        Ok(creds)
    }

    fn endpoint_path(&self, stream: bool) -> String {
        let enc_model = urlencoding::encode(&self.model);
        let action = if stream { "converse-stream" } else { "converse" };
        format!("/model/{enc_model}/{action}")
    }

    /// Build the signed request for a Converse call.
    async fn build_signed_request(
        &self,
        request: &ChatRequest<'_>,
        stream: bool,
    ) -> Result<reqwest::RequestBuilder> {
        let body = build_converse_body(request);
        let payload = serde_json::to_vec(&body).context("serializing Converse body")?;
        let path = self.endpoint_path(stream);
        let url = format!("https://{}{}", self.host, path);

        let creds = self.credentials().await?;
        let amz_date = chrono::Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let signed = sign(
            "POST",
            &path,
            "",
            &self.host,
            &amz_date,
            &self.region,
            SERVICE,
            &creds,
            &[("content-type", "application/json")],
            &payload,
            true,
        );

        let mut req = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .body(payload);
        for (name, value) in signed {
            req = req.header(name, value);
        }
        Ok(req)
    }
}

fn map_thinking_temperature(level: ThinkingLevel, base: f64) -> f64 {
    // Bedrock thinking is model-specific (not enabled here), so honor /think
    // generically via temperature, matching the fallback other non-native
    // providers use.
    match level {
        ThinkingLevel::Off => base,
        ThinkingLevel::Low => 0.5,
        ThinkingLevel::Medium => 0.3,
        ThinkingLevel::High => 0.15,
        ThinkingLevel::Max => 0.05,
    }
}

/// Build the Converse request body from a [`ChatRequest`].
fn build_converse_body(request: &ChatRequest<'_>) -> Value {
    let (system, messages) = build_converse_messages(request.system, request.messages);

    let mut body = json!({
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": request.max_tokens,
            "temperature": map_thinking_temperature(request.thinking_level, request.temperature),
        },
    });
    if let Some(system) = system {
        body["system"] = system;
    }
    if let Some(tools) = request.tools {
        let specs = convert_tools(tools);
        if !specs.is_empty() {
            body["toolConfig"] = json!({ "tools": specs });
        }
    }
    body
}

fn convert_tools(tools: &[crate::tools::traits::ToolSpec]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": { "json": t.parameters },
                }
            })
        })
        .collect()
}

/// Translate Fennec messages into Converse `(system, messages)`. Converse
/// requires strict user/assistant alternation (consecutive same-role messages
/// are merged) and a leading user turn; tool results ride in a user message.
fn build_converse_messages(
    system: Option<&str>,
    messages: &[ChatMessage],
) -> (Option<Value>, Vec<Value>) {
    let mut system_blocks: Vec<Value> = Vec::new();
    if let Some(s) = system {
        if !s.is_empty() {
            system_blocks.push(json!({ "text": s }));
        }
    }

    let mut converse: Vec<Value> = Vec::new();

    // Append content blocks to the last message if it shares `role`, else push
    // a new message — preserving Converse's required alternation.
    fn push_blocks(converse: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
        if let Some(last) = converse.last_mut() {
            if last["role"] == role {
                last["content"].as_array_mut().unwrap().extend(blocks);
                return;
            }
        }
        converse.push(json!({ "role": role, "content": blocks }));
    }

    fn text_block(s: &str) -> Value {
        // Bedrock rejects empty text blocks.
        json!({ "text": if s.is_empty() { " " } else { s } })
    }

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(c) = &msg.content {
                    if !c.is_empty() {
                        system_blocks.push(json!({ "text": c }));
                    }
                }
            }
            "tool" | "function" => {
                let tool_use_id = msg.tool_call_id.clone().unwrap_or_default();
                let content = msg.content.clone().unwrap_or_default();
                let block = json!({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{ "text": if content.is_empty() { " ".to_string() } else { content } }],
                    }
                });
                push_blocks(&mut converse, "user", vec![block]);
            }
            "assistant" => {
                let mut blocks: Vec<Value> = Vec::new();
                if let Some(c) = &msg.content {
                    if !c.is_empty() {
                        blocks.push(text_block(c));
                    }
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        let input = if tc.arguments.is_object() {
                            tc.arguments.clone()
                        } else {
                            json!({})
                        };
                        blocks.push(json!({
                            "toolUse": { "toolUseId": tc.id, "name": tc.name, "input": input }
                        }));
                    }
                }
                if blocks.is_empty() {
                    blocks.push(text_block(""));
                }
                push_blocks(&mut converse, "assistant", blocks);
            }
            _ => {
                let mut blocks: Vec<Value> = Vec::new();
                if let Some(c) = &msg.content {
                    if !c.is_empty() {
                        blocks.push(text_block(c));
                    }
                }
                if let Some(attachments) = &msg.attachments {
                    for a in attachments {
                        let format = a.mime_type.rsplit('/').next().unwrap_or("jpeg");
                        blocks.push(json!({
                            "image": { "format": format, "source": { "bytes": a.base64_data } }
                        }));
                    }
                }
                if blocks.is_empty() {
                    blocks.push(text_block(""));
                }
                push_blocks(&mut converse, "user", blocks);
            }
        }
    }

    // Converse requires the conversation to begin with a user turn.
    if converse
        .first()
        .map(|m| m["role"] != "user")
        .unwrap_or(false)
    {
        converse.insert(0, json!({ "role": "user", "content": [{ "text": " " }] }));
    }

    let system = if system_blocks.is_empty() {
        None
    } else {
        Some(Value::Array(system_blocks))
    };
    (system, converse)
}

fn parse_usage(usage: &Value) -> UsageInfo {
    UsageInfo {
        input_tokens: usage["inputTokens"].as_u64().unwrap_or(0),
        output_tokens: usage["outputTokens"].as_u64().unwrap_or(0),
        cache_read_tokens: usage["cacheReadInputTokens"].as_u64(),
        cache_write_tokens: usage["cacheWriteInputTokens"].as_u64(),
    }
}

/// Parse a non-streaming Converse response into a [`ChatResponse`].
fn parse_converse_response(body: &Value) -> Result<ChatResponse> {
    let blocks = body["output"]["message"]["content"].as_array();

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();

    if let Some(blocks) = blocks {
        for block in blocks {
            if let Some(t) = block["text"].as_str() {
                text.push_str(t);
            } else if let Some(rt) = block["reasoningContent"]["reasoningText"]["text"]
                .as_str()
                .or_else(|| block["reasoningContent"]["text"].as_str())
            {
                reasoning.push_str(rt);
            } else if block.get("toolUse").map(|v| v.is_object()).unwrap_or(false) {
                let tu = &block["toolUse"];
                tool_calls.push(ToolCall {
                    id: tu["toolUseId"].as_str().unwrap_or("").to_string(),
                    name: tu["name"].as_str().unwrap_or("").to_string(),
                    arguments: tu.get("input").cloned().unwrap_or_else(|| json!({})),
                });
            }
        }
    }

    let usage = body.get("usage").filter(|u| u.is_object()).map(parse_usage);

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

// ---------------------------------------------------------------------------
// AWS binary event-stream decoder
// ---------------------------------------------------------------------------

/// Accumulates response bytes and yields complete `vnd.amazon.eventstream`
/// frames as `(event_type, message_type, payload_json)`. CRC fields are not
/// validated — the bytes already arrive over TLS.
struct EventStreamDecoder {
    buf: Vec<u8>,
}

impl EventStreamDecoder {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn extend(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Pop the next complete frame, if the buffer holds one.
    fn next_frame(&mut self) -> Option<(String, String, Value)> {
        if self.buf.len() < 12 {
            return None;
        }
        let total = u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]) as usize;
        let headers_len =
            u32::from_be_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;
        if total < 16 || self.buf.len() < total || 12 + headers_len > total {
            return None;
        }

        let headers = parse_event_headers(&self.buf[12..12 + headers_len]);
        let payload = &self.buf[12 + headers_len..total - 4];
        let json: Value = serde_json::from_slice(payload).unwrap_or(Value::Null);
        let event_type = headers
            .iter()
            .find(|(k, _)| k == ":event-type")
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        let message_type = headers
            .iter()
            .find(|(k, _)| k == ":message-type")
            .map(|(_, v)| v.clone())
            .unwrap_or_default();

        self.buf.drain(..total);
        Some((event_type, message_type, json))
    }
}

/// Parse event-stream headers, returning the string-valued ones (the only kind
/// Bedrock sends: `:event-type`, `:message-type`, `:content-type`, …).
fn parse_event_headers(mut data: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    while data.len() >= 2 {
        let name_len = data[0] as usize;
        if 1 + name_len + 1 > data.len() {
            break;
        }
        let name = String::from_utf8_lossy(&data[1..1 + name_len]).to_string();
        let value_type = data[1 + name_len];
        let mut cursor = 1 + name_len + 1;
        // Type 7 = string: u16 length-prefixed UTF-8. Other types aren't used
        // by Bedrock's converse stream, so stop if we hit one.
        if value_type != 7 {
            break;
        }
        if cursor + 2 > data.len() {
            break;
        }
        let val_len = u16::from_be_bytes([data[cursor], data[cursor + 1]]) as usize;
        cursor += 2;
        if cursor + val_len > data.len() {
            break;
        }
        let value = String::from_utf8_lossy(&data[cursor..cursor + val_len]).to_string();
        out.push((name, value));
        data = &data[cursor + val_len..];
    }
    out
}

/// Dispatch a decoded Converse stream frame to `StreamEvent`s. Tracks the
/// in-flight toolUse id so each tool call is emitted as a complete
/// start/delta/end group. Returns `true` once the stream should terminate.
async fn dispatch_converse_frame(
    event_type: &str,
    message_type: &str,
    payload: &Value,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    current_tool: &mut Option<String>,
    usage_acc: &mut Option<UsageInfo>,
) -> bool {
    if message_type == "exception" || message_type == "error" {
        let msg = payload
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("Bedrock stream error")
            .to_string();
        let _ = tx.send(StreamEvent::Error(msg)).await;
        return true;
    }

    match event_type {
        "contentBlockStart" => {
            let tu = &payload["start"]["toolUse"];
            if tu.is_object() {
                let id = tu["toolUseId"].as_str().unwrap_or("").to_string();
                let name = tu["name"].as_str().unwrap_or("").to_string();
                *current_tool = Some(id.clone());
                let _ = tx.send(StreamEvent::ToolCallStart { id, name }).await;
            }
            false
        }
        "contentBlockDelta" => {
            let delta = &payload["delta"];
            if let Some(text) = delta["text"].as_str() {
                if !text.is_empty() {
                    let _ = tx.send(StreamEvent::Delta(text.to_string())).await;
                }
            } else if let Some(input) = delta["toolUse"]["input"].as_str() {
                if let Some(id) = current_tool.clone() {
                    let _ = tx
                        .send(StreamEvent::ToolCallDelta {
                            id,
                            arguments_delta: input.to_string(),
                        })
                        .await;
                }
            } else if let Some(rt) = delta["reasoningContent"]["text"].as_str() {
                if !rt.is_empty() {
                    let _ = tx.send(StreamEvent::Reasoning(rt.to_string())).await;
                }
            }
            false
        }
        "contentBlockStop" => {
            if let Some(id) = current_tool.take() {
                let _ = tx.send(StreamEvent::ToolCallEnd { id }).await;
            }
            false
        }
        "metadata" => {
            if payload["usage"].is_object() {
                *usage_acc = Some(parse_usage(&payload["usage"]));
            }
            false
        }
        "messageStop" => true,
        _ => false,
    }
}

#[async_trait]
impl Provider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let req = self.build_signed_request(&request, false).await?;
        let response = req.send().await.context("sending request to Bedrock")?;

        let status = response.status();
        let raw_body = response.bytes().await.context("reading Bedrock response body")?;
        if !status.is_success() {
            let msg = serde_json::from_slice::<Value>(&raw_body)
                .ok()
                .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(String::from))
                .unwrap_or_else(|| String::from_utf8_lossy(&raw_body).chars().take(200).collect());
            anyhow::bail!("Bedrock API error ({status}): {msg}");
        }

        let body: Value =
            serde_json::from_slice(&raw_body).context("parsing Bedrock response as JSON")?;
        parse_converse_response(&body)
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
        let req = self.build_signed_request(&request, true).await?;
        let response = req.send().await.context("sending streaming request to Bedrock")?;

        let status = response.status();
        if !status.is_success() {
            let raw_body = response
                .bytes()
                .await
                .context("reading Bedrock streaming error body")?;
            let msg = serde_json::from_slice::<Value>(&raw_body)
                .ok()
                .and_then(|v| v.get("message").and_then(|m| m.as_str()).map(String::from))
                .unwrap_or_else(|| String::from_utf8_lossy(&raw_body).chars().take(200).collect());
            anyhow::bail!("Bedrock API error ({status}): {msg}");
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let mut byte_stream = response.bytes_stream();

        tokio::spawn(async move {
            let mut decoder = EventStreamDecoder::new();
            let mut current_tool: Option<String> = None;
            let mut usage_acc: Option<UsageInfo> = None;

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                        return;
                    }
                };
                decoder.extend(&chunk);

                while let Some((event_type, message_type, payload)) = decoder.next_frame() {
                    let terminal = dispatch_converse_frame(
                        &event_type,
                        &message_type,
                        &payload,
                        &tx,
                        &mut current_tool,
                        &mut usage_acc,
                    )
                    .await;
                    if terminal {
                        if let Some(usage) = usage_acc.take() {
                            let _ = tx.send(StreamEvent::Usage(usage)).await;
                        }
                        let _ = tx.send(StreamEvent::Done).await;
                        return;
                    }
                }
            }

            if let Some(usage) = usage_acc.take() {
                let _ = tx.send(StreamEvent::Usage(usage)).await;
            }
            let _ = tx.send(StreamEvent::Done).await;
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::traits::ImageAttachmentRef;

    fn spec(name: &str, parameters: Value) -> crate::tools::traits::ToolSpec {
        crate::tools::traits::ToolSpec {
            name: name.to_string(),
            description: "d".to_string(),
            parameters,
        }
    }

    #[test]
    fn messages_split_system_and_require_leading_user() {
        let messages = vec![ChatMessage::user("hello")];
        let (system, converse) = build_converse_messages(Some("be terse"), &messages);
        assert_eq!(system.unwrap()[0]["text"], "be terse");
        assert_eq!(converse.len(), 1);
        assert_eq!(converse[0]["role"], "user");
        assert_eq!(converse[0]["content"][0]["text"], "hello");
    }

    #[test]
    fn assistant_tool_call_and_tool_result() {
        // Realistic order: a user turn, then the assistant's tool call, then
        // the tool result (which rides in a user turn).
        let user = ChatMessage::user("do it");
        let mut assistant = ChatMessage::assistant("");
        assistant.tool_calls = Some(vec![ToolCall {
            id: "tu_1".to_string(),
            name: "read".to_string(),
            arguments: json!({ "path": "/x" }),
        }]);
        let tool = ChatMessage::tool_result("tu_1", "body");

        let (_s, converse) = build_converse_messages(None, &[user, assistant, tool]);
        assert_eq!(converse.len(), 3);
        assert_eq!(converse[0]["role"], "user");
        // Assistant turn carries the toolUse block.
        assert_eq!(converse[1]["role"], "assistant");
        let tu = &converse[1]["content"][0]["toolUse"];
        assert_eq!(tu["toolUseId"], "tu_1");
        assert_eq!(tu["name"], "read");
        assert_eq!(tu["input"]["path"], "/x");
        // Tool result rides in a user turn as a toolResult block.
        assert_eq!(converse[2]["role"], "user");
        let tr = &converse[2]["content"][0]["toolResult"];
        assert_eq!(tr["toolUseId"], "tu_1");
        assert_eq!(tr["content"][0]["text"], "body");
    }

    #[test]
    fn consecutive_same_role_messages_merge() {
        // Two user messages in a row must merge (Converse needs alternation).
        let msgs = vec![ChatMessage::user("a"), ChatMessage::user("b")];
        let (_s, converse) = build_converse_messages(None, &msgs);
        assert_eq!(converse.len(), 1);
        assert_eq!(converse[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn leading_assistant_gets_synthetic_user_turn() {
        let msgs = vec![ChatMessage::assistant("hi there")];
        let (_s, converse) = build_converse_messages(None, &msgs);
        assert_eq!(converse[0]["role"], "user");
        assert_eq!(converse[1]["role"], "assistant");
    }

    #[test]
    fn user_image_attachment_becomes_image_block() {
        let mut msg = ChatMessage::user("what is this");
        msg.attachments = Some(vec![ImageAttachmentRef {
            mime_type: "image/png".to_string(),
            base64_data: "AAAA".to_string(),
            display_name: None,
        }]);
        let (_s, converse) = build_converse_messages(None, &[msg]);
        let content = converse[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["text"], "what is this");
        assert_eq!(content[1]["image"]["format"], "png");
        assert_eq!(content[1]["image"]["source"]["bytes"], "AAAA");
    }

    #[test]
    fn tools_become_tool_specs() {
        let specs = convert_tools(&[spec("t", json!({ "type": "object", "properties": {} }))]);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0]["toolSpec"]["name"], "t");
        assert!(specs[0]["toolSpec"]["inputSchema"]["json"].is_object());
    }

    #[test]
    fn build_body_sets_inference_config_and_tools() {
        let messages = vec![ChatMessage::user("hi")];
        let specs = vec![spec("t", json!({ "type": "object", "properties": {} }))];
        let request = ChatRequest {
            system: Some("sys"),
            messages: &messages,
            tools: Some(&specs),
            max_tokens: 4096,
            temperature: 0.5,
            thinking_level: ThinkingLevel::Off,
        };
        let body = build_converse_body(&request);
        assert_eq!(body["inferenceConfig"]["maxTokens"], 4096);
        assert_eq!(body["inferenceConfig"]["temperature"], 0.5);
        assert_eq!(body["system"][0]["text"], "sys");
        assert!(body["toolConfig"]["tools"].is_array());
    }

    #[test]
    fn thinking_level_lowers_temperature() {
        let messages = vec![ChatMessage::user("hi")];
        let request = ChatRequest {
            system: None,
            messages: &messages,
            tools: None,
            max_tokens: 100,
            temperature: 0.7,
            thinking_level: ThinkingLevel::High,
        };
        let body = build_converse_body(&request);
        assert_eq!(body["inferenceConfig"]["temperature"], 0.15);
    }

    #[test]
    fn parse_response_text_tooluse_and_usage() {
        let body = json!({
            "output": { "message": { "role": "assistant", "content": [
                { "text": "Hello" },
                { "toolUse": { "toolUseId": "tu_9", "name": "get", "input": { "x": 1 } } }
            ] } },
            "stopReason": "tool_use",
            "usage": { "inputTokens": 12, "outputTokens": 5, "cacheReadInputTokens": 3 }
        });
        let resp = parse_converse_response(&body).unwrap();
        assert_eq!(resp.content.as_deref(), Some("Hello"));
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "tu_9");
        assert_eq!(resp.tool_calls[0].arguments["x"], 1);
        let usage = resp.usage.unwrap();
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.cache_read_tokens, Some(3));
    }

    #[test]
    fn endpoint_path_encodes_model_id() {
        let p = BedrockProvider::new(
            Some("anthropic.claude-3-5-sonnet-20241022-v2:0".to_string()),
            None,
            None,
        );
        // The ':' in the version suffix must be percent-encoded in the path.
        assert_eq!(
            p.endpoint_path(false),
            "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse"
        );
        assert!(p.endpoint_path(true).ends_with("/converse-stream"));
    }

    /// Build a single AWS event-stream frame from headers + payload (CRCs are
    /// zeroed — the decoder doesn't validate them).
    fn frame(event_type: &str, payload: &Value) -> Vec<u8> {
        let mut headers = Vec::new();
        let mut put_str_header = |name: &str, value: &str| {
            headers.push(name.len() as u8);
            headers.extend_from_slice(name.as_bytes());
            headers.push(7u8); // string type
            headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
            headers.extend_from_slice(value.as_bytes());
        };
        put_str_header(":event-type", event_type);
        put_str_header(":message-type", "event");
        let payload_bytes = serde_json::to_vec(payload).unwrap();
        let total = 4 + 4 + 4 + headers.len() + payload_bytes.len() + 4;
        let mut out = Vec::new();
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]); // prelude CRC (ignored)
        out.extend_from_slice(&headers);
        out.extend_from_slice(&payload_bytes);
        out.extend_from_slice(&[0, 0, 0, 0]); // message CRC (ignored)
        out
    }

    #[test]
    fn decoder_yields_frames_including_split_across_chunks() {
        let f1 = frame("contentBlockDelta", &json!({ "delta": { "text": "Hi" } }));
        let f2 = frame("messageStop", &json!({ "stopReason": "end_turn" }));
        let mut dec = EventStreamDecoder::new();

        // Feed f1 split across two chunks.
        let mid = f1.len() / 2;
        dec.extend(&f1[..mid]);
        assert!(dec.next_frame().is_none(), "partial frame must not decode");
        dec.extend(&f1[mid..]);
        let (et, _mt, payload) = dec.next_frame().unwrap();
        assert_eq!(et, "contentBlockDelta");
        assert_eq!(payload["delta"]["text"], "Hi");

        dec.extend(&f2);
        let (et2, _mt, _p) = dec.next_frame().unwrap();
        assert_eq!(et2, "messageStop");
        assert!(dec.next_frame().is_none());
    }

    async fn drain(rx: &mut tokio::sync::mpsc::Receiver<StreamEvent>) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        while let Ok(e) = rx.try_recv() {
            out.push(e);
        }
        out
    }

    #[tokio::test]
    async fn dispatch_tool_use_emits_start_delta_end() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let mut current = None;
        let mut usage = None;
        dispatch_converse_frame(
            "contentBlockStart",
            "event",
            &json!({ "start": { "toolUse": { "toolUseId": "tu_1", "name": "run" } } }),
            &tx,
            &mut current,
            &mut usage,
        )
        .await;
        dispatch_converse_frame(
            "contentBlockDelta",
            "event",
            &json!({ "delta": { "toolUse": { "input": "{\"x\":1}" } } }),
            &tx,
            &mut current,
            &mut usage,
        )
        .await;
        dispatch_converse_frame(
            "contentBlockStop",
            "event",
            &json!({}),
            &tx,
            &mut current,
            &mut usage,
        )
        .await;
        let events = drain(&mut rx).await;
        assert!(matches!(&events[0], StreamEvent::ToolCallStart { id, name } if id == "tu_1" && name == "run"));
        assert!(matches!(&events[1], StreamEvent::ToolCallDelta { id, arguments_delta } if id == "tu_1" && arguments_delta.contains("\"x\"")));
        assert!(matches!(&events[2], StreamEvent::ToolCallEnd { id } if id == "tu_1"));
        assert!(current.is_none());
    }

    #[tokio::test]
    async fn dispatch_message_stop_is_terminal() {
        let (tx, mut _rx) = tokio::sync::mpsc::channel(16);
        let mut current = None;
        let mut usage = None;
        let terminal = dispatch_converse_frame(
            "messageStop",
            "event",
            &json!({ "stopReason": "end_turn" }),
            &tx,
            &mut current,
            &mut usage,
        )
        .await;
        assert!(terminal);
    }
}
