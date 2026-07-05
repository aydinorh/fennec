//! Vision tool — analyzes images via the configured vision-capable provider.
//!
//! Accepts a local file path or URL, base64-encodes the image, and sends it
//! to Anthropic, OpenAI, or Google Gemini's vision API with an optional
//! question. Returns the model's text response.
//!
//! Why this lives as a tool rather than native multimodal messages: the
//! existing agent message type is text-only. Wrapping the vision call in
//! a tool keeps the change fully additive — no provider trait updates.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde_json::{json, Value};

use crate::providers::openai::is_reasoning_model as openai_is_reasoning_model;
use crate::security::url_guard::{build_guarded_client, read_body_capped, validate_url_str};
use crate::security::PathSandbox;

use super::traits::{Tool, ToolResult};

/// Cap on image bytes downloaded for vision analysis. Big enough for any
/// plausible photo; keeps hostile URLs from OOMing the process.
const MAX_IMAGE_FETCH_BYTES: usize = 20 * 1024 * 1024;

/// Which vision backend to use (dispatched by provider name).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionBackend {
    Anthropic,
    OpenAi,
    Gemini,
}

impl VisionBackend {
    /// Pick a backend from a provider name. Returns None if the provider
    /// doesn't support vision.
    pub fn from_provider_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "anthropic" => Some(Self::Anthropic),
            "openai" => Some(Self::OpenAi),
            "google" | "gemini" => Some(Self::Gemini),
            _ => None,
        }
    }
}

pub struct VisionTool {
    backend: VisionBackend,
    api_key: String,
    model: String,
    client: reqwest::Client,
    /// Applied to local-path sources; URL sources are unaffected.
    sandbox: Arc<PathSandbox>,
}

impl VisionTool {
    /// Construct a VisionTool from a provider config. Returns None if the
    /// provider doesn't support vision (caller can then skip wiring this
    /// tool without failing startup).
    pub fn from_provider(
        provider_name: &str,
        api_key: String,
        model: Option<String>,
    ) -> Option<Self> {
        let backend = VisionBackend::from_provider_name(provider_name)?;
        if api_key.is_empty() {
            return None;
        }
        let model = model.unwrap_or_else(|| match backend {
            VisionBackend::Anthropic => "claude-sonnet-4-6".to_string(),
            VisionBackend::OpenAi => "gpt-4o".to_string(),
            VisionBackend::Gemini => "gemini-2.5-flash".to_string(),
        });
        let client = build_guarded_client(std::time::Duration::from_secs(60));
        Some(Self {
            backend,
            api_key,
            model,
            client,
            sandbox: Arc::new(PathSandbox::empty()),
        })
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Load image bytes + detect mime type.
    ///
    /// Accepts either a local file path or an `http(s)://` URL. URLs are
    /// fetched via reqwest; paths are read from disk.
    async fn load_image(&self, source: &str) -> Result<(Vec<u8>, String)> {
        if source.starts_with("http://") || source.starts_with("https://") {
            validate_url_str(source)?;
            let resp = self.client.get(source).send().await?;
            if !resp.status().is_success() {
                anyhow::bail!("HTTP {} fetching image", resp.status());
            }
            let mime = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
                .unwrap_or_else(|| "image/jpeg".to_string());
            let (bytes, truncated) = read_body_capped(resp, MAX_IMAGE_FETCH_BYTES).await?;
            if truncated {
                anyhow::bail!(
                    "image too large: exceeds max {} bytes",
                    MAX_IMAGE_FETCH_BYTES
                );
            }
            Ok((bytes, mime))
        } else {
            let resolved = self
                .sandbox
                .check(Path::new(source))
                .map_err(|e| anyhow::anyhow!("path rejected by sandbox: {}", e))?;
            let bytes = tokio::fs::read(&resolved).await?;
            let mime = guess_mime_from_path(source);
            Ok((bytes, mime))
        }
    }

    async fn analyze_anthropic(&self, image_b64: &str, mime: &str, question: &str) -> Result<String> {
        let body = json!({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": image_b64,
                        }
                    },
                    { "type": "text", "text": question }
                ]
            }]
        });
        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Anthropic vision API error ({}): {}",
                status,
                err_body
            ));
        }

        let parsed: Value = resp.json().await?;
        extract_anthropic_text(&parsed)
            .ok_or_else(|| anyhow::anyhow!("no text block in Anthropic response: {}", parsed))
    }

    async fn analyze_openai(&self, image_b64: &str, mime: &str, question: &str) -> Result<String> {
        let data_url = format!("data:{};base64,{}", mime, image_b64);
        // Reasoning-family models (o1/o3/o4-*, gpt-5*) REJECT `max_tokens`
        // and require `max_completion_tokens` at the Chat Completions
        // endpoint. See src/providers/openai.rs::is_reasoning_model.
        let token_field = if openai_is_reasoning_model(&self.model) {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        let body = json!({
            "model": self.model,
            token_field: 1024,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": question },
                    { "type": "image_url", "image_url": { "url": data_url } }
                ]
            }]
        });
        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OpenAI vision API error ({}): {}",
                status,
                err_body
            ));
        }

        let parsed: Value = resp.json().await?;
        extract_openai_text(&parsed)
            .ok_or_else(|| anyhow::anyhow!("no message content in OpenAI response: {}", parsed))
    }

    async fn analyze_gemini(&self, image_b64: &str, mime: &str, question: &str) -> Result<String> {
        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [
                    { "inlineData": { "mimeType": mime, "data": image_b64 } },
                    { "text": question }
                ]
            }],
            "generationConfig": { "maxOutputTokens": 1024 }
        });
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.model
        );
        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let err_body = resp.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Gemini vision API error ({}): {}",
                status,
                err_body
            ));
        }

        let parsed: Value = resp.json().await?;
        extract_gemini_text(&parsed)
            .ok_or_else(|| anyhow::anyhow!("no text part in Gemini response: {}", parsed))
    }
}

fn extract_anthropic_text(v: &Value) -> Option<String> {
    let blocks = v.get("content")?.as_array()?;
    let mut out = String::new();
    for block in blocks {
        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
            if let Some(t) = block.get("text").and_then(|x| x.as_str()) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

fn extract_openai_text(v: &Value) -> Option<String> {
    v.get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
        .map(String::from)
}

fn extract_gemini_text(v: &Value) -> Option<String> {
    let parts = v
        .get("candidates")?
        .as_array()?
        .first()?
        .get("content")?
        .get("parts")?
        .as_array()?;
    let mut out = String::new();
    for part in parts {
        // Skip reasoning ("thought") parts — only surface answer text.
        if part.get("thought").and_then(|t| t.as_bool()) == Some(true) {
            continue;
        }
        if let Some(t) = part.get("text").and_then(|x| x.as_str()) {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(t);
        }
    }
    (!out.is_empty()).then_some(out)
}

fn guess_mime_from_path(path: &str) -> String {
    let lower = path.to_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else {
        "image/jpeg"
    }
    .to_string()
}

#[async_trait]
impl Tool for VisionTool {
    fn name(&self) -> &str {
        "vision_describe"
    }

    fn description(&self) -> &str {
        "Analyze an image. Provide a file path or http(s) URL and an optional question. \
         Returns what the image shows — objects, text (OCR), diagram structure, colors, \
         layout, whatever the question asks about. Use for screenshots, photos, charts, \
         diagrams, UI mockups."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "Local file path (e.g. /tmp/screenshot.png) or http(s) URL of the image."
                },
                "question": {
                    "type": "string",
                    "description": "Optional — what to focus on. Defaults to a general description."
                }
            },
            "required": ["image"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolResult> {
        let image = match args.get("image").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: image".to_string()),
                });
            }
        };
        let question = args
            .get("question")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("Describe this image in detail.")
            .to_string();

        let (bytes, mime) = match self.load_image(&image).await {
            Ok(x) => x,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to load image: {}", e)),
                });
            }
        };
        let b64 = B64.encode(&bytes);

        let result = match self.backend {
            VisionBackend::Anthropic => self.analyze_anthropic(&b64, &mime, &question).await,
            VisionBackend::OpenAi => self.analyze_openai(&b64, &mime, &question).await,
            VisionBackend::Gemini => self.analyze_gemini(&b64, &mime, &question).await,
        };

        match result {
            Ok(text) => Ok(ToolResult {
                success: true,
                output: text,
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e.to_string()),
            }),
        }
    }

    fn is_read_only(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_from_anthropic() {
        assert_eq!(
            VisionBackend::from_provider_name("anthropic"),
            Some(VisionBackend::Anthropic)
        );
        assert_eq!(
            VisionBackend::from_provider_name("ANTHROPIC"),
            Some(VisionBackend::Anthropic)
        );
    }

    #[test]
    fn backend_from_openai() {
        assert_eq!(
            VisionBackend::from_provider_name("openai"),
            Some(VisionBackend::OpenAi)
        );
    }

    #[test]
    fn backend_from_gemini() {
        assert_eq!(
            VisionBackend::from_provider_name("gemini"),
            Some(VisionBackend::Gemini)
        );
        assert_eq!(
            VisionBackend::from_provider_name("google"),
            Some(VisionBackend::Gemini)
        );
        assert_eq!(
            VisionBackend::from_provider_name("Gemini"),
            Some(VisionBackend::Gemini)
        );
    }

    #[test]
    fn backend_from_unsupported_returns_none() {
        assert_eq!(VisionBackend::from_provider_name("ollama"), None);
        assert_eq!(VisionBackend::from_provider_name("kimi"), None);
        assert_eq!(VisionBackend::from_provider_name(""), None);
    }

    #[test]
    fn extract_gemini_text_joins_text_parts_and_skips_thoughts() {
        let v = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "thought": true, "text": "reasoning..." },
                        { "text": "A red bicycle" },
                        { "text": "leaning on a wall." }
                    ]
                }
            }]
        });
        assert_eq!(
            extract_gemini_text(&v).as_deref(),
            Some("A red bicycle\nleaning on a wall.")
        );
    }

    #[test]
    fn extract_gemini_text_none_when_empty() {
        let v = json!({ "candidates": [] });
        assert!(extract_gemini_text(&v).is_none());
    }

    #[test]
    fn from_provider_builds_gemini_with_default_model() {
        let tool = VisionTool::from_provider("gemini", "key".to_string(), None)
            .expect("gemini vision tool should build");
        assert_eq!(tool.backend, VisionBackend::Gemini);
        assert_eq!(tool.model, "gemini-2.5-flash");
    }

    #[test]
    fn from_provider_returns_none_on_unsupported() {
        assert!(VisionTool::from_provider("ollama", "key".to_string(), None).is_none());
    }

    #[test]
    fn from_provider_returns_none_on_empty_key() {
        assert!(VisionTool::from_provider("anthropic", String::new(), None).is_none());
    }

    #[test]
    fn from_provider_builds_for_anthropic() {
        let t = VisionTool::from_provider("anthropic", "sk-ant-test".to_string(), None);
        assert!(t.is_some());
    }

    #[test]
    fn mime_guess_handles_common_formats() {
        assert_eq!(guess_mime_from_path("foo.png"), "image/png");
        assert_eq!(guess_mime_from_path("foo.PNG"), "image/png");
        assert_eq!(guess_mime_from_path("foo.jpg"), "image/jpeg");
        assert_eq!(guess_mime_from_path("foo.jpeg"), "image/jpeg");
        assert_eq!(guess_mime_from_path("foo.gif"), "image/gif");
        assert_eq!(guess_mime_from_path("foo.webp"), "image/webp");
        assert_eq!(guess_mime_from_path("foo.bmp"), "image/jpeg");
    }

    #[test]
    fn extract_anthropic_text_single_block() {
        let v = json!({
            "content": [{"type": "text", "text": "a cat on a mat"}]
        });
        assert_eq!(extract_anthropic_text(&v), Some("a cat on a mat".to_string()));
    }

    #[test]
    fn extract_anthropic_text_multiple_blocks() {
        let v = json!({
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "text", "text": "line two"}
            ]
        });
        assert_eq!(
            extract_anthropic_text(&v),
            Some("line one\nline two".to_string())
        );
    }

    #[test]
    fn extract_anthropic_text_skips_non_text_blocks() {
        let v = json!({
            "content": [
                {"type": "tool_use", "id": "x", "name": "y", "input": {}},
                {"type": "text", "text": "real answer"}
            ]
        });
        assert_eq!(extract_anthropic_text(&v), Some("real answer".to_string()));
    }

    #[test]
    fn extract_openai_text_standard_response() {
        let v = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "looks like a pie chart"}
            }]
        });
        assert_eq!(
            extract_openai_text(&v),
            Some("looks like a pie chart".to_string())
        );
    }

    #[tokio::test]
    async fn execute_rejects_missing_image_param() {
        let t = VisionTool::from_provider("anthropic", "sk-test".to_string(), None).unwrap();
        let result = t.execute(json!({})).await.unwrap();
        assert!(!result.success);
        assert!(result.error.unwrap().contains("missing required parameter"));
    }

    #[test]
    fn tool_is_read_only() {
        let t = VisionTool::from_provider("anthropic", "sk-test".to_string(), None).unwrap();
        assert!(t.is_read_only());
    }

    #[test]
    fn openai_reasoning_model_helper_matches_reasoning_families() {
        // Sanity that the cross-module import works and classifies the
        // families the vision tool actually cares about.
        assert!(openai_is_reasoning_model("o1"));
        assert!(openai_is_reasoning_model("o3-mini"));
        assert!(openai_is_reasoning_model("gpt-5.1"));
        assert!(!openai_is_reasoning_model("gpt-4o"));
        assert!(!openai_is_reasoning_model("gpt-4o-mini"));
    }
}
