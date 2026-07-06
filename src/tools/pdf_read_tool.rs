//! PDF read tool — extracts text from a PDF file or URL.
//!
//! Uses `pdf-extract` (pure-Rust, no external binaries). Accepts a local
//! path or http(s) URL; fetched PDFs are written to a temp file so
//! `pdf-extract` can mmap them.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::security::url_guard::{build_guarded_client, read_body_capped, validate_url_str_resolved};
use crate::security::PathSandbox;

use super::traits::{Tool, ToolResult};

pub struct PdfReadTool {
    client: reqwest::Client,
    temp_dir: PathBuf,
    /// Max PDF size we'll accept before refusing (bytes).
    max_size_bytes: usize,
    /// Applied to local-path sources. URL-fetched PDFs bypass this (they
    /// land in the tool-owned temp_dir).
    sandbox: Arc<PathSandbox>,
}

impl PdfReadTool {
    pub fn new(temp_dir: PathBuf) -> Self {
        let client = build_guarded_client(std::time::Duration::from_secs(60));
        Self {
            client,
            temp_dir,
            max_size_bytes: 50 * 1024 * 1024, // 50 MB
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_max_size(mut self, bytes: usize) -> Self {
        self.max_size_bytes = bytes;
        self
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Resolve the source into a local filesystem path. URLs are downloaded
    /// to a temp file and the new path returned. Existing local paths are
    /// passed through unchanged.
    ///
    /// Returns (path, is_temp) — caller deletes the file when is_temp=true.
    async fn resolve_to_local(&self, source: &str) -> Result<(PathBuf, bool)> {
        if source.starts_with("http://") || source.starts_with("https://") {
            validate_url_str_resolved(source).await?;
            tokio::fs::create_dir_all(&self.temp_dir).await?;
            let resp = self.client.get(source).send().await?;
            if !resp.status().is_success() {
                anyhow::bail!("HTTP {} fetching PDF", resp.status());
            }
            // Stream with a cap so a 10 GB server response can't OOM the
            // process before any size check runs.
            let (bytes, truncated) = read_body_capped(resp, self.max_size_bytes).await?;
            if truncated {
                anyhow::bail!(
                    "PDF too large: exceeds max {} bytes",
                    self.max_size_bytes
                );
            }
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f");
            let path = self.temp_dir.join(format!("download_{}.pdf", ts));
            tokio::fs::write(&path, &bytes).await?;
            Ok((path, true))
        } else {
            // Local path: validate against the filesystem sandbox before
            // opening, so a prompt-injected source like ~/.ssh/id_rsa is
            // rejected rather than fed into pdf-extract.
            let resolved = self
                .sandbox
                .check(Path::new(source))
                .map_err(|e| anyhow::anyhow!("path rejected by sandbox: {}", e))?;
            if !resolved.exists() {
                anyhow::bail!("file does not exist: {}", source);
            }
            let meta = tokio::fs::metadata(&resolved).await?;
            if meta.len() as usize > self.max_size_bytes {
                anyhow::bail!(
                    "PDF too large: {} bytes (max {})",
                    meta.len(),
                    self.max_size_bytes
                );
            }
            Ok((resolved, false))
        }
    }
}

#[async_trait]
impl Tool for PdfReadTool {
    fn name(&self) -> &str {
        "pdf_read"
    }

    fn description(&self) -> &str {
        "Extract text from a PDF (local path or http(s) URL). Returns the \
         plain-text contents. Use for reading papers, invoices, contracts, \
         manuals. Does not OCR scanned images — those return empty text. \
         Max 50 MB per PDF."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Local file path or http(s) URL to the PDF."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Optional truncation after N chars. Default 100000."
                }
            },
            "required": ["source"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolResult> {
        let source = match args.get("source").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: source".to_string()),
                });
            }
        };
        let max_chars = args
            .get("max_chars")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(100_000);

        let (path, is_temp) = match self.resolve_to_local(&source).await {
            Ok(x) => x,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to load PDF: {}", e)),
                });
            }
        };

        // pdf-extract is synchronous and blocking — run on a blocking task.
        let path_for_task = path.clone();
        let text_result = tokio::task::spawn_blocking(move || {
            pdf_extract::extract_text(&path_for_task)
        })
        .await;

        if is_temp {
            let _ = tokio::fs::remove_file(&path).await;
        }

        let text = match text_result {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("PDF extraction failed: {}", e)),
                });
            }
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("extraction task failed: {}", e)),
                });
            }
        };

        let output = truncate_at_char_boundary(&text, max_chars);
        let note = if text.chars().count() > max_chars {
            format!(
                "\n\n[... truncated; {} chars total, showing first {}]",
                text.chars().count(),
                max_chars
            )
        } else {
            String::new()
        };

        Ok(ToolResult {
            success: true,
            output: format!("{}{}", output, note),
            error: None,
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }
}

fn truncate_at_char_boundary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_preserves_short_strings() {
        assert_eq!(truncate_at_char_boundary("hello", 100), "hello");
    }

    #[test]
    fn truncate_cuts_long_strings() {
        let s = "a".repeat(200);
        let out = truncate_at_char_boundary(&s, 50);
        assert_eq!(out.chars().count(), 50);
    }

    #[test]
    fn truncate_handles_multibyte_chars() {
        let s = "日本語".repeat(100);
        let out = truncate_at_char_boundary(&s, 10);
        assert_eq!(out.chars().count(), 10);
    }

    #[tokio::test]
    async fn execute_rejects_missing_source() {
        let t = PdfReadTool::new(std::env::temp_dir());
        let r = t.execute(json!({})).await.unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("source"));
    }

    #[tokio::test]
    async fn execute_rejects_nonexistent_file() {
        let t = PdfReadTool::new(std::env::temp_dir());
        let r = t
            .execute(json!({"source": "/nonexistent/file.pdf"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("does not exist"));
    }

    #[tokio::test]
    async fn resolve_to_local_rejects_oversize_local() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("big.pdf");
        std::fs::write(&path, vec![0u8; 1000]).unwrap();
        let t = PdfReadTool::new(std::env::temp_dir()).with_max_size(500);
        let r = t.resolve_to_local(path.to_str().unwrap()).await;
        assert!(r.is_err());
        assert!(r.err().unwrap().to_string().contains("too large"));
    }

    #[test]
    fn tool_is_read_only() {
        let t = PdfReadTool::new(std::env::temp_dir());
        assert!(t.is_read_only());
    }

    #[test]
    fn with_max_size_overrides_default() {
        let t = PdfReadTool::new(std::env::temp_dir()).with_max_size(1024);
        assert_eq!(t.max_size_bytes, 1024);
    }
}
