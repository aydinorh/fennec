//! Image info tool — returns dimensions, format, and byte size of an image.
//!
//! Lightweight companion to `vision_describe` — the agent can check "is this
//! actually an image and how big is it?" before sending to the vision API
//! (which costs more). Uses the `imagesize` crate: reads just the file
//! header, no full decode.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::security::url_guard::{build_guarded_client, read_body_capped, validate_url_str_resolved};
use crate::security::PathSandbox;

use super::traits::{Tool, ToolResult};

/// Hard cap on bytes fetched for an image — large enough for any reasonable
/// photo (20 MB) but bounded so a hostile URL can't OOM us.
const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

pub struct ImageInfoTool {
    client: reqwest::Client,
    temp_dir: PathBuf,
    /// Applied to local-path sources only; URL fetches use the tool's own
    /// temp dir.
    sandbox: Arc<PathSandbox>,
}

impl ImageInfoTool {
    pub fn new(temp_dir: PathBuf) -> Self {
        let client = build_guarded_client(Duration::from_secs(30));
        Self {
            client,
            temp_dir,
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }

    async fn load_bytes(&self, source: &str) -> Result<(Vec<u8>, Option<PathBuf>)> {
        if source.starts_with("http://") || source.starts_with("https://") {
            validate_url_str_resolved(source).await?;
            tokio::fs::create_dir_all(&self.temp_dir).await?;
            let resp = self.client.get(source).send().await?;
            if !resp.status().is_success() {
                anyhow::bail!("HTTP {} fetching image", resp.status());
            }
            let (bytes, truncated) = read_body_capped(resp, MAX_IMAGE_BYTES).await?;
            if truncated {
                anyhow::bail!(
                    "image too large: exceeds max {} bytes",
                    MAX_IMAGE_BYTES
                );
            }
            let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S_%3f");
            let path = self.temp_dir.join(format!("imginfo_{}", ts));
            tokio::fs::write(&path, &bytes).await?;
            Ok((bytes, Some(path)))
        } else {
            let resolved = self
                .sandbox
                .check(Path::new(source))
                .map_err(|e| anyhow::anyhow!("path rejected by sandbox: {}", e))?;
            let bytes = tokio::fs::read(&resolved).await?;
            Ok((bytes, None))
        }
    }
}

#[async_trait]
impl Tool for ImageInfoTool {
    fn name(&self) -> &str {
        "image_info"
    }

    fn description(&self) -> &str {
        "Inspect an image file and return its format, dimensions, and byte \
         size. Much cheaper than vision_describe — use first when you need \
         to know whether a file is a valid image and how big it is. Accepts \
         local path or http(s) URL. Supports PNG, JPEG, GIF, WebP, BMP, \
         TIFF, and more."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Local path or http(s) URL to the image."
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

        let (bytes, temp_path) = match self.load_bytes(&source).await {
            Ok(x) => x,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to read image: {}", e)),
                });
            }
        };

        let size = match imagesize::blob_size(&bytes) {
            Ok(s) => s,
            Err(e) => {
                if let Some(p) = temp_path {
                    let _ = tokio::fs::remove_file(&p).await;
                }
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("not a recognized image: {}", e)),
                });
            }
        };

        let format = imagesize::image_type(&bytes)
            .map(|t| format!("{:?}", t))
            .unwrap_or_else(|_| "unknown".to_string());

        let byte_size = bytes.len();
        let display_source = temp_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| source.clone());

        let output = format!(
            "Source:     {}\nFormat:     {}\nDimensions: {} x {}\nSize:       {} bytes ({:.1} KB)",
            display_source,
            format,
            size.width,
            size.height,
            byte_size,
            byte_size as f64 / 1024.0,
        );

        if let Some(p) = temp_path {
            let _ = tokio::fs::remove_file(&p).await;
        }

        Ok(ToolResult {
            success: true,
            output,
            error: None,
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }
}

#[allow(dead_code)]
fn pretty_image_type(t: imagesize::ImageType) -> &'static str {
    match t {
        imagesize::ImageType::Jpeg => "JPEG",
        imagesize::ImageType::Png => "PNG",
        imagesize::ImageType::Gif => "GIF",
        imagesize::ImageType::Webp => "WebP",
        imagesize::ImageType::Bmp => "BMP",
        imagesize::ImageType::Tiff => "TIFF",
        _ => "other",
    }
}

// Helper for tests: synthesize a minimal 1x1 PNG in memory.
#[cfg(test)]
fn make_tiny_png() -> Vec<u8> {
    // Minimal 1x1 PNG — public header bytes only.
    vec![
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // signature
        0x00, 0x00, 0x00, 0x0D, // IHDR length
        0x49, 0x48, 0x44, 0x52, // "IHDR"
        0x00, 0x00, 0x00, 0x01, // width = 1
        0x00, 0x00, 0x00, 0x01, // height = 1
        0x08, 0x06, 0x00, 0x00, 0x00, // bit depth, color type, etc.
        0x1F, 0x15, 0xC4, 0x89, // CRC (not validated by imagesize)
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn execute_rejects_missing_source() {
        let t = ImageInfoTool::new(std::env::temp_dir());
        let r = t.execute(json!({})).await.unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("source"));
    }

    #[tokio::test]
    async fn execute_rejects_nonexistent_file() {
        let t = ImageInfoTool::new(std::env::temp_dir());
        let r = t
            .execute(json!({"source": "/nonexistent/img.png"}))
            .await
            .unwrap();
        assert!(!r.success);
    }

    #[tokio::test]
    async fn execute_reads_minimal_png() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tiny.png");
        std::fs::write(&path, make_tiny_png()).unwrap();
        let t = ImageInfoTool::new(tmp.path().to_path_buf());
        let r = t
            .execute(json!({"source": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(r.success, "error: {:?}", r.error);
        assert!(r.output.contains("1 x 1"));
        assert!(r.output.contains("Png") || r.output.contains("PNG"));
    }

    #[tokio::test]
    async fn execute_rejects_non_image_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("fake.png");
        std::fs::write(&path, b"not an image at all").unwrap();
        let t = ImageInfoTool::new(tmp.path().to_path_buf());
        let r = t
            .execute(json!({"source": path.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("not a recognized"));
    }

    #[test]
    fn tool_is_read_only() {
        let t = ImageInfoTool::new(std::env::temp_dir());
        assert!(t.is_read_only());
    }

    #[test]
    fn pretty_type_mapping() {
        assert_eq!(pretty_image_type(imagesize::ImageType::Jpeg), "JPEG");
        assert_eq!(pretty_image_type(imagesize::ImageType::Png), "PNG");
    }
}
