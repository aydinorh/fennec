//! Image attachment support for `/image` and `/paste`.
//!
//! When the user types `/image <path>` (or pastes a clipboard
//! image via `/paste`), the file is read into memory, validated,
//! and held on the [`Agent`] until the next user turn. The next
//! call to `turn` / `turn_streaming` attaches it to the outbound
//! user message, which Anthropic / OpenAI provider impls
//! serialise as image content blocks alongside the text.
//!
//! The attached set drains on every turn — images survive
//! exactly one round-trip, matching upstream's
//! `session.attached_images` semantics in
//! `tui_gateway/server.py:3361-3401`.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;

/// Extensions the upstream's `image.attach` accepts. The provider may
/// still reject the upload if the format isn't natively
/// supported (Anthropic doesn't take BMP/SVG/ICO, OpenAI is
/// similar) — but matching the upstream's acceptance list keeps the
/// `/image` UX identical and surfaces the failure as a provider
/// error rather than a Fennec-side rejection.
const ALLOWED_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "svg", "ico",
];

/// One attached image carried across turns. `base64_data` is
/// pre-encoded so providers don't have to redo the work on
/// retries; `path` is kept for display in the chat ("Attached
/// image: foo.png · 192 tokens").
#[derive(Debug, Clone)]
pub struct ImageAttachment {
    pub path: String,
    pub display_name: String,
    pub mime_type: String,
    pub base64_data: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Rough cross-provider token cost for the image, computed
    /// at attach time via `ceil(w/512) * ceil(h/512) * 85`
    /// (matching the upstream's `_image_meta` at cli.py:411-423).
    /// `None` when dimensions couldn't be read.
    pub token_estimate: Option<u32>,
}

impl ImageAttachment {
    /// Read `path` into memory, validate the extension, infer the
    /// MIME type, and compute width/height + token estimate. The
    /// returned attachment is ready to be appended to a
    /// [`crate::providers::traits::ChatMessage`] for the next turn.
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        if !ALLOWED_EXTENSIONS.contains(&ext.as_str()) {
            return Err(anyhow!(
                "unsupported image extension '.{}': expected one of {}",
                ext,
                ALLOWED_EXTENSIONS.join(", ")
            ));
        }
        let bytes =
            fs::read(path).with_context(|| format!("reading image: {}", path.display()))?;
        let display_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_string();
        let mime_type = mime_for_extension(&ext);
        let base64_data = B64.encode(&bytes);

        // Dimensions are best-effort: if `imagesize` doesn't
        // recognise the format (rare for SVG/ICO), the attach
        // still succeeds but token_estimate stays None.
        let dimensions = imagesize::blob_size(&bytes).ok();
        let (width, height) = match dimensions {
            Some(d) => (Some(d.width as u32), Some(d.height as u32)),
            None => (None, None),
        };
        let token_estimate = match (width, height) {
            (Some(w), Some(h)) => Some(estimate_image_tokens(w, h)),
            _ => None,
        };

        Ok(ImageAttachment {
            path: path.to_string_lossy().to_string(),
            display_name,
            mime_type,
            base64_data,
            width,
            height,
            token_estimate,
        })
    }
}

/// Map a lowercase extension to its MIME type. Falls back to
/// `application/octet-stream` for unknown types so the provider
/// can still receive the bytes (the API will likely reject it,
/// matching the upstream's behavior of letting the upstream complain).
pub fn mime_for_extension(ext: &str) -> String {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "tiff" | "tif" => "image/tiff",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    }
    .to_string()
}

/// Token cost estimate matching the upstream's `_image_meta`
/// (`cli.py:411-423`): `ceil(w/512) * ceil(h/512) * 85`. Cheap
/// to compute and stable across providers; actual cost may
/// differ but the user gets a useful order-of-magnitude check.
pub fn estimate_image_tokens(width: u32, height: u32) -> u32 {
    let w_tiles = width.div_ceil(512).max(1);
    let h_tiles = height.div_ceil(512).max(1);
    w_tiles * h_tiles * 85
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_matches_upstream_formula() {
        // 1024x1024: 2*2*85 = 340
        assert_eq!(estimate_image_tokens(1024, 1024), 340);
        // 512x512: 1*1*85 = 85
        assert_eq!(estimate_image_tokens(512, 512), 85);
        // 100x100 (clamped to 1*1): 85
        assert_eq!(estimate_image_tokens(100, 100), 85);
        // 1500x800: ceil(1500/512)=3, ceil(800/512)=2 → 3*2*85 = 510
        assert_eq!(estimate_image_tokens(1500, 800), 510);
    }

    #[test]
    fn mime_for_extension_covers_each_supported_format() {
        for ext in ALLOWED_EXTENSIONS {
            let mime = mime_for_extension(ext);
            assert!(mime.starts_with("image/"), "{ext} -> {mime}");
        }
    }

    #[test]
    fn from_path_rejects_non_image_extension() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, b"plain text").unwrap();
        let err = ImageAttachment::from_path(&path).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("unsupported image extension"), "{msg}");
    }

    #[test]
    fn from_path_loads_png_with_dimensions() {
        // Smallest valid PNG: 1x1 transparent pixel.
        const ONE_PX_PNG: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x62, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("dot.png");
        std::fs::write(&path, ONE_PX_PNG).unwrap();
        let att = ImageAttachment::from_path(&path).unwrap();
        assert_eq!(att.mime_type, "image/png");
        assert_eq!(att.width, Some(1));
        assert_eq!(att.height, Some(1));
        assert_eq!(att.token_estimate, Some(85));
        assert_eq!(att.display_name, "dot.png");
        assert!(!att.base64_data.is_empty());
    }
}
