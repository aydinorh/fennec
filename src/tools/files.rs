use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

use crate::security::PathSandbox;

use super::traits::{Tool, ToolResult};

const MAX_READ_BYTES: usize = 10 * 1024 * 1024; // 10 MB hard cap on reads
const MAX_WRITE_BYTES: usize = 10 * 1024 * 1024; // 10 MB hard cap on writes

/// Convert a sandbox check error into a user-visible error line. The
/// canonical-path detail is preserved so the LLM knows which path was
/// rejected; the specific denylist pattern is shown too so the user can
/// adjust `security.forbidden_paths` if needed.
fn sandbox_err(e: anyhow::Error) -> String {
    format!("path rejected by sandbox: {}", e)
}

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

/// Reads the contents of a file. Output is truncated if the file is too large.
pub struct ReadFileTool {
    sandbox: Arc<PathSandbox>,
}

impl ReadFileTool {
    pub fn new() -> Self {
        Self {
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at a given path."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file"
                }
            },
            "required": ["path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;

        let resolved = match self.sandbox.check(Path::new(path)) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(sandbox_err(e)),
                });
            }
        };

        match tokio::fs::read_to_string(&resolved).await {
            Ok(content) => {
                let total = content.len();
                let output = if total > MAX_READ_BYTES {
                    let cut = char_boundary_prefix(&content, MAX_READ_BYTES);
                    format!(
                        "{cut}\n\n... [truncated, file is {total} bytes]",
                        cut = cut,
                        total = total
                    )
                } else {
                    content
                };
                Ok(ToolResult {
                    success: true,
                    output,
                    error: None,
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("failed to read file: {e}")),
            }),
        }
    }
}

/// Take a prefix of `s` up to `max_bytes` bytes, falling back to the
/// nearest char boundary at or before that index so we never split a
/// multibyte character.
fn char_boundary_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut idx = max_bytes;
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    &s[..idx]
}

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

/// Writes content to a file, creating parent directories as needed.
pub struct WriteFileTool {
    sandbox: Arc<PathSandbox>,
}

impl WriteFileTool {
    pub fn new() -> Self {
        Self {
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file, creating parent directories if needed."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: content"))?;

        if content.len() > MAX_WRITE_BYTES {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "refusing to write {} bytes — exceeds cap of {}",
                    content.len(),
                    MAX_WRITE_BYTES
                )),
            });
        }

        let resolved = match self.sandbox.check(Path::new(path)) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(sandbox_err(e)),
                });
            }
        };

        // Create parent directories if they don't exist.
        if let Some(parent) = resolved.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }

        // Atomic temp+rename write: a crash or concurrent reader never
        // sees a half-written file, and a failed overwrite leaves the
        // original intact.
        match crate::security::fs::write_atomic(&resolved, content.as_bytes()) {
            Ok(()) => Ok(ToolResult {
                success: true,
                output: format!("wrote {} bytes to {path}", content.len()),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("failed to write file: {e}")),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// ListDirTool
// ---------------------------------------------------------------------------

/// Lists directory entries, appending `/` to directory names.
pub struct ListDirTool {
    sandbox: Arc<PathSandbox>,
}

impl ListDirTool {
    pub fn new() -> Self {
        Self {
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List the contents of a directory."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list"
                }
            },
            "required": ["path"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;

        let resolved = match self.sandbox.check(Path::new(path)) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(sandbox_err(e)),
                });
            }
        };

        let mut entries = Vec::new();
        let mut read_dir = match tokio::fs::read_dir(&resolved).await {
            Ok(rd) => rd,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to read directory: {e}")),
                });
            }
        };

        while let Some(entry) = read_dir.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            let metadata = entry.metadata().await?;
            if metadata.is_dir() {
                entries.push(format!("{name}/"));
            } else {
                entries.push(name);
            }
        }

        entries.sort();

        Ok(ToolResult {
            success: true,
            output: entries.join("\n"),
            error: None,
        })
    }
}

// ---------------------------------------------------------------------------
// EditFileTool
// ---------------------------------------------------------------------------

/// Replaces a unique occurrence of `old_text` with `new_text` in a file.
/// Fails if `old_text` is not found or appears more than once.
pub struct EditFileTool {
    sandbox: Arc<PathSandbox>,
}

impl EditFileTool {
    pub fn new() -> Self {
        Self {
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing a unique occurrence of old_text with new_text. \
         Fails if old_text is not found or appears more than once."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file"
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace"
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text"
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: path"))?;
        let old_text = args
            .get("old_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: old_text"))?;
        let new_text = args
            .get("new_text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: new_text"))?;

        let resolved = match self.sandbox.check(Path::new(path)) {
            Ok(p) => p,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(sandbox_err(e)),
                });
            }
        };

        let content = match tokio::fs::read_to_string(&resolved).await {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to read file: {e}")),
                });
            }
        };

        let count = content.matches(old_text).count();
        if count == 0 {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("old_text not found in file".to_string()),
            });
        }
        if count > 1 {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "old_text appears {count} times in file; must be unique"
                )),
            });
        }

        let new_content = content.replacen(old_text, new_text, 1);
        // Atomic swap — same rationale as write_file above.
        match crate::security::fs::write_atomic(&resolved, new_content.as_bytes()) {
            Ok(()) => Ok(ToolResult {
                success: true,
                output: format!("edited {path}: replaced 1 occurrence"),
                error: None,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("failed to write file: {e}")),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// GlobTool
// ---------------------------------------------------------------------------

/// Searches for files matching a glob pattern.
pub struct GlobTool {
    sandbox: Arc<PathSandbox>,
}

impl GlobTool {
    pub fn new() -> Self {
        Self {
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern (e.g. \"**/*.rs\"). Returns matching file paths."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '**/*.rs', 'src/**/*.ts')"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: current directory)"
                }
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: pattern".to_string()),
                });
            }
        };

        let base = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        // Validate the base path is not denied. (The pattern itself may
        // escape base via '..' or '/'; we re-check each match after the
        // glob runs.)
        if let Err(e) = self.sandbox.check(Path::new(base)) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(sandbox_err(e)),
            });
        }

        let full_pattern = if pattern.starts_with('/') || pattern.starts_with('.') {
            pattern.to_string()
        } else {
            format!("{base}/{pattern}")
        };

        let sandbox = Arc::clone(&self.sandbox);
        let full_pattern_clone = full_pattern.clone();
        let result = tokio::task::spawn_blocking(move || {
            let mut paths = Vec::new();
            match glob::glob(&full_pattern_clone) {
                Ok(entries) => {
                    for entry in entries {
                        match entry {
                            Ok(path) => {
                                // Re-check every matched path — glob can
                                // resolve to files outside the original
                                // base (via '..' or symlinks).
                                if sandbox.check(&path).is_err() {
                                    continue;
                                }
                                paths.push(path.display().to_string());
                            }
                            Err(e) => {
                                tracing::debug!("glob entry error: {e}");
                            }
                        }
                    }
                    Ok(paths)
                }
                Err(e) => Err(format!("invalid glob pattern: {e}")),
            }
        })
        .await?;

        match result {
            Ok(paths) => {
                let count = paths.len();
                let output = if paths.is_empty() {
                    "no files matched".to_string()
                } else {
                    let display: Vec<&str> = paths.iter().take(500).map(|s| s.as_str()).collect();
                    let mut out = display.join("\n");
                    if count > 500 {
                        out.push_str(&format!("\n\n... and {} more files", count - 500));
                    }
                    out
                };
                Ok(ToolResult {
                    success: true,
                    output,
                    error: None,
                })
            }
            Err(e) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(e),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// GrepTool
// ---------------------------------------------------------------------------

/// Recursively searches files for a regex pattern and returns matching lines.
pub struct GrepTool {
    sandbox: Arc<PathSandbox>,
}

impl GrepTool {
    pub fn new() -> Self {
        Self {
            sandbox: Arc::new(PathSandbox::empty()),
        }
    }

    pub fn with_sandbox(mut self, sandbox: Arc<PathSandbox>) -> Self {
        self.sandbox = sandbox;
        self
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Recursively search files for a regex pattern. Returns matching lines in file:line:content format."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current directory)"
                },
                "file_type": {
                    "type": "string",
                    "description": "File extension filter (e.g. 'rs', 'py', 'js')"
                }
            },
            "required": ["pattern"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let pattern_str = match args.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: pattern".to_string()),
                });
            }
        };

        let base_path = args
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".")
            .to_string();

        if let Err(e) = self.sandbox.check(Path::new(&base_path)) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(sandbox_err(e)),
            });
        }

        let file_type = args
            .get("file_type")
            .and_then(|v| v.as_str())
            .map(|s| format!(".{s}"));

        let regex = match regex::Regex::new(&pattern_str) {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("invalid regex: {e}")),
                });
            }
        };

        let sandbox = Arc::clone(&self.sandbox);
        let result = tokio::task::spawn_blocking(move || {
            let mut matches = Vec::new();
            let path = PathBuf::from(&base_path);
            grep_recursive(&path, &regex, file_type.as_deref(), &mut matches, 0, &sandbox);
            matches
        })
        .await?;

        if result.is_empty() {
            return Ok(ToolResult {
                success: true,
                output: "no matches found".to_string(),
                error: None,
            });
        }

        let count = result.len();
        let display: Vec<&str> = result.iter().take(200).map(|s| s.as_str()).collect();
        let mut output = display.join("\n");
        if count > 200 {
            output.push_str(&format!("\n\n... and {} more matches", count - 200));
        }

        Ok(ToolResult {
            success: true,
            output,
            error: None,
        })
    }
}

/// Recursively search a directory for regex matches. `depth` guards against
/// excessive recursion. Each file/dir is re-checked against the sandbox
/// (symlink traversal might otherwise lead into a denied area).
fn grep_recursive(
    path: &Path,
    regex: &regex::Regex,
    file_type: Option<&str>,
    matches: &mut Vec<String>,
    depth: usize,
    sandbox: &PathSandbox,
) {
    const MAX_DEPTH: usize = 20;
    const MAX_MATCHES: usize = 1000;

    if depth > MAX_DEPTH || matches.len() >= MAX_MATCHES {
        return;
    }

    // Re-validate each path — a symlink inside the starting tree could
    // resolve into a denied area.
    if sandbox.check(path).is_err() {
        return;
    }

    // Skip symlinks outright to avoid cycles and filesystem escapes that
    // the check() sandbox-canonicalize alone can't always detect (e.g.
    // a symlink whose target doesn't match any denylist substring).
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => return,
        Err(_) => return,
        _ => {}
    }

    if path.is_file() {
        if let Some(ext) = file_type {
            let file_ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| format!(".{e}"));
            if file_ext.as_deref() != Some(ext) {
                return;
            }
        }

        let metadata = match std::fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return,
        };
        if metadata.len() > 1_000_000 {
            return;
        }

        if let Ok(content) = std::fs::read_to_string(path) {
            let display_path = path.display().to_string();
            for (line_num, line) in content.lines().enumerate() {
                if matches.len() >= MAX_MATCHES {
                    break;
                }
                if regex.is_match(line) {
                    matches.push(format!("{}:{}:{}", display_path, line_num + 1, line));
                }
            }
        }
    } else if path.is_dir() {
        if depth > 0 {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') || name == "node_modules" || name == "target" {
                    return;
                }
            }
        }

        if let Ok(entries) = std::fs::read_dir(path) {
            let mut sorted: Vec<_> = entries.filter_map(|e| e.ok()).collect();
            sorted.sort_by_key(|e| e.file_name());
            for entry in sorted {
                grep_recursive(&entry.path(), regex, file_type, matches, depth + 1, sandbox);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn denylist() -> Arc<PathSandbox> {
        Arc::new(PathSandbox::new(vec![
            "/etc".to_string(),
            ".ssh".to_string(),
            ".fennec/.secret_key".to_string(),
        ]))
    }

    #[tokio::test]
    async fn read_rejects_denied_path() {
        let tool = ReadFileTool::new().with_sandbox(denylist());
        let r = tool
            .execute(json!({"path": "/etc/passwd"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("sandbox"));
    }

    #[tokio::test]
    async fn write_rejects_denied_path() {
        let tool = WriteFileTool::new().with_sandbox(denylist());
        let r = tool
            .execute(json!({
                "path": "/etc/rogue.conf",
                "content": "injected"
            }))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("sandbox"));
    }

    #[tokio::test]
    async fn list_dir_rejects_denied_path() {
        let tool = ListDirTool::new().with_sandbox(denylist());
        let r = tool.execute(json!({"path": "/etc"})).await.unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("sandbox"));
    }

    #[tokio::test]
    async fn edit_rejects_denied_path() {
        let tool = EditFileTool::new().with_sandbox(denylist());
        let r = tool
            .execute(json!({
                "path": "/etc/hosts",
                "old_text": "localhost",
                "new_text": "evil.example"
            }))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("sandbox"));
    }

    #[tokio::test]
    async fn read_accepts_benign_path() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("ok.txt");
        std::fs::write(&file, b"hello").unwrap();
        let tool = ReadFileTool::new().with_sandbox(denylist());
        let r = tool
            .execute(json!({"path": file.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(r.success, "err: {:?}", r.error);
        assert_eq!(r.output, "hello");
    }

    #[tokio::test]
    async fn write_rejects_oversize_content() {
        let tool = WriteFileTool::new();
        let big = "a".repeat(MAX_WRITE_BYTES + 1);
        let r = tool
            .execute(json!({
                "path": "/tmp/fennec-oversize-test",
                "content": big
            }))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("cap"));
    }

    #[test]
    fn char_boundary_prefix_handles_multibyte() {
        let s = "日本語";
        // "日" is 3 bytes; ask for prefix of 2 bytes — must not split.
        assert_eq!(char_boundary_prefix(s, 2), "");
        // 3 bytes exactly: just "日".
        assert_eq!(char_boundary_prefix(s, 3), "日");
        // 4 bytes: still just "日" (since 4 isn't a boundary either; next is at 6).
        assert_eq!(char_boundary_prefix(s, 4), "日");
        // 6 bytes: "日本".
        assert_eq!(char_boundary_prefix(s, 6), "日本");
    }

    #[tokio::test]
    async fn read_without_sandbox_accepts_anything() {
        // Back-compat: a default ReadFileTool (no sandbox wired) still works.
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("ok.txt");
        std::fs::write(&file, b"hello").unwrap();
        let tool = ReadFileTool::new();
        let r = tool
            .execute(json!({"path": file.to_str().unwrap()}))
            .await
            .unwrap();
        assert!(r.success);
    }
}
