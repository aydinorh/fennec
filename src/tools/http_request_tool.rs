//! Generic HTTP request tool — the agent's Swiss Army knife for REST APIs.
//!
//! Any API that doesn't have a dedicated Fennec tool can still be called via
//! this primitive. Supports GET/POST/PUT/PATCH/DELETE, custom headers,
//! JSON or raw bodies. Returns status + headers + body.
//!
//! URLs are validated through [`crate::security::url_guard`] before every
//! request (and again on every redirect hop) so the agent can't be tricked
//! into hitting loopback, RFC1918, link-local, or cloud-metadata endpoints
//! via prompt injection. Set `FENNEC_ALLOW_PRIVATE_URLS=1` to opt out if the
//! agent legitimately needs to reach localhost.
//!
//! The response body is streamed with a hard byte cap, so a malicious server
//! returning 10 GB can't OOM the process the way `.bytes().await` would.

use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::security::url_guard::{build_guarded_client, read_body_capped, validate_url_str_resolved};

use super::traits::{Tool, ToolResult};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_BODY_BYTES: usize = 1_048_576; // 1 MB truncation

pub struct HttpRequestTool {
    client: reqwest::Client,
    max_body_bytes: usize,
}

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpRequestTool {
    pub fn new() -> Self {
        let client = build_guarded_client(Duration::from_secs(DEFAULT_TIMEOUT_SECS));
        Self {
            client,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        }
    }

    pub fn with_max_body(mut self, bytes: usize) -> Self {
        self.max_body_bytes = bytes;
        self
    }

    fn parse_method(s: &str) -> Option<reqwest::Method> {
        match s.to_uppercase().as_str() {
            "GET" => Some(reqwest::Method::GET),
            "POST" => Some(reqwest::Method::POST),
            "PUT" => Some(reqwest::Method::PUT),
            "PATCH" => Some(reqwest::Method::PATCH),
            "DELETE" => Some(reqwest::Method::DELETE),
            "HEAD" => Some(reqwest::Method::HEAD),
            _ => None,
        }
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn description(&self) -> &str {
        "Send an arbitrary HTTP request. Supports GET/POST/PUT/PATCH/DELETE/HEAD, \
         custom headers, JSON or raw body. Returns status, response headers, \
         and body (truncated at 1 MB). Use for any REST API that doesn't have \
         a dedicated tool."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                    "description": "HTTP method."
                },
                "url": {
                    "type": "string",
                    "description": "Full URL including scheme."
                },
                "headers": {
                    "type": "object",
                    "description": "Optional request headers as key/value pairs.",
                    "additionalProperties": { "type": "string" }
                },
                "json": {
                    "description": "Optional JSON body. Sets Content-Type to application/json automatically."
                },
                "body": {
                    "type": "string",
                    "description": "Optional raw string body. Ignored if 'json' is set."
                },
                "query": {
                    "type": "object",
                    "description": "Optional query parameters.",
                    "additionalProperties": { "type": "string" }
                }
            },
            "required": ["method", "url"]
        })
    }

    async fn execute(&self, args: Value) -> Result<ToolResult> {
        let method_str = match args.get("method").and_then(|v| v.as_str()) {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: method".to_string()),
                });
            }
        };
        let method = match Self::parse_method(&method_str) {
            Some(m) => m,
            None => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("unsupported HTTP method: {}", method_str)),
                });
            }
        };
        let url = match args.get("url").and_then(|v| v.as_str()) {
            Some(u) if !u.is_empty() => u.to_string(),
            _ => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some("missing required parameter: url".to_string()),
                });
            }
        };

        // SSRF guard: reject non-http(s), loopback, private, link-local,
        // cloud-metadata hosts before the request goes out.
        if let Err(e) = validate_url_str_resolved(&url).await {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("URL rejected: {}", e)),
            });
        }

        let mut builder = self.client.request(method, &url);

        if let Some(headers) = args.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in headers {
                if let Some(val) = v.as_str() {
                    builder = builder.header(k, val);
                }
            }
        }

        if let Some(query) = args.get("query").and_then(|v| v.as_object()) {
            let pairs: Vec<(String, String)> = query
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect();
            builder = builder.query(&pairs);
        }

        if let Some(json_body) = args.get("json") {
            if !json_body.is_null() {
                builder = builder.json(json_body);
            }
        } else if let Some(body) = args.get("body").and_then(|v| v.as_str()) {
            builder = builder.body(body.to_string());
        }

        let resp = match builder.send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("request failed: {}", e)),
                });
            }
        };

        let status = resp.status();
        let header_pairs: Vec<(String, String)> = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();

        // Stream the body with a hard byte cap — safer than resp.bytes()
        // which buffers the entire response before any size check.
        let (bytes, truncated) = match read_body_capped(resp, self.max_body_bytes).await {
            Ok(x) => x,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("failed to read response body: {}", e)),
                });
            }
        };

        let body_str = String::from_utf8_lossy(&bytes).to_string();

        let mut output = format!("HTTP {} {}\n\n", status.as_u16(), status.canonical_reason().unwrap_or(""));
        output.push_str("--- headers ---\n");
        for (k, v) in &header_pairs {
            output.push_str(&format!("{}: {}\n", k, v));
        }
        output.push_str("\n--- body ---\n");
        output.push_str(&body_str);
        if truncated {
            output.push_str(&format!(
                "\n\n[body truncated — cap {} bytes reached]",
                self.max_body_bytes
            ));
        }

        Ok(ToolResult {
            success: status.is_success(),
            output,
            error: None,
        })
    }

    fn is_read_only(&self) -> bool {
        // GET/HEAD are, but the tool as a whole isn't.
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_method_case_insensitive() {
        assert_eq!(
            HttpRequestTool::parse_method("get"),
            Some(reqwest::Method::GET)
        );
        assert_eq!(
            HttpRequestTool::parse_method("POST"),
            Some(reqwest::Method::POST)
        );
        assert_eq!(
            HttpRequestTool::parse_method("PaTcH"),
            Some(reqwest::Method::PATCH)
        );
    }

    #[test]
    fn parse_method_rejects_unknown() {
        assert!(HttpRequestTool::parse_method("FROBNICATE").is_none());
        assert!(HttpRequestTool::parse_method("").is_none());
    }

    #[test]
    fn parse_method_all_supported() {
        for m in ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"] {
            assert!(HttpRequestTool::parse_method(m).is_some(), "{} failed", m);
        }
    }

    #[tokio::test]
    async fn execute_rejects_missing_method() {
        let t = HttpRequestTool::new();
        let r = t.execute(json!({"url": "https://x.com"})).await.unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("method"));
    }

    #[tokio::test]
    async fn execute_rejects_missing_url() {
        let t = HttpRequestTool::new();
        let r = t.execute(json!({"method": "GET"})).await.unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("url"));
    }

    #[tokio::test]
    async fn execute_rejects_unsupported_method() {
        let t = HttpRequestTool::new();
        let r = t
            .execute(json!({"method": "OPTIONS", "url": "https://x.com"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("unsupported"));
    }

    #[tokio::test]
    async fn execute_rejects_loopback_url() {
        let t = HttpRequestTool::new();
        let r = t
            .execute(json!({"method": "GET", "url": "http://127.0.0.1:8080/admin"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("URL rejected"));
    }

    #[tokio::test]
    async fn execute_rejects_imds_url() {
        let t = HttpRequestTool::new();
        let r = t
            .execute(json!({"method": "GET", "url": "http://169.254.169.254/latest/"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("URL rejected"));
    }

    #[tokio::test]
    async fn execute_rejects_file_scheme() {
        let t = HttpRequestTool::new();
        let r = t
            .execute(json!({"method": "GET", "url": "file:///etc/passwd"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("URL rejected"));
    }

    #[test]
    fn with_max_body_overrides_default() {
        let t = HttpRequestTool::new().with_max_body(512);
        assert_eq!(t.max_body_bytes, 512);
    }

    #[test]
    fn default_max_body_is_one_mb() {
        let t = HttpRequestTool::new();
        assert_eq!(t.max_body_bytes, DEFAULT_MAX_BODY_BYTES);
    }
}
