//! Host functions exposed to WASM plugins.
//!
//! Each function is gated by the same security primitives the agent
//! itself uses:
//!
//! - HTTP goes through [`url_guard::validate_url_str`].
//! - File I/O goes through [`PathSandbox`].
//! - Memory writes are scoped to a per-plugin namespace.
//!
//! The plugin can never do anything the agent itself couldn't —
//! plugins are a *capability extension* mechanism, not a privilege
//! escalation.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::runtime::Handle;

use crate::bus::{MessageBus, OutboundMessage};
use crate::memory::traits::{Memory, MemoryCategory, MemoryEntry};
use crate::security::path_sandbox::PathSandbox;
use crate::security::url_guard;

/// Per-plugin host state. Held inside each plugin's `Store`.
///
/// Constructed by the loader once per plugin. Fields are the
/// **capability surface** — anything a plugin can do is brokered
/// through one of these handles.
pub struct PluginHostState {
    pub plugin_name: String,
    pub path_sandbox: Arc<PathSandbox>,
    pub memory: Arc<dyn Memory>,
    pub http_client: reqwest::Client,
    pub rt_handle: Handle,
    /// Plugin-scoped settings. Populated by the loader from
    /// `[plugins.settings.<plugin-name>]` in `config.toml`. Plugins
    /// see only their own settings.
    pub settings: HashMap<String, String>,
    /// Bus handle for outbound channel messages. `None` when
    /// running outside gateway mode (CLI / agent mode), in which
    /// case `channel-send` returns an error to the plugin.
    /// `MessageBus::clone()` is cheap (two `mpsc::Sender` Arc
    /// increments) so we hold the bus by value rather than via
    /// an extra `Arc` wrapper.
    pub bus: Option<MessageBus>,
}

impl PluginHostState {
    pub(super) fn memory_category(&self) -> MemoryCategory {
        MemoryCategory::Custom(format!("plugin:{}", self.plugin_name))
    }

    pub(super) fn memory_category_str(&self) -> String {
        format!("plugin:{}", self.plugin_name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

pub fn host_log(state: &PluginHostState, level: LogLevel, message: &str) {
    // tracing macros require a `'static` target literal, so we put
    // the plugin name in a structured field instead. Filter via
    // `RUST_LOG=fennec::plugins::wasm::host=info` to see only plugin
    // logs, then narrow to a specific plugin by grepping the
    // `plugin=...` field.
    let plugin = state.plugin_name.as_str();
    match level {
        LogLevel::Trace => tracing::trace!(plugin = %plugin, "{}", message),
        LogLevel::Debug => tracing::debug!(plugin = %plugin, "{}", message),
        LogLevel::Info => tracing::info!(plugin = %plugin, "{}", message),
        LogLevel::Warn => tracing::warn!(plugin = %plugin, "{}", message),
        LogLevel::Error => tracing::error!(plugin = %plugin, "{}", message),
    }
}

pub fn host_now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub struct WasmHttpRequest {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
}

pub struct WasmHttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

const MAX_HTTP_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

pub fn host_http_request(
    state: &PluginHostState,
    req: WasmHttpRequest,
) -> Result<WasmHttpResponse, String> {
    url_guard::validate_url_str(&req.url)
        .map_err(|e| format!("url rejected by sandbox: {e}"))?;
    // DNS-resolution check (a domain pointing at a private IP) runs
    // inside the blocking section below where we have the runtime.

    let method = match req.method.to_ascii_uppercase().as_str() {
        "GET" => reqwest::Method::GET,
        "POST" => reqwest::Method::POST,
        "PUT" => reqwest::Method::PUT,
        "PATCH" => reqwest::Method::PATCH,
        "DELETE" => reqwest::Method::DELETE,
        "HEAD" => reqwest::Method::HEAD,
        other => return Err(format!("unsupported method: {other}")),
    };

    let mut builder = state.http_client.request(method, &req.url);
    for (name, value) in &req.headers {
        builder = builder.header(name, value);
    }
    if let Some(body) = req.body {
        builder = builder.body(body);
    }

    // block_in_place tells the tokio multi-threaded runtime that
    // this worker is about to block on a future; another worker
    // takes over the async task pool while we drive the future to
    // completion. Without it, `Handle::block_on` panics with
    // "Cannot start a runtime from within a runtime" when called
    // from a context that's already on an async worker thread.
    tokio::task::block_in_place(|| state.rt_handle.block_on(async move {
        url_guard::validate_url_str_resolved(&req.url)
            .await
            .map_err(|e| format!("url rejected by sandbox: {e}"))?;
        let resp = builder
            .send()
            .await
            .map_err(|e| format!("http request failed: {e}"))?;
        let status = resp.status().as_u16();
        let headers: Vec<(String, String)> = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|s| (k.as_str().to_string(), s.to_string()))
            })
            .collect();
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| format!("http body read failed: {e}"))?;
        if bytes.len() > MAX_HTTP_RESPONSE_BYTES {
            return Err(format!(
                "http response too large: {} bytes (limit {})",
                bytes.len(),
                MAX_HTTP_RESPONSE_BYTES
            ));
        }
        Ok(WasmHttpResponse {
            status,
            headers,
            body: bytes.to_vec(),
        })
    }))
}

const MAX_READ_FILE_BYTES: u64 = 50 * 1024 * 1024;

pub fn host_read_file(state: &PluginHostState, path: &str) -> Result<Vec<u8>, String> {
    let resolved = state
        .path_sandbox
        .check(std::path::Path::new(path))
        .map_err(|e| format!("path rejected by sandbox: {e}"))?;

    let meta = std::fs::metadata(&resolved).map_err(|e| format!("stat failed: {e}"))?;
    if meta.len() > MAX_READ_FILE_BYTES {
        return Err(format!(
            "file too large: {} bytes (limit {})",
            meta.len(),
            MAX_READ_FILE_BYTES
        ));
    }
    std::fs::read(&resolved).map_err(|e| format!("read failed: {e}"))
}

pub fn host_write_file(
    state: &PluginHostState,
    path: &str,
    contents: &[u8],
) -> Result<(), String> {
    let resolved = state
        .path_sandbox
        .check(std::path::Path::new(path))
        .map_err(|e| format!("path rejected by sandbox: {e}"))?;
    if let Some(parent) = resolved.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir parent failed: {e}"))?;
    }
    std::fs::write(&resolved, contents).map_err(|e| format!("write failed: {e}"))
}

pub struct WasmMemoryEntry {
    pub key: String,
    pub content: String,
    pub category: String,
    pub created_at: String,
}

pub fn host_memory_recall(
    state: &PluginHostState,
    query: &str,
    limit: u32,
) -> Result<Vec<WasmMemoryEntry>, String> {
    let limit = limit.clamp(1, 50) as usize;
    let memory = Arc::clone(&state.memory);
    let q = query.to_string();
    // block_in_place tells the tokio multi-threaded runtime that
    // this worker is about to block on a future; another worker
    // takes over the async task pool while we drive the future to
    // completion. Without it, `Handle::block_on` panics with
    // "Cannot start a runtime from within a runtime" when called
    // from a context that's already on an async worker thread.
    tokio::task::block_in_place(|| state.rt_handle.block_on(async move {
        memory
            .recall(&q, limit)
            .await
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|e| WasmMemoryEntry {
                        key: e.key,
                        content: e.content,
                        category: format_category(&e.category),
                        created_at: e.created_at,
                    })
                    .collect()
            })
            .map_err(|e| format!("memory recall failed: {e}"))
    }))
}

pub fn host_memory_store(
    state: &PluginHostState,
    key: &str,
    content: &str,
) -> Result<(), String> {
    let category = state.memory_category();
    let now = chrono::Utc::now().to_rfc3339();
    let entry = MemoryEntry {
        key: key.to_string(),
        content: content.to_string(),
        category,
        created_at: now.clone(),
        updated_at: now,
        ..MemoryEntry::default()
    };
    let memory = Arc::clone(&state.memory);
    // block_in_place tells the tokio multi-threaded runtime that
    // this worker is about to block on a future; another worker
    // takes over the async task pool while we drive the future to
    // completion. Without it, `Handle::block_on` panics with
    // "Cannot start a runtime from within a runtime" when called
    // from a context that's already on an async worker thread.
    tokio::task::block_in_place(|| state.rt_handle.block_on(async move {
        memory
            .store(entry)
            .await
            .map_err(|e| format!("memory store failed: {e}"))
    }))
}

pub(super) fn format_category(c: &MemoryCategory) -> String {
    match c {
        MemoryCategory::Core => "core".into(),
        MemoryCategory::Daily => "daily".into(),
        MemoryCategory::Conversation => "conversation".into(),
        MemoryCategory::Custom(s) => s.clone(),
    }
}

// ---------------------------------------------------------------------------
// Memory: get + forget
// ---------------------------------------------------------------------------

/// Direct lookup of a single memory entry by key. Returns `None` if
/// no entry exists. Reads are not scoped — plugins can see any
/// memory entry, the same as the agent's `memory_recall` tool.
pub fn host_memory_get(
    state: &PluginHostState,
    key: &str,
) -> Result<Option<WasmMemoryEntry>, String> {
    let memory = Arc::clone(&state.memory);
    let key = key.to_string();
    // block_in_place tells the tokio multi-threaded runtime that
    // this worker is about to block on a future; another worker
    // takes over the async task pool while we drive the future to
    // completion. Without it, `Handle::block_on` panics with
    // "Cannot start a runtime from within a runtime" when called
    // from a context that's already on an async worker thread.
    tokio::task::block_in_place(|| state.rt_handle.block_on(async move {
        memory
            .get(&key)
            .await
            .map(|maybe| {
                maybe.map(|e| WasmMemoryEntry {
                    key: e.key,
                    content: e.content,
                    category: format_category(&e.category),
                    created_at: e.created_at,
                })
            })
            .map_err(|e| format!("memory get failed: {e}"))
    }))
}

/// Delete a memory entry. Scoped to this plugin's namespace —
/// `forget(key)` only succeeds if the entry's category is
/// `plugin:<this-plugin-name>`. Other-category entries are a silent
/// no-op (returns `Ok(false)`).
///
/// The scoping check is enforced by the host: we read the entry
/// first, verify its category, and only then call the underlying
/// `Memory::forget`. This means a plugin can't delete the agent's
/// Core memory or another plugin's entries even if it knows the key.
pub fn host_memory_forget(
    state: &PluginHostState,
    key: &str,
) -> Result<bool, String> {
    let memory = Arc::clone(&state.memory);
    let expected_category = state.memory_category_str();
    let key = key.to_string();
    // block_in_place tells the tokio multi-threaded runtime that
    // this worker is about to block on a future; another worker
    // takes over the async task pool while we drive the future to
    // completion. Without it, `Handle::block_on` panics with
    // "Cannot start a runtime from within a runtime" when called
    // from a context that's already on an async worker thread.
    tokio::task::block_in_place(|| state.rt_handle.block_on(async move {
        // Read the entry to verify category before deleting.
        let entry = match memory.get(&key).await {
            Ok(Some(e)) => e,
            Ok(None) => return Ok(false),  // already gone, idempotent
            Err(e) => return Err(format!("memory get-before-forget failed: {e}")),
        };
        let entry_category = format_category(&entry.category);
        if entry_category != expected_category {
            // Wrong namespace — refuse silently. Plugin sees a
            // "didn't exist" result, same as if the key was already
            // gone, so a plugin can't probe other categories'
            // contents through this call.
            tracing::warn!(
                plugin = %expected_category,
                attempted = %entry_category,
                key = %key,
                "Plugin tried to forget an entry outside its namespace; refusing"
            );
            return Ok(false);
        }
        memory
            .forget(&key)
            .await
            .map_err(|e| format!("memory forget failed: {e}"))
    }))
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Read a plugin-scoped string setting. Returns `None` if the key
/// isn't present in this plugin's `[plugins.settings.<name>]`
/// section. Plugins cannot see each other's settings or any other
/// Fennec config — the loader populates `state.settings` with only
/// this plugin's slice of the config map.
pub fn host_config_get_string(state: &PluginHostState, key: &str) -> Option<String> {
    state.settings.get(key).cloned()
}

// ---------------------------------------------------------------------------
// Channels
// ---------------------------------------------------------------------------

/// Publish an outbound message on the bus. Only available when
/// running in gateway mode (the bus is plumbed into the host state).
/// In CLI / agent mode the bus is `None` and this returns an error
/// the plugin can handle.
pub fn host_channel_send(
    state: &PluginHostState,
    channel: &str,
    chat_id: &str,
    content: &str,
) -> Result<(), String> {
    let bus = match state.bus.clone() {
        Some(b) => b,
        None => {
            return Err(
                "channel send is only available in gateway mode (no message bus is wired)"
                    .to_string(),
            );
        }
    };
    let outbound = OutboundMessage {
        content: content.to_string(),
        channel: channel.to_string(),
        chat_id: chat_id.to_string(),
        reply_to: None,
        metadata: std::collections::HashMap::new(),
        attachments: Vec::new(),
    };
    // block_in_place tells the tokio multi-threaded runtime that
    // this worker is about to block on a future; another worker
    // takes over the async task pool while we drive the future to
    // completion. Without it, `Handle::block_on` panics with
    // "Cannot start a runtime from within a runtime" when called
    // from a context that's already on an async worker thread.
    tokio::task::block_in_place(|| state.rt_handle.block_on(async move {
        bus.publish_outbound(outbound)
            .await
            .map_err(|e| format!("channel send failed: {e}"))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use crate::memory::traits::Memory;

    /// Minimal in-memory mock for the small subset of `Memory` the
    /// host functions touch (`get`, `store`, `forget`). Concurrency
    /// guard is `parking_lot::Mutex` because the rest of Fennec
    /// already standardised on it for non-async locks.
    struct MockMemory {
        entries: parking_lot::Mutex<Vec<MemoryEntry>>,
    }

    impl MockMemory {
        fn new() -> Self {
            Self {
                entries: parking_lot::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Memory for MockMemory {
        fn name(&self) -> &str {
            "mock"
        }
        async fn store(&self, entry: MemoryEntry) -> anyhow::Result<()> {
            self.entries.lock().push(entry);
            Ok(())
        }
        async fn recall(
            &self,
            _query: &str,
            _limit: usize,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(self.entries.lock().clone())
        }
        async fn get(&self, key: &str) -> anyhow::Result<Option<MemoryEntry>> {
            Ok(self.entries.lock().iter().find(|e| e.key == key).cloned())
        }
        async fn list(
            &self,
            _category: Option<&MemoryCategory>,
            _limit: usize,
        ) -> anyhow::Result<Vec<MemoryEntry>> {
            Ok(self.entries.lock().clone())
        }
        async fn forget(&self, key: &str) -> anyhow::Result<bool> {
            let mut entries = self.entries.lock();
            let before = entries.len();
            entries.retain(|e| e.key != key);
            Ok(entries.len() < before)
        }
        async fn count(
            &self,
            _category: Option<&MemoryCategory>,
        ) -> anyhow::Result<usize> {
            Ok(self.entries.lock().len())
        }
        async fn health_check(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn make_state(memory: Arc<dyn Memory>) -> PluginHostState {
        PluginHostState {
            plugin_name: "testplugin".to_string(),
            path_sandbox: Arc::new(PathSandbox::default()),
            memory,
            http_client: reqwest::Client::new(),
            rt_handle: Handle::current(),
            settings: HashMap::new(),
            bus: None,
        }
    }

    /// `host_config_get_string` returns the value when present and
    /// `None` when missing. Plugins see only their own slice — the
    /// scoping is enforced by the loader (which builds the
    /// `settings` map from the plugin's own section), so at the
    /// host-function level the call is a plain map lookup.
    // The host functions use `tokio::task::block_in_place` which
    // requires the multi-threaded scheduler, so every async test in
    // this module pins `flavor = "multi_thread"`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn config_get_string_present_and_missing() {
        let mut state = make_state(Arc::new(MockMemory::new()));
        state.settings.insert("market".to_string(), "US".to_string());

        assert_eq!(host_config_get_string(&state, "market"), Some("US".to_string()));
        assert_eq!(host_config_get_string(&state, "missing"), None);
    }

    /// `host_channel_send` with `bus = None` (CLI mode) must return
    /// an error rather than panicking. Plugins running outside the
    /// gateway should be able to detect "channels not available"
    /// gracefully.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn channel_send_without_bus_returns_error() {
        let state = make_state(Arc::new(MockMemory::new()));
        let r = host_channel_send(&state, "telegram", "12345", "hi");
        assert!(r.is_err());
        let msg = r.unwrap_err();
        assert!(msg.contains("gateway"), "expected gateway-mention in: {msg}");
    }

    /// `host_memory_get` returns `Some` for an entry that exists,
    /// `None` for one that doesn't.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn memory_get_round_trip() {
        let mem: Arc<dyn Memory> = Arc::new(MockMemory::new());
        let state = make_state(Arc::clone(&mem));
        // Seed an entry directly via the memory backend (bypassing
        // namespace scoping — this simulates the agent or another
        // plugin having written it).
        let now = chrono::Utc::now().to_rfc3339();
        mem.store(MemoryEntry {
            key: "k1".to_string(),
            content: "v1".to_string(),
            category: MemoryCategory::Custom("plugin:testplugin".to_string()),
            created_at: now.clone(),
            updated_at: now,
            ..MemoryEntry::default()
        })
        .await
        .unwrap();

        let got = host_memory_get(&state, "k1").unwrap();
        assert!(got.is_some());
        let entry = got.unwrap();
        assert_eq!(entry.key, "k1");
        assert_eq!(entry.content, "v1");
        assert_eq!(entry.category, "plugin:testplugin");

        let missing = host_memory_get(&state, "nope").unwrap();
        assert!(missing.is_none());
    }

    /// `host_memory_forget` returns `true` when the entry was in
    /// the plugin's own namespace and got deleted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn memory_forget_in_own_namespace_succeeds() {
        let mem: Arc<dyn Memory> = Arc::new(MockMemory::new());
        let state = make_state(Arc::clone(&mem));
        let now = chrono::Utc::now().to_rfc3339();
        mem.store(MemoryEntry {
            key: "mine".to_string(),
            content: "data".to_string(),
            category: MemoryCategory::Custom("plugin:testplugin".to_string()),
            created_at: now.clone(),
            updated_at: now,
            ..MemoryEntry::default()
        })
        .await
        .unwrap();

        let deleted = host_memory_forget(&state, "mine").unwrap();
        assert!(deleted, "expected forget to return true");
        // Verify the entry actually went away.
        let after = host_memory_get(&state, "mine").unwrap();
        assert!(after.is_none());
    }

    /// `host_memory_forget` returns `false` (not an error) when the
    /// entry exists but belongs to another category. Crucially, the
    /// entry must NOT be deleted — we're enforcing the plugin
    /// namespace boundary at the host level.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn memory_forget_other_namespace_refused() {
        let mem: Arc<dyn Memory> = Arc::new(MockMemory::new());
        let state = make_state(Arc::clone(&mem));
        let now = chrono::Utc::now().to_rfc3339();
        // Seed an entry in Core memory (not the plugin's namespace).
        mem.store(MemoryEntry {
            key: "core_secret".to_string(),
            content: "important".to_string(),
            category: MemoryCategory::Core,
            created_at: now.clone(),
            updated_at: now,
            ..MemoryEntry::default()
        })
        .await
        .unwrap();

        let result = host_memory_forget(&state, "core_secret").unwrap();
        assert!(!result, "expected forget to refuse cross-namespace delete");

        // Crucially: the entry must still be there.
        let still_there = host_memory_get(&state, "core_secret").unwrap();
        assert!(
            still_there.is_some(),
            "cross-namespace forget must NOT delete the entry"
        );
        assert_eq!(still_there.unwrap().category, "core");
    }

    /// `host_memory_forget` on a missing key is idempotent — returns
    /// `false`, doesn't error. This lets plugins call it without
    /// pre-checking existence.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn memory_forget_missing_key_is_idempotent() {
        let state = make_state(Arc::new(MockMemory::new()));
        let r = host_memory_forget(&state, "definitely-not-there").unwrap();
        assert!(!r);
    }
}
