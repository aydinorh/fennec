use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level Fennec configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FennecConfig {
    pub identity: IdentityConfig,
    pub provider: ProviderConfig,
    pub memory: MemoryConfig,
    pub security: SecurityConfig,
    pub agent: AgentConfig,
    pub channels: ChannelsConfig,
    pub gateway: GatewayConfig,
    pub cron: CronConfig,
    pub collective: CollectiveConfig,
    pub plugins: PluginsConfig,
    pub auxiliary: AuxiliaryConfigToml,
    pub skills: SkillsConfigToml,
    pub tools: ToolsConfig,
    pub tui: TuiConfig,
}

impl Default for FennecConfig {
    fn default() -> Self {
        Self {
            identity: IdentityConfig::default(),
            provider: ProviderConfig::default(),
            memory: MemoryConfig::default(),
            security: SecurityConfig::default(),
            agent: AgentConfig::default(),
            channels: ChannelsConfig::default(),
            gateway: GatewayConfig::default(),
            cron: CronConfig::default(),
            collective: CollectiveConfig::default(),
            plugins: PluginsConfig::default(),
            auxiliary: AuxiliaryConfigToml::default(),
            skills: SkillsConfigToml::default(),
            tools: ToolsConfig::default(),
            tui: TuiConfig::default(),
        }
    }
}

/// Plugin-system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PluginsConfig {
    pub enabled: Vec<String>,
    pub settings: std::collections::HashMap<String, std::collections::HashMap<String, String>>,
}

impl Default for PluginsConfig {
    fn default() -> Self {
        Self {
            enabled: Vec::new(),
            settings: std::collections::HashMap::new(),
        }
    }
}

/// Configuration for the skills subsystem.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SkillsConfigToml {
    pub guard: SkillsGuardConfigToml,
}

/// Static safety scanner configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SkillsGuardConfigToml {
    pub guard_agent_created: bool,
    pub disabled_categories: Vec<String>,
    pub disabled_rules: Vec<String>,
}

/// Per-task auxiliary client config.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AuxiliaryConfigToml {
    pub curator: AuxiliaryTaskToml,
    pub compression: AuxiliaryTaskToml,
    pub web_extract: AuxiliaryTaskToml,
    pub vision: AuxiliaryTaskToml,
    pub session_search: AuxiliaryTaskToml,
    pub smart_approval: AuxiliaryTaskToml,
    pub title: AuxiliaryTaskToml,
    pub custom: std::collections::HashMap<String, AuxiliaryTaskToml>,
}

/// Per-task config row.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AuxiliaryTaskToml {
    pub provider: String,
    pub model: String,
    pub timeout_secs: u64,
}

/// User-toggleable TUI display settings that survive a restart.
/// New entries here should match the command that toggles them —
/// `/compact`, `/details`, `/skin`, `/statusbar`, `/indicator`,
/// `/verbose`, `/busy`, `/reasoning`, `/personality`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiConfig {
    pub compact: bool,
    pub details: String,
    /// Per-section detail overrides — set via
    /// `/details <section> <mode>`. When a section name is
    /// present, its value wins over the global `details` mode
    /// for rendering that section. Valid section names:
    /// `thinking`, `tools`, `subagents`, `activity`. Mirrors
    /// Hermes' `details_mode.*` config keys + `ui.sections`.
    #[serde(default)]
    pub details_sections: std::collections::HashMap<String, String>,
    /// Status bar position. One of `"top"`, `"bottom"`, `"off"`.
    /// Empty / unknown falls back to `"bottom"` at runtime to
    /// preserve existing behaviour. Toggled by `/statusbar`.
    pub statusbar: String,
    /// Active skin (theme variant). Empty falls back to
    /// `"fennec-warm"` (current palette). Toggled by `/skin`.
    pub skin: String,
    /// Spinner indicator style. One of `"braille"` (default,
    /// matches existing behaviour), `"ascii"`, `"kaomoji"`,
    /// `"emoji"`, `"unicode"`. Toggled by `/indicator`.
    pub indicator: String,
    /// Tool-output verbosity. One of `"normal"` (default) or
    /// `"verbose"`. Toggled by `/verbose`.
    pub verbose: String,
    /// Behaviour when Enter is pressed mid-turn. One of
    /// `"interrupt"` (default), `"queue"`, `"steer"`. Toggled
    /// by `/busy`.
    pub busy: String,
    /// Active persona name. Empty = use IdentityConfig.persona
    /// as-is. Otherwise looks up a built-in or user preset.
    /// Toggled by `/personality`.
    pub personality: String,
}

/// Tool toggle configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    pub disabled: Vec<String>,
}

impl FennecConfig {
    /// Resolve the Fennec home directory.
    /// Priority: override arg > $FENNEC_HOME > ~/.fennec
    pub fn resolve_home(override_path: Option<&str>) -> PathBuf {
        if let Some(p) = override_path {
            return PathBuf::from(p);
        }
        if let Ok(env_home) = std::env::var("FENNEC_HOME") {
            return PathBuf::from(env_home);
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".fennec")
    }

    /// Resolve the home directory for a named profile.
    ///
    /// Profiles are a friendly convention layered on top of
    /// [`Self::resolve_home`]: instead of typing the full
    /// `--config-dir ~/.fennec/profiles/work` every time, the user
    /// types `--profile work` and we resolve the same path. The base
    /// directory follows the normal `$FENNEC_HOME` / `~/.fennec`
    /// resolution; the profile name is appended under a `profiles/`
    /// subdirectory.
    ///
    /// `name` is validated to keep it usable as a directory component
    /// — alphanumeric plus `-` and `_`, length 1-64. This rejects
    /// path-traversal attempts (`..`, `/foo`), surprising whitespace,
    /// and other shenanigans before we ever touch the filesystem.
    /// Returns `Err` with a clear message on rejection.
    pub fn resolve_profile_home(name: &str) -> Result<PathBuf> {
        validate_profile_name(name)?;
        let base = Self::resolve_home(None);
        Ok(base.join("profiles").join(name))
    }

    /// Load configuration from a TOML file at `path`.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: FennecConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Persist configuration to `path` as TOML. Used by `/model`,
    /// `/tools`, and other slash commands that mutate runtime
    /// config and need the change to survive a restart. Creates
    /// the parent directory if missing.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = toml::to_string_pretty(self)?;
        std::fs::write(path, body)?;
        Ok(())
    }
}

/// Profile name validation. Accepts ASCII alphanumeric + `-` + `_`,
/// 1-64 characters. Rejects everything else with a contextual error.
///
/// The constraints exist for two reasons:
///
/// 1. **Path safety.** A profile name is interpolated into a directory
///    path. Without validation, `--profile ../../../etc` would resolve
///    to a path outside `~/.fennec/profiles/` and we'd happily try to
///    write a `.key` file there. Restricting to `[A-Za-z0-9_-]` makes
///    every accepted name a valid single path component on every
///    supported platform, with no separator characters and no parent
///    references.
/// 2. **Predictability across shells.** Profile names show up in
///    shell history, scripts, and systemd unit names. Disallowing
///    spaces and quote characters means an operator can copy-paste
///    a profile name into any context without quoting hazards.
///
/// The 64-char cap is well above any realistic profile name and well
/// below any path-length limit. Empty names are rejected because they
/// would resolve to `~/.fennec/profiles/`, which is the parent of all
/// profile dirs and not a valid state directory itself.
fn validate_profile_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("profile name cannot be empty");
    }
    if name.len() > 64 {
        anyhow::bail!(
            "profile name '{}' is too long ({} chars; max 64)",
            name,
            name.len()
        );
    }
    for (i, ch) in name.chars().enumerate() {
        let ok = ch.is_ascii_alphanumeric() || ch == '-' || ch == '_';
        if !ok {
            anyhow::bail!(
                "profile name '{}' contains invalid character '{}' at position {}; \
                 allowed: ASCII letters, digits, '-', '_'",
                name,
                ch,
                i
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn validates_simple_name() {
        assert!(validate_profile_name("work").is_ok());
        assert!(validate_profile_name("personal").is_ok());
        assert!(validate_profile_name("test-1").is_ok());
        assert!(validate_profile_name("user_2").is_ok());
        assert!(validate_profile_name("a").is_ok());
        assert!(validate_profile_name("ABC123").is_ok());
    }

    #[test]
    fn rejects_empty_name() {
        let err = validate_profile_name("").unwrap_err().to_string();
        assert!(err.contains("empty"), "expected 'empty' in: {}", err);
    }

    #[test]
    fn rejects_path_traversal() {
        // Both `..` and slash-bearing names must be rejected — these are
        // the realistic attack shapes against an unvalidated profile name.
        assert!(validate_profile_name("..").is_err());
        assert!(validate_profile_name("../../etc").is_err());
        assert!(validate_profile_name("foo/bar").is_err());
        assert!(validate_profile_name("/absolute").is_err());
    }

    #[test]
    fn rejects_whitespace_and_quotes() {
        assert!(validate_profile_name("with space").is_err());
        assert!(validate_profile_name("with\ttab").is_err());
        assert!(validate_profile_name("with'quote").is_err());
        assert!(validate_profile_name("with\"dquote").is_err());
    }

    #[test]
    fn rejects_overlong_name() {
        let long = "a".repeat(65);
        let err = validate_profile_name(&long).unwrap_err().to_string();
        assert!(err.contains("too long"), "expected 'too long' in: {}", err);
        // Boundary: exactly 64 chars is fine.
        let limit = "a".repeat(64);
        assert!(validate_profile_name(&limit).is_ok());
    }

    #[test]
    fn rejects_dotfiles() {
        // A leading dot would create a hidden profile dir which conflicts
        // with the convention that everything visible under
        // `profiles/` is a real profile. Rejected by the alphanumeric
        // rule (`.` isn't alphanumeric or `-`/`_`).
        assert!(validate_profile_name(".hidden").is_err());
    }

    #[test]
    fn resolve_profile_home_appends_under_profiles() {
        // We don't assert the absolute base here because it depends on
        // $HOME / $FENNEC_HOME at test time; we just check the relative
        // shape: <base>/profiles/<name>.
        let resolved = FennecConfig::resolve_profile_home("work").unwrap();
        let s = resolved.to_string_lossy();
        assert!(
            s.ends_with("/profiles/work"),
            "expected path to end with /profiles/work, got: {}",
            s
        );
    }

    #[test]
    fn resolve_profile_home_rejects_invalid_name() {
        assert!(FennecConfig::resolve_profile_home("..").is_err());
        assert!(FennecConfig::resolve_profile_home("with/slash").is_err());
        assert!(FennecConfig::resolve_profile_home("").is_err());
    }
}

/// Identity configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IdentityConfig {
    pub name: String,
    pub persona: String,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            name: "Fennec".to_string(),
            persona: "Your personal AI agent — sharp, resourceful, and always on.".to_string(),
        }
    }
}

/// LLM provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub name: String,
    pub model: String,
    pub api_key: String,
    pub base_url: String,
    pub temperature: f64,
    pub max_tokens: u32,
    pub fallback_models: Vec<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            name: "anthropic".to_string(),
            model: "claude-sonnet-4-6".to_string(),
            api_key: String::new(),
            base_url: String::new(),
            temperature: 0.7,
            max_tokens: 8192,
            fallback_models: Vec::new(),
        }
    }
}

/// Memory subsystem configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub db_path: Option<String>,
    pub vector_weight: f64,
    pub keyword_weight: f64,
    pub half_life_days: f64,
    pub cache_max: usize,
    pub context_limit: usize,
    pub embedding_provider: String,
    pub embedding_api_key: String,
    pub consolidation_enabled: bool,
    /// Optional external memory provider plugin to run alongside
    /// the always-on built-in SQLite store. Default `"builtin"` (or
    /// empty) means built-in memory is the only memory layer
    /// active — current behavior of pre-C3 Fennec.
    ///
    /// Set this to a registered plugin's name (e.g.
    /// `provider = "honcho"`) to run that plugin's
    /// [`MemoryProvider`](crate::plugins::MemoryProvider)
    /// alongside the built-in store. The provider augments — it
    /// does NOT replace local SQLite. Your data stays local.
    ///
    /// Names that don't resolve to a registered provider produce
    /// a startup warning and fall back to builtin-only.
    pub provider: String,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            db_path: None,
            vector_weight: 0.7,
            keyword_weight: 0.3,
            half_life_days: 7.0,
            cache_max: 10000,
            context_limit: 5,
            embedding_provider: "noop".to_string(),
            embedding_api_key: String::new(),
            consolidation_enabled: true,
            provider: "builtin".to_string(),
        }
    }
}

/// Security configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SecurityConfig {
    pub prompt_guard_action: String,
    pub prompt_guard_sensitivity: f64,
    pub encrypt_secrets: bool,
    pub command_allowlist: Vec<String>,
    pub forbidden_paths: Vec<String>,
    pub command_timeout_secs: u64,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            prompt_guard_action: "warn".to_string(),
            prompt_guard_sensitivity: 0.7,
            encrypt_secrets: true,
            command_allowlist: vec![
                "git", "ls", "cat", "grep", "find", "echo", "pwd", "wc", "head", "tail", "date",
                "df", "du", "uname", "cargo", "npm", "node", "python", "python3", "pip",
                "curl", "wget", "which", "env", "sort", "uniq", "tr", "cut", "sed", "awk",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            forbidden_paths: vec![
                "/etc", "/root", "/boot", "/dev", "/proc", "/sys",
                ".fennec/config.toml", ".fennec/.secret_key", ".fennec/.anthropic_oauth.json",
                ".ssh", ".gnupg", ".aws", ".config/gcloud",
            ]
                .into_iter()
                .map(String::from)
                .collect(),
            command_timeout_secs: 60,
        }
    }
}

/// Agent behaviour configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentConfig {
    pub max_tool_iterations: u32,
    pub context_window: u64,
    pub compression_threshold: f64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_iterations: 90,
            context_window: 200_000,
            compression_threshold: 0.50,
        }
    }
}

/// Channel configuration for all supported messaging channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChannelsConfig {
    pub telegram: ChannelEntry,
    pub discord: ChannelEntry,
    pub slack: SlackChannelEntry,
    pub whatsapp: WhatsAppChannelEntry,
    pub email: EmailChannelEntry,
    pub webhook: WebhookChannelEntry,
    pub openai_compat: OpenAiCompatChannelEntry,
    pub signal: SignalChannelEntry,
    pub matrix: MatrixChannelEntry,
}

impl Default for ChannelsConfig {
    fn default() -> Self {
        Self {
            telegram: ChannelEntry::default(),
            discord: ChannelEntry::default(),
            slack: SlackChannelEntry::default(),
            whatsapp: WhatsAppChannelEntry::default(),
            email: EmailChannelEntry::default(),
            webhook: WebhookChannelEntry::default(),
            openai_compat: OpenAiCompatChannelEntry::default(),
            signal: SignalChannelEntry::default(),
            matrix: MatrixChannelEntry::default(),
        }
    }
}

/// Matrix channel — speaks the Matrix Client-Server API directly
/// (no SDK dependency). Auth is either a static `access_token` or
/// `user_id`+`password` (which the channel exchanges for a token at
/// startup). Provide `device_id` for stable session identity.
///
/// Room visibility:
///   - `allowed_users`: Matrix user-id allowlist for DM senders. Empty
///     means everyone is allowed. Format: `@user:server`.
///   - `allowed_rooms`: room-id allowlist. Empty disables group rooms;
///     `*` allows all; otherwise an explicit list of `!roomId:server`.
///   - `free_response_rooms`: rooms in which the bot replies even
///     without an explicit `@mention` (subset of `allowed_rooms`).
///
/// Threading:
///   - `auto_thread` (default true): in non-free rooms, replies are
///     wrapped in a thread anchored on the inbound mention.
///   - `dm_auto_thread` (default false): same behavior in DMs.
///   - `dm_mention_threads` (default false): in DMs, only thread when
///     the inbound message contains a mention.
///
/// `markdown_to_html` (default true) controls whether the agent's
/// markdown output is rendered to `formatted_body` HTML alongside the
/// plain `body`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MatrixChannelEntry {
    pub enabled: bool,
    /// Homeserver base URL, e.g. `https://matrix.org`.
    pub homeserver: String,
    /// Long-lived access token. Preferred over password.
    pub access_token: String,
    /// Bot user-id, e.g. `@fennec:matrix.org`. Required.
    pub user_id: String,
    /// Password (only used when `access_token` is empty).
    pub password: String,
    /// Stable device id; recommended even without E2EE so the
    /// homeserver doesn't churn devices on each restart.
    pub device_id: String,
    /// DM allowlist (matrix user-ids). Empty means everyone.
    pub allowed_users: Vec<String>,
    /// Room allowlist (`!roomId:server` or `*` or empty).
    pub allowed_rooms: Vec<String>,
    /// Rooms where the bot answers without requiring an `@mention`.
    pub free_response_rooms: Vec<String>,
    /// Whether to require an `@mention` in non-free rooms. Default true.
    pub require_mention: bool,
    /// Auto-thread replies in non-free rooms. Default true.
    pub auto_thread: bool,
    /// Auto-thread replies in DMs. Default false.
    pub dm_auto_thread: bool,
    /// In DMs, only thread when the inbound has a mention. Default false.
    pub dm_mention_threads: bool,
    /// Render markdown to HTML `formatted_body`. Default true.
    pub markdown_to_html: bool,
    /// Optional directory for caching inbound media files
    /// (`m.image` / `m.file` / `m.audio` / `m.video`). When set,
    /// the channel downloads the binary on receipt and surfaces the
    /// local path in `metadata.matrix_media_path` so vision /
    /// transcription tools can consume it directly. Empty disables
    /// auto-download (only the `mxc://` URL is surfaced).
    pub media_cache_dir: String,
    /// Optional path where the channel persists its `next_batch`
    /// sync token across restarts. When set, the channel resumes
    /// from the last seen point rather than re-running the initial
    /// sync. Empty disables persistence (every restart re-syncs).
    pub state_file: String,
    /// If non-zero, outbound text messages destined for the same
    /// chat within this many milliseconds are coalesced into a
    /// single send. Mirrors the upstream's batching behavior for
    /// rapid-fire LLM output. Default 0 (no batching) — Fennec
    /// emits one message per turn so this is mostly future-proofing
    /// for streaming-style integrations.
    pub text_batch_delay_ms: u64,
    /// Optional directory for the SqliteCryptoStore (matrix-e2ee
    /// feature only). When set and the feature is enabled, the
    /// channel reads / writes encrypted-room messages and persists
    /// Olm + Megolm sessions across restarts. Empty disables E2EE
    /// even when the feature is built; the channel falls back to
    /// the unencrypted path.
    pub crypto_store_dir: String,
    /// Optional passphrase encrypting the SQLite crypto store at
    /// rest. Empty leaves the store unencrypted (file-system
    /// permissions become the only protection on the key
    /// material).
    pub crypto_store_passphrase: String,
    /// Default destination for `send_message` calls without a chat
    /// id. Empty falls back to most-recent inbound.
    pub home_chat_id: String,
}

impl Default for MatrixChannelEntry {
    fn default() -> Self {
        Self {
            enabled: false,
            homeserver: String::new(),
            access_token: String::new(),
            user_id: String::new(),
            password: String::new(),
            device_id: String::new(),
            allowed_users: Vec::new(),
            allowed_rooms: Vec::new(),
            free_response_rooms: Vec::new(),
            require_mention: true,
            auto_thread: true,
            dm_auto_thread: false,
            dm_mention_threads: false,
            markdown_to_html: true,
            media_cache_dir: String::new(),
            state_file: String::new(),
            text_batch_delay_ms: 0,
            crypto_store_dir: String::new(),
            crypto_store_passphrase: String::new(),
            home_chat_id: String::new(),
        }
    }
}

/// Signal channel — connects to a `signal-cli` daemon over HTTP.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SignalChannelEntry {
    pub enabled: bool,
    pub http_url: String,
    pub account: String,
    pub allowed_users: Vec<String>,
    pub group_allowed_users: Vec<String>,
    pub ignore_stories: bool,
    pub home_chat_id: String,
}

impl Default for SignalChannelEntry {
    fn default() -> Self {
        Self {
            enabled: false,
            http_url: "http://127.0.0.1:8080".to_string(),
            account: String::new(),
            allowed_users: Vec::new(),
            group_allowed_users: Vec::new(),
            ignore_stories: true,
            home_chat_id: String::new(),
        }
    }
}

/// OpenAI-compatible HTTP API channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiCompatChannelEntry {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub api_key: String,
    pub model_name: String,
    pub cors_origins: String,
}

impl Default for OpenAiCompatChannelEntry {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "127.0.0.1".into(),
            port: 8642,
            api_key: String::new(),
            model_name: "fennec-agent".into(),
            cors_origins: String::new(),
        }
    }
}

/// Generic HTTP webhook channel: receives POSTs from external
/// systems (CI, monitoring, GitHub/GitLab events, etc.), validates
/// HMAC signatures, renders a prompt template against the payload,
/// and feeds the result to the agent loop. Outbound is a no-op:
/// webhooks are inbound-only, with replies routed through other
/// configured channels (telegram/discord/slack/etc.) by the agent's
/// regular send-message flow.
///
/// Routes are defined in `[channels.webhook.routes.<name>]`. Each
/// route has its own secret (HMAC), event allowlist, prompt
/// template, and optional skill list. A global secret in
/// `channels.webhook.secret` acts as the fallback when a route
/// doesn't supply its own.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WebhookChannelEntry {
    /// Master switch. Off means the HTTP server doesn't bind at all.
    pub enabled: bool,
    /// HTTP listen address. Default `0.0.0.0` so the server is
    /// reachable from the network; tighten to `127.0.0.1` for
    /// local-only.
    pub host: String,
    /// HTTP listen port. Default 8644 matches the upstream default.
    pub port: u16,
    /// Optional global HMAC secret used when a route doesn't define
    /// its own. Empty disables global fallback (each route must set
    /// `secret`).
    pub secret: String,
    /// Idempotency cache TTL in seconds. Webhooks frequently retry
    /// on failure; we de-dup by `(route, body-hash)` for this
    /// window so a duplicate POST doesn't double-fire the agent.
    /// Default 3600 (1 hour). Set 0 to disable.
    pub idempotency_ttl_secs: u64,
    /// Per-route rate limit, requests per minute. Default 30. The
    /// limiter is a fixed-window counter; bursts within one minute
    /// past this number get a 429 response.
    pub rate_limit_per_minute: u32,
    /// Per-route configuration map. The route name is the URL
    /// segment: `POST /webhook/<name>`.
    pub routes: std::collections::HashMap<String, WebhookRouteEntry>,
}

impl Default for WebhookChannelEntry {
    fn default() -> Self {
        Self {
            enabled: false,
            host: "0.0.0.0".into(),
            port: 8644,
            secret: String::new(),
            idempotency_ttl_secs: 3600,
            rate_limit_per_minute: 30,
            routes: std::collections::HashMap::new(),
        }
    }
}

/// Configuration for one webhook route under
/// `[channels.webhook.routes.<name>]`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct WebhookRouteEntry {
    /// Per-route HMAC secret. Falls back to `channels.webhook.secret`
    /// when empty. The literal string `INSECURE_NO_AUTH` skips
    /// signature checks entirely — only for testing on a trusted
    /// network. A startup warning is logged when a route uses it.
    pub secret: String,
    /// Optional event-type allowlist. Empty means "all events
    /// pass". For GitHub the event type is read from the
    /// `X-GitHub-Event` header; for GitLab from `X-Gitlab-Event`;
    /// otherwise from a top-level `event_type` field in the JSON
    /// body.
    pub events: Vec<String>,
    /// Prompt template rendered against the JSON payload. Supports
    /// dot-notation placeholders: `{pull_request.title}` looks up
    /// `payload["pull_request"]["title"]`. Missing keys render as
    /// the empty string with a debug log; the agent receives the
    /// rendered prompt as a user message.
    pub prompt: String,
    /// Optional list of skill names to load before the agent
    /// answers. Skills are loaded on top of the always-on set; the
    /// route can scope work to a specific tool surface.
    pub skills: Vec<String>,
    /// Where the rendered prompt's response should be delivered.
    /// Possible values: `"log"` (just log the response, no
    /// outbound message — the default), `"telegram"` / `"discord"`
    /// / `"slack"` / `"email"` (route through that channel via the
    /// usual send pipeline; needs `deliver_target` to specify
    /// `chat_id`).
    pub deliver: String,
    /// Free-form metadata for the deliver path: `chat_id` for
    /// telegram/discord/slack, `to` for email, GitHub repo + PR
    /// number for `github_comment` (later phase). Stored as a
    /// string→string map for now to avoid schema lock-in.
    pub deliver_extra: std::collections::HashMap<String, String>,
}

/// Generic channel entry (Telegram, Discord).
///
/// `home_chat_id`, when set, is the default destination the agent uses
/// when it calls `send_message` without an explicit chat_id (e.g. the
/// LLM asks to "send a reminder to telegram"). Empty string means no
/// home chat is configured; the tool then falls back to the most
/// recently-seen chat on this channel, or refuses if nothing has been
/// seen yet.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ChannelEntry {
    pub enabled: bool,
    pub token: String,
    pub allowed_users: Vec<String>,
    pub home_chat_id: String,
}

/// Slack-specific channel entry (requires bot_token + app_token).
///
/// See [`ChannelEntry::home_chat_id`] for the role of `home_chat_id`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SlackChannelEntry {
    pub enabled: bool,
    pub bot_token: String,
    pub app_token: String,
    pub allowed_users: Vec<String>,
    pub home_chat_id: String,
}

/// WhatsApp Cloud API channel entry.
///
/// `app_secret` is the Meta App Secret used to verify the
/// `X-Hub-Signature-256` HMAC on incoming webhook POSTs. If empty, signature
/// verification is skipped (for dev / not yet configured) and a warning is
/// logged at startup — do not leave it empty for internet-reachable deploys.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhatsAppChannelEntry {
    pub enabled: bool,
    pub phone_number_id: String,
    pub access_token: String,
    pub verify_token: String,
    pub app_secret: String,
    pub webhook_port: u16,
    pub allowed_users: Vec<String>,
    /// See [`ChannelEntry::home_chat_id`] for semantics.
    pub home_chat_id: String,
}

impl Default for WhatsAppChannelEntry {
    fn default() -> Self {
        Self {
            enabled: false,
            phone_number_id: String::new(),
            access_token: String::new(),
            verify_token: String::new(),
            app_secret: String::new(),
            webhook_port: 9443,
            allowed_users: Vec::new(),
            home_chat_id: String::new(),
        }
    }
}

/// Email channel entry (IMAP polling + SMTP sending).
///
/// See [`ChannelEntry::home_chat_id`] for the role of `home_chat_id` —
/// for email this is typically the user's own email address.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmailChannelEntry {
    pub enabled: bool,
    pub imap_host: String,
    pub imap_port: u16,
    pub imap_user: String,
    pub imap_password: String,
    pub smtp_host: String,
    pub smtp_port: u16,
    pub smtp_user: String,
    pub smtp_password: String,
    pub from_address: String,
    pub allowed_senders: Vec<String>,
    pub poll_interval_secs: u64,
    pub home_chat_id: String,
}

impl Default for EmailChannelEntry {
    fn default() -> Self {
        Self {
            enabled: false,
            imap_host: String::new(),
            imap_port: 993,
            imap_user: String::new(),
            imap_password: String::new(),
            smtp_host: String::new(),
            smtp_port: 587,
            smtp_user: String::new(),
            smtp_password: String::new(),
            from_address: String::new(),
            allowed_senders: Vec::new(),
            poll_interval_secs: 30,
            home_chat_id: String::new(),
        }
    }
}

/// HTTP gateway configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GatewayConfig {
    pub host: String,
    pub port: u16,
    pub auth_token: String,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8990,
            auth_token: String::new(),
        }
    }
}

/// Cron scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CronConfig {
    pub enabled: bool,
}

impl Default for CronConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// Collective intelligence configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CollectiveConfig {
    pub enabled: bool,
    pub api_key: String,
    pub base_url: String,
    pub publish_enabled: bool,
    pub search_enabled: bool,
    pub cache_ttl_days: u64,
}

impl Default for CollectiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: String::new(),
            base_url: "https://api.plurum.ai".to_string(),
            publish_enabled: true,
            search_enabled: true,
            cache_ttl_days: 30,
        }
    }
}
