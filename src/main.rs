use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use fennec::agent::AgentBuilder;
use fennec::auth;
use fennec::bus::MessageBus;
use fennec::channels::cli::CliChannel;
use fennec::channels::traits::{Channel, SendMessage};
use fennec::channels::ChannelManager;
use fennec::collective::cache::CollectiveCache;
use fennec::collective::mock::MockCollective;
use fennec::collective::plurum::PlurumlClient;
use fennec::collective::search::CollectiveSearch;
use fennec::collective::traits::CollectiveLayer;
use fennec::config::{FennecConfig, ProviderConfig};
use fennec::cron::{CronScheduler, JobStore};
use fennec::gateway::GatewayServer;
use fennec::memory::embedding::{NoopEmbedding, OpenAIEmbedding};
use fennec::memory::snapshot;
use fennec::memory::sqlite::SqliteMemory;
use fennec::memory::Memory;
use fennec::providers::anthropic::AnthropicProvider;
use fennec::providers::openai::OpenAIProvider;
use fennec::providers::gemini::GeminiProvider;
use fennec::providers::ollama::OllamaProvider;
use fennec::providers::traits::Provider;
use fennec::security::prompt_guard::{GuardAction, PromptGuard};
use fennec::security::{PathSandbox, SecretStore};
use fennec::tools::collective_tools::{CollectiveGetExperienceTool, CollectivePublishTool, CollectiveReportTool, CollectiveSearchTool};
use fennec::tools::cron_tool::{CronOrigin, CronTool};
use fennec::tools::files::{ListDirTool, ReadFileTool, WriteFileTool};
use fennec::tools::memory_tools::{MemoryForgetTool, MemoryRecallTool, MemoryStoreTool};
use fennec::tools::send_message_tool::SendMessageTool;
use fennec::tools::shell::ShellTool;
use fennec::tools::todo_tool::TodoTool;
use fennec::tools::ask_user_tool::AskUserTool;
use fennec::channels::{ChannelMapHandle, new_channel_map};
use fennec::tools::web::{WebFetchTool, WebSearchTool};
use fennec::tools::browser_tool::BrowserTool;
use fennec::tools::vision_tool::VisionTool;
use fennec::tools::image_gen_tool::{default_output_dir as image_output_dir, ImageGenTool};
use fennec::tools::code_exec_tool::CodeExecTool;
use fennec::tools::voice_tool::{
    default_tts_output_dir, resolve_openai_key as voice_resolve_openai_key,
    TextToSpeechTool, TranscribeAudioTool,
};
use fennec::tools::pdf_read_tool::PdfReadTool;
use fennec::tools::screenshot_tool::{default_screenshot_dir, ScreenshotTool};
use fennec::tools::http_request_tool::HttpRequestTool;
use fennec::tools::weather_tool::WeatherTool;
use fennec::tools::image_info_tool::ImageInfoTool;
use fennec::tools::claude_code_cli_tool::ClaudeCodeCliTool;
use fennec::skills::{Skill, SkillsLoader};
use fennec::tools::skills_tool::SkillsTool;
use fennec::tools::delegate_tool::DelegateTool;
use fennec::tools::traits::Tool;

#[derive(Parser, Debug)]
#[command(name = "fennec", version, about = "The fastest personal AI agent with collective intelligence")]
struct Cli {
    /// Override the Fennec home directory directly. Use this when you
    /// want full control over the path; otherwise prefer `--profile`.
    /// Mutually exclusive with `--profile`.
    #[arg(long, global = true)]
    config_dir: Option<String>,

    /// Run against a named profile under `~/.fennec/profiles/<name>/`.
    /// Each profile gets its own config, memory, secrets key, cron
    /// jobs, and OAuth tokens — fully isolated from every other
    /// profile and from the default `~/.fennec/` install.
    ///
    /// Names are restricted to `[A-Za-z0-9_-]` (1-64 chars).
    /// Mutually exclusive with `--config-dir`.
    #[arg(long, global = true)]
    profile: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start interactive agent session
    Agent {
        /// Single-shot message (non-interactive)
        #[arg(short, long)]
        message: Option<String>,

        /// Override the LLM model
        #[arg(long)]
        model: Option<String>,

        /// Open the terminal UI (sessions / chat / channels panels)
        /// instead of the line-by-line CLI. Existing CLI mode is
        /// the default.
        #[arg(long)]
        tui: bool,
    },
    /// Show agent status
    Status,
    /// Start gateway serving all configured channels
    Gateway {
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        port: Option<u16>,
    },
    /// Run interactive setup wizard
    Onboard {
        /// Overwrite existing config
        #[arg(long)]
        force: bool,
    },
    /// Authenticate with Anthropic via OAuth
    Login,
    /// Run diagnostic checks — provider reachability, API key, memory DB, Plurum, config.
    Doctor,
    /// Manage the skill curator — periodic background consolidation
    /// of agent-created skills.
    Curator {
        #[command(subcommand)]
        action: CuratorAction,
    },
    /// MCP (Model Context Protocol) integration.
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
}

#[derive(Subcommand, Debug)]
enum CuratorAction {
    /// Print last-run timestamp, summary, paused state, and run count.
    Status,
    /// Run the curator now, ignoring the idle/interval gate.
    Run,
    /// Suspend automatic curator runs (manual `run` still works).
    Pause,
    /// Resume automatic curator runs.
    Resume,
    /// Pin a skill so it is exempt from auto-transitions and curator consolidation.
    Pin { name: String },
    /// Remove a pin.
    Unpin { name: String },
    /// Restore a previously archived skill from `<home>/skills/.archive/`.
    Restore { name: String },
}

#[derive(Subcommand, Debug)]
enum McpAction {
    /// Run Fennec as an MCP server on stdio.
    Serve {
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Decrypt a channel-config secret through the [`SecretStore`].
fn decrypt_channel_secret(
    secret_store: &SecretStore,
    raw: &str,
    field_name: &str,
) -> Option<String> {
    if raw.is_empty() {
        return Some(String::new());
    }
    match secret_store.decrypt(raw) {
        Ok(value) => Some(value),
        Err(e) => {
            tracing::error!(
                "Failed to decrypt {field_name}: {e}; disabling associated channel \
                 (set a fresh encrypted value or paste plaintext)"
            );
            None
        }
    }
}

/// Resolve the API key from config or provider-specific environment variable.
fn resolve_api_key(config: &FennecConfig, secret_store: &SecretStore) -> Result<String> {
    // Try config value first.
    if !config.provider.api_key.is_empty() {
        let decrypted = secret_store
            .decrypt(&config.provider.api_key)
            .context("decrypting API key from config")?;
        return Ok(decrypted);
    }

    // Fall back to provider-specific environment variable.
    let env_var = match config.provider.name.as_str() {
        "anthropic" => "ANTHROPIC_API_KEY",
        "openai" => "OPENAI_API_KEY",
        "kimi" | "moonshot" => "KIMI_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "google" | "gemini" => "GEMINI_API_KEY",
        "ollama" => return Ok(String::new()), // Ollama needs no key
        _ => "ANTHROPIC_API_KEY",
    };

    std::env::var(env_var)
        .with_context(|| format!("API key not found: set provider.api_key in config or {} env var", env_var))
}

/// Build the auxiliary client. Used by background tasks (curator,
/// future title generation, smart-approval LLM, etc.) so they
/// don't dirty the main provider's prompt cache or pay primary-
/// model rates.
///
/// Resolution chain:
///
///  1. The user's primary provider always leads — single-key setups
///     work without extra config.
///  2. Opportunistic entries for whichever other provider keys are
///     configured (env vars or `[provider] api_key`). We call
///     `build_provider` with each backend's name so the resulting
///     `Arc<dyn Provider>` shares the existing routing logic.
///
/// Vision chain mirrors the text chain but skips Ollama (we don't
/// currently model multimodal capability per provider, so the
/// vision chain just excludes the local-only provider).
fn build_auxiliary_client(
    config: &FennecConfig,
    primary: Arc<dyn Provider>,
    secret_store: &SecretStore,
) -> fennec::providers::AuxiliaryClient {
    let mut text_chain: Vec<fennec::providers::ChainEntry> = Vec::new();
    let mut vision_chain: Vec<fennec::providers::ChainEntry> = Vec::new();

    // Primary always first.
    let primary_name = config.provider.name.clone();
    let primary_entry = fennec::providers::ChainEntry {
        name: primary_name.clone(),
        provider: primary,
    };
    text_chain.push(primary_entry.clone());
    if primary_name != "ollama" {
        vision_chain.push(primary_entry);
    }

    // Opportunistic additions. Each branch checks whether a key is
    // available from env (the primary's `[provider] api_key` is
    // already used above; here we check ONLY env vars to avoid
    // double-adding the primary when the configured api_key
    // happens to match an env var key).
    let try_add = |name: &str,
                   env_var: &str,
                   text_chain: &mut Vec<fennec::providers::ChainEntry>,
                   vision_chain: &mut Vec<fennec::providers::ChainEntry>,
                   include_vision: bool| {
        if text_chain.iter().any(|e| e.name == name) {
            return; // already covered by primary
        }
        let key = match std::env::var(env_var) {
            Ok(k) if !k.is_empty() => k,
            _ => return,
        };
        // Build a config-on-the-fly so build_provider picks the right
        // shape for each backend (Anthropic native vs OpenAI-compat).
        let aux_config = FennecConfig {
            provider: ProviderConfig {
                name: name.to_string(),
                model: String::new(), // empty → provider's own default
                api_key: String::new(), // not used; we pass key directly
                base_url: String::new(),
                temperature: 0.7,
                max_tokens: 8192,
                fallback_models: Vec::new(),
            },
            ..config.clone()
        };
        let provider_box = build_provider(&aux_config, key, None);
        let provider: Arc<dyn Provider> = Arc::from(provider_box);
        let entry = fennec::providers::ChainEntry {
            name: name.to_string(),
            provider,
        };
        text_chain.push(entry.clone());
        if include_vision {
            vision_chain.push(entry);
        }
    };

    try_add(
        "openrouter",
        "OPENROUTER_API_KEY",
        &mut text_chain,
        &mut vision_chain,
        true,
    );
    try_add(
        "anthropic",
        "ANTHROPIC_API_KEY",
        &mut text_chain,
        &mut vision_chain,
        true,
    );
    try_add(
        "openai",
        "OPENAI_API_KEY",
        &mut text_chain,
        &mut vision_chain,
        true,
    );
    try_add(
        "kimi",
        "KIMI_API_KEY",
        &mut text_chain,
        &mut vision_chain,
        false, // Kimi vision support is variable; conservative skip
    );
    try_add(
        "gemini",
        "GEMINI_API_KEY",
        &mut text_chain,
        &mut vision_chain,
        true, // Gemini is natively multimodal
    );
    let _ = secret_store; // reserved for future encrypted-aux-key
                          // resolution; placeholder so callers can
                          // pass it without breaking when we wire it.

    let aux_config = fennec::providers::AuxiliaryConfig::from_toml(&config.auxiliary);
    fennec::providers::AuxiliaryClient::new(aux_config, text_chain, vision_chain)
}

/// Resolve a provider Arc for the given config + new model name.
fn resolve_provider_with_model(
    config: &FennecConfig,
    home_dir: &std::path::Path,
    model: &str,
) -> Result<Arc<dyn Provider>> {
    let secret_store =
        SecretStore::new(home_dir.to_path_buf()).context("creating secret store")?;
    let provider: Box<dyn Provider> = if config.provider.name == "anthropic" {
        if let Ok(Some(oauth_token)) = auth::load_oauth_token(home_dir) {
            Box::new(AnthropicProvider::new_with_oauth(
                oauth_token,
                Some(model.to_string()),
            ))
        } else {
            let api_key = resolve_api_key(config, &secret_store)?;
            build_provider(config, api_key, Some(model.to_string()))
        }
    } else {
        let api_key = resolve_api_key(config, &secret_store)?;
        build_provider(config, api_key, Some(model.to_string()))
    };
    Ok(Arc::from(provider))
}

/// Build the LLM provider based on config.
fn build_provider(
    config: &FennecConfig,
    api_key: String,
    model_override: Option<String>,
) -> Box<dyn Provider> {
    let model = model_override.unwrap_or_else(|| config.provider.model.clone());
    let base_url = if config.provider.base_url.is_empty() {
        None
    } else {
        Some(config.provider.base_url.clone())
    };

    match config.provider.name.as_str() {
        "anthropic" => {
            Box::new(AnthropicProvider::new(api_key, Some(model)))
        }
        "openai" => {
            Box::new(OpenAIProvider::new(api_key, Some(model), base_url, None))
        }
        "kimi" | "moonshot" => {
            let kimi_url = base_url.unwrap_or_else(|| {
                // Route by key prefix: sk-kimi-* → api.kimi.com, otherwise → api.moonshot.ai
                if api_key.starts_with("sk-kimi-") {
                    "https://api.kimi.com/coding/v1".to_string()
                } else {
                    "https://api.moonshot.ai/v1".to_string()
                }
            });
            // Detect "user kept the Anthropic-flavored default and switched
            // provider to Kimi" — fall back to a Kimi-shaped default. We
            // accept BOTH the new default (claude-sonnet-4-6, current) and
            // the old default (claude-sonnet-4-20250514, deprecated June
            // 2026) so users whose configs still hold the legacy string
            // continue to get the right behavior.
            let kimi_model = if model.is_empty()
                || model == "claude-sonnet-4-6"
                || model == "claude-sonnet-4-20250514"
                || model == "moonshot-v1-128k"
            {
                "kimi-k2.5".to_string()
            } else {
                model
            };
            Box::new(OpenAIProvider::new(api_key, Some(kimi_model), Some(kimi_url), Some(262_144)))
        }
        "openrouter" => {
            let or_url = base_url.unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string());
            Box::new(OpenAIProvider::new(api_key, Some(model), Some(or_url), None))
        }
        "google" | "gemini" => {
            // Back-compat: if the user switched provider to Gemini but kept an
            // Anthropic-flavored default model string, fall back to Gemini's
            // own default rather than passing a non-Gemini model id.
            let gemini_model = if model.is_empty()
                || model == "claude-sonnet-4-6"
                || model == "claude-sonnet-4-20250514"
            {
                "gemini-2.5-flash".to_string()
            } else {
                model
            };
            Box::new(GeminiProvider::new(api_key, Some(gemini_model), base_url, None))
        }
        "ollama" => {
            // Same back-compat as kimi: accept both new and old Anthropic
            // defaults when falling back to Ollama's local default.
            let ollama_model = if model.is_empty()
                || model == "claude-sonnet-4-6"
                || model == "claude-sonnet-4-20250514"
            {
                "llama3.1".to_string()
            } else {
                model
            };
            Box::new(OllamaProvider::new(
                Some(ollama_model),
                base_url,
                None,
            ))
        }
        other => {
            // Unknown provider — treat as OpenAI-compatible with custom base_url
            tracing::warn!("Unknown provider '{other}', treating as OpenAI-compatible");
            Box::new(OpenAIProvider::new(api_key, Some(model), base_url, None))
        }
    }
}

/// Parse the guard action string from config into a GuardAction enum.
fn parse_guard_action(action: &str) -> GuardAction {
    match action.to_lowercase().as_str() {
        "block" => GuardAction::Block,
        "sanitize" => GuardAction::Sanitize,
        _ => GuardAction::Warn,
    }
}

/// Build an Agent from configuration. Shared by both `run_agent` and `run_gateway`.
///
/// Returns `(agent, memory, cron_origin_handle, pending_replies, chat_directory)`.
/// The latter two are used by the gateway's inbound dispatch:
///
///   - `pending_replies` lets `ask_user`-style tools register a wait
///     for the next inbound from a specific chat. The gateway's main
///     loop must call `take_and_deliver` *before* forwarding to the
///     agent; if a waiter is registered, the message is consumed by
///     the tool and not turned into a fresh agent turn.
///   - `chat_directory` is the source of truth for the `send_message`
///     tool's `list` action and for resolving "send to telegram" without
///     an explicit chat_id. The gateway's main loop must call
///     `record(channel, chat_id)` on every inbound (including those
///     consumed by pending replies, so the directory stays up to date).
// `plugin_bus` is the bus handle plumbed into WASM plugin host
// imports. `None` in CLI mode (no channels); `Some(bus.clone())` in
// gateway mode so plugins can use `channel-send`.
async fn build_agent(
    config: &FennecConfig,
    home_dir: &std::path::Path,
    model_override: Option<String>,
    channel_map: Option<ChannelMapHandle>,
    plugin_bus: Option<MessageBus>,
) -> Result<(
    fennec::agent::Agent,
    Arc<dyn Memory>,
    Arc<Mutex<Option<CronOrigin>>>,
    fennec::bus::PendingReplies,
    fennec::bus::ChatDirectory,
    Arc<dyn Provider>,
)> {
    build_agent_with_callbacks(config, home_dir, model_override, channel_map, None, None, None, None).await
}

async fn build_agent_with_callbacks(
    config: &FennecConfig,
    home_dir: &std::path::Path,
    model_override: Option<String>,
    channel_map: Option<ChannelMapHandle>,
    callbacks: Option<fennec::agent::callbacks::CallbacksHandle>,
    plugin_bus: Option<MessageBus>,
    delegation_registry: Option<fennec::agent::DelegationRegistry>,
    interrupt_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<(
    fennec::agent::Agent,
    Arc<dyn Memory>,
    Arc<Mutex<Option<CronOrigin>>>,
    fennec::bus::PendingReplies,
    fennec::bus::ChatDirectory,
    Arc<dyn Provider>,
)> {
    // Create SecretStore.
    let secret_store =
        SecretStore::new(home_dir.to_path_buf()).context("creating secret store")?;

    // Resolve provider: for Anthropic, try OAuth token first, then fall back to API key.
    let provider: Box<dyn Provider> = if config.provider.name == "anthropic" {
        if let Ok(Some(oauth_token)) = auth::load_oauth_token(home_dir) {
            tracing::info!("Using Anthropic OAuth token");
            let model = model_override.unwrap_or_else(|| config.provider.model.clone());
            Box::new(AnthropicProvider::new_with_oauth(oauth_token, Some(model)))
        } else {
            let api_key = resolve_api_key(config, &secret_store)?;
            build_provider(config, api_key, model_override)
        }
    } else {
        let api_key = resolve_api_key(config, &secret_store)?;
        build_provider(config, api_key, model_override)
    };

    // Promote the provider to `Arc` so it can be shared with DelegateTool
    // (which needs its own handle for sub-agent runs) while still being
    // passed into the AgentBuilder.
    let provider: Arc<dyn Provider> = Arc::from(provider);

    // Build the auxiliary client chain. Primary provider goes
    // first (so single-key setups Just Work), then opportunistic
    // entries for whichever provider env vars are also set. The
    // vision chain excludes Ollama (no vision support) and any
    // provider not known to handle multimodal input.
    let aux_client = build_auxiliary_client(
        &config,
        Arc::clone(&provider),
        &secret_store,
    );
    let aux_client = Arc::new(aux_client);

    // Create embedding provider based on config.
    let embedder: Arc<dyn fennec::memory::embedding::EmbeddingProvider> =
        match config.memory.embedding_provider.as_str() {
            "openai" => {
                let embedding_key = if !config.memory.embedding_api_key.is_empty() {
                    config.memory.embedding_api_key.clone()
                } else {
                    std::env::var("OPENAI_API_KEY").unwrap_or_default()
                };
                if embedding_key.is_empty() {
                    tracing::warn!("embedding_provider is 'openai' but no API key found; falling back to noop");
                    Arc::new(NoopEmbedding::new(1536))
                } else {
                    Arc::new(OpenAIEmbedding::new(embedding_key, None, None, None))
                }
            }
            _ => Arc::new(NoopEmbedding::new(1536)),
        };

    // Create SqliteMemory.
    let db_path = config
        .memory
        .db_path
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| home_dir.join("memory").join("brain.db"));
    let memory = SqliteMemory::new(
        db_path.clone(),
        config.memory.vector_weight as f32,
        config.memory.keyword_weight as f32,
        config.memory.cache_max,
        embedder,
    )
    .context("creating sqlite memory")?;
    let memory = Arc::new(memory);

    // Soul snapshot: hydrate from snapshot if DB is fresh and snapshot exists.
    let snapshot_path = home_dir.join("MEMORY_SNAPSHOT.md");
    let db_is_new = !db_path.exists() || {
        let entries = memory.list(None, 1).await.unwrap_or_default();
        entries.is_empty()
    };
    if db_is_new && snapshot_path.exists() {
        match snapshot::hydrate_from_snapshot(memory.as_ref(), &snapshot_path).await {
            Ok(count) => {
                tracing::info!(count, "hydrated memories from snapshot");
            }
            Err(e) => {
                tracing::warn!("failed to hydrate from snapshot: {e}");
            }
        }
    }

    // Create tools.
    let shell_tool = ShellTool::new(
        config.security.command_allowlist.clone(),
        config.security.forbidden_paths.clone(),
        config.security.command_timeout_secs,
    );
    // Shared filesystem sandbox driven by `security.forbidden_paths` from
    // config. Every tool that takes a file path from the LLM clones this
    // Arc so the denylist is enforced in one place.
    let path_sandbox = Arc::new(PathSandbox::new(config.security.forbidden_paths.clone()));
    let read_file_tool = ReadFileTool::new().with_sandbox(Arc::clone(&path_sandbox));
    let write_file_tool = WriteFileTool::new().with_sandbox(Arc::clone(&path_sandbox));
    let list_dir_tool = ListDirTool::new().with_sandbox(Arc::clone(&path_sandbox));
    let web_fetch_tool = WebFetchTool::new();
    let web_search_tool = WebSearchTool::new();
    let browser_tool = BrowserTool::new();

    // Vision tool: only wired when the configured provider supports vision
    // (anthropic, openai, gemini) AND we can resolve an API key. OAuth-only
    // users and non-vision providers silently skip it.
    let vision_api_key = resolve_api_key(config, &secret_store)
        .ok()
        .unwrap_or_default();
    let vision_tool = VisionTool::from_provider(
        &config.provider.name,
        vision_api_key,
        Some(config.provider.model.clone()),
    )
    .map(|t| t.with_sandbox(Arc::clone(&path_sandbox)));
    match &vision_tool {
        Some(_) => tracing::info!("Vision tool enabled ({})", config.provider.name),
        None => tracing::debug!(
            "Vision tool disabled: provider '{}' unsupported or no API key",
            config.provider.name
        ),
    }

    // Image generation tool: independent of the primary provider. Pulls an
    // OpenAI key from config (when provider is openai) or OPENAI_API_KEY env.
    // Users with Anthropic as primary can still generate images via their
    // OpenAI key.
    let img_config_key = resolve_api_key(config, &secret_store)
        .ok()
        .unwrap_or_default();
    let openai_key = ImageGenTool::resolve_openai_key(&config.provider.name, &img_config_key);
    let image_gen_tool = ImageGenTool::new_with_key(
        openai_key,
        image_output_dir(home_dir),
        None,
    );
    match &image_gen_tool {
        Some(_) => tracing::info!("Image generation tool enabled (OpenAI DALL-E 3)"),
        None => tracing::debug!("Image generation tool disabled: no OpenAI API key"),
    }

    // Code execution tool (python/node/bash subprocess, no sandbox). Always
    // wired — runners are checked at call time, not startup.
    let code_exec_tool = CodeExecTool::new(
        config.security.command_timeout_secs,
        home_dir.join("codeexec"),
    );
    tracing::info!("Code execution tool enabled (python/node/bash)");

    // Voice tools (transcription + TTS). Both use OpenAI; shared key
    // resolution with the image gen tool.
    let voice_key = voice_resolve_openai_key(&config.provider.name, &img_config_key);
    // Wire memory into the transcribe tool so the agent can opt into
    // FTS5-indexing transcripts via the new `index` param. Existing
    // transcription calls without `index=true` behave identically.
    let transcribe_memory: Arc<dyn Memory> = memory.clone();
    let transcribe_tool = TranscribeAudioTool::new_with_key(voice_key.clone(), None)
        .map(|t| {
            t.with_sandbox(Arc::clone(&path_sandbox))
                .with_memory(Arc::clone(&transcribe_memory))
        });
    let tts_tool = TextToSpeechTool::new_with_key(
        voice_key,
        default_tts_output_dir(home_dir),
        None,
        None,
    );
    match (&transcribe_tool, &tts_tool) {
        (Some(_), Some(_)) => tracing::info!("Voice tools enabled (Whisper + OpenAI TTS)"),
        _ => tracing::debug!("Voice tools disabled: no OpenAI API key"),
    }

    // Create CronTool with shared origin handle. The same handle is
    // reused by AskUserTool so it knows the (channel, chat_id) the
    // current turn came from.
    let cron_origin: Arc<Mutex<Option<CronOrigin>>> = Arc::new(Mutex::new(None));
    let cron_tool = CronTool::new(
        home_dir.join("cron_jobs.json"),
        Arc::clone(&cron_origin),
    );

    // Create prompt guard from config security settings.
    let guard_action = parse_guard_action(&config.security.prompt_guard_action);
    let prompt_guard = PromptGuard::new(guard_action, config.security.prompt_guard_sensitivity);

    // Set up collective intelligence layer.
    let collective_search: Option<Arc<CollectiveSearch>> = if config.collective.enabled {
        // Resolve collective API key. Fail-closed: if the configured value is
        // present but cannot be decrypted, treat the integration as
        // unavailable rather than leaking the encrypted blob to the
        // collective endpoint as if it were the plaintext key.
        let collective_api_key = if !config.collective.api_key.is_empty() {
            match secret_store.decrypt(&config.collective.api_key) {
                Ok(key) => key,
                Err(e) => {
                    tracing::error!(
                        "Failed to decrypt collective API key: {e}; disabling collective \
                         (set a fresh encrypted key or unset to use PLURUM_API_KEY env var)"
                    );
                    String::new()
                }
            }
        } else {
            std::env::var("PLURUM_API_KEY").unwrap_or_default()
        };

        let remote: Option<Arc<dyn CollectiveLayer>> = if !collective_api_key.is_empty() {
            let base_url = if config.collective.base_url.is_empty() {
                None
            } else {
                Some(config.collective.base_url.clone())
            };
            // Failing to build the Plurum reqwest client (broken TLS
            // roots, proxy config, etc.) used to panic the process at
            // startup. Treat it as "remote disabled" instead — local
            // collective still works.
            match PlurumlClient::new(collective_api_key, base_url) {
                Ok(client) => {
                    tracing::info!("Collective intelligence enabled (Plurum remote)");
                    Some(Arc::new(client))
                }
                Err(e) => {
                    tracing::error!(
                        "Failed to build Plurum client: {e}; running collective in local-only mode"
                    );
                    None
                }
            }
        } else {
            tracing::info!("Collective intelligence enabled (local only, no API key)");
            None
        };

        let cache = CollectiveCache::new(memory.clone());
        let search_guard = PromptGuard::new(
            parse_guard_action(&config.security.prompt_guard_action),
            config.security.prompt_guard_sensitivity,
        );
        let search = CollectiveSearch::new(
            memory.clone(),
            cache,
            remote,
            Some(search_guard),
        );
        Some(Arc::new(search))
    } else {
        tracing::debug!("Collective intelligence disabled");
        None
    };

    // Build Agent.
    let mut builder = AgentBuilder::new()
        .provider(Arc::clone(&provider))
        .memory(memory.clone())
        .tool(Box::new(shell_tool))
        .tool(Box::new(read_file_tool))
        .tool(Box::new(write_file_tool))
        .tool(Box::new(list_dir_tool))
        .tool(Box::new(web_fetch_tool))
        .tool(Box::new(web_search_tool))
        .tool(Box::new(browser_tool))
        .tool(Box::new(cron_tool))
        .tool(Box::new(MemoryStoreTool::new(memory.clone())))
        .tool(Box::new(MemoryRecallTool::new(memory.clone())))
        .tool(Box::new(MemoryForgetTool::new(memory.clone())))
        .tool(Box::new(TodoTool::new()));

    if let Some(vt) = vision_tool {
        builder = builder.tool(Box::new(vt));
    }
    if let Some(igt) = image_gen_tool {
        builder = builder.tool(Box::new(igt));
    }
    builder = builder.tool(Box::new(code_exec_tool));
    builder = builder.tool(Box::new(
        PdfReadTool::new(home_dir.join("pdf_cache")).with_sandbox(Arc::clone(&path_sandbox)),
    ));
    builder = builder.tool(Box::new(ScreenshotTool::new(default_screenshot_dir(home_dir))));
    builder = builder.tool(Box::new(HttpRequestTool::new()));
    builder = builder.tool(Box::new(WeatherTool::new()));
    builder = builder.tool(Box::new(
        ImageInfoTool::new(home_dir.join("image_cache")).with_sandbox(Arc::clone(&path_sandbox)),
    ));
    if let Some(claude_tool) = ClaudeCodeCliTool::detect() {
        tracing::info!("Claude Code CLI tool enabled");
        builder = builder.tool(Box::new(claude_tool));
    } else {
        tracing::debug!("Claude Code CLI tool disabled: `claude` binary not on PATH");
    }
    if let Some(t) = transcribe_tool {
        builder = builder.tool(Box::new(t));
    }
    if let Some(t) = tts_tool {
        builder = builder.tool(Box::new(t));
    }

    // Shared turn-context handles. These are returned from build_agent so
    // the gateway's inbound dispatch can keep them populated.
    let pending_replies = fennec::bus::PendingReplies::new();
    let chat_directory = fennec::bus::ChatDirectory::new();

    // Add send_message + ask_user tools when running in gateway mode
    // (channel map and turn-origin available).
    if let Some(ref ch_map) = channel_map {
        // Pull each enabled channel's home_chat_id from config so the
        // send_message tool can route default-target sends ("send to
        // telegram") without the LLM having to know the chat_id.
        let mut home_chats: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        if config.channels.telegram.enabled
            && !config.channels.telegram.home_chat_id.is_empty()
        {
            home_chats.insert(
                "telegram".to_string(),
                config.channels.telegram.home_chat_id.clone(),
            );
        }
        if config.channels.discord.enabled
            && !config.channels.discord.home_chat_id.is_empty()
        {
            home_chats.insert(
                "discord".to_string(),
                config.channels.discord.home_chat_id.clone(),
            );
        }
        if config.channels.slack.enabled
            && !config.channels.slack.home_chat_id.is_empty()
        {
            home_chats.insert(
                "slack".to_string(),
                config.channels.slack.home_chat_id.clone(),
            );
        }
        if config.channels.whatsapp.enabled
            && !config.channels.whatsapp.home_chat_id.is_empty()
        {
            home_chats.insert(
                "whatsapp".to_string(),
                config.channels.whatsapp.home_chat_id.clone(),
            );
        }
        if config.channels.email.enabled
            && !config.channels.email.home_chat_id.is_empty()
        {
            home_chats.insert(
                "email".to_string(),
                config.channels.email.home_chat_id.clone(),
            );
        }

        builder = builder.tool(Box::new(SendMessageTool::new(
            Arc::clone(ch_map),
            chat_directory.clone(),
            home_chats,
        )));

        builder = builder.tool(Box::new(AskUserTool::new(
            Arc::clone(ch_map),
            Arc::clone(&cron_origin),
            pending_replies.clone(),
        )));
    }

    // Add collective tools if enabled.
    if let Some(ref search) = collective_search {
        builder = builder.tool(Box::new(CollectiveSearchTool::new(Arc::clone(search))));

        // CollectiveReportTool needs a CollectiveLayer; create one based on config.
        if config.collective.publish_enabled {
            // Fail-closed on decrypt error: empty key → MockCollective fallback
            // below, so we never publish using the encrypted blob as plaintext.
            let collective_api_key = if !config.collective.api_key.is_empty() {
                match secret_store.decrypt(&config.collective.api_key) {
                    Ok(key) => key,
                    Err(e) => {
                        tracing::error!(
                            "Failed to decrypt collective API key for publish layer: {e}; \
                             falling back to MockCollective (publish disabled)"
                        );
                        String::new()
                    }
                }
            } else {
                std::env::var("PLURUM_API_KEY").unwrap_or_default()
            };

            let report_layer: Arc<dyn CollectiveLayer> = if !collective_api_key.is_empty() {
                let base_url = if config.collective.base_url.is_empty() {
                    None
                } else {
                    Some(config.collective.base_url.clone())
                };
                // Same fallback as the search-layer constructor above:
                // a Plurum reqwest-build failure now degrades to
                // MockCollective rather than panicking the process.
                match PlurumlClient::new(collective_api_key, base_url) {
                    Ok(client) => Arc::new(client) as Arc<dyn CollectiveLayer>,
                    Err(e) => {
                        tracing::error!(
                            "Failed to build Plurum publish client: {e}; using MockCollective"
                        );
                        Arc::new(MockCollective::new())
                    }
                }
            } else {
                Arc::new(MockCollective::new())
            };
            builder = builder.tool(Box::new(CollectiveGetExperienceTool::new(Arc::clone(&report_layer))));
            builder = builder.tool(Box::new(CollectiveReportTool::new(Arc::clone(&report_layer))));
            builder = builder.tool(Box::new(CollectivePublishTool::new(report_layer, Arc::clone(search))));
        }
    }

    // Wire collective search into agent.
    if let Some(search) = collective_search {
        builder = builder.collective(search);
    }

    // Skills directory: seed bundled skills (best-effort), then load
    // everything with full provenance and lifecycle state populated so
    // the curator and `skill_manage` can distinguish bundled / hub /
    // agent-created skills and respect pin/state flags.
    let skills_dir = home_dir.join("skills");
    match fennec::skills::sync::sync_bundled(&skills_dir) {
        Ok(counts) => tracing::info!(
            considered = counts.considered,
            installed = counts.installed,
            updated = counts.updated,
            skipped_customized = counts.skipped_customized,
            skipped_deleted = counts.skipped_deleted,
            up_to_date = counts.up_to_date,
            errors = counts.errors,
            "bundled skills synced",
        ),
        Err(e) => tracing::warn!(error = %e, "bundled skill sync failed; continuing"),
    }

    let usage_store = Arc::new(fennec::skills::UsageStore::open(&skills_dir));
    let bundled_manifest = fennec::skills::BundledManifest::load(&skills_dir);
    let hub_lock = fennec::skills::HubLock::load(&skills_dir);
    let loaded_skills = SkillsLoader::load_with_provenance(
        &skills_dir,
        Some(&bundled_manifest),
        Some(&hub_lock),
        Some(&usage_store),
    )
    .context("loading skills directory")?;
    let available_skills: Vec<Skill> = SkillsLoader::filter_available(&loaded_skills)
        .into_iter()
        .filter(|s| s.state != fennec::skills::SkillState::Archived)
        .cloned()
        .collect();
    tracing::info!(
        total = loaded_skills.len(),
        available = available_skills.len(),
        dir = %skills_dir.display(),
        "skills loaded",
    );
    let skills_prompt = SkillsLoader::build_skills_prompt(&available_skills);
    let guard_config =
        fennec::skills::guard::GuardConfig::from_toml(&config.skills.guard);
    builder = builder
        .skills_prompt(skills_prompt)
        .tool(Box::new(SkillsTool::with_usage(
            available_skills,
            Arc::clone(&usage_store),
        )))
        .tool(Box::new(
            fennec::tools::SkillManageTool::new(
                skills_dir.clone(),
                Arc::clone(&usage_store),
            )
            .with_guard(guard_config),
        ));

    // Wire DelegateTool so the agent can spawn read-only sub-agents for
    // bounded research / investigation tasks. Toolkit is intentionally
    // read-only: anything that writes files, spends money, or touches
    // live systems stays with the main agent.
    let delegate_subagent_tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool::new()),
        Arc::new(ListDirTool::new()),
        Arc::new(WebFetchTool::new()),
        Arc::new(WebSearchTool::new()),
    ];
    let mut delegate_tool = DelegateTool::new(
        Arc::clone(&provider),
        memory.clone(),
        delegate_subagent_tools,
    );
    if let Some(ref cb) = callbacks {
        // Wire the parent's callback bridge so each spawned
        // sub-agent's lifecycle events surface in the TUI's
        // /agents overlay.
        delegate_tool = delegate_tool.with_callbacks(Arc::clone(cb));
    }
    if let Some(ref reg) = delegation_registry {
        // Wire the shared delegation registry so pause / caps /
        // active map / interrupt flag are honoured by every
        // sub-agent spawn through this DelegateTool.
        delegate_tool = delegate_tool.with_registry(reg.clone());
    }
    builder = builder.tool(Box::new(delegate_tool));

    // Plugin system: bundled + WASM plugin discovery.
    let mut plugin_registry = fennec::plugins::PluginRegistry::new();
    if let Err(e) = plugin_registry.load_bundled(&config.plugins.enabled) {
        tracing::error!(
            "Plugin registry failed to load bundled: {e}; continuing without bundled plugins"
        );
    }
    let wasm_resources = fennec::plugins::WasmHostResources {
        path_sandbox: Arc::clone(&path_sandbox),
        memory: memory.clone(),
        http_client: fennec::tools::http::shared_client(),
        rt_handle: tokio::runtime::Handle::current(),
        settings: config.plugins.settings.clone(),
        bus: plugin_bus,
    };
    let plugins_root = home_dir.join("plugins");
    if let Err(e) = plugin_registry.load_wasm(
        &plugins_root,
        &config.plugins.enabled,
        wasm_resources,
    ) {
        tracing::error!(
            "Plugin registry failed to load WASM plugins: {e}; continuing without WASM plugins"
        );
    }
    let runtime = plugin_registry.into_runtime(&config.memory.provider);
    for plugin_tool in runtime.tools {
        builder = builder.tool(plugin_tool);
    }
    builder = builder.hooks(Arc::new(runtime.hooks));
    builder = builder.memory_manager(Arc::new(runtime.memory_manager));
    builder = builder.home_dir(home_dir.to_path_buf());

    let mut configured_builder = builder
        .identity_name(&config.identity.name)
        .identity_persona(&config.identity.persona)
        .max_tool_iterations(config.agent.max_tool_iterations as usize)
        .max_tokens(config.provider.max_tokens as usize)
        .temperature(config.provider.temperature)
        .memory_context_limit(config.memory.context_limit)
        .half_life_days(config.memory.half_life_days)
        .prompt_guard(prompt_guard)
        .auxiliary_client(Arc::clone(&aux_client));
    if let Some(handle) = callbacks {
        configured_builder = configured_builder.callbacks(handle);
    }
    if let Some(flag) = interrupt_flag {
        // Wire the shared interrupt flag so `/busy interrupt`
        // can cooperatively cancel an in-flight turn.
        configured_builder = configured_builder.interrupt_flag(flag);
    }
    let mut agent = configured_builder.build().context("building agent")?;

    // Apply any persisted /tools disable list from config.
    if !config.tools.disabled.is_empty() {
        agent.set_disabled_tools(config.tools.disabled.iter().cloned());
    }

    Ok((agent, memory, cron_origin, pending_replies, chat_directory, provider))
}

async fn run_agent(
    config: FennecConfig,
    home_dir: std::path::PathBuf,
    message: Option<String>,
    model: Option<String>,
) -> Result<()> {
    // CLI mode doesn't run the gateway, so the channel-aware handles are
    // unused here. We still need to bind them to suppress unused-let
    // warnings without losing the build_agent return-shape.
    let (mut agent, memory, _cron_origin, _pending_replies, _chat_directory, _provider) =
        build_agent(&config, &home_dir, model, None, None).await?;

    if let Some(msg) = message {
        // Single-shot mode.
        let response = agent.turn(&msg).await?;
        println!("{response}");
    } else {
        // Interactive mode.
        println!("Fennec v{} — interactive mode", env!("CARGO_PKG_VERSION"));
        println!("Type /quit or /exit to leave.\n");

        let channel = CliChannel::new();
        let (tx, mut rx) = tokio::sync::mpsc::channel(32);

        // Spawn the channel listener in the background.
        let listen_handle = tokio::spawn(async move {
            if let Err(e) = channel.listen(tx).await {
                eprintln!("Channel error: {e}");
            }
        });

        print!("You: ");
        use std::io::Write;
        let _ = std::io::stdout().flush();

        let consolidation_enabled = config.memory.consolidation_enabled;

        // Process messages.
        while let Some(msg) = rx.recv().await {
            if msg.content.is_empty() {
                continue;
            }

            match agent.turn(&msg.content).await {
                Ok(response) => {
                    let send_msg = SendMessage::new(&response, "user");
                    // Print via channel send (writes to stdout).
                    println!("Fennec: {}", send_msg.content);

                    // Fire-and-forget consolidation hint.
                    if consolidation_enabled {
                        tracing::debug!(
                            "consolidation: would consolidate after turn (provider sharing needed for full implementation)"
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                }
            }
        }

        // Clean exit -- export soul snapshot.
        let snapshot_path = home_dir.join("MEMORY_SNAPSHOT.md");
        match snapshot::export_snapshot(memory.as_ref(), &snapshot_path).await {
            Ok(count) => {
                tracing::info!(count, path = %snapshot_path.display(), "exported soul snapshot on exit");
            }
            Err(e) => {
                tracing::warn!("failed to export soul snapshot on exit: {e}");
            }
        }

        listen_handle.abort();
    }

    Ok(())
}

/// Launch the terminal UI (`fennec agent --tui`).
///
/// Builds a real agent with `TuiBridge` callbacks so streaming
/// text deltas, tool starts/completes, status updates, and turn
/// boundaries fan out into the TUI's app state. The renderer
/// thread reads `App` for each frame; the agent loop runs on a
/// tokio task and produces events into an mpsc channel; a drain
/// task on the same runtime applies events to `App`.
async fn run_tui(
    config: FennecConfig,
    home_dir: std::path::PathBuf,
    model_override: Option<String>,
    log_ring: Option<fennec::tui::log_ring::LogRing>,
) -> Result<()> {
    use fennec::sessions::store::SessionStore;
    use fennec::tui::app::{App, ChatLine, SessionRow};
    use fennec::tui::callbacks::TuiBridge;

    // Open (or create) the session store. Failure is non-fatal:
    // the TUI keeps running, /resume just reports "no store".
    let session_db = home_dir.join("sessions.db");
    let session_store_handle = {
        let path = session_db.clone();
        match tokio::task::spawn_blocking(move || SessionStore::new(&path)).await {
            Ok(Ok(s)) => Some(std::sync::Arc::new(s)),
            Ok(Err(e)) => {
                tracing::warn!("session store init failed: {e}; sessions list will be empty");
                None
            }
            Err(_) => None,
        }
    };
    let prior_sessions = if let Some(ref store) = session_store_handle {
        store.list_sessions(20).await.unwrap_or_default()
    } else {
        Vec::new()
    };

    // Provision a row for the current TUI session so /title and
    // per-turn message persistence have somewhere to write. If
    // the store isn't available we leave `current_session_id`
    // unset and the persistence hook becomes a no-op.
    let current_session_id = if let Some(ref store) = session_store_handle {
        match store.create_session("cli").await {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::warn!("create session failed: {e}; persistence disabled this run");
                None
            }
        }
    } else {
        None
    };

    let mut app = App::new();
    app.current_session_id = current_session_id.clone();
    app.skills_dir = Some(home_dir.join("skills"));
    // Hydrate persisted TUI display toggles so user choices
    // survive across runs.
    app.compact_mode = config.tui.compact;
    if let Some(mode) = fennec::tui::app::DetailsMode::parse(&config.tui.details) {
        app.details_mode = mode;
    }
    // Per-section overrides set via `/details <section> <mode>`.
    for (section, raw) in &config.tui.details_sections {
        if let Some(mode) = fennec::tui::app::DetailsMode::parse(raw) {
            app.details_section_overrides
                .insert(section.clone(), mode);
        }
    }
    if let Some(p) = fennec::tui::app::StatusBarPosition::parse(&config.tui.statusbar) {
        app.statusbar_position = p;
    }
    if let Some(s) = fennec::tui::app::IndicatorStyle::parse(&config.tui.indicator) {
        app.indicator_style = s;
    }
    if let Some(v) = fennec::tui::app::VerbosityMode::parse(&config.tui.verbose) {
        app.verbosity = v;
    }
    if let Some(b) = fennec::tui::app::BusyMode::parse(&config.tui.busy) {
        app.busy_mode = b;
    }
    app.personality_name = config.tui.personality.clone();
    app.skin_name = config.tui.skin.clone();
    // Resolve the skin name on startup so the renderer reads the
    // user's chosen palette from the first frame onwards.
    match fennec::tui::skin::Skin::resolve(&config.tui.skin, &home_dir) {
        Ok(s) => app.skin = s,
        Err(e) => {
            tracing::warn!("skin resolve failed; falling back to fennec-warm: {e}");
        }
    }
    // Current TUI session pinned to the top.
    app.sessions.push(SessionRow {
        code: "$ ".into(),
        who: "local".into(),
        subject: "current session".into(),
        count: "0".into(),
        has_unread: false,
    });
    // Historical sessions from the store, transformed into the
    // sessions-panel row shape.
    for rec in prior_sessions.iter().take(20) {
        let (code, who) = channel_label(&rec.channel);
        let subject = rec
            .summary
            .clone()
            .unwrap_or_else(|| short_id(&rec.id));
        app.sessions.push(SessionRow {
            code,
            who,
            subject,
            count: rec
                .ended_at
                .as_deref()
                .map(|_| "—".to_string())
                .unwrap_or_else(|| "·".to_string()),
            has_unread: false,
        });
    }
    app.selected_session = 0;

    // Real channels list from config — every enabled channel is
    // surfaced with a status reflecting whether the gateway is
    // running it (in TUI mode the gateway isn't started, so all
    // channels except cli show as 'not running').
    app.channels = build_channels_panel(&config);

    app.chat = vec![ChatLine::System {
        time: chrono::Local::now().format("%H:%M:%S").to_string(),
        body: format!(
            "session resumed · model {} · {} prior sessions · ready",
            if config.provider.model.is_empty() {
                "default"
            } else {
                &config.provider.model
            },
            prior_sessions.len()
        ),
    }];
    // Shared delegation state — pause flag + caps + active map.
    // Held by both `App` (so /agents commands can read it) and
    // `DelegateTool` (so spawns honour the gate).
    let delegation_registry = fennec::agent::DelegationRegistry::default();
    app.delegation_registry = Some(delegation_registry.clone());
    if let Some(ring) = log_ring {
        app.log_ring = ring;
    }
    // Main-agent interrupt flag shared with the submit loop so
    // `/busy interrupt` can cancel an in-flight turn at its next
    // tool-iteration boundary. Agent::turn clears the flag at
    // the top of each turn so the user's next prompt runs.
    let main_interrupt_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    app.main_interrupt_flag = Some(std::sync::Arc::clone(&main_interrupt_flag));
    let app = std::sync::Arc::new(parking_lot::Mutex::new(app));

    // Build the agent with the TUI bridge wired in as callbacks.
    let (bridge, mut event_rx) = TuiBridge::new(app.clone());
    let bridge_handle: fennec::agent::callbacks::CallbacksHandle =
        std::sync::Arc::new(bridge);
    let (agent, _memory, _cron_origin, _pending_replies, _chat_directory, _provider) =
        build_agent_with_callbacks(
            &config,
            &home_dir,
            model_override,
            None,
            Some(bridge_handle),
            None,
            Some(delegation_registry),
            Some(std::sync::Arc::clone(&main_interrupt_flag)),
        )
        .await?;
    let agent = std::sync::Arc::new(tokio::sync::Mutex::new(agent));

    // Drain task: consume agent events, mutate app state. Runs on
    // the same runtime as the agent itself; the renderer reads
    // app state under its own parking_lot mutex on the main
    // thread.
    let drain_app = app.clone();
    tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            apply_tui_event(&drain_app, ev);
        }
    });

    // Submit-handler task: watches the input editor for new
    // history entries (each Enter records one) and routes them
    // either to the slash-command registry or to a fresh agent
    // turn. Polling is intentional — the TUI's render loop runs
    // synchronously on the main thread so we can't share a
    // tokio mpsc with it without extra plumbing.
    let registry = std::sync::Arc::new(fennec::tui::commands::CommandRegistry::with_builtins());
    let submit_app = app.clone();
    let submit_agent = agent.clone();
    let submit_store = session_store_handle.clone();
    let submit_config = config.clone();
    let submit_home = home_dir.clone();
    let submit_handle = tokio::spawn(async move {
        let mut last_submitted: Option<String> = None;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let pending = {
                let guard = submit_app.lock();
                if guard.should_quit {
                    return;
                }
                guard.input.history.front().cloned()
            };
            let prompt = match pending {
                Some(p) if last_submitted.as_ref() != Some(&p) => p,
                _ => continue,
            };
            last_submitted = Some(prompt.clone());

            // Slash command? Run it through the registry under
            // the app lock briefly, then handle any AgentAction
            // outcome with the agent lock released.
            if let Some((name, args)) = fennec::tui::commands::parse(&prompt) {
                let outcome = {
                    let mut guard = submit_app.lock();
                    registry.dispatch(name, args, &mut guard).unwrap_or_else(|e| {
                        fennec::tui::commands::CommandOutcome::Status(format!("error: {e}"))
                    })
                };
                handle_command_outcome(
                    outcome,
                    &submit_app,
                    &submit_agent,
                    submit_store.as_ref(),
                    &submit_config,
                    &submit_home,
                )
                .await;
                continue;
            }

            // Busy-mode gate: if a turn is already in progress
            // (the agent mutex is held), `/busy` decides what
            // Enter should do.
            //   - Queue: defer the prompt to `queued_input`; it
            //     fires after the current turn completes via the
            //     existing queue-drain hook.
            //   - Steer: route the prompt through `/steer`'s
            //     injection path so it lands as user-guidance
            //     after the next tool batch.
            //   - Interrupt (default): set the agent's interrupt
            //     flag so the current turn bails at its next
            //     iteration boundary; queue the new prompt to
            //     fire when the lock frees.
            use fennec::tui::app::BusyMode;
            if submit_agent.try_lock().is_err() {
                let busy_mode = submit_app.lock().busy_mode;
                match busy_mode {
                    BusyMode::Queue => {
                        let mut g = submit_app.lock();
                        g.queued_input.push_back(prompt.clone());
                        let pending = g.queued_input.len();
                        g.set_status(format!(
                            "queued (agent busy) · {pending} pending"
                        ));
                        continue;
                    }
                    BusyMode::Steer => {
                        handle_steer(prompt.clone(), &submit_app, &submit_agent).await;
                        continue;
                    }
                    BusyMode::Interrupt => {
                        if let Some(flag) = submit_app.lock().main_interrupt_flag.clone() {
                            flag.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                        // Re-queue ourselves so the next loop
                        // iteration runs the new prompt once the
                        // interrupted turn releases the lock.
                        let mut g = submit_app.lock();
                        g.queued_input.push_back(prompt.clone());
                        g.set_status("interrupting current turn".to_string());
                        continue;
                    }
                }
            }

            // Plain text — run as an agent turn (streaming, so
            // text deltas + tool events surface live in the TUI).
            // After the turn drains, replay the freshly-appended
            // history slice into the SessionStore so /resume can
            // restore it later.
            let session_id = submit_app.lock().current_session_id.clone();
            let mut agent_guard = submit_agent.lock().await;
            let history_before = agent_guard.history_len();
            let _ = agent_guard.turn_streaming(&prompt).await;
            let appended: Vec<_> = agent_guard
                .history_slice(history_before)
                .iter()
                .cloned()
                .collect();
            drop(agent_guard);
            if let (Some(store), Some(sid)) = (submit_store.as_ref(), session_id) {
                for msg in &appended {
                    let role = msg.role.as_str();
                    let content = msg.content.clone().unwrap_or_default();
                    let tool_calls_json = msg.tool_calls.as_ref().and_then(|tcs| {
                        // Persist as JSON so resume can reconstruct
                        // the structured ToolCall list. Skip on
                        // empty array — no point recording "this
                        // assistant turn requested zero tools."
                        if tcs.is_empty() {
                            None
                        } else {
                            serde_json::to_string(tcs).ok()
                        }
                    });
                    // Empty content is fine for assistant messages
                    // that ONLY contain tool_calls (no preamble
                    // text), and for tool-result rows that
                    // genuinely return empty output. Skip only
                    // when there's literally nothing to record:
                    // no content AND no tool structure.
                    if content.is_empty() && tool_calls_json.is_none() && msg.tool_call_id.is_none()
                    {
                        continue;
                    }
                    if let Err(e) = store
                        .add_message_full(
                            &sid,
                            role,
                            &content,
                            tool_calls_json.as_deref(),
                            msg.tool_call_id.as_deref(),
                        )
                        .await
                    {
                        tracing::warn!("session store add_message failed: {e}");
                    }
                }
                // Record a checkpoint at the end of each turn so
                // `/rollback` can step back to "state right after
                // this exchange." Preview text is the most recent
                // assistant reply for legibility in `/rollback list`.
                let assistant_preview = appended
                    .iter()
                    .rev()
                    .find(|m| m.role.as_str() == "assistant")
                    .and_then(|m| m.content.clone())
                    .unwrap_or_else(|| "<no reply>".to_string());
                if let Err(e) = store
                    .record_checkpoint(&sid, &assistant_preview)
                    .await
                {
                    tracing::warn!("checkpoint record failed: {e}");
                }
            }

            // Drain one /queue entry into the input history so the
            // next loop iteration picks it up. One per turn keeps
            // the agent's pace reasonable; bursting all queued
            // messages at once would run them back-to-back without
            // letting the user inspect the in-between outputs.
            //
            // Reset `last_submitted` after draining so the queued
            // prompt re-fires even when it happens to equal the
            // most recently submitted prompt — without this the
            // dedup-by-last_submitted gate above silently skips
            // /busy queue and /busy interrupt re-fires.
            let drained = {
                let mut g = submit_app.lock();
                g.queued_input.pop_front()
            };
            if let Some(next) = drained {
                submit_app.lock().input.history.push_front(next);
                last_submitted = None;
            }
        }
    });

    // Voice transcription polling task: when /voice off has
    // queued a WAV file, transcribe it via the user-configured
    // OpenAI Whisper key (same key the agent's
    // TranscribeAudioTool uses) and deliver the text back into
    // the TUI's input box on the next tick.
    let voice_app = app.clone();
    let voice_config = config.clone();
    let voice_home = home_dir.clone();
    let voice_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            let pending = {
                let guard = voice_app.lock();
                if guard.should_quit {
                    return;
                }
                guard.voice.take_pending_wav()
            };
            let Some(path) = pending else { continue };
            // Build a transcription tool on demand. The OpenAI
            // key is resolved the same way the agent's tool does
            // (config.api_key first, then OPENAI_API_KEY env).
            let key = fennec::tools::voice_tool::resolve_openai_key(
                &voice_config.provider.name,
                &voice_config.provider.api_key,
            );
            if key.is_empty() {
                voice_app.lock().voice.deliver_error(
                    "OPENAI_API_KEY not set; cannot transcribe".into(),
                );
                continue;
            }
            let _ = voice_home.clone(); // reserved for future cache-dir use
            let tool =
                match fennec::tools::voice_tool::TranscribeAudioTool::new_with_key(
                    key,
                    None,
                ) {
                    Some(t) => t,
                    None => {
                        voice_app.lock().voice.deliver_error(
                            "transcription tool init failed".into(),
                        );
                        continue;
                    }
                };
            // The tool expects a JSON arg payload with `path`.
            let args = serde_json::json!({ "path": path.to_string_lossy() });
            use fennec::tools::traits::Tool;
            match tool.execute(args).await {
                Ok(result) => {
                    if result.success {
                        voice_app.lock().voice.deliver_transcription(result.output);
                    } else {
                        voice_app.lock().voice.deliver_error(format!(
                            "transcribe: {}",
                            result.error.unwrap_or(result.output)
                        ));
                    }
                }
                Err(e) => voice_app
                    .lock()
                    .voice
                    .deliver_error(format!("transcribe: {e}")),
            }
        }
    });

    // Run the renderer (blocks until the user quits).
    let result = fennec::tui::run(app.clone());

    // Mark the active session ended (if persistence is on) so
    // the sessions list shows a clean ended_at instead of a
    // dangling "in progress" row. Failure here is non-fatal —
    // the chat data is already on disk from the per-turn
    // persistence loop above.
    let exit_session_id = app.lock().current_session_id.clone();
    if let (Some(store), Some(sid)) = (session_store_handle.as_ref(), exit_session_id) {
        if let Err(e) = store.end_session(&sid, None).await {
            tracing::warn!("end_session failed for {sid}: {e}");
        }
    }

    // Cleanly stop background tasks.
    {
        let mut guard = app.lock();
        guard.should_quit = true;
    }
    submit_handle.abort();
    voice_handle.abort();

    result
}

/// Build the TUI's CHANNELS panel from the running config.
/// The gateway isn't started in `--tui` mode so non-CLI channels
/// show as "not running" — when fennec gateway is active in a
/// separate process the user would still see them as "available"
/// rather than "connected" (live channel-manager wiring is out
/// of scope for F1-1; that's a richer integration that needs
/// a shared bus across the gateway and TUI processes, which
/// belongs to F2's dashboard work).
fn build_channels_panel(config: &FennecConfig) -> Vec<fennec::tui::app::ChannelState> {
    use fennec::tui::app::{ChannelConnState, ChannelState};
    let mut out = vec![ChannelState {
        code: "$ ".into(),
        name: "cli".into(),
        state: ChannelConnState::Attached,
        detail: "this session".into(),
    }];
    let cs = &config.channels;
    if cs.telegram.enabled {
        out.push(ChannelState {
            code: "TG".into(),
            name: "telegram".into(),
            state: ChannelConnState::Idle,
            detail: "configured".into(),
        });
    }
    if cs.discord.enabled {
        out.push(ChannelState {
            code: "DC".into(),
            name: "discord".into(),
            state: ChannelConnState::Idle,
            detail: "configured".into(),
        });
    }
    if cs.slack.enabled {
        out.push(ChannelState {
            code: "SL".into(),
            name: "slack".into(),
            state: ChannelConnState::Idle,
            detail: "configured".into(),
        });
    }
    if cs.whatsapp.enabled {
        out.push(ChannelState {
            code: "WA".into(),
            name: "whatsapp".into(),
            state: ChannelConnState::Idle,
            detail: "configured".into(),
        });
    }
    if cs.email.enabled {
        out.push(ChannelState {
            code: "@ ".into(),
            name: "email".into(),
            state: ChannelConnState::Idle,
            detail: format!("imap.{}", first_dot_segment(&cs.email.imap_host)),
        });
    }
    out
}

/// Map a channel name (as stored in `SessionRecord.channel`) to
/// the (code, who) pair the sessions panel renders.
fn channel_label(channel: &str) -> (String, String) {
    match channel {
        "cli" | "" => ("$ ".into(), "local".into()),
        "telegram" => ("TG".into(), "telegram".into()),
        "discord" => ("DC".into(), "discord".into()),
        "slack" => ("SL".into(), "slack".into()),
        "signal" => ("SG".into(), "signal".into()),
        "matrix" => ("MX".into(), "matrix".into()),
        "whatsapp" => ("WA".into(), "whatsapp".into()),
        "email" => ("@ ".into(), "email".into()),
        other => ("? ".into(), other.to_string()),
    }
}

fn short_id(s: &str) -> String {
    s.chars().take(8).collect()
}

fn first_dot_segment(s: &str) -> String {
    s.split('.').next().unwrap_or(s).to_string()
}

/// Apply a `CommandOutcome` to the app state and (if needed)
/// dispatch the queued `AgentAction` against the agent. Lives
/// here rather than inside the registry because executing
/// agent operations needs both the parking_lot app mutex and
/// the tokio agent mutex; the registry is sync.
async fn handle_command_outcome(
    outcome: fennec::tui::commands::CommandOutcome,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    session_store: Option<&std::sync::Arc<fennec::sessions::SessionStore>>,
    config: &FennecConfig,
    home_dir: &std::path::Path,
) {
    use fennec::tui::commands::{AgentAction, CommandOutcome};
    match outcome {
        CommandOutcome::Quit => {
            app.lock().should_quit = true;
        }
        CommandOutcome::Text(body) => {
            let mut guard = app.lock();
            guard.chat.push(fennec::tui::app::ChatLine::System {
                time: chrono::Local::now().format("%H:%M:%S").to_string(),
                body,
            });
        }
        CommandOutcome::Status(msg) => {
            app.lock().set_status(msg);
        }
        CommandOutcome::Unknown(name) => {
            app.lock().set_status(format!("unknown command: /{name}"));
        }
        CommandOutcome::NotImplemented(name) => {
            app.lock()
                .set_status(format!("/{name}: not yet implemented"));
        }
        CommandOutcome::Agent(action) => match action {
            AgentAction::Clear => {
                let mut guard = agent.lock().await;
                guard.clear_history();
            }
            AgentAction::Undo => {
                handle_undo(app, agent).await;
            }
            AgentAction::Retry => {
                handle_retry(app, agent).await;
            }
            AgentAction::Steer(note) => {
                handle_steer(note, app, agent).await;
            }
            AgentAction::Run(prompt) => {
                let mut guard = agent.lock().await;
                let _ = guard.turn(&prompt).await;
            }
            AgentAction::ShowUsage => {
                let snapshot = {
                    let guard = agent.lock().await;
                    guard.token_usage()
                };
                let body = fennec::tui::usage_panel::render(&snapshot);
                let mut guard = app.lock();
                guard.chat.push(fennec::tui::app::ChatLine::System {
                    time: chrono::Local::now().format("%H:%M:%S").to_string(),
                    body,
                });
            }
            AgentAction::SessionTitle(payload) => {
                handle_session_title(payload, app, session_store).await;
            }
            AgentAction::SessionResume(target) => {
                handle_session_resume(target, app, agent, session_store).await;
            }
            AgentAction::SwitchModel(payload) => {
                handle_switch_model(payload, app, agent, config, home_dir).await;
            }
            AgentAction::ToolsToggle(payload) => {
                handle_tools_toggle(payload, app, agent, config, home_dir).await;
            }
            AgentAction::ReloadEnv => {
                handle_reload_env(app, home_dir);
            }
            AgentAction::ReloadMcp => {
                handle_reload_mcp(app);
            }
            AgentAction::AttachImage(path) => {
                handle_attach_image(path, app, agent).await;
            }
            AgentAction::PasteClipboardImage => {
                handle_paste_clipboard(app, agent, home_dir).await;
            }
            AgentAction::CopyAssistantMessage(n) => {
                handle_copy_assistant(n, app);
            }
            AgentAction::PersistTuiSettings => {
                handle_persist_tui_settings(app, config, home_dir);
            }
            AgentAction::SetThinkingLevel(level) => {
                let mut g = agent.lock().await;
                g.set_thinking_level(level);
                drop(g);
                let line = format!("thinking level set: {:?}", level);
                let mut app_g = app.lock();
                app_g.set_status(line);
            }
            AgentAction::SetPersona(persona) => {
                let mut g = agent.lock().await;
                g.set_persona(persona);
                drop(g);
                handle_persist_tui_settings(app, config, home_dir);
            }
            AgentAction::BranchSession(title) => {
                handle_branch_session(title, app, agent, session_store).await;
            }
            AgentAction::ReloadSkills => {
                handle_reload_skills(app, agent, home_dir).await;
            }
            AgentAction::CompressHistory(topic) => {
                handle_compress_history(topic, app, agent).await;
            }
            AgentAction::RollbackList => {
                handle_rollback_list(app, session_store).await;
            }
            AgentAction::RollbackTo(hash) => {
                handle_rollback_to(hash, app, agent, session_store).await;
            }
            AgentAction::ApplyUserSkin(name) => {
                handle_apply_user_skin(name, app, config, home_dir);
            }
        },
    }
}

/// `/title` worker — reads or writes the current session's
/// title via the SessionStore. `payload = None` reads, `Some`
/// writes. Mirrors Hermes' `session.title` RPC: read returns
/// "title: <name>" or "no title set"; write returns "session
/// title set: <name>" with an optional "(queued while session
/// initializes)" suffix when the row hasn't been created yet.
async fn handle_session_title(
    payload: Option<String>,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    store: Option<&std::sync::Arc<fennec::sessions::SessionStore>>,
) {
    let (sid, current_title) = {
        let guard = app.lock();
        (
            guard.current_session_id.clone(),
            guard.current_session_title.clone(),
        )
    };
    let body = match (sid.as_deref(), store, payload) {
        (None, _, _) | (_, None, _) => "no active session".to_string(),
        (Some(_), Some(_), None) => match current_title {
            Some(t) if !t.is_empty() => format!("title: {t}"),
            _ => "no title set".to_string(),
        },
        (Some(sid), Some(store), Some(new_title)) => {
            match store.set_session_title(sid, &new_title).await {
                Ok(0) => format!(
                    "session title set: {new_title} (queued while session initializes)"
                ),
                Ok(_) => {
                    app.lock().current_session_title = if new_title.trim().is_empty() {
                        None
                    } else {
                        Some(new_title.trim().to_string())
                    };
                    format!("session title set: {}", new_title.trim())
                }
                Err(e) => format!("title write failed: {e}"),
            }
        }
    };
    let mut guard = app.lock();
    guard.chat.push(fennec::tui::app::ChatLine::System {
        time: chrono::Local::now().format("%H:%M:%S").to_string(),
        body,
    });
}

/// `/resume` worker — looks up the target session by id, then
/// by exact title, fetches its message history, and replays
/// it into the agent. Mirrors Hermes' `session.resume`
/// (`tui_gateway/server.py:2180-2221`): reset agent, load
/// messages as conversation, re-emit a system line so the
/// chat shows what was loaded. Empty store / unknown id
/// produce informative error lines.
async fn handle_session_resume(
    target: String,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    store: Option<&std::sync::Arc<fennec::sessions::SessionStore>>,
) {
    let Some(store) = store else {
        let mut guard = app.lock();
        guard.chat.push(fennec::tui::app::ChatLine::System {
            time: chrono::Local::now().format("%H:%M:%S").to_string(),
            body: "session store unavailable — cannot resume".into(),
        });
        return;
    };
    let record = match store.get_session(&target).await {
        Ok(Some(r)) => r,
        Ok(None) => {
            let mut guard = app.lock();
            guard.chat.push(fennec::tui::app::ChatLine::System {
                time: chrono::Local::now().format("%H:%M:%S").to_string(),
                body: format!("session not found: {target}"),
            });
            return;
        }
        Err(e) => {
            let mut guard = app.lock();
            guard.chat.push(fennec::tui::app::ChatLine::System {
                time: chrono::Local::now().format("%H:%M:%S").to_string(),
                body: format!("session lookup failed: {e}"),
            });
            return;
        }
    };
    let messages = match store.get_session_messages(&record.id).await {
        Ok(m) => m,
        Err(e) => {
            let mut guard = app.lock();
            guard.chat.push(fennec::tui::app::ChatLine::System {
                time: chrono::Local::now().format("%H:%M:%S").to_string(),
                body: format!("loading messages failed: {e}"),
            });
            return;
        }
    };

    // Translate stored messages back into full ChatMessage
    // form, including the tool-call structure. Assistant rows
    // with persisted tool_calls JSON re-attach the parsed
    // Vec<ToolCall>; tool-result rows preserve their
    // tool_call_id so the provider can match them back to the
    // originating call. Rows that fail to deserialise (corrupt
    // JSON from a hand-edited db) fall back to text-only
    // replay rather than dropping the message entirely.
    use fennec::providers::traits::{ChatMessage, ToolCall};
    let chat_messages: Vec<ChatMessage> = messages
        .iter()
        .filter_map(|m| match m.role.as_str() {
            "user" => Some(ChatMessage::user(&m.content)),
            "assistant" => {
                let mut msg = ChatMessage::assistant(&m.content);
                if let Some(json) = m.tool_calls.as_deref() {
                    match serde_json::from_str::<Vec<ToolCall>>(json) {
                        Ok(tcs) if !tcs.is_empty() => msg.tool_calls = Some(tcs),
                        Ok(_) => {} // empty array — leave as text-only
                        Err(e) => tracing::warn!(
                            "could not parse persisted tool_calls JSON for resumed assistant message: {e}"
                        ),
                    }
                }
                Some(msg)
            }
            "tool" => {
                // Tool result rows. Preserve the tool_call_id so
                // the provider serializer can match the result
                // back to the assistant's tool_calls entry — a
                // mismatched id makes the next turn's request
                // bail on the API side.
                let id = m.tool_call_id.as_deref().unwrap_or("");
                Some(ChatMessage::tool_result(id, &m.content))
            }
            "system" => Some(ChatMessage::system(&m.content)),
            _ => None,
        })
        .collect();
    let count = chat_messages.len();
    {
        let mut g = agent.lock().await;
        g.replace_history(chat_messages);
    }

    // Mutate app state to reflect the new session, then push a
    // confirmation system message into the chat scrollback.
    let mut guard = app.lock();
    guard.current_session_id = Some(record.id.clone());
    guard.current_session_title = record.title.clone();
    let title_label = record
        .title
        .clone()
        .unwrap_or_else(|| short_id(&record.id));
    guard.chat.push(fennec::tui::app::ChatLine::System {
        time: chrono::Local::now().format("%H:%M:%S").to_string(),
        body: format!("resumed: {title_label} ({count} messages)"),
    });
}

/// `/model` worker — read or write the active provider's
/// model. `payload = None` renders an inline panel showing
/// the current model + the snapshot of known models so the
/// user can pick one. `Some(name)` rebuilds the provider with
/// the new model, swaps it on the live agent, and confirms.
///
/// Mid-turn swap is rejected (matching Hermes'
/// `_apply_model_switch` at server.py:1067-1145, which raises
/// "session busy"). Detection here is a `try_lock` on the
/// agent's tokio mutex — if it's held, a turn is in flight.
async fn handle_switch_model(
    payload: Option<String>,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    config: &FennecConfig,
    home_dir: &std::path::Path,
) {
    use fennec::agent::pricing;

    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let push = |body: String, app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>| {
        let mut g = app.lock();
        g.chat.push(fennec::tui::app::ChatLine::System {
            time: now.clone(),
            body,
        });
    };

    match payload {
        None => {
            // Read path: snapshot the active model and render the
            // known-models list inline. No agent mutation.
            let current = {
                let guard = agent.lock().await;
                guard.provider().model().to_string()
            };
            let current_label = if current.is_empty() {
                "(unknown)".to_string()
            } else {
                current
            };
            let known: Vec<&'static str> = pricing::known_models();
            let mut body = format!("Active model: {current_label}\nKnown models:\n");
            for m in &known {
                body.push_str(&format!("  · {m}\n"));
            }
            body.push_str("\nUse /model <name> to switch live (rejected mid-turn).");
            push(body.trim_end().to_string(), app);
        }
        Some(new_model) => {
            // Write path: reject if the agent is mid-turn, then
            // rebuild + swap the provider. The new provider Arc
            // also surfaces the new model in /usage's pricing
            // lookup automatically since `provider().model()` is
            // the source of truth.
            let mut agent_lock = match agent.try_lock() {
                Ok(g) => g,
                Err(_) => {
                    push(
                        "session busy — finish the current turn before /model".to_string(),
                        app,
                    );
                    return;
                }
            };
            let provider = match resolve_provider_with_model(config, home_dir, &new_model) {
                Ok(p) => p,
                Err(e) => {
                    drop(agent_lock);
                    push(format!("model swap failed: {e}"), app);
                    return;
                }
            };
            agent_lock.set_provider(provider);
            drop(agent_lock);
            // Persist the new model to disk so the change survives
            // a restart, mirroring Hermes' `_persist_model_switch`.
            // A failure here is non-fatal — the live agent already
            // has the swap applied; we just warn the user.
            let mut persisted = config.clone();
            persisted.provider.model = new_model.clone();
            let config_path = home_dir.join("config.toml");
            let persist_msg = match persisted.save(&config_path) {
                Ok(()) => format!("model → {new_model}"),
                Err(e) => format!(
                    "model → {new_model} (in-memory only; config.toml save failed: {e})"
                ),
            };
            push(persist_msg, app);
        }
    }
}

/// `/tools` worker — list every registered tool with its
/// enabled/disabled status, or toggle the listed names. After
/// any toggle, persist the new disabled set to
/// `~/.fennec/config.toml` and clear the agent's chat history
/// (matching Hermes' `_reset_session_agent` behavior on
/// tools.configure: previously-fired tool_calls in history
/// would otherwise reference tools the model can no longer
/// invoke). Mid-turn toggles are rejected.
async fn handle_tools_toggle(
    payload: Option<(bool, Vec<String>)>,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    config: &FennecConfig,
    home_dir: &std::path::Path,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let push = |body: String, app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>| {
        let mut g = app.lock();
        g.chat.push(fennec::tui::app::ChatLine::System {
            time: now.clone(),
            body,
        });
    };

    match payload {
        None => {
            // List path — read-only, no agent mutation.
            let guard = agent.lock().await;
            let mut names = guard.tool_names();
            names.sort();
            let mut body = String::from("Tools\n");
            for n in &names {
                let mark = if guard.is_tool_enabled(n) { "✓" } else { "✗" };
                body.push_str(&format!("  {mark} {n}\n"));
            }
            body.push_str("\nUse /tools enable|disable <name> to toggle.");
            drop(guard);
            push(body.trim_end().to_string(), app);
        }
        Some((enable, requested)) => {
            // Write path — try-lock, mutate, persist, clear history.
            let mut agent_lock = match agent.try_lock() {
                Ok(g) => g,
                Err(_) => {
                    push(
                        "session busy — finish the current turn before /tools".to_string(),
                        app,
                    );
                    return;
                }
            };
            let mut changed: Vec<String> = Vec::new();
            let mut unknown: Vec<String> = Vec::new();
            for n in &requested {
                if agent_lock.set_tool_enabled(n, enable) {
                    changed.push(n.clone());
                } else if !agent_lock
                    .tool_names()
                    .iter()
                    .any(|t| t == n)
                {
                    unknown.push(n.clone());
                }
                // If the tool exists but is already in the requested
                // state, set_tool_enabled returns false — no error,
                // just no-op (matches Hermes' silent idempotence).
            }
            // Capture the new disabled set for persistence.
            let new_disabled = agent_lock.disabled_tool_names();
            // Tool change must clear chat history (Hermes' behavior).
            if !changed.is_empty() {
                agent_lock.clear_history();
            }
            drop(agent_lock);

            // Persist. Best-effort.
            let mut persisted = config.clone();
            persisted.tools.disabled = new_disabled;
            let path = home_dir.join("config.toml");
            if let Err(e) = persisted.save(&path) {
                tracing::warn!("config save failed after /tools toggle: {e}");
            }

            // Reset visible chat history too so the user sees the
            // reset Hermes also performs.
            if !changed.is_empty() {
                let mut g = app.lock();
                g.chat.clear();
                g.in_flight_bot_idx = None;
                g.live_tool = None;
            }

            let action = if enable { "enabled" } else { "disabled" };
            let mut body = String::new();
            if !changed.is_empty() {
                body.push_str(&format!("{action} {}\n", changed.join(", ")));
                body.push_str("session reset. new tool configuration is active.");
            } else {
                body.push_str(&format!("nothing to {action} (already in that state)"));
            }
            if !unknown.is_empty() {
                body.push_str(&format!("\nunknown tools: {}", unknown.join(", ")));
            }
            push(body, app);
        }
    }
}

/// `/reload` worker — re-read `~/.fennec/.env` into the
/// running process. Newly-set keys take effect on the next
/// provider call without a restart. Already-built provider
/// Arcs keep their cached credentials, same as Hermes (which
/// docstrings this same caveat at server.py:4147-4165).
fn handle_reload_env(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    home_dir: &std::path::Path,
) {
    let env_path = home_dir.join(".env");
    let body = match fennec::auth_env::reload_env_file(&env_path) {
        Ok(0) if !env_path.exists() => {
            format!("no .env file at {}", env_path.display())
        }
        Ok(n) => {
            let noun = if n == 1 { "var" } else { "vars" };
            format!("reloaded .env ({n} {noun} updated)")
        }
        Err(e) => format!("reload failed: {e}"),
    };
    let mut g = app.lock();
    g.chat.push(fennec::tui::app::ChatLine::System {
        time: chrono::Local::now().format("%H:%M:%S").to_string(),
        body,
    });
}

/// Persist the App's TUI display state (compact, details) into
/// `~/.fennec/config.toml` so toggles survive a restart. Called
/// by `/compact` and `/details` after they mutate the live
/// state. Failure is logged but doesn't surface to the user —
/// the in-memory toggle already took effect.
fn handle_persist_tui_settings(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    config: &FennecConfig,
    home_dir: &std::path::Path,
) {
    let snap = {
        let g = app.lock();
        let sections: std::collections::HashMap<String, String> = g
            .details_section_overrides
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().to_string()))
            .collect();
        (
            g.compact_mode,
            g.details_mode.as_str().to_string(),
            sections,
            g.statusbar_position.as_str().to_string(),
            g.indicator_style.as_str().to_string(),
            g.verbosity.as_str().to_string(),
            g.busy_mode.as_str().to_string(),
            g.personality_name.clone(),
            g.skin_name.clone(),
        )
    };
    let mut persisted = config.clone();
    persisted.tui.compact = snap.0;
    persisted.tui.details = snap.1;
    persisted.tui.details_sections = snap.2;
    persisted.tui.statusbar = snap.3;
    persisted.tui.indicator = snap.4;
    persisted.tui.verbose = snap.5;
    persisted.tui.busy = snap.6;
    persisted.tui.personality = snap.7;
    persisted.tui.skin = snap.8;
    let path = home_dir.join("config.toml");
    if let Err(e) = persisted.save(&path) {
        tracing::warn!("config save failed after TUI settings change: {e}");
    }
}

/// `/branch` worker — clone the current session's history into a
/// fresh row in the SessionStore. The new session inherits every
/// stored message (including tool calls / results) so the user
/// can continue from the same context but explore a different
/// path. The current session is kept active; switching to the
/// new branch happens via `/resume <id>`.
async fn handle_branch_session(
    title: Option<String>,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    _agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    store: Option<&std::sync::Arc<fennec::sessions::SessionStore>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let store = match store {
        Some(s) => s.clone(),
        None => {
            let mut g = app.lock();
            g.chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: "/branch: no session store available".into(),
            });
            return;
        }
    };
    let parent_id = match app.lock().current_session_id.clone() {
        Some(id) => id,
        None => {
            let mut g = app.lock();
            g.chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: "/branch: no active session to fork".into(),
            });
            return;
        }
    };
    let messages = match store.get_session_messages(&parent_id).await {
        Ok(m) => m,
        Err(e) => {
            let mut g = app.lock();
            g.chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/branch: failed to read parent history: {e}"),
            });
            return;
        }
    };
    let new_id = match store.create_session("cli").await {
        Ok(id) => id,
        Err(e) => {
            let mut g = app.lock();
            g.chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/branch: failed to create new session: {e}"),
            });
            return;
        }
    };
    // Copy each stored message into the new session, preserving
    // role + content (tool-call structure isn't surfaced through
    // add_message; that's a known limitation — assistant tool
    // calls round-trip as plain text).
    for m in &messages {
        if let Err(e) = store
            .add_message(&new_id, &m.role, &m.content)
            .await
        {
            tracing::warn!("/branch: failed copying message: {e}");
            break;
        }
    }
    if let Some(title) = title.as_deref().filter(|t| !t.trim().is_empty()) {
        let _ = store.set_session_title(&new_id, title).await;
    }
    let body = match title.as_deref().filter(|t| !t.trim().is_empty()) {
        Some(t) => format!(
            "branched → {t}  ·  /resume {} to switch",
            short_session_id(&new_id)
        ),
        None => format!(
            "branched → {} (untitled)  ·  /resume to switch",
            short_session_id(&new_id)
        ),
    };
    let mut g = app.lock();
    g.chat.push(fennec::tui::app::ChatLine::System { time: now, body });
}

fn short_session_id(id: &str) -> String {
    if id.len() > 8 {
        id[..8].to_string()
    } else {
        id.to_string()
    }
}

/// `/skin <name>` worker for user-defined skins — built-ins are
/// applied directly in the command handler. Loads
/// `~/.fennec/skins/<name>.toml`, applies on success, persists.
fn handle_apply_user_skin(
    name: String,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    config: &FennecConfig,
    home_dir: &std::path::Path,
) {
    use fennec::tui::skin::Skin as SkinTy;
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    match SkinTy::resolve(&name, home_dir) {
        Ok(loaded) => {
            {
                let mut g = app.lock();
                g.skin = loaded;
                g.skin_name = name.clone();
            }
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/skin: loaded user skin '{name}'"),
            });
            handle_persist_tui_settings(app, config, home_dir);
        }
        Err(e) => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/skin: {e}"),
            });
        }
    }
}

/// `/compress` worker — call `Agent::compress_history` and
/// surface (messages removed, summary preview) into chat.
async fn handle_compress_history(
    topic: Option<String>,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let result = {
        let mut g = agent.lock().await;
        g.compress_history(topic).await
    };
    let body = match result {
        Ok((0, _)) => "/compress: nothing to compress yet".to_string(),
        Ok((removed, preview)) => format!(
            "/compress: summarised {removed} message{} · {preview}",
            if removed == 1 { "" } else { "s" }
        ),
        Err(e) => format!("/compress failed: {e}"),
    };
    app.lock().chat.push(fennec::tui::app::ChatLine::System { time: now, body });
}

/// `/rollback list` worker — fetch checkpoints from the store
/// and render them as a system-message table.
async fn handle_rollback_list(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    store: Option<&std::sync::Arc<fennec::sessions::SessionStore>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let store = match store {
        Some(s) => s.clone(),
        None => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: "/rollback: no session store available".into(),
            });
            return;
        }
    };
    let sid = match app.lock().current_session_id.clone() {
        Some(id) => id,
        None => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: "/rollback: no active session".into(),
            });
            return;
        }
    };
    let entries = match store.list_checkpoints(&sid).await {
        Ok(v) => v,
        Err(e) => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/rollback: list failed: {e}"),
            });
            return;
        }
    };
    let body = if entries.is_empty() {
        "/rollback: no checkpoints in this session yet".to_string()
    } else {
        let mut s = format!(
            "{} checkpoint{}:\n",
            entries.len(),
            if entries.len() == 1 { "" } else { "s" }
        );
        for c in &entries {
            let preview: String = c.preview.chars().take(60).collect();
            let suffix = if c.preview.chars().count() > 60 { "…" } else { "" };
            s.push_str(&format!(
                "  {hash}  ({n} msg)  {preview}{suffix}\n",
                hash = c.hash,
                n = c.message_count
            ));
        }
        s.trim_end().to_string()
    };
    app.lock().chat.push(fennec::tui::app::ChatLine::System { time: now, body });
}

/// `/rollback <hash>` worker — restore the session to the
/// checkpoint's `message_count`. Truncates session_messages on
/// disk and pops the in-memory agent history past that index.
async fn handle_rollback_to(
    hash: String,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    store: Option<&std::sync::Arc<fennec::sessions::SessionStore>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let store = match store {
        Some(s) => s.clone(),
        None => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: "/rollback: no session store available".into(),
            });
            return;
        }
    };
    let sid = match app.lock().current_session_id.clone() {
        Some(id) => id,
        None => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: "/rollback: no active session".into(),
            });
            return;
        }
    };
    let checkpoint = match store.get_checkpoint(&sid, &hash).await {
        Ok(Some(c)) => c,
        Ok(None) => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/rollback: no checkpoint '{hash}' in this session"),
            });
            return;
        }
        Err(e) => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/rollback: lookup failed: {e}"),
            });
            return;
        }
    };
    let removed = match store
        .restore_to_message_count(&sid, checkpoint.message_count)
        .await
    {
        Ok(n) => n,
        Err(e) => {
            app.lock().chat.push(fennec::tui::app::ChatLine::System {
                time: now,
                body: format!("/rollback: restore failed: {e}"),
            });
            return;
        }
    };
    // Truncate the agent's in-memory history to the same count.
    // We can't trim "the first N chat messages" because the
    // agent's history also includes the system message + tool
    // messages that don't appear in session_messages. The
    // simplest faithful behaviour: clear the agent history and
    // let the next turn rebuild context from scratch. The chat
    // pane truncates separately below.
    {
        let mut g = agent.lock().await;
        g.clear_history();
    }
    // Trim chat scrollback to roughly match. `checkpoint.message_count`
    // counts both user AND assistant rows in the SQL table (one
    // per send-or-receive). The chat pane mixes those plus tool
    // calls / system lines, so we count User+Bot lines as
    // "messages" and stop once we've passed that many. System +
    // tool-result lines stay (they're context for the kept
    // messages); anything visually past the cut is dropped.
    {
        let mut g = app.lock();
        truncate_chat_to_message_count(&mut g, checkpoint.message_count);
    }
    app.lock().chat.push(fennec::tui::app::ChatLine::System {
        time: now,
        body: format!(
            "/rollback → {hash}: {removed} message{} removed · {n} preserved",
            if removed == 1 { "" } else { "s" },
            n = checkpoint.message_count,
        ),
    });
}

/// Trim chat scrollback so it has at most `keep_count` chat
/// messages (counting `User` + `Bot` lines, matching the SQL
/// table's row count). Used by `/rollback`. System / tool lines
/// before the cut point stay (they're context for the kept
/// messages); everything visually past the cut is dropped.
fn truncate_chat_to_message_count(app: &mut fennec::tui::App, keep_count: usize) {
    use fennec::tui::app::ChatLine;
    let mut msg_seen = 0usize;
    let mut cut_at = app.chat.len();
    for (i, line) in app.chat.iter().enumerate() {
        let is_msg = matches!(line, ChatLine::User { .. } | ChatLine::Bot { .. });
        if is_msg {
            msg_seen += 1;
            if msg_seen > keep_count {
                cut_at = i;
                break;
            }
        }
    }
    app.chat.truncate(cut_at);
}

/// `/reload-skills` worker — re-run `SkillsLoader::load_from_directory`
/// against `~/.fennec/skills/` and rebuild the live agent's
/// `skills_prompt`. The agent's cached `system_prompt` is also
/// cleared so the new skills land on the next turn.
async fn handle_reload_skills(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    home_dir: &std::path::Path,
) {
    use fennec::skills::SkillsLoader;
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let skills_dir = home_dir.join("skills");
    let loaded = if skills_dir.exists() {
        match tokio::task::spawn_blocking({
            let dir = skills_dir.clone();
            move || SkillsLoader::load_from_directory(&dir)
        })
        .await
        {
            Ok(Ok(skills)) => skills,
            Ok(Err(e)) => {
                let mut g = app.lock();
                g.chat.push(fennec::tui::app::ChatLine::System {
                    time: now,
                    body: format!("/reload-skills: load failed: {e}"),
                });
                return;
            }
            Err(e) => {
                let mut g = app.lock();
                g.chat.push(fennec::tui::app::ChatLine::System {
                    time: now,
                    body: format!("/reload-skills: task panicked: {e}"),
                });
                return;
            }
        }
    } else {
        Vec::new()
    };
    let count = loaded.len();
    let prompt = SkillsLoader::build_skills_prompt(&loaded);
    {
        let mut g = agent.lock().await;
        g.set_skills_prompt(prompt);
    }
    let mut guard = app.lock();
    guard.chat.push(fennec::tui::app::ChatLine::System {
        time: now,
        body: format!(
            "/reload-skills: {count} skill{} reloaded · effective next turn",
            if count == 1 { "" } else { "s" }
        ),
    });
}

/// `/steer` worker — queue text on the agent so it's injected
/// after the next tool batch with the "User guidance:" marker.
/// If a turn isn't currently running (try_lock succeeds), the
/// queued text still lands on the next tool result of whatever
/// turn fires next, mirroring Hermes' "no active turn — queued
/// for next" fallback (`core.ts:527-563`).
///
/// `/steer` itself returns immediately — actual injection
/// happens inside `Agent::turn` / `turn_streaming`'s tool loop
/// via `apply_pending_steer_to_tool_results`.
async fn handle_steer(
    note: String,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let trimmed = note.trim().to_string();
    let body = if trimmed.is_empty() {
        "usage: /steer <text>".to_string()
    } else {
        // The agent lock might be held by an in-flight turn —
        // that's exactly the case where /steer is most useful,
        // and Agent::steer is fine to call against a held
        // agent because the queue is locked separately. But
        // we still need *some* reference; use try_lock first to
        // avoid blocking the submit task on a long-running turn.
        let preview = truncate_for_status(&trimmed);
        match agent.try_lock() {
            Ok(mut g) => {
                if g.steer(&trimmed) {
                    format!("steer queued — arrives after next tool call: {preview}")
                } else {
                    "steer rejected".to_string()
                }
            }
            Err(_) => {
                // Turn is mid-flight. We still want the steer to
                // land — fall back to a non-blocking lock that
                // suspends the submit task briefly. Hermes' RPC
                // path takes the agent lock unconditionally, same
                // shape.
                let mut g = agent.lock().await;
                if g.steer(&trimmed) {
                    format!("steer queued — arrives after next tool call: {preview}")
                } else {
                    "steer rejected".to_string()
                }
            }
        }
    };
    let mut g = app.lock();
    g.chat.push(fennec::tui::app::ChatLine::System {
        time: now,
        body,
    });
}

/// `/undo` worker — drop the last user / assistant exchange
/// from the agent's history. Mid-turn rejection via try_lock
/// matches Hermes' "session busy" guard at server.py:2424-2449.
/// The chat-side cleanup (popping ChatLines) was done by the
/// command handler before we got here.
async fn handle_undo(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let body = {
        let mut agent_lock = match agent.try_lock() {
            Ok(g) => g,
            Err(_) => {
                let mut g = app.lock();
                g.chat.push(fennec::tui::app::ChatLine::System {
                    time: now,
                    body: "session busy — finish the current turn before /undo".into(),
                });
                return;
            }
        };
        match agent_lock.pop_last_turn() {
            Some((n, _user_text)) => {
                let plural = if n == 1 { "message" } else { "messages" };
                format!("undid {n} {plural}")
            }
            None => "nothing to undo".to_string(),
        }
    };
    let mut g = app.lock();
    g.chat.push(fennec::tui::app::ChatLine::System {
        time: now,
        body,
    });
}

/// `/retry` worker — pop the last user / assistant exchange,
/// then re-submit the user message as a fresh streaming turn.
/// Mid-turn rejection via try_lock. If there's no prior user
/// message to retry, surfaces "nothing to retry" rather than
/// silently no-op (matches Hermes' core.ts:587-610).
async fn handle_retry(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let push = |body: String, app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>| {
        let mut g = app.lock();
        g.chat.push(fennec::tui::app::ChatLine::System {
            time: now.clone(),
            body,
        });
    };

    // Pop + capture the user text under the agent lock, then
    // drop the lock so turn_streaming can take it back for the
    // re-run.
    let user_text = {
        let mut agent_lock = match agent.try_lock() {
            Ok(g) => g,
            Err(_) => {
                push(
                    "session busy — finish the current turn before /retry".into(),
                    app,
                );
                return;
            }
        };
        match agent_lock.pop_last_turn() {
            Some((_n, text)) => text,
            None => {
                drop(agent_lock);
                push("nothing to retry".into(), app);
                return;
            }
        }
    };

    push(format!("retrying: {}", truncate_for_status(&user_text)), app);
    // Re-submit. We hold the agent lock for the full streaming
    // turn so concurrent commands wait — same as the regular
    // submit path.
    let mut agent_lock = agent.lock().await;
    let _ = agent_lock.turn_streaming(&user_text).await;
}

fn truncate_for_status(s: &str) -> String {
    const MAX: usize = 60;
    let mut iter = s.chars();
    let head: String = iter.by_ref().take(MAX).collect();
    if iter.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

/// `/paste` worker — read an image from the OS clipboard, save
/// as PNG under `~/.fennec/clipboard/clip-YYYYMMDD-HHMMSS.png`,
/// and queue it on the agent the same way `/image` does. Pure
/// text paste isn't handled here because the user's terminal
/// already gets that for free via Cmd-V / Ctrl-Shift-V into
/// the input box.
async fn handle_paste_clipboard(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
    home_dir: &std::path::Path,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let push = |body: String, app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>| {
        let mut g = app.lock();
        g.chat.push(fennec::tui::app::ChatLine::System {
            time: now.clone(),
            body,
        });
    };

    // arboard's Clipboard::new is sync + can block briefly on
    // Linux X11. Run on a blocking pool so the submit task's
    // tokio runtime stays responsive.
    let read_result =
        tokio::task::spawn_blocking(fennec::tui::clipboard::read_image_rgba).await;
    let img = match read_result {
        Ok(Ok(img)) => img,
        Ok(Err(e)) => {
            push(format!("paste failed: {e}"), app);
            return;
        }
        Err(e) => {
            push(format!("paste failed (join error): {e}"), app);
            return;
        }
    };

    let target_dir = home_dir.join("clipboard");
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let target = target_dir.join(format!("clip-{stamp}.png"));
    if let Err(e) = img.write_png(&target) {
        push(format!("paste failed: {e}"), app);
        return;
    }

    // Reuse the /image attach path so dimensions, token estimate,
    // and message echoing all match the keyboard /image command.
    let body = {
        let mut guard = agent.lock().await;
        match guard.attach_image(&target) {
            Ok(att) => {
                let queued = guard.pending_attachment_count();
                let dims = match (att.width, att.height) {
                    (Some(w), Some(h)) => format!("{w}×{h}"),
                    _ => "?".to_string(),
                };
                let tokens = att
                    .token_estimate
                    .map(|t| format!("{t} tokens"))
                    .unwrap_or_else(|| "size unknown".to_string());
                format!(
                    "📎 pasted: {name} · {dims} · ~{tokens} · {queued} pending",
                    name = att.display_name,
                )
            }
            Err(e) => format!("paste attach failed: {e}"),
        }
    };
    push(body, app);
}

/// `/copy` worker — copy the Nth assistant message (1-indexed,
/// or the last when `None`) to the OS clipboard via arboard.
/// Falls back to OSC52 escape on failure (SSH sessions and
/// sandboxed terminals). Empty chat → "nothing to copy".
fn handle_copy_assistant(
    n: Option<usize>,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let body = {
        let bots: Vec<String> = {
            let guard = app.lock();
            guard
                .chat
                .iter()
                .filter_map(|l| match l {
                    fennec::tui::app::ChatLine::Bot { body, .. } => Some(body.clone()),
                    _ => None,
                })
                .collect()
        };
        if bots.is_empty() {
            "nothing to copy — no assistant messages yet".to_string()
        } else {
            let idx = match n {
                None => bots.len() - 1,
                Some(v) => v.saturating_sub(1).min(bots.len() - 1),
            };
            let target = bots[idx].clone();
            match fennec::tui::clipboard::write_with_fallback(&target) {
                fennec::tui::clipboard::CopyResult::Native => {
                    format!(
                        "copied message {}/{} to clipboard",
                        idx + 1,
                        bots.len()
                    )
                }
                fennec::tui::clipboard::CopyResult::Osc52 => {
                    format!(
                        "sent OSC52 copy sequence (terminal must honor it) — message {}/{}",
                        idx + 1,
                        bots.len()
                    )
                }
                fennec::tui::clipboard::CopyResult::Failed => {
                    "copy failed: native clipboard + OSC52 both unavailable".to_string()
                }
            }
        }
    };
    let mut g = app.lock();
    g.chat.push(fennec::tui::app::ChatLine::System {
        time: now,
        body,
    });
}

/// `/image` worker — load + base64-encode the file, queue it
/// on the agent for the next user turn, and echo metadata back
/// to the chat (filename, dimensions, token estimate). Mirrors
/// Hermes' image.attach RPC (`tui_gateway/server.py:3361-3401`).
async fn handle_attach_image(
    path: std::path::PathBuf,
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    agent: &std::sync::Arc<tokio::sync::Mutex<fennec::agent::Agent>>,
) {
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let body = {
        let mut guard = agent.lock().await;
        match guard.attach_image(&path) {
            Ok(att) => {
                let queued = guard.pending_attachment_count();
                let dims = match (att.width, att.height) {
                    (Some(w), Some(h)) => format!("{w}×{h}"),
                    _ => "?".to_string(),
                };
                let tokens = att
                    .token_estimate
                    .map(|t| format!("{t} tokens"))
                    .unwrap_or_else(|| "size unknown".to_string());
                format!(
                    "📎 attached: {name} · {dims} · ~{tokens} · {queued} pending",
                    name = att.display_name,
                )
            }
            Err(e) => format!("attach failed: {e}"),
        }
    };
    let mut g = app.lock();
    g.chat.push(fennec::tui::app::ChatLine::System {
        time: now,
        body,
    });
}

/// `/reload_mcp` worker — Hermes calls
/// `shutdown_mcp_servers` + `discover_mcp_tools` against the
/// active session's MCP clients. Fennec's agent doesn't
/// currently boot MCP clients (the `mcp` module exists but
/// isn't wired into `build_agent_with_callbacks`), so the
/// honest behavior is to surface that state rather than fake
/// a reload. When MCP wiring lands, swap this body for a real
/// rescan.
fn handle_reload_mcp(app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>) {
    let body = "no MCP servers loaded in this build — MCP client wiring \
                attaches in the upcoming MCP-integration PR; /reload_mcp \
                will rescan once that's in place"
        .to_string();
    let mut g = app.lock();
    g.chat.push(fennec::tui::app::ChatLine::System {
        time: chrono::Local::now().format("%H:%M:%S").to_string(),
        body,
    });
}

/// Apply a single `TuiEvent` to the app state. Called from the
/// drain task; runs under the parking_lot mutex briefly.
fn apply_tui_event(
    app: &std::sync::Arc<parking_lot::Mutex<fennec::tui::App>>,
    ev: fennec::tui::callbacks::TuiEvent,
) {
    use fennec::tui::app::{ChatLine, LiveTool, ToolHistoryEntry};
    use fennec::tui::callbacks::TuiEvent;
    let now = chrono::Local::now().format("%H:%M:%S").to_string();
    let mut guard = app.lock();
    match ev {
        TuiEvent::TurnStart(prompt) => {
            guard.chat.push(ChatLine::User {
                time: now,
                body: prompt,
            });
            guard.finalize_bot_message();
            guard.finalize_reasoning_block();
        }
        TuiEvent::TurnComplete(summary) => {
            // If streaming was active, the in-flight bot message
            // already contains the full text — just close it. If
            // streaming wasn't used (e.g. the non-streaming
            // `turn()` path), the bot reply hasn't been pushed
            // yet, so push it now.
            match guard.in_flight_bot_idx {
                Some(_) => {
                    guard.finalize_bot_message();
                }
                None if !summary.is_empty() => {
                    guard.chat.push(ChatLine::Bot {
                        time: now,
                        body: summary,
                    });
                }
                _ => {}
            }
            guard.finalize_reasoning_block();
            guard.live_tool = None;
        }
        TuiEvent::TextDelta(delta) => {
            // Bot's text reply starting — close any in-flight
            // reasoning block's live cursor since thinking is
            // logically done.
            guard.finalize_reasoning_block();
            guard.append_bot_delta(&delta);
        }
        TuiEvent::ReasoningDelta(delta) => {
            guard.append_reasoning_delta(&delta);
        }
        TuiEvent::ToolStart(s) => {
            // A tool call breaks the streaming continuity — close
            // the in-flight bot message so the next text delta
            // (after the tool result) starts a fresh message.
            guard.finalize_bot_message();
            let started = std::time::Instant::now();
            guard.chat.push(ChatLine::ToolCall {
                call: format!("{}({})", s.name, s.preview),
            });
            // Inline running indicator under the tool-call line.
            // Replaced with ToolResult on completion so the
            // scrollback ends up as call → result, with the spinner
            // visible only while the tool is in flight.
            guard.chat.push(fennec::tui::app::ChatLine::ToolRunning {
                label: "running…".into(),
                started_at: started,
            });
            guard.live_tool = Some(LiveTool {
                name: s.name,
                args_preview: s.preview,
                started_at: started,
                progress: None,
            });
        }
        TuiEvent::ToolProgress(p) => {
            if let Some(ref mut lt) = guard.live_tool {
                lt.args_preview = p.preview;
            }
        }
        TuiEvent::ToolComplete(c) => {
            let summary = c
                .summary
                .clone()
                .unwrap_or_else(|| "(done)".to_string());
            // Drop the most recent inline running indicator before
            // pushing the result so the chat reads call → result.
            if let Some(idx) = guard
                .chat
                .iter()
                .rposition(|l| matches!(l, fennec::tui::app::ChatLine::ToolRunning { .. }))
            {
                guard.chat.remove(idx);
            }
            guard.chat.push(ChatLine::ToolResult { summary });
            guard.recent_tools.insert(
                0,
                ToolHistoryEntry {
                    ok: c.error.is_none(),
                    name: c.name,
                    note: format!("{}ms", c.duration_ms),
                },
            );
            guard.recent_tools.truncate(8);
            guard.live_tool = None;
        }
        TuiEvent::Status(msg) => {
            guard.set_status(msg);
        }
        TuiEvent::ApprovalRequest { request, resp_tx } => {
            fennec::tui::callbacks::install_approval_modal(&mut guard, request, resp_tx);
        }
        TuiEvent::ClarifyRequest { request, resp_tx } => {
            fennec::tui::callbacks::install_clarify_modal(&mut guard, request, resp_tx);
        }
        TuiEvent::SecretRequest { request, resp_tx } => {
            fennec::tui::callbacks::install_secret_modal(&mut guard, request, resp_tx);
        }
        TuiEvent::SubagentSpawn(spawn) => {
            // When a fresh root spawn arrives and we'd been
            // viewing a history snapshot, snap back to live so
            // the user sees the new turn's tree.
            if spawn.parent_id.is_none() && guard.agents_history_index > 0 {
                guard.agents_history_index = 0;
                guard.agents_cursor = None;
            }
            // Default the overlay cursor to the first root we
            // see so /agents has something selected when the
            // user opens it. Subsequent spawns leave the cursor
            // alone.
            if guard.agents_cursor.is_none() {
                guard.agents_cursor = Some(spawn.id.clone());
            }
            guard.spawn_tree.on_spawn(spawn);
        }
        TuiEvent::SubagentStart(id) => {
            guard.spawn_tree.on_start(&id);
        }
        TuiEvent::SubagentText { id, delta: _ } => {
            // Sub-agent text doesn't append to the main chat
            // scrollback (avoids duplicate output); the spawn-tree
            // node accumulates it as `output` on completion. For
            // now we only track that the sub-agent emitted text
            // (no per-token detail panel yet).
            let _ = id;
        }
        TuiEvent::SubagentThinking { id, delta } => {
            guard.spawn_tree.on_thinking(&id, delta);
        }
        TuiEvent::SubagentTool { id, start } => {
            guard.spawn_tree.on_tool(&id, start);
        }
        TuiEvent::SubagentProgress { id, note } => {
            guard.spawn_tree.on_progress(&id, note);
        }
        TuiEvent::SubagentComplete(complete) => {
            guard.spawn_tree.on_complete(complete);
            // When every node has reached a terminal status, snapshot
            // the live tree onto history and reset for the next spawn
            // round. Auto-promote-on-settle behavior.
            if guard.spawn_tree.is_settled() {
                let settled = std::mem::take(&mut guard.spawn_tree);
                guard.spawn_history.push(settled);
                guard.agents_cursor = None;
            }
        }
    }
}

async fn run_gateway(
    config: FennecConfig,
    home_dir: std::path::PathBuf,
    host_override: Option<String>,
    port_override: Option<u16>,
) -> Result<()> {
    // 1. Create MessageBus and channel map handle.
    let (bus, mut receiver) = MessageBus::new(256);
    let channel_map = new_channel_map();

    // 2. Build Agent (pass channel map so ask_user tool can reach channels).
    let (agent, _memory, cron_origin, pending_replies, chat_directory, provider) =
        build_agent(
            &config,
            &home_dir,
            None,
            Some(channel_map.clone()),
            // Plugins running under the gateway get the bus so
            // their `channel-send` host import works. The CLI
            // path passes None above.
            Some(bus.clone()),
        )
        .await?;
    let agent = Arc::new(tokio::sync::Mutex::new(agent));

    // 3. Build channel list from config.
    //
    // Each channel's secret fields (Telegram bot token, Discord bot token,
    // Slack bot/app tokens, WhatsApp access/verify/app_secret, Email IMAP
    // and SMTP passwords) is routed through `SecretStore::decrypt` so that
    // `enc2:`-prefixed values from `fennec encrypt` round-trip to plaintext
    // before being handed to the channel constructor — matching the
    // `provider.api_key` and `collective.api_key` paths.
    let secret_store =
        SecretStore::new(home_dir.clone()).context("creating secret store for channel tokens")?;
    let mut channels: Vec<Arc<dyn Channel>> = Vec::new();

    let ch_config = &config.channels;

    if ch_config.telegram.enabled && !ch_config.telegram.token.is_empty() {
        if let Some(token) = decrypt_channel_secret(
            &secret_store,
            &ch_config.telegram.token,
            "channels.telegram.token",
        )
        .filter(|t| !t.is_empty())
        {
            let ch = fennec::channels::TelegramChannel::new(
                token,
                ch_config.telegram.allowed_users.clone(),
            );
            channels.push(Arc::new(ch));
            tracing::info!("Telegram channel enabled");
        }
    }

    if ch_config.discord.enabled && !ch_config.discord.token.is_empty() {
        if let Some(token) = decrypt_channel_secret(
            &secret_store,
            &ch_config.discord.token,
            "channels.discord.token",
        )
        .filter(|t| !t.is_empty())
        {
            let ch = fennec::channels::DiscordChannel::new(
                token,
                ch_config.discord.allowed_users.clone(),
            );
            channels.push(Arc::new(ch));
            tracing::info!("Discord channel enabled");
        }
    }

    if ch_config.slack.enabled
        && !ch_config.slack.bot_token.is_empty()
        && !ch_config.slack.app_token.is_empty()
    {
        let bot_token = decrypt_channel_secret(
            &secret_store,
            &ch_config.slack.bot_token,
            "channels.slack.bot_token",
        );
        let app_token = decrypt_channel_secret(
            &secret_store,
            &ch_config.slack.app_token,
            "channels.slack.app_token",
        );
        if let (Some(bot), Some(app)) = (bot_token, app_token) {
            if !bot.is_empty() && !app.is_empty() {
                let ch = fennec::channels::SlackChannel::new(
                    bot,
                    app,
                    ch_config.slack.allowed_users.clone(),
                );
                channels.push(Arc::new(ch));
                tracing::info!("Slack channel enabled");
            }
        }
    }

    if ch_config.whatsapp.enabled && !ch_config.whatsapp.access_token.is_empty() {
        let access_token = decrypt_channel_secret(
            &secret_store,
            &ch_config.whatsapp.access_token,
            "channels.whatsapp.access_token",
        );
        let verify_token = decrypt_channel_secret(
            &secret_store,
            &ch_config.whatsapp.verify_token,
            "channels.whatsapp.verify_token",
        );
        let app_secret = decrypt_channel_secret(
            &secret_store,
            &ch_config.whatsapp.app_secret,
            "channels.whatsapp.app_secret",
        );
        // verify_token / app_secret can be empty (existing behavior — only
        // access_token is required by the outer guard) but if the user
        // *did* configure them as encrypted blobs that fail to decrypt we
        // refuse to construct the channel rather than handing a ciphertext
        // string into webhook signature checks.
        if let (Some(access), Some(verify), Some(secret)) =
            (access_token, verify_token, app_secret)
        {
            if !access.is_empty() {
                let ch = fennec::channels::WhatsAppChannel::new(
                    ch_config.whatsapp.phone_number_id.clone(),
                    access,
                    verify,
                    ch_config.whatsapp.webhook_port,
                    ch_config.whatsapp.allowed_users.clone(),
                    secret,
                );
                channels.push(Arc::new(ch));
                tracing::info!("WhatsApp channel enabled");
            }
        }
    }

    if ch_config.email.enabled
        && !ch_config.email.smtp_host.is_empty()
        && !ch_config.email.imap_host.is_empty()
    {
        let imap_password = decrypt_channel_secret(
            &secret_store,
            &ch_config.email.imap_password,
            "channels.email.imap_password",
        );
        let smtp_password = decrypt_channel_secret(
            &secret_store,
            &ch_config.email.smtp_password,
            "channels.email.smtp_password",
        );
        if let (Some(imap_pw), Some(smtp_pw)) = (imap_password, smtp_password) {
            let ch = fennec::channels::EmailChannel::new(
                ch_config.email.imap_host.clone(),
                ch_config.email.imap_port,
                ch_config.email.imap_user.clone(),
                imap_pw,
                ch_config.email.smtp_host.clone(),
                ch_config.email.smtp_port,
                ch_config.email.smtp_user.clone(),
                smtp_pw,
                ch_config.email.from_address.clone(),
                ch_config.email.allowed_senders.clone(),
                ch_config.email.poll_interval_secs,
            );
            channels.push(Arc::new(ch));
            tracing::info!("Email channel enabled");
        }
    }

    if let Some(ch) = fennec::channels::WebhookChannel::from_config(&ch_config.webhook) {
        channels.push(Arc::new(ch));
        tracing::info!(
            host = %ch_config.webhook.host,
            port = ch_config.webhook.port,
            routes = ch_config.webhook.routes.len(),
            "Webhook channel enabled"
        );
    }

    if ch_config.openai_compat.enabled {
        let openai_session_store = Arc::new(
            fennec::sessions::SessionStore::new(&home_dir.join("sessions.db"))
                .context("opening sessions.db for openai_compat")?,
        );
        let openai_response_store = Arc::new(
            fennec::channels::openai_compat::ResponseStore::new(
                &home_dir.join("openai_responses.db"),
            )
            .context("opening openai_responses.db")?,
        );
        if let Some(ch) =
            fennec::channels::OpenAiCompatChannel::from_config_with_sessions(
                &ch_config.openai_compat,
                Arc::clone(&agent),
                openai_session_store,
                openai_response_store,
            )
        {
            channels.push(Arc::new(ch));
            tracing::info!(
                host = %ch_config.openai_compat.host,
                port = ch_config.openai_compat.port,
                model = %ch_config.openai_compat.model_name,
                backend = "agent",
                "OpenAI-compat channel enabled (session + response stores wired)"
            );
        }
    }
    let _ = provider;

    if let Some(ch) = fennec::channels::SignalChannel::from_config(&ch_config.signal) {
        channels.push(Arc::new(ch));
        tracing::info!(
            account = %ch_config.signal.account,
            http_url = %ch_config.signal.http_url,
            "Signal channel enabled"
        );
    }

    if let Some(ch) = fennec::channels::MatrixChannel::from_config(&ch_config.matrix) {
        channels.push(Arc::new(ch));
        tracing::info!(
            homeserver = %ch_config.matrix.homeserver,
            user_id = %ch_config.matrix.user_id,
            "Matrix channel enabled"
        );
    }

    // 3a. Populate the channel map so tools (e.g. ask_user) can reach channels.
    {
        let mut map = channel_map.write();
        for ch in &channels {
            map.insert(ch.name().to_string(), Arc::clone(ch));
        }
    }

    // 4. Create ChannelManager, start all channels.
    //
    // The listener handles (Telegram/Discord/Slack/WhatsApp/Email pollers)
    // and the outbound-dispatch handle were previously bound to `_`-prefixed
    // names, which keeps them alive only for the lexical scope but never
    // aborts them on shutdown — they kept polling vendor APIs (and printing
    // reconnect errors to the log) until the runtime was finally dropped at
    // process exit. Bind them so the shutdown block at the bottom can abort
    // them explicitly along with the gateway / agent loop / cron handles.
    let manager = ChannelManager::new(channels, bus.clone());
    let listener_handles = manager.start_all();
    let dispatch_handle = manager.spawn_outbound_dispatch(receiver.outbound_rx);

    // 5. Start CronScheduler — always runs in gateway mode so user-created
    //    reminders/jobs are delivered. The config.cron.enabled flag is reserved
    //    for future heartbeat/auto-task features.
    let _cron_handle = {
        let cron_path = home_dir.join("cron_jobs.json");
        let mut store = JobStore::new(cron_path);
        if let Err(e) = store.load() {
            tracing::warn!("Failed to load cron jobs: {e}");
        }
        let mut scheduler = CronScheduler::new(store, bus.clone(), None);
        tracing::info!("Cron scheduler started");
        Some(tokio::spawn(async move {
            scheduler.run().await;
        }))
    };

    // 6. Start GatewayServer in a background task.
    let host = host_override.unwrap_or_else(|| config.gateway.host.clone());
    let port = port_override.unwrap_or(config.gateway.port);
    let addr = format!("{host}:{port}");

    let auth_token = if config.gateway.auth_token.is_empty() {
        None
    } else {
        Some(config.gateway.auth_token.clone())
    };

    let gateway = GatewayServer::new(addr);
    let gateway_agent = Arc::clone(&agent);
    // Channel used to ask the gateway to stop accepting new connections
    // and drain in-flight requests. The shutdown handler at the bottom
    // of this function sends `true` here on Ctrl-C; the gateway's
    // `with_graceful_shutdown` future awaits it. We await the gateway's
    // JoinHandle (with a small grace timeout) instead of `.abort()`-ing
    // it, so a request that's mid-turn doesn't lose its agent lock or
    // its outbound publish.
    let (gateway_shutdown_tx, mut gateway_shutdown_rx) =
        tokio::sync::watch::channel::<bool>(false);
    let gateway_handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            // Wait until the watch value flips to `true`. If the sender
            // is dropped before that (e.g. main panics), `changed()`
            // returns Err — treat as shutdown so we don't dangle.
            while gateway_shutdown_rx.changed().await.is_ok() {
                if *gateway_shutdown_rx.borrow() {
                    break;
                }
            }
        };
        if let Err(e) = gateway
            .run_with_shutdown(gateway_agent, auth_token, shutdown_signal)
            .await
        {
            tracing::error!("Gateway server error: {e}");
        }
    });

    // 7. Main loop: consume inbound messages from bus, run agent, publish outbound.
    let agent_loop = {
        let agent = Arc::clone(&agent);
        let bus = bus.clone();
        let manager_ref = Arc::new(manager);
        let cron_origin = Arc::clone(&cron_origin);
        let pending_replies = pending_replies.clone();
        let chat_directory = chat_directory.clone();
        tokio::spawn(async move {
            while let Some(msg) = receiver.inbound_rx.recv().await {
                // Always record the (channel, chat_id) pair in the
                // directory, even for messages we'll consume via a
                // pending reply or treat as a /reset command. This
                // keeps the directory's "recently seen" view honest.
                chat_directory.record(&msg.channel, &msg.chat_id);

                // Cron-fired messages carry `metadata["source"] = "cron"`
                // and represent the agent's own scheduled prompt, not a
                // user reply. They must not be redirected to a pending
                // ask_user wait (a cron output happening to land while
                // the agent is mid-question would otherwise be consumed
                // as the user's "answer"). Skip the pending check for
                // anything sourced internally.
                let is_user_initiated = msg
                    .metadata
                    .get("source")
                    .map(|s| s.as_str() != "cron")
                    .unwrap_or(true);

                if is_user_initiated {
                    let pending_origin = fennec::bus::TurnOrigin {
                        channel: msg.channel.clone(),
                        chat_id: msg.chat_id.clone(),
                    };
                    if pending_replies.take_and_deliver(&pending_origin, msg.clone()) {
                        tracing::debug!(
                            "Inbound from {}:{} consumed by pending tool reply",
                            msg.channel,
                            msg.chat_id,
                        );
                        continue;
                    }
                }

                // Handle /new and /reset commands: clear agent history and
                // send a confirmation instead of running a full agent turn.
                if msg.metadata.get("command").map(|s| s.as_str()) == Some("reset") {
                    // Drop the agent lock before publishing on the bus so
                    // gateway HTTP /chat (which also takes agent.lock()) is
                    // not serialized through the outbound publish await.
                    {
                        let mut agent_lock = agent.lock().await;
                        agent_lock.clear_history();
                    }
                    let outbound = fennec::bus::OutboundMessage {
                        content: "Session reset! Starting fresh.".to_string(),
                        channel: msg.channel.clone(),
                        chat_id: msg.chat_id.clone(),
                        reply_to: Some(msg.id.clone()),
                        metadata: std::collections::HashMap::new(),
                        attachments: Vec::new(),
                    };
                    let _ = bus.publish_outbound(outbound).await;
                    continue;
                }

                // Spawn continuous typing indicator that fires every 4 seconds
                // until the agent finishes processing.
                let typing_channel: Option<Arc<dyn Channel>> = manager_ref.get_channel(&msg.channel);
                let typing_chat_id = msg.chat_id.clone();
                let typing_handle = tokio::spawn(async move {
                    if let Some(ch) = typing_channel {
                        loop {
                            let _ = ch.send_typing(&typing_chat_id).await;
                            tokio::time::sleep(tokio::time::Duration::from_secs(4)).await;
                        }
                    }
                });

                // Set the CronTool's origin so any jobs created during this
                // turn know which channel/chat to deliver results to.
                // Recover from a poisoned mutex via `into_inner` so that a
                // single panic-while-locked elsewhere can't kill all
                // subsequent inbound turns. (Same recovery pattern as the
                // CronTool::execute lock site.)
                {
                    let mut origin = cron_origin
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    *origin = Some(CronOrigin {
                        channel: msg.channel.clone(),
                        chat_id: msg.chat_id.clone(),
                    });
                }

                // Hold the agent lock only for the LLM turn itself. All
                // subsequent I/O (typing-indicator abort, streaming
                // delivery, bus publish) runs without the lock so that
                // gateway HTTP /chat — which also takes agent.lock() —
                // doesn't serialize behind a finished agent's outbound
                // publish.
                let turn_result = {
                    let mut agent_lock = agent.lock().await;
                    agent_lock.turn(&msg.content).await
                };
                match turn_result {
                    Ok(response) => {
                        // Stop typing indicator.
                        typing_handle.abort();

                        // If the response starts with "[SILENT]", the agent
                        // decided nothing needs to be said — skip outbound.
                        if response.starts_with("[SILENT]") {
                            tracing::debug!(
                                "Agent response marked [SILENT], suppressing outbound"
                            );
                            continue;
                        }

                        // Check if the channel supports streaming and use
                        // streaming delivery for a progressive typing effect.
                        let streaming_channel = manager_ref.get_channel(&msg.channel)
                            .filter(|ch| ch.supports_streaming());

                        if let Some(ch) = streaming_channel {
                            match ch.send_streaming_start(&msg.chat_id).await {
                                Ok(Some(mid)) => {
                                    // Deliver the response in chunks to simulate
                                    // streaming on the channel side.
                                    let mut accumulated = String::new();
                                    let chunk_size = 80;
                                    let mut chars = response.chars().peekable();

                                    while chars.peek().is_some() {
                                        let chunk: String = chars.by_ref().take(chunk_size).collect();
                                        accumulated.push_str(&chunk);
                                        let _ = ch
                                            .send_streaming_delta(
                                                &msg.chat_id,
                                                &mid,
                                                &accumulated,
                                            )
                                            .await;
                                        // Small delay between chunks for visual effect;
                                        // the channel's own rate limiter handles throttling.
                                        tokio::time::sleep(tokio::time::Duration::from_millis(50))
                                            .await;
                                    }

                                    let _ = ch
                                        .send_streaming_end(
                                            &msg.chat_id,
                                            &mid,
                                            &accumulated,
                                        )
                                        .await;
                                    // Streaming already delivered — skip outbound bus.
                                    continue;
                                }
                                Ok(None) | Err(_) => {
                                    // Streaming start failed; fall through to
                                    // normal outbound delivery below.
                                    tracing::warn!(
                                        "Streaming start failed for channel '{}', falling back",
                                        msg.channel
                                    );
                                }
                            }
                        }

                        let outbound = fennec::bus::OutboundMessage {
                            content: response,
                            channel: msg.channel.clone(),
                            chat_id: msg.chat_id.clone(),
                            reply_to: Some(msg.id.clone()),
                            metadata: std::collections::HashMap::new(),
                            attachments: Vec::new(),
                        };
                        if let Err(e) = bus.publish_outbound(outbound).await {
                            tracing::error!("Failed to publish outbound: {e}");
                        }
                    }
                    Err(e) => {
                        // Stop typing indicator.
                        typing_handle.abort();

                        tracing::error!(
                            "Agent turn failed for message from {}: {e}",
                            msg.channel
                        );

                        // Send error message back to the user.
                        let error_msg = format!(
                            "Something went wrong: {}\n\nTry /new to start a fresh session.",
                            e
                        );
                        let outbound = fennec::bus::OutboundMessage {
                            content: error_msg,
                            channel: msg.channel.clone(),
                            chat_id: msg.chat_id.clone(),
                            reply_to: Some(msg.id.clone()),
                            metadata: std::collections::HashMap::new(),
                            attachments: Vec::new(),
                        };
                        let _ = bus.publish_outbound(outbound).await;
                    }
                }
            }
        })
    };

    // 8. Wait for SIGINT or SIGTERM and shut down all background tasks.
    //
    // Previously only `tokio::signal::ctrl_c()` was awaited. Under
    // `systemd stop` / `docker stop` / `kill` the process receives
    // SIGTERM, which terminates without ever reaching the abort block —
    // listener handles kept hammering vendor APIs until the runtime was
    // dropped. We listen for both signals on Unix and fall back to
    // ctrl_c() on Windows (where SIGTERM doesn't exist).
    wait_for_shutdown_signal().await;

    // Tell the gateway to stop accepting new connections and drain.
    // `send` only fails if all receivers have been dropped, which would
    // already mean the gateway task ended; either way, ignore the result
    // and proceed with the rest of shutdown.
    let _ = gateway_shutdown_tx.send(true);

    // Give the gateway up to 30s to drain. Most in-flight requests
    // complete within seconds; the per-request 10-minute hard cap means
    // even a stuck request can't block this forever, but the request
    // timeout would extend shutdown to ~10 minutes worst-case which is
    // unfriendly. Cap the drain wait so a stuck request doesn't hold
    // the whole process. If drain takes longer than the cap, we abort.
    let gateway_shutdown_grace = std::time::Duration::from_secs(30);
    match tokio::time::timeout(gateway_shutdown_grace, gateway_handle).await {
        Ok(Ok(())) => tracing::info!("Gateway drained cleanly"),
        Ok(Err(e)) => tracing::warn!("Gateway task ended with error: {}", e),
        Err(_) => {
            tracing::warn!(
                "Gateway didn't drain within {:?}; some in-flight requests \
                 may have been cut off",
                gateway_shutdown_grace,
            );
        }
    }

    // Abort the rest of the supervised tasks.
    agent_loop.abort();
    if let Some(h) = _cron_handle {
        h.abort();
    }
    for h in listener_handles {
        h.abort();
    }
    dispatch_handle.abort();

    Ok(())
}

/// Wait for SIGINT (Ctrl-C) or SIGTERM and log which one fired.
///
/// On Unix, listens for both via `tokio::select!`. On Windows there's no
/// SIGTERM, so we just await `ctrl_c()`. The `signal` feature is enabled
/// in `Cargo.toml`.
async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        // `signal()` can fail if the underlying syscall registration fails
        // (e.g. fd exhaustion). If SIGTERM registration fails we degrade
        // to SIGINT-only rather than aborting startup — better to run
        // without graceful systemd shutdown than to refuse to come up.
        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    "Failed to install SIGTERM handler: {e}; falling back to SIGINT only"
                );
                let _ = tokio::signal::ctrl_c().await;
                tracing::info!("Received SIGINT, shutting down...");
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                tracing::info!("Received SIGINT, shutting down...");
            }
            _ = sigterm.recv() => {
                tracing::info!("Received SIGTERM, shutting down...");
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("Received Ctrl-C, shutting down...");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // -- Two-phase argv handling ----------------------------------------
    //
    // Plugin-contributed subcommands (`fennec spotify play ...`) are
    // discovered at runtime, so clap can't resolve them via derive
    // alone. We do a cheap first pass to learn the plugin command
    // names from manifests (no plugin instantiation), then build a
    // dynamic clap Command that includes both the derive-defined
    // built-ins AND the plugin subcommands.
    //
    // To learn `[plugins].enabled` and the home directory, we have
    // to look at `--config-dir` and load config BEFORE the full
    // parse. The first pass is allowed to be permissive — it just
    // peeks for those two values.

    // Resolve the Fennec home directory from raw argv BEFORE clap
    // parses, so plugin CLI discovery (below) sees the right
    // `[plugins].enabled` list. `--config-dir` and `--profile` are
    // mutually exclusive — each picks the same thing (where state
    // lives) so accepting both at once would force an arbitrary
    // precedence rule. Reject the combination here.
    let raw_args: Vec<String> = std::env::args().collect();
    let pre_config_dir = preparse_config_dir(&raw_args);
    let pre_profile = preparse_profile(&raw_args);
    if pre_config_dir.is_some() && pre_profile.is_some() {
        anyhow::bail!(
            "--config-dir and --profile are mutually exclusive; use one or the other"
        );
    }
    let pre_home = match (pre_config_dir.as_deref(), pre_profile.as_deref()) {
        (Some(path), None) => FennecConfig::resolve_home(Some(path)),
        (None, Some(name)) => FennecConfig::resolve_profile_home(name)
            .with_context(|| format!("resolving profile '{}'", name))?,
        _ => FennecConfig::resolve_home(None),
    };
    let pre_config_path = pre_home.join("config.toml");
    let pre_config = if pre_config_path.exists() {
        FennecConfig::load(&pre_config_path).unwrap_or_default()
    } else {
        FennecConfig::default()
    };

    // Lightweight CLI command discovery — no plugin instantiation.
    // Returns the spec list the plugins want exposed as `fennec
    // <name>` subcommands, validated and deduplicated against
    // built-in command names and against each other.
    let plugin_cli_specs = discover_plugin_cli_specs(&pre_home, &pre_config.plugins.enabled);

    // Build the clap command tree: derive-defined built-ins plus
    // dynamic plugin subcommands. Each plugin subcommand accepts
    // `args = [...]` as positional, trailing-var-arg, so anything
    // after `fennec spotify` lands intact in the plugin handler.
    let mut clap_cmd = <Cli as clap::CommandFactory>::command();
    for spec in &plugin_cli_specs {
        // Leak the spec name into a 'static str so clap's
        // string-pool is happy. Plugin command names are
        // bounded (1-32 chars, validated above) and the leak is
        // bounded to startup-time discovery — not a real memory
        // concern.
        let static_name: &'static str = Box::leak(spec.name.clone().into_boxed_str());
        let static_about: &'static str = Box::leak(spec.description.clone().into_boxed_str());
        clap_cmd = clap_cmd.subcommand(
            clap::Command::new(static_name)
                .about(static_about)
                .arg(
                    clap::Arg::new("args")
                        .num_args(0..)
                        .trailing_var_arg(true)
                        .allow_hyphen_values(true)
                        .help("Arguments forwarded to the plugin"),
                ),
        );
    }
    let matches = clap_cmd.get_matches_from(&raw_args);

    // If a plugin subcommand matched, dispatch to the plugin
    // BEFORE doing any agent / gateway work. Plugin commands run
    // in the foreground and return an exit code.
    if let Some((sub_name, sub_matches)) = matches.subcommand() {
        if plugin_cli_specs.iter().any(|s| s.name == sub_name) {
            // Now do the full plugin load to get the handler.
            return dispatch_plugin_cli(
                &pre_config,
                &pre_home,
                sub_name,
                sub_matches,
            )
            .await;
        }
    }

    // No plugin command matched — re-parse via the derive-typed
    // path so we get the strongly-typed `Commands` enum for the
    // built-in subcommands. This second parse is essentially free
    // and gives us back the structured arguments without
    // hand-rolling translation from `ArgMatches`.
    let cli = Cli::parse_from(&raw_args);

    // Load config: try from config dir, fall back to defaults.
    // (Re-uses pre_home and pre_config so we don't re-read config.toml.)
    let home_dir = pre_home;
    let config = pre_config;

    match cli.command {
        Commands::Agent { message, model, tui } => {
            if tui {
                run_tui(config, home_dir, model, None).await?;
            } else {
                run_agent(config, home_dir, message, model).await?;
            }
        }
        Commands::Status => {
            println!("Fennec v{}", env!("CARGO_PKG_VERSION"));
            println!("Status: ready");
        }
        Commands::Gateway { host, port } => {
            run_gateway(config, home_dir, host, port).await?;
        }
        Commands::Onboard { force } => {
            let config_path = home_dir.join("config.toml");
            if config_path.exists() && !force {
                eprintln!(
                    "Config already exists at {}. Use --force to overwrite.",
                    config_path.display()
                );
                std::process::exit(1);
            }
            fennec::onboard::run_wizard(&home_dir)?;
        }
        Commands::Login => {
            auth::run_oauth_login(&home_dir)?;
        }
        Commands::Doctor => {
            run_doctor(&config, &home_dir).await?;
        }
        Commands::Curator { action } => {
            run_curator_command(&config, &home_dir, action).await?;
        }
        Commands::Mcp { action } => match action {
            McpAction::Serve { verbose } => {
                run_mcp_serve(home_dir, verbose).await?;
            }
        },
    }

    Ok(())
}

/// Pre-parse the `--config-dir` flag from raw argv. Used at startup
/// before clap is fully configured (we need to know the home dir
/// to read config + discover plugin CLI commands BEFORE clap can
/// build a complete subcommand tree).
///
/// Permissive: ignores unknown flags, doesn't fail on malformed
/// argv. Worst case we miss the override and fall back to the
/// default home dir, which clap will catch on the real parse.
fn preparse_config_dir(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == "--config-dir" {
            return iter.next().cloned();
        }
        if let Some(v) = a.strip_prefix("--config-dir=") {
            return Some(v.to_string());
        }
    }
    None
}

/// Pre-parse the `--profile` flag from raw argv, same pattern as
/// `preparse_config_dir`. Plugin CLI discovery has to happen before
/// clap parses, so both flags need pre-extraction.
fn preparse_profile(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == "--profile" {
            return iter.next().cloned();
        }
        if let Some(v) = a.strip_prefix("--profile=") {
            return Some(v.to_string());
        }
    }
    None
}

/// Lightweight discovery of plugin-contributed CLI command specs.
/// Iterates bundled plugin static refs (cheap — just trait method
/// calls, no instantiation) and reads WASM `plugin.toml` manifests
/// from `<home>/plugins/`. Used by `main` to extend clap before
/// argv is parsed.
///
/// Validation, dedup, and reserved-name checks are applied here —
/// the returned specs are safe to feed into clap as subcommands.
fn discover_plugin_cli_specs(
    home_dir: &std::path::Path,
    enabled: &[String],
) -> Vec<fennec::plugins::CliCommandSpec> {
    use fennec::plugins::PluginEntry;
    use std::collections::HashSet;

    let want: HashSet<&str> = enabled.iter().map(String::as_str).collect();
    if want.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<fennec::plugins::CliCommandSpec> = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    // Bundled plugins — call cli_commands() on each enabled static
    // reference. No `register()` invocation, no agent build.
    for entry in fennec::plugins::inventory::iter::<PluginEntry> {
        let manifest = entry.plugin.manifest();
        if !want.contains(manifest.name.as_str()) {
            continue;
        }
        for spec in entry.plugin.cli_commands() {
            if let Err(e) = fennec::plugins::validate_command_name(&spec.name) {
                tracing::warn!(
                    plugin = %manifest.name,
                    command = %spec.name,
                    "Bundled plugin CLI command rejected at discovery: {e}"
                );
                continue;
            }
            if !seen_names.insert(spec.name.clone()) {
                tracing::warn!(
                    plugin = %manifest.name,
                    command = %spec.name,
                    "Plugin CLI command name '{}' already taken; dropping",
                    spec.name
                );
                continue;
            }
            out.push(spec);
        }
    }

    // WASM plugins — read manifests from disk. The `cli_commands`
    // field of `plugin.toml` declares names + descriptions; we
    // bind handlers later when a plugin command actually matches.
    let plugins_root = home_dir.join("plugins");
    if let Ok(discovered) =
        fennec::plugins::wasm::discover_wasm_plugins(&plugins_root)
    {
        for d in discovered {
            if !want.contains(d.manifest.name.as_str()) {
                continue;
            }
            for spec in &d.manifest.cli_commands {
                if let Err(e) = fennec::plugins::validate_command_name(&spec.name) {
                    tracing::warn!(
                        plugin = %d.manifest.name,
                        command = %spec.name,
                        "WASM plugin CLI command rejected at discovery: {e}"
                    );
                    continue;
                }
                if !seen_names.insert(spec.name.clone()) {
                    tracing::warn!(
                        plugin = %d.manifest.name,
                        command = %spec.name,
                        "Plugin CLI command name '{}' already taken; dropping",
                        spec.name
                    );
                    continue;
                }
                out.push(spec.clone());
            }
        }
    }

    out
}

/// Dispatch a matched plugin CLI command. Does the full plugin
/// load (bundled + WASM) so the matched command's handler is
/// available, then runs it.
///
/// The agent is NOT built and NOT started — plugin CLI commands
/// run as standalone tools. Returns `Ok(())` regardless of the
/// plugin's exit code; the exit code is propagated via
/// `std::process::exit` so shell pipelines see the right thing.
async fn dispatch_plugin_cli(
    config: &FennecConfig,
    home_dir: &std::path::Path,
    sub_name: &str,
    sub_matches: &clap::ArgMatches,
) -> Result<()> {
    // Pull args following `fennec <plugin-cmd>`. clap stores them
    // under the "args" id we set on the dynamic subcommand.
    let args: Vec<String> = sub_matches
        .get_many::<String>("args")
        .map(|vs| vs.cloned().collect())
        .unwrap_or_default();

    // Full plugin load. Most fields aren't relevant for a CLI-only
    // dispatch (we don't need memory, channel bus, etc.), but the
    // load path requires them. Use stub-friendly choices for the
    // resources the plugins might touch.
    let mut registry = fennec::plugins::PluginRegistry::new();
    if let Err(e) = registry.load_bundled(&config.plugins.enabled) {
        tracing::error!(
            "Plugin registry failed to load bundled for CLI dispatch: {e}"
        );
    }

    // For WASM, we still need the full host resources because the
    // plugin's CLI handler may use `http_request` / `read_file` /
    // etc. as host imports. Construct an in-memory `SqliteMemory`
    // (under home/memory.db) for the plugin to use; channel bus
    // is None because we're not in gateway mode.
    let path_sandbox = Arc::new(PathSandbox::new(config.security.forbidden_paths.clone()));
    let memory_path = home_dir.join("memory.db");
    let memory: Arc<dyn Memory> = Arc::new(
        SqliteMemory::new(
            memory_path,
            config.memory.vector_weight as f32,
            config.memory.keyword_weight as f32,
            config.memory.cache_max,
            Arc::new(NoopEmbedding::new(1536)),
        )
        .context("opening memory store for plugin CLI dispatch")?,
    );
    let wasm_resources = fennec::plugins::WasmHostResources {
        path_sandbox,
        memory,
        http_client: fennec::tools::http::shared_client(),
        rt_handle: tokio::runtime::Handle::current(),
        settings: config.plugins.settings.clone(),
        bus: None,
    };
    let plugins_root = home_dir.join("plugins");
    if let Err(e) = registry.load_wasm(
        &plugins_root,
        &config.plugins.enabled,
        wasm_resources,
    ) {
        tracing::error!("Plugin registry failed to load WASM for CLI dispatch: {e}");
    }

    let runtime = registry.into_runtime(&config.memory.provider);

    let exit_code = match runtime.dispatch_cli(sub_name, args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("plugin command '{sub_name}' failed: {e}");
            1
        }
    };
    std::process::exit(exit_code);
}

/// Dispatch a `fennec curator <action>` invocation. Each action
/// touches at most three filesystem entries (skills root, usage
/// sidecar, curator state file) so they are cheap to wire here
/// without going through the full agent build.
async fn run_curator_command(
    config: &FennecConfig,
    home_dir: &std::path::Path,
    action: CuratorAction,
) -> Result<()> {
    use fennec::skills::curator::{
        CuratorScheduleConfig, CuratorStateStore, RunContext, run_curator,
    };
    use fennec::skills::{UsageStore, archive};

    let skills_dir = home_dir.join("skills");
    std::fs::create_dir_all(&skills_dir).context("ensuring skills dir exists")?;
    let logs_dir = home_dir.join("logs");
    let usage = Arc::new(UsageStore::open(&skills_dir));
    let state = Arc::new(CuratorStateStore::open(&skills_dir));

    match action {
        CuratorAction::Status => {
            let snap = state.snapshot();
            println!("Curator status");
            println!("  paused: {}", snap.paused);
            println!("  run_count: {}", snap.run_count);
            match snap.last_run_at {
                Some(t) => {
                    println!("  last_run_at: {}", t);
                    println!("  last_run_duration: {:.2}s", snap.last_run_duration_seconds);
                    println!("  last_run_summary: {}", snap.last_run_summary);
                    if let Some(p) = &snap.last_report_path {
                        println!("  last_report: {}", p.display());
                    }
                }
                None => {
                    println!("  last_run_at: never");
                }
            }
            // Show schedule decision against current state for
            // operator visibility.
            let sched = CuratorScheduleConfig::default();
            let decision = fennec::skills::curator::should_auto_run(
                &sched,
                &snap,
                None,
                chrono::Utc::now(),
            );
            match decision {
                fennec::skills::curator::AutoRunDecision::Run => {
                    println!("  next: would run on next idle session");
                }
                fennec::skills::curator::AutoRunDecision::Skip(reason) => {
                    println!("  next: skipped — {}", reason.as_human_string());
                }
            }
        }
        CuratorAction::Run => {
            let primary = build_provider_for_curator(config, home_dir).await?;
            let aux = primary.map(|p| Arc::new(p));
            let mut ctx = RunContext::new(
                skills_dir.clone(),
                logs_dir,
                Arc::clone(&usage),
                Arc::clone(&state),
            );
            ctx.aux = aux;
            let summary = run_curator(&ctx).await?;
            println!("Curator finished");
            println!("  duration: {:.2}s", summary.duration_seconds);
            println!("  summary: {}", summary.one_line_summary);
            if let Some(p) = summary.report_dir {
                println!("  report: {}", p.display());
            }
        }
        CuratorAction::Pause => {
            state.set_paused(true).context("setting paused flag")?;
            println!("Curator paused. Manual `fennec curator run` still works.");
        }
        CuratorAction::Resume => {
            state.set_paused(false).context("clearing paused flag")?;
            println!("Curator resumed.");
        }
        CuratorAction::Pin { name } => {
            usage.set_pinned(&name, true);
            println!("Pinned {:?}.", name);
        }
        CuratorAction::Unpin { name } => {
            usage.set_pinned(&name, false);
            println!("Unpinned {:?}.", name);
        }
        CuratorAction::Restore { name } => {
            let dest = archive::restore(&skills_dir, &name)
                .with_context(|| format!("restoring {:?}", name))?;
            usage.set_state(&name, fennec::skills::SkillState::Active);
            println!("Restored {:?} to {}.", name, dest.display());
        }
    }

    Ok(())
}

/// Build the auxiliary client for the curator CLI command. Returns
/// `Ok(None)` when the operator hasn't configured any provider — the
/// curator falls back to auto-only mode rather than failing.
async fn build_provider_for_curator(
    config: &FennecConfig,
    home_dir: &std::path::Path,
) -> Result<Option<fennec::providers::AuxiliaryClient>> {
    let secret_store = SecretStore::new(home_dir.to_path_buf())
        .context("creating secret store for curator")?;
    let api_key = match resolve_api_key(config, &secret_store) {
        Ok(k) if !k.is_empty() => k,
        _ => {
            tracing::info!("no provider key for curator — running auto-transition phase only");
            return Ok(None);
        }
    };
    let primary: Arc<dyn Provider> = build_provider(config, api_key, None).into();
    Ok(Some(build_auxiliary_client(config, primary, &secret_store)))
}

/// Run Fennec as an MCP server on stdio. Logs to stderr (stdout is
/// the protocol transport — anything written there desyncs the
/// JSON-RPC stream). Read-only mode: the server doesn't bring up
/// channel listeners, just exposes the session DB and a stub
/// channel list. `messages_send` returns a clear error in that
/// mode; users who need send must run `fennec gateway` separately
/// and configure the MCP server to share its channel manager
/// (future work).
async fn run_mcp_serve(home_dir: std::path::PathBuf, verbose: bool) -> Result<()> {
    use fennec::mcp::serve::{EventBridge, McpServerHandler, ServerState, run_stdio};
    use fennec::sessions::SessionStore;

    // Re-init logging to stderr only — silencing the default
    // tracing-subscriber that may have been pointed at stdout.
    let level = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::WARN
    };
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(level)
        .try_init();

    let db_path = home_dir.join("sessions.db");
    let store = Arc::new(
        SessionStore::new(&db_path)
            .with_context(|| format!("opening sessions DB at {}", db_path.display()))?,
    );

    let bridge = EventBridge::new();
    bridge
        .seed_from_store(&store)
        .await
        .context("seeding event-bridge cursor from session store")?;
    let _poller = bridge.spawn_poller(Arc::clone(&store));

    let state = ServerState::new(home_dir.clone(), Arc::clone(&store));
    // Read-only mode: no channels attached. messages_send and
    // channels_list will report empty. The user can still consume
    // every read tool and the events stream.
    let handler = McpServerHandler::new(state, bridge);

    tracing::info!(
        db = %db_path.display(),
        "Fennec MCP server starting on stdio"
    );
    run_stdio(handler).await
}

async fn run_doctor(config: &FennecConfig, home_dir: &std::path::Path) -> Result<()> {
    use fennec::doctor;
    let secret_store = SecretStore::new(home_dir.to_path_buf()).context("creating secret store")?;
    let use_color = console::Term::stdout().is_term()
        && std::env::var("NO_COLOR").is_err();

    let heading = if use_color {
        console::style("Fennec Doctor").cyan().bold().to_string()
    } else {
        "Fennec Doctor".to_string()
    };
    println!();
    println!("  {}", heading);
    let rule = "─".repeat(40);
    let rule_styled = if use_color {
        console::style(rule).dim().to_string()
    } else {
        rule
    };
    println!("  {}", rule_styled);
    println!();

    let results = doctor::run_all(config, home_dir, &secret_store).await;
    for r in &results {
        println!("  {}", doctor::render_result(r, use_color));
    }
    println!();
    let (summary, any_failed) = doctor::render_summary(&results, use_color);
    println!("  {}", summary);
    println!();
    if any_failed {
        std::process::exit(1);
    }
    Ok(())
}
