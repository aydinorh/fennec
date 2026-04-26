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
use fennec::config::FennecConfig;
use fennec::cron::{CronScheduler, JobStore};
use fennec::gateway::GatewayServer;
use fennec::memory::embedding::{NoopEmbedding, OpenAIEmbedding};
use fennec::memory::snapshot;
use fennec::memory::sqlite::SqliteMemory;
use fennec::memory::Memory;
use fennec::providers::anthropic::AnthropicProvider;
use fennec::providers::openai::OpenAIProvider;
use fennec::providers::ollama::OllamaProvider;
use fennec::providers::traits::Provider;
use fennec::security::prompt_guard::{GuardAction, PromptGuard};
use fennec::security::SecretStore;
use fennec::tools::collective_tools::{CollectiveGetExperienceTool, CollectivePublishTool, CollectiveReportTool, CollectiveSearchTool};
use fennec::tools::cron_tool::{CronOrigin, CronTool};
use fennec::tools::files::{GlobTool, GrepTool, ListDirTool, ReadFileTool, WriteFileTool};
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
use fennec::tools::delegate_tool::DelegateTool;
use fennec::tools::traits::Tool;

#[derive(Parser, Debug)]
#[command(name = "fennec", version, about = "The fastest personal AI agent with collective intelligence")]
struct Cli {
    #[arg(long, global = true)]
    config_dir: Option<String>,

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
        "ollama" => return Ok(String::new()), // Ollama needs no key
        _ => "ANTHROPIC_API_KEY",
    };

    std::env::var(env_var)
        .with_context(|| format!("API key not found: set provider.api_key in config or {} env var", env_var))
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
            let kimi_model = if model.is_empty() || model == "claude-sonnet-4-20250514" || model == "moonshot-v1-128k" {
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
        "ollama" => {
            let ollama_model = if model.is_empty() || model == "claude-sonnet-4-20250514" {
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
/// Returns `(agent, memory, cron_origin_handle)`. The cron origin handle is
/// an `Arc<Mutex<Option<CronOrigin>>>` that the gateway sets before each
/// agent turn so the CronTool knows the originating channel/chat.
async fn build_agent(
    config: &FennecConfig,
    home_dir: &std::path::Path,
    model_override: Option<String>,
    channel_map: Option<ChannelMapHandle>,
) -> Result<(fennec::agent::Agent, Arc<dyn Memory>, Arc<Mutex<Option<CronOrigin>>>)> {
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
    let read_file_tool = ReadFileTool::new();
    let write_file_tool = WriteFileTool::new();
    let list_dir_tool = ListDirTool::new();
    let web_fetch_tool = WebFetchTool::new();
    let web_search_tool = WebSearchTool::new();
    let browser_tool = BrowserTool::new();

    // Vision tool: only wired when the configured provider supports vision
    // (anthropic, openai) AND we can resolve an API key. OAuth-only users and
    // non-vision providers silently skip it.
    let vision_api_key = resolve_api_key(config, &secret_store)
        .ok()
        .unwrap_or_default();
    let vision_tool = VisionTool::from_provider(
        &config.provider.name,
        vision_api_key,
        Some(config.provider.model.clone()),
    );
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
    let transcribe_tool = TranscribeAudioTool::new_with_key(voice_key.clone(), None);
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

    // Create CronTool with shared origin handle.
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
        // Resolve collective API key.
        let collective_api_key = if !config.collective.api_key.is_empty() {
            match secret_store.decrypt(&config.collective.api_key) {
                Ok(key) => key,
                Err(e) => {
                    tracing::warn!("Failed to decrypt collective API key: {e}; trying raw value");
                    config.collective.api_key.clone()
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
            let client = PlurumlClient::new(collective_api_key, base_url);
            tracing::info!("Collective intelligence enabled (Plurum remote)");
            Some(Arc::new(client))
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
    builder = builder.tool(Box::new(PdfReadTool::new(home_dir.join("pdf_cache"))));
    builder = builder.tool(Box::new(ScreenshotTool::new(default_screenshot_dir(home_dir))));
    builder = builder.tool(Box::new(HttpRequestTool::new()));
    builder = builder.tool(Box::new(WeatherTool::new()));
    builder = builder.tool(Box::new(ImageInfoTool::new(home_dir.join("image_cache"))));
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

    // Add send_message tool when running in gateway mode (bus available via channel_map).
    if let Some(ref ch_map) = channel_map {
        builder = builder.tool(Box::new(SendMessageTool::new(Arc::clone(ch_map))));
    }

    // Add ask_user tool when running in gateway mode (channel map available).
    if let Some(ref ch_map) = channel_map {
        // Default to "telegram" — the most common gateway channel.
        let default_ch = "telegram".to_string();
        builder = builder.tool(Box::new(AskUserTool::new(Arc::clone(ch_map), default_ch)));
    }

    // Add collective tools if enabled.
    if let Some(ref search) = collective_search {
        builder = builder.tool(Box::new(CollectiveSearchTool::new(Arc::clone(search))));

        // CollectiveReportTool needs a CollectiveLayer; create one based on config.
        if config.collective.publish_enabled {
            let collective_api_key = if !config.collective.api_key.is_empty() {
                secret_store
                    .decrypt(&config.collective.api_key)
                    .unwrap_or_else(|_| config.collective.api_key.clone())
            } else {
                std::env::var("PLURUM_API_KEY").unwrap_or_default()
            };

            let report_layer: Arc<dyn CollectiveLayer> = if !collective_api_key.is_empty() {
                let base_url = if config.collective.base_url.is_empty() {
                    None
                } else {
                    Some(config.collective.base_url.clone())
                };
                Arc::new(PlurumlClient::new(collective_api_key, base_url))
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

    // Wire DelegateTool so the agent can spawn read-only sub-agents for
    // bounded research / investigation tasks. The sub-agent runs
    // synchronously, capped at 10 tool iterations, with a fresh history.
    // Toolkit is intentionally read-only: anything that writes files,
    // spends money, or touches live systems stays with the main agent.
    //
    // We started with just file-read + web-read. That turned out to be
    // too narrow: the most common delegate pattern is "search the
    // codebase for X" or "summarize these files matching Y", which
    // needs grep + glob, and "summarize this paper at <URL>" which
    // needs pdf_read. All three are strictly read-only by definition.
    let delegate_subagent_tools: Vec<Arc<dyn Tool>> = vec![
        Arc::new(ReadFileTool::new()),
        Arc::new(ListDirTool::new()),
        Arc::new(GlobTool::new()),
        Arc::new(GrepTool::new()),
        Arc::new(WebFetchTool::new()),
        Arc::new(WebSearchTool::new()),
        Arc::new(PdfReadTool::new(home_dir.join("pdf_cache"))),
    ];
    builder = builder.tool(Box::new(DelegateTool::new(
        Arc::clone(&provider),
        memory.clone(),
        delegate_subagent_tools,
    )));

    let agent = builder
        .identity_name(&config.identity.name)
        .identity_persona(&config.identity.persona)
        .max_tool_iterations(config.agent.max_tool_iterations as usize)
        .max_tokens(config.provider.max_tokens as usize)
        .temperature(config.provider.temperature)
        .memory_context_limit(config.memory.context_limit)
        .half_life_days(config.memory.half_life_days)
        .prompt_guard(prompt_guard)
        .build()
        .context("building agent")?;

    Ok((agent, memory, cron_origin))
}

async fn run_agent(
    config: FennecConfig,
    home_dir: std::path::PathBuf,
    message: Option<String>,
    model: Option<String>,
) -> Result<()> {
    let (mut agent, memory, _cron_origin) = build_agent(&config, &home_dir, model, None).await?;

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
    let (agent, _memory, cron_origin) = build_agent(&config, &home_dir, None, Some(channel_map.clone())).await?;
    let agent = Arc::new(tokio::sync::Mutex::new(agent));

    // 3. Build channel list from config.
    let mut channels: Vec<Arc<dyn Channel>> = Vec::new();

    let ch_config = &config.channels;

    if ch_config.telegram.enabled && !ch_config.telegram.token.is_empty() {
        let ch = fennec::channels::TelegramChannel::new(
            ch_config.telegram.token.clone(),
            ch_config.telegram.allowed_users.clone(),
        );
        channels.push(Arc::new(ch));
        tracing::info!("Telegram channel enabled");
    }

    if ch_config.discord.enabled && !ch_config.discord.token.is_empty() {
        let ch = fennec::channels::DiscordChannel::new(
            ch_config.discord.token.clone(),
            ch_config.discord.allowed_users.clone(),
        );
        channels.push(Arc::new(ch));
        tracing::info!("Discord channel enabled");
    }

    if ch_config.slack.enabled
        && !ch_config.slack.bot_token.is_empty()
        && !ch_config.slack.app_token.is_empty()
    {
        let ch = fennec::channels::SlackChannel::new(
            ch_config.slack.bot_token.clone(),
            ch_config.slack.app_token.clone(),
            ch_config.slack.allowed_users.clone(),
        );
        channels.push(Arc::new(ch));
        tracing::info!("Slack channel enabled");
    }

    if ch_config.whatsapp.enabled && !ch_config.whatsapp.access_token.is_empty() {
        let ch = fennec::channels::WhatsAppChannel::new(
            ch_config.whatsapp.phone_number_id.clone(),
            ch_config.whatsapp.access_token.clone(),
            ch_config.whatsapp.verify_token.clone(),
            ch_config.whatsapp.webhook_port,
            ch_config.whatsapp.allowed_users.clone(),
        );
        channels.push(Arc::new(ch));
        tracing::info!("WhatsApp channel enabled");
    }

    if ch_config.email.enabled
        && !ch_config.email.smtp_host.is_empty()
        && !ch_config.email.imap_host.is_empty()
    {
        let ch = fennec::channels::EmailChannel::new(
            ch_config.email.imap_host.clone(),
            ch_config.email.imap_port,
            ch_config.email.imap_user.clone(),
            ch_config.email.imap_password.clone(),
            ch_config.email.smtp_host.clone(),
            ch_config.email.smtp_port,
            ch_config.email.smtp_user.clone(),
            ch_config.email.smtp_password.clone(),
            ch_config.email.from_address.clone(),
            ch_config.email.allowed_senders.clone(),
            ch_config.email.poll_interval_secs,
        );
        channels.push(Arc::new(ch));
        tracing::info!("Email channel enabled");
    }

    // 3a. Populate the channel map so tools (e.g. ask_user) can reach channels.
    {
        let mut map = channel_map.write();
        for ch in &channels {
            map.insert(ch.name().to_string(), Arc::clone(ch));
        }
    }

    // 4. Create ChannelManager, start all channels.
    let manager = ChannelManager::new(channels, bus.clone());
    let _listener_handles = manager.start_all();
    let _dispatch_handle = manager.spawn_outbound_dispatch(receiver.outbound_rx);

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
    let gateway_handle = tokio::spawn(async move {
        if let Err(e) = gateway.run(gateway_agent, auth_token).await {
            tracing::error!("Gateway server error: {e}");
        }
    });

    // 7. Main loop: consume inbound messages from bus, run agent, publish outbound.
    let agent_loop = {
        let agent = Arc::clone(&agent);
        let bus = bus.clone();
        let manager_ref = Arc::new(manager);
        let cron_origin = Arc::clone(&cron_origin);
        tokio::spawn(async move {
            while let Some(msg) = receiver.inbound_rx.recv().await {
                // Handle /new and /reset commands: clear agent history and
                // send a confirmation instead of running a full agent turn.
                if msg.metadata.get("command").map(|s| s.as_str()) == Some("reset") {
                    let mut agent_lock = agent.lock().await;
                    agent_lock.clear_history();
                    let outbound = fennec::bus::OutboundMessage {
                        content: "Session reset! Starting fresh.".to_string(),
                        channel: msg.channel.clone(),
                        chat_id: msg.chat_id.clone(),
                        reply_to: Some(msg.id.clone()),
                        metadata: std::collections::HashMap::new(),
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
                {
                    let mut origin = cron_origin.lock().unwrap();
                    *origin = Some(CronOrigin {
                        channel: msg.channel.clone(),
                        chat_id: msg.chat_id.clone(),
                    });
                }

                let mut agent_lock = agent.lock().await;
                match agent_lock.turn(&msg.content).await {
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
                            // Drop agent lock before streaming delivery.
                            drop(agent_lock);

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
                        };
                        let _ = bus.publish_outbound(outbound).await;
                    }
                }
            }
        })
    };

    // 8. Handle SIGINT gracefully.
    tokio::signal::ctrl_c().await?;
    tracing::info!("Received SIGINT, shutting down...");

    // Abort tasks.
    gateway_handle.abort();
    agent_loop.abort();
    if let Some(h) = _cron_handle {
        h.abort();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // Load config: try from config dir, fall back to defaults.
    let home_dir = FennecConfig::resolve_home(cli.config_dir.as_deref());
    let config_path = home_dir.join("config.toml");
    let config = if config_path.exists() {
        FennecConfig::load(&config_path)
            .with_context(|| format!("loading config from {}", config_path.display()))?
    } else {
        FennecConfig::default()
    };

    match cli.command {
        Commands::Agent { message, model } => {
            run_agent(config, home_dir, message, model).await?;
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
    }

    Ok(())
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
