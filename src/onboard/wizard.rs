use crate::onboard::frame::{existing_config_at, FinalSummary, StepSummary, WizardFrame};
use crate::security::fs::{create_dir_private, write_secure};
use console::style;
use dialoguer::{Confirm, Input, Select};

/// Run the interactive setup wizard that creates `~/.fennec/config.toml`.
///
/// The framed flow renders each question on an alternate terminal screen,
/// collapsing completed steps to dim summary lines. Set
/// `FENNEC_CLASSIC_WIZARD=1` to restore the original unframed behavior
/// (rollback escape hatch).
pub fn run_wizard(fennec_home: &std::path::Path) -> anyhow::Result<()> {
    let classic = std::env::var("FENNEC_CLASSIC_WIZARD")
        .ok()
        .is_some_and(|v| !v.is_empty());
    if classic {
        return run_wizard_classic(fennec_home);
    }

    // Pre-wizard: prompt before overwriting an existing config.
    if existing_config_at(fennec_home) {
        println!();
        println!("  {}", style("Fennec Setup").cyan().bold());
        println!("  {}", style("─".repeat(40)).dim());
        println!(
            "  Config already exists at {}",
            fennec_home.join("config.toml").display()
        );
        println!();
        let choices = vec!["Cancel (keep existing)", "Replace it"];
        let idx = Select::new().items(&choices).default(0).interact()?;
        if idx == 0 {
            println!("  {} Kept existing config.", style("✓").green());
            return Ok(());
        }
    }

    let mut frame = WizardFrame::new(6);
    frame.start()?;

    // Step 1: Provider
    frame.begin_step(1, "Provider");
    frame.redraw()?;
    let providers = vec![
        "Anthropic (Claude)",
        "OpenAI (GPT-4o)",
        "Kimi (Moonshot)",
        "OpenRouter (any model)",
        "Ollama (local)",
    ];
    let provider_idx = Select::new()
        .with_prompt("Choose your LLM provider")
        .items(&providers)
        .default(0)
        .interact()?;

    let (provider_name, default_model, env_var) = match provider_idx {
        0 => ("anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
        1 => ("openai", "gpt-4o", "OPENAI_API_KEY"),
        2 => ("kimi", "kimi-k2.5", "KIMI_API_KEY"),
        3 => (
            "openrouter",
            "anthropic/claude-sonnet-4",
            "OPENROUTER_API_KEY",
        ),
        4 => ("ollama", "llama3.1", ""),
        _ => ("anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    };
    frame.complete_step(StepSummary::done(
        "Provider",
        providers[provider_idx].to_string(),
    ));

    // Step 2: Authentication
    frame.begin_step(2, "Authentication");
    frame.redraw()?;
    let (api_key, auth_summary) = if provider_name == "anthropic" {
        let auth_methods = vec![
            "OAuth (sign in with your Claude account — free with subscription)",
            "API key (from console.anthropic.com — pay per use)",
        ];
        let auth_idx = Select::new()
            .with_prompt("How do you want to authenticate")
            .items(&auth_methods)
            .default(0)
            .interact()?;
        if auth_idx == 0 {
            println!();
            println!("  {}", style("Starting OAuth login...").dim());
            match crate::auth::run_oauth_login(fennec_home) {
                Ok(_creds) => {
                    println!("  {} Authenticated with Claude!", style("✓").green());
                    (String::new(), "OAuth".to_string())
                }
                Err(e) => {
                    println!("  {} OAuth failed: {}", style("✗").yellow(), e);
                    println!("  {}", style("Falling back to API key...").dim());
                    let key = Input::<String>::new()
                        .with_prompt("Enter your Anthropic API key")
                        .allow_empty(true)
                        .interact_text()?;
                    let summary = if key.is_empty() {
                        "skipped".to_string()
                    } else {
                        "API key".to_string()
                    };
                    (key, summary)
                }
            }
        } else {
            let key = if let Ok(k) = std::env::var(env_var) {
                println!("  {} Using {} from environment", style("✓").green(), env_var);
                k
            } else {
                Input::<String>::new()
                    .with_prompt("Enter your Anthropic API key")
                    .allow_empty(true)
                    .interact_text()?
            };
            let summary = if key.is_empty() {
                "skipped".to_string()
            } else {
                "API key".to_string()
            };
            (key, summary)
        }
    } else if !env_var.is_empty() {
        let key = if let Ok(k) = std::env::var(env_var) {
            println!("  {} Using {} from environment", style("✓").green(), env_var);
            k
        } else {
            Input::<String>::new()
                .with_prompt("Enter your API key")
                .allow_empty(true)
                .interact_text()?
        };
        let summary = if key.is_empty() {
            "skipped".to_string()
        } else {
            "API key".to_string()
        };
        (key, summary)
    } else {
        (String::new(), "Ollama (no key)".to_string())
    };
    frame.complete_step(StepSummary::done("Authentication", auth_summary));

    // Step 3: Agent name
    frame.begin_step(3, "Agent name");
    frame.redraw()?;
    let agent_name: String = Input::new()
        .with_prompt("Agent name")
        .default("Fennec".to_string())
        .interact_text()?;
    frame.complete_step(StepSummary::done("Agent name", agent_name.clone()));

    // Step 4: Telegram
    frame.begin_step(4, "Telegram");
    frame.redraw()?;
    let setup_telegram = Confirm::new()
        .with_prompt("Set up Telegram?")
        .default(false)
        .interact()?;

    let (telegram_token, telegram_user_id, telegram_configured) = if setup_telegram {
        println!(
            "  {}",
            style("Create a bot via @BotFather on Telegram").dim()
        );
        let token: String = Input::new()
            .with_prompt("Telegram bot token")
            .interact_text()?;
        let user_id: String = Input::new()
            .with_prompt("Your Telegram user ID (message @userinfobot on Telegram to find it)")
            .allow_empty(true)
            .interact_text()?;
        (token, user_id, true)
    } else {
        (String::new(), String::new(), false)
    };
    if telegram_configured {
        frame.complete_step(StepSummary::done("Telegram", "Configured"));
    } else {
        frame.complete_step(StepSummary::skipped("Telegram"));
    }

    // Step 5: Discord
    frame.begin_step(5, "Discord");
    frame.redraw()?;
    let setup_discord = Confirm::new()
        .with_prompt("Set up Discord?")
        .default(false)
        .interact()?;

    let (discord_token, discord_configured) = if setup_discord {
        let token = Input::<String>::new()
            .with_prompt("Discord bot token")
            .interact_text()?;
        (token, true)
    } else {
        (String::new(), false)
    };
    if discord_configured {
        frame.complete_step(StepSummary::done("Discord", "Configured"));
    } else {
        frame.complete_step(StepSummary::skipped("Discord"));
    }

    // Step 6: Collective (Plurum)
    frame.begin_step(6, "Collective");
    frame.redraw()?;
    let enable_collective = Confirm::new()
        .with_prompt("Enable collective intelligence (Plurum)?")
        .default(true)
        .interact()?;

    let (plurum_key, collective_summary) = if enable_collective {
        println!(
            "  {} Registering with Plurum...",
            style("\u{27F3}").yellow()
        );
        match auto_register_plurum(&agent_name) {
            Ok(key) => {
                println!(
                    "  {} Registered! Key: {}...{}",
                    style("✓").green(),
                    &key[..key.len().min(16)],
                    &key[key.len().saturating_sub(4)..]
                );
                (key, "Registered with Plurum".to_string())
            }
            Err(e) => {
                println!("  {} Auto-register failed: {}", style("✗").yellow(), e);
                let key = Input::<String>::new()
                    .with_prompt("Plurum API key (or Enter to skip)")
                    .allow_empty(true)
                    .interact_text()?;
                let summary = if key.is_empty() {
                    String::new()
                } else {
                    "Manual key".to_string()
                };
                (key, summary)
            }
        }
    } else {
        (String::new(), String::new())
    };
    if collective_summary.is_empty() {
        frame.complete_step(StepSummary::skipped("Collective"));
    } else {
        frame.complete_step(StepSummary::done("Collective", collective_summary));
    }

    // Write config (byte-identical output to the classic path).
    let config = build_config_toml(
        &agent_name,
        provider_name,
        default_model,
        &api_key,
        &telegram_token,
        &telegram_user_id,
        &discord_token,
        &plurum_key,
    );

    std::fs::create_dir_all(fennec_home)?;
    // memory/, skills/, pairing/ hold secrets or secret-adjacent state;
    // create them with 0700 on Unix so other local users can't enumerate.
    create_dir_private(&fennec_home.join("memory"))?;
    create_dir_private(&fennec_home.join("skills"))?;
    create_dir_private(&fennec_home.join("pairing"))?;

    // config.toml contains API keys / bot tokens; write atomically with 0600
    // from the start so there is no readable-for-a-moment window.
    let config_path = fennec_home.join("config.toml");
    write_secure(&config_path, config.as_bytes())?;

    frame.finish(FinalSummary {
        config_path,
        quick_start: vec![
            ("fennec agent".to_string(), "Interactive chat".to_string()),
            (
                "fennec agent -m 'Hello'".to_string(),
                "Single message".to_string(),
            ),
            (
                "fennec gateway".to_string(),
                "Start all channels".to_string(),
            ),
        ],
    })?;

    Ok(())
}

/// Classic unframed wizard — byte-identical behavior to pre-frame main.
/// Invoked via `FENNEC_CLASSIC_WIZARD=1` as an instant rollback.
fn run_wizard_classic(fennec_home: &std::path::Path) -> anyhow::Result<()> {
    println!();
    println!("  {}", style("Welcome to Fennec").bold().cyan());
    println!(
        "  {}",
        style("The fastest AI agent with collective intelligence").dim()
    );
    println!();

    // Step 1: Provider selection
    let providers = vec![
        "Anthropic (Claude)",
        "OpenAI (GPT-4o)",
        "Kimi (Moonshot)",
        "OpenRouter (any model)",
        "Ollama (local)",
    ];
    let provider_idx = Select::new()
        .with_prompt("Choose your LLM provider")
        .items(&providers)
        .default(0)
        .interact()?;

    let (provider_name, default_model, env_var) = match provider_idx {
        0 => ("anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
        1 => ("openai", "gpt-4o", "OPENAI_API_KEY"),
        2 => ("kimi", "kimi-k2.5", "KIMI_API_KEY"),
        3 => (
            "openrouter",
            "anthropic/claude-sonnet-4",
            "OPENROUTER_API_KEY",
        ),
        4 => ("ollama", "llama3.1", ""),
        _ => ("anthropic", "claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
    };

    let api_key = if provider_name == "anthropic" {
        let auth_methods = vec![
            "OAuth (sign in with your Claude account — free with subscription)",
            "API key (from console.anthropic.com — pay per use)",
        ];
        let auth_idx = Select::new()
            .with_prompt("How do you want to authenticate")
            .items(&auth_methods)
            .default(0)
            .interact()?;

        if auth_idx == 0 {
            println!();
            println!("  {}", style("Starting OAuth login...").dim());
            match crate::auth::run_oauth_login(fennec_home) {
                Ok(_creds) => {
                    println!("  {} Authenticated with Claude!", style("✓").green());
                    String::new()
                }
                Err(e) => {
                    println!("  {} OAuth failed: {}", style("✗").red(), e);
                    println!("  {}", style("Falling back to API key...").dim());
                    Input::<String>::new()
                        .with_prompt("Enter your Anthropic API key")
                        .allow_empty(true)
                        .interact_text()?
                }
            }
        } else if let Ok(key) = std::env::var(env_var) {
            println!("  {} Using {} from environment", style("✓").green(), env_var);
            key
        } else {
            Input::<String>::new()
                .with_prompt("Enter your Anthropic API key")
                .allow_empty(true)
                .interact_text()?
        }
    } else if !env_var.is_empty() {
        if let Ok(key) = std::env::var(env_var) {
            println!("  {} Using {} from environment", style("✓").green(), env_var);
            key
        } else {
            Input::<String>::new()
                .with_prompt("Enter your API key")
                .allow_empty(true)
                .interact_text()?
        }
    } else {
        String::new()
    };

    let agent_name: String = Input::new()
        .with_prompt("Agent name")
        .default("Fennec".to_string())
        .interact_text()?;

    let setup_telegram = Confirm::new()
        .with_prompt("Set up Telegram?")
        .default(false)
        .interact()?;

    let (telegram_token, telegram_user_id) = if setup_telegram {
        println!(
            "  {}",
            style("Create a bot via @BotFather on Telegram").dim()
        );
        let token: String = Input::new()
            .with_prompt("Telegram bot token")
            .interact_text()?;
        let user_id: String = Input::new()
            .with_prompt("Your Telegram user ID (message @userinfobot on Telegram to find it)")
            .allow_empty(true)
            .interact_text()?;
        (token, user_id)
    } else {
        (String::new(), String::new())
    };

    let setup_discord = Confirm::new()
        .with_prompt("Set up Discord?")
        .default(false)
        .interact()?;

    let discord_token = if setup_discord {
        Input::<String>::new()
            .with_prompt("Discord bot token")
            .interact_text()?
    } else {
        String::new()
    };

    let enable_collective = Confirm::new()
        .with_prompt("Enable collective intelligence (Plurum)?")
        .default(true)
        .interact()?;

    let plurum_key = if enable_collective {
        println!(
            "  {} Registering with Plurum...",
            style("\u{27F3}").yellow()
        );
        match auto_register_plurum(&agent_name) {
            Ok(key) => {
                println!(
                    "  {} Registered! Key: {}...{}",
                    style("✓").green(),
                    &key[..key.len().min(16)],
                    &key[key.len().saturating_sub(4)..]
                );
                key
            }
            Err(e) => {
                println!("  {} Auto-register failed: {}", style("✗").red(), e);
                Input::<String>::new()
                    .with_prompt("Plurum API key (or Enter to skip)")
                    .allow_empty(true)
                    .interact_text()?
            }
        }
    } else {
        String::new()
    };

    let config = build_config_toml(
        &agent_name,
        provider_name,
        default_model,
        &api_key,
        &telegram_token,
        &telegram_user_id,
        &discord_token,
        &plurum_key,
    );

    std::fs::create_dir_all(fennec_home)?;
    create_dir_private(&fennec_home.join("memory"))?;
    create_dir_private(&fennec_home.join("skills"))?;
    create_dir_private(&fennec_home.join("pairing"))?;

    let config_path = fennec_home.join("config.toml");
    write_secure(&config_path, config.as_bytes())?;

    println!();
    println!(
        "  {} Config written to {}",
        style("✓").green(),
        config_path.display()
    );
    println!();
    println!("  {}", style("Quick start:").bold());
    println!("    fennec agent               # Interactive chat");
    println!("    fennec agent -m 'Hello'    # Single message");
    println!("    fennec gateway             # Start all channels");
    println!();

    Ok(())
}

fn auto_register_plurum(agent_name: &str) -> anyhow::Result<String> {
    let username = format!(
        "{}-{}",
        agent_name.to_lowercase().replace(' ', "-"),
        &uuid::Uuid::new_v4().to_string()[..8]
    );
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("https://api.plurum.ai/api/v1/agents/register")
        .json(&serde_json::json!({"name": agent_name, "username": username}))
        .timeout(std::time::Duration::from_secs(10))
        .send()?;
    let json: serde_json::Value = resp.json()?;
    json["api_key"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| anyhow::anyhow!("No api_key in response"))
}

/// Serialize a string as a TOML basic string (quoted, with proper escapes).
///
/// Hand-rolled rather than pulling in a full `toml::Value::to_string` because
/// we want to control the exact quoting (basic strings, not literal strings)
/// and avoid the top-level-table requirement of `toml::to_string`. Matches
/// TOML v1.0 basic-string rules.
fn toml_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn build_config_toml(
    name: &str,
    provider: &str,
    model: &str,
    api_key: &str,
    telegram_token: &str,
    telegram_user_id: &str,
    discord_token: &str,
    plurum_key: &str,
) -> String {
    // User-typed values get escaped via toml_str so stray quotes, backslashes,
    // or newlines can't break out of their field and inject arbitrary TOML
    // sections.
    let name_lit = toml_str(name);
    let provider_lit = toml_str(provider);
    let model_lit = toml_str(model);
    let api_key_lit = toml_str(api_key);
    let telegram_token_lit = toml_str(telegram_token);
    let discord_token_lit = toml_str(discord_token);
    let plurum_key_lit = toml_str(plurum_key);

    let allowed_users_line = if telegram_user_id.is_empty() {
        "allowed_users = []".to_string()
    } else {
        format!("allowed_users = [{}]", toml_str(telegram_user_id))
    };

    format!(
        r#"[identity]
name = {name_lit}
persona = "A fast, helpful AI assistant with collective intelligence."

[provider]
name = {provider_lit}
model = {model_lit}
api_key = {api_key_lit}
base_url = ""
temperature = 0.7
max_tokens = 8192

[memory]
vector_weight = 0.7
keyword_weight = 0.3
half_life_days = 7.0
consolidation_enabled = true

[security]
prompt_guard_action = "warn"
prompt_guard_sensitivity = 0.7
encrypt_secrets = true
command_timeout_secs = 60

[agent]
max_tool_iterations = 90
context_window = 200000

[channels.telegram]
enabled = {telegram_enabled}
token = {telegram_token_lit}
{allowed_users_line}

[channels.discord]
enabled = {discord_enabled}
token = {discord_token_lit}

[channels.slack]
enabled = false
bot_token = ""
app_token = ""

[gateway]
host = "0.0.0.0"
port = 8990

[cron]
enabled = false

[collective]
enabled = {collective_enabled}
api_key = {plurum_key_lit}
base_url = "https://api.plurum.ai"
publish_enabled = true
search_enabled = true
"#,
        name_lit = name_lit,
        provider_lit = provider_lit,
        model_lit = model_lit,
        api_key_lit = api_key_lit,
        telegram_token_lit = telegram_token_lit,
        allowed_users_line = allowed_users_line,
        telegram_enabled = if telegram_token.is_empty() {
            "false"
        } else {
            "true"
        },
        discord_token_lit = discord_token_lit,
        discord_enabled = if discord_token.is_empty() {
            "false"
        } else {
            "true"
        },
        plurum_key_lit = plurum_key_lit,
        collective_enabled = if plurum_key.is_empty() {
            "false"
        } else {
            "true"
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_config_toml_defaults() {
        let config = build_config_toml(
            "TestBot",
            "anthropic",
            "claude-sonnet-4-6",
            "sk-test-key",
            "",
            "",
            "",
            "",
        );
        assert!(config.contains("name = \"TestBot\""));
        assert!(config.contains("name = \"anthropic\""));
        assert!(config.contains("api_key = \"sk-test-key\""));
        assert!(config.contains("[channels.telegram]\nenabled = false"));
        assert!(config.contains("[channels.discord]\nenabled = false"));
        assert!(config.contains("[collective]\nenabled = false"));
    }

    #[test]
    fn test_build_config_toml_with_channels() {
        let config = build_config_toml(
            "Fennec",
            "openai",
            "gpt-4o",
            "sk-openai",
            "123:ABC",
            "987654321",
            "discord-token",
            "plurum-key",
        );
        assert!(config.contains("[channels.telegram]\nenabled = true"));
        assert!(config.contains("token = \"123:ABC\""));
        assert!(config.contains("allowed_users = [\"987654321\"]"));
        assert!(config.contains("[channels.discord]\nenabled = true"));
        assert!(config.contains("token = \"discord-token\""));
        assert!(config.contains("[collective]\nenabled = true"));
        assert!(config.contains("api_key = \"plurum-key\""));
    }

    #[test]
    fn toml_str_quotes_plain_ascii() {
        assert_eq!(toml_str("hello"), "\"hello\"");
        assert_eq!(toml_str(""), "\"\"");
    }

    #[test]
    fn toml_str_escapes_quote_and_backslash() {
        assert_eq!(toml_str("a\"b"), "\"a\\\"b\"");
        assert_eq!(toml_str("a\\b"), "\"a\\\\b\"");
    }

    #[test]
    fn toml_str_escapes_control_chars() {
        assert_eq!(toml_str("a\nb"), "\"a\\nb\"");
        assert_eq!(toml_str("a\tb"), "\"a\\tb\"");
        assert_eq!(toml_str("a\rb"), "\"a\\rb\"");
        assert_eq!(toml_str("a\x00b"), "\"a\\u0000b\"");
    }

    #[test]
    fn build_config_toml_escapes_injection_in_agent_name() {
        // An attacker-controlled or accidentally-pasted agent_name containing
        // a closing quote and a fake TOML section MUST NOT inject a new
        // section. Re-parsing the produced TOML must preserve the name
        // verbatim.
        let hostile = r#"x"]
[provider]
api_key = "stolen"
[identity]
name = "gotcha"#;
        let config = build_config_toml(
            hostile,
            "anthropic",
            "claude-sonnet-4-20250514",
            "sk-real",
            "",
            "",
            "",
            "",
        );
        let parsed: toml::Value =
            toml::from_str(&config).expect("escaped config must be valid TOML");
        let parsed_name = parsed
            .get("identity")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .expect("identity.name must be present");
        assert_eq!(
            parsed_name, hostile,
            "agent_name must round-trip verbatim — injection attempt was not escaped"
        );
        // Provider key must still be the one we passed, not the injected one.
        let parsed_api_key = parsed
            .get("provider")
            .and_then(|v| v.get("api_key"))
            .and_then(|v| v.as_str())
            .expect("provider.api_key must be present");
        assert_eq!(parsed_api_key, "sk-real");
    }

    #[test]
    fn build_config_toml_escapes_injection_in_telegram_user_id() {
        // telegram_user_id lands inside allowed_users = [...] — same escape
        // path, different call site.
        let hostile = r#"1"], other = [1"#;
        let config = build_config_toml(
            "Fennec",
            "anthropic",
            "m",
            "",
            "tok",
            hostile,
            "",
            "",
        );
        let parsed: toml::Value =
            toml::from_str(&config).expect("escaped config must be valid TOML");
        let users = parsed
            .get("channels")
            .and_then(|v| v.get("telegram"))
            .and_then(|v| v.get("allowed_users"))
            .and_then(|v| v.as_array())
            .expect("allowed_users must be an array");
        assert_eq!(users.len(), 1);
        assert_eq!(users[0].as_str(), Some(hostile));
    }

    #[test]
    fn build_config_toml_roundtrips_to_schema() {
        // End-to-end: the wizard's output must parse as a FennecConfig.
        let config = build_config_toml(
            "Fennec",
            "anthropic",
            "claude-sonnet-4-20250514",
            "sk-test",
            "123:ABC",
            "987654321",
            "dc-token",
            "plrm_test",
        );
        let cfg: crate::config::FennecConfig =
            toml::from_str(&config).expect("must parse as FennecConfig");
        assert_eq!(cfg.identity.name, "Fennec");
        assert_eq!(cfg.provider.api_key, "sk-test");
        assert_eq!(cfg.channels.telegram.token, "123:ABC");
        assert_eq!(cfg.channels.telegram.allowed_users, vec!["987654321"]);
        assert_eq!(cfg.collective.api_key, "plrm_test");
    }
}
