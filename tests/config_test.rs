use fennec::config::schema::*;
use std::io::Write;

#[test]
fn test_default_identity() {
    let cfg = FennecConfig::default();
    assert_eq!(cfg.identity.name, "Fennec");
    assert_eq!(
        cfg.identity.persona,
        "Your personal AI agent — sharp, resourceful, and always on."
    );
}

#[test]
fn test_default_provider() {
    let cfg = FennecConfig::default();
    assert_eq!(cfg.provider.name, "anthropic");
    // Default tracks the current Anthropic flagship — was
    // `claude-sonnet-4-20250514`, bumped to `claude-sonnet-4-6` when
    // the new generation landed. This assertion lagged the bump.
    assert_eq!(cfg.provider.model, "claude-sonnet-4-6");
    assert_eq!(cfg.provider.api_key, "");
    assert!((cfg.provider.temperature - 0.7).abs() < f64::EPSILON);
    assert_eq!(cfg.provider.max_tokens, 8192);
}

#[test]
fn test_default_memory() {
    let cfg = FennecConfig::default();
    assert!(cfg.memory.db_path.is_none());
    assert!((cfg.memory.vector_weight - 0.7).abs() < f64::EPSILON);
    assert!((cfg.memory.keyword_weight - 0.3).abs() < f64::EPSILON);
    assert!((cfg.memory.half_life_days - 7.0).abs() < f64::EPSILON);
    assert_eq!(cfg.memory.cache_max, 10000);
    assert_eq!(cfg.memory.context_limit, 5);
}

#[test]
fn test_default_security() {
    let cfg = FennecConfig::default();
    assert_eq!(cfg.security.prompt_guard_action, "warn");
    assert!((cfg.security.prompt_guard_sensitivity - 0.7).abs() < f64::EPSILON);
    assert!(cfg.security.encrypt_secrets);
    assert!(cfg.security.command_allowlist.contains(&"git".to_string()));
    assert!(cfg.security.command_allowlist.contains(&"cargo".to_string()));
    assert_eq!(cfg.security.command_allowlist.len(), 30);
    assert!(cfg.security.forbidden_paths.contains(&"/etc".to_string()));
    assert_eq!(cfg.security.command_timeout_secs, 60);
}

#[test]
fn test_default_agent() {
    let cfg = FennecConfig::default();
    assert_eq!(cfg.agent.max_tool_iterations, 90);
    assert_eq!(cfg.agent.context_window, 200_000);
}

#[test]
fn test_toml_deserialization() {
    let toml_str = r#"
[identity]
name = "TestBot"
persona = "A test bot."

[provider]
name = "openai"
model = "gpt-4"
api_key = "sk-test"
temperature = 0.9
max_tokens = 4096

[memory]
db_path = "/tmp/test.db"
vector_weight = 0.5
keyword_weight = 0.5
half_life_days = 14.0
cache_max = 5000
context_limit = 10

[security]
prompt_guard_action = "block"
prompt_guard_sensitivity = 0.9
encrypt_secrets = false
command_allowlist = ["git", "ls"]
forbidden_paths = ["/etc"]
command_timeout_secs = 30

[agent]
max_tool_iterations = 25
context_window = 128000
"#;

    let cfg: FennecConfig = toml::from_str(toml_str).expect("failed to parse TOML");
    assert_eq!(cfg.identity.name, "TestBot");
    assert_eq!(cfg.provider.name, "openai");
    assert_eq!(cfg.provider.model, "gpt-4");
    assert_eq!(cfg.provider.api_key, "sk-test");
    assert!((cfg.provider.temperature - 0.9).abs() < f64::EPSILON);
    assert_eq!(cfg.provider.max_tokens, 4096);
    assert_eq!(cfg.memory.db_path, Some("/tmp/test.db".to_string()));
    assert!((cfg.memory.vector_weight - 0.5).abs() < f64::EPSILON);
    assert_eq!(cfg.security.prompt_guard_action, "block");
    assert!(!cfg.security.encrypt_secrets);
    assert_eq!(cfg.security.command_allowlist.len(), 2);
    assert_eq!(cfg.agent.max_tool_iterations, 25);
    assert_eq!(cfg.agent.context_window, 128000);
}

#[test]
fn test_toml_partial_deserialization() {
    // Only provide identity — everything else should use defaults
    let toml_str = r#"
[identity]
name = "Partial"
"#;

    let cfg: FennecConfig = toml::from_str(toml_str).expect("failed to parse partial TOML");
    assert_eq!(cfg.identity.name, "Partial");
    // Persona should be default since it wasn't specified
    assert_eq!(
        cfg.identity.persona,
        "Your personal AI agent — sharp, resourceful, and always on."
    );
    // Provider should be fully default
    assert_eq!(cfg.provider.name, "anthropic");
    assert_eq!(cfg.provider.max_tokens, 8192);
}

#[test]
fn test_load_from_file() {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let config_path = dir.path().join("fennec.toml");
    let mut file = std::fs::File::create(&config_path).expect("failed to create file");
    writeln!(
        file,
        r#"
[identity]
name = "FromFile"

[provider]
model = "claude-opus-4-20250514"
"#
    )
    .expect("failed to write");

    let cfg = FennecConfig::load(&config_path).expect("failed to load config");
    assert_eq!(cfg.identity.name, "FromFile");
    assert_eq!(cfg.provider.model, "claude-opus-4-20250514");
    // Defaults for unspecified fields
    assert_eq!(cfg.provider.name, "anthropic");
}

#[test]
fn test_resolve_home_override() {
    let home = FennecConfig::resolve_home(Some("/custom/path"));
    assert_eq!(home, std::path::PathBuf::from("/custom/path"));
}

#[test]
fn test_resolve_home_default() {
    // Clear FENNEC_HOME to test the default path
    // SAFETY: This test is single-threaded and does not rely on FENNEC_HOME elsewhere.
    unsafe {
        std::env::remove_var("FENNEC_HOME");
    }
    let home = FennecConfig::resolve_home(None);
    assert!(home.ends_with(".fennec"));
}
