//! `fennec doctor` — diagnostics that surface common setup problems.
//!
//! Runs a set of independent checks and prints a pass/fail/warn line for
//! each. Exits 0 when everything is green, 1 when any check fails. Designed
//! to be the first thing a user runs when something feels off — "is my key
//! even valid? is memory initialized? is Plurum reachable?"

use std::path::Path;
use std::time::Duration;

use console::style;

use crate::config::schema::FennecConfig;
use crate::security::SecretStore;

/// Outcome of a single check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckStatus {
    Pass,
    /// Non-fatal (e.g. an optional feature isn't configured).
    Warn,
    Fail,
}

impl CheckStatus {
    pub fn glyph(&self, use_color: bool) -> String {
        if !use_color {
            return match self {
                Self::Pass => "[OK]".to_string(),
                Self::Warn => "[WARN]".to_string(),
                Self::Fail => "[FAIL]".to_string(),
            };
        }
        match self {
            Self::Pass => style("✓").green().to_string(),
            Self::Warn => style("⚠").yellow().to_string(),
            Self::Fail => style("✗").red().to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

impl CheckResult {
    pub fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Pass,
            detail: detail.into(),
        }
    }

    pub fn warn(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Warn,
            detail: detail.into(),
        }
    }

    pub fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Fail,
            detail: detail.into(),
        }
    }
}

/// Check that the config has required fields populated.
pub fn check_config(config: &FennecConfig) -> CheckResult {
    if config.identity.name.is_empty() {
        return CheckResult::fail("config", "identity.name is empty");
    }
    if config.provider.name.is_empty() {
        return CheckResult::fail("config", "provider.name is empty");
    }
    CheckResult::pass(
        "config",
        format!(
            "agent={} provider={} model={}",
            config.identity.name, config.provider.name, config.provider.model
        ),
    )
}

/// Check that a provider API key is available (config or env).
pub fn check_api_key(config: &FennecConfig, secret_store: &SecretStore) -> CheckResult {
    // Provider ollama doesn't need a key.
    if config.provider.name == "ollama" {
        return CheckResult::pass("api_key", "ollama requires no key");
    }

    // gemini-cloudcode authenticates via Google OAuth, not an API key.
    if matches!(config.provider.name.as_str(), "gemini-cloudcode" | "google-cloudcode") {
        return CheckResult::pass(
            "api_key",
            "OAuth provider — sign in with `fennec login --provider gemini-cloudcode`",
        );
    }

    if !config.provider.api_key.is_empty() {
        match secret_store.decrypt(&config.provider.api_key) {
            Ok(k) if !k.is_empty() => {
                return CheckResult::pass("api_key", "from config.toml");
            }
            Ok(_) => {
                return CheckResult::fail("api_key", "config.toml key decrypts to empty");
            }
            Err(e) => {
                return CheckResult::fail(
                    "api_key",
                    format!("failed to decrypt config key: {}", e),
                );
            }
        }
    }

    let env_var = match config.provider.name.as_str() {
        "anthropic" => "ANTHROPIC_API_KEY",
        "openai" => "OPENAI_API_KEY",
        "kimi" | "moonshot" => "KIMI_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "google" | "gemini" => "GEMINI_API_KEY",
        _ => "ANTHROPIC_API_KEY",
    };
    match std::env::var(env_var) {
        Ok(v) if !v.is_empty() => CheckResult::pass("api_key", format!("from {} env", env_var)),
        _ => CheckResult::fail(
            "api_key",
            format!("no key in config.toml or {} env var", env_var),
        ),
    }
}

/// Check that the memory DB is accessible.
pub fn check_memory_db(fennec_home: &Path) -> CheckResult {
    let db_path = fennec_home.join("memory").join("brain.db");
    if !db_path.exists() {
        return CheckResult::warn(
            "memory_db",
            format!("{} does not exist yet (will be created on first run)", db_path.display()),
        );
    }
    // Open + basic query.
    match rusqlite::Connection::open(&db_path) {
        Ok(conn) => {
            match conn.query_row::<i64, _, _>("SELECT 1", [], |row| row.get(0)) {
                Ok(_) => CheckResult::pass(
                    "memory_db",
                    format!("opened {}", db_path.display()),
                ),
                Err(e) => CheckResult::fail("memory_db", format!("query failed: {}", e)),
            }
        }
        Err(e) => CheckResult::fail("memory_db", format!("open failed: {}", e)),
    }
}

/// Check that the configured provider responds to a trivial request.
/// Skipped when no key is available — that's a separate check.
pub async fn check_provider_reachable(
    config: &FennecConfig,
    api_key: &str,
) -> CheckResult {
    let is_oauth_provider =
        matches!(config.provider.name.as_str(), "gemini-cloudcode" | "google-cloudcode");
    if is_oauth_provider {
        return CheckResult::warn(
            "provider_reachable",
            "skipped — OAuth provider; verify with `fennec login --provider gemini-cloudcode`",
        );
    }
    if api_key.is_empty() && config.provider.name != "ollama" {
        return CheckResult::warn(
            "provider_reachable",
            "skipped — no API key available",
        );
    }

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
    {
        Ok(c) => c,
        Err(e) => return CheckResult::fail("provider_reachable", format!("client build: {}", e)),
    };

    let (url, builder) = match config.provider.name.as_str() {
        "anthropic" => {
            let url = "https://api.anthropic.com/v1/models".to_string();
            let req = client
                .get(&url)
                .header("x-api-key", api_key)
                .header("anthropic-version", "2023-06-01");
            (url, req)
        }
        "openai" => {
            let url = "https://api.openai.com/v1/models".to_string();
            let req = client.get(&url).bearer_auth(api_key);
            (url, req)
        }
        "openrouter" => {
            let url = "https://openrouter.ai/api/v1/models".to_string();
            let req = client.get(&url).bearer_auth(api_key);
            (url, req)
        }
        "kimi" | "moonshot" => {
            let url = "https://api.moonshot.ai/v1/models".to_string();
            let req = client.get(&url).bearer_auth(api_key);
            (url, req)
        }
        "google" | "gemini" => {
            let base = if config.provider.base_url.is_empty() {
                "https://generativelanguage.googleapis.com/v1beta"
            } else {
                config.provider.base_url.trim_end_matches('/')
            };
            let url = format!("{}/models", base);
            let req = client.get(&url).header("x-goog-api-key", api_key);
            (url, req)
        }
        "ollama" => {
            let base = if config.provider.base_url.is_empty() {
                "http://localhost:11434"
            } else {
                config.provider.base_url.as_str()
            };
            let url = format!("{}/api/tags", base);
            let req = client.get(&url);
            (url, req)
        }
        other => {
            return CheckResult::warn(
                "provider_reachable",
                format!("unknown provider '{}' — skipping probe", other),
            );
        }
    };

    match builder.send().await {
        Ok(resp) if resp.status().is_success() => {
            CheckResult::pass("provider_reachable", format!("{} responded 2xx", url))
        }
        Ok(resp) => {
            let status = resp.status();
            CheckResult::fail(
                "provider_reachable",
                format!("{} responded {} — key may be invalid", url, status),
            )
        }
        Err(e) => CheckResult::fail("provider_reachable", format!("{}: {}", url, e)),
    }
}

/// Check Plurum /health endpoint when collective is enabled.
pub async fn check_plurum(config: &FennecConfig) -> CheckResult {
    if !config.collective.enabled {
        return CheckResult::warn("plurum", "collective disabled in config");
    }
    let base = if config.collective.base_url.is_empty() {
        "https://api.plurum.ai"
    } else {
        config.collective.base_url.as_str()
    };
    let url = format!("{}/health", base);
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => return CheckResult::fail("plurum", format!("client build: {}", e)),
    };
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            CheckResult::pass("plurum", format!("{} responded 2xx", url))
        }
        Ok(resp) => CheckResult::fail("plurum", format!("{} responded {}", url, resp.status())),
        Err(e) => CheckResult::fail("plurum", format!("{}: {}", url, e)),
    }
}

/// Check that expected fennec directories exist (or are creatable).
pub fn check_directories(fennec_home: &Path) -> CheckResult {
    let required = ["memory", "skills", "pairing"];
    let mut missing = Vec::new();
    for dir in required {
        let p = fennec_home.join(dir);
        if !p.exists() {
            missing.push(dir);
        }
    }
    if missing.is_empty() {
        CheckResult::pass("directories", format!("all present under {}", fennec_home.display()))
    } else {
        CheckResult::warn(
            "directories",
            format!("missing: {} (created on first run)", missing.join(", ")),
        )
    }
}

/// Render a single result as a line.
pub fn render_result(r: &CheckResult, use_color: bool) -> String {
    format!(
        "{} {}  {}",
        r.status.glyph(use_color),
        format!("{:<20}", r.name),
        r.detail
    )
}

/// Render a summary line given a list of results. Returns (line, any_failed).
pub fn render_summary(results: &[CheckResult], use_color: bool) -> (String, bool) {
    let pass = results
        .iter()
        .filter(|r| r.status == CheckStatus::Pass)
        .count();
    let warn = results
        .iter()
        .filter(|r| r.status == CheckStatus::Warn)
        .count();
    let fail = results
        .iter()
        .filter(|r| r.status == CheckStatus::Fail)
        .count();
    let any_failed = fail > 0;

    if !use_color {
        return (
            format!("{} passed, {} warned, {} failed", pass, warn, fail),
            any_failed,
        );
    }
    let line = format!(
        "{} passed, {} warned, {} failed",
        style(pass).green(),
        style(warn).yellow(),
        style(fail).red(),
    );
    (line, any_failed)
}

/// Run all diagnostic checks and return the results in order.
pub async fn run_all(
    config: &FennecConfig,
    fennec_home: &Path,
    secret_store: &SecretStore,
) -> Vec<CheckResult> {
    let mut results = Vec::new();

    results.push(check_config(config));
    results.push(check_directories(fennec_home));
    results.push(check_memory_db(fennec_home));

    let api_key_result = check_api_key(config, secret_store);
    let api_key_for_probe = match api_key_result.status {
        CheckStatus::Pass => {
            // Re-resolve; either config value or env var.
            if !config.provider.api_key.is_empty() {
                secret_store
                    .decrypt(&config.provider.api_key)
                    .unwrap_or_default()
            } else {
                let env_var = match config.provider.name.as_str() {
                    "anthropic" => "ANTHROPIC_API_KEY",
                    "openai" => "OPENAI_API_KEY",
                    "kimi" | "moonshot" => "KIMI_API_KEY",
                    "openrouter" => "OPENROUTER_API_KEY",
                    "google" | "gemini" => "GEMINI_API_KEY",
                    _ => "",
                };
                std::env::var(env_var).unwrap_or_default()
            }
        }
        _ => String::new(),
    };
    results.push(api_key_result);

    results.push(check_provider_reachable(config, &api_key_for_probe).await);
    results.push(check_plurum(config).await);

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_status_glyph_with_color_has_ansi() {
        let s = CheckStatus::Pass.glyph(true);
        assert!(s.contains("\x1b[") || s.contains("✓"));
    }

    #[test]
    fn check_status_glyph_ascii_fallback() {
        assert_eq!(CheckStatus::Pass.glyph(false), "[OK]");
        assert_eq!(CheckStatus::Warn.glyph(false), "[WARN]");
        assert_eq!(CheckStatus::Fail.glyph(false), "[FAIL]");
    }

    #[test]
    fn check_config_fails_on_empty_identity() {
        let mut cfg = FennecConfig::default();
        cfg.identity.name = String::new();
        cfg.provider.name = "anthropic".to_string();
        let r = check_config(&cfg);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn check_config_fails_on_empty_provider() {
        let mut cfg = FennecConfig::default();
        cfg.identity.name = "Test".to_string();
        cfg.provider.name = String::new();
        let r = check_config(&cfg);
        assert_eq!(r.status, CheckStatus::Fail);
    }

    #[test]
    fn check_config_passes_on_valid() {
        let mut cfg = FennecConfig::default();
        cfg.identity.name = "Fennec".to_string();
        cfg.provider.name = "anthropic".to_string();
        cfg.provider.model = "claude-sonnet-4-6".to_string();
        let r = check_config(&cfg);
        assert_eq!(r.status, CheckStatus::Pass);
        assert!(r.detail.contains("Fennec"));
        assert!(r.detail.contains("anthropic"));
    }

    #[test]
    fn check_memory_db_warns_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let r = check_memory_db(tmp.path());
        assert_eq!(r.status, CheckStatus::Warn);
    }

    #[test]
    fn check_directories_warns_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let r = check_directories(tmp.path());
        assert_eq!(r.status, CheckStatus::Warn);
        assert!(r.detail.contains("missing"));
    }

    #[test]
    fn check_directories_passes_when_all_present() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("memory")).unwrap();
        std::fs::create_dir_all(tmp.path().join("skills")).unwrap();
        std::fs::create_dir_all(tmp.path().join("pairing")).unwrap();
        let r = check_directories(tmp.path());
        assert_eq!(r.status, CheckStatus::Pass);
    }

    #[tokio::test]
    async fn check_plurum_warns_when_disabled() {
        let mut cfg = FennecConfig::default();
        cfg.collective.enabled = false;
        let r = check_plurum(&cfg).await;
        assert_eq!(r.status, CheckStatus::Warn);
    }

    #[test]
    fn render_result_includes_name_and_detail() {
        let r = CheckResult::pass("db", "ok");
        let line = render_result(&r, false);
        assert!(line.contains("db"));
        assert!(line.contains("ok"));
        assert!(line.contains("[OK]"));
    }

    #[test]
    fn render_summary_counts_each_status() {
        let results = vec![
            CheckResult::pass("a", ""),
            CheckResult::pass("b", ""),
            CheckResult::warn("c", ""),
            CheckResult::fail("d", ""),
        ];
        let (line, failed) = render_summary(&results, false);
        assert!(line.contains("2 passed"));
        assert!(line.contains("1 warned"));
        assert!(line.contains("1 failed"));
        assert!(failed);
    }

    #[test]
    fn render_summary_no_fail_when_all_pass() {
        let results = vec![CheckResult::pass("a", ""), CheckResult::warn("b", "")];
        let (_, failed) = render_summary(&results, false);
        assert!(!failed);
    }

    #[tokio::test]
    async fn check_api_key_ollama_always_passes() {
        let mut cfg = FennecConfig::default();
        cfg.provider.name = "ollama".to_string();
        let tmp = tempfile::tempdir().unwrap();
        let store = SecretStore::new(tmp.path().to_path_buf()).unwrap();
        let r = check_api_key(&cfg, &store);
        assert_eq!(r.status, CheckStatus::Pass);
    }
}
