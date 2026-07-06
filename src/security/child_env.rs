//! Scrubbed child-process environments.
//!
//! Subprocesses spawned on behalf of the model (the `code_exec` runners)
//! must not inherit Fennec's secrets: the parent process env carries
//! provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) and
//! whatever else the user exported, and a single `print(os.environ)` —
//! or any library that phones home with its env — exfiltrates all of it.
//!
//! Model: deny-first, then allowlist. A variable survives only when ALL
//! of these hold:
//!   1. its name contains no secret substring (`KEY`, `TOKEN`, `SECRET`,
//!      `PASSWORD`, `PASSWD`, `CREDENTIAL`, `AUTH`, `DSN`, `WEBHOOK`) —
//!      checked first so a safe prefix can never smuggle a secret
//!      (`PATH_TOKEN` stays out), and
//!   2. its name starts with a safe operational prefix (`PATH`, `HOME`,
//!      `LANG`, …), is an exact allowed Fennec runtime-location var, or
//!      (on Windows) is one of the OS-essential names the CRT needs.
//!
//! Everything else — including the broad `FENNEC_*` namespace — is
//! dropped. Children that legitimately need a blocked variable should
//! receive it explicitly at the call site, not by inheritance.

/// Name substrings that mark a variable as secret-bearing. Checked
/// case-insensitively against the variable NAME (not value), before any
/// allow rule runs.
const SECRET_SUBSTRINGS: [&str; 9] = [
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "DSN",
    "WEBHOOK",
];

/// Safe operational name prefixes a child process may inherit.
const SAFE_PREFIXES: [&str; 15] = [
    "PATH", "HOME", "USER", "LANG", "LC_", "TERM", "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
    "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA",
];

/// Fennec runtime-location variables a child may need by exact name.
/// None of these match `SECRET_SUBSTRINGS`; the broad `FENNEC_` prefix is
/// deliberately NOT allowed (config-bearing vars without a secret-looking
/// name would leak through).
const FENNEC_CHILD_ALLOWED: [&str; 3] = ["FENNEC_HOME", "FENNEC_PROFILE", "FENNEC_CONFIG"];

/// Windows-only OS/CRT essentials. Without these even stdlib calls fail
/// (Winsock needs SYSTEMROOT; subprocess needs COMSPEC). Well-known OS
/// paths, not secrets; the secret-substring deny still runs first.
#[cfg(windows)]
const WINDOWS_ESSENTIAL: [&str; 21] = [
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "OS",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "PUBLIC",
    "ALLUSERSPROFILE",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMW6432",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "USERDOMAIN",
    "USERNAME",
    "HOMEDRIVE",
    "HOMEPATH",
];

/// Whether a single variable (by name) survives scrubbing.
fn is_allowed_var(name: &str) -> bool {
    let upper = name.to_uppercase();

    // Deny first: a secret-looking name never survives, regardless of
    // any allow rule below.
    if SECRET_SUBSTRINGS.iter().any(|s| upper.contains(s)) {
        return false;
    }

    if SAFE_PREFIXES.iter().any(|p| upper.starts_with(p)) {
        return true;
    }
    if FENNEC_CHILD_ALLOWED.iter().any(|v| *v == upper) {
        return true;
    }
    #[cfg(windows)]
    if WINDOWS_ESSENTIAL.iter().any(|v| *v == upper) {
        return true;
    }
    false
}

/// Build the scrubbed environment for a model-spawned child process from
/// the current process env.
pub fn scrubbed_child_env() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(name, _)| is_allowed_var(name))
        .collect()
}

/// Apply the scrubbed env to a tokio command: clear everything, then set
/// only the surviving variables.
pub fn apply_scrubbed_env(cmd: &mut tokio::process::Command) {
    cmd.env_clear();
    for (k, v) in scrubbed_child_env() {
        cmd.env(k, v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_api_keys_blocked() {
        assert!(!is_allowed_var("ANTHROPIC_API_KEY"));
        assert!(!is_allowed_var("OPENAI_API_KEY"));
        assert!(!is_allowed_var("GEMINI_API_KEY"));
        assert!(!is_allowed_var("GITHUB_TOKEN"));
        assert!(!is_allowed_var("DATABASE_PASSWORD"));
        assert!(!is_allowed_var("SLACK_WEBHOOK"));
        assert!(!is_allowed_var("PGPASSWORD"));
        assert!(!is_allowed_var("SENTRY_DSN"));
    }

    #[test]
    fn secret_substring_beats_safe_prefix() {
        // Deny runs first: a safe prefix can't smuggle a secret-named var.
        assert!(!is_allowed_var("PATH_TOKEN"));
        assert!(!is_allowed_var("HOME_SECRET"));
        assert!(!is_allowed_var("XDG_AUTH_FILE"));
    }

    #[test]
    fn operational_vars_allowed() {
        assert!(is_allowed_var("PATH"));
        assert!(is_allowed_var("HOME"));
        assert!(is_allowed_var("LANG"));
        assert!(is_allowed_var("LC_ALL"));
        assert!(is_allowed_var("TERM"));
        assert!(is_allowed_var("TMPDIR"));
        assert!(is_allowed_var("SHELL"));
        assert!(is_allowed_var("XDG_CACHE_HOME"));
        assert!(is_allowed_var("VIRTUAL_ENV"));
        assert!(is_allowed_var("PYTHONPATH"));
    }

    #[test]
    fn fennec_namespace_blocked_except_exact_allowlist() {
        assert!(is_allowed_var("FENNEC_HOME"));
        assert!(is_allowed_var("FENNEC_PROFILE"));
        assert!(!is_allowed_var("FENNEC_BASE_URL"));
        assert!(!is_allowed_var("FENNEC_CRON_MAX_PARALLEL"));
    }

    #[test]
    fn case_insensitive_matching() {
        assert!(!is_allowed_var("anthropic_api_key"));
        assert!(is_allowed_var("path"));
    }

    #[test]
    fn unknown_vars_blocked_by_default() {
        assert!(!is_allowed_var("AWS_PROFILE")); // not in any allow rule
        assert!(!is_allowed_var("RANDOM_APP_CONFIG"));
    }

    #[test]
    fn scrubbed_env_drops_injected_secret() {
        // SAFETY: test-scoped env mutation; removed before assertion ends.
        unsafe {
            std::env::set_var("FENNEC_TEST_FAKE_API_KEY", "sk-fake");
        }
        let env = scrubbed_child_env();
        let leaked = env.iter().any(|(k, _)| k == "FENNEC_TEST_FAKE_API_KEY");
        unsafe {
            std::env::remove_var("FENNEC_TEST_FAKE_API_KEY");
        }
        assert!(!leaked, "secret-named var must not survive scrubbing");
    }
}
