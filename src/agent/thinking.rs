use serde_json::Value;

/// Controls how much "thinking" or reasoning effort the model should apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThinkingLevel {
    Off,
    Low,
    Medium,
    High,
    Max,
}

impl ThinkingLevel {
    /// Budget tokens for Anthropic's extended thinking feature.
    fn anthropic_budget_tokens(self) -> Option<u64> {
        match self {
            ThinkingLevel::Off => None,
            ThinkingLevel::Low => Some(1024),
            ThinkingLevel::Medium => Some(4096),
            ThinkingLevel::High => Some(10240),
            ThinkingLevel::Max => Some(32768),
        }
    }

    /// OpenAI reasoning_effort string.
    fn openai_effort(self) -> Option<&'static str> {
        match self {
            ThinkingLevel::Off => None,
            ThinkingLevel::Low => Some("low"),
            ThinkingLevel::Medium => Some("medium"),
            ThinkingLevel::High | ThinkingLevel::Max => Some("high"),
        }
    }

    /// Temperature adjustment for providers without native thinking support.
    /// Lower temperature = more deterministic / "careful" reasoning.
    fn fallback_temperature(self) -> Option<f64> {
        match self {
            ThinkingLevel::Off => None,
            ThinkingLevel::Low => Some(0.5),
            ThinkingLevel::Medium => Some(0.3),
            ThinkingLevel::High => Some(0.15),
            ThinkingLevel::Max => Some(0.05),
        }
    }
}

/// Scan a user message for a `/think:<level>` directive.
///
/// Returns the detected thinking level (if any) and the message with the
/// directive stripped out.
pub fn parse_thinking_directive(message: &str) -> (Option<ThinkingLevel>, String) {
    let directives = [
        ("/think:off", ThinkingLevel::Off),
        ("/think:low", ThinkingLevel::Low),
        ("/think:medium", ThinkingLevel::Medium),
        ("/think:high", ThinkingLevel::High),
        ("/think:max", ThinkingLevel::Max),
    ];

    for (token, level) in &directives {
        if let Some(pos) = message.find(token) {
            let mut cleaned = String::with_capacity(message.len());
            cleaned.push_str(&message[..pos]);
            cleaned.push_str(&message[pos + token.len()..]);
            let cleaned = cleaned.trim().to_string();
            return (Some(*level), cleaned);
        }
    }

    // Bare `/think` (no `:level`) — enable thinking at a sensible default
    // (High) so the command is recognized instead of being passed to the
    // agent as text (which replied "no command by that name"). The
    // `:level` forms are matched first, so this only fires for a standalone
    // `/think` token.
    if let Some(pos) = find_bare_think(message) {
        let token_len = "/think".len();
        let mut cleaned = String::with_capacity(message.len());
        cleaned.push_str(&message[..pos]);
        cleaned.push_str(&message[pos + token_len..]);
        let cleaned = cleaned.trim().to_string();
        return (Some(ThinkingLevel::High), cleaned);
    }

    (None, message.to_string())
}

/// Find a standalone `/think` token — not the `/think:` prefix of a
/// `:level` form (handled earlier) and not a longer word like `/thinking`.
/// Requires the character after `/think` to be absent or a non-`:`,
/// non-alphanumeric boundary.
fn find_bare_think(message: &str) -> Option<usize> {
    let mut search_from = 0;
    while let Some(rel) = message[search_from..].find("/think") {
        let pos = search_from + rel;
        let after = message[pos + "/think".len()..].chars().next();
        let is_boundary = match after {
            None => true,
            Some(c) => c != ':' && !c.is_ascii_alphanumeric(),
        };
        if is_boundary {
            return Some(pos);
        }
        search_from = pos + "/think".len();
    }
    None
}

/// Mutate a JSON request body to apply thinking / reasoning parameters based
/// on the provider.
pub fn apply_thinking_params(request_body: &mut Value, level: ThinkingLevel, provider_name: &str) {
    match provider_name {
        "anthropic" => {
            if let Some(budget) = level.anthropic_budget_tokens() {
                request_body["thinking"] = serde_json::json!({
                    "type": "enabled",
                    "budget_tokens": budget
                });
                // Anthropic requires temperature 1.0 when thinking is enabled.
                request_body["temperature"] = serde_json::json!(1.0);
            }
        }
        "openai" | "openrouter" => {
            if let Some(effort) = level.openai_effort() {
                request_body["reasoning_effort"] = serde_json::json!(effort);
            }
        }
        _ => {
            // Generic fallback: adjust temperature.
            if let Some(temp) = level.fallback_temperature() {
                request_body["temperature"] = serde_json::json!(temp);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_no_directive() {
        let (level, cleaned) = parse_thinking_directive("Hello world");
        assert!(level.is_none());
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn parse_high_directive() {
        let (level, cleaned) = parse_thinking_directive("/think:high Solve this complex problem");
        assert_eq!(level, Some(ThinkingLevel::High));
        assert_eq!(cleaned, "Solve this complex problem");
    }

    #[test]
    fn parse_bare_think_enables_high() {
        // Bare `/think` (no :level) is now recognized and defaults to High.
        let (level, cleaned) = parse_thinking_directive("/think");
        assert_eq!(level, Some(ThinkingLevel::High));
        assert_eq!(cleaned, "");

        let (level, cleaned) = parse_thinking_directive("/think please solve this");
        assert_eq!(level, Some(ThinkingLevel::High));
        assert_eq!(cleaned, "please solve this");
    }

    #[test]
    fn bare_think_does_not_shadow_level_forms_or_words() {
        // The `:level` forms still win over the bare token.
        let (level, _) = parse_thinking_directive("/think:off quiet please");
        assert_eq!(level, Some(ThinkingLevel::Off));
        let (level, _) = parse_thinking_directive("/think:max");
        assert_eq!(level, Some(ThinkingLevel::Max));
        // A longer word like "/thinking" must NOT trigger the bare token.
        let (level, cleaned) = parse_thinking_directive("/thinking about it");
        assert!(level.is_none());
        assert_eq!(cleaned, "/thinking about it");
    }

    #[test]
    fn parse_directive_in_middle() {
        let (level, cleaned) =
            parse_thinking_directive("Please /think:max analyze this carefully");
        assert_eq!(level, Some(ThinkingLevel::Max));
        assert_eq!(cleaned, "Please  analyze this carefully");
    }

    #[test]
    fn parse_off_directive() {
        let (level, cleaned) = parse_thinking_directive("/think:off Just answer quickly");
        assert_eq!(level, Some(ThinkingLevel::Off));
        assert_eq!(cleaned, "Just answer quickly");
    }

    #[test]
    fn apply_anthropic_thinking() {
        let mut body = serde_json::json!({
            "model": "claude-sonnet-4-6",
            "temperature": 0.7
        });
        apply_thinking_params(&mut body, ThinkingLevel::High, "anthropic");
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 10240);
        assert_eq!(body["temperature"], 1.0);
    }

    #[test]
    fn apply_openai_thinking() {
        let mut body = serde_json::json!({
            "model": "o1-preview"
        });
        apply_thinking_params(&mut body, ThinkingLevel::Medium, "openai");
        assert_eq!(body["reasoning_effort"], "medium");
    }

    #[test]
    fn apply_fallback_thinking() {
        let mut body = serde_json::json!({
            "model": "llama-3",
            "temperature": 0.7
        });
        apply_thinking_params(&mut body, ThinkingLevel::High, "ollama");
        assert_eq!(body["temperature"], 0.15);
    }

    #[test]
    fn apply_off_does_nothing() {
        let mut body = serde_json::json!({
            "model": "claude-sonnet-4-6",
            "temperature": 0.7
        });
        apply_thinking_params(&mut body, ThinkingLevel::Off, "anthropic");
        assert!(body.get("thinking").is_none());
        assert_eq!(body["temperature"], 0.7);
    }
}
