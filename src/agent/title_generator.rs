//! Auto-generate a short title for a session from its opening exchange.
//!
//! Runs on the cheap auxiliary model (so it doesn't dirty the main provider's
//! prompt cache or pay primary-model rates) via [`TaskKind::Title`]. Best-effort
//! — returns `None` when no auxiliary provider is configured or the call fails;
//! callers spawn it in the background so it never blocks a turn.

use crate::providers::{AuxiliaryClient, ChatMessage, ChatRequest, TaskKind};

const TITLE_PROMPT: &str = "You generate a concise title for a conversation. \
Reply with ONLY a 3-6 word title — no quotes, no punctuation, no preamble, \
no markdown. Capture the topic, not the format.";

/// Max characters kept from each side of the opening exchange (keeps the
/// request small and cheap).
const SNIPPET_CHARS: usize = 500;

/// Generate a session title from the first user message + assistant reply.
/// Returns `None` if no auxiliary provider is available or on any error.
pub async fn generate_title(
    aux: &AuxiliaryClient,
    user_message: &str,
    assistant_response: &str,
) -> Option<String> {
    if !aux.is_available() {
        return None;
    }

    let user: String = user_message.chars().take(SNIPPET_CHARS).collect();
    let assistant: String = assistant_response.chars().take(SNIPPET_CHARS).collect();
    let content = format!("User: {user}\n\nAssistant: {assistant}");
    let messages = vec![ChatMessage::user(content)];

    let request = ChatRequest {
        system: Some(TITLE_PROMPT),
        messages: &messages,
        tools: None,
        max_tokens: 32,
        temperature: 0.3,
        thinking_level: crate::agent::thinking::ThinkingLevel::Off,
    };

    let response = aux.call_for(TaskKind::Title, request).await.ok()?;
    let title = sanitize_title(&response.content?);
    if title.is_empty() {
        None
    } else {
        Some(title)
    }
}

/// Clean a model-generated title: take the first line, strip wrapping quotes
/// and trailing punctuation, and cap the length so a chatty model can't
/// produce an oversized title.
pub fn sanitize_title(raw: &str) -> String {
    let line = raw.trim().lines().next().unwrap_or("").trim();
    let line = line.trim_matches(|c| c == '"' || c == '\'' || c == '`');
    let line = line.trim_end_matches(['.', '!', '?', ':', ';', ',']).trim();
    // Cap to a sane length (≈8 words / 60 chars) without splitting a char.
    line.chars().take(60).collect::<String>().trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_quotes_and_trailing_punctuation() {
        assert_eq!(sanitize_title("\"Fix the login bug\""), "Fix the login bug");
        assert_eq!(sanitize_title("Refactor the parser."), "Refactor the parser");
        assert_eq!(sanitize_title("  Plan the migration  "), "Plan the migration");
    }

    #[test]
    fn sanitize_takes_first_line_only() {
        assert_eq!(
            sanitize_title("Database schema design\nHere's why: ..."),
            "Database schema design"
        );
    }

    #[test]
    fn sanitize_caps_length() {
        let long = "a".repeat(200);
        assert_eq!(sanitize_title(&long).chars().count(), 60);
    }

    #[test]
    fn sanitize_handles_empty() {
        assert_eq!(sanitize_title(""), "");
        assert_eq!(sanitize_title("   \n  "), "");
    }
}
