use anyhow::{Context, Result};
use serde::Deserialize;

use crate::collective::scrub::scrub_text;
use crate::memory::traits::{Memory, MemoryCategory, MemoryEntry};
use crate::providers::traits::{ChatMessage, ChatRequest, Provider};

/// Cap on every content string we persist out of the consolidator.
/// Model-generated daily summaries of a 10-message conversation don't need
/// to be longer than this; oversized model output going into persistent
/// memory bloats the DB and blows out downstream prompts. Separate caps
/// for three classes of content that have different reasonable sizes.
const MAX_DAILY_SUMMARY_BYTES: usize = 8 * 1024;
const MAX_FACT_CONTENT_BYTES: usize = 4 * 1024;
const MAX_FALLBACK_BYTES: usize = 16 * 1024;
/// Keys are short identifiers (think slugs). Extractions that return
/// huge keys are either buggy or hostile — either way, cap.
const MAX_KEY_BYTES: usize = 256;

/// Extracts structured information from conversations and stores it in memory.
pub struct MemoryConsolidator {
    provider: std::sync::Arc<dyn Provider>,
}

/// JSON schema returned by the extraction prompt.
#[derive(Debug, Deserialize)]
struct ExtractionResult {
    daily_summary: String,
    #[serde(default)]
    core_facts: Vec<CoreFact>,
}

#[derive(Debug, Deserialize)]
struct CoreFact {
    key: String,
    content: String,
}

impl MemoryConsolidator {
    /// Create a consolidator that uses the given (cheap) provider for extraction.
    pub fn new(provider: Box<dyn Provider>) -> Self {
        Self {
            provider: std::sync::Arc::from(provider),
        }
    }

    /// Like [`Self::new`] but shares an existing provider handle — used
    /// by the agent's session-end consolidation, which holds its
    /// provider behind an `Arc`.
    pub fn from_arc(provider: std::sync::Arc<dyn Provider>) -> Self {
        Self { provider }
    }

    /// Analyse the most recent conversation messages and extract structured
    /// information, storing daily summaries and core facts into memory.
    ///
    /// Every string that lands in the memory store is run through
    /// [`scrub_text`] and capped at a per-field byte limit first, so that
    /// secrets the model echoed into its output (or a misbehaving model's
    /// multi-megabyte response) can't become persistent leaks.
    pub async fn consolidate(
        &self,
        memory: &dyn Memory,
        conversation: &[ChatMessage],
        session_id: &str,
    ) -> Result<()> {
        // Take the last 10 messages (or fewer).
        let recent: Vec<&ChatMessage> = conversation.iter().rev().take(10).collect::<Vec<_>>();
        let recent: Vec<&&ChatMessage> = recent.iter().rev().collect();

        if recent.is_empty() {
            return Ok(());
        }

        // Build conversation text for the prompt. Each message's
        // content is run through `scrub_text` first so credentials
        // (API keys, tokens, JWTs, AWS/GCP keys, PEM blocks, emails)
        // that may have appeared in the raw chat don't get shipped to
        // the consolidation provider — which is often a cheap third-
        // party API the user wouldn't necessarily want to see those
        // secrets. The OUTPUT path already scrubs symmetrically (see
        // `clean_for_store` below); this is the matching INPUT-side
        // protection.
        let mut conversation_text = String::new();
        for msg in &recent {
            let role = &msg.role;
            let raw_content = msg.content.as_deref().unwrap_or("(no content)");
            let scrubbed = scrub_text(raw_content);
            conversation_text.push_str(&format!("{role}: {scrubbed}\n"));
        }

        let prompt = format!(
            "Analyze this conversation and extract structured information.\n\
             Return valid JSON with this exact format:\n\
             {{\"daily_summary\": \"Brief summary of what happened\", \"core_facts\": [{{\"key\": \"unique-key\", \"content\": \"The fact or preference\"}}]}}\n\
             Only include core_facts for genuinely new, durable information (user preferences, project decisions, important facts). \
             If nothing new was learned, return an empty core_facts array.\n\
             Conversation:\n{conversation_text}"
        );

        let messages = vec![ChatMessage::user(prompt)];
        let request = ChatRequest {
            system: None,
            messages: &messages,
            tools: None,
            max_tokens: 1024,
            temperature: 0.0,
            thinking_level: crate::agent::thinking::ThinkingLevel::Off,
        };

        let response = self
            .provider
            .chat(request)
            .await
            .context("consolidation provider call")?;

        let raw_text = response.content.unwrap_or_default();

        let timestamp = chrono::Utc::now().to_rfc3339();

        // Try to parse as JSON. On failure, store the raw text as a daily summary.
        match serde_json::from_str::<ExtractionResult>(&raw_text) {
            Ok(extraction) => {
                // Scrub + cap the model-extracted daily summary. The
                // extraction model can echo secrets it saw in the
                // conversation into the "summary" field.
                let summary_content =
                    clean_for_store(&extraction.daily_summary, MAX_DAILY_SUMMARY_BYTES);
                let summary_key = format!("daily-{session_id}-{timestamp}");
                let entry = MemoryEntry {
                    key: summary_key,
                    content: summary_content,
                    category: MemoryCategory::Daily,
                    session_id: Some(session_id.to_string()),
                    ..MemoryEntry::default()
                };
                memory.store(entry).await.context("storing daily summary")?;

                for fact in extraction.core_facts {
                    // Cap keys (no scrub — keys are slugs and tend to be
                    // short; scrubbing could mangle a legitimate slug
                    // containing e.g. "password-policy").
                    let fact_key = truncate_at_char_boundary(&fact.key, MAX_KEY_BYTES);
                    // Scrub + cap fact content.
                    let fact_content = clean_for_store(&fact.content, MAX_FACT_CONTENT_BYTES);
                    let entry = MemoryEntry {
                        key: fact_key,
                        content: fact_content,
                        category: MemoryCategory::Core,
                        session_id: Some(session_id.to_string()),
                        ..MemoryEntry::default()
                    };
                    memory.store(entry).await.context("storing core fact")?;
                }
            }
            Err(_) => {
                // Graceful fallback: the model emitted something that
                // isn't JSON. Store the raw response as a daily summary —
                // but scrub secrets and cap size first. Previously this
                // persisted whatever the model said verbatim, which
                // meant any secret echoed in the raw output landed in
                // permanent memory unredacted.
                let fallback_content = clean_for_store(&raw_text, MAX_FALLBACK_BYTES);
                let summary_key = format!("daily-{session_id}-{timestamp}");
                let entry = MemoryEntry {
                    key: summary_key,
                    content: fallback_content,
                    category: MemoryCategory::Daily,
                    session_id: Some(session_id.to_string()),
                    ..MemoryEntry::default()
                };
                memory.store(entry).await.context("storing fallback summary")?;
            }
        }

        Ok(())
    }
}

/// Pipeline every persisted consolidator string through: scrub → cap.
/// The order matters: scrub first so the redaction markers can occupy
/// space before the truncation check, rather than having the truncation
/// cut off the tail of a `[REDACTED_…]` marker mid-word.
fn clean_for_store(raw: &str, cap_bytes: usize) -> String {
    let scrubbed = scrub_text(raw);
    truncate_at_char_boundary(&scrubbed, cap_bytes)
}

/// Truncate `s` to at most `max_bytes`, stepping back to the nearest
/// UTF-8 char boundary if needed. Appends a visible `… [truncated]`
/// marker when the content was actually cut.
fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}… [truncated]", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::traits::{
        default_chat_stream, ChatMessage, ChatResponse, StreamEvent,
    };
    use std::sync::Arc;

    // -- Helpers for scrub+cap --------------------------------------

    #[test]
    fn clean_for_store_scrubs_then_caps() {
        // A secret inside the content must be redacted BEFORE the
        // truncation check, otherwise the `[REDACTED_…]` marker might
        // be cut off and a partial secret could survive.
        let raw = format!("hello AKIAIOSFODNN7EXAMPLE world");
        let cleaned = clean_for_store(&raw, 128);
        assert!(cleaned.contains("[REDACTED_AWS_KEY]"));
        assert!(!cleaned.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[test]
    fn clean_for_store_truncates_oversized() {
        let raw = "a".repeat(10_000);
        let cleaned = clean_for_store(&raw, 200);
        assert!(cleaned.len() < 10_000);
        assert!(cleaned.contains("truncated"));
    }

    #[test]
    fn truncate_at_char_boundary_respects_utf8() {
        // "日" is 3 bytes. Asking for 2 bytes must not panic; the
        // returned prefix may be empty, followed by the truncation
        // marker.
        let s = "日本語";
        let out = truncate_at_char_boundary(s, 2);
        assert!(out.contains("truncated"));
        // 3 bytes → "日…"
        let out = truncate_at_char_boundary(s, 3);
        assert!(out.starts_with("日"));
        // 100 bytes → unchanged.
        let out = truncate_at_char_boundary(s, 100);
        assert_eq!(out, "日本語");
    }

    #[test]
    fn default_caps_are_sensible() {
        assert_eq!(MAX_DAILY_SUMMARY_BYTES, 8 * 1024);
        assert_eq!(MAX_FACT_CONTENT_BYTES, 4 * 1024);
        assert_eq!(MAX_FALLBACK_BYTES, 16 * 1024);
        assert_eq!(MAX_KEY_BYTES, 256);
    }

    // -- Integration tests with minimal mocks -----------------------

    /// Minimal Memory mock: captures every `store` call into a
    /// Mutex-guarded Vec so tests can inspect the stored entries.
    struct MockMemory {
        stored: parking_lot::Mutex<Vec<MemoryEntry>>,
    }

    impl MockMemory {
        fn new() -> Self {
            Self {
                stored: parking_lot::Mutex::new(Vec::new()),
            }
        }
        fn snapshot(&self) -> Vec<MemoryEntry> {
            self.stored.lock().clone()
        }
    }

    #[async_trait::async_trait]
    impl Memory for MockMemory {
        fn name(&self) -> &str {
            "mock"
        }
        async fn store(&self, entry: MemoryEntry) -> Result<()> {
            self.stored.lock().push(entry);
            Ok(())
        }
        async fn recall(&self, _query: &str, _limit: usize) -> Result<Vec<MemoryEntry>> {
            Ok(vec![])
        }
        async fn get(&self, _key: &str) -> Result<Option<MemoryEntry>> {
            Ok(None)
        }
        async fn list(
            &self,
            _category: Option<&MemoryCategory>,
            _limit: usize,
        ) -> Result<Vec<MemoryEntry>> {
            Ok(self.stored.lock().clone())
        }
        async fn forget(&self, _key: &str) -> Result<bool> {
            Ok(false)
        }
        async fn count(&self, _category: Option<&MemoryCategory>) -> Result<usize> {
            Ok(self.stored.lock().len())
        }
        async fn health_check(&self) -> Result<()> {
            Ok(())
        }
    }

    /// Minimal Provider mock: returns a canned response for every
    /// `chat()` call. Enough to drive `consolidate`.
    struct MockProvider {
        response_text: String,
    }

    #[async_trait::async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            "mock"
        }
        async fn chat(&self, _req: ChatRequest<'_>) -> Result<ChatResponse> {
            Ok(ChatResponse {
                content: Some(self.response_text.clone()),
                tool_calls: vec![],
                usage: None,
                reasoning: None,
            })
        }
        fn supports_tool_calling(&self) -> bool {
            false
        }
        fn context_window(&self) -> usize {
            128_000
        }
        async fn chat_stream(
            &self,
            request: ChatRequest<'_>,
        ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
            default_chat_stream(self, request).await
        }
    }

    fn convo() -> Vec<ChatMessage> {
        vec![
            ChatMessage::user("please remember my aws key AKIAIOSFODNN7EXAMPLE"),
            ChatMessage::assistant("got it"),
        ]
    }

    /// Regression for T3-G: when the model returns non-JSON, the raw
    /// text used to be stored verbatim. Any secret the model echoed
    /// now gets scrubbed first.
    #[tokio::test]
    async fn fallback_summary_scrubs_secrets() {
        let provider = MockProvider {
            // Model returned something that isn't valid JSON.
            response_text: "Here's what I noticed: your AWS key is AKIAIOSFODNN7EXAMPLE"
                .to_string(),
        };
        let consolidator = MemoryConsolidator::new(Box::new(provider));
        let memory = Arc::new(MockMemory::new());
        consolidator
            .consolidate(memory.as_ref(), &convo(), "session-1")
            .await
            .unwrap();

        let stored = memory.snapshot();
        assert_eq!(stored.len(), 1, "expected one daily entry");
        let entry = &stored[0];
        assert!(
            entry.content.contains("[REDACTED_AWS_KEY]"),
            "fallback must scrub: got {:?}",
            entry.content
        );
        assert!(
            !entry.content.contains("AKIAIOSFODNN7EXAMPLE"),
            "raw secret leaked: {:?}",
            entry.content
        );
    }

    /// Regression for T3-G: oversized fallback content must be capped,
    /// not stored in full.
    #[tokio::test]
    async fn fallback_summary_caps_oversized_content() {
        let provider = MockProvider {
            response_text: "junk ".repeat(50_000), // ~250 KB
        };
        let consolidator = MemoryConsolidator::new(Box::new(provider));
        let memory = Arc::new(MockMemory::new());
        consolidator
            .consolidate(memory.as_ref(), &convo(), "session-2")
            .await
            .unwrap();

        let stored = memory.snapshot();
        assert_eq!(stored.len(), 1);
        // Whatever the cap is, the stored entry must be much shorter
        // than the 250 KB input.
        assert!(stored[0].content.len() <= MAX_FALLBACK_BYTES + 64);
        assert!(stored[0].content.contains("truncated"));
    }

    /// Valid-JSON path: the extracted daily_summary and each fact.content
    /// must also be scrubbed. The model could echo secrets from the
    /// conversation into its "extracted" output.
    #[tokio::test]
    async fn extracted_fields_are_scrubbed() {
        // Valid JSON, but with a secret embedded in the summary and a
        // fact.
        let json = r#"{
            "daily_summary": "User shared AKIAIOSFODNN7EXAMPLE with me",
            "core_facts": [
                {"key": "aws-note", "content": "Key: AKIAIOSFODNN7EXAMPLE"}
            ]
        }"#;
        let provider = MockProvider {
            response_text: json.to_string(),
        };
        let consolidator = MemoryConsolidator::new(Box::new(provider));
        let memory = Arc::new(MockMemory::new());
        consolidator
            .consolidate(memory.as_ref(), &convo(), "session-3")
            .await
            .unwrap();

        let stored = memory.snapshot();
        // One daily summary + one core fact.
        assert_eq!(stored.len(), 2);
        for entry in &stored {
            assert!(
                !entry.content.contains("AKIAIOSFODNN7EXAMPLE"),
                "secret survived in {} entry: {:?}",
                match entry.category {
                    MemoryCategory::Core => "core",
                    MemoryCategory::Daily => "daily",
                    _ => "other",
                },
                entry.content
            );
        }
    }

    /// Oversized fact content (from a misbehaving extraction) must be
    /// capped at `MAX_FACT_CONTENT_BYTES`.
    #[tokio::test]
    async fn oversized_fact_content_is_capped() {
        let big_content = "x".repeat(20_000);
        let json = format!(
            r#"{{"daily_summary":"ok","core_facts":[{{"key":"k","content":"{}"}}]}}"#,
            big_content
        );
        let provider = MockProvider {
            response_text: json,
        };
        let consolidator = MemoryConsolidator::new(Box::new(provider));
        let memory = Arc::new(MockMemory::new());
        consolidator
            .consolidate(memory.as_ref(), &convo(), "session-4")
            .await
            .unwrap();

        let stored = memory.snapshot();
        // Pick the core-fact entry.
        let fact = stored
            .iter()
            .find(|e| matches!(e.category, MemoryCategory::Core))
            .expect("missing core fact");
        assert!(fact.content.len() <= MAX_FACT_CONTENT_BYTES + 64);
        assert!(fact.content.contains("truncated"));
    }

    /// Oversized fact key gets truncated but not scrubbed.
    #[tokio::test]
    async fn oversized_fact_key_is_truncated() {
        let big_key = "k".repeat(500);
        let json = format!(
            r#"{{"daily_summary":"ok","core_facts":[{{"key":"{}","content":"c"}}]}}"#,
            big_key
        );
        let provider = MockProvider {
            response_text: json,
        };
        let consolidator = MemoryConsolidator::new(Box::new(provider));
        let memory = Arc::new(MockMemory::new());
        consolidator
            .consolidate(memory.as_ref(), &convo(), "session-5")
            .await
            .unwrap();

        let stored = memory.snapshot();
        let fact = stored
            .iter()
            .find(|e| matches!(e.category, MemoryCategory::Core))
            .unwrap();
        assert!(fact.key.len() <= MAX_KEY_BYTES + 64);
    }
}
