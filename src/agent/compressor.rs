use anyhow::Result;

use crate::providers::traits::{ChatMessage, ChatRequest, Provider};

/// Compresses the conversation context when it grows too large relative to the
/// provider's context window.
#[derive(Debug, Clone)]
pub struct ContextCompressor {
    /// Fraction of the context window at which compression triggers.
    threshold_percent: f64,
    /// Number of messages at the start of history to leave untouched.
    protect_first: usize,
    /// Number of messages at the end of history to leave untouched.
    protect_last: usize,
}

impl ContextCompressor {
    /// Create a new compressor with the given thresholds.
    pub fn new(threshold_percent: f64, protect_first: usize, protect_last: usize) -> Self {
        Self {
            threshold_percent,
            protect_first,
            protect_last,
        }
    }

    /// Estimate whether the messages exceed the threshold for the given context
    /// window. Token count is approximated as total chars / 4.
    pub fn should_compress(&self, messages: &[ChatMessage], context_window: usize) -> bool {
        let total_chars: usize = messages
            .iter()
            .map(|m| m.content.as_deref().unwrap_or("").len())
            .sum();
        let estimated_tokens = total_chars / 4;
        let threshold = (context_window as f64 * self.threshold_percent) as usize;
        estimated_tokens > threshold
    }

    /// Attempt to compress the conversation history in-place.
    ///
    /// Returns `Ok(true)` if any compression was applied, `Ok(false)` if the
    /// messages are still under threshold and no action was needed.
    pub async fn compress(
        &self,
        messages: &mut Vec<ChatMessage>,
        provider: &dyn Provider,
        context_window: usize,
    ) -> Result<bool> {
        if !self.should_compress(messages, context_window) {
            return Ok(false);
        }

        let mut compressed = false;

        // Phase 1 — prune old tool results.
        let end = messages.len().saturating_sub(self.protect_last);
        for i in self.protect_first..end {
            if messages[i].role == "tool"
                && messages[i]
                    .content
                    .as_ref()
                    .map_or(false, |c| c.len() > 200)
            {
                messages[i].content =
                    Some("[Old tool output cleared to save context]".to_string());
                compressed = true;
            }
        }

        // Phase 2 — LLM summarization if still over threshold.
        if self.should_compress(messages, context_window) {
            let len = messages.len();
            if len > self.protect_first + self.protect_last {
                let middle_start = self.protect_first;
                let middle_end = len - self.protect_last;

                // Build text from middle block.
                let middle_text: String = messages[middle_start..middle_end]
                    .iter()
                    .map(|m| {
                        format!(
                            "{}: {}",
                            m.role,
                            m.content.as_deref().unwrap_or("")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let summary_prompt = format!(
                    "Summarize this conversation segment concisely, preserving key decisions, file paths, and progress:\n\n{middle_text}"
                );

                let request = ChatRequest {
                    system: None,
                    messages: &[ChatMessage::user(&summary_prompt)],
                    tools: None,
                    max_tokens: 1024,
                    temperature: 0.3,
                    thinking_level: crate::agent::thinking::ThinkingLevel::Off,
                };

                if let Ok(response) = provider.chat(request).await {
                    let summary = response.content.unwrap_or_default();
                    let summary_msg = ChatMessage::assistant(format!(
                        "[Context compressed]\n{summary}"
                    ));

                    // Replace the middle block with a single summary message.
                    messages.splice(middle_start..middle_end, std::iter::once(summary_msg));
                    compressed = true;
                }
            }
        }

        // Phase 3 — sanitize orphaned tool pairs.
        self.sanitize_orphans(messages);

        Ok(compressed)
    }

    /// The configured `protect_first` value.
    pub fn protect_first_val(&self) -> usize {
        self.protect_first
    }

    /// The configured `protect_last` value.
    pub fn protect_last_val(&self) -> usize {
        self.protect_last
    }

    /// Remove tool_result messages whose `tool_call_id` does not match any
    /// surviving assistant `tool_calls` entry. For surviving tool_calls entries
    /// with missing results, add a stub result.
    pub fn sanitize_orphans(&self, messages: &mut Vec<ChatMessage>) {
        // Collect all tool_call IDs from surviving assistant messages.
        let mut live_call_ids = std::collections::HashSet::new();
        for msg in messages.iter() {
            if msg.role == "assistant" {
                if let Some(ref calls) = msg.tool_calls {
                    for tc in calls {
                        live_call_ids.insert(tc.id.clone());
                    }
                }
            }
        }

        // Collect all tool_result IDs.
        let mut result_ids = std::collections::HashSet::new();
        for msg in messages.iter() {
            if msg.role == "tool" {
                if let Some(ref id) = msg.tool_call_id {
                    result_ids.insert(id.clone());
                }
            }
        }

        // Remove orphaned tool results (those whose call_id is not in live_call_ids).
        messages.retain(|msg| {
            if msg.role == "tool" {
                if let Some(ref id) = msg.tool_call_id {
                    return live_call_ids.contains(id);
                }
            }
            true
        });

        // Add stub results for tool_calls that lost their results.
        let missing: Vec<String> = live_call_ids
            .iter()
            .filter(|id| !result_ids.contains(*id))
            .cloned()
            .collect();

        for id in missing {
            messages.push(ChatMessage::tool_result(
                id,
                "[Tool result lost during context compression]",
            ));
        }
    }
}

impl Default for ContextCompressor {
    fn default() -> Self {
        Self::new(0.50, 3, 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let c = ContextCompressor::default();
        assert!((c.threshold_percent - 0.50).abs() < f64::EPSILON);
        assert_eq!(c.protect_first, 3);
        assert_eq!(c.protect_last, 4);
    }
}
