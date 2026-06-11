use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use parking_lot::Mutex;
use rand::Rng;

use super::traits::{ChatRequest, ChatResponse, Provider, StreamEvent};

/// Default cooldown duration for rate-limited providers (60 seconds).
const RATE_LIMIT_COOLDOWN: Duration = Duration::from_secs(60);

/// Default overall deadline for a single `chat` call across all provider
/// retries. The audit flagged that a 4-provider outage with
/// `max_retries=3` could previously spend ~28s blocked in backoff
/// alone, plus per-provider request latencies on top — easy to blow
/// past a minute total. Callers that want longer or shorter deadlines
/// can set their own via [`ReliableProvider::with_max_total_duration`].
const DEFAULT_MAX_TOTAL_DURATION: Duration = Duration::from_secs(60);

/// A provider wrapper that adds automatic failover and retry logic.
///
/// `ReliableProvider` holds a prioritized list of providers and tries each
/// one in order. On rate limits it applies a cooldown; on other errors it
/// retries with exponential backoff before moving to the next provider.
/// An overall deadline caps the whole operation so a multi-provider
/// outage can't block a caller for minutes.
pub struct ReliableProvider {
    providers: Vec<Box<dyn Provider>>,
    max_retries: usize,
    cooldowns: Arc<Mutex<HashMap<usize, Instant>>>,
    max_total_duration: Duration,
}

impl ReliableProvider {
    /// Create a new reliable provider wrapper.
    ///
    /// - `providers`: Prioritized list of providers to try in order.
    /// - `max_retries`: Maximum retry attempts per provider (default 3).
    pub fn new(providers: Vec<Box<dyn Provider>>, max_retries: Option<usize>) -> Self {
        Self {
            providers,
            max_retries: max_retries.unwrap_or(3),
            cooldowns: Arc::new(Mutex::new(HashMap::new())),
            max_total_duration: DEFAULT_MAX_TOTAL_DURATION,
        }
    }

    /// Override the overall `chat` deadline from its default of 60 seconds.
    pub fn with_max_total_duration(mut self, d: Duration) -> Self {
        self.max_total_duration = d;
        self
    }

    /// Check whether a provider index is currently on cooldown.
    fn is_on_cooldown(&self, index: usize) -> bool {
        let cooldowns = self.cooldowns.lock();
        if let Some(until) = cooldowns.get(&index) {
            Instant::now() < *until
        } else {
            false
        }
    }

    /// Mark a provider as rate-limited, putting it on cooldown.
    fn mark_rate_limited(&self, index: usize) {
        let mut cooldowns = self.cooldowns.lock();
        cooldowns.insert(index, Instant::now() + RATE_LIMIT_COOLDOWN);
    }

    /// Check if an error looks like a rate limit (429).
    fn is_rate_limit_error(err: &anyhow::Error) -> bool {
        let msg = err.to_string().to_lowercase();
        msg.contains("429") || msg.contains("rate limit")
    }

    /// Apply ±20% jitter to `base` to avoid thundering-herd retries when
    /// multiple callers hit the same transient failure simultaneously.
    fn jitter(base: Duration) -> Duration {
        let factor: f64 = rand::rng().random_range(0.8..=1.2);
        let millis = (base.as_millis() as f64 * factor) as u64;
        Duration::from_millis(millis)
    }
}

#[async_trait]
impl Provider for ReliableProvider {
    fn name(&self) -> &str {
        "reliable"
    }

    fn model(&self) -> &str {
        // /usage pricing lookup + /model panel header read this; an
        // empty default would break both the moment failover wraps
        // the primary.
        self.providers.first().map(|p| p.model()).unwrap_or("")
    }

    async fn chat(&self, request: ChatRequest<'_>) -> Result<ChatResponse> {
        let start = Instant::now();
        let mut last_error: Option<anyhow::Error> = None;

        for (index, provider) in self.providers.iter().enumerate() {
            // Overall deadline check at the provider boundary — if we're
            // already out of budget, don't even try the next provider.
            if start.elapsed() >= self.max_total_duration {
                tracing::warn!(
                    "ReliableProvider: overall deadline {:?} exceeded before trying {} more provider(s)",
                    self.max_total_duration,
                    self.providers.len().saturating_sub(index)
                );
                break;
            }

            // Skip providers on cooldown.
            if self.is_on_cooldown(index) {
                tracing::debug!(
                    "Skipping provider {} (index {}): on cooldown",
                    provider.name(),
                    index
                );
                continue;
            }

            // Retry loop for this provider.
            for attempt in 0..self.max_retries {
                // Re-check the deadline at each attempt — a slow-to-error
                // provider shouldn't keep consuming budget.
                if start.elapsed() >= self.max_total_duration {
                    break;
                }

                // Build a new request referencing the same data.
                let req = ChatRequest {
                    system: request.system,
                    messages: request.messages,
                    tools: request.tools,
                    max_tokens: request.max_tokens,
                    temperature: request.temperature,
                    thinking_level: request.thinking_level,
                };

                match provider.chat(req).await {
                    Ok(response) => return Ok(response),
                    Err(err) => {
                        if Self::is_rate_limit_error(&err) {
                            tracing::warn!(
                                "Provider {} rate limited, applying cooldown",
                                provider.name()
                            );
                            self.mark_rate_limited(index);
                            last_error = Some(err);
                            break; // Move to next provider.
                        }

                        tracing::warn!(
                            "Provider {} attempt {}/{} failed: {}",
                            provider.name(),
                            attempt + 1,
                            self.max_retries,
                            err
                        );
                        last_error = Some(err);

                        // Exponential backoff with jitter. Cap the sleep
                        // at the remaining deadline budget so we don't
                        // "sleep past" the caller's deadline.
                        if attempt + 1 < self.max_retries {
                            let base = Duration::from_secs(1 << attempt);
                            let jittered = Self::jitter(base);
                            let remaining =
                                self.max_total_duration.saturating_sub(start.elapsed());
                            let sleep_for = jittered.min(remaining);
                            if sleep_for.is_zero() {
                                break;
                            }
                            tokio::time::sleep(sleep_for).await;
                        }
                    }
                }
            }
        }

        // All providers exhausted or deadline exceeded.
        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "ReliableProvider: retry deadline of {:?} exceeded with no providers available",
                self.max_total_duration
            )
        }))
    }

    fn supports_tool_calling(&self) -> bool {
        self.providers
            .iter()
            .any(|p| p.supports_tool_calling())
    }

    fn context_window(&self) -> usize {
        self.providers
            .first()
            .map(|p| p.context_window())
            .unwrap_or(0)
    }

    fn supports_streaming(&self) -> bool {
        self.providers
            .iter()
            .any(|p| p.supports_streaming())
    }

    async fn chat_stream(
        &self,
        request: ChatRequest<'_>,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        // Mirror the retry+backoff+deadline structure from `chat()`. The
        // original `chat_stream` failover gave each provider exactly one
        // shot before moving on, with no backoff between providers and
        // no per-provider retry — so a single transient hiccup
        // (network blip, 502 from Cloudflare) on the first provider
        // bypassed retries entirely. Streaming and non-streaming
        // callers should get the same reliability profile.
        let start = Instant::now();
        let mut last_error: Option<anyhow::Error> = None;

        for (index, provider) in self.providers.iter().enumerate() {
            if start.elapsed() >= self.max_total_duration {
                tracing::warn!(
                    "ReliableProvider::chat_stream: deadline {:?} exceeded before trying {} more provider(s)",
                    self.max_total_duration,
                    self.providers.len().saturating_sub(index)
                );
                break;
            }

            if self.is_on_cooldown(index) {
                tracing::debug!(
                    "Skipping streaming provider {} (index {}): on cooldown",
                    provider.name(),
                    index
                );
                continue;
            }

            for attempt in 0..self.max_retries {
                if start.elapsed() >= self.max_total_duration {
                    break;
                }

                let req = ChatRequest {
                    system: request.system,
                    messages: request.messages,
                    tools: request.tools,
                    max_tokens: request.max_tokens,
                    temperature: request.temperature,
                    thinking_level: request.thinking_level,
                };

                match provider.chat_stream(req).await {
                    Ok(rx) => return Ok(rx),
                    Err(err) => {
                        if Self::is_rate_limit_error(&err) {
                            tracing::warn!(
                                "Streaming provider {} rate limited, applying cooldown",
                                provider.name()
                            );
                            self.mark_rate_limited(index);
                            last_error = Some(err);
                            break; // Move to next provider.
                        }

                        tracing::warn!(
                            "Streaming provider {} attempt {}/{} failed: {}",
                            provider.name(),
                            attempt + 1,
                            self.max_retries,
                            err
                        );
                        last_error = Some(err);

                        if attempt + 1 < self.max_retries {
                            let base = Duration::from_secs(1 << attempt);
                            let jittered = Self::jitter(base);
                            let remaining =
                                self.max_total_duration.saturating_sub(start.elapsed());
                            let sleep_for = jittered.min(remaining);
                            if sleep_for.is_zero() {
                                break;
                            }
                            tokio::time::sleep(sleep_for).await;
                        }
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "ReliableProvider::chat_stream: deadline of {:?} exceeded with no providers available",
                self.max_total_duration
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_produces_value_in_expected_range() {
        let base = Duration::from_millis(1000);
        for _ in 0..50 {
            let j = ReliableProvider::jitter(base);
            let ms = j.as_millis();
            assert!(
                (800..=1200).contains(&ms),
                "jitter out of range: {}ms for base 1000ms",
                ms
            );
        }
    }

    #[test]
    fn with_max_total_duration_overrides_default() {
        let p = ReliableProvider::new(vec![], Some(3))
            .with_max_total_duration(Duration::from_secs(10));
        assert_eq!(p.max_total_duration, Duration::from_secs(10));
    }

    #[test]
    fn default_max_total_duration_is_sixty_seconds() {
        let p = ReliableProvider::new(vec![], Some(3));
        assert_eq!(p.max_total_duration, DEFAULT_MAX_TOTAL_DURATION);
        assert_eq!(p.max_total_duration, Duration::from_secs(60));
    }

    #[test]
    fn is_rate_limit_error_matches_common_phrasings() {
        assert!(ReliableProvider::is_rate_limit_error(&anyhow::anyhow!(
            "HTTP 429 Too Many Requests"
        )));
        assert!(ReliableProvider::is_rate_limit_error(&anyhow::anyhow!(
            "rate limit exceeded on tier 1"
        )));
        assert!(ReliableProvider::is_rate_limit_error(&anyhow::anyhow!(
            "RATE LIMIT"
        )));
        assert!(!ReliableProvider::is_rate_limit_error(&anyhow::anyhow!(
            "service unavailable"
        )));
    }
}
