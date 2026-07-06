use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use parking_lot::Mutex;
use rand::Rng;

use super::error_classifier::{classify, ErrorClass};
use super::traits::{ChatRequest, ChatResponse, Provider, StreamEvent};

/// What the failover loop should do with a failed attempt.
#[derive(Debug, PartialEq, Eq)]
enum FailoverAction {
    /// Provider is throttled/busy — cool it down and move to the next.
    Cooldown,
    /// This provider can't serve the request (auth/billing/model) — move to
    /// the next without retrying.
    NextProvider,
    /// Transient — retry this provider (with backoff) before moving on.
    Retry,
    /// Retrying or failing over won't help (context-overflow / malformed) —
    /// surface the error immediately.
    Abort,
}

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

    /// Decide what the failover loop should do with a failed attempt, based on
    /// the error's classification.
    fn failover_action(err: &anyhow::Error) -> FailoverAction {
        match classify(err) {
            ErrorClass::RateLimit | ErrorClass::Overloaded => FailoverAction::Cooldown,
            ErrorClass::Auth
            | ErrorClass::InsufficientCredits
            | ErrorClass::ModelNotFound => FailoverAction::NextProvider,
            ErrorClass::ContextOverflow | ErrorClass::FormatError => FailoverAction::Abort,
            ErrorClass::ServerError | ErrorClass::Network | ErrorClass::Unknown => {
                FailoverAction::Retry
            }
        }
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
                    Err(err) => match Self::failover_action(&err) {
                        FailoverAction::Cooldown => {
                            tracing::warn!(
                                "Provider {} throttled/overloaded, applying cooldown",
                                provider.name()
                            );
                            self.mark_rate_limited(index);
                            last_error = Some(err);
                            break; // Move to next provider.
                        }
                        FailoverAction::NextProvider => {
                            tracing::warn!(
                                "Provider {} can't serve request ({}); trying next",
                                provider.name(),
                                err
                            );
                            last_error = Some(err);
                            break; // Move to next provider, no retry.
                        }
                        FailoverAction::Abort => {
                            // Context-overflow / malformed request — retrying
                            // or failing over won't help; surface immediately.
                            return Err(err);
                        }
                        FailoverAction::Retry => {
                            tracing::warn!(
                                "Provider {} attempt {}/{} failed: {}",
                                provider.name(),
                                attempt + 1,
                                self.max_retries,
                                err
                            );
                            last_error = Some(err);

                            // Exponential backoff with jitter, capped at the
                            // remaining deadline budget.
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
                    },
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
                    Err(err) => match Self::failover_action(&err) {
                        FailoverAction::Cooldown => {
                            tracing::warn!(
                                "Streaming provider {} throttled/overloaded, applying cooldown",
                                provider.name()
                            );
                            self.mark_rate_limited(index);
                            last_error = Some(err);
                            break; // Move to next provider.
                        }
                        FailoverAction::NextProvider => {
                            tracing::warn!(
                                "Streaming provider {} can't serve request ({}); trying next",
                                provider.name(),
                                err
                            );
                            last_error = Some(err);
                            break;
                        }
                        FailoverAction::Abort => {
                            return Err(err);
                        }
                        FailoverAction::Retry => {
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
                    },
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
    fn failover_action_maps_errors_correctly() {
        let act = |s: &str| ReliableProvider::failover_action(&anyhow::anyhow!("{}", s));
        // Throttled / overloaded → cooldown + rotate.
        assert_eq!(act("HTTP 429 Too Many Requests"), FailoverAction::Cooldown);
        assert_eq!(act("Anthropic API error (529): overloaded"), FailoverAction::Cooldown);
        // Can't-serve-here → next provider, no retry.
        assert_eq!(act("OpenAI API error (401): invalid api key"), FailoverAction::NextProvider);
        // Pointless to retry or fail over.
        assert_eq!(
            act("API error (400): prompt is too long: too many tokens"),
            FailoverAction::Abort
        );
        // Transient → retry.
        assert_eq!(act("API error (500): internal server error"), FailoverAction::Retry);
        assert_eq!(act("error sending request: connection reset"), FailoverAction::Retry);
    }
}
