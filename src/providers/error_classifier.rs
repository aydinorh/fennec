//! Classify provider errors so failover/retry decisions match the failure.
//!
//! [`ReliableProvider`](super::reliable::ReliableProvider) previously treated
//! every non-429 error the same way — retry with backoff, then move on — which
//! wastes the retry/deadline budget on errors that retrying can't fix (a bad
//! API key, a malformed request, a context-overflow). This maps an error
//! message onto a coarse [`ErrorClass`] so callers can react appropriately:
//! back off and rotate on rate-limit/overload, move on (no retry) for
//! auth/billing/model-not-found, give up immediately for context-overflow /
//! malformed-request (where failover won't help), and retry transient
//! server/network errors.
//!
//! Classification is substring/status-code matching on the error text — the
//! same pragmatic approach the upstream uses — since provider errors surface as
//! strings like `"OpenAI API error (429): ..."`.

/// Coarse classification of a provider error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// 429 / throttling — back off and rotate to another provider.
    RateLimit,
    /// 503 / 529 / "overloaded" — provider is busy; back off and rotate.
    Overloaded,
    /// 401 / 403 — credentials rejected; this provider won't work, try another.
    Auth,
    /// 402 / quota / billing / insufficient balance — try another provider.
    InsufficientCredits,
    /// Prompt exceeds the model's context window — retrying/failing over won't
    /// help; surface immediately (the agent should compress instead).
    ContextOverflow,
    /// 404 / invalid model — try a provider that has the model.
    ModelNotFound,
    /// 400 / malformed request — non-retryable; surface immediately.
    FormatError,
    /// 5xx (500/502/504) — transient server error; retry then rotate.
    ServerError,
    /// Timeout / connection / DNS / TLS — transient; retry then rotate.
    Network,
    /// Unrecognized — treat as transient (retry then rotate).
    Unknown,
}

impl ErrorClass {
    /// Whether retrying the *same* provider may succeed.
    pub fn retryable(self) -> bool {
        matches!(self, Self::ServerError | Self::Network | Self::Unknown)
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Classify a provider error from its message text.
pub fn classify(err: &anyhow::Error) -> ErrorClass {
    let msg = err.to_string().to_lowercase();

    // Context overflow first — Anthropic returns it as a 400, so it must be
    // matched before the generic 400/format rule.
    if contains_any(
        &msg,
        &[
            "context length",
            "context window",
            "maximum context",
            "context_length_exceeded",
            "prompt is too long",
            "too many tokens",
            "reduce the length",
            "input is too long",
        ],
    ) {
        return ErrorClass::ContextOverflow;
    }

    // Rate limit before billing — a 429 mentioning "quota" is throttling.
    if contains_any(&msg, &["(429)", "429", "rate limit", "rate_limit", "too many requests"]) {
        return ErrorClass::RateLimit;
    }

    if contains_any(
        &msg,
        &[
            "insufficient",
            "(402)",
            "billing",
            "payment required",
            "exceeded your current quota",
            "insufficient_quota",
            "credit balance",
            "out of credits",
        ],
    ) {
        return ErrorClass::InsufficientCredits;
    }

    if contains_any(
        &msg,
        &["(503)", "(529)", "529", "overloaded", "service unavailable", "at capacity"],
    ) {
        return ErrorClass::Overloaded;
    }

    if contains_any(
        &msg,
        &[
            "(401)",
            "(403)",
            "unauthorized",
            "forbidden",
            "authentication",
            "invalid api key",
            "invalid_api_key",
            "permission denied",
            "could not authenticate",
        ],
    ) {
        return ErrorClass::Auth;
    }

    // ModelNotFound requires model-specific phrasing. A bare 404 is NOT
    // enough: local / OpenAI-compatible endpoints (Ollama, LM Studio,
    // vLLM behind a proxy) return 404 for a wrong base_url path or a
    // briefly-restarting server, and the upstream deliberately treats
    // generic 404 as retryable-unknown for exactly that reason —
    // classifying it as ModelNotFound permanently skips the provider in
    // failover.
    if contains_any(
        &msg,
        &[
            "model not found",
            "invalid model",
            "no such model",
            "model_not_found",
            "unknown model",
        ],
    ) || (msg.contains("(404)") && msg.contains("model"))
        || (msg.contains("does not exist") && msg.contains("model"))
    {
        return ErrorClass::ModelNotFound;
    }

    if contains_any(
        &msg,
        &["(400)", "invalid_request", "invalid request", "bad request", "malformed", "unprocessable"],
    ) {
        return ErrorClass::FormatError;
    }

    if contains_any(
        &msg,
        &["(500)", "(502)", "(504)", "internal server error", "bad gateway", "gateway timeout"],
    ) {
        return ErrorClass::ServerError;
    }

    if contains_any(
        &msg,
        &[
            "timeout",
            "timed out",
            "connection",
            "dns",
            "broken pipe",
            "connection reset",
            "unreachable",
            "tls",
            "error sending request",
        ],
    ) {
        return ErrorClass::Network;
    }

    ErrorClass::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(s: &str) -> anyhow::Error {
        anyhow::anyhow!("{}", s)
    }

    #[test]
    fn classifies_rate_limit() {
        assert_eq!(classify(&err("OpenAI API error (429): rate limit reached")), ErrorClass::RateLimit);
        assert_eq!(classify(&err("Too Many Requests")), ErrorClass::RateLimit);
    }

    #[test]
    fn context_overflow_beats_400() {
        // Anthropic returns context overflow as a 400 — must classify as
        // overflow, not a generic format error.
        let e = err("Anthropic API error (400): prompt is too long: 250000 tokens > 200000 maximum");
        assert_eq!(classify(&e), ErrorClass::ContextOverflow);
        assert!(!classify(&e).retryable());
    }

    #[test]
    fn classifies_auth_billing_model_format() {
        assert_eq!(classify(&err("OpenAI API error (401): invalid api key")), ErrorClass::Auth);
        assert_eq!(classify(&err("error (402): insufficient_quota")), ErrorClass::InsufficientCredits);
        assert_eq!(classify(&err("API error (404): model not found")), ErrorClass::ModelNotFound);
        assert_eq!(classify(&err("API error (400): invalid_request_error")), ErrorClass::FormatError);
    }

    /// A bare 404 with no model-specific phrasing must NOT classify as
    /// ModelNotFound: local / OpenAI-compatible endpoints return 404
    /// for wrong base_url paths and mid-restart windows, and a
    /// ModelNotFound verdict permanently skips the provider in
    /// failover. It stays retryable Unknown instead.
    #[test]
    fn bare_404_is_not_model_not_found() {
        let e = err("API error (404): Not Found");
        assert_eq!(classify(&e), ErrorClass::Unknown);
        assert!(classify(&e).retryable());
        // 404 + model phrasing still classifies.
        assert_eq!(
            classify(&err("API error (404): The model `x` does not exist")),
            ErrorClass::ModelNotFound
        );
    }

    #[test]
    fn classifies_overloaded_and_server_and_network() {
        assert_eq!(classify(&err("Anthropic API error (529): overloaded")), ErrorClass::Overloaded);
        assert_eq!(classify(&err("API error (500): internal server error")), ErrorClass::ServerError);
        assert_eq!(classify(&err("error sending request for url: connection reset")), ErrorClass::Network);
    }

    #[test]
    fn retryable_only_for_transient() {
        assert!(ErrorClass::ServerError.retryable());
        assert!(ErrorClass::Network.retryable());
        assert!(ErrorClass::Unknown.retryable());
        assert!(!ErrorClass::RateLimit.retryable());
        assert!(!ErrorClass::Auth.retryable());
        assert!(!ErrorClass::ContextOverflow.retryable());
        assert!(!ErrorClass::FormatError.retryable());
    }

    #[test]
    fn unknown_falls_through() {
        assert_eq!(classify(&err("something totally weird happened")), ErrorClass::Unknown);
    }
}
