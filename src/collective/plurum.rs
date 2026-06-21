use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::traits::*;
use crate::memory::experience::{Attempt, Experience, ExperienceContext};

pub struct PlurumlClient {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

// ---------------------------------------------------------------------------
// Wire types for Plurum API serialization/deserialization
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct SearchRequest {
    query: String,
    limit: usize,
}

/// Wrapper for the search response envelope.
#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    results: Vec<PlurumlSearchResult>,
}

#[derive(Deserialize)]
struct PlurumlSearchResult {
    id: String,
    goal: String,
    solution: Option<String>,
    #[serde(default, deserialize_with = "deserialize_gotchas")]
    gotchas: Vec<String>,
    /// Plurum may send `trust_score` or the older `quality_score` name.
    trust_score: Option<f64>,
    quality_score: Option<f64>,
    relevance_score: Option<f64>,
    #[serde(default)]
    outcome_reports: Option<PlurumlOutcomeReports>,
}

/// Gotchas can be either plain strings or `{"warning": "...", "context": "..."}` objects.
fn deserialize_gotchas<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(s),
            serde_json::Value::Object(obj) => obj
                .get("warning")
                .and_then(|w| w.as_str())
                .map(String::from),
            _ => None,
        })
        .collect())
}

#[derive(Deserialize)]
struct PlurumlOutcomeReports {
    success: u32,
    failure: u32,
}

/// Plurum experience response — may use `attempts` or the legacy
/// `dead_ends` + `breakthroughs` fields.
#[derive(Deserialize)]
struct PlurumlExperience {
    id: String,
    goal: String,
    #[serde(default)]
    context: Option<PlurumlContext>,
    #[serde(default)]
    attempts: Option<Vec<PlurumlAttempt>>,
    #[serde(default)]
    dead_ends: Option<Vec<String>>,
    #[serde(default)]
    breakthroughs: Option<Vec<String>>,
    solution: Option<String>,
    #[serde(default, deserialize_with = "deserialize_gotchas")]
    gotchas: Vec<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "default_confidence")]
    confidence: f32,
    session_id: Option<String>,
    #[serde(default = "default_timestamp")]
    created_at: String,
}

fn default_confidence() -> f32 {
    0.5
}

fn default_timestamp() -> String {
    String::new()
}

#[derive(Deserialize)]
struct PlurumlContext {
    #[serde(default)]
    tools_used: Vec<String>,
    #[serde(default)]
    environment: String,
    #[serde(default)]
    constraints: String,
}

#[derive(Deserialize)]
struct PlurumlAttempt {
    action: String,
    outcome: String,
    #[serde(default)]
    dead_end: bool,
    #[serde(default)]
    insight: String,
}

#[derive(Serialize)]
struct PublishRequest {
    goal: String,
    domain: String,
    outcome: String,
    context_structured: PublishContext,
    attempts: Vec<PublishAttempt>,
    solution: Option<String>,
    gotchas: Vec<String>,
    tags: Vec<String>,
    confidence: f32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools_used: Vec<String>,
    status: String,
}

#[derive(Serialize)]
struct PublishContext {
    tools_used: Vec<String>,
    environment: String,
    constraints: String,
}

#[derive(Serialize)]
struct PublishAttempt {
    action: String,
    outcome: String,
    dead_end: bool,
    insight: String,
}

#[derive(Deserialize)]
struct PublishResponse {
    id: String,
}

#[derive(Serialize)]
struct OutcomeReportRequest {
    success: bool,
    execution_time_ms: Option<u64>,
    error_message: Option<String>,
    context_notes: Option<String>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    #[serde(default)]
    message: String,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl PlurumlClient {
    /// Build a Plurum API client.
    ///
    /// Returns `Err` if the underlying reqwest client cannot be built
    /// (e.g. missing TLS roots, broken proxy config, DNS-resolver init
    /// failure). The previous `.expect("failed to build reqwest client")`
    /// would panic the entire process at startup on a misconfigured
    /// rustls install, taking down all of Fennec for what is really an
    /// optional integration. Callers in `main.rs` treat the error the
    /// same way as a missing API key — collective integration is
    /// disabled, the rest of the agent runs fine.
    ///
    /// The client also enforces an HTTPS-only policy and a 5-second
    /// connect timeout in addition to the existing 10-second total
    /// request timeout. `https_only(true)` blocks accidental HTTP
    /// downgrade if the configured `base_url` is malformed; the
    /// connect timeout caps the worst-case wait when the Plurum
    /// endpoint is unreachable so a slow startup probe doesn't hang
    /// the agent for the full 10 seconds.
    pub fn new(api_key: String, base_url: Option<String>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .https_only(true)
            .build()
            .context("building Plurum reqwest client")?;

        Ok(Self {
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.plurum.ai".to_string()),
            client,
        })
    }

    /// Build a request with standard headers.
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        self.client
            .request(method, &url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
    }

    /// Parse an error response body into a contextual anyhow error.
    ///
    /// 429 (Too Many Requests) and 503 (Service Unavailable) are
    /// classified explicitly so callers / log aggregators can tell a
    /// rate-limit / availability issue apart from a real API error
    /// (4xx schema mismatch, 5xx bug). When the upstream sends a
    /// `Retry-After` header it's surfaced in the error message so the
    /// number lands in logs even without dedicated retry plumbing.
    async fn parse_error(resp: reqwest::Response) -> anyhow::Error {
        let status = resp.status();
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let body = resp.text().await.unwrap_or_default();

        // Surface the raw status + body at warn level so an intermittent
        // failure (e.g. a 422 the server rejects with an empty body) is
        // diagnosable from the default log level instead of only via the
        // returned error, which callers may downgrade or swallow. The
        // request side (query/limit) is logged at debug just before the
        // call, so enabling debug for one run correlates the two.
        if !status.is_success() {
            tracing::warn!(
                status = %status,
                body = %if body.is_empty() { "<empty>" } else { body.as_str() },
                "Plurum API error response"
            );
        }

        let message = serde_json::from_str::<ErrorResponse>(&body)
            .map(|e| e.message)
            .unwrap_or(body);

        let label = match status.as_u16() {
            422 => "validation rejected",
            429 => "rate limited",
            503 => "service unavailable",
            _ => "API error",
        };
        match retry_after {
            Some(ra) => anyhow::anyhow!(
                "Plurum {} ({}, retry-after: {}): {}",
                label,
                status,
                ra,
                message
            ),
            None => anyhow::anyhow!("Plurum {} ({}): {}", label, status, message),
        }
    }
}

#[async_trait]
impl CollectiveLayer for PlurumlClient {
    fn name(&self) -> &str {
        "plurum"
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<CollectiveSearchResult>> {
        let body = SearchRequest {
            query: query.to_string(),
            limit,
        };

        tracing::debug!(query = %query, limit = limit, "Plurum search request");

        let resp = self
            .request(reqwest::Method::POST, "/api/v1/experiences/search")
            .json(&body)
            .send()
            .await
            .context("Plurum search request failed")?;

        if !resp.status().is_success() {
            return Err(Self::parse_error(resp).await);
        }

        let envelope: SearchResponse = resp
            .json()
            .await
            .context("failed to parse Plurum search response")?;
        let results = envelope.results;

        Ok(results
            .into_iter()
            .map(|r| {
                let trust_score = r.trust_score.or(r.quality_score).unwrap_or(0.0);
                let reports = r.outcome_reports.unwrap_or(PlurumlOutcomeReports {
                    success: 0,
                    failure: 0,
                });
                CollectiveSearchResult {
                    id: r.id,
                    goal: r.goal,
                    solution: r.solution,
                    gotchas: r.gotchas,
                    trust_score,
                    relevance_score: r.relevance_score.unwrap_or(0.0),
                    outcome_reports: OutcomeReports {
                        success: reports.success,
                        failure: reports.failure,
                    },
                }
            })
            .collect())
    }

    async fn get_experience(&self, id: &str) -> anyhow::Result<Option<Experience>> {
        let resp = self
            .request(reqwest::Method::GET, &format!("/api/v1/experiences/{}", id))
            .send()
            .await
            .context("Plurum get_experience request failed")?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(Self::parse_error(resp).await);
        }

        let pe: PlurumlExperience = resp
            .json()
            .await
            .context("failed to parse Plurum experience response")?;

        let context = pe.context.map_or_else(
            || ExperienceContext {
                tools_used: vec![],
                environment: String::new(),
                constraints: String::new(),
            },
            |c| ExperienceContext {
                tools_used: c.tools_used,
                environment: c.environment,
                constraints: c.constraints,
            },
        );

        // Use `attempts` if present; otherwise synthesise from legacy fields.
        let attempts = if let Some(atts) = pe.attempts {
            atts.into_iter()
                .map(|a| Attempt {
                    action: a.action,
                    outcome: a.outcome,
                    dead_end: a.dead_end,
                    insight: a.insight,
                })
                .collect()
        } else {
            let mut synth = Vec::new();
            if let Some(dead_ends) = pe.dead_ends {
                for de in dead_ends {
                    synth.push(Attempt {
                        action: de.clone(),
                        outcome: "dead end".to_string(),
                        dead_end: true,
                        insight: de,
                    });
                }
            }
            if let Some(breakthroughs) = pe.breakthroughs {
                for bt in breakthroughs {
                    synth.push(Attempt {
                        action: bt.clone(),
                        outcome: "breakthrough".to_string(),
                        dead_end: false,
                        insight: bt,
                    });
                }
            }
            synth
        };

        Ok(Some(Experience {
            id: pe.id,
            goal: pe.goal,
            context,
            attempts,
            solution: pe.solution,
            gotchas: pe.gotchas,
            tags: pe.tags,
            confidence: pe.confidence,
            session_id: pe.session_id,
            created_at: pe.created_at,
        }))
    }

    async fn publish(&self, experience: &Experience) -> anyhow::Result<String> {
        // Infer domain from tags or default to "general".
        let domain = experience.tags.first().cloned().unwrap_or_else(|| "general".to_string());
        // Infer outcome from whether a solution exists.
        let outcome = if experience.solution.is_some() { "success" } else { "partial" }.to_string();

        let body = PublishRequest {
            goal: experience.goal.clone(),
            domain,
            outcome,
            context_structured: PublishContext {
                tools_used: experience.context.tools_used.clone(),
                environment: experience.context.environment.clone(),
                constraints: experience.context.constraints.clone(),
            },
            attempts: experience
                .attempts
                .iter()
                .map(|a| PublishAttempt {
                    action: a.action.clone(),
                    outcome: a.outcome.clone(),
                    dead_end: a.dead_end,
                    insight: a.insight.clone(),
                })
                .collect(),
            solution: experience.solution.clone(),
            gotchas: experience.gotchas.clone(),
            tags: experience.tags.clone(),
            confidence: experience.confidence,
            tools_used: experience.context.tools_used.clone(),
            status: "published".to_string(),
        };

        let resp = self
            .request(reqwest::Method::POST, "/api/v1/experiences")
            .json(&body)
            .send()
            .await
            .context("Plurum publish request failed")?;

        if !resp.status().is_success() {
            return Err(Self::parse_error(resp).await);
        }

        let created: PublishResponse = resp
            .json()
            .await
            .context("failed to parse Plurum publish response")?;

        Ok(created.id)
    }

    async fn report_outcome(
        &self,
        experience_id: &str,
        report: &OutcomeReport,
    ) -> anyhow::Result<()> {
        let body = OutcomeReportRequest {
            success: report.success,
            execution_time_ms: report.execution_time_ms,
            error_message: report.error_message.clone(),
            context_notes: report.context_notes.clone(),
        };

        let resp = self
            .request(
                reqwest::Method::POST,
                &format!("/api/v1/experiences/{}/outcome", experience_id),
            )
            .json(&body)
            .send()
            .await
            .context("Plurum report_outcome request failed")?;

        if !resp.status().is_success() {
            return Err(Self::parse_error(resp).await);
        }

        Ok(())
    }

    async fn health_check(&self) -> bool {
        let resp = self
            .request(reqwest::Method::GET, "/health")
            .send()
            .await;

        matches!(resp, Ok(r) if r.status().is_success())
    }
}
