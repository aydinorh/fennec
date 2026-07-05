use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail};

use crate::collective::search::{CollectiveSearch, RankedExperience, SearchConfidence};
use crate::memory::decay::apply_time_decay;
use crate::memory::traits::Memory;
use crate::providers::traits::{ChatMessage, ChatRequest, ChatResponse, Provider, StreamEvent};
use crate::security::prompt_guard::{PromptGuard, ScanResult};
use crate::tools::traits::{Tool, ToolSpec};

use super::context::SystemPromptBuilder;
use super::scrub;
use super::thinking::{self, ThinkingLevel};

/// Cumulative session token usage + estimated cost. Returned by
/// [`Agent::token_usage`] for the `/usage` slash command. All
/// counters are u64; `cost_usd` is `None` when the active model
/// isn't in the pricing snapshot.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub api_calls: u64,
    /// Input + cache_read + cache_write from the most recent
    /// provider response. Used to compute `last / context_max`
    /// for the context-window utilisation row.
    pub last_prompt_tokens: u64,
    /// Configured context window of the active model.
    pub context_max: usize,
    pub cost_usd: Option<f64>,
}

impl TokenUsage {
    /// Total tokens across all classes (input + cache_read +
    /// cache_write + output).
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.cache_read_tokens + self.cache_write_tokens + self.output_tokens
    }

    /// Context utilisation as a 0..=100 percentage based on the
    /// most recent prompt size. Returns `None` when no provider
    /// call has happened yet or the model has no context window.
    pub fn context_percent(&self) -> Option<u8> {
        if self.context_max == 0 || self.last_prompt_tokens == 0 {
            return None;
        }
        let pct = (self.last_prompt_tokens as f64 * 100.0) / (self.context_max as f64);
        Some(pct.clamp(0.0, 100.0) as u8)
    }
}

/// The core agent that orchestrates provider calls, tool execution, and memory.
pub struct Agent {
    provider: Arc<dyn Provider>,
    /// Tools the agent can dispatch to. Stored as `Arc<dyn Tool>` so a
    /// single tool instance can be shared across multiple Agent
    /// instances (the OpenAI-compat channel constructs a fresh agent
    /// per HTTP request and reuses the same tool registry — see
    /// E-2-3). The legacy `AgentBuilder::tool(Box<dyn Tool>)` API is
    /// preserved by transparently converting `Box -> Arc` at insertion
    /// time, so existing call sites don't change.
    tools: Vec<Arc<dyn Tool>>,
    tool_specs: Vec<ToolSpec>,
    /// Tool names the user has disabled via `/tools disable`.
    /// Disabled tools stay registered (so re-enabling is cheap)
    /// but are filtered out of `tool_specs` shown to the model
    /// and rejected at dispatch time.
    disabled_tools: HashSet<String>,
    /// Image attachments queued by `/image` and `/paste` for
    /// the next user turn. Drained at the top of `turn` /
    /// `turn_streaming` and attached to the outbound user
    /// `ChatMessage`. Mirrors Hermes'
    /// `session["attached_images"]` (`tui_gateway/server.py:3361-3401`).
    pending_attachments: Vec<super::attachment::ImageAttachment>,
    /// Steer text queued via `/steer` while a turn is running
    /// (or pre-queued for the next turn). Drained after each
    /// tool batch; appended as "User guidance: <text>" to the
    /// last tool result so the model sees it before its next
    /// reply. Multiple steer calls concatenate with newlines.
    /// Mirrors `run_agent.py:4493-4527` + `:4545-4600`.
    pending_steer: Option<String>,
    memory: Arc<dyn Memory>,
    prompt_builder: SystemPromptBuilder,
    max_tool_iterations: usize,
    history: Vec<ChatMessage>,
    system_prompt: Option<String>,
    max_tokens: usize,
    temperature: f64,
    memory_context_limit: usize,
    half_life_days: f64,
    prompt_guard: Option<PromptGuard>,
    collective: Option<Arc<CollectiveSearch>>,
    /// Pre-rendered skills prompt fragment injected into the system prompt
    /// (see [`crate::skills::SkillsLoader::build_skills_prompt`]). Empty when
    /// no skills are loaded.
    skills_prompt: String,
    thinking_level: ThinkingLevel,
    // Token usage tracking
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cache_read_tokens: u64,
    total_cache_write_tokens: u64,
    /// Number of provider HTTP calls made in this session.
    total_api_calls: u64,
    /// Input-token count from the most recent provider response.
    /// Drives the context-window utilisation panel in `/usage`.
    last_prompt_tokens: u64,
    turn_count: u64,
    /// Plugin-registered lifecycle hooks.
    hooks: Arc<crate::plugins::HookRegistry>,
    /// Pluggable memory augmentation layer.
    memory_manager: Arc<crate::plugins::MemoryManager>,
    /// Resolved Fennec home directory.
    home_dir: std::path::PathBuf,
    /// Stable session identifier for the current session.
    session_id: String,
    /// Whether the current session's `on_session_start` has fired.
    session_started: bool,
    /// Auxiliary client for background tasks (curator, etc.).
    auxiliary_client: Arc<crate::providers::AuxiliaryClient>,
    /// Iterations the most recent `turn` / `turn_streaming` call
    /// consumed (the count of provider rounds it took to settle
    /// on a final assistant message). Reset at the top of each
    /// turn; surfaced via [`Self::last_turn_iterations`] so
    /// sub-agent observers can record it.
    last_turn_iterations: u32,
    /// Cooperative-interrupt flag. When set, the tool-call loop
    /// bails at the next iteration boundary with an
    /// `interrupted` error. Wired by [`crate::agent::delegation`]
    /// for sub-agents the user kills via `/agents x` / `X`. Main
    /// agents leave it null — interrupt isn't surfaced for them.
    interrupt_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Lifecycle hooks for real-time frontends (TUI, dashboard).
    callbacks: super::callbacks::CallbacksHandle,
}

/// Result of a single `turn_with_history` call.
#[derive(Debug, Clone)]
pub struct TurnWithHistoryResult {
    pub response: String,
    pub new_messages: Vec<ChatMessage>,
}

impl Agent {
    /// Execute a single turn against a caller-supplied conversation
    /// history, instead of the agent's own accumulating history.
    /// The agent's own `history` field is preserved; this is a
    /// strictly additive variant that doesn't observe or mutate
    /// the agent's default-session state.
    ///
    /// Used by the OpenAI-compat channel for session-scoped
    /// conversations: each `/v1/chat/completions` (with
    /// `X-Fennec-Session-Id`) and `/v1/responses` request loads its
    /// own history from disk, runs the turn, then persists the new
    /// messages back. The default-session agent path (CLI, channel
    /// listeners) is untouched.
    ///
    /// Implementation: swap `self.history` for the supplied vec,
    /// call `turn()`, capture the mutated history, swap back. On
    /// error the original `self.history` is restored before the
    /// error propagates.
    pub async fn turn_with_history(
        &mut self,
        supplied_history: Vec<ChatMessage>,
        user_message: &str,
    ) -> Result<TurnWithHistoryResult> {
        let pre_len = supplied_history.len();
        // Stash the agent's own history; install the caller's.
        let original = std::mem::replace(&mut self.history, supplied_history);
        // Run the turn. `self.history` now holds the supplied
        // messages and may have grown if turn() succeeded; or it
        // may be in a half-mutated state if turn() failed midway.
        let result = self.turn(user_message).await;
        // Take whatever final state turn() left and restore the
        // agent's own history. After this, self.history matches
        // pre-call exactly; the supplied-history-plus-new-messages
        // lives in `final_history`.
        let final_history = std::mem::replace(&mut self.history, original);
        let response = result?;
        // The new messages are everything turn() added on top of
        // the supplied prefix.
        let new_messages = if final_history.len() > pre_len {
            final_history[pre_len..].to_vec()
        } else {
            Vec::new()
        };
        Ok(TurnWithHistoryResult {
            response,
            new_messages,
        })
    }

    /// Execute a single conversational turn.
    ///
    /// On the first turn the system prompt is built (with memory context) and
    /// frozen for the remainder of the session. Subsequent turns reuse it.
    pub async fn turn(&mut self, user_message: &str) -> Result<String> {
        // Notify any registered frontend that a turn is starting.
        // This is the very first hook so a TUI / dashboard sees
        // even the prompt-guard rejection path light up.
        self.callbacks.on_turn_start(user_message);

        // Parse thinking directive before any other processing.
        let (thinking_override, user_message) = thinking::parse_thinking_directive(user_message);
        if let Some(level) = thinking_override {
            self.thinking_level = level;
            tracing::info!(?level, "Thinking level set via /think directive");
        }
        let user_message: &str = &user_message;

        // Memory provider observer hook: turn-start. Logged
        // failures are absorbed inside the manager so a misbehaving
        // provider can't abort the turn.
        self.memory_manager.on_turn_start(user_message).await;

        // Prompt guard scan.
        if let Some(ref guard) = self.prompt_guard {
            match guard.scan(user_message) {
                ScanResult::Blocked(reason) => {
                    bail!("{reason}");
                }
                ScanResult::Suspicious(categories, score) => {
                    tracing::warn!(
                        ?categories,
                        score,
                        "Prompt guard: suspicious input detected"
                    );
                }
                ScanResult::Safe => {}
            }
        }

        // Step 3: Search collective for relevant experiences.
        let collective_context = if let Some(ref collective) = self.collective {
            match collective.search(user_message, 3).await {
                Ok(result) => match result.confidence {
                    SearchConfidence::High => {
                        let formatted =
                            self.format_collective_results(&result.experiences, true);
                        Some(formatted)
                    }
                    SearchConfidence::Partial => {
                        let formatted =
                            self.format_collective_results(&result.experiences, false);
                        Some(formatted)
                    }
                    SearchConfidence::None => None,
                },
                Err(e) => {
                    tracing::warn!("Collective search failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Build system prompt on first turn.
        if self.system_prompt.is_none() {
            let memory_context = self.load_memory_context(user_message).await?;
            let tool_names: Vec<String> = self.tools.iter().map(|t| t.name().to_string()).collect();
            let prompt = self.prompt_builder.build(
                &memory_context,
                &tool_names,
                &self.skills_prompt,
            );
            self.system_prompt = Some(prompt);
        }

        // Fire on_session_start exactly once per session, right
        // before the first turn produces user-visible output.
        // Subsequent turns no-op; clear_history resets the flag so
        // a fresh conversation re-fires the hook.
        if !self.session_started {
            self.hooks.fire_on_session_start(&self.session_id);

            // Memory provider per-session initialize. Runs once per
            // session, after `clear_history` resets the flag.
            // `fennec_home` is the resolved profile-aware path the
            // agent was built with — providers reading this field
            // get `~/.fennec/profiles/<name>/` when `--profile` is
            // active, not a hardcoded `~/.fennec`. Failures are
            // logged inside the manager and do not abort the turn.
            let init_ctx = crate::plugins::MemoryProviderContext {
                session_id: self.session_id.clone(),
                fennec_home: self.home_dir.clone(),
                platform: "agent".to_string(),
            };
            if let Err(e) = self.memory_manager.initialize(&init_ctx).await {
                tracing::warn!(
                    "MemoryProvider initialize failed: {e}; continuing without it"
                );
            }
            self.session_started = true;
        }

        // Memory provider prefetch — additional context the
        // external provider contributes for THIS turn. Returned as a
        // formatted string the agent embeds in the user message.
        // Empty when no external provider is wired.
        let provider_prefetch = self.memory_manager.prefetch_for_turn(user_message).await;

        // Inject collective + provider context into the user message.
        // Both layers are optional; when both are absent the
        // effective_message is identical to the original user
        // message (current behavior).
        let effective_message = match (collective_context, provider_prefetch.is_empty()) {
            (Some(ctx), false) => format!(
                "[Collective matches]\n{ctx}\n[Memory provider]\n{}\n[User message]\n{}",
                provider_prefetch, user_message
            ),
            (Some(ctx), true) => {
                format!("[Collective matches]\n{ctx}\n[User message]\n{user_message}")
            }
            (None, false) => format!(
                "[Memory provider]\n{}\n[User message]\n{}",
                provider_prefetch, user_message
            ),
            (None, true) => user_message.to_string(),
        };

        // Push user message to history, attaching any images
        // queued via /image or /paste so the provider can send
        // them inline with this turn's prompt.
        let mut user_msg = ChatMessage::user(&effective_message);
        user_msg.attachments = self.take_pending_attachments_for_turn();
        self.history.push(user_msg);

        self.turn_count += 1;
        self.last_turn_iterations = 0;
        // Clear any stale interrupt flag from a prior turn so
        // the user's next prompt actually runs. `/busy interrupt`
        // sets the flag mid-turn; this is where we reset it.
        if let Some(ref flag) = self.interrupt_flag {
            flag.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        let turn_start = std::time::Instant::now();
        let tokens_before_input = self.total_input_tokens;
        let tokens_before_output = self.total_output_tokens;

        // Tool call loop.
        for _iteration in 0..self.max_tool_iterations {
            self.last_turn_iterations = self.last_turn_iterations.saturating_add(1);
            if self.is_interrupted() {
                bail!("interrupted by user");
            }
            self.callbacks.on_status("calling provider");
            let response = self.call_provider().await?;
            self.total_api_calls += 1;

            // Track token usage from this API call.
            if let Some(ref usage) = response.usage {
                self.total_input_tokens += usage.input_tokens;
                self.total_output_tokens += usage.output_tokens;
                self.last_prompt_tokens = usage.input_tokens
                    + usage.cache_read_tokens.unwrap_or(0)
                    + usage.cache_write_tokens.unwrap_or(0);
                if let Some(cache) = usage.cache_read_tokens {
                    self.total_cache_read_tokens += cache;
                }
                if let Some(cache) = usage.cache_write_tokens {
                    self.total_cache_write_tokens += cache;
                }
            }

            // Forward reasoning text (if any) before regular content
            // so frontends render thinking blocks first, matching the
            // streaming-path order.
            if let Some(ref reasoning) = response.reasoning {
                self.callbacks.on_reasoning_delta(reasoning);
            }

            if response.tool_calls.is_empty() {
                // No tool calls — final assistant response.
                let text = response.content.unwrap_or_default();
                self.history.push(ChatMessage::assistant(&text));
                self.callbacks.on_turn_complete(&text);

                // Log token usage for this turn.
                let turn_input = self.total_input_tokens - tokens_before_input;
                let turn_output = self.total_output_tokens - tokens_before_output;
                let elapsed = turn_start.elapsed();
                tracing::info!(
                    turn = self.turn_count,
                    turn_input_tokens = turn_input,
                    turn_output_tokens = turn_output,
                    turn_total_tokens = turn_input + turn_output,
                    turn_time_ms = elapsed.as_millis() as u64,
                    session_total_input = self.total_input_tokens,
                    session_total_output = self.total_output_tokens,
                    session_total_tokens = self.total_input_tokens + self.total_output_tokens,
                    session_cache_read = self.total_cache_read_tokens,
                    "Turn complete"
                );

                // Memory provider observer: completed turn. Pass
                // both messages so the provider can update its
                // model. Failures are absorbed inside the manager.
                self.memory_manager
                    .sync_turn(user_message, &text)
                    .await;

                return Ok(text);
            }

            // Push assistant message with tool calls. Preserve the
            // turn's reasoning text alongside it — Kimi K2.5 and a
            // few other reasoning-family endpoints 400 on the next
            // request if a tool-call assistant message echoes back
            // without its accompanying `reasoning_content`.
            let mut assistant_msg = ChatMessage::assistant(
                response.content.as_deref().unwrap_or(""),
            );
            assistant_msg.tool_calls = Some(response.tool_calls.clone());
            assistant_msg.reasoning = response.reasoning.clone();
            self.history.push(assistant_msg);

            // Execute each tool call and push results. Combines:
            //  - plugin lifecycle hooks (pre/post can skip or rewrite),
            //  - frontend callbacks (start + complete for live UI).
            for tc in &response.tool_calls {
                tracing::info!(tool = %tc.name, "Executing tool call");
                self.callbacks.on_tool_start(super::callbacks::ToolStart {
                    tool_id: tc.id.clone(),
                    name: tc.name.clone(),
                    preview: preview_for_args(&tc.arguments),
                    args: tc.arguments.clone(),
                });
                let started = std::time::Instant::now();
                let (final_output, final_success) =
                    match self.hooks.fire_pre_tool(&tc.name, &tc.arguments) {
                        crate::plugins::PreToolResolution::Skip { reason } => {
                            tracing::warn!(
                                tool = %tc.name,
                                reason = %reason,
                                "Tool call skipped by plugin pre_tool_call hook"
                            );
                            (format!("[skipped by plugin: {reason}]"), false)
                        }
                        crate::plugins::PreToolResolution::Continue { effective_args } => {
                            let (output, success) =
                                self.execute_tool(&tc.name, &effective_args).await;
                            let post = self.hooks.fire_post_tool(
                                &tc.name,
                                &effective_args,
                                &output,
                                success,
                            );
                            (post.output, post.success)
                        }
                    };
                tracing::info!(tool = %tc.name, success = %final_success, "Tool call complete");
                self.callbacks.on_tool_complete(super::callbacks::ToolComplete {
                    tool_id: tc.id.clone(),
                    name: tc.name.clone(),
                    duration_ms: started.elapsed().as_millis() as u64,
                    error: if final_success { None } else { Some(final_output.clone()) },
                    summary: Some(truncate_summary(&final_output)),
                });
                self.history
                    .push(ChatMessage::tool_result(&tc.id, &final_output));
            }

            // Drain any /steer text queued during the tool
            // batch so the model sees it before its next
            // response. If no tool ran (impossible at this
            // point — the loop body ran), apply_pending_steer
            // is a no-op.
            self.apply_pending_steer_to_tool_results();
        }

        // Budget exhausted: end gracefully with a tool-less summary instead of
        // erroring the whole turn. See `summarize_after_max_iterations`.
        self.callbacks
            .on_status("max iterations reached — requesting summary");
        self.summarize_after_max_iterations().await
    }

    /// Execute a single conversational turn with streaming.
    ///
    /// Returns a receiver of [`StreamEvent`]s. Text deltas are forwarded in
    /// real-time while tool calls are processed internally (their results are
    /// *not* streamed back; only the final text response is).
    ///
    /// Brings the same setup work as [`Self::turn`] to bear: parses
    /// `/think:<level>` directives, runs the prompt guard, injects
    /// collective-search context, builds the system prompt on first call.
    /// The caller is responsible for recording the final assistant text via
    /// [`Self::record_streamed_response`] once the stream drains.
    pub async fn turn_streamed(
        &mut self,
        user_message: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        // Parse thinking directive before any other processing — same as turn().
        let (thinking_override, user_message) = thinking::parse_thinking_directive(user_message);
        if let Some(level) = thinking_override {
            self.thinking_level = level;
            tracing::info!(?level, "Thinking level set via /think directive (stream)");
        }
        let user_message: &str = &user_message;

        // Memory provider observer hook: turn-start. Logged
        // failures are absorbed inside the manager so a misbehaving
        // provider can't abort the turn.
        self.memory_manager.on_turn_start(user_message).await;

        // Prompt guard scan.
        if let Some(ref guard) = self.prompt_guard {
            match guard.scan(user_message) {
                ScanResult::Blocked(reason) => {
                    bail!("{reason}");
                }
                ScanResult::Suspicious(categories, score) => {
                    tracing::warn!(
                        ?categories,
                        score,
                        "Prompt guard: suspicious input detected (stream)"
                    );
                }
                ScanResult::Safe => {}
            }
        }

        // Search collective for relevant experiences (mirrors turn()).
        let collective_context = if let Some(ref collective) = self.collective {
            match collective.search(user_message, 3).await {
                Ok(result) => match result.confidence {
                    SearchConfidence::High => Some(
                        self.format_collective_results(&result.experiences, true),
                    ),
                    SearchConfidence::Partial => Some(
                        self.format_collective_results(&result.experiences, false),
                    ),
                    SearchConfidence::None => None,
                },
                Err(e) => {
                    tracing::warn!("Collective search failed (stream): {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Build system prompt on first turn.
        if self.system_prompt.is_none() {
            let memory_context = self.load_memory_context(user_message).await?;
            let tool_names: Vec<String> = self.tools.iter().map(|t| t.name().to_string()).collect();
            let prompt = self.prompt_builder.build(
                &memory_context,
                &tool_names,
                &self.skills_prompt,
            );
            self.system_prompt = Some(prompt);
        }

        // Fire on_session_start exactly once per session, right
        // before the first turn produces user-visible output.
        // Subsequent turns no-op; clear_history resets the flag so
        // a fresh conversation re-fires the hook.
        if !self.session_started {
            self.hooks.fire_on_session_start(&self.session_id);
            self.session_started = true;
        }

        // Inject collective context into user message if available.
        let effective_message = if let Some(ref ctx) = collective_context {
            format!("[Collective matches]\n{ctx}\n[User message]\n{user_message}")
        } else {
            user_message.to_string()
        };

        // Push user message to history, attaching any images
        // queued via /image or /paste so the provider can send
        // them inline with this turn's prompt.
        let mut user_msg = ChatMessage::user(&effective_message);
        user_msg.attachments = self.take_pending_attachments_for_turn();
        self.history.push(user_msg);

        // Get streaming receiver from provider.
        let system = self.system_prompt.as_deref();
        let enabled = self.enabled_tool_specs();
        let tools = if enabled.is_empty() {
            None
        } else {
            Some(enabled.as_slice())
        };

        let request = ChatRequest {
            system,
            messages: &self.history,
            tools,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            thinking_level: self.thinking_level,
        };

        let rx = self.provider.chat_stream(request).await?;
        Ok(rx)
    }

    /// Streaming version of [`Self::turn`] that fans text deltas
    /// through the registered callbacks as they arrive. Same
    /// final-result contract as `turn()` (returns the assistant
    /// text on success, errors on max-iterations exceeded /
    /// guard rejection / provider failure), with the addition
    /// that frontends see live updates throughout.
    ///
    /// The default no-op `AgentCallbacks` impl makes this safe
    /// for callers that don't care about streaming — the deltas
    /// fire into nothing. Channels (Telegram / etc.) keep using
    /// `turn()` since they only consume final messages and don't
    /// need the streaming overhead.
    pub async fn turn_streaming(&mut self, user_message: &str) -> Result<String> {
        use crate::providers::traits::{ChatRequest, StreamEvent, ToolCall};

        self.callbacks.on_turn_start(user_message);

        // Same setup as turn(): /think parsing, prompt guard,
        // collective search, system prompt, history push.
        let (thinking_override, user_message) = thinking::parse_thinking_directive(user_message);
        if let Some(level) = thinking_override {
            self.thinking_level = level;
        }
        let user_message: &str = &user_message;

        if let Some(ref guard) = self.prompt_guard {
            match guard.scan(user_message) {
                ScanResult::Blocked(reason) => bail!("{reason}"),
                ScanResult::Suspicious(_, _) | ScanResult::Safe => {}
            }
        }

        let collective_context = if let Some(ref collective) = self.collective {
            match collective.search(user_message, 3).await {
                Ok(result) => match result.confidence {
                    SearchConfidence::High => {
                        Some(self.format_collective_results(&result.experiences, true))
                    }
                    SearchConfidence::Partial => {
                        Some(self.format_collective_results(&result.experiences, false))
                    }
                    SearchConfidence::None => None,
                },
                Err(_) => None,
            }
        } else {
            None
        };

        if self.system_prompt.is_none() {
            let memory_context = self.load_memory_context(user_message).await?;
            let tool_names: Vec<String> = self.tools.iter().map(|t| t.name().to_string()).collect();
            let prompt = self.prompt_builder.build(
                &memory_context,
                &tool_names,
                &self.skills_prompt,
            );
            self.system_prompt = Some(prompt);
        }

        let effective_message = if let Some(ref ctx) = collective_context {
            format!("[Collective matches]\n{ctx}\n[User message]\n{user_message}")
        } else {
            user_message.to_string()
        };
        self.history.push(ChatMessage::user(&effective_message));
        self.turn_count += 1;
        self.last_turn_iterations = 0;
        // Clear any stale interrupt flag from a prior turn. See
        // matching note in `turn` above.
        if let Some(ref flag) = self.interrupt_flag {
            flag.store(false, std::sync::atomic::Ordering::SeqCst);
        }

        // Tool-iteration loop, streaming each LLM call.
        for _iteration in 0..self.max_tool_iterations {
            self.last_turn_iterations = self.last_turn_iterations.saturating_add(1);
            if self.is_interrupted() {
                bail!("interrupted by user");
            }
            self.callbacks.on_status("calling provider");

            let enabled = self.enabled_tool_specs();
            let request = ChatRequest {
                system: self.system_prompt.as_deref(),
                messages: &self.history,
                tools: if enabled.is_empty() {
                    None
                } else {
                    Some(enabled.as_slice())
                },
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                thinking_level: self.thinking_level,
            };
            let mut rx = self.provider.chat_stream(request).await?;
            self.total_api_calls += 1;

            let mut accumulated_text = String::new();
            let mut accumulated_reasoning = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            // (id, name, args_buffer) for the in-flight tool call.
            let mut current_tool: Option<(String, String, String)> = None;

            while let Some(ev) = rx.recv().await {
                match ev {
                    StreamEvent::Delta(text) => {
                        accumulated_text.push_str(&text);
                        self.callbacks.on_text_delta(&text);
                    }
                    StreamEvent::Reasoning(text) => {
                        // Forward reasoning deltas to the
                        // frontend AND accumulate them so we can
                        // attach the full reasoning trace to the
                        // assistant message — Kimi K2.5 and other
                        // reasoning-family providers 400 on the
                        // next request when a tool-call assistant
                        // message echoes back without its
                        // accompanying `reasoning_content`.
                        accumulated_reasoning.push_str(&text);
                        self.callbacks.on_reasoning_delta(&text);
                    }
                    StreamEvent::ToolCallStart { id, name } => {
                        current_tool = Some((id, name, String::new()));
                    }
                    StreamEvent::ToolCallDelta {
                        id: _,
                        arguments_delta,
                    } => {
                        if let Some((_, _, ref mut buf)) = current_tool {
                            buf.push_str(&arguments_delta);
                        }
                    }
                    StreamEvent::ToolCallEnd { id: _ } => {
                        if let Some((id, name, args)) = current_tool.take() {
                            let arguments: serde_json::Value =
                                serde_json::from_str(&args).unwrap_or_else(|_| {
                                    serde_json::Value::Object(serde_json::Map::new())
                                });
                            tool_calls.push(ToolCall {
                                id,
                                name,
                                arguments,
                            });
                        }
                    }
                    StreamEvent::Usage(usage) => {
                        self.total_input_tokens += usage.input_tokens;
                        self.total_output_tokens += usage.output_tokens;
                        self.last_prompt_tokens = usage.input_tokens
                            + usage.cache_read_tokens.unwrap_or(0)
                            + usage.cache_write_tokens.unwrap_or(0);
                        if let Some(cache) = usage.cache_read_tokens {
                            self.total_cache_read_tokens += cache;
                        }
                        if let Some(cache) = usage.cache_write_tokens {
                            self.total_cache_write_tokens += cache;
                        }
                    }
                    StreamEvent::Done => break,
                    StreamEvent::Error(e) => bail!("provider stream error: {e}"),
                }
            }

            if tool_calls.is_empty() {
                // Final assistant response.
                self.history.push(ChatMessage::assistant(&accumulated_text));
                self.callbacks.on_turn_complete(&accumulated_text);
                return Ok(accumulated_text);
            }

            // Has tool calls — push the assistant message + tool
            // results, then loop for the next provider call.
            let mut assistant_msg = ChatMessage::assistant(&accumulated_text);
            assistant_msg.tool_calls = Some(tool_calls.clone());
            if !accumulated_reasoning.is_empty() {
                assistant_msg.reasoning = Some(accumulated_reasoning.clone());
            }
            self.history.push(assistant_msg);

            for tc in &tool_calls {
                self.callbacks.on_tool_start(super::callbacks::ToolStart {
                    tool_id: tc.id.clone(),
                    name: tc.name.clone(),
                    preview: preview_for_args(&tc.arguments),
                    args: tc.arguments.clone(),
                });
                let started = std::time::Instant::now();
                let (output, success) = self.execute_tool(&tc.name, &tc.arguments).await;
                self.callbacks.on_tool_complete(super::callbacks::ToolComplete {
                    tool_id: tc.id.clone(),
                    name: tc.name.clone(),
                    duration_ms: started.elapsed().as_millis() as u64,
                    error: if success { None } else { Some(output.clone()) },
                    summary: Some(truncate_summary(&output)),
                });
                self.history
                    .push(ChatMessage::tool_result(&tc.id, &output));
            }

            // Drain any /steer text queued during the tool batch
            // (streaming path). Same hook as turn(): inject after
            // tool results before looping for the next provider call.
            self.apply_pending_steer_to_tool_results();
        }

        // Budget exhausted (streaming path): same graceful tool-less summary.
        self.callbacks
            .on_status("max iterations reached — requesting summary");
        self.summarize_after_max_iterations().await
    }

    /// Record the final assistant text after a [`Self::turn_streamed`]
    /// receiver has drained, so the next turn sees it in history. Callers
    /// that drive `turn_streamed` must invoke this once the stream's
    /// [`StreamEvent::Done`] arrives — `turn_streamed` itself can't capture
    /// the final text because it returns before the model has finished.
    pub fn record_streamed_response(&mut self, text: impl Into<String>) {
        self.history.push(ChatMessage::assistant(text.into()));
    }

    /// Call the provider with the current history and system prompt.
    async fn call_provider(&self) -> Result<ChatResponse> {
        let system = self.system_prompt.as_deref();
        let enabled = self.enabled_tool_specs();
        let tools = if enabled.is_empty() {
            None
        } else {
            Some(enabled.as_slice())
        };

        let request = ChatRequest {
            system,
            messages: &self.history,
            tools,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            thinking_level: self.thinking_level,
        };

        // pre_llm_call observers — serialise the message list only
        // when at least one hook is registered. The default path
        // (no plugins) has an empty `pre_llm` Vec and pays nothing.
        if self.hooks.has_pre_llm() {
            let messages_json = serde_json::to_string(&self.history)
                .unwrap_or_else(|_| "[]".to_string());
            self.hooks.fire_pre_llm(&messages_json);
        }

        let response = self.provider.chat(request).await;

        // post_llm_call observers — same gating. Errors are not
        // surfaced to observers today; if a use case for "observe
        // failures" emerges we can add a separate hook kind later.
        if let Ok(ref resp) = response {
            if self.hooks.has_post_llm() {
                let response_json =
                    serde_json::to_string(resp).unwrap_or_else(|_| "{}".to_string());
                self.hooks.fire_post_llm(&response_json);
            }
        }

        response
    }

    /// Graceful end-of-budget handler. When the tool-iteration cap is hit we
    /// do NOT error the turn — instead we make one final provider call with
    /// tools disabled, nudging the model to summarise what it accomplished,
    /// and return that text. A budget-exhausted turn should still give the
    /// user a usable answer, not a stack-trace-style failure.
    ///
    /// The summary instruction is appended to the trailing tool-result message
    /// — the same channel `/steer` uses — so providers that require strict
    /// user/assistant role alternation (Anthropic in particular) never see two
    /// consecutive user turns. When there is no trailing tool result to attach
    /// to (degenerate case, e.g. a zero-iteration cap), we fall back to a
    /// standalone user message.
    async fn summarize_after_max_iterations(&mut self) -> Result<String> {
        const SUMMARY_INSTRUCTION: &str = "[System] You've reached the maximum \
            number of tool-calling iterations allowed for this turn. Provide a \
            final response summarising what you found and accomplished so far. \
            Do not attempt to call any more tools.";

        match self.history.last_mut() {
            Some(last) if last.role == "tool" => {
                let content = last.content.get_or_insert_with(String::new);
                content.push_str("\n\n");
                content.push_str(SUMMARY_INSTRUCTION);
            }
            _ => {
                self.history.push(ChatMessage::user(SUMMARY_INSTRUCTION));
            }
        }

        // Tools stripped — the model must answer in text. Clone the provider
        // Arc so the request can borrow `self` immutably across the await
        // without conflicting with the provider handle.
        let provider = Arc::clone(&self.provider);
        let request = ChatRequest {
            system: self.system_prompt.as_deref(),
            messages: &self.history,
            tools: None,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            thinking_level: self.thinking_level,
        };
        let response = provider.chat(request).await?;
        self.total_api_calls += 1;

        // Track usage from the summary call so /usage stays accurate.
        if let Some(ref usage) = response.usage {
            self.total_input_tokens += usage.input_tokens;
            self.total_output_tokens += usage.output_tokens;
            self.last_prompt_tokens = usage.input_tokens
                + usage.cache_read_tokens.unwrap_or(0)
                + usage.cache_write_tokens.unwrap_or(0);
            if let Some(c) = usage.cache_read_tokens {
                self.total_cache_read_tokens += c;
            }
            if let Some(c) = usage.cache_write_tokens {
                self.total_cache_write_tokens += c;
            }
        }

        let text = response.content.unwrap_or_default();
        self.history.push(ChatMessage::assistant(&text));
        self.callbacks.on_turn_complete(&text);
        Ok(text)
    }

    /// Find a tool by name and execute it. Returns the formatted output string
    /// with credentials scrubbed, plus the structured success flag from the
    /// tool itself (`true` only when the tool reported success — tool output
    /// containing the substring "error" must not be confused with failure).
    async fn execute_tool(&self, name: &str, args: &serde_json::Value) -> (String, bool) {
        // Disabled tools must not run.
        if self.disabled_tools.contains(name) {
            return (
                format!("Error: tool '{name}' is currently disabled"),
                false,
            );
        }

        // Route memory-provider-contributed tools to the manager first.
        if self.memory_manager.handles_tool(name) {
            let (raw, success) = match self
                .memory_manager
                .handle_tool_call(name, args.clone())
                .await
            {
                Ok(result) => {
                    if result.success {
                        (result.output, true)
                    } else {
                        (
                            format!(
                                "Error: {}",
                                result.error.unwrap_or_else(|| "unknown error".to_string())
                            ),
                            false,
                        )
                    }
                }
                Err(e) => (format!("Memory provider tool failed: {e}"), false),
            };
            return (scrub::scrub_credentials(&raw), success);
        }

        let (raw, success) = match self.tools.iter().find(|t| t.name() == name) {
            Some(t) => match t.execute(args.clone()).await {
                Ok(result) => {
                    if result.success {
                        (result.output, true)
                    } else {
                        (
                            format!(
                                "Error: {}",
                                result.error.unwrap_or_else(|| "unknown error".to_string())
                            ),
                            false,
                        )
                    }
                }
                Err(e) => (format!("Tool execution failed: {e}"), false),
            },
            None => (format!("Unknown tool: {name}"), false),
        };
        (scrub::scrub_credentials(&raw), success)
    }

    /// Load memory context for the given query.
    ///
    /// Recalls entries from memory, applies time decay, sorts by score
    /// descending, and formats as `"[key] content"` strings.
    async fn load_memory_context(&self, query: &str) -> Result<Vec<String>> {
        let mut entries = self.memory.recall(query, self.memory_context_limit).await?;

        apply_time_decay(&mut entries, self.half_life_days);

        // Sort by score descending.
        entries.sort_by(|a, b| {
            let sa = a.score.unwrap_or(0.0);
            let sb = b.score.unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let formatted: Vec<String> = entries
            .iter()
            .map(|e| format!("[{}] {}", e.key, e.content))
            .collect();

        Ok(formatted)
    }

    /// Format collective search results for injection into the user message.
    ///
    /// Only includes goal titles and IDs — the agent can use the
    /// `collective_get_experience` tool to fetch full details for any
    /// experience it finds relevant.
    fn format_collective_results(
        &self,
        results: &[RankedExperience],
        high_confidence: bool,
    ) -> String {
        let mut output = if high_confidence {
            "The collective has relevant experiences. Use collective_get_experience to get details if useful:\n\n".to_string()
        } else {
            "Possibly related experiences from the collective. Use collective_get_experience to get details if useful:\n\n".to_string()
        };
        for (i, ranked) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. \"{}\" (id: {}, score: {:.2})\n",
                i + 1,
                ranked.result.goal,
                ranked.result.id,
                ranked.final_score
            ));
        }
        output
    }

    /// Publish an experience to the collective if configured and publish is enabled.
    ///
    /// This can be called when an experience is manually extracted or when
    /// consolidation creates one. It scrubs the experience before publishing.
    pub async fn publish_experience_if_configured(
        &self,
        experience: &crate::memory::experience::Experience,
    ) -> Result<Option<String>> {
        let _collective = match self.collective {
            Some(ref c) => c,
            None => return Ok(None),
        };
        // We only have search access; publishing requires the remote layer.
        // For now log that publishing would occur.
        tracing::info!(
            experience_id = %experience.id,
            "Would publish experience to collective (requires direct remote access)"
        );
        Ok(None)
    }

    /// Clear conversation history and system prompt, resetting for a new session.
    ///
    /// Fires `on_session_end` for the outgoing session (if it had
    /// reached `on_session_start`), tells the active memory provider
    /// (if any) to shut down, generates a fresh `session_id`, and
    /// resets the start flag so the next turn re-fires
    /// `on_session_start`.
    ///
    /// Synchronous (matches the existing public API) — the provider
    /// shutdown is best-effort: we spawn it onto the current Tokio
    /// runtime so the future runs but `clear_history` returns
    /// immediately. A misbehaving provider can't make `/new` block.
    pub fn clear_history(&mut self) {
        if self.session_started {
            self.hooks.fire_on_session_end(&self.session_id);
            // Best-effort provider shutdown — only fire if a Tokio
            // runtime is currently available (gateway / agent paths
            // always have one). The spawn detaches; we don't await.
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let manager = Arc::clone(&self.memory_manager);
                handle.spawn(async move {
                    manager.shutdown().await;
                });
            }
        }
        self.history.clear();
        self.system_prompt = None;
        self.session_id = uuid::Uuid::new_v4().to_string();
        self.session_started = false;
    }

    /// Replace the agent's working history with a prior session's
    /// messages. Used by `/resume` to repopulate `history` from
    /// the persisted [`crate::sessions::SessionStore`] so the
    /// next turn sees the full conversation.
    ///
    /// Resets the cached system prompt so it's rebuilt with
    /// memory context relevant to whatever the user types after
    /// resuming, mirroring `clear_history`'s semantics.
    pub fn replace_history(&mut self, messages: Vec<ChatMessage>) {
        self.history = messages;
        self.system_prompt = None;
    }

    /// Number of messages currently in `history`. Used by the
    /// per-turn persistence hook in the TUI: snapshot the length
    /// before a turn, persist the slice from that point after
    /// the turn completes.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Read-only view of `history[start..]`. Returns an empty
    /// slice if `start` is out of bounds.
    pub fn history_slice(&self, start: usize) -> &[ChatMessage] {
        if start >= self.history.len() {
            &[]
        } else {
            &self.history[start..]
        }
    }

    /// Pop messages from the tail of `history` until (and
    /// including) the most recent user-role message is removed.
    /// Returns `(count_popped, user_text)` on success, `None`
    /// when no user message exists (history is left untouched
    /// in that case).
    ///
    /// Used by `/undo` (to drop the last exchange) and `/retry`
    /// (to drop + re-submit the user message). Mirrors Hermes'
    /// `session.undo` (`tui_gateway/server.py:2424-2449`) which
    /// pops in reverse until a user-role row is found.
    pub fn pop_last_turn(&mut self) -> Option<(usize, String)> {
        // First locate the index of the most recent user
        // message — only then do we mutate. Avoids corrupting
        // history if the tail somehow has no user row (a
        // pathological state, but we shouldn't panic on it).
        let user_idx = self
            .history
            .iter()
            .rposition(|m| m.role == "user")?;
        let user_text = self.history[user_idx]
            .content
            .clone()
            .unwrap_or_default();
        let popped = self.history.len() - user_idx;
        self.history.truncate(user_idx);
        Some((popped, user_text))
    }

    /// Snapshot of cumulative session token usage + cost. Returned
    /// by [`Self::token_usage`] for the `/usage` command.
    ///
    /// `cost_usd` is `None` when the active model isn't in the
    /// pricing snapshot — callers render that as "—" or skip the
    /// row, matching upstream's `cost_status: "unknown"` behavior.
    /// Iterations consumed by the most recent `turn` /
    /// `turn_streaming` call. 0 before the first turn. Sub-agent
    /// observers read this after `turn` returns to record the
    /// `iteration` field on `SubagentComplete`.
    pub fn last_turn_iterations(&self) -> u32 {
        self.last_turn_iterations
    }

    /// True when an external interrupt flag has been set on this
    /// agent. Cooperative — the tool loop bails at the next
    /// iteration boundary; in-flight provider calls finish first.
    fn is_interrupted(&self) -> bool {
        match self.interrupt_flag.as_ref() {
            Some(flag) => flag.load(std::sync::atomic::Ordering::SeqCst),
            None => false,
        }
    }

    pub fn token_usage(&self) -> TokenUsage {
        let model = self.provider.model().to_string();
        let context_max = self.provider.context_window();
        let cost_usd = super::pricing::estimate_cost(
            &model,
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_cache_read_tokens,
            self.total_cache_write_tokens,
        );
        TokenUsage {
            model,
            input_tokens: self.total_input_tokens,
            output_tokens: self.total_output_tokens,
            cache_read_tokens: self.total_cache_read_tokens,
            cache_write_tokens: self.total_cache_write_tokens,
            api_calls: self.total_api_calls,
            last_prompt_tokens: self.last_prompt_tokens,
            context_max,
            cost_usd,
        }
    }

    /// Get a reference to the shared provider.
    /// Handle to the auxiliary client. Curator + future background
    /// tasks call into this for non-prompt-cache-polluting LLM
    /// calls. Always present (empty client when no providers are
    /// available); call sites should `is_available()` first.
    pub fn auxiliary_client(&self) -> &Arc<crate::providers::AuxiliaryClient> {
        &self.auxiliary_client
    }

    pub fn provider(&self) -> &Arc<dyn Provider> {
        &self.provider
    }

    /// Swap the live provider — used by `/model` to switch
    /// models without restarting the process. Resets the cached
    /// system prompt because the new provider may want
    /// different prompt-injection conventions on its first
    /// turn (and to keep parity with `clear_history`'s prompt
    /// reset). Existing `history` is preserved.
    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        self.provider = provider;
        self.system_prompt = None;
    }

    /// Swap the persona string injected into the system prompt.
    /// Rebuilds the cached `SystemPromptBuilder` with the same
    /// name + new persona and clears `system_prompt` so the
    /// next turn rebuilds from scratch. Used by `/personality`.
    pub fn set_persona(&mut self, persona: impl Into<String>) {
        let name = self.prompt_builder.identity_name().to_string();
        self.prompt_builder = SystemPromptBuilder::new(name, persona);
        self.system_prompt = None;
    }

    /// Replace the pre-rendered skills prompt fragment and
    /// invalidate the cached system prompt so the next turn
    /// picks up the new skills. Used by `/reload-skills`.
    pub fn set_skills_prompt(&mut self, prompt: String) {
        self.skills_prompt = prompt;
        self.system_prompt = None;
    }

    /// LLM-driven history compression. Asks the provider to
    /// summarise everything except the last `keep_recent_turns`
    /// turns and replaces the older history with a single
    /// `system` message containing the summary. Returns
    /// `(messages_removed, summary_preview)` for the caller to
    /// surface to the user.
    ///
    /// `focus_topic` optionally biases the summarisation toward
    /// a topic the user named — matches the upstream's
    /// `/compress <topic>` argument.
    ///
    /// Conservative defaults: keeps the last 4 chat messages
    /// (user + assistant pairs) intact so the immediate context
    /// for the next turn is preserved.
    pub async fn compress_history(
        &mut self,
        focus_topic: Option<String>,
    ) -> Result<(usize, String)> {
        const KEEP_RECENT: usize = 4;
        let total = self.history.len();
        if total <= KEEP_RECENT + 1 {
            return Ok((0, String::new()));
        }
        let split_at = total - KEEP_RECENT;
        let older: Vec<ChatMessage> = self.history[..split_at].to_vec();
        let recent: Vec<ChatMessage> = self.history[split_at..].to_vec();

        // Build the summarisation request. Use the provider
        // directly with a fresh system prompt — we don't want
        // the agent's tools / persona overriding this call.
        let mut instr = String::from(
            "Summarise the following conversation concisely. Preserve key facts, \
             decisions made, open questions, and any in-flight tasks. Reply with \
             the summary text only — no preamble, no markdown headers.",
        );
        if let Some(topic) = focus_topic.as_deref().filter(|t| !t.is_empty()) {
            instr.push_str(&format!(
                "\nFocus the summary on the following topic: {topic}"
            ));
        }

        let mut summary_messages = older;
        summary_messages.push(ChatMessage::user(&instr));
        let request = ChatRequest {
            system: Some("You are a summarisation assistant. Produce a single concise summary."),
            messages: &summary_messages,
            tools: None,
            max_tokens: 2048,
            temperature: 0.3,
            thinking_level: super::thinking::ThinkingLevel::Off,
        };
        let response = self.provider.chat(request).await?;
        let summary = response
            .content
            .clone()
            .unwrap_or_default()
            .trim()
            .to_string();
        if summary.is_empty() {
            anyhow::bail!("provider returned empty summary");
        }

        // Replace older history with a single system message.
        let summary_marker = format!("[/compress summary] {summary}");
        let mut new_history = Vec::with_capacity(recent.len() + 1);
        new_history.push(ChatMessage::system(&summary_marker));
        new_history.extend(recent);
        let removed = self.history.len() - new_history.len();
        self.history = new_history;
        self.system_prompt = None;

        let preview: String = summary.chars().take(160).collect();
        let suffix = if summary.chars().count() > 160 { "…" } else { "" };
        Ok((removed, format!("{preview}{suffix}")))
    }

    /// Enumerate every registered tool's name, regardless of
    /// enabled state. Used by `/tools` (no arg) to show the user
    /// what's installed.
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name().to_string()).collect()
    }

    /// Whether `name` is currently enabled (not in the disabled
    /// set). Returns `true` if the tool isn't registered at all
    /// — callers wanting "exists and enabled" should pair this
    /// with `tool_names`.
    pub fn is_tool_enabled(&self, name: &str) -> bool {
        !self.disabled_tools.contains(name)
    }

    /// Toggle `name`'s enabled state. `enabled = false` adds it
    /// to the disabled set; `enabled = true` removes it. Returns
    /// `true` if the call changed anything (the tool exists and
    /// the state actually flipped). Resets the cached system
    /// prompt so the next turn rebuilds the tool-list section.
    pub fn set_tool_enabled(&mut self, name: &str, enabled: bool) -> bool {
        let exists = self.tools.iter().any(|t| t.name() == name);
        if !exists {
            return false;
        }
        let changed = if enabled {
            self.disabled_tools.remove(name)
        } else {
            self.disabled_tools.insert(name.to_string())
        };
        if changed {
            self.system_prompt = None;
        }
        changed
    }

    /// Replace the disabled set wholesale — used at agent
    /// construction to seed from `FennecConfig.tools.disabled`.
    pub fn set_disabled_tools<I, S>(&mut self, names: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.disabled_tools = names.into_iter().map(|s| s.into()).collect();
        self.system_prompt = None;
    }

    /// Sorted list of currently-disabled tool names, suitable for
    /// persistence into `FennecConfig.tools.disabled`.
    pub fn disabled_tool_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.disabled_tools.iter().cloned().collect();
        v.sort();
        v
    }

    /// Queue an image for the next user turn. The bytes are
    /// loaded + base64-encoded eagerly so the next provider
    /// call doesn't pay disk I/O. Returns the attachment so
    /// `/image` can echo dimensions + token estimate to the
    /// user.
    pub fn attach_image(
        &mut self,
        path: &std::path::Path,
    ) -> anyhow::Result<super::attachment::ImageAttachment> {
        let attached = super::attachment::ImageAttachment::from_path(path)?;
        self.pending_attachments.push(attached.clone());
        Ok(attached)
    }

    /// Number of attachments queued for the next turn. Powers
    /// `/usage`-style "X images attached" hints in the UI.
    pub fn pending_attachment_count(&self) -> usize {
        self.pending_attachments.len()
    }

    /// Drop all queued attachments without consuming them. Used
    /// by `/clear` and similar commands that reset turn state.
    pub fn clear_pending_attachments(&mut self) {
        self.pending_attachments.clear();
    }

    /// Append `text` to the pending-steer queue. Multiple calls
    /// before the next tool batch concatenate with newlines, so
    /// the model sees them as one block. Returns `true` if the
    /// text was accepted (matches Hermes' `agent.steer` return
    /// at `run_agent.py:4493-4527`).
    pub fn steer(&mut self, text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return false;
        }
        match &mut self.pending_steer {
            Some(existing) => {
                existing.push('\n');
                existing.push_str(trimmed);
            }
            None => self.pending_steer = Some(trimmed.to_string()),
        }
        true
    }

    /// Whether a steer message is currently queued. Used by the
    /// `/steer` worker to confirm acceptance.
    pub fn has_pending_steer(&self) -> bool {
        self.pending_steer.is_some()
    }

    /// Drain any pending steer text onto the most recent tool
    /// result message in `history`, formatted with the
    /// "User guidance:" marker that mirrors Hermes'
    /// `_apply_pending_steer_to_tool_results`
    /// (`run_agent.py:4545-4600`). Returns `true` if a steer
    /// was applied — callers can use this to decide whether to
    /// loop another tool iteration.
    ///
    /// If no tool result exists in history (i.e. the turn
    /// produced no tool calls before terminating), the steer
    /// stays queued for the next turn so it isn't lost.
    fn apply_pending_steer_to_tool_results(&mut self) -> bool {
        let Some(text) = self.pending_steer.clone() else {
            return false;
        };
        // Walk back to find the most recent tool-result row.
        let last_tool_idx = self
            .history
            .iter()
            .rposition(|m| m.role == "tool");
        let Some(idx) = last_tool_idx else {
            // No tool result this turn — leave the queue as-is
            // so the next turn picks it up.
            return false;
        };
        let appended = format!("\n\nUser guidance: {text}");
        if let Some(content) = self.history[idx].content.as_mut() {
            content.push_str(&appended);
        } else {
            self.history[idx].content = Some(appended);
        }
        self.pending_steer = None;
        true
    }

    /// Internal: drain queued attachments into the
    /// provider-facing form, used at the top of `turn` /
    /// `turn_streaming` to attach them to the outbound user
    /// message.
    fn take_pending_attachments_for_turn(
        &mut self,
    ) -> Option<Vec<crate::providers::traits::ImageAttachmentRef>> {
        if self.pending_attachments.is_empty() {
            return None;
        }
        let drained = std::mem::take(&mut self.pending_attachments);
        let refs: Vec<_> = drained
            .into_iter()
            .map(|a| crate::providers::traits::ImageAttachmentRef {
                mime_type: a.mime_type,
                base64_data: a.base64_data,
                display_name: Some(a.display_name),
            })
            .collect();
        Some(refs)
    }

    /// Subset of `tool_specs` actually advertised to the model
    /// — disabled entries are filtered out so the LLM doesn't
    /// see them as available.
    fn enabled_tool_specs(&self) -> Vec<ToolSpec> {
        self.tool_specs
            .iter()
            .filter(|spec| !self.disabled_tools.contains(&spec.name))
            .cloned()
            .collect()
    }

    /// Get a reference to the shared memory.
    pub fn memory(&self) -> &Arc<dyn Memory> {
        &self.memory
    }

    /// Get the current thinking level.
    pub fn thinking_level(&self) -> ThinkingLevel {
        self.thinking_level
    }

    /// Set the thinking level programmatically.
    pub fn set_thinking_level(&mut self, level: ThinkingLevel) {
        self.thinking_level = level;
    }
}

// ---------------------------------------------------------------------------
// AgentBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing an [`Agent`] with validated configuration.
///
/// Tools are stored as `Arc<dyn Tool>` (shared) rather than
/// `Box<dyn Tool>` (owned), so a single tool instance can be reused
/// across many `Agent` constructions. The legacy
/// `tool(Box<dyn Tool>)` and `tools(Vec<Box<dyn Tool>>)` setters
/// transparently convert via `Arc::from(Box)` so existing call sites
/// don't have to change. New callers (the OpenAI-compat session-
/// scoped agent factory) can use `tool_arc(Arc<dyn Tool>)` and
/// `tools_arc(Vec<Arc<dyn Tool>>)` to share a pre-built tool
/// registry across many fresh agents without reconstructing each
/// tool per request.
pub struct AgentBuilder {
    provider: Option<Arc<dyn Provider>>,
    tools: Vec<Arc<dyn Tool>>,
    memory: Option<Arc<dyn Memory>>,
    identity_name: Option<String>,
    identity_persona: Option<String>,
    max_tool_iterations: Option<usize>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    memory_context_limit: Option<usize>,
    half_life_days: Option<f64>,
    prompt_guard: Option<PromptGuard>,
    collective: Option<Arc<CollectiveSearch>>,
    skills_prompt: Option<String>,
    hooks: Option<Arc<crate::plugins::HookRegistry>>,
    memory_manager: Option<Arc<crate::plugins::MemoryManager>>,
    home_dir: Option<std::path::PathBuf>,
    auxiliary_client: Option<Arc<crate::providers::AuxiliaryClient>>,
    callbacks: Option<super::callbacks::CallbacksHandle>,
    interrupt_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            provider: None,
            tools: Vec::new(),
            memory: None,
            identity_name: None,
            identity_persona: None,
            max_tool_iterations: None,
            max_tokens: None,
            temperature: None,
            memory_context_limit: None,
            half_life_days: None,
            prompt_guard: None,
            collective: None,
            skills_prompt: None,
            hooks: None,
            memory_manager: None,
            home_dir: None,
            auxiliary_client: None,
            callbacks: None,
            interrupt_flag: None,
        }
    }

    /// Wire the resolved Fennec home directory.
    pub fn home_dir(mut self, path: std::path::PathBuf) -> Self {
        self.home_dir = Some(path);
        self
    }

    /// Wire in a plugin lifecycle [`HookRegistry`].
    pub fn hooks(mut self, hooks: Arc<crate::plugins::HookRegistry>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Wire in the plugin [`MemoryManager`].
    pub fn memory_manager(
        mut self,
        manager: Arc<crate::plugins::MemoryManager>,
    ) -> Self {
        self.memory_manager = Some(manager);
        self
    }

    /// Wire in an auxiliary client. If not set, `Agent::build`
    /// substitutes an empty client so consumers (curator, plugin
    /// background tasks) can always call `auxiliary_client()` and
    /// check `is_available()` without a None-check.
    pub fn auxiliary_client(
        mut self,
        client: Arc<crate::providers::AuxiliaryClient>,
    ) -> Self {
        self.auxiliary_client = Some(client);
        self
    }

    /// Wire a cooperative-interrupt flag. The agent loop polls it
    /// at each tool-iteration boundary and bails when set. Used
    /// by sub-agents the user kills via the spawn-tree overlay.
    pub fn interrupt_flag(
        mut self,
        flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        self.interrupt_flag = Some(flag);
        self
    }

    pub fn provider(mut self, provider: impl Into<Arc<dyn Provider>>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Add a single tool. Accepts the legacy `Box<dyn Tool>` shape
    /// to preserve every existing call site (every `Box::new(SomeTool::new(...))`
    /// across main.rs, channel constructors, tests, etc. continues
    /// to work unchanged). Internally the box is converted to an
    /// `Arc` so the tool can later be shared across multiple Agent
    /// instances.
    pub fn tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(Arc::from(tool));
        self
    }

    /// Replace the tool list. Same compatibility shim as
    /// [`Self::tool`] — accepts `Box<dyn Tool>` for back-compat.
    pub fn tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools = tools.into_iter().map(Arc::from).collect();
        self
    }

    /// Add a single Arc-shared tool. Used by callers that already
    /// hold an `Arc<dyn Tool>` (the OpenAI-compat per-request agent
    /// factory in particular) and want to skip the `Box -> Arc`
    /// indirection.
    pub fn tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Replace the tool list with pre-shared Arc'd tools.
    pub fn tools_arc(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn identity_name(mut self, name: impl Into<String>) -> Self {
        self.identity_name = Some(name.into());
        self
    }

    pub fn identity_persona(mut self, persona: impl Into<String>) -> Self {
        self.identity_persona = Some(persona.into());
        self
    }

    pub fn max_tool_iterations(mut self, n: usize) -> Self {
        self.max_tool_iterations = Some(n);
        self
    }

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn memory_context_limit(mut self, n: usize) -> Self {
        self.memory_context_limit = Some(n);
        self
    }

    pub fn half_life_days(mut self, d: f64) -> Self {
        self.half_life_days = Some(d);
        self
    }

    pub fn prompt_guard(mut self, guard: PromptGuard) -> Self {
        self.prompt_guard = Some(guard);
        self
    }

    pub fn collective(mut self, search: Arc<CollectiveSearch>) -> Self {
        self.collective = Some(search);
        self
    }

    /// Set the pre-rendered skills prompt fragment. Typically the output of
    /// [`crate::skills::SkillsLoader::build_skills_prompt`]. Empty string (or
    /// never calling this) means no skills are injected.
    pub fn skills_prompt(mut self, prompt: String) -> Self {
        self.skills_prompt = Some(prompt);
        self
    }

    /// Register a frontend callback handle. Used by the TUI and
    /// (later) the dashboard to receive turn / tool / status
    /// events as they happen. Callers that don't set this get a
    /// no-op handle; existing code paths (channel sends, batch
    /// runs) behave exactly as before.
    pub fn callbacks(mut self, handle: super::callbacks::CallbacksHandle) -> Self {
        self.callbacks = Some(handle);
        self
    }

    /// Build the [`Agent`], validating that required fields are set.
    pub fn build(self) -> Result<Agent> {
        let provider = self
            .provider
            .ok_or_else(|| anyhow::anyhow!("AgentBuilder: provider is required"))?;
        let memory = self
            .memory
            .ok_or_else(|| anyhow::anyhow!("AgentBuilder: memory is required"))?;

        let name = self.identity_name.unwrap_or_else(|| "Fennec".to_string());
        let persona = self.identity_persona.unwrap_or_else(|| {
            "A fast, helpful AI assistant with collective intelligence.".to_string()
        });
        let prompt_builder = SystemPromptBuilder::new(name, persona);

        // Resolve the memory manager once (default = empty) so we
        // can merge its tool schemas into the LLM-visible tool list
        // AND store it on the Agent without rebuilding.
        let memory_manager = self
            .memory_manager
            .unwrap_or_else(|| Arc::new(crate::plugins::MemoryManager::empty()));

        // Built-in tools first, then any tool schemas contributed
        // by the active memory provider. When no provider is wired,
        // `tool_schemas()` returns an empty Vec and the merge is a
        // no-op (default path stays byte-identical).
        let mut tool_specs: Vec<ToolSpec> = self.tools.iter().map(|t| t.spec()).collect();
        tool_specs.extend(memory_manager.tool_schemas());

        Ok(Agent {
            provider,
            tools: self.tools,
            tool_specs,
            disabled_tools: HashSet::new(),
            pending_attachments: Vec::new(),
            pending_steer: None,
            memory,
            prompt_builder,
            max_tool_iterations: self.max_tool_iterations.unwrap_or(90),
            history: Vec::new(),
            system_prompt: None,
            max_tokens: self.max_tokens.unwrap_or(8192),
            temperature: self.temperature.unwrap_or(0.7),
            memory_context_limit: self.memory_context_limit.unwrap_or(5),
            half_life_days: self.half_life_days.unwrap_or(7.0),
            prompt_guard: self.prompt_guard,
            collective: self.collective,
            skills_prompt: self.skills_prompt.unwrap_or_default(),
            thinking_level: ThinkingLevel::Off,
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_read_tokens: 0,
            total_cache_write_tokens: 0,
            total_api_calls: 0,
            last_prompt_tokens: 0,
            turn_count: 0,
            hooks: self
                .hooks
                .unwrap_or_else(|| Arc::new(crate::plugins::HookRegistry::new())),
            memory_manager,
            home_dir: self.home_dir.unwrap_or_else(|| {
                dirs::home_dir()
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
                    .join(".fennec")
            }),
            session_id: uuid::Uuid::new_v4().to_string(),
            session_started: false,
            auxiliary_client: self.auxiliary_client.unwrap_or_else(|| {
                Arc::new(crate::providers::AuxiliaryClient::new(
                    crate::providers::AuxiliaryConfig::default(),
                    Vec::new(),
                    Vec::new(),
                ))
            }),
            last_turn_iterations: 0,
            interrupt_flag: self.interrupt_flag,
            callbacks: self
                .callbacks
                .unwrap_or_else(super::callbacks::noop_callbacks),
        })
    }
}

// ---------------------------------------------------------------------------
// Callback helpers
// ---------------------------------------------------------------------------

/// One-line preview of a tool call's args for the inline display
/// and the TOOL LIVE panel header. We render JSON compactly and
/// truncate aggressively — frontends show the full args in their
/// own collapsed view.
fn preview_for_args(args: &serde_json::Value) -> String {
    let s = match args {
        serde_json::Value::Object(map) if map.is_empty() => return String::from("()"),
        serde_json::Value::Null => return String::from("()"),
        _ => args.to_string(),
    };
    truncate_summary(&s)
}

/// Truncate any text to a single short line for inline display.
fn truncate_summary(s: &str) -> String {
    let single_line: String = s.lines().next().unwrap_or("").to_string();
    const MAX: usize = 80;
    if single_line.chars().count() <= MAX {
        single_line
    } else {
        let mut out: String = single_line.chars().take(MAX - 1).collect();
        out.push('…');
        out
    }
}
