//! Bridge from `AgentCallbacks` to the TUI's app state.
//!
//! The agent loop runs on its own task; the TUI renderer reads
//! from `App` on the main thread. To bridge them, the agent's
//! callbacks push events into a channel; the TUI's tick handler
//! drains the channel and applies the events to `App`.
//!
//! Most events are fire-and-forget (text deltas, tool start /
//! complete, status). The blocking-prompt events
//! (`on_approval_request`, `on_clarify_request`,
//! `on_secret_request`) need bidirectional flow: the agent
//! awaits a oneshot receiver while the TUI's keyboard handler
//! resolves the modal and sends the user's choice through the
//! corresponding sender. Implemented via `tokio::sync::oneshot`
//! per request.

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};

use crate::agent::callbacks::{
    AgentCallbacks, ApprovalRequest, ClarifyRequest, SecretRequest, SubagentComplete,
    SubagentSpawn, ToolComplete, ToolProgress, ToolStart,
};

use super::app::App;
use super::modal::{ApprovalChoice, Modal};

/// Events emitted by the agent that the TUI's event loop drains
/// into `App` mutations.
///
/// `Approval` / `Clarify` / `Secret` carry the request payload
/// plus a oneshot `Sender` for resolution. The drain task
/// installs a `Modal` on `App` carrying that sender; when the
/// user resolves the modal in the keyboard handler, the choice
/// flows back through the oneshot to the awaiting agent task.
#[derive(Debug)]
pub enum TuiEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolStart(ToolStart),
    ToolProgress(ToolProgress),
    ToolComplete(ToolComplete),
    Status(String),
    TurnStart(String),
    TurnComplete(String),
    /// Open an approval modal. Drain task moves the request +
    /// sender into `App.modal`; keyboard handler resolves.
    ApprovalRequest {
        request: ApprovalRequest,
        resp_tx: oneshot::Sender<ApprovalChoice>,
    },
    ClarifyRequest {
        request: ClarifyRequest,
        resp_tx: oneshot::Sender<Option<String>>,
    },
    SecretRequest {
        request: SecretRequest,
        resp_tx: oneshot::Sender<Option<String>>,
    },
    /// New sub-agent spawned via the `delegate` tool. Carries the
    /// id, optional parent_id (for nested spawns), and the goal
    /// the parent passed.
    SubagentSpawn(SubagentSpawn),
    /// Sub-agent's main loop began — separate from spawn so the
    /// overlay can show the queued → running transition.
    SubagentStart(String),
    /// Streaming text delta from a sub-agent.
    SubagentText { id: String, delta: String },
    /// Streaming reasoning delta from a sub-agent.
    SubagentThinking { id: String, delta: String },
    /// A tool started inside a sub-agent.
    SubagentTool { id: String, start: ToolStart },
    /// Free-text progress note from a sub-agent.
    SubagentProgress { id: String, note: String },
    /// Sub-agent finished — final event for that branch.
    SubagentComplete(SubagentComplete),
}

/// Sender side of the bridge — held by the `AgentCallbacks` impl.
pub type EventSender = mpsc::UnboundedSender<TuiEvent>;
/// Receiver side — owned by the TUI's drain loop.
pub type EventReceiver = mpsc::UnboundedReceiver<TuiEvent>;

/// `AgentCallbacks` implementation that forwards every event onto
/// a channel. Fire-and-forget for text deltas / tool events;
/// blocking events `await` the user's modal resolution via
/// oneshot.
pub struct TuiBridge {
    tx: EventSender,
    /// Held so the bridge can opt to mutate app state directly
    /// for events that don't need to round-trip the channel
    /// (e.g. status updates that should land instantly).
    pub app: Arc<Mutex<App>>,
}

impl TuiBridge {
    pub fn new(app: Arc<Mutex<App>>) -> (Self, EventReceiver) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx, app }, rx)
    }
}

#[async_trait]
impl AgentCallbacks for TuiBridge {
    fn on_text_delta(&self, delta: &str) {
        let _ = self.tx.send(TuiEvent::TextDelta(delta.to_string()));
    }

    fn on_reasoning_delta(&self, delta: &str) {
        let _ = self.tx.send(TuiEvent::ReasoningDelta(delta.to_string()));
    }

    fn on_tool_start(&self, start: ToolStart) {
        let _ = self.tx.send(TuiEvent::ToolStart(start));
    }

    fn on_tool_progress(&self, progress: ToolProgress) {
        let _ = self.tx.send(TuiEvent::ToolProgress(progress));
    }

    fn on_tool_complete(&self, complete: ToolComplete) {
        let _ = self.tx.send(TuiEvent::ToolComplete(complete));
    }

    fn on_status(&self, message: &str) {
        let _ = self.tx.send(TuiEvent::Status(message.to_string()));
    }

    fn on_turn_start(&self, prompt: &str) {
        let _ = self.tx.send(TuiEvent::TurnStart(prompt.to_string()));
    }

    fn on_turn_complete(&self, summary: &str) {
        let _ = self.tx.send(TuiEvent::TurnComplete(summary.to_string()));
    }

    async fn on_approval_request(&self, request: ApprovalRequest) -> bool {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self
            .tx
            .send(TuiEvent::ApprovalRequest { request, resp_tx })
            .is_err()
        {
            // TUI shut down — fail closed.
            return false;
        }
        match resp_rx.await {
            Ok(choice) => choice.is_allow(),
            // Modal was dropped without resolution (e.g. TUI
            // exited while the agent was waiting). Treat as deny.
            Err(_) => false,
        }
    }

    async fn on_clarify_request(&self, request: ClarifyRequest) -> Option<String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self
            .tx
            .send(TuiEvent::ClarifyRequest { request, resp_tx })
            .is_err()
        {
            return None;
        }
        resp_rx.await.unwrap_or(None)
    }

    async fn on_secret_request(&self, request: SecretRequest) -> Option<String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        if self
            .tx
            .send(TuiEvent::SecretRequest { request, resp_tx })
            .is_err()
        {
            return None;
        }
        resp_rx.await.unwrap_or(None)
    }

    fn on_subagent_spawn(&self, spawn: SubagentSpawn) {
        let _ = self.tx.send(TuiEvent::SubagentSpawn(spawn));
    }

    fn on_subagent_start(&self, id: &str) {
        let _ = self.tx.send(TuiEvent::SubagentStart(id.to_string()));
    }

    fn on_subagent_text(&self, id: &str, delta: &str) {
        let _ = self.tx.send(TuiEvent::SubagentText {
            id: id.to_string(),
            delta: delta.to_string(),
        });
    }

    fn on_subagent_thinking(&self, id: &str, delta: &str) {
        let _ = self.tx.send(TuiEvent::SubagentThinking {
            id: id.to_string(),
            delta: delta.to_string(),
        });
    }

    fn on_subagent_tool(&self, id: &str, start: ToolStart) {
        let _ = self.tx.send(TuiEvent::SubagentTool {
            id: id.to_string(),
            start,
        });
    }

    fn on_subagent_progress(&self, id: &str, note: &str) {
        let _ = self.tx.send(TuiEvent::SubagentProgress {
            id: id.to_string(),
            note: note.to_string(),
        });
    }

    fn on_subagent_complete(&self, complete: SubagentComplete) {
        let _ = self.tx.send(TuiEvent::SubagentComplete(complete));
    }
}

/// Apply an `ApprovalRequest` event to the app's modal slot.
/// Splits out from the drain task so it's directly testable.
/// If a modal is already open, the new request replaces it (the
/// previous request's oneshot is dropped, which makes the
/// awaiting agent task receive an `Err` and treat the prompt as
/// denied — matches the upstream's last-write-wins semantics, which we
/// document as a deliberate gap rather than a guarantee).
pub fn install_approval_modal(
    app: &mut App,
    request: ApprovalRequest,
    resp_tx: oneshot::Sender<ApprovalChoice>,
) {
    app.modal = Some(Modal::Approval {
        request,
        cursor: 0,
        resp_tx,
    });
}

pub fn install_clarify_modal(
    app: &mut App,
    request: ClarifyRequest,
    resp_tx: oneshot::Sender<Option<String>>,
) {
    let cursor = if request.options.is_empty() {
        None
    } else {
        Some(0)
    };
    app.modal = Some(Modal::Clarify {
        request,
        cursor,
        text: String::new(),
        text_col: 0,
        resp_tx,
    });
}

pub fn install_secret_modal(
    app: &mut App,
    request: SecretRequest,
    resp_tx: oneshot::Sender<Option<String>>,
) {
    app.modal = Some(Modal::Secret {
        request,
        text: String::new(),
        text_col: 0,
        resp_tx,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_app() -> Arc<Mutex<App>> {
        Arc::new(Mutex::new(App::new()))
    }

    #[tokio::test]
    async fn approval_modal_round_trip_resolves_callback() {
        let app = fresh_app();
        let (bridge, mut rx) = TuiBridge::new(app.clone());

        // Spawn the bridge call in the background; it will await
        // the oneshot until the modal is resolved.
        let pending = tokio::spawn(async move {
            bridge
                .on_approval_request(ApprovalRequest {
                    command: "rm -rf /".into(),
                    description: "scary".into(),
                })
                .await
        });

        // Drain the request from the bridge channel.
        let evt = rx.recv().await.unwrap();
        let TuiEvent::ApprovalRequest { request, resp_tx } = evt else {
            panic!("expected ApprovalRequest, got {evt:?}");
        };
        // Install the modal as the drain task would.
        install_approval_modal(&mut app.lock(), request, resp_tx);
        assert!(app.lock().is_blocked());

        // Resolve as if the user pressed '3' (Always).
        app.lock()
            .handle_key(crossterm::event::KeyCode::Char('3'), crossterm::event::KeyModifiers::NONE);
        assert!(!app.lock().is_blocked());

        let allowed = pending.await.unwrap();
        assert!(allowed, "Always should be an allow");
    }

    #[tokio::test]
    async fn approval_ctrl_c_resolves_with_deny() {
        let app = fresh_app();
        let (bridge, mut rx) = TuiBridge::new(app.clone());
        let pending = tokio::spawn(async move {
            bridge
                .on_approval_request(ApprovalRequest {
                    command: "do bad thing".into(),
                    description: "".into(),
                })
                .await
        });
        let TuiEvent::ApprovalRequest { request, resp_tx } = rx.recv().await.unwrap() else {
            panic!()
        };
        install_approval_modal(&mut app.lock(), request, resp_tx);
        // Ctrl-C → deny.
        app.lock().handle_key(
            crossterm::event::KeyCode::Char('c'),
            crossterm::event::KeyModifiers::CONTROL,
        );
        assert!(!app.lock().is_blocked());
        let allowed = pending.await.unwrap();
        assert!(!allowed);
    }

    #[tokio::test]
    async fn clarify_modal_returns_picked_option() {
        let app = fresh_app();
        let (bridge, mut rx) = TuiBridge::new(app.clone());
        let pending = tokio::spawn(async move {
            bridge
                .on_clarify_request(ClarifyRequest {
                    question: "go ahead?".into(),
                    options: vec!["yes".into(), "no".into()],
                })
                .await
        });
        let TuiEvent::ClarifyRequest { request, resp_tx } = rx.recv().await.unwrap() else {
            panic!()
        };
        install_clarify_modal(&mut app.lock(), request, resp_tx);
        // Press '1' → first option ("yes").
        app.lock().handle_key(
            crossterm::event::KeyCode::Char('1'),
            crossterm::event::KeyModifiers::NONE,
        );
        let answer = pending.await.unwrap();
        assert_eq!(answer.as_deref(), Some("yes"));
    }

    #[tokio::test]
    async fn secret_modal_returns_typed_text() {
        let app = fresh_app();
        let (bridge, mut rx) = TuiBridge::new(app.clone());
        let pending = tokio::spawn(async move {
            bridge
                .on_secret_request(SecretRequest {
                    label: "API key?".into(),
                })
                .await
        });
        let TuiEvent::SecretRequest { request, resp_tx } = rx.recv().await.unwrap() else {
            panic!()
        };
        install_secret_modal(&mut app.lock(), request, resp_tx);
        // Type "abc" then Enter.
        for c in ['a', 'b', 'c'] {
            app.lock().handle_key(
                crossterm::event::KeyCode::Char(c),
                crossterm::event::KeyModifiers::NONE,
            );
        }
        app.lock().handle_key(
            crossterm::event::KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        );
        let secret = pending.await.unwrap();
        assert_eq!(secret.as_deref(), Some("abc"));
    }
}
