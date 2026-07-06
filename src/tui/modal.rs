//! Modal overlays for human-in-the-loop prompts.
//!
//! Six modal types matching the upstream's `prompts.tsx`:
//!
//! 1. **Approval** — agent wants to run a privileged command;
//!    user picks `Once` / `Session` / `Always` / `Deny`.
//! 2. **Clarify** — agent needs the user to answer a question;
//!    optionally with multiple-choice presets.
//! 3. **Confirm** — local destructive action (e.g. `/clear`)
//!    asks for `Yes` / `No`. Triggered by slash commands, not
//!    agent events.
//! 4. **Secret** — agent needs an API key / token; masked input.
//! 5. **Sudo** — sudo password prompt; masked input.
//! 6. **Pager** — full-screen scrollable text viewer for `/logs`,
//!    long `/help` output, etc. Triggered locally.
//!
//! Resolution is driven by a `tokio::sync::oneshot` per modal:
//! the TUI bridge's blocking callback awaits the receiver, the
//! keyboard handler sends the user's choice through the sender
//! when the modal is dismissed. Once resolved the modal is
//! cleared from `App.modal`.
//!
//! Upstream anchors: `ui-tui/src/components/prompts.tsx:14-217`,
//! `ui-tui/src/app/overlayStore.ts`, `ui-tui/src/app/useInputHandlers.ts:70-108`
//! (cancelOverlayFromCtrlC).

use tokio::sync::oneshot;

use crate::agent::callbacks::{ApprovalRequest, ClarifyRequest, SecretRequest};

/// A privilege-approval choice. Maps onto the upstream's four quick-pick
/// options. `Once` and `Deny` are per-request; `Session` and
/// `Always` are intended to be remembered (session-scoped or
/// config-persisted respectively) — the policy engine that
/// decides what to remember lives outside this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalChoice {
    Once,
    Session,
    Always,
    Deny,
}

impl ApprovalChoice {
    /// Whether this choice grants the agent permission to proceed.
    /// `Deny` is the only `false` case.
    pub fn is_allow(self) -> bool {
        !matches!(self, ApprovalChoice::Deny)
    }

    /// Numeric quick-pick (1-indexed) matching the modal's
    /// rendered order — the upstream uses 1=Once, 2=Session, 3=Always,
    /// 4=Deny.
    pub fn from_quick_pick(n: u8) -> Option<Self> {
        match n {
            1 => Some(ApprovalChoice::Once),
            2 => Some(ApprovalChoice::Session),
            3 => Some(ApprovalChoice::Always),
            4 => Some(ApprovalChoice::Deny),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ApprovalChoice::Once => "Allow once",
            ApprovalChoice::Session => "Allow this session",
            ApprovalChoice::Always => "Always allow",
            ApprovalChoice::Deny => "Deny",
        }
    }
}

/// Confirmation outcome for `/clear`-style local prompts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmChoice {
    Yes,
    No,
}

/// State carried by an open modal. Each variant owns the oneshot
/// `Sender` for resolution; sending through it unblocks the
/// `AgentCallbacks` impl that fired the request.
///
/// `Confirm` and `Pager` are local-only (no agent callback); they
/// carry an `FnOnce` action instead of a oneshot. The action runs
/// when the user confirms.
pub enum Modal {
    Approval {
        request: ApprovalRequest,
        /// 0..=3 highlight index for ↑/↓ navigation. The
        /// keyboard handler maps this back to `ApprovalChoice`
        /// via `quick_pick_for_index`.
        cursor: usize,
        resp_tx: oneshot::Sender<ApprovalChoice>,
    },
    Clarify {
        request: ClarifyRequest,
        /// Index into `request.options + 1` (the +1 sentinel is
        /// the "Other (type your answer)" entry that flips the
        /// modal into free-text mode). `None` while the user is
        /// already in free-text mode.
        cursor: Option<usize>,
        /// Free-text buffer when the user picked "Other" or
        /// when `request.options` is empty.
        text: String,
        /// Cursor column inside `text` (char-index, like
        /// `InputState.col`).
        text_col: usize,
        resp_tx: oneshot::Sender<Option<String>>,
    },
    Confirm {
        title: String,
        detail: Option<String>,
        danger: bool,
        cursor: ConfirmChoice,
        /// Action to run on `Yes`. Wrapped in `Option` so we can
        /// `.take()` it in the resolve handler since `FnOnce`
        /// can't be moved out of `&mut self` directly.
        on_confirm: Option<Box<dyn FnOnce() + Send>>,
    },
    Secret {
        request: SecretRequest,
        text: String,
        text_col: usize,
        resp_tx: oneshot::Sender<Option<String>>,
    },
    Sudo {
        prompt: String,
        text: String,
        text_col: usize,
        resp_tx: oneshot::Sender<Option<String>>,
    },
    Pager {
        title: Option<String>,
        lines: Vec<String>,
        offset: usize,
    },
}

impl std::fmt::Debug for Modal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Modal::Approval { request, cursor, .. } => f
                .debug_struct("Modal::Approval")
                .field("request", request)
                .field("cursor", cursor)
                .finish_non_exhaustive(),
            Modal::Clarify {
                request,
                cursor,
                text,
                ..
            } => f
                .debug_struct("Modal::Clarify")
                .field("request", request)
                .field("cursor", cursor)
                .field("text_len", &text.len())
                .finish_non_exhaustive(),
            Modal::Confirm {
                title,
                detail,
                danger,
                cursor,
                ..
            } => f
                .debug_struct("Modal::Confirm")
                .field("title", title)
                .field("detail", detail)
                .field("danger", danger)
                .field("cursor", cursor)
                .finish_non_exhaustive(),
            Modal::Secret { request, .. } => f
                .debug_struct("Modal::Secret")
                .field("request", request)
                .finish_non_exhaustive(),
            Modal::Sudo { prompt, .. } => f
                .debug_struct("Modal::Sudo")
                .field("prompt", prompt)
                .finish_non_exhaustive(),
            Modal::Pager {
                title,
                lines,
                offset,
            } => f
                .debug_struct("Modal::Pager")
                .field("title", title)
                .field("line_count", &lines.len())
                .field("offset", offset)
                .finish(),
        }
    }
}

impl Modal {
    /// Map an `Approval` cursor index (0..=3) to its choice.
    pub fn approval_choice_at(index: usize) -> Option<ApprovalChoice> {
        ApprovalChoice::from_quick_pick((index + 1) as u8)
    }

    /// Number of clarify options including the "Other" sentinel
    /// row. Used by the keyboard handler to clamp ↑/↓ navigation.
    pub fn clarify_total_rows(request: &ClarifyRequest) -> usize {
        request.options.len() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approval_quick_pick_maps_to_choice() {
        assert_eq!(
            ApprovalChoice::from_quick_pick(1),
            Some(ApprovalChoice::Once)
        );
        assert_eq!(
            ApprovalChoice::from_quick_pick(4),
            Some(ApprovalChoice::Deny)
        );
        assert_eq!(ApprovalChoice::from_quick_pick(0), None);
        assert_eq!(ApprovalChoice::from_quick_pick(5), None);
    }

    #[test]
    fn approval_choice_is_allow_only_false_for_deny() {
        assert!(ApprovalChoice::Once.is_allow());
        assert!(ApprovalChoice::Session.is_allow());
        assert!(ApprovalChoice::Always.is_allow());
        assert!(!ApprovalChoice::Deny.is_allow());
    }

    #[test]
    fn approval_cursor_index_maps_back_to_choice() {
        assert_eq!(Modal::approval_choice_at(0), Some(ApprovalChoice::Once));
        assert_eq!(Modal::approval_choice_at(3), Some(ApprovalChoice::Deny));
        assert_eq!(Modal::approval_choice_at(99), None);
    }

    #[test]
    fn clarify_total_rows_includes_other_sentinel() {
        let req = ClarifyRequest {
            question: "favourite colour?".into(),
            options: vec!["red".into(), "blue".into()],
        };
        assert_eq!(Modal::clarify_total_rows(&req), 3);
        let empty = ClarifyRequest {
            question: "what?".into(),
            options: vec![],
        };
        assert_eq!(Modal::clarify_total_rows(&empty), 1);
    }
}
