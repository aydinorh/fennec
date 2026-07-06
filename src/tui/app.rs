//! TUI application state.
//!
//! Holds the data the renderer reads and the input handlers
//! mutate: sessions list + selection, chat scrollback, current
//! tool execution, channel statuses, input buffer, modal state,
//! ticker for animations.
//!
//! Real wiring to the agent / session store / channel manager
//! comes in a follow-up commit. This is the state shape and the
//! event-handling skeleton — enough for the renderer to draw
//! against and for keyboard navigation to feel right.

use std::collections::VecDeque;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyModifiers};

use super::spawn_tree::SpawnTree;

/// Walk a subtree depth-first appending ids in render order,
/// with children sorted by the active overlay sort mode. Keeps
/// the rendered row order stable when the cursor walks the flat
/// id list.
fn collect_subtree_ids_sorted(
    tree: &SpawnTree,
    id: &str,
    sort: AgentsSortMode,
    out: &mut Vec<String>,
) {
    out.push(id.to_string());
    if let Some(node) = tree.nodes.get(id) {
        let mut kids: Vec<&str> = node.children.iter().map(String::as_str).collect();
        kids.sort_by(|a, b| compare_nodes(tree, a, b, sort));
        for child in kids {
            collect_subtree_ids_sorted(tree, child, sort, out);
        }
    }
}

/// Comparator used to order sibling rows under the active
/// overlay sort mode. Mirrors the upstream's `SORT_COMPARATORS` table
/// (`agentsOverlay.tsx:67-72`).
fn compare_nodes(
    tree: &SpawnTree,
    a: &str,
    b: &str,
    sort: AgentsSortMode,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let na = tree.nodes.get(a);
    let nb = tree.nodes.get(b);
    match (na, nb) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(na), Some(nb)) => match sort {
            AgentsSortMode::DepthFirst => a.cmp(b),
            AgentsSortMode::ToolsDesc => {
                let am = tree.aggregate(a).subtree_tools;
                let bm = tree.aggregate(b).subtree_tools;
                bm.cmp(&am).then_with(|| a.cmp(b))
            }
            AgentsSortMode::DurationDesc => {
                let am = tree.aggregate(a).subtree_duration_ms;
                let bm = tree.aggregate(b).subtree_duration_ms;
                bm.cmp(&am).then_with(|| a.cmp(b))
            }
            AgentsSortMode::Status => {
                status_rank(na.status)
                    .cmp(&status_rank(nb.status))
                    .then_with(|| a.cmp(b))
            }
        },
    }
}

/// Ranks statuses for the "Status" sort mode. Matches the upstream's
/// `STATUS_RANK` (`agentsOverlay.tsx:59-65`) so failed agents
/// surface first, completed last.
fn status_rank(status: super::spawn_tree::SubagentStatus) -> u8 {
    use super::spawn_tree::SubagentStatus;
    match status {
        SubagentStatus::Failed => 0,
        SubagentStatus::Interrupted => 1,
        SubagentStatus::Running => 2,
        SubagentStatus::Queued => 3,
        SubagentStatus::Completed => 4,
    }
}

/// Whether a node passes the active filter. Mirrors the upstream's
/// `FILTER_PREDICATES` table. Note: `leaf` checks for zero
/// children in the *spawn tree*, not the registry, so an
/// archived snapshot with no recorded children counts as a leaf.
fn filter_matches(
    filter: AgentsFilterMode,
    node: &super::spawn_tree::SubagentNode,
    tree: &SpawnTree,
) -> bool {
    use super::spawn_tree::SubagentStatus;
    match filter {
        AgentsFilterMode::All => true,
        AgentsFilterMode::Running => matches!(
            node.status,
            SubagentStatus::Running | SubagentStatus::Queued
        ),
        AgentsFilterMode::Failed => matches!(
            node.status,
            SubagentStatus::Failed | SubagentStatus::Interrupted
        ),
        AgentsFilterMode::Leaf => {
            tree.nodes
                .get(&node.id)
                .map(|n| n.children.is_empty())
                .unwrap_or(true)
        }
    }
}

/// One row in the sessions panel.
#[derive(Debug, Clone)]
pub struct SessionRow {
    /// Two-character source code: `TG` (telegram), `SL` (slack),
    /// `DC` (discord), `SG` (signal), `MX` (matrix), `@ ` (email),
    /// `$ ` (cli).
    pub code: String,
    /// Display name (sender / channel / user).
    pub who: String,
    /// One-line subject preview.
    pub subject: String,
    /// Render-ready count text ("47" or "2/31" with unread).
    pub count: String,
    /// Whether this row counts as having unread activity (drives
    /// the amber-bold render style).
    pub has_unread: bool,
}

/// One displayable line in the chat panel.
#[derive(Debug, Clone)]
pub enum ChatLine {
    /// `sys` line — session info, model, context window.
    System { time: String, body: String },
    /// User-typed message.
    User { time: String, body: String },
    /// Assistant message (possibly streaming).
    Bot { time: String, body: String },
    /// Inline `▸ tool · name(args)` line.
    ToolCall { call: String },
    /// Inline `↳ {result}` line under a tool call.
    ToolResult { summary: String },
    /// Inline running-tool marker with a spinner the renderer
    /// animates on tick. Replaced with `ToolResult` once the
    /// tool completes.
    ToolRunning { label: String, started_at: Instant },
    /// Extended-thinking / reasoning block from the model
    /// (Anthropic `thinking` content blocks; OpenAI o1/o3
    /// `message.reasoning` field). Renders dim above the next
    /// bot reply. `is_streaming = true` while deltas are still
    /// arriving so the renderer shows a live cursor; flipped
    /// to false when the bot's text reply starts (or the turn
    /// ends without one).
    ReasoningBlock {
        body: String,
        started_at: Instant,
        is_streaming: bool,
    },
}

/// Currently-executing tool, surfaced in the right column's
/// TOOL LIVE panel. `None` between tool calls.
#[derive(Debug, Clone)]
pub struct LiveTool {
    pub name: String,
    pub args_preview: String,
    pub started_at: Instant,
    /// 0..=100 progress hint. Tools that don't report progress
    /// just leave this at the animated default (looping pulse
    /// the renderer overlays).
    pub progress: Option<u8>,
}

/// Recent tool history shown under the live tool indicator.
#[derive(Debug, Clone)]
pub struct ToolHistoryEntry {
    pub ok: bool,
    pub name: String,
    pub note: String,
}

/// One channel + its current connection state. Drives the right
/// column's CHANNELS panel.
#[derive(Debug, Clone)]
pub struct ChannelState {
    pub code: String,
    pub name: String,
    pub state: ChannelConnState,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelConnState {
    Connected,
    Polling,
    Attached,
    Idle,
    Disconnected,
    Error,
}

/// Pane focus — drives keyboard routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Sessions,
    Chat,
    Input,
}

/// Visibility mode for inline tool blocks (and, once F1-2 lands
/// reasoning rendering, thinking blocks). Settable via
/// `/details hidden|collapsed|expanded`. Default is `Expanded`,
/// matching the F1-1 renderer's existing behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetailsMode {
    /// Skip tool/reasoning blocks entirely.
    Hidden,
    /// Show only the header (`▸ tool · name`) without args /
    /// result body / spinner.
    Collapsed,
    /// Show everything (current F1-1 behavior).
    #[default]
    Expanded,
}

impl DetailsMode {
    /// Parse a string the user typed at the command line.
    /// Returns `None` for unknown values so callers can surface
    /// "unknown mode" error rather than silently defaulting.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "hidden" | "off" | "h" => Some(DetailsMode::Hidden),
            "collapsed" | "c" => Some(DetailsMode::Collapsed),
            "expanded" | "on" | "full" | "e" => Some(DetailsMode::Expanded),
            _ => None,
        }
    }

    /// Inverse of `parse` — used for config.toml persistence
    /// and `/details` (no arg) read-back.
    pub fn as_str(&self) -> &'static str {
        match self {
            DetailsMode::Hidden => "hidden",
            DetailsMode::Collapsed => "collapsed",
            DetailsMode::Expanded => "expanded",
        }
    }
}

/// Per-input editor state — multi-line buffer with cursor,
/// undo/redo stack, and history navigation.
///
/// Cursor coordinates are `(row, col)` in **char** units (not
/// bytes). All editing operations preserve this invariant. The
/// undo/redo stack records snapshots before each mutation; cap
/// is 200 to keep memory bounded.
#[derive(Debug, Default, Clone)]
pub struct InputState {
    pub lines: Vec<String>,
    pub row: usize,
    pub col: usize,
    /// Recently submitted messages. ↑↓ at the top/bottom edges
    /// of the buffer cycle through these.
    pub history: VecDeque<String>,
    /// Cursor index into `history`. `None` = editing live buffer;
    /// `Some(i)` = previewing `history[i]`.
    pub history_cursor: Option<usize>,
    /// Snapshot stack for Ctrl-Z. Each entry is the full
    /// `(lines, row, col)` triple captured before a mutation.
    undo_stack: Vec<EditSnapshot>,
    /// Snapshots popped by Ctrl-Z and re-applicable via Ctrl-Y.
    /// Cleared on any new mutation.
    redo_stack: Vec<EditSnapshot>,
    /// Saved live-buffer when the user starts cycling history;
    /// restored when they cycle back past the most-recent entry.
    saved_live_buffer: Option<EditSnapshot>,
}

#[derive(Debug, Clone)]
struct EditSnapshot {
    lines: Vec<String>,
    row: usize,
    col: usize,
}

const UNDO_STACK_CAP: usize = 200;
const HISTORY_CAP: usize = 64;

impl InputState {
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            row: 0,
            col: 0,
            history: VecDeque::with_capacity(HISTORY_CAP),
            history_cursor: None,
            undo_stack: Vec::with_capacity(32),
            redo_stack: Vec::new(),
            saved_live_buffer: None,
        }
    }

    /// Current buffer rendered as a single string with `\n`
    /// separators.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Replace the buffer with `s` and place the cursor at the
    /// end. Records an undo snapshot.
    pub fn set(&mut self, s: &str) {
        self.snapshot_for_undo();
        self.lines = if s.is_empty() {
            vec![String::new()]
        } else {
            s.split('\n').map(String::from).collect()
        };
        self.row = self.lines.len().saturating_sub(1);
        self.col = self.lines[self.row].chars().count();
    }

    pub fn clear(&mut self) {
        self.snapshot_for_undo();
        self.lines = vec![String::new()];
        self.row = 0;
        self.col = 0;
        self.history_cursor = None;
        self.saved_live_buffer = None;
    }

    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(|l| l.is_empty())
    }

    pub fn push_history(&mut self, s: String) {
        if s.is_empty() {
            return;
        }
        if self.history.front().map(|h| h == &s).unwrap_or(false) {
            return;
        }
        self.history.push_front(s);
        while self.history.len() > HISTORY_CAP {
            self.history.pop_back();
        }
    }

    // -- Editor operations ---------------------------------------

    /// Insert one char at the cursor. Records an undo snapshot.
    pub fn insert_char(&mut self, c: char) {
        self.snapshot_for_undo();
        let line = &mut self.lines[self.row];
        let byte_idx = char_index_to_byte(line, self.col);
        line.insert(byte_idx, c);
        self.col += 1;
    }

    /// Insert a newline (Shift+Enter on most terminals).
    pub fn insert_newline(&mut self) {
        self.snapshot_for_undo();
        let line = self.lines[self.row].clone();
        let byte_idx = char_index_to_byte(&line, self.col);
        let (left, right) = line.split_at(byte_idx);
        self.lines[self.row] = left.to_string();
        self.lines.insert(self.row + 1, right.to_string());
        self.row += 1;
        self.col = 0;
    }

    /// Backspace — delete one char to the left of the cursor.
    pub fn backspace(&mut self) {
        self.snapshot_for_undo();
        if self.col > 0 {
            let line = &mut self.lines[self.row];
            let prev = char_index_to_byte(line, self.col - 1);
            let cur = char_index_to_byte(line, self.col);
            line.replace_range(prev..cur, "");
            self.col -= 1;
        } else if self.row > 0 {
            // Join with previous line.
            let cur_line = self.lines.remove(self.row);
            self.row -= 1;
            self.col = self.lines[self.row].chars().count();
            self.lines[self.row].push_str(&cur_line);
        }
    }

    /// Delete the word backward (Ctrl-W).
    pub fn delete_word_backward(&mut self) {
        self.snapshot_for_undo();
        let line = &self.lines[self.row].clone();
        let mut new_col = self.col;
        let chars: Vec<char> = line.chars().collect();
        // Skip trailing whitespace before the cursor.
        while new_col > 0 && chars[new_col - 1].is_whitespace() {
            new_col -= 1;
        }
        // Skip the word.
        while new_col > 0 && !chars[new_col - 1].is_whitespace() {
            new_col -= 1;
        }
        let from = char_index_to_byte(line, new_col);
        let to = char_index_to_byte(line, self.col);
        self.lines[self.row].replace_range(from..to, "");
        self.col = new_col;
    }

    /// Delete from cursor to start of line (Ctrl-U).
    pub fn delete_to_line_start(&mut self) {
        self.snapshot_for_undo();
        let line = &self.lines[self.row].clone();
        let to = char_index_to_byte(line, self.col);
        self.lines[self.row].replace_range(..to, "");
        self.col = 0;
    }

    /// Delete from cursor to end of line (Ctrl-K).
    pub fn delete_to_line_end(&mut self) {
        self.snapshot_for_undo();
        let line = &self.lines[self.row].clone();
        let from = char_index_to_byte(line, self.col);
        self.lines[self.row].replace_range(from.., "");
    }

    pub fn move_left(&mut self) {
        if self.col > 0 {
            self.col -= 1;
        } else if self.row > 0 {
            self.row -= 1;
            self.col = self.lines[self.row].chars().count();
        }
    }

    pub fn move_right(&mut self) {
        let line_len = self.lines[self.row].chars().count();
        if self.col < line_len {
            self.col += 1;
        } else if self.row + 1 < self.lines.len() {
            self.row += 1;
            self.col = 0;
        }
    }

    pub fn move_word_left(&mut self) {
        let chars: Vec<char> = self.lines[self.row].chars().collect();
        if self.col == 0 {
            self.move_left();
            return;
        }
        let mut c = self.col;
        while c > 0 && chars[c - 1].is_whitespace() {
            c -= 1;
        }
        while c > 0 && !chars[c - 1].is_whitespace() {
            c -= 1;
        }
        self.col = c;
    }

    pub fn move_word_right(&mut self) {
        let chars: Vec<char> = self.lines[self.row].chars().collect();
        let len = chars.len();
        if self.col >= len {
            self.move_right();
            return;
        }
        let mut c = self.col;
        while c < len && !chars[c].is_whitespace() {
            c += 1;
        }
        while c < len && chars[c].is_whitespace() {
            c += 1;
        }
        self.col = c;
    }

    pub fn move_line_start(&mut self) {
        self.col = 0;
    }

    pub fn move_line_end(&mut self) {
        self.col = self.lines[self.row].chars().count();
    }

    pub fn move_up(&mut self) {
        if self.row > 0 {
            self.row -= 1;
            let len = self.lines[self.row].chars().count();
            self.col = self.col.min(len);
        }
    }

    pub fn move_down(&mut self) {
        if self.row + 1 < self.lines.len() {
            self.row += 1;
            let len = self.lines[self.row].chars().count();
            self.col = self.col.min(len);
        }
    }

    /// Whether the cursor is on the topmost row — for ↑ to flip
    /// into history navigation.
    pub fn at_first_row(&self) -> bool {
        self.row == 0
    }

    /// Whether the cursor is on the bottom row.
    pub fn at_last_row(&self) -> bool {
        self.row + 1 == self.lines.len()
    }

    // -- History navigation --------------------------------------

    /// Cycle one step backward in history (older). Saves the live
    /// buffer on first invocation so cycling forward past the
    /// most-recent entry restores it.
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.history_cursor.is_none() {
            // Save the live buffer before overwriting it.
            self.saved_live_buffer = Some(self.snapshot());
        }
        let next = match self.history_cursor {
            None => 0,
            Some(i) => (i + 1).min(self.history.len() - 1),
        };
        self.history_cursor = Some(next);
        let snapshot = self.history[next].clone();
        self.lines = if snapshot.is_empty() {
            vec![String::new()]
        } else {
            snapshot.split('\n').map(String::from).collect()
        };
        self.row = self.lines.len().saturating_sub(1);
        self.col = self.lines[self.row].chars().count();
    }

    /// Cycle one step forward in history (newer). When already at
    /// the most-recent entry, restores the saved live buffer.
    pub fn history_next(&mut self) {
        match self.history_cursor {
            None => {}
            Some(0) => {
                self.history_cursor = None;
                if let Some(saved) = self.saved_live_buffer.take() {
                    self.restore(saved);
                }
            }
            Some(i) => {
                self.history_cursor = Some(i - 1);
                let snapshot = self.history[i - 1].clone();
                self.lines = if snapshot.is_empty() {
                    vec![String::new()]
                } else {
                    snapshot.split('\n').map(String::from).collect()
                };
                self.row = self.lines.len().saturating_sub(1);
                self.col = self.lines[self.row].chars().count();
            }
        }
    }

    // -- Undo / redo --------------------------------------------

    pub fn undo(&mut self) {
        if let Some(snapshot) = self.undo_stack.pop() {
            self.redo_stack.push(self.snapshot());
            self.restore(snapshot);
        }
    }

    pub fn redo(&mut self) {
        if let Some(snapshot) = self.redo_stack.pop() {
            self.undo_stack.push(self.snapshot());
            self.restore(snapshot);
        }
    }

    fn snapshot(&self) -> EditSnapshot {
        EditSnapshot {
            lines: self.lines.clone(),
            row: self.row,
            col: self.col,
        }
    }

    fn restore(&mut self, snapshot: EditSnapshot) {
        self.lines = snapshot.lines;
        self.row = snapshot.row;
        self.col = snapshot.col;
    }

    fn snapshot_for_undo(&mut self) {
        self.redo_stack.clear();
        self.undo_stack.push(self.snapshot());
        if self.undo_stack.len() > UNDO_STACK_CAP {
            self.undo_stack.remove(0);
        }
    }
}

/// Translate a char index to a byte index within `s`. Cursor
/// positions are stored as char indices because Unicode-aware
/// editing is the default on modern terminals (emoji, CJK,
/// combining marks).
fn char_index_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

/// Overall TUI app state.
pub struct App {
    pub focus: Focus,
    pub sessions: Vec<SessionRow>,
    pub selected_session: usize,
    pub chat: Vec<ChatLine>,
    /// Scroll offset from the bottom of the chat. 0 = pinned to
    /// latest; positive = scrolled up.
    pub chat_scroll: u16,
    pub live_tool: Option<LiveTool>,
    pub recent_tools: Vec<ToolHistoryEntry>,
    pub channels: Vec<ChannelState>,
    pub input: InputState,
    pub started_at: Instant,
    /// Whether the input is "ready" (cursor visible). Toggled by
    /// the tick handler so the cursor blinks.
    pub cursor_visible: bool,
    /// Brief status message shown at the bottom of the chat panel
    /// for ~3 seconds (e.g. "session resumed", "command not found").
    pub transient_status: Option<(String, Instant)>,
    /// Set to true when the event loop should exit.
    pub should_quit: bool,
    /// `/compact` toggle — hides metadata + blank lines in the
    /// chat panel for tighter rendering.
    pub compact_mode: bool,
    /// `/details` mode — controls visibility of inline tool
    /// details (call args, running spinner, result summary) and
    /// reasoning blocks when those land in F1-2. Three settings
    /// matching the upstream (`ui-tui/src/app/slash/commands/ops.ts`):
    ///   - Hidden: skip tool blocks entirely
    ///   - Collapsed: render only `▸ tool · name` (no args / no
    ///     result body)
    ///   - Expanded: full detail (current F1-1 behavior)
    pub details_mode: DetailsMode,
    /// Per-section overrides set via `/details <section> <mode>`.
    /// When a key (e.g. `"thinking"`) is present, the renderer
    /// uses that mode for the matching section instead of the
    /// global `details_mode`. Mirrors Hermes' `ui.sections`
    /// (`domain/details.ts:51-74`).
    pub details_section_overrides: std::collections::HashMap<String, DetailsMode>,
    /// `/mouse` toggle — drives crossterm's mouse-tracking
    /// enable/disable on the next frame. Off by default to avoid
    /// stealing the user's terminal-native scroll wheel.
    pub mouse_enabled: bool,
    /// Index into `chat` of the assistant message currently being
    /// streamed. `None` between turns; `Some(i)` while text deltas
    /// are arriving so they accumulate into one growing message
    /// instead of pushing a new line per delta.
    pub in_flight_bot_idx: Option<usize>,
    /// Index of the in-flight `ChatLine::ReasoningBlock` for
    /// streaming thinking deltas. Same pattern as
    /// `in_flight_bot_idx` — `None` between turns.
    pub in_flight_reasoning_idx: Option<usize>,
    /// Voice subsystem handle. The `/voice` command flips its
    /// state; the tick handler polls it for transcriptions /
    /// errors and updates the UI accordingly.
    pub voice: super::voice::VoiceController,
    /// Identifier of the SessionStore row backing the current
    /// chat. Set when `run_tui` provisions or resumes a session;
    /// `None` if the store init failed (the TUI still works,
    /// just without persistence). `/title` and `/resume` mutate
    /// this through the submit-loop's `AgentAction` plumbing.
    pub current_session_id: Option<String>,
    /// User-visible label of the current session. Mirrors the
    /// `title` column of `current_session_id`'s row, kept on
    /// `App` so the chat header can show it without re-querying
    /// the store every frame.
    pub current_session_title: Option<String>,
    /// Filesystem location of `~/.fennec/skills/` (or override).
    /// `/skills` reads this lazily via SkillsLoader so the user
    /// sees a fresh list every time without a config reload.
    pub skills_dir: Option<std::path::PathBuf>,
    /// Active modal overlay, if any.
    pub modal: Option<super::modal::Modal>,
    /// Set by `/edit` (or Ctrl-G) to the current input buffer text.
    pub pending_editor: Option<String>,
    /// Live spawn tree of delegated sub-agents. Updated as
    /// `TuiEvent::Subagent*` events flow in from the agent
    /// callback bridge. Read by the `/agents` overlay renderer.
    pub spawn_tree: super::spawn_tree::SpawnTree,
    /// FIFO history of completed spawn trees (cap 10). Pushed
    /// when the live tree is_settled — `/agents` then auto-pins
    /// the most recent snapshot when the live tree empties.
    pub spawn_history: super::spawn_tree::SpawnHistory,
    /// Whether the `/agents` fullscreen overlay is open.
    /// Toggled by the `/agents` slash command. When true, the
    /// keyboard router sends ↑↓/jk/g/G/Enter/q/Esc to the
    /// overlay handler instead of focus-dependent panes.
    pub show_agents_overlay: bool,
    /// Selected node id within the overlay's tree pane.
    pub agents_cursor: Option<String>,
    /// Active sort mode for the tree-pane row list.
    pub agents_sort: AgentsSortMode,
    /// Active filter mode for the tree-pane row list.
    pub agents_filter: AgentsFilterMode,
    /// 0 = live spawn tree; 1..N pulls the Nth-most-recent
    /// snapshot from `spawn_history`. Bumped by `[` `<` / `]` `>`.
    /// When the live tree clears mid-turn, `on_tick` auto-follows
    /// onto index 1 with a flash message — matches the upstream's
    /// "turn finished · inspect freely" pattern.
    pub agents_history_index: usize,
    /// Transient one-liner shown in the overlay footer.
    /// Cleared after ~2s by `on_tick`.
    pub agents_flash: Option<(String, Instant)>,
    /// When `true`, keyboard scrolls the detail pane instead of
    /// walking the tree list. Toggled by `Enter`/`l`/`→` (enter)
    /// and `Esc`/`h`/`←` (exit; falls back to closing the overlay
    /// when already on the list).
    pub agents_detail_focused: bool,
    /// Scroll offset (in rows) for the detail pane. Adjusted by
    /// PgUp/PgDn/Ctrl-D/Ctrl-U/j/k when the detail is focused.
    /// Reset when the cursor changes nodes.
    pub agents_detail_scroll: u16,
    /// Shared delegation registry — pause flag + caps + active
    /// map. Set when the TUI is running against a real agent
    /// (main.rs path). `None` in unit tests + smoke tests so
    /// `App::new()` stays cheap and dependency-free.
    pub delegation_registry: Option<crate::agent::DelegationRegistry>,
    /// Ring buffer of recent tracing log lines. Powers `/logs`.
    /// Empty in test mode (no tracing layer installed); the
    /// TUI bootstrap injects a live ring shared with the
    /// tracing subscriber.
    pub log_ring: super::log_ring::LogRing,
    /// Status-bar position. Driven by `/statusbar`. Persisted in
    /// `TuiConfig.statusbar`.
    pub statusbar_position: StatusBarPosition,
    /// Indicator (spinner) style. Driven by `/indicator`.
    /// Persisted in `TuiConfig.indicator`.
    pub indicator_style: IndicatorStyle,
    /// Tool-output verbosity. Driven by `/verbose`. Persisted in
    /// `TuiConfig.verbose`.
    pub verbosity: VerbosityMode,
    /// Behaviour when Enter is pressed mid-turn. Driven by
    /// `/busy`. Persisted in `TuiConfig.busy`.
    pub busy_mode: BusyMode,
    /// Active personality preset name. Driven by `/personality`.
    /// Persisted in `TuiConfig.personality`. Empty = use the
    /// IdentityConfig.persona as-loaded.
    pub personality_name: String,
    /// Messages enqueued via `/queue` to send as the next user
    /// turn (front of queue first). Drained by the submit loop.
    pub queued_input: std::collections::VecDeque<String>,
    /// Active skin name. Driven by `/skin`. Persisted in
    /// `TuiConfig.skin`. Empty = default fennec-warm palette.
    pub skin_name: String,
    /// Whether reasoning blocks are shown in the chat. Driven by
    /// `/reasoning hide|show`. Default `true` to preserve existing
    /// behaviour. Visual effect is rendered by F1-2-B's reasoning
    /// pane (this branch just stores + persists the flag).
    pub show_reasoning: bool,
    /// Indices `(a, b)` (1-based into `spawn_history`) of the
    /// snapshot pair to render in the agents overlay's diff view.
    /// `None` = normal single-tree mode. Set by `/replay-diff`.
    pub agents_diff_pair: Option<(usize, usize)>,
    /// Active skin (theme variant). Renderers read every colour
    /// from this struct rather than the `theme::*` constants so
    /// `/skin <name>` can swap palettes at runtime. Default is
    /// the fennec-warm palette (literally the same RGB values as
    /// the existing `theme::*` constants).
    pub skin: super::skin::Skin,
    /// Shared with the main `Agent` so `/busy interrupt` can
    /// cooperatively cancel the running turn. `None` outside TUI
    /// mode; populated by the bootstrap before the agent builds.
    pub main_interrupt_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// Status-bar position toggled by `/statusbar`. Mirrors the
/// upstream's `statusbar` config field with the same canonical
/// string forms; default is `Bottom` to preserve existing
/// behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StatusBarPosition {
    #[default]
    Bottom,
    Top,
    Off,
}

impl StatusBarPosition {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "bottom" | "on" => Some(Self::Bottom),
            "top" => Some(Self::Top),
            "off" | "hidden" => Some(Self::Off),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bottom => "bottom",
            Self::Top => "top",
            Self::Off => "off",
        }
    }
    /// 3-state cycle: Bottom → Top → Off → Bottom. Visits every
    /// position so `/statusbar toggle` from the default actually
    /// surfaces Top instead of jumping straight to Off.
    pub fn toggle(self) -> Self {
        match self {
            Self::Bottom => Self::Top,
            Self::Top => Self::Off,
            Self::Off => Self::Bottom,
        }
    }
}

/// Spinner-indicator style. `Braille` is the existing 10-frame
/// braille animation; the alternatives mirror the upstream
/// so config files round-trip between agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndicatorStyle {
    #[default]
    Braille,
    Ascii,
    Kaomoji,
    Emoji,
    Unicode,
}

impl IndicatorStyle {
    pub const ALL: [Self; 5] = [
        Self::Braille,
        Self::Ascii,
        Self::Kaomoji,
        Self::Emoji,
        Self::Unicode,
    ];
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "braille" | "" => Some(Self::Braille),
            "ascii" => Some(Self::Ascii),
            "kaomoji" => Some(Self::Kaomoji),
            "emoji" => Some(Self::Emoji),
            "unicode" => Some(Self::Unicode),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Braille => "braille",
            Self::Ascii => "ascii",
            Self::Kaomoji => "kaomoji",
            Self::Emoji => "emoji",
            Self::Unicode => "unicode",
        }
    }
    /// Pick the current frame based on elapsed ms since session
    /// start. 10 fps for braille, slower for face animations so
    /// they're readable.
    pub fn frame(&self, elapsed_ms: u128) -> &'static str {
        match self {
            Self::Braille => {
                const F: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                F[(elapsed_ms / 100) as usize % F.len()]
            }
            Self::Ascii => {
                const F: [&str; 4] = ["|", "/", "-", "\\"];
                F[(elapsed_ms / 150) as usize % F.len()]
            }
            Self::Kaomoji => {
                const F: [&str; 4] = ["(･ω･)", "(･ｪ･)", "(•́ω•̀)", "(•̀ω•́)"];
                F[(elapsed_ms / 350) as usize % F.len()]
            }
            Self::Emoji => {
                const F: [&str; 4] = ["🦊", "🌅", "✨", "🔥"];
                F[(elapsed_ms / 400) as usize % F.len()]
            }
            Self::Unicode => {
                const F: [&str; 6] = ["◐", "◓", "◑", "◒", "◴", "◵"];
                F[(elapsed_ms / 150) as usize % F.len()]
            }
        }
    }
}

/// Tool-output verbosity. `Normal` (default) shows the existing
/// collapsed/expanded view from `/details`; `Verbose` doesn't
/// truncate tool previews and shows full args.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VerbosityMode {
    #[default]
    Normal,
    Verbose,
}

impl VerbosityMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "off" | "normal" | "" => Some(Self::Normal),
            "on" | "verbose" => Some(Self::Verbose),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Verbose => "verbose",
        }
    }
    pub fn toggle(self) -> Self {
        match self {
            Self::Normal => Self::Verbose,
            Self::Verbose => Self::Normal,
        }
    }
}

/// Behaviour when the user presses Enter while the agent is busy.
/// `Interrupt` (default) cancels the running turn and starts a
/// new one; `Queue` parks the message until the turn finishes;
/// `Steer` routes it through the existing `/steer` injection
/// path so it lands as user-guidance after the next tool batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BusyMode {
    #[default]
    Interrupt,
    Queue,
    Steer,
}

impl BusyMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "interrupt" | "" => Some(Self::Interrupt),
            "queue" => Some(Self::Queue),
            "steer" => Some(Self::Steer),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Interrupt => "interrupt",
            Self::Queue => "queue",
            Self::Steer => "steer",
        }
    }
}

/// Sort modes for the `/agents` overlay tree-pane row list.
/// `s` cycles forward through these in render order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentsSortMode {
    DepthFirst,
    ToolsDesc,
    DurationDesc,
    Status,
}

impl AgentsSortMode {
    pub const ALL: [Self; 4] = [
        Self::DepthFirst,
        Self::ToolsDesc,
        Self::DurationDesc,
        Self::Status,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::DepthFirst => "spawn order",
            Self::ToolsDesc => "busiest",
            Self::DurationDesc => "slowest",
            Self::Status => "status",
        }
    }

    pub fn next(self) -> Self {
        let idx = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }
}

/// Filter modes for the `/agents` overlay. `f` cycles them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentsFilterMode {
    All,
    Running,
    Failed,
    Leaf,
}

impl AgentsFilterMode {
    pub const ALL: [Self; 4] =
        [Self::All, Self::Running, Self::Failed, Self::Leaf];

    pub fn label(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Running => "running",
            Self::Failed => "failed",
            Self::Leaf => "leaves",
        }
    }

    pub fn next(self) -> Self {
        let idx = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            focus: Focus::Input,
            sessions: Vec::new(),
            selected_session: 0,
            chat: Vec::new(),
            chat_scroll: 0,
            live_tool: None,
            recent_tools: Vec::new(),
            channels: Vec::new(),
            input: InputState::new(),
            started_at: Instant::now(),
            cursor_visible: true,
            transient_status: None,
            should_quit: false,
            compact_mode: false,
            details_mode: DetailsMode::Expanded,
            details_section_overrides: std::collections::HashMap::new(),
            mouse_enabled: false,
            in_flight_bot_idx: None,
            in_flight_reasoning_idx: None,
            voice: super::voice::VoiceController::new(),
            current_session_id: None,
            current_session_title: None,
            skills_dir: None,
            modal: None,
            pending_editor: None,
            spawn_tree: super::spawn_tree::SpawnTree::new(),
            spawn_history: super::spawn_tree::SpawnHistory::new(),
            show_agents_overlay: false,
            agents_cursor: None,
            agents_sort: AgentsSortMode::DepthFirst,
            agents_filter: AgentsFilterMode::All,
            agents_history_index: 0,
            agents_flash: None,
            agents_detail_focused: false,
            agents_detail_scroll: 0,
            delegation_registry: None,
            log_ring: super::log_ring::LogRing::new(),
            statusbar_position: StatusBarPosition::default(),
            indicator_style: IndicatorStyle::default(),
            verbosity: VerbosityMode::default(),
            busy_mode: BusyMode::default(),
            personality_name: String::new(),
            queued_input: std::collections::VecDeque::new(),
            skin_name: String::new(),
            show_reasoning: true,
            agents_diff_pair: None,
            skin: super::skin::Skin::default(),
            main_interrupt_flag: None,
        }
    }

    /// Whether a modal overlay is currently consuming input.
    /// Mirrors Hermes' `$isBlocked` computed atom: when `true`,
    /// the global key router sends input to the modal handler
    /// instead of focus-dependent panes.
    pub fn is_blocked(&self) -> bool {
        self.modal.is_some()
    }

    /// Open a local confirmation prompt. Used by destructive
    /// slash commands that want a "are you sure?" gate before
    /// running. `on_confirm` runs only on `Yes`; `No` / Esc /
    /// Ctrl-C drop the closure unrun.
    ///
    /// `danger=true` switches the modal's icon (⚠) and border
    /// color (terracotta) to flag risk; safe confirmations use
    /// `?` and amber.
    pub fn show_confirm(
        &mut self,
        title: impl Into<String>,
        detail: Option<String>,
        danger: bool,
        on_confirm: Box<dyn FnOnce() + Send>,
    ) {
        self.modal = Some(super::modal::Modal::Confirm {
            title: title.into(),
            detail,
            danger,
            cursor: super::modal::ConfirmChoice::No,
            on_confirm: Some(on_confirm),
        });
    }

    /// Open a fullscreen text pager. Lines wrap inside the
    /// pager's body; `j/k`, `PgUp/PgDn`, `g/G`, `q/Esc` for
    /// navigation. Used by future `/logs` and long `/help`
    /// output.
    pub fn show_pager(&mut self, title: Option<String>, lines: Vec<String>) {
        self.modal = Some(super::modal::Modal::Pager {
            title,
            lines,
            offset: 0,
        });
    }

    /// Append a streaming text delta to the in-flight assistant
    /// message. Pushes a new `Bot` line if there isn't one yet
    /// for this turn (or if a tool call broke the streaming
    /// continuity by setting `in_flight_bot_idx` back to `None`).
    pub fn append_bot_delta(&mut self, delta: &str) {
        if let Some(idx) = self.in_flight_bot_idx {
            if let Some(ChatLine::Bot { body, .. }) = self.chat.get_mut(idx) {
                body.push_str(delta);
                return;
            }
            // Fall through to "fresh push" if the index points
            // at a non-Bot line (shouldn't happen, but be safe).
            self.in_flight_bot_idx = None;
        }
        self.chat.push(ChatLine::Bot {
            time: chrono::Local::now().format("%H:%M:%S").to_string(),
            body: delta.to_string(),
        });
        self.in_flight_bot_idx = Some(self.chat.len() - 1);
    }

    /// Mark the in-flight assistant message complete (a tool call
    /// is starting, or the turn is ending). Subsequent
    /// `append_bot_delta` will start a fresh message.
    pub fn finalize_bot_message(&mut self) {
        self.in_flight_bot_idx = None;
    }

    /// Append a streaming reasoning delta to the in-flight
    /// thinking block. Mirrors `append_bot_delta`'s pattern but
    /// for `ChatLine::ReasoningBlock`. The block stays
    /// `is_streaming = true` until [`Self::finalize_reasoning_block`]
    /// is called (typically when the bot's text reply starts or
    /// the turn ends).
    pub fn append_reasoning_delta(&mut self, delta: &str) {
        if let Some(idx) = self.in_flight_reasoning_idx {
            if let Some(ChatLine::ReasoningBlock { body, .. }) =
                self.chat.get_mut(idx)
            {
                body.push_str(delta);
                return;
            }
            self.in_flight_reasoning_idx = None;
        }
        self.chat.push(ChatLine::ReasoningBlock {
            body: delta.to_string(),
            started_at: Instant::now(),
            is_streaming: true,
        });
        self.in_flight_reasoning_idx = Some(self.chat.len() - 1);
    }

    /// Stop the live cursor on the in-flight reasoning block —
    /// usually called when the assistant's text reply begins
    /// (thinking is done streaming).
    pub fn finalize_reasoning_block(&mut self) {
        if let Some(idx) = self.in_flight_reasoning_idx.take() {
            if let Some(ChatLine::ReasoningBlock { is_streaming, .. }) =
                self.chat.get_mut(idx)
            {
                *is_streaming = false;
            }
        }
    }

    /// Effective `DetailsMode` for the `thinking` section.
    /// Per-section override (`details_section_overrides["thinking"]`)
    /// wins; otherwise falls back to the global
    /// `details_mode`. Mirrors `domain/details.ts:69-74`'s
    /// `sectionMode()`.
    pub fn reasoning_mode(&self) -> DetailsMode {
        self.details_section_overrides
            .get("thinking")
            .copied()
            .unwrap_or(self.details_mode)
    }

    /// Set or clear a per-section override.
    pub fn set_section_override(
        &mut self,
        section: impl Into<String>,
        mode: Option<DetailsMode>,
    ) {
        match mode {
            Some(m) => {
                self.details_section_overrides.insert(section.into(), m);
            }
            None => {
                self.details_section_overrides.remove(&section.into());
            }
        }
    }

    /// Handle a key press. Routes to the focused pane; some keys
    /// are global (Tab cycles focus, Ctrl-C/D exits — handled in
    /// the event loop).
    ///
    /// When [`Self::is_blocked`] returns true (a modal is active)
    /// keys go to the modal handler instead — Tab does NOT cycle
    /// focus, so the user can't accidentally rotate panes while
    /// a prompt is open.
    pub fn handle_key(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        // Overlays consume input first — Tab does NOT cycle focus
        // when an overlay is up so the user can't accidentally
        // rotate panes behind it.
        if self.show_agents_overlay {
            self.handle_key_agents_overlay(code, modifiers);
            return;
        }
        if self.is_blocked() {
            self.handle_key_modal(code, modifiers);
            return;
        }
        // Global quit chords. Deliberately AFTER overlay/modal routing:
        // a modal Ctrl-C denies the pending request (handled above)
        // instead of killing the app. q/Esc are NOT global — `q` is a
        // letter the user types in the composer; Esc/q quit only from
        // the non-typing panes (see the focus handlers).
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        if ctrl && matches!(code, KeyCode::Char('c') | KeyCode::Char('d')) {
            self.should_quit = true;
            return;
        }
        // Global hotkeys.
        if matches!(code, KeyCode::Tab) {
            self.cycle_focus();
            return;
        }

        match self.focus {
            Focus::Sessions => self.handle_key_sessions(code),
            Focus::Chat => self.handle_key_chat(code),
            Focus::Input => self.handle_key_input(code, modifiers),
        }
    }

    /// Modal-active key handler. Each variant has a distinct key
    /// map matching Hermes' per-modal `useInput` handlers in
    /// `prompts.tsx`. On resolve the modal is removed from
    /// `self.modal` and (for callback-driven modals) the user's
    /// choice is sent through the oneshot.
    fn handle_key_modal(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        use super::modal::{ApprovalChoice, ConfirmChoice, Modal};
        let ctrl_c =
            matches!(code, KeyCode::Char('c')) && modifiers.contains(KeyModifiers::CONTROL);
        let esc = matches!(code, KeyCode::Esc);
        // Take the modal out so we can reason about it without
        // borrowing self mutably twice. Re-insert if not resolved.
        let Some(modal) = self.modal.take() else { return };
        match modal {
            Modal::Approval { request, mut cursor, resp_tx } => {
                if ctrl_c {
                    let _ = resp_tx.send(ApprovalChoice::Deny);
                    return;
                }
                match code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        cursor = cursor.saturating_sub(1);
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        cursor = (cursor + 1).min(3);
                    }
                    KeyCode::Char(c @ '1'..='4') => {
                        let n = (c as u8) - b'0';
                        if let Some(choice) = ApprovalChoice::from_quick_pick(n) {
                            let _ = resp_tx.send(choice);
                            return;
                        }
                    }
                    KeyCode::Enter => {
                        if let Some(choice) = Modal::approval_choice_at(cursor) {
                            let _ = resp_tx.send(choice);
                        } else {
                            let _ = resp_tx.send(ApprovalChoice::Deny);
                        }
                        return;
                    }
                    _ => {}
                }
                self.modal = Some(Modal::Approval {
                    request,
                    cursor,
                    resp_tx,
                });
            }
            Modal::Clarify {
                request,
                mut cursor,
                mut text,
                mut text_col,
                resp_tx,
            } => {
                if ctrl_c {
                    let _ = resp_tx.send(None);
                    return;
                }
                let in_text_mode = cursor.is_none() || request.options.is_empty();
                if in_text_mode {
                    match code {
                        KeyCode::Esc => {
                            // Back to choice list if we have one;
                            // otherwise cancel.
                            if request.options.is_empty() {
                                let _ = resp_tx.send(None);
                                return;
                            }
                            cursor = Some(0);
                            text.clear();
                            text_col = 0;
                        }
                        KeyCode::Enter => {
                            if text.trim().is_empty() {
                                let _ = resp_tx.send(None);
                            } else {
                                let _ = resp_tx.send(Some(text));
                            }
                            return;
                        }
                        KeyCode::Backspace => {
                            if text_col > 0 {
                                let prev = char_index_to_byte(&text, text_col - 1);
                                let cur = char_index_to_byte(&text, text_col);
                                text.replace_range(prev..cur, "");
                                text_col -= 1;
                            }
                        }
                        KeyCode::Left => {
                            text_col = text_col.saturating_sub(1);
                        }
                        KeyCode::Right => {
                            let len = text.chars().count();
                            text_col = (text_col + 1).min(len);
                        }
                        KeyCode::Char(c)
                            if !modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            let byte_idx = char_index_to_byte(&text, text_col);
                            text.insert(byte_idx, c);
                            text_col += 1;
                        }
                        _ => {}
                    }
                } else {
                    let total = Modal::clarify_total_rows(&request);
                    match code {
                        KeyCode::Esc => {
                            let _ = resp_tx.send(None);
                            return;
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            cursor = Some(cursor.unwrap_or(0).saturating_sub(1));
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            cursor =
                                Some((cursor.unwrap_or(0) + 1).min(total - 1));
                        }
                        KeyCode::Char(c @ '1'..='9') => {
                            // ASCII digit → 0-indexed slot. `c as
                            // usize` is the codepoint (49 for '1'),
                            // so subtract `b'1'` to get the offset.
                            let n = (c as u8 - b'1') as usize;
                            if n < total {
                                if n == request.options.len() {
                                    // "Other" → enter free-text mode.
                                    cursor = None;
                                    text.clear();
                                    text_col = 0;
                                } else {
                                    let chosen = request.options[n].clone();
                                    let _ = resp_tx.send(Some(chosen));
                                    return;
                                }
                            }
                        }
                        KeyCode::Enter => {
                            let idx = cursor.unwrap_or(0);
                            if idx == request.options.len() {
                                cursor = None;
                                text.clear();
                                text_col = 0;
                            } else if let Some(opt) = request.options.get(idx) {
                                let _ = resp_tx.send(Some(opt.clone()));
                                return;
                            }
                        }
                        _ => {}
                    }
                }
                self.modal = Some(Modal::Clarify {
                    request,
                    cursor,
                    text,
                    text_col,
                    resp_tx,
                });
            }
            Modal::Confirm {
                title,
                detail,
                danger,
                mut cursor,
                mut on_confirm,
            } => {
                if ctrl_c || esc {
                    // Cancel — drop the action.
                    return;
                }
                match code {
                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        if let Some(action) = on_confirm.take() {
                            action();
                        }
                        return;
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') => {
                        return;
                    }
                    KeyCode::Up | KeyCode::Down | KeyCode::Tab => {
                        cursor = match cursor {
                            ConfirmChoice::No => ConfirmChoice::Yes,
                            ConfirmChoice::Yes => ConfirmChoice::No,
                        };
                    }
                    KeyCode::Enter => {
                        if cursor == ConfirmChoice::Yes {
                            if let Some(action) = on_confirm.take() {
                                action();
                            }
                        }
                        return;
                    }
                    _ => {}
                }
                self.modal = Some(Modal::Confirm {
                    title,
                    detail,
                    danger,
                    cursor,
                    on_confirm,
                });
            }
            Modal::Secret {
                request,
                mut text,
                mut text_col,
                resp_tx,
            } => {
                if ctrl_c || esc {
                    let _ = resp_tx.send(None);
                    return;
                }
                match code {
                    KeyCode::Enter => {
                        if text.is_empty() {
                            let _ = resp_tx.send(None);
                        } else {
                            let _ = resp_tx.send(Some(text));
                        }
                        return;
                    }
                    KeyCode::Backspace => {
                        if text_col > 0 {
                            let prev = char_index_to_byte(&text, text_col - 1);
                            let cur = char_index_to_byte(&text, text_col);
                            text.replace_range(prev..cur, "");
                            text_col -= 1;
                        }
                    }
                    KeyCode::Char(c)
                        if !modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        let byte_idx = char_index_to_byte(&text, text_col);
                        text.insert(byte_idx, c);
                        text_col += 1;
                    }
                    _ => {}
                }
                self.modal = Some(Modal::Secret {
                    request,
                    text,
                    text_col,
                    resp_tx,
                });
            }
            Modal::Sudo {
                prompt,
                mut text,
                mut text_col,
                resp_tx,
            } => {
                if ctrl_c || esc {
                    let _ = resp_tx.send(None);
                    return;
                }
                match code {
                    KeyCode::Enter => {
                        if text.is_empty() {
                            let _ = resp_tx.send(None);
                        } else {
                            let _ = resp_tx.send(Some(text));
                        }
                        return;
                    }
                    KeyCode::Backspace => {
                        if text_col > 0 {
                            let prev = char_index_to_byte(&text, text_col - 1);
                            let cur = char_index_to_byte(&text, text_col);
                            text.replace_range(prev..cur, "");
                            text_col -= 1;
                        }
                    }
                    KeyCode::Char(c)
                        if !modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        let byte_idx = char_index_to_byte(&text, text_col);
                        text.insert(byte_idx, c);
                        text_col += 1;
                    }
                    _ => {}
                }
                self.modal = Some(Modal::Sudo {
                    prompt,
                    text,
                    text_col,
                    resp_tx,
                });
            }
            Modal::Pager {
                title,
                lines,
                mut offset,
            } => {
                if ctrl_c || esc {
                    return;
                }
                let page = 10usize;
                match code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => return,
                    KeyCode::Up | KeyCode::Char('k') => {
                        offset = offset.saturating_sub(1);
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        offset = (offset + 1).min(lines.len().saturating_sub(1));
                    }
                    KeyCode::PageUp | KeyCode::Char('b') => {
                        offset = offset.saturating_sub(page);
                    }
                    KeyCode::PageDown | KeyCode::Char(' ') | KeyCode::Enter => {
                        offset = (offset + page).min(lines.len().saturating_sub(1));
                    }
                    KeyCode::Char('g') => {
                        offset = 0;
                    }
                    KeyCode::Char('G') => {
                        offset = lines.len().saturating_sub(1);
                    }
                    _ => {}
                }
                self.modal = Some(Modal::Pager {
                    title,
                    lines,
                    offset,
                });
            }
        }
    }

    /// Keyboard handler active while the `/agents` overlay is
    /// open. Layered behaviours:
    ///
    /// - **always**: `q`/`Q`/`Esc`/`Ctrl-C` close
    /// - **navigation**: `↑/k` `↓/j` walk the flat tree;
    ///   `g`/`G` top/bottom
    /// - **sort/filter**: `s` cycles sort, `f` cycles filter
    /// - **history**: `[`/`<` step older, `]`/`>` step toward live
    /// - **live-only actions**: `x` kill node, `X` kill subtree,
    ///   `p` toggle delegation pause. In replay mode these
    ///   surface "replay mode — controls disabled" via the flash.
    pub fn handle_key_agents_overlay(
        &mut self,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) {
        let ctrl_c = matches!(code, KeyCode::Char('c'))
            && modifiers.contains(KeyModifiers::CONTROL);
        if ctrl_c || matches!(code, KeyCode::Char('q') | KeyCode::Char('Q')) {
            self.show_agents_overlay = false;
            self.agents_detail_focused = false;
            return;
        }
        // Esc returns from detail focus to list focus when in
        // detail mode; only closes the overlay from list mode.
        if matches!(code, KeyCode::Esc) {
            if self.agents_detail_focused {
                self.agents_detail_focused = false;
                self.agents_detail_scroll = 0;
            } else {
                self.show_agents_overlay = false;
            }
            return;
        }

        // Detail-focused mode owns scroll keys.
        if self.agents_detail_focused {
            match code {
                KeyCode::Left | KeyCode::Char('h') => {
                    self.agents_detail_focused = false;
                    self.agents_detail_scroll = 0;
                    return;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    self.agents_detail_scroll = self.agents_detail_scroll.saturating_sub(2);
                    return;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.agents_detail_scroll = self.agents_detail_scroll.saturating_add(2);
                    return;
                }
                KeyCode::PageUp => {
                    self.agents_detail_scroll = self.agents_detail_scroll.saturating_sub(10);
                    return;
                }
                KeyCode::PageDown => {
                    self.agents_detail_scroll = self.agents_detail_scroll.saturating_add(10);
                    return;
                }
                KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.agents_detail_scroll = self.agents_detail_scroll.saturating_add(10);
                    return;
                }
                KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.agents_detail_scroll = self.agents_detail_scroll.saturating_sub(10);
                    return;
                }
                KeyCode::Char('g') => {
                    self.agents_detail_scroll = 0;
                    return;
                }
                KeyCode::Char('G') => {
                    self.agents_detail_scroll = u16::MAX / 2;
                    return;
                }
                _ => {}
            }
        }

        // History nav + sort/filter are mode-independent.
        match code {
            KeyCode::Char('s') => {
                self.agents_sort = self.agents_sort.next();
                self.set_agents_flash(format!("sort · {}", self.agents_sort.label()));
                return;
            }
            KeyCode::Char('f') => {
                self.agents_filter = self.agents_filter.next();
                self.set_agents_flash(format!("filter · {}", self.agents_filter.label()));
                return;
            }
            KeyCode::Char('[') | KeyCode::Char('<') => {
                self.agents_step_history(1);
                return;
            }
            KeyCode::Char(']') | KeyCode::Char('>') => {
                self.agents_step_history(-1);
                return;
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                self.agents_detail_focused = true;
                self.agents_detail_scroll = 0;
                return;
            }
            _ => {}
        }

        // Live-only actions: x / X / p surface a flash in replay
        // mode rather than mutating state.
        let replay_mode = self.agents_history_index > 0;
        match code {
            KeyCode::Char('p') => {
                if replay_mode {
                    self.set_agents_flash(
                        "replay mode — controls disabled".to_string(),
                    );
                    return;
                }
                if let Some(reg) = self.delegation_registry.clone() {
                    let now_paused = reg.set_paused(!reg.is_paused());
                    self.set_agents_flash(if now_paused {
                        "spawning paused".to_string()
                    } else {
                        "spawning resumed".to_string()
                    });
                } else {
                    self.set_agents_flash(
                        "no delegation registry attached".to_string(),
                    );
                }
                return;
            }
            KeyCode::Char('x') => {
                if replay_mode {
                    self.set_agents_flash(
                        "replay mode — controls disabled".to_string(),
                    );
                    return;
                }
                if let Some(id) = self.agents_cursor.clone() {
                    if let Some(reg) = self.delegation_registry.as_ref() {
                        if reg.interrupt(&id) {
                            self.set_agents_flash(format!("killing {id}"));
                        }
                    }
                    self.spawn_tree.interrupt(&id, false);
                }
                return;
            }
            KeyCode::Char('X') => {
                if replay_mode {
                    self.set_agents_flash(
                        "replay mode — controls disabled".to_string(),
                    );
                    return;
                }
                if let Some(id) = self.agents_cursor.clone() {
                    if let Some(reg) = self.delegation_registry.as_ref() {
                        let n = reg.interrupt_subtree(&id);
                        self.set_agents_flash(format!(
                            "killing subtree · {n} node{}",
                            if n == 1 { "" } else { "s" }
                        ));
                    }
                    self.spawn_tree.interrupt(&id, true);
                }
                return;
            }
            _ => {}
        }

        let flat = self.agents_flat_node_ids();
        if flat.is_empty() {
            return;
        }
        let cur_idx = self
            .agents_cursor
            .as_ref()
            .and_then(|id| flat.iter().position(|x| x == id))
            .unwrap_or(0);
        let prev_cursor = self.agents_cursor.clone();
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                let next = cur_idx.saturating_sub(1);
                self.agents_cursor = Some(flat[next].clone());
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let next = (cur_idx + 1).min(flat.len() - 1);
                self.agents_cursor = Some(flat[next].clone());
            }
            KeyCode::Char('g') => {
                self.agents_cursor = flat.first().cloned();
            }
            KeyCode::Char('G') => {
                self.agents_cursor = flat.last().cloned();
            }
            _ => {}
        }
        if self.agents_cursor != prev_cursor {
            // Reset detail scroll so the new node opens from the
            // top — otherwise the user is mid-section on the new
            // record without realising the cursor moved.
            self.agents_detail_scroll = 0;
        }
    }

    /// Step the history cursor by `delta` (`+1` = older,
    /// `-1` = newer). Clamps to `[0, spawn_history.len()]`.
    /// Resets the row cursor + emits a flash when the index
    /// changes.
    pub fn agents_step_history(&mut self, delta: i32) {
        let max = self.spawn_history.len();
        let cur = self.agents_history_index as i32;
        let next = (cur + delta).clamp(0, max as i32) as usize;
        if next == self.agents_history_index {
            return;
        }
        self.agents_history_index = next;
        self.agents_cursor = self.agents_flat_node_ids().first().cloned();
        let flash = if next == 0 {
            "live turn".to_string()
        } else {
            format!("replay · {next}/{max}")
        };
        self.set_agents_flash(flash);
    }

    /// Emit a transient flash for the overlay footer. Expires
    /// after ~2s via `on_tick`. Replaces any in-flight flash.
    pub fn set_agents_flash(&mut self, body: String) {
        self.agents_flash = Some((body, Instant::now()));
    }

    /// Borrow the spawn tree that the overlay should render at
    /// the current history index. Live tree when `index == 0`,
    /// snapshot otherwise. Falls back to the live (possibly
    /// empty) tree when the requested index is out of bounds.
    pub fn agents_effective_tree(&self) -> &super::spawn_tree::SpawnTree {
        if self.agents_history_index == 0 {
            &self.spawn_tree
        } else if let Some(snap) =
            self.spawn_history.get(self.agents_history_index - 1)
        {
            &snap.tree
        } else {
            &self.spawn_tree
        }
    }

    /// Whether the overlay is currently in replay mode (showing
    /// a history snapshot rather than the live tree).
    pub fn agents_replay_mode(&self) -> bool {
        self.agents_history_index > 0
    }

    /// Flattened list of every node id in the active spawn tree
    /// (live or selected history snapshot), in render order with
    /// sort + filter applied. Used by the overlay's keyboard
    /// navigation + the tree-pane renderer.
    pub fn agents_flat_node_ids(&self) -> Vec<String> {
        let tree = self.agents_effective_tree();
        // Auto-fallback: when live tree is empty + we're at
        // index 0, peek at history[0] so the user isn't stuck
        // with a blank overlay.
        let tree = if tree.is_empty() && self.agents_history_index == 0 {
            self.spawn_history
                .get(0)
                .map(|s| &s.tree)
                .unwrap_or(tree)
        } else {
            tree
        };
        if tree.is_empty() {
            return Vec::new();
        }
        let mut roots: Vec<&str> = tree.root_ids.iter().map(String::as_str).collect();
        let sort = self.agents_sort;
        roots.sort_by(|a, b| compare_nodes(tree, a, b, sort));
        let mut out = Vec::new();
        for root in roots {
            collect_subtree_ids_sorted(tree, root, sort, &mut out);
        }
        // Filter retains only matching ids; render order is the
        // depth-first sorted walk above.
        let filter = self.agents_filter;
        out.into_iter()
            .filter(|id| {
                tree.nodes
                    .get(id)
                    .map(|n| filter_matches(filter, n, tree))
                    .unwrap_or(false)
            })
            .collect()
    }

    fn handle_key_sessions(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.prev_session(),
            KeyCode::Down | KeyCode::Char('j') => self.next_session(),
            // Non-typing pane: q / Esc quit the TUI.
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            _ => {}
        }
    }

    fn handle_key_chat(&mut self, code: KeyCode) {
        match code {
            KeyCode::PageUp => {
                self.chat_scroll = self.chat_scroll.saturating_add(10);
            }
            KeyCode::PageDown => {
                self.chat_scroll = self.chat_scroll.saturating_sub(10);
            }
            KeyCode::Up => {
                self.chat_scroll = self.chat_scroll.saturating_add(1);
            }
            KeyCode::Down => {
                self.chat_scroll = self.chat_scroll.saturating_sub(1);
            }
            // Non-typing pane: q / Esc quit the TUI.
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            _ => {}
        }
    }

    fn handle_key_input(&mut self, code: KeyCode, modifiers: KeyModifiers) {
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        let alt = modifiers.contains(KeyModifiers::ALT);
        let shift = modifiers.contains(KeyModifiers::SHIFT);
        match code {
            // Submit / newline.
            KeyCode::Enter if shift || alt => self.input.insert_newline(),
            KeyCode::Enter => {
                let text = self.input.text().trim().to_string();
                if !text.is_empty() {
                    self.input.push_history(text.clone());
                    self.input.clear();
                    self.set_status(format!("submitted: {}", truncate(&text, 40)));
                }
            }

            // Editing — undo / redo.
            KeyCode::Char('z') if ctrl => self.input.undo(),
            KeyCode::Char('y') if ctrl => self.input.redo(),

            // Open $EDITOR with the current input pre-filled.
            // Hermes uses Cmd-G (macOS) / Ctrl-G (Linux/Windows)
            // with Alt-G as the VSCode/Cursor fallback (those
            // terminals intercept Ctrl-G as "Find Next" before
            // we see it). Crossterm reports Alt as the META
            // modifier, so we accept either.
            KeyCode::Char('g') if ctrl || alt => {
                self.pending_editor = Some(self.input.text());
            }

            // Editing — bulk delete.
            KeyCode::Char('w') if ctrl => self.input.delete_word_backward(),
            KeyCode::Char('u') if ctrl => self.input.delete_to_line_start(),
            KeyCode::Char('k') if ctrl => self.input.delete_to_line_end(),

            // Editing — single-char delete.
            KeyCode::Backspace => self.input.backspace(),

            // Cursor movement — char.
            KeyCode::Left if ctrl || alt => self.input.move_word_left(),
            KeyCode::Right if ctrl || alt => self.input.move_word_right(),
            KeyCode::Left => self.input.move_left(),
            KeyCode::Right => self.input.move_right(),

            // Cursor movement — line / vertical / history.
            KeyCode::Home => self.input.move_line_start(),
            KeyCode::End => self.input.move_line_end(),
            KeyCode::Char('a') if ctrl => self.input.move_line_start(),
            KeyCode::Char('e') if ctrl => self.input.move_line_end(),
            KeyCode::Up if self.input.at_first_row() => self.input.history_prev(),
            KeyCode::Up => self.input.move_up(),
            KeyCode::Down if self.input.at_last_row() => self.input.history_next(),
            KeyCode::Down => self.input.move_down(),

            // Esc in the composer: clear a non-empty input; quit when
            // already empty (Esc-Esc exits without leaving the keyboard).
            KeyCode::Esc => {
                if self.input.text().is_empty() {
                    self.should_quit = true;
                } else {
                    self.input.clear();
                    self.set_status("input cleared (Esc again to quit)".to_string());
                }
            }

            // Char input. Skipping ASCII control chars (0x00-0x1F)
            // so Ctrl-key combos we don't bind don't spew junk.
            KeyCode::Char(c) if !ctrl => self.input.insert_char(c),

            _ => {}
        }
    }

    /// Called every ~100ms by the event loop. Drives cursor blink
    /// and any time-based UI state changes.
    pub fn on_tick(&mut self) {
        let elapsed_500 = (self.started_at.elapsed().as_millis() / 500) % 2;
        self.cursor_visible = elapsed_500 == 0;
        // Auto-clear transient status after 3 seconds.
        if let Some((_, t)) = &self.transient_status {
            if t.elapsed().as_secs() >= 3 {
                self.transient_status = None;
            }
        }
        // Voice subsystem: a delivered transcription drops into
        // the input box; a delivered error shows as a transient
        // status so the user sees mic-open failures etc.
        if let Some(text) = self.voice.take_transcription() {
            self.input.set(&text);
            self.set_status(format!(
                "transcribed ({} chars) — review + Enter to send",
                text.chars().count()
            ));
        }
        if let Some(err) = self.voice.take_error() {
            self.set_status(format!("voice: {err}"));
        }
        // Overlay flash expires after ~2.5 seconds so the footer
        // returns to the static legend.
        if let Some((_, t)) = &self.agents_flash {
            if t.elapsed().as_millis() >= 2500 {
                self.agents_flash = None;
            }
        }
        // Auto-follow: when the live tree clears mid-turn and we
        // were viewing it, hop onto history[0] so the user isn't
        // dropped into an empty overlay. Only fires once per
        // clearing — we check that the history has at least one
        // entry and we're currently at index 0.
        if self.show_agents_overlay
            && self.agents_history_index == 0
            && self.spawn_tree.is_empty()
            && !self.spawn_history.is_empty()
        {
            // Already on index 0; bump to 1 only if there's a
            // snapshot newer than what we just settled (the upstream
            // uses the same "snapshot we just pushed" trigger).
            // SpawnHistory.push happens elsewhere when a tree
            // settles, so by the time we observe is_empty +
            // history.len() >= 1, the just-finished tree IS
            // history[0]. Jump to it.
            self.agents_history_index = 1;
            self.agents_cursor = self.agents_flat_node_ids().first().cloned();
            self.set_agents_flash(
                "turn finished · inspect freely · q to close".to_string(),
            );
        }
    }

    pub fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Sessions => Focus::Chat,
            Focus::Chat => Focus::Input,
            Focus::Input => Focus::Sessions,
        };
    }

    pub fn next_session(&mut self) {
        if self.selected_session + 1 < self.sessions.len() {
            self.selected_session += 1;
        }
    }

    pub fn prev_session(&mut self) {
        self.selected_session = self.selected_session.saturating_sub(1);
    }

    pub fn set_status(&mut self, msg: impl Into<String>) {
        self.transient_status = Some((msg.into(), Instant::now()));
    }

    /// Animated spinner glyph for live tool indicators. Style
    /// follows `App.indicator_style` (toggleable via `/indicator`).
    pub fn spinner_glyph(&self) -> &'static str {
        self.indicator_style.frame(self.started_at.elapsed().as_millis())
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install_pager(app: &mut App, lines: Vec<&str>) {
        app.show_pager(
            Some("test pager".into()),
            lines.into_iter().map(String::from).collect(),
        );
    }

    // -- Quit-key routing (regression: q/Esc/Ctrl-C were intercepted
    // -- by the event loop BEFORE the app saw them, so typing "q" in
    // -- the composer exited the TUI and Esc killed the app mid-modal).

    #[test]
    fn typing_q_in_composer_inserts_not_quits() {
        let mut app = App::new();
        app.focus = Focus::Input;
        app.handle_key(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(!app.should_quit, "typing 'q' must not quit");
        assert_eq!(app.input.text(), "q");
    }

    #[test]
    fn esc_in_composer_clears_then_quits() {
        let mut app = App::new();
        app.focus = Focus::Input;
        app.handle_key(KeyCode::Char('h'), KeyModifiers::NONE);
        app.handle_key(KeyCode::Char('i'), KeyModifiers::NONE);
        app.handle_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(!app.should_quit, "first Esc clears, must not quit");
        assert!(app.input.text().is_empty());
        app.handle_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(app.should_quit, "Esc on an empty composer quits");
    }

    #[test]
    fn q_and_esc_quit_from_non_typing_panes() {
        let mut app = App::new();
        app.focus = Focus::Chat;
        app.handle_key(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(app.should_quit);

        let mut app = App::new();
        app.focus = Focus::Sessions;
        app.handle_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(app.should_quit);
    }

    #[test]
    fn ctrl_c_quits_globally_but_not_inside_modals() {
        let mut app = App::new();
        app.focus = Focus::Input;
        app.handle_key(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(app.should_quit, "Ctrl-C quits from the main surface");

        // With a modal up, Ctrl-C resolves the modal (deny/close path)
        // instead of quitting the app.
        let mut app = App::new();
        install_pager(&mut app, vec!["line"]);
        assert!(app.is_blocked());
        app.handle_key(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(!app.should_quit, "modal must consume Ctrl-C");
    }

    #[test]
    fn esc_inside_modal_closes_modal_not_app() {
        let mut app = App::new();
        install_pager(&mut app, vec!["line"]);
        assert!(app.is_blocked());
        app.handle_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(!app.should_quit, "Esc must reach the modal, not quit");
        assert!(!app.is_blocked(), "pager closes on Esc");
    }

    #[test]
    fn pager_arrow_keys_scroll_within_bounds() {
        use super::super::modal::Modal;
        let mut app = App::new();
        let lines: Vec<&str> = (0..30).map(|_| "line").collect();
        install_pager(&mut app, lines);
        assert!(app.is_blocked());
        app.handle_key(KeyCode::Down, KeyModifiers::NONE);
        if let Some(Modal::Pager { offset, .. }) = &app.modal {
            assert_eq!(*offset, 1);
        } else {
            panic!("pager not present");
        }
        app.handle_key(KeyCode::Char('g'), KeyModifiers::NONE);
        if let Some(Modal::Pager { offset, .. }) = &app.modal {
            assert_eq!(*offset, 0);
        } else {
            panic!("pager not present");
        }
        app.handle_key(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(!app.is_blocked());
    }

    #[test]
    fn show_confirm_marks_blocked_and_runs_action_on_yes() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let fired = Arc::new(AtomicBool::new(false));
        let fired_for_closure = fired.clone();
        let mut app = App::new();
        app.show_confirm(
            "delete?",
            Some("this will delete X".into()),
            true,
            Box::new(move || fired_for_closure.store(true, Ordering::SeqCst)),
        );
        assert!(app.is_blocked());
        // 'y' confirms.
        app.handle_key(KeyCode::Char('y'), KeyModifiers::NONE);
        assert!(!app.is_blocked());
        assert!(fired.load(Ordering::SeqCst));
    }

    #[test]
    fn show_confirm_does_not_run_action_on_no() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let fired = Arc::new(AtomicBool::new(false));
        let fired_for_closure = fired.clone();
        let mut app = App::new();
        app.show_confirm(
            "delete?",
            None,
            true,
            Box::new(move || fired_for_closure.store(true, Ordering::SeqCst)),
        );
        // 'n' rejects.
        app.handle_key(KeyCode::Char('n'), KeyModifiers::NONE);
        assert!(!app.is_blocked());
        assert!(!fired.load(Ordering::SeqCst));
    }

    #[test]
    fn show_confirm_esc_drops_action_unrun() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let fired = Arc::new(AtomicBool::new(false));
        let fired_for_closure = fired.clone();
        let mut app = App::new();
        app.show_confirm(
            "delete?",
            None,
            true,
            Box::new(move || fired_for_closure.store(true, Ordering::SeqCst)),
        );
        app.handle_key(KeyCode::Esc, KeyModifiers::NONE);
        assert!(!app.is_blocked());
        assert!(!fired.load(Ordering::SeqCst));
    }

    #[test]
    fn modal_blocks_focus_cycle() {
        let mut app = App::new();
        let initial_focus = app.focus;
        app.show_pager(None, vec!["a".into(), "b".into()]);
        // Tab would normally cycle focus; with modal active it shouldn't.
        app.handle_key(KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(app.focus, initial_focus);
        assert!(app.is_blocked());
    }

    #[test]
    fn details_mode_parses_known_aliases() {
        assert_eq!(DetailsMode::parse("hidden"), Some(DetailsMode::Hidden));
        assert_eq!(DetailsMode::parse("h"), Some(DetailsMode::Hidden));
        assert_eq!(DetailsMode::parse("off"), Some(DetailsMode::Hidden));
        assert_eq!(
            DetailsMode::parse("collapsed"),
            Some(DetailsMode::Collapsed)
        );
        assert_eq!(DetailsMode::parse("c"), Some(DetailsMode::Collapsed));
        assert_eq!(
            DetailsMode::parse("expanded"),
            Some(DetailsMode::Expanded)
        );
        assert_eq!(DetailsMode::parse("on"), Some(DetailsMode::Expanded));
        assert_eq!(DetailsMode::parse("FULL"), Some(DetailsMode::Expanded));
    }

    #[test]
    fn details_mode_parse_returns_none_for_unknown() {
        assert!(DetailsMode::parse("foobar").is_none());
        assert!(DetailsMode::parse("").is_none());
    }

    #[test]
    fn details_mode_round_trips_via_as_str() {
        for m in [DetailsMode::Hidden, DetailsMode::Collapsed, DetailsMode::Expanded] {
            let serialized = m.as_str();
            assert_eq!(
                DetailsMode::parse(serialized),
                Some(m),
                "{serialized} did not round-trip"
            );
        }
    }

    #[test]
    fn input_state_starts_empty() {
        let s = InputState::new();
        assert!(s.is_empty());
        assert_eq!(s.text(), "");
    }

    #[test]
    fn input_state_set_and_clear() {
        let mut s = InputState::new();
        s.set("hello\nworld");
        assert_eq!(s.text(), "hello\nworld");
        assert_eq!(s.row, 1);
        assert_eq!(s.col, 5);
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn input_state_dedups_history() {
        let mut s = InputState::new();
        s.push_history("hi".into());
        s.push_history("hi".into()); // duplicate of front
        s.push_history("there".into());
        assert_eq!(s.history.len(), 2);
        assert_eq!(s.history.front().map(String::as_str), Some("there"));
    }

    #[test]
    fn input_state_caps_history_at_64() {
        let mut s = InputState::new();
        for i in 0..100 {
            s.push_history(format!("msg{i}"));
        }
        assert_eq!(s.history.len(), 64);
    }

    #[test]
    fn cycle_focus_round_trip() {
        let mut app = App::new();
        assert_eq!(app.focus, Focus::Input);
        app.cycle_focus();
        assert_eq!(app.focus, Focus::Sessions);
        app.cycle_focus();
        assert_eq!(app.focus, Focus::Chat);
        app.cycle_focus();
        assert_eq!(app.focus, Focus::Input);
    }

    #[test]
    fn navigation_clamps_at_bounds() {
        let mut app = App::new();
        app.sessions = vec![
            SessionRow {
                code: "TG".into(),
                who: "Alice".into(),
                subject: "x".into(),
                count: "1".into(),
                has_unread: false,
            },
            SessionRow {
                code: "SL".into(),
                who: "#a".into(),
                subject: "y".into(),
                count: "2".into(),
                has_unread: false,
            },
        ];
        app.focus = Focus::Sessions;
        app.next_session();
        assert_eq!(app.selected_session, 1);
        app.next_session(); // clamped
        assert_eq!(app.selected_session, 1);
        app.prev_session();
        assert_eq!(app.selected_session, 0);
        app.prev_session(); // clamped
        assert_eq!(app.selected_session, 0);
    }

    #[test]
    fn submitting_input_records_history_and_clears() {
        let mut app = App::new();
        app.focus = Focus::Input;
        app.handle_key(KeyCode::Char('h'), KeyModifiers::NONE);
        app.handle_key(KeyCode::Char('i'), KeyModifiers::NONE);
        app.handle_key(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.input.is_empty());
        assert_eq!(app.input.history.front().map(String::as_str), Some("hi"));
    }

    #[test]
    fn shift_enter_inserts_newline_does_not_submit() {
        let mut app = App::new();
        app.focus = Focus::Input;
        app.handle_key(KeyCode::Char('a'), KeyModifiers::NONE);
        app.handle_key(KeyCode::Enter, KeyModifiers::SHIFT);
        app.handle_key(KeyCode::Char('b'), KeyModifiers::NONE);
        assert_eq!(app.input.text(), "a\nb");
        assert_eq!(app.input.history.len(), 0);
    }

    #[test]
    fn ctrl_w_deletes_word_backward() {
        let mut s = InputState::new();
        for c in "hello world ".chars() {
            s.insert_char(c);
        }
        s.delete_word_backward();
        // First Ctrl-W eats trailing whitespace + "world".
        assert_eq!(s.text(), "hello ");
        s.delete_word_backward();
        // Second eats "hello" + the trailing space before it.
        assert_eq!(s.text(), "");
    }

    #[test]
    fn ctrl_u_deletes_to_line_start() {
        let mut s = InputState::new();
        for c in "hello world".chars() {
            s.insert_char(c);
        }
        s.move_line_start();
        s.move_word_right();
        s.delete_to_line_start();
        assert_eq!(s.text(), "world");
    }

    #[test]
    fn ctrl_k_deletes_to_line_end() {
        let mut s = InputState::new();
        for c in "hello world".chars() {
            s.insert_char(c);
        }
        s.col = 5;
        s.delete_to_line_end();
        assert_eq!(s.text(), "hello");
    }

    #[test]
    fn undo_redo_round_trip() {
        let mut s = InputState::new();
        for c in "abc".chars() {
            s.insert_char(c);
        }
        assert_eq!(s.text(), "abc");
        s.undo();
        s.undo();
        s.undo();
        assert_eq!(s.text(), "");
        s.redo();
        s.redo();
        s.redo();
        assert_eq!(s.text(), "abc");
    }

    #[test]
    fn history_navigation_preserves_live_buffer() {
        let mut s = InputState::new();
        s.push_history("first".into());
        s.push_history("second".into());
        // Type something the user hasn't submitted.
        for c in "draft".chars() {
            s.insert_char(c);
        }
        assert_eq!(s.text(), "draft");
        // ↑ goes to most-recent history entry.
        s.history_prev();
        assert_eq!(s.text(), "second");
        // ↑ goes further back.
        s.history_prev();
        assert_eq!(s.text(), "first");
        // ↑ at the oldest pins.
        s.history_prev();
        assert_eq!(s.text(), "first");
        // ↓ steps back forward.
        s.history_next();
        assert_eq!(s.text(), "second");
        // ↓ past most-recent restores the saved live buffer.
        s.history_next();
        assert_eq!(s.text(), "draft");
    }

    #[test]
    fn cursor_word_movement_skips_words_and_whitespace() {
        let mut s = InputState::new();
        for c in "the quick brown".chars() {
            s.insert_char(c);
        }
        s.move_line_start();
        s.move_word_right();
        assert_eq!(s.col, 4); // start of "quick"
        s.move_word_right();
        assert_eq!(s.col, 10); // start of "brown"
        s.move_word_left();
        assert_eq!(s.col, 4);
    }

    #[test]
    fn append_bot_delta_creates_then_extends_in_flight_message() {
        let mut app = App::new();
        app.append_bot_delta("hello");
        app.append_bot_delta(", world");
        assert_eq!(app.chat.len(), 1);
        match &app.chat[0] {
            ChatLine::Bot { body, .. } => assert_eq!(body, "hello, world"),
            other => panic!("expected Bot line, got {:?}", other),
        }
        assert_eq!(app.in_flight_bot_idx, Some(0));
    }

    #[test]
    fn finalize_then_append_starts_fresh_message() {
        let mut app = App::new();
        app.append_bot_delta("first reply");
        app.finalize_bot_message();
        app.append_bot_delta("second reply");
        assert_eq!(app.chat.len(), 2);
        assert_eq!(app.in_flight_bot_idx, Some(1));
    }

    #[test]
    fn reasoning_deltas_accumulate_into_streaming_block() {
        let mut app = App::new();
        app.append_reasoning_delta("let me ");
        app.append_reasoning_delta("think...");
        assert_eq!(app.chat.len(), 1);
        match &app.chat[0] {
            ChatLine::ReasoningBlock {
                body,
                is_streaming,
                ..
            } => {
                assert_eq!(body, "let me think...");
                assert!(*is_streaming);
            }
            other => panic!("expected ReasoningBlock, got {:?}", other),
        }
        assert_eq!(app.in_flight_reasoning_idx, Some(0));
    }

    #[test]
    fn finalize_reasoning_block_stops_streaming_flag() {
        let mut app = App::new();
        app.append_reasoning_delta("draft");
        app.finalize_reasoning_block();
        assert_eq!(app.in_flight_reasoning_idx, None);
        match &app.chat[0] {
            ChatLine::ReasoningBlock { is_streaming, .. } => {
                assert!(!*is_streaming);
            }
            other => panic!("expected ReasoningBlock, got {:?}", other),
        }
    }

    #[test]
    fn reasoning_mode_prefers_section_override_over_global() {
        let mut app = App::new();
        app.details_mode = DetailsMode::Hidden;
        // No override → global wins.
        assert_eq!(app.reasoning_mode(), DetailsMode::Hidden);
        // Set override → wins.
        app.set_section_override("thinking", Some(DetailsMode::Expanded));
        assert_eq!(app.reasoning_mode(), DetailsMode::Expanded);
        // Reset override → fall back to global again.
        app.set_section_override("thinking", None);
        assert_eq!(app.reasoning_mode(), DetailsMode::Hidden);
    }

    #[test]
    fn unicode_aware_editing() {
        let mut s = InputState::new();
        for c in "héllo".chars() {
            s.insert_char(c);
        }
        assert_eq!(s.col, 5);
        s.backspace();
        s.backspace();
        assert_eq!(s.text(), "hél");
        assert_eq!(s.col, 3);
    }

    fn spawn(id: &str, parent: Option<&str>) -> crate::agent::callbacks::SubagentSpawn {
        crate::agent::callbacks::SubagentSpawn {
            id: id.to_string(),
            parent_id: parent.map(str::to_string),
            goal: format!("goal {id}"),
            depth: 0,
            index: 0,
            model: None,
            toolsets: Vec::new(),
        }
    }

    #[test]
    fn agents_overlay_arrow_keys_walk_flat_tree() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        app.spawn_tree.on_spawn(spawn("root", None));
        app.spawn_tree.on_spawn(spawn("child", Some("root")));
        app.spawn_tree.on_spawn(spawn("leaf", Some("child")));
        app.agents_cursor = Some("root".into());

        app.handle_key_agents_overlay(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("child"));
        app.handle_key_agents_overlay(KeyCode::Char('j'), KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("leaf"));
        // Already at the bottom — clamps, does not wrap.
        app.handle_key_agents_overlay(KeyCode::Down, KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("leaf"));
        app.handle_key_agents_overlay(KeyCode::Char('k'), KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("child"));
        app.handle_key_agents_overlay(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("root"));
        // Already at the top — clamps.
        app.handle_key_agents_overlay(KeyCode::Up, KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("root"));
    }

    #[test]
    fn agents_overlay_gG_jump_to_top_and_bottom() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        app.spawn_tree.on_spawn(spawn("root", None));
        app.spawn_tree.on_spawn(spawn("child", Some("root")));
        app.spawn_tree.on_spawn(spawn("leaf", Some("child")));
        app.agents_cursor = Some("child".into());

        app.handle_key_agents_overlay(KeyCode::Char('G'), KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("leaf"));
        app.handle_key_agents_overlay(KeyCode::Char('g'), KeyModifiers::NONE);
        assert_eq!(app.agents_cursor.as_deref(), Some("root"));
    }

    #[test]
    fn agents_overlay_q_and_esc_close_overlay() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        app.handle_key_agents_overlay(KeyCode::Char('q'), KeyModifiers::NONE);
        assert!(!app.show_agents_overlay);

        app.show_agents_overlay = true;
        app.handle_key_agents_overlay(KeyCode::Esc, KeyModifiers::NONE);
        assert!(!app.show_agents_overlay);

        app.show_agents_overlay = true;
        app.handle_key_agents_overlay(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(!app.show_agents_overlay);
    }

    #[test]
    fn agents_overlay_x_interrupts_single_node_capital_x_interrupts_subtree() {
        use crate::tui::spawn_tree::SubagentStatus;
        let mut app = App::new();
        app.show_agents_overlay = true;
        app.spawn_tree.on_spawn(spawn("root", None));
        app.spawn_tree.on_spawn(spawn("child", Some("root")));
        app.spawn_tree.on_spawn(spawn("leaf", Some("child")));

        // `x` on `child` interrupts only that node.
        app.agents_cursor = Some("child".into());
        app.handle_key_agents_overlay(KeyCode::Char('x'), KeyModifiers::NONE);
        assert_eq!(
            app.spawn_tree.nodes.get("child").map(|n| n.status),
            Some(SubagentStatus::Interrupted)
        );
        assert_ne!(
            app.spawn_tree.nodes.get("leaf").map(|n| n.status),
            Some(SubagentStatus::Interrupted)
        );

        // `X` on `root` interrupts the whole subtree (including leaf).
        app.agents_cursor = Some("root".into());
        app.handle_key_agents_overlay(KeyCode::Char('X'), KeyModifiers::NONE);
        assert_eq!(
            app.spawn_tree.nodes.get("root").map(|n| n.status),
            Some(SubagentStatus::Interrupted)
        );
        assert_eq!(
            app.spawn_tree.nodes.get("leaf").map(|n| n.status),
            Some(SubagentStatus::Interrupted)
        );
    }

    #[test]
    fn agents_overlay_s_cycles_sort_modes() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        assert_eq!(app.agents_sort, AgentsSortMode::DepthFirst);
        app.handle_key_agents_overlay(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.agents_sort, AgentsSortMode::ToolsDesc);
        app.handle_key_agents_overlay(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.agents_sort, AgentsSortMode::DurationDesc);
        app.handle_key_agents_overlay(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.agents_sort, AgentsSortMode::Status);
        app.handle_key_agents_overlay(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(app.agents_sort, AgentsSortMode::DepthFirst);
    }

    #[test]
    fn agents_overlay_f_cycles_filter_modes() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        assert_eq!(app.agents_filter, AgentsFilterMode::All);
        app.handle_key_agents_overlay(KeyCode::Char('f'), KeyModifiers::NONE);
        assert_eq!(app.agents_filter, AgentsFilterMode::Running);
        app.handle_key_agents_overlay(KeyCode::Char('f'), KeyModifiers::NONE);
        assert_eq!(app.agents_filter, AgentsFilterMode::Failed);
        app.handle_key_agents_overlay(KeyCode::Char('f'), KeyModifiers::NONE);
        assert_eq!(app.agents_filter, AgentsFilterMode::Leaf);
        app.handle_key_agents_overlay(KeyCode::Char('f'), KeyModifiers::NONE);
        assert_eq!(app.agents_filter, AgentsFilterMode::All);
    }

    #[test]
    fn agents_overlay_filter_running_hides_completed_nodes() {
        use crate::tui::spawn_tree::SubagentStatus;
        let mut app = App::new();
        app.show_agents_overlay = true;
        app.spawn_tree.on_spawn(spawn("a", None));
        app.spawn_tree.on_spawn(spawn("b", None));
        app.spawn_tree.on_start("a");
        app.spawn_tree.on_complete(crate::agent::callbacks::SubagentComplete {
            id: "a".into(),
            success: true,
            ..Default::default()
        });
        // a is completed, b is queued.
        app.agents_filter = AgentsFilterMode::Running;
        let flat = app.agents_flat_node_ids();
        assert_eq!(flat, vec!["b".to_string()]);
        // Sanity: a's status is terminal.
        assert_eq!(
            app.spawn_tree.nodes.get("a").map(|n| n.status),
            Some(SubagentStatus::Completed)
        );
    }

    #[test]
    fn agents_overlay_history_step_moves_through_snapshots() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        // Push 2 snapshots into history.
        let mut t1 = crate::tui::spawn_tree::SpawnTree::new();
        t1.on_spawn(spawn("s1", None));
        app.spawn_history.push(t1);
        let mut t2 = crate::tui::spawn_tree::SpawnTree::new();
        t2.on_spawn(spawn("s2", None));
        app.spawn_history.push(t2);
        assert_eq!(app.spawn_history.len(), 2);
        assert_eq!(app.agents_history_index, 0);

        app.handle_key_agents_overlay(KeyCode::Char('['), KeyModifiers::NONE);
        assert_eq!(app.agents_history_index, 1);
        app.handle_key_agents_overlay(KeyCode::Char('['), KeyModifiers::NONE);
        assert_eq!(app.agents_history_index, 2);
        // Clamps at history length.
        app.handle_key_agents_overlay(KeyCode::Char('['), KeyModifiers::NONE);
        assert_eq!(app.agents_history_index, 2);
        app.handle_key_agents_overlay(KeyCode::Char(']'), KeyModifiers::NONE);
        assert_eq!(app.agents_history_index, 1);
    }

    #[test]
    fn agents_overlay_replay_mode_disables_kill_keys() {
        use crate::tui::spawn_tree::SubagentStatus;
        let mut app = App::new();
        app.show_agents_overlay = true;
        let mut t = crate::tui::spawn_tree::SpawnTree::new();
        t.on_spawn(spawn("a", None));
        app.spawn_history.push(t);
        app.agents_history_index = 1;
        // Cursor on the history-tree node.
        app.agents_cursor = Some("a".into());
        app.handle_key_agents_overlay(KeyCode::Char('x'), KeyModifiers::NONE);
        // History snapshot's node must remain unchanged
        // (interrupt should NOT touch it in replay mode).
        let history_node_status = app
            .spawn_history
            .get(0)
            .unwrap()
            .tree
            .nodes
            .get("a")
            .map(|n| n.status);
        assert_eq!(history_node_status, Some(SubagentStatus::Queued));
        // And a flash was set explaining why.
        let flash = app.agents_flash.as_ref().map(|(b, _)| b.clone()).unwrap();
        assert!(flash.contains("replay mode"));
    }

    #[test]
    fn agents_overlay_detail_focus_toggles_and_routes_scroll_keys() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        assert!(!app.agents_detail_focused);
        app.handle_key_agents_overlay(KeyCode::Enter, KeyModifiers::NONE);
        assert!(app.agents_detail_focused);
        // While focused, `j` scrolls instead of navigating list.
        app.handle_key_agents_overlay(KeyCode::Char('j'), KeyModifiers::NONE);
        assert_eq!(app.agents_detail_scroll, 2);
        app.handle_key_agents_overlay(KeyCode::PageDown, KeyModifiers::NONE);
        assert_eq!(app.agents_detail_scroll, 12);
        // `h` exits detail focus + resets scroll.
        app.handle_key_agents_overlay(KeyCode::Char('h'), KeyModifiers::NONE);
        assert!(!app.agents_detail_focused);
        assert_eq!(app.agents_detail_scroll, 0);
        // Re-enter + Esc behaviour: Esc unfocuses (does NOT close).
        app.handle_key_agents_overlay(KeyCode::Char('l'), KeyModifiers::NONE);
        assert!(app.agents_detail_focused);
        app.handle_key_agents_overlay(KeyCode::Esc, KeyModifiers::NONE);
        assert!(!app.agents_detail_focused);
        assert!(app.show_agents_overlay, "esc from detail should NOT close overlay");
        // Esc again now closes since we're in list mode.
        app.handle_key_agents_overlay(KeyCode::Esc, KeyModifiers::NONE);
        assert!(!app.show_agents_overlay);
    }

    #[test]
    fn agents_overlay_p_toggles_delegation_pause_when_registry_attached() {
        let mut app = App::new();
        app.show_agents_overlay = true;
        let reg = crate::agent::DelegationRegistry::default();
        app.delegation_registry = Some(reg.clone());
        assert!(!reg.is_paused());
        app.handle_key_agents_overlay(KeyCode::Char('p'), KeyModifiers::NONE);
        assert!(reg.is_paused());
        let flash = app.agents_flash.as_ref().map(|(b, _)| b.clone()).unwrap();
        assert_eq!(flash, "spawning paused");
        app.handle_key_agents_overlay(KeyCode::Char('p'), KeyModifiers::NONE);
        assert!(!reg.is_paused());
    }
}
