//! Per-turn tool-call loop guardrails.
//!
//! Ported from the upstream's pure guardrail controller: the controller
//! tracks per-turn observations and returns decisions; the agent loop
//! owns whether those decisions become warning guidance appended to
//! tool results, synthetic failed results, or a controlled turn halt.
//!
//! Three patterns, all FAILURE- or RESULT-aware (the previous detector
//! in this module counted identical *calls* regardless of outcome,
//! which false-positived on legitimate repetition — re-reading a file
//! after edits, polling a process whose status changes):
//!
//! - **Exact failure repeat**: the same tool called with identical
//!   arguments keeps FAILING. Warn after 2, block after 5.
//! - **Same-tool failure**: one tool keeps failing this turn across
//!   different arguments. Warn after 3, halt the turn after 8.
//! - **Idempotent no-progress**: a read-only call repeated with
//!   identical arguments returns the IDENTICAL result. Warn after 2,
//!   block after 5. A success with a different result resets the
//!   counter, so polling that observes change never triggers.
//!
//! Successes clear the failure counters for that signature/tool.
//! Warnings are on by default and never prevent execution; hard stops
//! (block/halt) are explicit config opt-in so interactive sessions get
//! a nudge unless the user enables circuit-breaker behavior.
//!
//! Divergence from the upstream (deliberate, an improvement): the
//! idempotent set is derived from each tool's own `is_read_only()`
//! declaration instead of a hardcoded name list, so new tools are
//! covered automatically.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Thresholds for per-turn tool-call loop detection. Mirrors the
/// upstream's defaults.
#[derive(Debug, Clone)]
pub struct LoopGuardConfig {
    /// Append warning guidance to tool results (never blocks).
    pub warnings_enabled: bool,
    /// Enable block/halt circuit-breaker behavior. Off by default.
    pub hard_stop_enabled: bool,
    pub exact_failure_warn_after: u32,
    pub exact_failure_block_after: u32,
    pub same_tool_failure_warn_after: u32,
    pub same_tool_failure_halt_after: u32,
    pub no_progress_warn_after: u32,
    pub no_progress_block_after: u32,
}

impl Default for LoopGuardConfig {
    fn default() -> Self {
        Self {
            warnings_enabled: true,
            hard_stop_enabled: false,
            exact_failure_warn_after: 2,
            exact_failure_block_after: 5,
            same_tool_failure_warn_after: 3,
            same_tool_failure_halt_after: 8,
            no_progress_warn_after: 2,
            no_progress_block_after: 5,
        }
    }
}

/// What the guard wants the agent loop to do with a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardAction {
    /// Execute normally.
    Allow,
    /// Execute, but append this guidance to the tool result.
    Warn(String),
    /// Do NOT execute; feed this message back as a failed tool result.
    Block(String),
    /// Stop the whole turn after this batch; surface this message.
    Halt(String),
}

impl GuardAction {
    pub fn allows_execution(&self) -> bool {
        matches!(self, GuardAction::Allow | GuardAction::Warn(_))
    }
}

/// Stable identity for a tool name plus canonical arguments.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CallSignature {
    tool_name: String,
    args_hash: u64,
}

impl CallSignature {
    fn new(tool_name: &str, args: &serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            args_hash: hash_value(&canonical_args(args)),
        }
    }
}

/// Canonicalize arguments: objects get sorted keys so `{a,b}` and
/// `{b,a}` produce the same signature (serde_json's Value preserves
/// insertion order unless the `preserve_order` feature is off; sort
/// explicitly to be independent of that).
fn canonical_args(args: &serde_json::Value) -> String {
    fn sort(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
                entries.sort_by_key(|(k, _)| k.as_str().to_string());
                let mut out = serde_json::Map::new();
                for (k, val) in entries {
                    out.insert(k.clone(), sort(val));
                }
                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(sort).collect())
            }
            other => other.clone(),
        }
    }
    sort(args).to_string()
}

fn hash_value(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Per-turn controller for repeated failed / non-progressing tool calls.
pub struct LoopGuard {
    config: LoopGuardConfig,
    exact_failure_counts: HashMap<CallSignature, u32>,
    same_tool_failure_counts: HashMap<String, u32>,
    /// signature → (result hash, consecutive identical-result count).
    no_progress: HashMap<CallSignature, (u64, u32)>,
    halted: Option<String>,
}

impl LoopGuard {
    pub fn new(config: LoopGuardConfig) -> Self {
        Self {
            config,
            exact_failure_counts: HashMap::new(),
            same_tool_failure_counts: HashMap::new(),
            no_progress: HashMap::new(),
            halted: None,
        }
    }

    /// Clear all per-turn state. Call at the start of every turn.
    pub fn reset_for_turn(&mut self) {
        self.exact_failure_counts.clear();
        self.same_tool_failure_counts.clear();
        self.no_progress.clear();
        self.halted = None;
    }

    /// The halt message, when a halt decision fired this turn.
    pub fn halt_message(&self) -> Option<&str> {
        self.halted.as_deref()
    }

    /// Check BEFORE executing a tool call. Only ever blocks when
    /// `hard_stop_enabled`.
    pub fn before_call(&mut self, tool_name: &str, args: &serde_json::Value) -> GuardAction {
        if !self.config.hard_stop_enabled {
            return GuardAction::Allow;
        }
        let signature = CallSignature::new(tool_name, args);

        let exact = self.exact_failure_counts.get(&signature).copied().unwrap_or(0);
        if exact >= self.config.exact_failure_block_after {
            let msg = format!(
                "Blocked {tool_name}: the same tool call failed {exact} times with \
                 identical arguments. Stop retrying it unchanged; change strategy \
                 or explain the blocker."
            );
            self.halted = Some(msg.clone());
            return GuardAction::Block(msg);
        }

        if let Some((_hash, repeats)) = self.no_progress.get(&signature) {
            if *repeats >= self.config.no_progress_block_after {
                let msg = format!(
                    "Blocked {tool_name}: this read-only call returned the same result \
                     {repeats} times. Stop repeating it unchanged; use the result \
                     already provided or try a different query."
                );
                self.halted = Some(msg.clone());
                return GuardAction::Block(msg);
            }
        }

        GuardAction::Allow
    }

    /// Record the outcome AFTER executing a tool call.
    ///
    /// `failed` comes from the tool's structured success flag;
    /// `read_only` from the tool's own `is_read_only()` declaration.
    pub fn after_call(
        &mut self,
        tool_name: &str,
        args: &serde_json::Value,
        result: &str,
        failed: bool,
        read_only: bool,
    ) -> GuardAction {
        let signature = CallSignature::new(tool_name, args);

        if failed {
            let exact = self.exact_failure_counts.entry(signature.clone()).or_insert(0);
            *exact += 1;
            let exact = *exact;
            self.no_progress.remove(&signature);

            let same = self
                .same_tool_failure_counts
                .entry(tool_name.to_string())
                .or_insert(0);
            *same += 1;
            let same = *same;

            if self.config.hard_stop_enabled && same >= self.config.same_tool_failure_halt_after {
                let msg = format!(
                    "Stopped {tool_name}: it failed {same} times this turn. Stop \
                     retrying the same failing tool path and choose a different approach."
                );
                self.halted = Some(msg.clone());
                return GuardAction::Halt(msg);
            }
            if self.config.warnings_enabled && exact >= self.config.exact_failure_warn_after {
                return GuardAction::Warn(format!(
                    "{tool_name} has failed {exact} times with identical arguments. \
                     This looks like a loop; inspect the error and change strategy \
                     instead of retrying it unchanged."
                ));
            }
            if self.config.warnings_enabled && same >= self.config.same_tool_failure_warn_after {
                return GuardAction::Warn(format!(
                    "{tool_name} has failed {same} times this turn (different \
                     arguments). Consider a different tool or approach."
                ));
            }
            return GuardAction::Allow;
        }

        // Success clears the failure counters.
        self.exact_failure_counts.remove(&signature);
        self.same_tool_failure_counts.remove(tool_name);

        if !read_only {
            self.no_progress.remove(&signature);
            return GuardAction::Allow;
        }

        // Idempotent no-progress: identical call returning the identical
        // result. A different result resets the streak, so polling that
        // observes change never triggers.
        let result_hash = hash_value(result);
        let repeats = match self.no_progress.get(&signature) {
            Some((prev_hash, prev_count)) if *prev_hash == result_hash => prev_count + 1,
            _ => 1,
        };
        self.no_progress.insert(signature, (result_hash, repeats));

        if self.config.warnings_enabled && repeats >= self.config.no_progress_warn_after {
            return GuardAction::Warn(format!(
                "{tool_name} returned the identical result {repeats} times for the \
                 same arguments. The answer is already above — use it, or change \
                 the query."
            ));
        }
        GuardAction::Allow
    }
}

impl Default for LoopGuard {
    fn default() -> Self {
        Self::new(LoopGuardConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn hard_stop_guard() -> LoopGuard {
        LoopGuard::new(LoopGuardConfig {
            hard_stop_enabled: true,
            ..LoopGuardConfig::default()
        })
    }

    #[test]
    fn repeated_exact_failures_warn_then_block() {
        let mut g = hard_stop_guard();
        let args = json!({"path": "/x"});
        // First failure: allow. Second: warn.
        assert_eq!(g.after_call("patch", &args, "err", true, false), GuardAction::Allow);
        assert!(matches!(
            g.after_call("patch", &args, "err", true, false),
            GuardAction::Warn(_)
        ));
        for _ in 0..3 {
            g.after_call("patch", &args, "err", true, false);
        }
        // 5 exact failures recorded → before_call blocks + sets halt.
        assert!(matches!(g.before_call("patch", &args), GuardAction::Block(_)));
        assert!(g.halt_message().is_some());
    }

    #[test]
    fn success_resets_failure_counters() {
        let mut g = hard_stop_guard();
        let args = json!({"path": "/x"});
        g.after_call("patch", &args, "err", true, false);
        g.after_call("patch", &args, "ok", false, false);
        // Counter was cleared — next failure is count 1 again (allow).
        assert_eq!(g.after_call("patch", &args, "err", true, false), GuardAction::Allow);
    }

    #[test]
    fn same_tool_failures_across_args_halt() {
        let mut g = hard_stop_guard();
        for i in 0..7 {
            let action = g.after_call("terminal", &json!({"i": i}), "err", true, false);
            assert!(!matches!(action, GuardAction::Halt(_)), "halted early at {i}");
        }
        assert!(matches!(
            g.after_call("terminal", &json!({"i": 99}), "err", true, false),
            GuardAction::Halt(_)
        ));
    }

    #[test]
    fn identical_readonly_results_warn_then_block() {
        let mut g = hard_stop_guard();
        let args = json!({"path": "/x"});
        assert_eq!(
            g.after_call("read_file", &args, "same body", false, true),
            GuardAction::Allow
        );
        assert!(matches!(
            g.after_call("read_file", &args, "same body", false, true),
            GuardAction::Warn(_)
        ));
        for _ in 0..3 {
            g.after_call("read_file", &args, "same body", false, true);
        }
        assert!(matches!(
            g.before_call("read_file", &args),
            GuardAction::Block(_)
        ));
    }

    /// Polling that observes change must never trigger: a different
    /// result resets the no-progress streak.
    #[test]
    fn changing_results_reset_no_progress() {
        let mut g = hard_stop_guard();
        let args = json!({"id": "proc1"});
        for i in 0..10 {
            let action = g.after_call("read_file", &args, &format!("status {i}"), false, true);
            assert_eq!(action, GuardAction::Allow, "changing results must not warn");
        }
    }

    /// Repeated identical MUTATING calls that succeed never trigger —
    /// the old detector's headline false positive.
    #[test]
    fn repeated_successful_mutations_are_fine() {
        let mut g = hard_stop_guard();
        let args = json!({"path": "/x", "content": "y"});
        for _ in 0..10 {
            assert_eq!(
                g.after_call("write_file", &args, "ok", false, false),
                GuardAction::Allow
            );
        }
        assert_eq!(g.before_call("write_file", &args), GuardAction::Allow);
    }

    #[test]
    fn defaults_never_block_only_warn() {
        let mut g = LoopGuard::default(); // hard_stop_enabled = false
        let args = json!({"path": "/x"});
        for _ in 0..20 {
            g.after_call("patch", &args, "err", true, false);
        }
        // Plenty of failures, but blocks/halts require the opt-in.
        assert!(matches!(g.before_call("patch", &args), GuardAction::Allow));
        assert!(g.halt_message().is_none());
        // Warnings still fire.
        assert!(matches!(
            g.after_call("patch", &args, "err", true, false),
            GuardAction::Warn(_)
        ));
    }

    #[test]
    fn args_key_order_does_not_change_signature() {
        let mut g = hard_stop_guard();
        let a = json!({"x": 1, "y": 2});
        let b = json!({"y": 2, "x": 1});
        g.after_call("patch", &a, "err", true, false);
        assert!(matches!(
            g.after_call("patch", &b, "err", true, false),
            GuardAction::Warn(_)
        ), "key order must not split the failure count");
    }

    #[test]
    fn reset_clears_everything() {
        let mut g = hard_stop_guard();
        let args = json!({"path": "/x"});
        for _ in 0..6 {
            g.after_call("patch", &args, "err", true, false);
        }
        g.reset_for_turn();
        assert_eq!(g.before_call("patch", &args), GuardAction::Allow);
        assert!(g.halt_message().is_none());
    }
}
