//! Integration tests for the per-turn tool-loop guardrails.
//!
//! The guard is failure- and result-aware (ported from the upstream's
//! controller): it reacts to repeated FAILED calls and to read-only
//! calls returning IDENTICAL results — never to mere repetition, which
//! the previous detector punished even when every call succeeded and
//! made progress.

use serde_json::json;

use fennec::agent::loop_::{GuardAction, LoopGuard, LoopGuardConfig};

fn hard_stop_guard() -> LoopGuard {
    LoopGuard::new(LoopGuardConfig {
        hard_stop_enabled: true,
        ..LoopGuardConfig::default()
    })
}

#[test]
fn normal_mixed_sequence_stays_allowed() {
    let mut g = hard_stop_guard();
    assert_eq!(
        g.after_call("read_file", &json!({"path": "/a.txt"}), "contents a", false, true),
        GuardAction::Allow
    );
    assert_eq!(
        g.after_call("shell", &json!({"cmd": "ls"}), "files", false, false),
        GuardAction::Allow
    );
    assert_eq!(
        g.after_call("write_file", &json!({"path": "/b.txt"}), "ok", false, false),
        GuardAction::Allow
    );
}

#[test]
fn repeated_identical_failures_warn_then_block() {
    let mut g = hard_stop_guard();
    let args = json!({"path": "/a.txt"});
    assert_eq!(g.after_call("patch", &args, "no match", true, false), GuardAction::Allow);
    match g.after_call("patch", &args, "no match", true, false) {
        GuardAction::Warn(msg) => {
            assert!(msg.contains("patch"), "message should name the tool: {msg}");
            assert!(msg.contains('2'), "message should mention count: {msg}");
        }
        other => panic!("expected Warn, got {other:?}"),
    }
    for _ in 0..3 {
        g.after_call("patch", &args, "no match", true, false);
    }
    assert!(matches!(g.before_call("patch", &args), GuardAction::Block(_)));
    assert!(g.halt_message().is_some());
}

#[test]
fn successful_repetition_never_triggers() {
    // The old detector's headline false positive: re-reading a file
    // after edits / re-running a passing command tripped it. The
    // guard only reacts to failures and identical read-only results.
    let mut g = hard_stop_guard();
    let args = json!({"path": "/a.txt"});
    for i in 0..12 {
        let result = format!("contents v{i}");
        assert_eq!(
            g.after_call("read_file", &args, &result, false, true),
            GuardAction::Allow,
            "changing results must stay allowed"
        );
    }
    assert_eq!(g.before_call("read_file", &args), GuardAction::Allow);
}

#[test]
fn identical_readonly_results_block_in_hard_stop_mode() {
    let mut g = hard_stop_guard();
    let args = json!({"q": "same search"});
    for _ in 0..5 {
        g.after_call("web_search", &args, "same ten results", false, true);
    }
    assert!(matches!(g.before_call("web_search", &args), GuardAction::Block(_)));
}

#[test]
fn warnings_only_by_default_no_blocks() {
    let mut g = LoopGuard::default();
    let args = json!({"path": "/a.txt"});
    for _ in 0..10 {
        g.after_call("patch", &args, "no match", true, false);
    }
    assert_eq!(g.before_call("patch", &args), GuardAction::Allow);
    assert!(g.halt_message().is_none());
}

#[test]
fn same_tool_failures_across_args_halt_in_hard_stop_mode() {
    let mut g = hard_stop_guard();
    let mut halted = false;
    for i in 0..9 {
        if matches!(
            g.after_call("terminal", &json!({"cmd": format!("try {i}")}), "exit 1", true, false),
            GuardAction::Halt(_)
        ) {
            halted = true;
            break;
        }
    }
    assert!(halted, "8 same-tool failures must halt in hard-stop mode");
}
