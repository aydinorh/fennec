//! Shared helpers for building safe SQLite FTS5 MATCH expressions.
//!
//! FTS5's MATCH grammar is a mini-DSL. Bare words are terms, quoted
//! strings are phrases, and a handful of characters carry syntactic
//! meaning: `*` (prefix wildcard), `+` / `-` (required / negated),
//! `AND` / `OR` / `NOT` / `NEAR` (operators), `:` (column qualifier),
//! `(`/`)` (grouping), `"` (phrase delimiter).
//!
//! Passing user input verbatim into a MATCH expression therefore risks
//! two problems:
//!
//! 1. **Syntax errors**: an input containing `"` breaks out of the
//!    `"<word>"` quoting Fennec uses everywhere. Users searching for
//!    `foo"bar` previously hit an opaque `"fts5: syntax error"` which
//!    bubbled up to the agent as a generic "search failed". Same for
//!    unmatched `(`, trailing `-`, etc.
//! 2. **Query-intent override** (minor): operators in user input let
//!    the caller craft a query the caller's outer intent didn't
//!    authorize — e.g. a user searching for "foo" accidentally running
//!    a prefix wildcard by including `*`.
//!
//! [`build_match_query`] tokenises user input on whitespace, strips the
//! FTS5-special characters from each token, drops empty tokens, and
//! OR-joins the surviving quoted phrases. The result is either a safe
//! MATCH expression or `None` if no usable tokens remained.

/// Build a safe FTS5 MATCH expression from free-form user input,
/// OR-joining the tokens (ANY token matches). Used by memory recall and
/// the collective cache, where forgiving multi-word recall beats
/// precision.
///
/// Returns `None` if the input produces no usable tokens — callers
/// should treat this as "no matches" rather than running the query.
pub fn build_match_query(query: &str) -> Option<String> {
    join_sanitized(query, " OR ")
}

/// Build a safe FTS5 MATCH expression AND-joining the tokens (ALL
/// tokens must match). Used by session search: a user searching
/// "docker timeout" wants the conversation about docker timeouts, not
/// every session mentioning either word.
pub fn build_match_query_all(query: &str) -> Option<String> {
    join_sanitized(query, " AND ")
}

fn join_sanitized(query: &str, op: &str) -> Option<String> {
    let parts: Vec<String> = query
        .split_whitespace()
        .filter_map(sanitize_token)
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(op))
    }
}

/// Sanitise one whitespace-separated token into an FTS5-safe quoted
/// phrase, or `None` if the token is empty after stripping specials.
///
/// Stripped characters: `"` (breaks out of the quoted phrase), the
/// FTS5 operator set `* + - : ( )`, and `^` (column-rank prefix).
fn sanitize_token(word: &str) -> Option<String> {
    let cleaned: String = word
        .chars()
        .filter(|c| !matches!(*c, '"' | '*' | '+' | '-' | ':' | '(' | ')' | '^'))
        .collect();
    if cleaned.is_empty() {
        None
    } else {
        Some(format!("\"{}\"", cleaned))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_words_are_or_joined() {
        assert_eq!(
            build_match_query("foo bar"),
            Some("\"foo\" OR \"bar\"".into())
        );
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(build_match_query("").is_none());
        assert!(build_match_query("   ").is_none());
        assert!(build_match_query_all("").is_none());
    }

    /// Session search AND-joins: every term must match. (Upstream's
    /// session search defaults to AND; OR made multi-word searches
    /// return every session mentioning any single word.)
    #[test]
    fn all_variant_and_joins() {
        assert_eq!(
            build_match_query_all("docker timeout"),
            Some("\"docker\" AND \"timeout\"".into())
        );
        // Same sanitization as the OR variant.
        assert_eq!(
            build_match_query_all("foo\"bar baz*"),
            Some("\"foobar\" AND \"baz\"".into())
        );
    }

    /// Regression for the headline FTS5 injection bug: `foo"bar`
    /// used to render as `"foo"bar"` which is an FTS5 syntax error.
    /// Now becomes `"foobar"`.
    #[test]
    fn strips_inner_double_quote() {
        let q = build_match_query("foo\"bar").unwrap();
        assert!(!q.contains("\"bar\""));
        assert_eq!(q, "\"foobar\"");
    }

    /// Regression: FTS5 operator characters in user input used to be
    /// passed through, enabling unintended wildcards / negation /
    /// column qualifiers / grouping in the final MATCH expression.
    #[test]
    fn strips_fts5_operator_chars() {
        let q = build_match_query("foo* bar+ -baz col:val (x) y^2").unwrap();
        for bad in ['*', '+', '(', ')', ':', '^'] {
            assert!(!q.contains(bad), "query still contains '{}': {}", bad, q);
        }
        // `-` is inside tokens but also a valid leading negation — we
        // strip it in both positions for simplicity.
        assert!(!q.contains(" -"));
        // Real tokens still land in the output.
        assert!(q.contains("\"foo\""));
        assert!(q.contains("\"bar\""));
        assert!(q.contains("\"baz\""));
    }

    #[test]
    fn tokens_that_become_empty_are_dropped() {
        let q = build_match_query("foo --- bar").unwrap();
        assert_eq!(q, "\"foo\" OR \"bar\"");
    }

    #[test]
    fn all_operator_input_returns_none() {
        assert!(build_match_query("*** --- +++").is_none());
        assert!(build_match_query("\"\"\"").is_none());
    }

    #[test]
    fn unicode_tokens_are_preserved() {
        let q = build_match_query("日本 語検索").unwrap();
        assert!(q.contains("日本"));
        assert!(q.contains("語検索"));
    }
}
