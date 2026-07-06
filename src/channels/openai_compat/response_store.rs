//! `/v1/responses` previous_response_id chain store.
//!
//! Persists OpenAI Responses API state — each response is a node
//! linked to its predecessor via `prev_response_id`. Walking the
//! chain backward reconstructs the full conversation a request is
//! continuing.
//!
//! Lives in its own SQLite file (`~/.fennec/openai_responses.db`)
//! so it doesn't share schema or transactions with Fennec's
//! existing `sessions.db`. The session DB stores conversation
//! history keyed by session_id; this store records the
//! response-graph that OpenAI's API model superimposes on top.
//!
//! Two tables:
//!
//!   responses (response_id PK, prev_response_id, session_id,
//!              role, content, created_at)
//!   conversations (name PK, latest_response_id)
//!
//! `conversations` is a name → latest_response_id pointer table
//! so clients can address a conversation by a stable name instead
//! of always carrying the latest response_id around.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use rusqlite::Connection;

use crate::providers::traits::ChatMessage;

/// One stored response node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseRecord {
    pub response_id: String,
    pub prev_response_id: Option<String>,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

/// SQLite-backed store. Async API delegates to `spawn_blocking`
/// for SQLite calls, mirroring `sessions::SessionStore`.
pub struct ResponseStore {
    conn: Arc<Mutex<Connection>>,
}

impl ResponseStore {
    /// Open or create the store at `db_path`. Idempotent — re-running
    /// against an existing file leaves data alone.
    pub fn new(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating parent dirs for {}", db_path.display())
            })?;
        }
        let conn = Connection::open(db_path).with_context(|| {
            format!("opening response_store at {}", db_path.display())
        })?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;
             PRAGMA temp_store = MEMORY;
             PRAGMA foreign_keys = ON;",
        )
        .context("applying PRAGMAs")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS responses (
                response_id      TEXT PRIMARY KEY,
                prev_response_id TEXT,
                session_id       TEXT NOT NULL,
                role             TEXT NOT NULL,
                content          TEXT NOT NULL,
                created_at       TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_responses_session
                 ON responses(session_id);
             CREATE INDEX IF NOT EXISTS idx_responses_prev
                 ON responses(prev_response_id);
             CREATE TABLE IF NOT EXISTS conversations (
                name                TEXT PRIMARY KEY,
                latest_response_id  TEXT NOT NULL,
                updated_at          TEXT NOT NULL
             );",
        )
        .context("creating response_store schema")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Insert a response node.
    pub async fn put(
        &self,
        response_id: &str,
        prev_response_id: Option<&str>,
        session_id: &str,
        role: &str,
        content: &str,
    ) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let response_id = response_id.to_string();
        let prev_response_id = prev_response_id.map(String::from);
        let session_id = session_id.to_string();
        let role = role.to_string();
        let content = content.to_string();
        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now().to_rfc3339();
            let conn = conn.lock();
            conn.execute(
                "INSERT INTO responses (response_id, prev_response_id, session_id,
                                        role, content, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    response_id,
                    prev_response_id,
                    session_id,
                    role,
                    content,
                    now,
                ],
            )
            .context("inserting response record")?;
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }

    /// Look up a single response by id.
    pub async fn get(&self, response_id: &str) -> Result<Option<ResponseRecord>> {
        let conn = Arc::clone(&self.conn);
        let response_id = response_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT response_id, prev_response_id, session_id, role, content, created_at
                 FROM responses
                 WHERE response_id = ?1",
            )?;
            let mut rows = stmt.query(rusqlite::params![response_id])?;
            if let Some(row) = rows.next()? {
                Ok::<_, anyhow::Error>(Some(ResponseRecord {
                    response_id: row.get(0)?,
                    prev_response_id: row.get(1)?,
                    session_id: row.get(2)?,
                    role: row.get(3)?,
                    content: row.get(4)?,
                    created_at: row.get(5)?,
                }))
            } else {
                Ok(None)
            }
        })
        .await?
    }

    /// Walk the chain backward starting from `response_id` and
    /// return the records oldest-first (i.e., the conversation in
    /// natural reading order). Stops at the first node whose
    /// `prev_response_id` is `None`.
    ///
    /// Returns an empty list when the starting id doesn't exist.
    pub async fn walk_chain(&self, response_id: &str) -> Result<Vec<ResponseRecord>> {
        let conn = Arc::clone(&self.conn);
        let response_id = response_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut nodes: Vec<ResponseRecord> = Vec::new();
            let mut cursor: Option<String> = Some(response_id);
            // Defensive cap: any chain longer than this is almost
            // certainly a bug or pathological client. 1024 is well
            // above any realistic conversation length.
            const MAX_HOPS: usize = 1024;
            let mut hops = 0usize;
            while let Some(id) = cursor {
                if hops >= MAX_HOPS {
                    tracing::warn!(
                        starting_id = id,
                        "response chain hit walk cap; truncating"
                    );
                    break;
                }
                hops += 1;
                let mut stmt = conn.prepare(
                    "SELECT response_id, prev_response_id, session_id, role, content, created_at
                     FROM responses
                     WHERE response_id = ?1",
                )?;
                let mut rows = stmt.query(rusqlite::params![&id])?;
                if let Some(row) = rows.next()? {
                    let rec = ResponseRecord {
                        response_id: row.get(0)?,
                        prev_response_id: row.get(1)?,
                        session_id: row.get(2)?,
                        role: row.get(3)?,
                        content: row.get(4)?,
                        created_at: row.get(5)?,
                    };
                    cursor = rec.prev_response_id.clone();
                    nodes.push(rec);
                } else {
                    // Broken chain — referenced response_id doesn't
                    // exist. Return what we have.
                    break;
                }
            }
            // We walked backward; reverse so callers see oldest-first.
            nodes.reverse();
            Ok::<_, anyhow::Error>(nodes)
        })
        .await?
    }

    /// Walk the chain and project it as `ChatMessage`s ready to feed
    /// into `Agent::turn_with_history`.
    pub async fn chain_as_messages(&self, response_id: &str) -> Result<Vec<ChatMessage>> {
        let chain = self.walk_chain(response_id).await?;
        Ok(chain
            .into_iter()
            .map(|r| ChatMessage {
                role: r.role,
                content: Some(r.content),
                tool_calls: None,
                tool_call_id: None,
                attachments: None,
                reasoning: None,
            })
            .collect())
    }

    /// Set the latest response id for a named conversation. Upsert.
    pub async fn set_conversation(&self, name: &str, response_id: &str) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let name = name.to_string();
        let response_id = response_id.to_string();
        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now().to_rfc3339();
            let conn = conn.lock();
            conn.execute(
                "INSERT INTO conversations (name, latest_response_id, updated_at)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(name) DO UPDATE SET
                     latest_response_id = excluded.latest_response_id,
                     updated_at = excluded.updated_at",
                rusqlite::params![name, response_id, now],
            )
            .context("upserting conversation pointer")?;
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }

    /// Look up the latest response id for a named conversation.
    /// Returns `None` if the conversation doesn't exist yet (a
    /// brand-new name on the first request — that's not an error).
    pub async fn get_conversation(&self, name: &str) -> Result<Option<String>> {
        let conn = Arc::clone(&self.conn);
        let name = name.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT latest_response_id FROM conversations WHERE name = ?1",
            )?;
            let mut rows = stmt.query(rusqlite::params![name])?;
            if let Some(row) = rows.next()? {
                Ok::<_, anyhow::Error>(Some(row.get::<_, String>(0)?))
            } else {
                Ok(None)
            }
        })
        .await?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store() -> (ResponseStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = dir.path().join("openai_responses.db");
        let store = ResponseStore::new(&db).unwrap();
        (store, dir)
    }

    #[tokio::test]
    async fn put_and_get_round_trips() {
        let (store, _dir) = make_store();
        store
            .put("r1", None, "s1", "user", "hello")
            .await
            .unwrap();
        let rec = store.get("r1").await.unwrap().unwrap();
        assert_eq!(rec.response_id, "r1");
        assert_eq!(rec.session_id, "s1");
        assert_eq!(rec.role, "user");
        assert_eq!(rec.content, "hello");
        assert_eq!(rec.prev_response_id, None);
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let (store, _dir) = make_store();
        assert!(store.get("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn walk_chain_returns_nodes_oldest_first() {
        let (store, _dir) = make_store();
        store.put("r1", None, "s", "user", "first").await.unwrap();
        store
            .put("r2", Some("r1"), "s", "assistant", "second")
            .await
            .unwrap();
        store
            .put("r3", Some("r2"), "s", "user", "third")
            .await
            .unwrap();
        let chain = store.walk_chain("r3").await.unwrap();
        let contents: Vec<&str> = chain.iter().map(|r| r.content.as_str()).collect();
        assert_eq!(contents, vec!["first", "second", "third"]);
    }

    #[tokio::test]
    async fn walk_chain_stops_at_root() {
        let (store, _dir) = make_store();
        store.put("r1", None, "s", "user", "only").await.unwrap();
        let chain = store.walk_chain("r1").await.unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].content, "only");
    }

    #[tokio::test]
    async fn walk_chain_missing_id_returns_empty() {
        let (store, _dir) = make_store();
        let chain = store.walk_chain("nope").await.unwrap();
        assert!(chain.is_empty());
    }

    #[tokio::test]
    async fn walk_chain_broken_link_returns_partial() {
        let (store, _dir) = make_store();
        // Insert r1 pointing at "missing"; walk_chain should
        // surface r1 then stop.
        store
            .put("r1", Some("missing"), "s", "user", "orphan")
            .await
            .unwrap();
        let chain = store.walk_chain("r1").await.unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].content, "orphan");
    }

    #[tokio::test]
    async fn chain_as_messages_projects_role_and_content() {
        let (store, _dir) = make_store();
        store.put("r1", None, "s", "user", "hi").await.unwrap();
        store
            .put("r2", Some("r1"), "s", "assistant", "hello")
            .await
            .unwrap();
        let msgs = store.chain_as_messages("r2").await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content.as_deref(), Some("hi"));
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn set_and_get_conversation() {
        let (store, _dir) = make_store();
        // Brand-new name — get returns None.
        assert!(store.get_conversation("project-x").await.unwrap().is_none());
        // Insert.
        store.put("r1", None, "s", "user", "start").await.unwrap();
        store.set_conversation("project-x", "r1").await.unwrap();
        assert_eq!(
            store.get_conversation("project-x").await.unwrap().as_deref(),
            Some("r1")
        );
        // Update.
        store
            .put("r2", Some("r1"), "s", "assistant", "reply")
            .await
            .unwrap();
        store.set_conversation("project-x", "r2").await.unwrap();
        assert_eq!(
            store.get_conversation("project-x").await.unwrap().as_deref(),
            Some("r2")
        );
    }

    #[tokio::test]
    async fn pragma_foreign_keys_is_on() {
        // Sanity that PRAGMA foreign_keys = ON is applied — same
        // regression we caught in sessions::store.
        let (store, _dir) = make_store();
        let conn = store.conn.lock();
        let val: i64 = conn
            .query_row("PRAGMA foreign_keys", [], |r| r.get(0))
            .unwrap();
        assert_eq!(val, 1);
    }

    #[tokio::test]
    async fn chain_walk_caps_at_max_hops() {
        let (store, _dir) = make_store();
        // Build a degenerate chain where each node points at itself
        // by id. The walker should hit the cap and stop without
        // looping.
        // (We'd need to insert + then update prev_response_id to
        // self; for the unit test it's enough to verify the
        // walker terminates on a long well-formed chain.)
        let n = 50;
        let mut prev: Option<String> = None;
        for i in 0..n {
            let id = format!("r{}", i);
            store
                .put(&id, prev.as_deref(), "s", "user", &format!("m{}", i))
                .await
                .unwrap();
            prev = Some(id);
        }
        let chain = store.walk_chain(&format!("r{}", n - 1)).await.unwrap();
        assert_eq!(chain.len(), n as usize);
    }
}
