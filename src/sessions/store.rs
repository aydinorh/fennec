use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use rusqlite::Connection;

/// A search hit from FTS5 across session messages.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub score: f64,
}

/// One message read from a session, in time order. Returned by
/// [`SessionStore::list_messages_for_session`] and
/// [`SessionStore::list_messages_after`] (the MCP server's
/// `messages_read` and event-bridge polling both consume this).
#[derive(Debug, Clone)]
pub struct MessageRow {
    /// Auto-increment primary key. Used by the MCP server's event
    /// bridge as a polling cursor — ask for messages with id > the
    /// last-seen cursor and forward them as `message` events.
    pub id: i64,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

/// Metadata for a session.
#[derive(Debug, Clone)]
pub struct SessionRecord {
    pub id: String,
    pub channel: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub summary: Option<String>,
    /// Human label set by `/title <name>`. Distinct from
    /// `summary` (which is the system-generated end-of-session
    /// description). `None` until the user names the session.
    pub title: Option<String>,
}

/// One persisted message inside a session — replayed into
/// [`Agent::replace_history`] when `/resume` loads a prior
/// conversation.
///
/// `tool_calls` is the JSON-encoded `Vec<ToolCall>` for
/// assistant messages that requested tool invocations;
/// `tool_call_id` is set on tool-result rows so they can be
/// linked back to the originating call. Both are `None` for
/// plain user/assistant text messages, which keeps the table
/// small for non-tool turns.
#[derive(Debug, Clone)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
}

/// SQLite-backed session store with FTS5 search.
///
/// All read/write methods are `async` and internally delegate to
/// `tokio::task::spawn_blocking` so that holding the connection mutex
/// during a synchronous SQLite operation does not block a tokio worker
/// thread. The `&self` methods are therefore safe to call from any
/// async context (agent loop, Telegram handler, etc.).
pub struct SessionStore {
    conn: Arc<Mutex<Connection>>,
}

impl SessionStore {
    /// Open (or create) the session database at `db_path` and initialise the
    /// schema including FTS5 virtual table and sync triggers.
    pub fn new(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating parent dirs for {}", db_path.display()))?;
        }

        let conn = Connection::open(db_path)
            .with_context(|| format!("opening sessions db at {}", db_path.display()))?;

        // Performance + integrity PRAGMAs. `foreign_keys = ON` is required
        // for the declared `session_messages.session_id REFERENCES
        // sessions(id)` clause to actually be enforced — without it SQLite
        // parses the constraint but never checks it, so orphan messages
        // pointing at a deleted or never-created session can accumulate.
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA temp_store = MEMORY;
             PRAGMA foreign_keys = ON;",
        )
        .context("applying PRAGMAs")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                summary TEXT,
                title TEXT
            );

            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tool_calls TEXT,
                tool_call_id TEXT
            );

            -- Indexes for per-channel session lookup and for the
            -- session_id FK join in search. Without these, every
            -- `list_sessions` scan and every JOIN in search had to
            -- full-scan.
            CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(channel);
            CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
                ON session_messages(session_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS session_messages_fts USING fts5(
                content,
                content=session_messages,
                content_rowid=id
            );

            -- FTS sync triggers
            CREATE TRIGGER IF NOT EXISTS sm_ai AFTER INSERT ON session_messages BEGIN
                INSERT INTO session_messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS sm_ad AFTER DELETE ON session_messages BEGIN
                INSERT INTO session_messages_fts(session_messages_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS sm_au AFTER UPDATE ON session_messages BEGIN
                INSERT INTO session_messages_fts(session_messages_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO session_messages_fts(rowid, content)
                VALUES (new.id, new.content);
            END;",
        )
        .context("creating sessions schema")?;

        // Migration for older dbs that pre-date the `title` column.
        // SQLite has no `IF NOT EXISTS` for ADD COLUMN, so check the
        // table's pragma list and only run the ALTER when needed.
        // Failure here is non-fatal — if the column already exists
        // the duplicate-column error is the expected case.
        let has_title_col: bool = {
            let mut stmt = conn
                .prepare("PRAGMA table_info(sessions)")
                .context("preparing pragma table_info(sessions)")?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .context("querying table_info(sessions)")?;
            let mut found = false;
            for r in rows {
                if r? == "title" {
                    found = true;
                    break;
                }
            }
            found
        };
        if !has_title_col {
            conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT", [])
                .context("adding title column to sessions")?;
        }

        // Migration: tool_calls + tool_call_id columns on
        // session_messages. Existing dbs predate F1-1's
        // tool-call replay support; this ALTER lets prior
        // sessions co-exist with new ones (their tool messages
        // just have NULL in both columns and replay as text-only,
        // matching the F1-1 amendment's first-cut behavior).
        let mut has_tool_calls_col = false;
        let mut has_tool_call_id_col = false;
        {
            let mut stmt = conn
                .prepare("PRAGMA table_info(session_messages)")
                .context("preparing pragma table_info(session_messages)")?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .context("querying table_info(session_messages)")?;
            for r in rows {
                match r?.as_str() {
                    "tool_calls" => has_tool_calls_col = true,
                    "tool_call_id" => has_tool_call_id_col = true,
                    _ => {}
                }
            }
        }
        if !has_tool_calls_col {
            conn.execute(
                "ALTER TABLE session_messages ADD COLUMN tool_calls TEXT",
                [],
            )
            .context("adding tool_calls column to session_messages")?;
        }
        if !has_tool_call_id_col {
            conn.execute(
                "ALTER TABLE session_messages ADD COLUMN tool_call_id TEXT",
                [],
            )
            .context("adding tool_call_id column to session_messages")?;
        }

        // Conversation-only checkpoint table. Each row records
        // "at this point in time, the session had N messages."
        // /rollback restores by truncating session_messages to
        // that count + telling the Agent to drop its in-memory
        // history past that index. No filesystem rollback —
        // deliberate scope choice (see F1-2 plan).
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS session_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                hash TEXT NOT NULL,
                preview TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_session_checkpoints_session
                ON session_checkpoints(session_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_session_checkpoints_hash
                ON session_checkpoints(session_id, hash);",
        )
        .context("creating session_checkpoints schema")?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }


    /// List messages for a session in time order, capped by `limit`.
    /// Used by the OpenAI-compat channel to load per-session
    /// conversation history into the agent on each request.
    pub async fn list_messages_for_session(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<MessageRow>> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, session_id, role, content, timestamp
                 FROM session_messages
                 WHERE session_id = ?1
                 ORDER BY id ASC
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(
                rusqlite::params![session_id, limit as i64],
                |row| {
                    Ok(MessageRow {
                        id: row.get(0)?,
                        session_id: row.get(1)?,
                        role: row.get(2)?,
                        content: row.get(3)?,
                        timestamp: row.get(4)?,
                    })
                },
            )?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            Ok::<_, anyhow::Error>(out)
        })
        .await?
    }

    /// Create a session row with a caller-supplied id and channel.
    /// Idempotent — uses `INSERT OR IGNORE`, so a second call with
    /// the same id is a no-op.
    ///
    /// Used by the OpenAI-compat channel, which honors a client-
    /// supplied `X-Fennec-Session-Id` header (prefixed with
    /// `openai_compat:`). Other channels generate UUIDs via
    /// [`Self::create_session`].
    pub async fn create_session_with_id(
        &self,
        session_id: &str,
        channel: &str,
    ) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        let channel = channel.to_string();
        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now().to_rfc3339();
            let conn = conn.lock();
            conn.execute(
                "INSERT OR IGNORE INTO sessions (id, channel, started_at) VALUES (?1, ?2, ?3)",
                rusqlite::params![session_id, channel, now],
            )
            .context("inserting session with supplied id")?;
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }

    /// Create a new session for the given channel. Returns the session ID.
    pub async fn create_session(&self, channel: &str) -> Result<String> {
        let conn = Arc::clone(&self.conn);
        let channel = channel.to_string();
        tokio::task::spawn_blocking(move || {
            let id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            let conn = conn.lock();
            conn.execute(
                "INSERT INTO sessions (id, channel, started_at) VALUES (?1, ?2, ?3)",
                rusqlite::params![id, channel, now],
            )
            .context("inserting session")?;
            Ok::<_, anyhow::Error>(id)
        })
        .await?
    }

    /// Add a plain text message (no tool-call structure) to an
    /// existing session. Convenience wrapper over
    /// [`Self::add_message_full`] for callers that don't need
    /// to persist tool_calls / tool_call_id.
    pub async fn add_message(&self, session_id: &str, role: &str, content: &str) -> Result<()> {
        self.add_message_full(session_id, role, content, None, None).await
    }

    /// Add a message including optional tool-call structure.
    /// Used by the TUI's per-turn persistence loop so resumed
    /// sessions replay tool turns faithfully — assistant
    /// messages keep their `tool_calls` array, tool-result
    /// rows keep their `tool_call_id`.
    pub async fn add_message_full(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
        tool_calls: Option<&str>,
        tool_call_id: Option<&str>,
    ) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        let role = role.to_string();
        let content = content.to_string();
        let tool_calls = tool_calls.map(|s| s.to_string());
        let tool_call_id = tool_call_id.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now().to_rfc3339();
            let conn = conn.lock();
            conn.execute(
                "INSERT INTO session_messages
                    (session_id, role, content, timestamp, tool_calls, tool_call_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    session_id,
                    role,
                    content,
                    now,
                    tool_calls,
                    tool_call_id
                ],
            )
            .context("inserting session message")?;
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }

    /// Mark a session as ended, optionally attaching a summary.
    pub async fn end_session(&self, session_id: &str, summary: Option<&str>) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        let summary = summary.map(|s| s.to_string());
        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now().to_rfc3339();
            let conn = conn.lock();
            conn.execute(
                "UPDATE sessions SET ended_at = ?1, summary = ?2 WHERE id = ?3",
                rusqlite::params![now, summary, session_id],
            )
            .context("ending session")?;
            Ok::<_, anyhow::Error>(())
        })
        .await?
    }

    /// Full-text search across all session messages. Returns results ranked by
    /// BM25 relevance score.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let Some(fts_query) = crate::memory::fts::build_match_query(query) else {
            return Ok(vec![]);
        };

        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT m.session_id, m.role, m.content, m.timestamp, bm25(session_messages_fts) AS score
                 FROM session_messages_fts AS f
                 JOIN session_messages AS m ON m.id = f.rowid
                 WHERE session_messages_fts MATCH ?1
                 ORDER BY score
                 LIMIT ?2",
            )?;

            let rows = stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
                let raw_score: f64 = row.get(4)?;
                Ok(SearchHit {
                    session_id: row.get(0)?,
                    role: row.get(1)?,
                    content: row.get(2)?,
                    timestamp: row.get(3)?,
                    score: -raw_score, // negate BM25 so higher = more relevant
                })
            })?;

            let mut results = Vec::new();
            for row in rows {
                results.push(row?);
            }
            Ok::<_, anyhow::Error>(results)
        })
        .await?
    }

    /// List every message inserted with rowid greater than `after_id`,
    /// across every session, oldest first. The MCP server's event
    /// bridge calls this on each poll: it remembers the highest id
    /// it has emitted as `message` events and forwards anything new.
    /// `limit` is a safety cap so a backlog after a long idle gap
    /// can't pull thousands of rows in one tick.
    pub async fn list_messages_after(
        &self,
        after_id: i64,
        limit: usize,
    ) -> Result<Vec<MessageRow>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, session_id, role, content, timestamp
                 FROM session_messages
                 WHERE id > ?1
                 ORDER BY id ASC
                 LIMIT ?2",
            )?;
            let rows = stmt.query_map(rusqlite::params![after_id, limit as i64], |row| {
                Ok(MessageRow {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    timestamp: row.get(4)?,
                })
            })?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            Ok::<_, anyhow::Error>(out)
        })
        .await?
    }

    /// Highest message id currently in the store. Used by the event
    /// bridge to seed its cursor at startup so we don't replay every
    /// message in history as "new".
    pub async fn max_message_id(&self) -> Result<i64> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let id: i64 = conn
                .query_row(
                    "SELECT COALESCE(MAX(id), 0) FROM session_messages",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            Ok::<_, anyhow::Error>(id)
        })
        .await?
    }

    /// List the most recent sessions.
    pub async fn list_sessions(&self, limit: usize) -> Result<Vec<SessionRecord>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, channel, started_at, ended_at, summary, title
                 FROM sessions
                 ORDER BY started_at DESC
                 LIMIT ?1",
            )?;

            let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
                Ok(SessionRecord {
                    id: row.get(0)?,
                    channel: row.get(1)?,
                    started_at: row.get(2)?,
                    ended_at: row.get(3)?,
                    summary: row.get(4)?,
                    title: row.get(5)?,
                })
            })?;

            let mut results = Vec::new();
            for row in rows {
                results.push(row?);
            }
            Ok::<_, anyhow::Error>(results)
        })
        .await?
    }

    /// Look up a single session by exact id, then by exact title
    /// as a fallback. Mirrors the upstream's `db.get_session_by_title()`
    /// fallback (`tui_gateway/server.py:2180-2221`) so users who
    /// type `/resume my-experiment` instead of the UUID can still
    /// land on the right session. Returns `None` when neither
    /// match.
    pub async fn get_session(&self, id_or_title: &str) -> Result<Option<SessionRecord>> {
        let conn = Arc::clone(&self.conn);
        let key = id_or_title.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            // Try id first.
            let mut stmt = conn.prepare(
                "SELECT id, channel, started_at, ended_at, summary, title
                 FROM sessions WHERE id = ?1 LIMIT 1",
            )?;
            let mut rows = stmt.query_map(rusqlite::params![key], |row| {
                Ok(SessionRecord {
                    id: row.get(0)?,
                    channel: row.get(1)?,
                    started_at: row.get(2)?,
                    ended_at: row.get(3)?,
                    summary: row.get(4)?,
                    title: row.get(5)?,
                })
            })?;
            if let Some(row) = rows.next() {
                return Ok::<_, anyhow::Error>(Some(row?));
            }
            // Title fallback.
            let mut stmt = conn.prepare(
                "SELECT id, channel, started_at, ended_at, summary, title
                 FROM sessions WHERE title = ?1 ORDER BY started_at DESC LIMIT 1",
            )?;
            let mut rows = stmt.query_map(rusqlite::params![key], |row| {
                Ok(SessionRecord {
                    id: row.get(0)?,
                    channel: row.get(1)?,
                    started_at: row.get(2)?,
                    ended_at: row.get(3)?,
                    summary: row.get(4)?,
                    title: row.get(5)?,
                })
            })?;
            if let Some(row) = rows.next() {
                return Ok(Some(row?));
            }
            Ok(None)
        })
        .await?
    }

    /// Set (or clear) the user-facing title for a session. Empty
    /// title clears the field. Returns the number of rows
    /// affected — `0` means the session id wasn't found.
    pub async fn set_session_title(&self, id: &str, title: &str) -> Result<usize> {
        let conn = Arc::clone(&self.conn);
        let id = id.to_string();
        let title = if title.trim().is_empty() {
            None
        } else {
            Some(title.trim().to_string())
        };
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let n = conn
                .execute(
                    "UPDATE sessions SET title = ?1 WHERE id = ?2",
                    rusqlite::params![title, id],
                )
                .context("updating session title")?;
            Ok::<_, anyhow::Error>(n)
        })
        .await?
    }

    /// Fetch every persisted message for `session_id` in
    /// chronological order. Used by `/resume` to repopulate
    /// `Agent::history` so the next turn sees the full prior
    /// context, matching the upstream's `db.get_messages_as_conversation`
    /// (`tui_gateway/server.py:2180-2221`).
    pub async fn get_session_messages(&self, session_id: &str) -> Result<Vec<StoredMessage>> {
        let conn = Arc::clone(&self.conn);
        let sid = session_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT role, content, timestamp, tool_calls, tool_call_id
                 FROM session_messages
                 WHERE session_id = ?1
                 ORDER BY id ASC",
            )?;
            let rows = stmt.query_map(rusqlite::params![sid], |row| {
                Ok(StoredMessage {
                    role: row.get(0)?,
                    content: row.get(1)?,
                    timestamp: row.get(2)?,
                    tool_calls: row.get(3)?,
                    tool_call_id: row.get(4)?,
                })
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok::<_, anyhow::Error>(out)
        })
        .await?
    }

    /// Record a checkpoint at the current message count for
    /// `session_id`. The 8-char hash is generated from a uuid4 so
    /// the user-facing id is stable + collision-free per session.
    /// `preview` is a short label (typically the last assistant
    /// message text) shown in `/rollback list`. Returns the hash.
    pub async fn record_checkpoint(
        &self,
        session_id: &str,
        preview: &str,
    ) -> Result<String> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        let preview: String = preview.chars().take(120).collect();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM session_messages WHERE session_id = ?1",
                rusqlite::params![&session_id],
                |row| row.get(0),
            )
            .context("counting session_messages for checkpoint")?;
            // 8-char hash — uuid4 simple form sliced.
            let hash: String =
                uuid::Uuid::new_v4().simple().to_string().chars().take(8).collect();
            let now = chrono::Utc::now().to_rfc3339();
            conn.execute(
                "INSERT INTO session_checkpoints
                 (session_id, hash, preview, message_count, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![session_id, hash, preview, count, now],
            )
            .context("inserting session_checkpoint")?;
            Ok::<_, anyhow::Error>(hash)
        })
        .await?
    }

    /// All checkpoints for `session_id`, newest first.
    pub async fn list_checkpoints(
        &self,
        session_id: &str,
    ) -> Result<Vec<CheckpointRecord>> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn
                .prepare(
                    "SELECT hash, preview, message_count, created_at
                     FROM session_checkpoints
                     WHERE session_id = ?1
                     ORDER BY id DESC",
                )
                .context("preparing list_checkpoints")?;
            let rows = stmt.query_map(rusqlite::params![&session_id], |row| {
                Ok(CheckpointRecord {
                    hash: row.get(0)?,
                    preview: row.get(1)?,
                    message_count: row.get::<_, i64>(2)? as usize,
                    created_at: row.get(3)?,
                })
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok::<_, anyhow::Error>(out)
        })
        .await?
    }

    /// Look up a checkpoint by `(session_id, hash)`. Returns
    /// `Ok(None)` when no row matches — distinct from a SQL
    /// error.
    pub async fn get_checkpoint(
        &self,
        session_id: &str,
        hash: &str,
    ) -> Result<Option<CheckpointRecord>> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        let hash = hash.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let result = conn.query_row(
                "SELECT hash, preview, message_count, created_at
                 FROM session_checkpoints
                 WHERE session_id = ?1 AND hash = ?2",
                rusqlite::params![&session_id, &hash],
                |row| {
                    Ok(CheckpointRecord {
                        hash: row.get(0)?,
                        preview: row.get(1)?,
                        message_count: row.get::<_, i64>(2)? as usize,
                        created_at: row.get(3)?,
                    })
                },
            );
            match result {
                Ok(r) => Ok::<_, anyhow::Error>(Some(r)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(anyhow::Error::from(e)),
            }
        })
        .await?
    }

    /// Truncate `session_messages` for `session_id` to the first
    /// `message_count` rows (ordered by id ascending). Returns
    /// the number of rows actually removed. Also deletes
    /// checkpoints that pointed past the new tail since they're
    /// invalidated by the rollback.
    pub async fn restore_to_message_count(
        &self,
        session_id: &str,
        message_count: usize,
    ) -> Result<usize> {
        let conn = Arc::clone(&self.conn);
        let session_id = session_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            // Delete messages beyond message_count, keeping the
            // first `message_count` rows ordered by id.
            let removed = conn.execute(
                "DELETE FROM session_messages
                 WHERE session_id = ?1
                   AND id IN (
                     SELECT id FROM session_messages
                     WHERE session_id = ?1
                     ORDER BY id ASC
                     LIMIT -1 OFFSET ?2
                   )",
                rusqlite::params![&session_id, message_count as i64],
            )
            .context("truncating session_messages on rollback")?;
            // Invalidate checkpoints that pointed past the new tail.
            conn.execute(
                "DELETE FROM session_checkpoints
                 WHERE session_id = ?1 AND message_count > ?2",
                rusqlite::params![&session_id, message_count as i64],
            )
            .context("trimming stale checkpoints after rollback")?;
            Ok::<_, anyhow::Error>(removed)
        })
        .await?
    }
}

/// One row in `session_checkpoints`. Returned by
/// [`SessionStore::list_checkpoints`] / [`SessionStore::get_checkpoint`].
#[derive(Debug, Clone)]
pub struct CheckpointRecord {
    pub hash: String,
    pub preview: String,
    pub message_count: usize,
    pub created_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store() -> (SessionStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let db = dir.path().join("sessions.db");
        let store = SessionStore::new(&db).unwrap();
        (store, dir)
    }

    #[tokio::test]
    async fn create_and_list_sessions() {
        let (store, _dir) = make_store();
        let id = store.create_session("telegram").await.unwrap();
        let sessions = store.list_sessions(10).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, id);
        assert_eq!(sessions[0].channel, "telegram");
        assert!(sessions[0].ended_at.is_none());
    }

    #[tokio::test]
    async fn add_message_and_search() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store
            .add_message(&sid, "user", "How do I configure Fennec?")
            .await
            .unwrap();
        store
            .add_message(&sid, "assistant", "You can edit fennec.toml to configure Fennec.")
            .await
            .unwrap();

        let hits = store.search("configure Fennec", 10).await.unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].content.contains("Fennec"));
    }

    #[tokio::test]
    async fn end_session_with_summary() {
        let (store, _dir) = make_store();
        let sid = store.create_session("discord").await.unwrap();
        store
            .end_session(&sid, Some("Discussed configuration"))
            .await
            .unwrap();
        let sessions = store.list_sessions(10).await.unwrap();
        assert!(sessions[0].ended_at.is_some());
        assert_eq!(
            sessions[0].summary.as_deref(),
            Some("Discussed configuration")
        );
    }

    #[tokio::test]
    async fn search_empty_query_returns_empty() {
        let (store, _dir) = make_store();
        let hits = store.search("", 10).await.unwrap();
        assert!(hits.is_empty());
    }

    /// Regression: a user query containing `"` used to produce an
    /// FTS5 syntax error because the hand-rolled tokenizer emitted
    /// `"foo"bar"` instead of a safe quoted phrase. Now strips the
    /// inner quote and searches for `"foobar"`.
    #[tokio::test]
    async fn search_tolerates_embedded_quote() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.add_message(&sid, "user", "foobar works fine").await.unwrap();
        let hits = store.search("foo\"bar", 10).await.unwrap();
        // Either finds the message or returns empty — the critical
        // invariant is that the call does not error.
        let _ = hits;
    }

    #[tokio::test]
    async fn set_and_get_session_title_round_trip() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        // Initially title is None.
        let rec = store.get_session(&sid).await.unwrap().unwrap();
        assert!(rec.title.is_none());
        // Set title.
        let n = store
            .set_session_title(&sid, "experiment-1")
            .await
            .unwrap();
        assert_eq!(n, 1);
        let rec = store.get_session(&sid).await.unwrap().unwrap();
        assert_eq!(rec.title.as_deref(), Some("experiment-1"));
        // Empty title clears.
        let n = store.set_session_title(&sid, "  ").await.unwrap();
        assert_eq!(n, 1);
        let rec = store.get_session(&sid).await.unwrap().unwrap();
        assert!(rec.title.is_none());
    }

    #[tokio::test]
    async fn get_session_falls_back_to_title_when_id_missing() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.set_session_title(&sid, "alpha").await.unwrap();
        // Lookup by exact title string.
        let rec = store.get_session("alpha").await.unwrap().unwrap();
        assert_eq!(rec.id, sid);
    }

    #[tokio::test]
    async fn add_message_full_round_trips_tool_call_structure() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.add_message(&sid, "user", "what's the weather?").await.unwrap();
        // Assistant turn with a tool_calls array.
        let tcs_json =
            r#"[{"id":"call_1","name":"weather","arguments":{"city":"SF"}}]"#;
        store
            .add_message_full(&sid, "assistant", "", Some(tcs_json), None)
            .await
            .unwrap();
        // Tool-result row referring to that id.
        store
            .add_message_full(
                &sid,
                "tool",
                "{\"temp\": 64}",
                None,
                Some("call_1"),
            )
            .await
            .unwrap();
        store
            .add_message(&sid, "assistant", "It's 64°F in SF.")
            .await
            .unwrap();

        let msgs = store.get_session_messages(&sid).await.unwrap();
        assert_eq!(msgs.len(), 4);
        // The assistant tool-call message must come back with
        // its tool_calls JSON intact.
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].tool_calls.as_deref(), Some(tcs_json));
        // The tool result row must preserve tool_call_id.
        assert_eq!(msgs[2].role, "tool");
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msgs[2].content, "{\"temp\": 64}");
        // Plain rows have None on both new fields.
        assert!(msgs[0].tool_calls.is_none());
        assert!(msgs[0].tool_call_id.is_none());
        assert!(msgs[3].tool_calls.is_none());
        assert!(msgs[3].tool_call_id.is_none());
    }

    #[tokio::test]
    async fn list_messages_for_session_returns_in_time_order() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.add_message(&sid, "user", "first").await.unwrap();
        store.add_message(&sid, "assistant", "second").await.unwrap();
        store.add_message(&sid, "user", "third").await.unwrap();
        let rows = store
            .list_messages_for_session(&sid, 100)
            .await
            .unwrap();
        let contents: Vec<&str> = rows.iter().map(|r| r.content.as_str()).collect();
        assert_eq!(contents, vec!["first", "second", "third"]);
    }

    #[tokio::test]
    async fn list_messages_for_session_respects_limit() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        for i in 0..5 {
            store
                .add_message(&sid, "user", &format!("msg{}", i))
                .await
                .unwrap();
        }
        let rows = store
            .list_messages_for_session(&sid, 3)
            .await
            .unwrap();
        assert_eq!(rows.len(), 3);
    }

    #[tokio::test]
    async fn list_messages_after_pagination_works() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        for i in 0..4 {
            store
                .add_message(&sid, "user", &format!("m{}", i))
                .await
                .unwrap();
        }
        let all = store
            .list_messages_for_session(&sid, 100)
            .await
            .unwrap();
        let cursor = all[1].id;
        let after = store.list_messages_after(cursor, 100).await.unwrap();
        let contents: Vec<&str> = after.iter().map(|r| r.content.as_str()).collect();
        assert_eq!(contents, vec!["m2", "m3"]);
    }

    #[tokio::test]
    async fn max_message_id_zero_when_empty() {
        let (store, _dir) = make_store();
        assert_eq!(store.max_message_id().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn max_message_id_tracks_inserts() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.add_message(&sid, "user", "x").await.unwrap();
        let after_one = store.max_message_id().await.unwrap();
        store.add_message(&sid, "user", "y").await.unwrap();
        let after_two = store.max_message_id().await.unwrap();
        assert!(after_two > after_one);
    }

    #[tokio::test]
    async fn get_session_returns_record() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        let rec = store.get_session(&sid).await.unwrap();
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().channel, "cli");
    }

    #[tokio::test]
    async fn get_session_none_when_missing() {
        let (store, _dir) = make_store();
        assert!(store.get_session("nope").await.unwrap().is_none());
    }

    /// Regression: the store's FK declaration (session_messages
    /// REFERENCES sessions) was not being enforced because
    /// `PRAGMA foreign_keys = ON` was missing. Inserting a message
    /// pointing at a non-existent session must now fail.
    #[tokio::test]
    async fn foreign_key_constraint_is_enforced() {
        let (store, _dir) = make_store();
        let r = store
            .add_message("non-existent-session-id", "user", "oops")
            .await;
        assert!(
            r.is_err(),
            "message with dangling session_id must be rejected (got Ok) — PRAGMA foreign_keys not applied?"
        );
    }

    #[tokio::test]
    async fn checkpoint_round_trip() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.add_message(&sid, "user", "hello").await.unwrap();
        store.add_message(&sid, "assistant", "hi there").await.unwrap();
        let hash = store
            .record_checkpoint(&sid, "after first exchange")
            .await
            .unwrap();
        assert_eq!(hash.len(), 8);
        let record = store
            .get_checkpoint(&sid, &hash)
            .await
            .unwrap()
            .expect("checkpoint should exist");
        assert_eq!(record.message_count, 2);
        assert_eq!(record.preview, "after first exchange");
    }

    #[tokio::test]
    async fn list_checkpoints_returns_newest_first() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.add_message(&sid, "user", "u1").await.unwrap();
        let _h1 = store.record_checkpoint(&sid, "first").await.unwrap();
        store.add_message(&sid, "user", "u2").await.unwrap();
        let h2 = store.record_checkpoint(&sid, "second").await.unwrap();
        let list = store.list_checkpoints(&sid).await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].hash, h2);
        assert_eq!(list[0].preview, "second");
    }

    #[tokio::test]
    async fn restore_to_message_count_trims_messages() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        for i in 0..6 {
            store
                .add_message(&sid, "user", &format!("u{i}"))
                .await
                .unwrap();
        }
        // Trim to 2 messages; 4 should be removed.
        let removed = store.restore_to_message_count(&sid, 2).await.unwrap();
        assert_eq!(removed, 4);
        let remaining = store.get_session_messages(&sid).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].content, "u0");
        assert_eq!(remaining[1].content, "u1");
    }

    #[tokio::test]
    async fn restore_invalidates_checkpoints_past_new_tail() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        store.add_message(&sid, "user", "u1").await.unwrap();
        let h1 = store.record_checkpoint(&sid, "after u1").await.unwrap();
        store.add_message(&sid, "user", "u2").await.unwrap();
        let h2 = store.record_checkpoint(&sid, "after u2").await.unwrap();
        store.restore_to_message_count(&sid, 1).await.unwrap();
        // h1 should survive (message_count == 1, still valid).
        // h2 should be deleted (message_count == 2, past new tail).
        assert!(store.get_checkpoint(&sid, &h1).await.unwrap().is_some());
        assert!(store.get_checkpoint(&sid, &h2).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn get_checkpoint_returns_none_for_unknown_hash() {
        let (store, _dir) = make_store();
        let sid = store.create_session("cli").await.unwrap();
        let res = store.get_checkpoint(&sid, "deadbeef").await.unwrap();
        assert!(res.is_none());
    }
}
