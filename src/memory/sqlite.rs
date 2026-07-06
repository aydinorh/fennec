use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use parking_lot::Mutex;
use rusqlite::Connection;
use sha2::{Sha256, Digest};

use crate::memory::embedding::EmbeddingProvider;
use crate::memory::experience::{Experience, ExperienceContext};
use crate::memory::traits::{Memory, MemoryCategory, MemoryEntry};
use crate::memory::vector::{cosine_similarity, hybrid_merge, ScoredResult};

/// SQLite-backed memory store.
pub struct SqliteMemory {
    conn: Arc<Mutex<Connection>>,
    #[allow(dead_code)]
    db_path: PathBuf,
    vector_weight: f32,
    keyword_weight: f32,
    /// Cap on `embedding_cache` row count. The cache table is bounded
    /// by LRU eviction in [`Memory::store`] — when an insert pushes
    /// the row count past `cache_max`, the oldest rows by
    /// `accessed_at` are deleted. `0` is treated as "unlimited".
    cache_max: usize,
    embedder: Arc<dyn EmbeddingProvider>,
}

impl SqliteMemory {
    /// Create a new SQLite memory backend.
    ///
    /// Opens (or creates) the database at `db_path`, applies performance
    /// PRAGMAs, and initialises the schema (tables, FTS5, triggers, cache).
    pub fn new(
        db_path: PathBuf,
        vector_weight: f32,
        keyword_weight: f32,
        cache_max: usize,
        embedder: Arc<dyn EmbeddingProvider>,
    ) -> Result<Self> {
        // Create parent directories if they don't exist.
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating parent dirs for {}", db_path.display()))?;
        }

        let conn = Connection::open(&db_path)
            .with_context(|| format!("opening sqlite db at {}", db_path.display()))?;

        // Performance + integrity PRAGMAs. `foreign_keys = ON` is
        // required for `REFERENCES` clauses to actually be enforced —
        // without it, declared FKs are parsed but ignored, so a bad row
        // can silently land with a dangling reference (the audit flagged
        // this for memories.superseded_by).
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;
             PRAGMA mmap_size = 8388608;
             PRAGMA cache_size = -2000;
             PRAGMA temp_store = MEMORY;
             PRAGMA foreign_keys = ON;",
        )
        .context("applying PRAGMAs")?;

        // Schema
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id            TEXT PRIMARY KEY,
                key           TEXT UNIQUE NOT NULL,
                content       TEXT NOT NULL,
                category      TEXT NOT NULL DEFAULT 'core',
                embedding     BLOB,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                session_id    TEXT,
                namespace     TEXT NOT NULL DEFAULT 'default',
                importance    REAL,
                superseded_by TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
            -- Session/namespace queries previously table-scanned because no
            -- index backed these columns. `superseded_by` is touched by
            -- cleanup queries that look for dangling pointers.
            CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
            CREATE INDEX IF NOT EXISTS idx_memories_superseded_by ON memories(superseded_by);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                key, content, content='memories', content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, key, content)
                VALUES (new.rowid, new.key, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, content)
                VALUES ('delete', old.rowid, old.key, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, key, content)
                VALUES ('delete', old.rowid, old.key, old.content);
                INSERT INTO memories_fts(rowid, key, content)
                VALUES (new.rowid, new.key, new.content);
            END;

            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding    BLOB,
                created_at   TEXT NOT NULL,
                accessed_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiences (
                id          TEXT PRIMARY KEY,
                goal        TEXT NOT NULL,
                context_json TEXT NOT NULL DEFAULT '{}',
                attempts_json TEXT NOT NULL DEFAULT '[]',
                solution    TEXT,
                gotchas_json TEXT NOT NULL DEFAULT '[]',
                tags_json   TEXT NOT NULL DEFAULT '[]',
                confidence  REAL NOT NULL DEFAULT 0.5,
                session_id  TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts USING fts5(
                goal, solution, content=experiences, content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS experiences_ai AFTER INSERT ON experiences BEGIN
                INSERT INTO experiences_fts(rowid, goal, solution)
                VALUES (new.rowid, new.goal, COALESCE(new.solution, ''));
            END;

            CREATE TRIGGER IF NOT EXISTS experiences_ad AFTER DELETE ON experiences BEGIN
                INSERT INTO experiences_fts(experiences_fts, rowid, goal, solution)
                VALUES ('delete', old.rowid, old.goal, COALESCE(old.solution, ''));
            END;

            CREATE TRIGGER IF NOT EXISTS experiences_au AFTER UPDATE ON experiences BEGIN
                INSERT INTO experiences_fts(experiences_fts, rowid, goal, solution)
                VALUES ('delete', old.rowid, old.goal, COALESCE(old.solution, ''));
                INSERT INTO experiences_fts(rowid, goal, solution)
                VALUES (new.rowid, new.goal, COALESCE(new.solution, ''));
            END;

            CREATE TABLE IF NOT EXISTS collective_cache (
                id TEXT PRIMARY KEY,
                original_id TEXT NOT NULL,
                source_server TEXT NOT NULL,
                goal TEXT NOT NULL,
                solution TEXT,
                gotchas_json TEXT NOT NULL DEFAULT '[]',
                attempts_json TEXT NOT NULL DEFAULT '[]',
                tags_json TEXT NOT NULL DEFAULT '[]',
                trust_score REAL NOT NULL DEFAULT 0.3,
                relevance_score REAL,
                outcome_success INTEGER NOT NULL DEFAULT 0,
                outcome_failure INTEGER NOT NULL DEFAULT 0,
                cached_at TEXT NOT NULL,
                last_used TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS collective_cache_fts USING fts5(
                goal, solution, content=collective_cache, content_rowid=rowid
            );

            CREATE TRIGGER IF NOT EXISTS cc_ai AFTER INSERT ON collective_cache BEGIN
                INSERT INTO collective_cache_fts(rowid, goal, solution)
                VALUES (new.rowid, new.goal, COALESCE(new.solution, ''));
            END;
            CREATE TRIGGER IF NOT EXISTS cc_ad AFTER DELETE ON collective_cache BEGIN
                INSERT INTO collective_cache_fts(collective_cache_fts, rowid, goal, solution)
                VALUES ('delete', old.rowid, old.goal, COALESCE(old.solution, ''));
            END;
            CREATE TRIGGER IF NOT EXISTS cc_au AFTER UPDATE ON collective_cache BEGIN
                INSERT INTO collective_cache_fts(collective_cache_fts, rowid, goal, solution)
                VALUES ('delete', old.rowid, old.goal, COALESCE(old.solution, ''));
                INSERT INTO collective_cache_fts(rowid, goal, solution)
                VALUES (new.rowid, new.goal, COALESCE(new.solution, ''));
            END;

            -- Idempotent dedupe of duplicate `original_id` rows before the
            -- UNIQUE index goes in. Previously cache_result did a
            -- DELETE + INSERT pair with no constraint, so two concurrent
            -- callers could land two rows for the same original_id.
            -- Keeps the row with the smallest rowid per original_id group.
            DELETE FROM collective_cache
            WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM collective_cache GROUP BY original_id
            );

            -- UNIQUE index on original_id serves two purposes: makes
            -- subsequent accidental duplicates fail-loud at INSERT time,
            -- and gives cache_result's new ON CONFLICT upsert a target
            -- to reference.
            CREATE UNIQUE INDEX IF NOT EXISTS idx_collective_cache_original_id
                ON collective_cache(original_id);",
        )
        .context("creating schema")?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path,
            vector_weight,
            keyword_weight,
            cache_max,
            embedder,
        })
    }
}

impl SqliteMemory {
    /// Expose the underlying connection for use by other modules (e.g. collective cache).
    pub fn connection(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
    }
}

/// Convert a `MemoryCategory` to its string representation.
fn category_to_str(cat: &MemoryCategory) -> &str {
    match cat {
        MemoryCategory::Core => "core",
        MemoryCategory::Daily => "daily",
        MemoryCategory::Conversation => "conversation",
        MemoryCategory::Custom(s) => s.as_str(),
    }
}

/// Convert a string to a `MemoryCategory`.
fn str_to_category(s: &str) -> MemoryCategory {
    match s {
        "core" => MemoryCategory::Core,
        "daily" => MemoryCategory::Daily,
        "conversation" => MemoryCategory::Conversation,
        other => MemoryCategory::Custom(other.to_string()),
    }
}

/// Perform a keyword search against the FTS5 index using BM25 scoring.
///
/// Returns `(id, score)` pairs sorted by relevance. BM25 scores from SQLite
/// are negative (lower is better), so we negate them to produce positive
/// scores where higher is better.
fn keyword_search(
    conn: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<(String, f64)>> {
    let Some(fts_query) = super::fts::build_match_query(query) else {
        return Ok(vec![]);
    };

    let mut stmt = conn.prepare(
        "SELECT m.id, bm25(memories_fts) AS score
         FROM memories_fts AS f
         JOIN memories AS m ON m.rowid = f.rowid
         WHERE memories_fts MATCH ?1
         ORDER BY score
         LIMIT ?2",
    )?;

    let rows = stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
        let id: String = row.get(0)?;
        let raw_score: f64 = row.get(1)?;
        Ok((id, -raw_score)) // negate BM25 score
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }

    // Sort by score descending (highest relevance first).
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Serialize a Vec<f32> as a little-endian byte blob.
fn serialize_embedding(vec: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for &v in vec {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Deserialize a little-endian byte blob back into a Vec<f32>.
fn deserialize_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Perform a vector search by computing cosine similarity against all stored embeddings.
///
/// Returns `(id, score)` pairs sorted by similarity descending. Only rows with
/// non-NULL embeddings are considered.
fn vector_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> Result<Vec<(String, f64)>> {
    let mut stmt = conn.prepare(
        "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL",
    )?;

    let rows = stmt.query_map([], |row| {
        let id: String = row.get(0)?;
        let blob: Vec<u8> = row.get(1)?;
        Ok((id, blob))
    })?;

    let mut results: Vec<(String, f64)> = Vec::new();
    for row in rows {
        let (id, blob) = row?;
        let stored_embedding = deserialize_embedding(&blob);
        if stored_embedding.len() != query_embedding.len() {
            continue; // Skip mismatched dimensions.
        }
        let similarity = cosine_similarity(query_embedding, &stored_embedding);
        results.push((id, similarity));
    }

    // Sort by similarity descending.
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    Ok(results)
}

/// Read a full `MemoryEntry` row from the database.
fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEntry> {
    let cat_str: String = row.get(3)?;
    let importance: Option<f64> = row.get(9)?;
    let superseded_by: Option<String> = row.get(10)?;

    Ok(MemoryEntry {
        id: row.get(0)?,
        key: row.get(1)?,
        content: row.get(2)?,
        category: str_to_category(&cat_str),
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
        session_id: row.get(6)?,
        namespace: row.get(7)?,
        importance,
        score: None,
        superseded_by,
    })
}

#[async_trait]
impl Memory for SqliteMemory {
    fn name(&self) -> &str {
        "sqlite"
    }

    async fn store(&self, entry: MemoryEntry) -> Result<()> {
        // Compute embedding if the embedder is not noop.
        //
        // The cache lookup and accessed_at update — both blocking
        // SQLite calls — used to run inline on the async runtime
        // thread. Under any concurrent load that serialised every
        // other async task behind a lock+query+update cycle. Now both
        // sides of the cache cycle run inside `spawn_blocking` so the
        // executor stays free.
        //
        // The accessed_at update doubles as the LRU touch for the
        // cache_max enforcement at insert-time below: rows that
        // haven't been accessed recently get evicted first.
        let embedding_blob = if self.embedder.name() != "noop" {
            let content_hash = {
                let mut hasher = Sha256::new();
                hasher.update(entry.content.as_bytes());
                hex::encode(hasher.finalize())
            };

            // Phase 1: cache lookup + accessed_at touch. Returns the
            // cached blob if present, or None to signal "embed it".
            let cache_arc = Arc::clone(&self.conn);
            let hash_for_lookup = content_hash.clone();
            let cached: Option<Vec<u8>> = tokio::task::spawn_blocking(move || -> Result<Option<Vec<u8>>> {
                let conn = cache_arc.lock();
                let mut stmt = conn.prepare(
                    "SELECT embedding FROM embedding_cache WHERE content_hash = ?1",
                )?;
                let mut rows = stmt.query(rusqlite::params![hash_for_lookup])?;
                if let Some(row) = rows.next()? {
                    let blob: Vec<u8> = row.get(0)?;
                    drop(rows);
                    drop(stmt);
                    let now = chrono::Utc::now().to_rfc3339();
                    conn.execute(
                        "UPDATE embedding_cache SET accessed_at = ?1 WHERE content_hash = ?2",
                        rusqlite::params![now, hash_for_lookup],
                    )?;
                    Ok(Some(blob))
                } else {
                    Ok(None)
                }
            })
            .await??;

            if let Some(blob) = cached {
                Some(blob)
            } else {
                // Compute embedding (the only legitimately-async step
                // — the embedder calls a remote API).
                let vec = self.embedder.embed(&entry.content).await?;
                let blob = serialize_embedding(&vec);

                // Phase 2: cache insert + LRU eviction. Run as one
                // blocking task so we don't take/release the conn
                // lock between the insert and the size check.
                let cache_arc = Arc::clone(&self.conn);
                let cache_max = self.cache_max;
                let blob_for_cache = blob.clone();
                let hash_for_insert = content_hash.clone();
                tokio::task::spawn_blocking(move || -> Result<()> {
                    let conn = cache_arc.lock();
                    let now = chrono::Utc::now().to_rfc3339();
                    conn.execute(
                        "INSERT OR REPLACE INTO embedding_cache \
                         (content_hash, embedding, created_at, accessed_at) \
                         VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![hash_for_insert, blob_for_cache, now, now],
                    )?;

                    // LRU enforcement: if the cache exceeded its
                    // configured cap, delete the least-recently-used
                    // rows. `cache_max == 0` is treated as "unlimited"
                    // for back-compat with callers that don't care.
                    if cache_max > 0 {
                        let count: i64 = conn.query_row(
                            "SELECT COUNT(*) FROM embedding_cache",
                            [],
                            |r| r.get(0),
                        )?;
                        if (count as usize) > cache_max {
                            let to_delete = (count as usize) - cache_max;
                            // ORDER BY accessed_at ASC LIMIT N gives us
                            // the oldest N rows. Delete them in a
                            // single statement so we hold the conn
                            // lock once.
                            conn.execute(
                                "DELETE FROM embedding_cache \
                                 WHERE content_hash IN ( \
                                     SELECT content_hash FROM embedding_cache \
                                     ORDER BY accessed_at ASC LIMIT ?1 \
                                 )",
                                rusqlite::params![to_delete as i64],
                            )?;
                        }
                    }
                    Ok(())
                })
                .await??;

                Some(blob)
            }
        } else {
            None
        };

        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            conn.execute(
                "INSERT INTO memories (id, key, content, category, embedding, created_at, updated_at, session_id, namespace, importance, superseded_by)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                 ON CONFLICT(key) DO UPDATE SET
                     content = excluded.content,
                     category = excluded.category,
                     embedding = excluded.embedding,
                     updated_at = excluded.updated_at,
                     session_id = excluded.session_id,
                     namespace = excluded.namespace,
                     importance = excluded.importance,
                     superseded_by = excluded.superseded_by",
                rusqlite::params![
                    entry.id,
                    entry.key,
                    entry.content,
                    category_to_str(&entry.category),
                    embedding_blob,
                    entry.created_at,
                    entry.updated_at,
                    entry.session_id,
                    entry.namespace,
                    entry.importance,
                    entry.superseded_by,
                ],
            )?;
            Ok(())
        })
        .await?
    }

    async fn recall(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let use_vector = self.embedder.name() != "noop";

        // Compute query embedding if we have a real embedder.
        let query_embedding = if use_vector {
            Some(self.embedder.embed(query).await?)
        } else {
            None
        };

        let conn = Arc::clone(&self.conn);
        let query = query.to_string();
        let vector_weight = self.vector_weight as f64;
        let keyword_weight = self.keyword_weight as f64;
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();

            // Keyword search via FTS5.
            let keyword_results = keyword_search(&conn, &query, limit * 2)?;

            // Vector search if we have a query embedding.
            let merged_ids: Vec<(String, f64)> = if let Some(ref qe) = query_embedding {
                let vector_results = vector_search(&conn, qe, limit * 2)?;

                // Convert to ScoredResult for hybrid_merge.
                let vec_scored: Vec<ScoredResult> = vector_results
                    .iter()
                    .map(|(id, score)| ScoredResult {
                        id: id.clone(),
                        score: *score,
                    })
                    .collect();
                let kw_scored: Vec<ScoredResult> = keyword_results
                    .iter()
                    .map(|(id, score)| ScoredResult {
                        id: id.clone(),
                        score: *score,
                    })
                    .collect();

                let merged = hybrid_merge(
                    &vec_scored,
                    &kw_scored,
                    vector_weight,
                    keyword_weight,
                    limit,
                );

                merged.into_iter().map(|r| (r.id, r.score)).collect()
            } else {
                // Keyword-only.
                keyword_results
                    .iter()
                    .take(limit)
                    .cloned()
                    .collect()
            };

            if merged_ids.is_empty() {
                return Ok(vec![]);
            }

            // Fetch full rows for matched IDs.
            let ids: Vec<String> = merged_ids.iter().map(|(id, _)| id.clone()).collect();
            let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id, key, content, category, created_at, updated_at, session_id, namespace, embedding, importance, superseded_by
                 FROM memories WHERE id IN ({})",
                placeholders
            );

            let mut stmt = conn.prepare(&sql)?;

            let params: Vec<Box<dyn rusqlite::types::ToSql>> = ids
                .iter()
                .map(|id| Box::new(id.clone()) as Box<dyn rusqlite::types::ToSql>)
                .collect();
            let param_refs: Vec<&dyn rusqlite::types::ToSql> =
                params.iter().map(|p| p.as_ref()).collect();

            let rows = stmt.query_map(param_refs.as_slice(), |row| row_to_entry(row))?;

            let mut entries: Vec<MemoryEntry> = Vec::new();
            for row in rows {
                let mut entry = row?;
                // Attach the merged score.
                if let Some((_, score)) = merged_ids.iter().find(|(id, _)| *id == entry.id) {
                    entry.score = Some(*score);
                }
                entries.push(entry);
            }

            // Sort by score descending.
            entries.sort_by(|a, b| {
                let sa = a.score.unwrap_or(0.0);
                let sb = b.score.unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            entries.truncate(limit);

            Ok(entries)
        })
        .await?
    }

    async fn get(&self, key: &str) -> Result<Option<MemoryEntry>> {
        let conn = Arc::clone(&self.conn);
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, key, content, category, created_at, updated_at, session_id, namespace, embedding, importance, superseded_by
                 FROM memories WHERE key = ?1",
            )?;
            let mut rows = stmt.query_map(rusqlite::params![key], |row| row_to_entry(row))?;
            match rows.next() {
                Some(Ok(entry)) => Ok(Some(entry)),
                Some(Err(e)) => Err(e.into()),
                None => Ok(None),
            }
        })
        .await?
    }

    async fn list(
        &self,
        category: Option<&MemoryCategory>,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let conn = Arc::clone(&self.conn);
        let cat_str = category.map(|c| category_to_str(c).to_string());
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let entries = if let Some(ref cat) = cat_str {
                let mut stmt = conn.prepare(
                    "SELECT id, key, content, category, created_at, updated_at, session_id, namespace, embedding, importance, superseded_by
                     FROM memories WHERE category = ?1 ORDER BY created_at DESC LIMIT ?2",
                )?;
                let rows = stmt.query_map(rusqlite::params![cat, limit as i64], |row| {
                    row_to_entry(row)
                })?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            } else {
                let mut stmt = conn.prepare(
                    "SELECT id, key, content, category, created_at, updated_at, session_id, namespace, embedding, importance, superseded_by
                     FROM memories ORDER BY created_at DESC LIMIT ?1",
                )?;
                let rows = stmt
                    .query_map(rusqlite::params![limit as i64], |row| row_to_entry(row))?;
                rows.collect::<rusqlite::Result<Vec<_>>>()?
            };
            Ok(entries)
        })
        .await?
    }

    async fn forget(&self, key: &str) -> Result<bool> {
        let conn = Arc::clone(&self.conn);
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let deleted = conn.execute("DELETE FROM memories WHERE key = ?1", rusqlite::params![key])?;
            Ok(deleted > 0)
        })
        .await?
    }

    async fn count(&self, category: Option<&MemoryCategory>) -> Result<usize> {
        let conn = Arc::clone(&self.conn);
        let cat_str = category.map(|c| category_to_str(c).to_string());
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let count: i64 = if let Some(ref cat) = cat_str {
                conn.query_row(
                    "SELECT COUNT(*) FROM memories WHERE category = ?1",
                    rusqlite::params![cat],
                    |row| row.get(0),
                )?
            } else {
                conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?
            };
            Ok(count as usize)
        })
        .await?
    }

    async fn health_check(&self) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            conn.query_row("SELECT 1", [], |_row| Ok(()))?;
            Ok(())
        })
        .await?
    }
}

// ---------------------------------------------------------------------------
// Experience methods (not part of the Memory trait)
// ---------------------------------------------------------------------------

impl SqliteMemory {
    /// Store an experience, serialising JSON fields.
    pub async fn store_experience(&self, exp: &Experience) -> Result<()> {
        let exp = exp.clone();
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let context_json = serde_json::to_string(&exp.context)?;
            let attempts_json = serde_json::to_string(&exp.attempts)?;
            let gotchas_json = serde_json::to_string(&exp.gotchas)?;
            let tags_json = serde_json::to_string(&exp.tags)?;

            conn.execute(
                "INSERT OR REPLACE INTO experiences
                 (id, goal, context_json, attempts_json, solution, gotchas_json, tags_json, confidence, session_id, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                rusqlite::params![
                    exp.id,
                    exp.goal,
                    context_json,
                    attempts_json,
                    exp.solution,
                    gotchas_json,
                    tags_json,
                    exp.confidence,
                    exp.session_id,
                    exp.created_at,
                ],
            )?;
            Ok(())
        })
        .await?
    }

    /// Full-text search on experiences by goal / solution text.
    pub async fn search_experiences(&self, query: &str, limit: usize) -> Result<Vec<Experience>> {
        let conn = Arc::clone(&self.conn);
        let query = query.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();

            let Some(fts_query) = super::fts::build_match_query(&query) else {
                return Ok(vec![]);
            };

            let mut stmt = conn.prepare(
                "SELECT e.id, e.goal, e.context_json, e.attempts_json, e.solution,
                        e.gotchas_json, e.tags_json, e.confidence, e.session_id, e.created_at
                 FROM experiences_fts AS f
                 JOIN experiences AS e ON e.rowid = f.rowid
                 WHERE experiences_fts MATCH ?1
                 LIMIT ?2",
            )?;

            let rows = stmt.query_map(rusqlite::params![fts_query, limit as i64], |row| {
                let context_json: String = row.get(2)?;
                let attempts_json: String = row.get(3)?;
                let gotchas_json: String = row.get(5)?;
                let tags_json: String = row.get(6)?;

                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    context_json,
                    attempts_json,
                    row.get::<_, Option<String>>(4)?,
                    gotchas_json,
                    tags_json,
                    row.get::<_, f32>(7)?,
                    row.get::<_, Option<String>>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })?;

            let mut experiences = Vec::new();
            for row in rows {
                let (id, goal, ctx_json, att_json, solution, got_json, tag_json, confidence, session_id, created_at) = row?;
                let context: ExperienceContext = serde_json::from_str(&ctx_json).unwrap_or(ExperienceContext {
                    tools_used: vec![],
                    environment: String::new(),
                    constraints: String::new(),
                });
                let attempts = serde_json::from_str(&att_json).unwrap_or_default();
                let gotchas = serde_json::from_str(&got_json).unwrap_or_default();
                let tags = serde_json::from_str(&tag_json).unwrap_or_default();

                experiences.push(Experience {
                    id,
                    goal,
                    context,
                    attempts,
                    solution,
                    gotchas,
                    tags,
                    confidence,
                    session_id,
                    created_at,
                });
            }

            Ok(experiences)
        })
        .await?
    }

    /// List the most recent experiences.
    pub async fn list_experiences(&self, limit: usize) -> Result<Vec<Experience>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let mut stmt = conn.prepare(
                "SELECT id, goal, context_json, attempts_json, solution,
                        gotchas_json, tags_json, confidence, session_id, created_at
                 FROM experiences
                 ORDER BY created_at DESC
                 LIMIT ?1",
            )?;

            let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
                let context_json: String = row.get(2)?;
                let attempts_json: String = row.get(3)?;
                let gotchas_json: String = row.get(5)?;
                let tags_json: String = row.get(6)?;

                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    context_json,
                    attempts_json,
                    row.get::<_, Option<String>>(4)?,
                    gotchas_json,
                    tags_json,
                    row.get::<_, f32>(7)?,
                    row.get::<_, Option<String>>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })?;

            let mut experiences = Vec::new();
            for row in rows {
                let (id, goal, ctx_json, att_json, solution, got_json, tag_json, confidence, session_id, created_at) = row?;
                let context: ExperienceContext = serde_json::from_str(&ctx_json).unwrap_or(ExperienceContext {
                    tools_used: vec![],
                    environment: String::new(),
                    constraints: String::new(),
                });
                let attempts = serde_json::from_str(&att_json).unwrap_or_default();
                let gotchas = serde_json::from_str(&got_json).unwrap_or_default();
                let tags = serde_json::from_str(&tag_json).unwrap_or_default();

                experiences.push(Experience {
                    id,
                    goal,
                    context,
                    attempts,
                    solution,
                    gotchas,
                    tags,
                    confidence,
                    session_id,
                    created_at,
                });
            }

            Ok(experiences)
        })
        .await?
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use crate::memory::embedding::EmbeddingProvider;
    use crate::memory::traits::{Memory, MemoryEntry};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    /// Test embedder whose `name()` is NOT "noop" so the cache path
    /// in `SqliteMemory::store` actually runs. Each text gets a fresh
    /// fixed-length vector — the embedding values don't matter, only
    /// the content_hash key (derived from the text) does.
    struct CountingEmbedder {
        dims: usize,
        calls: AtomicUsize,
    }

    impl CountingEmbedder {
        fn new(dims: usize) -> Self {
            Self {
                dims,
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl EmbeddingProvider for CountingEmbedder {
        fn name(&self) -> &str {
            "counting-test"
        }
        fn dimensions(&self) -> usize {
            self.dims
        }
        async fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.0; self.dims])
        }
    }

    fn cache_row_count(mem: &SqliteMemory) -> usize {
        let conn = mem.conn.lock();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM embedding_cache", [], |r| r.get(0))
            .unwrap();
        count as usize
    }

    #[tokio::test]
    async fn cache_max_evicts_oldest_when_capacity_exceeded() {
        let dir = TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let embedder = Arc::new(CountingEmbedder::new(8));
        // cache_max=3 — after 4 distinct stores, one row should be
        // evicted (the least-recently-accessed).
        let mem = SqliteMemory::new(db, 0.7, 0.3, 3, embedder).unwrap();

        for i in 0..4 {
            let mut e = MemoryEntry::default();
            e.key = format!("k{i}");
            e.content = format!("unique-content-{i}");
            mem.store(e).await.unwrap();
        }

        assert_eq!(
            cache_row_count(&mem),
            3,
            "embedding_cache must shed rows past cache_max"
        );
    }

    #[tokio::test]
    async fn cache_max_zero_keeps_all_rows() {
        let dir = TempDir::new().unwrap();
        let db = dir.path().join("test.db");
        let embedder = Arc::new(CountingEmbedder::new(8));
        // cache_max=0 means "unlimited" per the doc comment on the field.
        let mem = SqliteMemory::new(db, 0.7, 0.3, 0, embedder).unwrap();

        for i in 0..5 {
            let mut e = MemoryEntry::default();
            e.key = format!("k{i}");
            e.content = format!("unique-content-{i}");
            mem.store(e).await.unwrap();
        }

        assert_eq!(
            cache_row_count(&mem),
            5,
            "cache_max=0 must keep every row"
        );
    }
}
