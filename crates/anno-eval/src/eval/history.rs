//! Evaluation history tracking with optional SQLite index.
//!
//! This module provides:
//! - JSONL storage (primary, git-friendly, human-readable)
//! - Optional SQLite index for queries (time-series, comparisons, aggregations)
//!
//! # Design Philosophy
//!
//! - **JSONL is source of truth**: Always append to JSONL first
//! - **SQLite is queryable index**: Automatically maintained for fast queries
//! - **Both by default**: SQLite enabled with `eval` feature (no separate flag needed)
//!
//! # Usage
//!
//! ```rust,ignore
//! use anno_eval::eval::history::EvalHistory;
//!
//! let history = EvalHistory::new("reports/eval-results.jsonl")?;
//!
//! // Append result (writes to JSONL, optionally updates SQLite)
//! history.append_result(&result)?;
//!
//! // Query (uses SQLite if available, falls back to JSONL scan)
//! let recent = history.query_recent("gliner", 10)?;
//! let trends = history.query_trends("gliner", 30)?;
//! ```

use crate::eval::task_evaluator::TaskEvalResult;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Cached git commit hash (avoids spawning `git` on every SQLite insert).
fn cached_git_commit() -> Option<String> {
    static COMMIT: OnceLock<Option<String>> = OnceLock::new();
    COMMIT
        .get_or_init(|| {
            std::env::var("ANNO_GIT_COMMIT").ok().or_else(|| {
                std::process::Command::new("git")
                    .args(["rev-parse", "--short", "HEAD"])
                    .output()
                    .ok()
                    .and_then(|o| {
                        if o.status.success() {
                            String::from_utf8(o.stdout)
                                .ok()
                                .map(|s| s.trim().to_string())
                        } else {
                            None
                        }
                    })
            })
        })
        .clone()
}

/// Evaluation result entry for history tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalHistoryEntry {
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// Backend name
    pub backend: String,
    /// Dataset identifier
    pub dataset: String,
    /// Task type (NER, Coref, etc.)
    pub task: String,
    /// Random seed used
    pub seed: u64,
    /// F1 score (0.0-1.0)
    pub f1: Option<f64>,
    /// Precision (0.0-1.0)
    pub precision: Option<f64>,
    /// Recall (0.0-1.0)
    pub recall: Option<f64>,
    /// Number of examples evaluated
    pub n: usize,
    /// Duration in milliseconds
    pub duration_ms: Option<f64>,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional metadata (JSON string for flexibility)
    pub metadata: Option<String>,
}

/// On-disk JSONL representation of an evaluation entry.
///
/// Git provenance belongs to the observation, rather than the SQLite index: the index can be
/// rebuilt at a later checkout, but that must not change which source revision produced a result.
/// `git_commit` is optional so JSONL files written before provenance was added remain readable.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredEvalHistoryEntry {
    #[serde(flatten)]
    entry: EvalHistoryEntry,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    git_commit: Option<String>,
}

impl From<&TaskEvalResult> for EvalHistoryEntry {
    fn from(result: &TaskEvalResult) -> Self {
        let f1 = result.metrics.get("f1").copied();
        let precision = result.metrics.get("precision").copied();
        let recall = result.metrics.get("recall").copied();

        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            backend: result.backend.clone(),
            dataset: result.dataset.name().to_string(),
            task: format!("{:?}", result.task),
            seed: result.seed,
            f1,
            precision,
            recall,
            n: result.num_examples,
            duration_ms: result.duration_ms,
            error: result.error.clone(),
            metadata: serde_json::to_string(result).ok(),
        }
    }
}

/// Evaluation history manager.
///
/// Handles both JSONL storage (primary) and optional SQLite indexing.
pub struct EvalHistory {
    jsonl_path: PathBuf,
    sqlite_path: Option<PathBuf>,
}

impl EvalHistory {
    /// Create a new evaluation history manager.
    ///
    /// # Arguments
    ///
    /// * `jsonl_path` - Path to JSONL file (source of truth)
    pub fn new(jsonl_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let jsonl_path = jsonl_path.as_ref().to_path_buf();

        // Ensure parent directory exists
        if let Some(parent) = jsonl_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // SQLite index in same directory as JSONL
        let sqlite_path = jsonl_path
            .parent()
            .map(|p| p.join("eval-history.db"))
            .or_else(|| Some(PathBuf::from("eval-history.db")));

        // Initialize SQLite schema under the database-owned lock. A corrupt index is tolerated
        // here so callers can construct history and invoke the explicit rebuild recovery.
        if let Some(ref db_path) = sqlite_path {
            let _lock = Self::lock_sqlite(db_path)?;
            // Do not let SQLite open a database while an external WAL or journal exists: opening
            // a corrupt database can clean up sidecars that this process does not own.
            Self::reject_sqlite_sidecars(db_path)?;
            match Self::init_sqlite(db_path) {
                Ok(()) => {}
                Err(error) if Self::is_corrupt_sqlite_error(&error) => {}
                Err(error) => return Err(std::io::Error::other(format!("SQLite error: {error}"))),
            }
        }

        Ok(Self {
            jsonl_path,
            sqlite_path,
        })
    }

    /// Append a result to history.
    ///
    /// Writes to JSONL (primary) and optionally updates SQLite index.
    pub fn append_result(&self, result: &TaskEvalResult) -> std::io::Result<()> {
        let entry = EvalHistoryEntry::from(result);
        self.append_entry(&entry)
    }

    /// Append an entry to history.
    ///
    /// Lower-level method that accepts a pre-constructed entry.
    /// Useful when you need to customize the entry (e.g., set seed from config).
    pub fn append_entry(&self, entry: &EvalHistoryEntry) -> std::io::Result<()> {
        let _lock = self.lock_history()?;
        let git_commit = cached_git_commit();

        // Always write to JSONL first (source of truth)
        self.append_jsonl(entry, git_commit.as_deref())?;

        // Update SQLite index for fast queries
        if let Some(ref db_path) = self.sqlite_path {
            if let Err(error) = self.insert_sqlite(entry, git_commit.as_deref(), db_path) {
                return Err(std::io::Error::new(
                    error.kind(),
                    format!(
                        "evaluation entry was persisted to JSONL, but the SQLite index update failed: {error}; \
                         run `anno history --history-file {} rebuild` to recover the index",
                        self.jsonl_path.display()
                    ),
                ));
            }
        }

        Ok(())
    }

    /// Serialize JSONL and SQLite updates across processes using a persistent database sidecar.
    /// The lock is advisory, so external SQLite clients must not mutate this index directly.
    fn lock_history(&self) -> std::io::Result<File> {
        let db_path = self.sqlite_path.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Unsupported, "SQLite index is disabled")
        })?;
        Self::lock_sqlite(db_path)
    }

    fn lock_sqlite(db_path: &Path) -> std::io::Result<File> {
        let lock_path = Self::sqlite_lock_path(db_path);
        let lock = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)?;
        lock.lock_exclusive()?;
        Ok(lock)
    }

    fn sqlite_lock_path(db_path: &Path) -> PathBuf {
        db_path.with_extension("db.lock")
    }

    /// Append entry to JSONL file.
    fn append_jsonl(
        &self,
        entry: &EvalHistoryEntry,
        git_commit: Option<&str>,
    ) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .truncate(false)
            .open(&self.jsonl_path)?;

        let record = StoredEvalHistoryEntry {
            entry: entry.clone(),
            git_commit: git_commit.map(str::to_owned),
        };
        let line = serde_json::to_string(&record)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(file, "{}", line)?;
        file.sync_data()?;
        Ok(())
    }

    /// Load all entries from JSONL file.
    pub fn load_all(&self) -> std::io::Result<Vec<EvalHistoryEntry>> {
        if !self.jsonl_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.jsonl_path)?;
        let reader = BufReader::new(file);
        Self::read_jsonl_records(reader)
            .map(|records| records.into_iter().map(|record| record.entry).collect())
    }

    fn load_stored_entries_strict(&self) -> std::io::Result<Vec<StoredEvalHistoryEntry>> {
        if !self.jsonl_path.exists() {
            return Ok(Vec::new());
        }

        let reader = BufReader::new(File::open(&self.jsonl_path)?);
        let mut records = Vec::new();
        for (line_number, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record =
                serde_json::from_str::<StoredEvalHistoryEntry>(&line).map_err(|error| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "invalid history JSONL entry on line {}: {error}",
                            line_number + 1
                        ),
                    )
                })?;
            records.push(record);
        }
        Ok(records)
    }

    fn read_jsonl_records(reader: impl BufRead) -> std::io::Result<Vec<StoredEvalHistoryEntry>> {
        let mut records = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(record) = serde_json::from_str::<StoredEvalHistoryEntry>(&line) {
                records.push(record);
            }
        }
        Ok(records)
    }

    /// Get all unique backends in history.
    pub fn backends(&self) -> std::io::Result<Vec<String>> {
        let entries = self.load_all()?;
        let backends: std::collections::HashSet<String> =
            entries.iter().map(|e| e.backend.clone()).collect();
        let mut result: Vec<String> = backends.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Get all unique datasets in history.
    pub fn datasets(&self) -> std::io::Result<Vec<String>> {
        let entries = self.load_all()?;
        let datasets: std::collections::HashSet<String> =
            entries.iter().map(|e| e.dataset.clone()).collect();
        let mut result: Vec<String> = datasets.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Get statistics about the history.
    pub fn stats(&self) -> std::io::Result<HistoryStats> {
        let entries = self.load_all()?;

        let mut by_backend: HashMap<String, usize> = HashMap::new();
        let mut by_dataset: HashMap<String, usize> = HashMap::new();
        let mut total_f1: f64 = 0.0;
        let mut f1_count: usize = 0;

        for entry in &entries {
            *by_backend.entry(entry.backend.clone()).or_insert(0) += 1;
            *by_dataset.entry(entry.dataset.clone()).or_insert(0) += 1;
            if let Some(f1) = entry.f1 {
                total_f1 += f1;
                f1_count += 1;
            }
        }

        Ok(HistoryStats {
            total_entries: entries.len(),
            by_backend,
            by_dataset,
            avg_f1: if f1_count > 0 {
                Some(total_f1 / f1_count as f64)
            } else {
                None
            },
        })
    }

    fn init_sqlite(db_path: &Path) -> rusqlite::Result<()> {
        use rusqlite::Connection;

        let conn = Connection::open(db_path)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                backend TEXT NOT NULL,
                dataset TEXT NOT NULL,
                task TEXT NOT NULL,
                seed INTEGER NOT NULL,
                f1 REAL,
                precision REAL,
                recall REAL,
                n INTEGER NOT NULL,
                duration_ms REAL,
                error TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        // Create indexes for common queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backend_dataset ON eval_results(backend, dataset)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON eval_results(timestamp)",
            [],
        )?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_f1 ON eval_results(f1)", [])?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backend_timestamp ON eval_results(backend, timestamp)",
            [],
        )?;

        // Schema evolution: add git_commit column if not present.
        // This enables change-point detection tied to specific code versions.
        let _ = conn.execute("ALTER TABLE eval_results ADD COLUMN git_commit TEXT", []); // Silently ignore if column already exists.

        Ok(())
    }

    fn insert_sqlite(
        &self,
        entry: &EvalHistoryEntry,
        git_commit: Option<&str>,
        db_path: &Path,
    ) -> std::io::Result<()> {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        Self::insert_sqlite_into(&conn, entry, git_commit)
    }

    fn insert_sqlite_into(
        conn: &rusqlite::Connection,
        entry: &EvalHistoryEntry,
        git_commit: Option<&str>,
    ) -> std::io::Result<()> {
        use rusqlite::params;

        conn.execute(
            "INSERT INTO eval_results (
                timestamp, backend, dataset, task, seed,
                f1, precision, recall, n, duration_ms, error, metadata, git_commit
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                entry.timestamp,
                entry.backend,
                entry.dataset,
                entry.task,
                entry.seed,
                entry.f1,
                entry.precision,
                entry.recall,
                entry.n,
                entry.duration_ms,
                entry.error,
                entry.metadata,
                git_commit,
            ],
        )
        .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        Ok(())
    }

    /// Query recent results for a backend.
    ///
    /// Returns the N most recent results, ordered by timestamp descending.
    pub fn query_recent(
        &self,
        backend: &str,
        limit: usize,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        if let Some(ref db_path) = self.sqlite_path {
            return Self::query_recent_sqlite(db_path, backend, limit);
        }

        // Fallback to JSONL scan
        let mut entries = self.load_all()?;
        entries.retain(|e| e.backend == backend);
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(limit);
        Ok(entries)
    }

    fn query_recent_sqlite(
        db_path: &Path,
        backend: &str,
        limit: usize,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        use rusqlite::params;

        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
        let mut stmt = conn
            .prepare(
                "SELECT timestamp, backend, dataset, task, seed, f1, precision, recall, n, duration_ms, error, metadata
             FROM eval_results
             WHERE backend = ?1
             ORDER BY timestamp DESC
             LIMIT ?2",
            )
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        let rows = stmt
            .query_map(params![backend, limit as i64], |row| {
                Ok(EvalHistoryEntry {
                    timestamp: row.get(0)?,
                    backend: row.get(1)?,
                    dataset: row.get(2)?,
                    task: row.get(3)?,
                    seed: row.get(4)?,
                    f1: row.get(5)?,
                    precision: row.get(6)?,
                    recall: row.get(7)?,
                    n: row.get(8)?,
                    duration_ms: row.get(9)?,
                    error: row.get(10)?,
                    metadata: row.get(11)?,
                })
            })
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?);
        }
        Ok(entries)
    }

    /// Query best results for a backend-dataset combination.
    ///
    /// Returns results ordered by F1 score descending.
    pub fn query_best(
        &self,
        backend: &str,
        dataset: Option<&str>,
        limit: usize,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        if let Some(ref db_path) = self.sqlite_path {
            return Self::query_best_sqlite(db_path, backend, dataset, limit);
        }

        // Fallback to JSONL scan
        let mut entries = self.load_all()?;
        entries.retain(|e| {
            e.backend == backend
                && match dataset {
                    None => true,
                    Some(d) => e.dataset == d,
                }
        });
        entries.sort_by(|a, b| {
            let a_f1 = a.f1.unwrap_or(0.0);
            let b_f1 = b.f1.unwrap_or(0.0);
            b_f1.partial_cmp(&a_f1).unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(limit);
        Ok(entries)
    }

    fn query_best_sqlite(
        db_path: &Path,
        backend: &str,
        dataset: Option<&str>,
        limit: usize,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        use rusqlite::params;

        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        let mut entries = Vec::new();

        if let Some(ds) = dataset {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, backend, dataset, task, seed, f1, precision, recall, n, duration_ms, error, metadata
                     FROM eval_results
                     WHERE backend = ?1 AND dataset = ?2 AND f1 IS NOT NULL
                     ORDER BY f1 DESC
                     LIMIT ?3",
                )
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
            let rows = stmt
                .query_map(params![backend, ds, limit as i64], |row| {
                    Ok(EvalHistoryEntry {
                        timestamp: row.get(0)?,
                        backend: row.get(1)?,
                        dataset: row.get(2)?,
                        task: row.get(3)?,
                        seed: row.get(4)?,
                        f1: row.get(5)?,
                        precision: row.get(6)?,
                        recall: row.get(7)?,
                        n: row.get(8)?,
                        duration_ms: row.get(9)?,
                        error: row.get(10)?,
                        metadata: row.get(11)?,
                    })
                })
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

            for row in rows {
                entries
                    .push(row.map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?);
            }
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, backend, dataset, task, seed, f1, precision, recall, n, duration_ms, error, metadata
                     FROM eval_results
                     WHERE backend = ?1 AND f1 IS NOT NULL
                     ORDER BY f1 DESC
                     LIMIT ?2",
                )
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
            let rows = stmt
                .query_map(params![backend, limit as i64], |row| {
                    Ok(EvalHistoryEntry {
                        timestamp: row.get(0)?,
                        backend: row.get(1)?,
                        dataset: row.get(2)?,
                        task: row.get(3)?,
                        seed: row.get(4)?,
                        f1: row.get(5)?,
                        precision: row.get(6)?,
                        recall: row.get(7)?,
                        n: row.get(8)?,
                        duration_ms: row.get(9)?,
                        error: row.get(10)?,
                        metadata: row.get(11)?,
                    })
                })
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

            for row in rows {
                entries
                    .push(row.map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?);
            }
        }

        Ok(entries)
    }

    /// Query results by date range.
    ///
    /// Returns all results between `start_date` and `end_date` (inclusive).
    /// Dates should be in ISO 8601 format (e.g., "2024-01-01T00:00:00Z").
    pub fn query_by_date_range(
        &self,
        start_date: &str,
        end_date: &str,
        backend: Option<&str>,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        if let Some(ref db_path) = self.sqlite_path {
            return Self::query_by_date_range_sqlite(db_path, start_date, end_date, backend);
        }

        // Fallback to JSONL scan
        let mut entries = self.load_all()?;
        entries.retain(|e| {
            e.timestamp.as_str() >= start_date
                && e.timestamp.as_str() <= end_date
                && match backend {
                    None => true,
                    Some(b) => e.backend == b,
                }
        });
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(entries)
    }

    fn query_by_date_range_sqlite(
        db_path: &Path,
        start_date: &str,
        end_date: &str,
        backend: Option<&str>,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        use rusqlite::params;

        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        let mut entries = Vec::new();

        if let Some(b) = backend {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, backend, dataset, task, seed, f1, precision, recall, n, duration_ms, error, metadata
                     FROM eval_results
                     WHERE timestamp >= ?1 AND timestamp <= ?2 AND backend = ?3
                     ORDER BY timestamp DESC",
                )
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
            let rows = stmt
                .query_map(params![start_date, end_date, b], |row| {
                    Ok(EvalHistoryEntry {
                        timestamp: row.get(0)?,
                        backend: row.get(1)?,
                        dataset: row.get(2)?,
                        task: row.get(3)?,
                        seed: row.get(4)?,
                        f1: row.get(5)?,
                        precision: row.get(6)?,
                        recall: row.get(7)?,
                        n: row.get(8)?,
                        duration_ms: row.get(9)?,
                        error: row.get(10)?,
                        metadata: row.get(11)?,
                    })
                })
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

            for row in rows {
                entries
                    .push(row.map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?);
            }
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, backend, dataset, task, seed, f1, precision, recall, n, duration_ms, error, metadata
                     FROM eval_results
                     WHERE timestamp >= ?1 AND timestamp <= ?2
                     ORDER BY timestamp DESC",
                )
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
            let rows = stmt
                .query_map(params![start_date, end_date], |row| {
                    Ok(EvalHistoryEntry {
                        timestamp: row.get(0)?,
                        backend: row.get(1)?,
                        dataset: row.get(2)?,
                        task: row.get(3)?,
                        seed: row.get(4)?,
                        f1: row.get(5)?,
                        precision: row.get(6)?,
                        recall: row.get(7)?,
                        n: row.get(8)?,
                        duration_ms: row.get(9)?,
                        error: row.get(10)?,
                        metadata: row.get(11)?,
                    })
                })
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

            for row in rows {
                entries
                    .push(row.map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?);
            }
        }

        Ok(entries)
    }

    /// Compare two backends on the same dataset.
    ///
    /// Returns entries for both backends, ordered by timestamp.
    pub fn compare_backends(
        &self,
        backend1: &str,
        backend2: &str,
        dataset: Option<&str>,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        if let Some(ref db_path) = self.sqlite_path {
            return Self::compare_backends_sqlite(db_path, backend1, backend2, dataset);
        }

        // Fallback to JSONL scan
        let mut entries = self.load_all()?;
        entries.retain(|e| {
            (e.backend == backend1 || e.backend == backend2)
                && match dataset {
                    None => true,
                    Some(d) => e.dataset == d,
                }
        });
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(entries)
    }

    fn compare_backends_sqlite(
        db_path: &Path,
        backend1: &str,
        backend2: &str,
        dataset: Option<&str>,
    ) -> std::io::Result<Vec<EvalHistoryEntry>> {
        use rusqlite::params;

        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        let mut entries = Vec::new();

        if let Some(ds) = dataset {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, backend, dataset, task, seed, f1, precision, recall, n, duration_ms, error, metadata
                     FROM eval_results
                     WHERE (backend = ?1 OR backend = ?2) AND dataset = ?3
                     ORDER BY timestamp DESC",
                )
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
            let rows = stmt
                .query_map(params![backend1, backend2, ds], |row| {
                    Ok(EvalHistoryEntry {
                        timestamp: row.get(0)?,
                        backend: row.get(1)?,
                        dataset: row.get(2)?,
                        task: row.get(3)?,
                        seed: row.get(4)?,
                        f1: row.get(5)?,
                        precision: row.get(6)?,
                        recall: row.get(7)?,
                        n: row.get(8)?,
                        duration_ms: row.get(9)?,
                        error: row.get(10)?,
                        metadata: row.get(11)?,
                    })
                })
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

            for row in rows {
                entries
                    .push(row.map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?);
            }
        } else {
            let mut stmt = conn
                .prepare(
                    "SELECT timestamp, backend, dataset, task, seed, f1, precision, recall, n, duration_ms, error, metadata
                     FROM eval_results
                     WHERE backend = ?1 OR backend = ?2
                     ORDER BY timestamp DESC",
                )
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
            let rows = stmt
                .query_map(params![backend1, backend2], |row| {
                    Ok(EvalHistoryEntry {
                        timestamp: row.get(0)?,
                        backend: row.get(1)?,
                        dataset: row.get(2)?,
                        task: row.get(3)?,
                        seed: row.get(4)?,
                        f1: row.get(5)?,
                        precision: row.get(6)?,
                        recall: row.get(7)?,
                        n: row.get(8)?,
                        duration_ms: row.get(9)?,
                        error: row.get(10)?,
                        metadata: row.get(11)?,
                    })
                })
                .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

            for row in rows {
                entries
                    .push(row.map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?);
            }
        }

        Ok(entries)
    }

    /// Return observation counts per (backend, dataset) cell from the SQLite index.
    ///
    /// This is the quality matrix coverage map: each entry tells you how many times
    /// a (backend, dataset) pair has been evaluated.  Used by the Estimate strategy
    /// to prioritize cells with fewest observations.
    ///
    /// Falls back to JSONL scan if SQLite is unavailable.
    pub fn cell_observation_counts(&self) -> std::io::Result<HashMap<(String, String), u64>> {
        if let Some(ref db_path) = self.sqlite_path {
            if let Ok(conn) = rusqlite::Connection::open(db_path) {
                let mut stmt = conn
                    .prepare(
                        "SELECT backend, dataset, COUNT(*) FROM eval_results GROUP BY backend, dataset",
                    )
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                let mut counts = HashMap::new();
                let rows = stmt
                    .query_map([], |row| {
                        let backend: String = row.get(0)?;
                        let dataset: String = row.get(1)?;
                        let count: u64 = row.get(2)?;
                        Ok((backend, dataset, count))
                    })
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                for (b, d, c) in rows.flatten() {
                    counts.insert((b, d), c);
                }
                return Ok(counts);
            }
        }
        // Fallback: scan JSONL
        let entries = self.load_all()?;
        let mut counts = HashMap::new();
        for e in entries {
            *counts
                .entry((e.backend.clone(), e.dataset.clone()))
                .or_insert(0u64) += 1;
        }
        Ok(counts)
    }

    /// Return total observation counts per dataset across all backends.
    ///
    /// Used by the Estimate strategy to find least-observed datasets.
    pub fn dataset_observation_counts(&self) -> std::io::Result<HashMap<String, u64>> {
        if let Some(ref db_path) = self.sqlite_path {
            if let Ok(conn) = rusqlite::Connection::open(db_path) {
                let mut stmt = conn
                    .prepare("SELECT dataset, COUNT(*) FROM eval_results GROUP BY dataset")
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                let mut counts = HashMap::new();
                let rows = stmt
                    .query_map([], |row| {
                        let dataset: String = row.get(0)?;
                        let count: u64 = row.get(1)?;
                        Ok((dataset, count))
                    })
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                for (d, c) in rows.flatten() {
                    counts.insert(d, c);
                }
                return Ok(counts);
            }
        }
        let entries = self.load_all()?;
        let mut counts = HashMap::new();
        for e in entries {
            *counts.entry(e.dataset.clone()).or_insert(0u64) += 1;
        }
        Ok(counts)
    }

    /// Return total observation counts per backend across all datasets.
    pub fn backend_observation_counts(&self) -> std::io::Result<HashMap<String, u64>> {
        if let Some(ref db_path) = self.sqlite_path {
            if let Ok(conn) = rusqlite::Connection::open(db_path) {
                let mut stmt = conn
                    .prepare("SELECT backend, COUNT(*) FROM eval_results GROUP BY backend")
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                let mut counts = HashMap::new();
                let rows = stmt
                    .query_map([], |row| {
                        let backend: String = row.get(0)?;
                        let count: u64 = row.get(1)?;
                        Ok((backend, count))
                    })
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                for (b, c) in rows.flatten() {
                    counts.insert(b, c);
                }
                return Ok(counts);
            }
        }
        let entries = self.load_all()?;
        let mut counts = HashMap::new();
        for e in entries {
            *counts.entry(e.backend.clone()).or_insert(0u64) += 1;
        }
        Ok(counts)
    }

    /// Detect regressions: cells where recent F1 is significantly lower than historical.
    ///
    /// For each (backend, dataset) cell with enough observations, splits the data at
    /// a time threshold (default: median timestamp), computes the mean F1 for "before"
    /// and "after", and flags cells where the drop exceeds `min_drop`.
    ///
    /// This is the **change point detection** mechanism for the quality matrix:
    /// code changes that break a backend on a dataset will show up as a drop in
    /// recent F1 relative to historical F1.
    ///
    /// Returns a list of `(backend, dataset, old_mean, new_mean, drop, n_old, n_new)`.
    pub fn detect_regressions(
        &self,
        min_observations: u64,
        min_drop: f64,
    ) -> std::io::Result<Vec<RegressionAlert>> {
        let Some(ref db_path) = self.sqlite_path else {
            return Ok(Vec::new());
        };
        let conn = rusqlite::Connection::open(db_path).map_err(std::io::Error::other)?;

        // For each cell, split observations by the median timestamp and compare means.
        let mut stmt = conn
            .prepare(
                "SELECT backend, dataset, timestamp, f1 FROM eval_results \
                 WHERE f1 IS NOT NULL AND error IS NULL \
                 ORDER BY backend, dataset, timestamp",
            )
            .map_err(std::io::Error::other)?;

        let mut cells: HashMap<(String, String), Vec<(String, f64)>> = HashMap::new();
        let rows = stmt
            .query_map([], |row| {
                let b: String = row.get(0)?;
                let d: String = row.get(1)?;
                let ts: String = row.get(2)?;
                let f1: f64 = row.get(3)?;
                Ok((b, d, ts, f1))
            })
            .map_err(std::io::Error::other)?;
        for row in rows.flatten() {
            let (b, d, ts, f1) = row;
            cells.entry((b, d)).or_default().push((ts, f1));
        }

        let mut alerts = Vec::new();
        for ((backend, dataset), obs) in &cells {
            let n = obs.len() as u64;
            if n < min_observations {
                continue;
            }
            // Split at the median index (first half = historical, second half = recent).
            let mid = obs.len() / 2;
            let old_vals: Vec<f64> = obs[..mid].iter().map(|(_, f)| *f).collect();
            let new_vals: Vec<f64> = obs[mid..].iter().map(|(_, f)| *f).collect();

            if old_vals.is_empty() || new_vals.is_empty() {
                continue;
            }

            let old_mean = old_vals.iter().sum::<f64>() / old_vals.len() as f64;
            let new_mean = new_vals.iter().sum::<f64>() / new_vals.len() as f64;
            let drop = old_mean - new_mean;

            if drop >= min_drop {
                alerts.push(RegressionAlert {
                    backend: backend.clone(),
                    dataset: dataset.clone(),
                    old_mean,
                    new_mean,
                    drop,
                    n_old: old_vals.len() as u64,
                    n_new: new_vals.len() as u64,
                    split_timestamp: obs[mid].0.clone(),
                });
            }
        }

        alerts.sort_by(|a, b| b.drop.total_cmp(&a.drop));
        Ok(alerts)
    }

    /// Detect regressions using a recent-window comparison with sample-size normalization.
    ///
    /// For each (backend, dataset) cell, compares the last `recent_n` observations to all
    /// earlier observations.  Only compares observations with similar evaluation size (n)
    /// to avoid false alarms from the n-dependent F1 variance (small n = high F1 variance).
    ///
    /// Uses Cohen's d effect size: d = (old_mean - new_mean) / pooled_sd.
    /// Flags cells where d > `min_effect_size` (default 0.8 = "large" effect).
    pub fn detect_regressions_recent(
        &self,
        recent_n: usize,
        min_effect_size: f64,
        min_total: u64,
    ) -> std::io::Result<Vec<RegressionAlert>> {
        let Some(ref db_path) = self.sqlite_path else {
            return Ok(Vec::new());
        };
        let conn = rusqlite::Connection::open(db_path).map_err(std::io::Error::other)?;

        // Include evaluation size (n) so we can match comparable observations.
        let mut stmt = conn
            .prepare(
                "SELECT backend, dataset, timestamp, f1, n FROM eval_results \
                 WHERE f1 IS NOT NULL AND error IS NULL \
                 ORDER BY backend, dataset, timestamp",
            )
            .map_err(std::io::Error::other)?;

        // (backend, dataset) → [(timestamp, f1, n)]
        type CellKey = (String, String);
        type CellRow = (String, f64, i64);
        let mut cells: HashMap<CellKey, Vec<CellRow>> = HashMap::new();
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })
            .map_err(std::io::Error::other)?;
        for row in rows.flatten() {
            cells
                .entry((row.0, row.1))
                .or_default()
                .push((row.2, row.3, row.4));
        }

        let mut alerts = Vec::new();
        for ((backend, dataset), obs) in &cells {
            if (obs.len() as u64) < min_total || obs.len() <= recent_n {
                continue;
            }
            let split = obs.len() - recent_n;

            // Determine the typical evaluation size in the recent window.
            let recent_n_vals: Vec<i64> = obs[split..].iter().map(|(_, _, n)| *n).collect();
            let median_recent_n = {
                let mut sorted = recent_n_vals.clone();
                sorted.sort();
                sorted[sorted.len() / 2]
            };

            // Only compare against historical observations with similar evaluation size
            // (within 2x) to avoid the n-dependent variance confound.
            let old: Vec<f64> = obs[..split]
                .iter()
                .filter(|(_, _, n)| *n >= median_recent_n / 2 && *n <= median_recent_n * 2)
                .map(|(_, f, _)| *f)
                .collect();
            let new: Vec<f64> = obs[split..].iter().map(|(_, f, _)| *f).collect();

            if old.len() < 3 || new.is_empty() {
                continue; // not enough comparable historical observations
            }

            let old_mean = old.iter().sum::<f64>() / old.len() as f64;
            let new_mean = new.iter().sum::<f64>() / new.len() as f64;
            let drop = old_mean - new_mean;
            if drop <= 0.0 {
                continue;
            }

            let old_var = old.iter().map(|x| (x - old_mean).powi(2)).sum::<f64>()
                / (old.len() as f64 - 1.0).max(1.0);
            let new_var = new.iter().map(|x| (x - new_mean).powi(2)).sum::<f64>()
                / (new.len() as f64 - 1.0).max(1.0);
            let pooled_sd = ((old_var + new_var) / 2.0).sqrt();
            if pooled_sd < 1e-12 {
                continue;
            }
            let d = drop / pooled_sd;

            if d >= min_effect_size {
                alerts.push(RegressionAlert {
                    backend: backend.clone(),
                    dataset: dataset.clone(),
                    old_mean,
                    new_mean,
                    drop,
                    n_old: old.len() as u64,
                    n_new: new.len() as u64,
                    split_timestamp: obs[split].0.clone(),
                });
            }
        }

        alerts.sort_by(|a, b| b.drop.total_cmp(&a.drop));
        Ok(alerts)
    }

    /// Detect regressions between two git commits.
    ///
    /// This is the most precise change-point detection: it directly compares F1 scores
    /// from evaluations tagged with `old_commit` to those tagged with `new_commit`.
    /// Only works after the git_commit column is populated (evaluations run after this
    /// code change).
    pub fn detect_regressions_by_commit(
        &self,
        old_commit: &str,
        new_commit: &str,
        min_drop: f64,
    ) -> std::io::Result<Vec<RegressionAlert>> {
        let Some(ref db_path) = self.sqlite_path else {
            return Ok(Vec::new());
        };
        let conn = rusqlite::Connection::open(db_path).map_err(std::io::Error::other)?;

        // Check if the git_commit column exists.
        let has_column = conn
            .prepare("SELECT git_commit FROM eval_results LIMIT 0")
            .is_ok();
        if !has_column {
            return Ok(Vec::new());
        }

        let mut stmt = conn
            .prepare(
                "SELECT backend, dataset, git_commit, f1 FROM eval_results \
                 WHERE f1 IS NOT NULL AND error IS NULL \
                 AND git_commit IN (?1, ?2) \
                 ORDER BY backend, dataset",
            )
            .map_err(std::io::Error::other)?;

        let mut old_cells: HashMap<(String, String), Vec<f64>> = HashMap::new();
        let mut new_cells: HashMap<(String, String), Vec<f64>> = HashMap::new();
        let rows = stmt
            .query_map(rusqlite::params![old_commit, new_commit], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, f64>(3)?,
                ))
            })
            .map_err(std::io::Error::other)?;
        for row in rows.flatten() {
            let (b, d, commit, f1) = row;
            if commit == old_commit {
                old_cells.entry((b, d)).or_default().push(f1);
            } else {
                new_cells.entry((b, d)).or_default().push(f1);
            }
        }

        let mut alerts = Vec::new();
        for ((backend, dataset), old_vals) in &old_cells {
            let Some(new_vals) = new_cells.get(&(backend.clone(), dataset.clone())) else {
                continue;
            };
            if old_vals.is_empty() || new_vals.is_empty() {
                continue;
            }
            let old_mean = old_vals.iter().sum::<f64>() / old_vals.len() as f64;
            let new_mean = new_vals.iter().sum::<f64>() / new_vals.len() as f64;
            let drop = old_mean - new_mean;
            if drop >= min_drop {
                alerts.push(RegressionAlert {
                    backend: backend.clone(),
                    dataset: dataset.clone(),
                    old_mean,
                    new_mean,
                    drop,
                    n_old: old_vals.len() as u64,
                    n_new: new_vals.len() as u64,
                    split_timestamp: format!("{} -> {}", old_commit, new_commit),
                });
            }
        }

        alerts.sort_by(|a, b| b.drop.total_cmp(&a.drop));
        Ok(alerts)
    }

    fn rebuild_initialized_index_transactionally(
        db_path: &Path,
        entries: &[StoredEvalHistoryEntry],
    ) -> std::io::Result<()> {
        let mut conn = rusqlite::Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {e}")))?;
        let transaction = conn.transaction().map_err(std::io::Error::other)?;
        transaction
            .execute("DELETE FROM eval_results", [])
            .map_err(std::io::Error::other)?;

        // Insert recorded provenance, never resolving the current checkout. Historic JSONL
        // lines without provenance intentionally remain NULL.
        for record in entries {
            Self::insert_sqlite_into(&transaction, &record.entry, record.git_commit.as_deref())?;
        }
        transaction.commit().map_err(std::io::Error::other)
    }

    fn is_corrupt_sqlite_error(error: &rusqlite::Error) -> bool {
        matches!(
            error.sqlite_error_code(),
            Some(rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase)
        )
    }

    fn sqlite_sidecar_paths(db_path: &Path) -> Vec<PathBuf> {
        ["-wal", "-shm", "-journal"]
            .into_iter()
            .map(|suffix| {
                let mut path = db_path.as_os_str().to_os_string();
                path.push(suffix);
                PathBuf::from(path)
            })
            .collect()
    }

    fn reject_sqlite_sidecars(db_path: &Path) -> std::io::Result<()> {
        let sidecars: Vec<_> = Self::sqlite_sidecar_paths(db_path)
            .into_iter()
            .filter(|path| path.exists())
            .collect();
        if sidecars.is_empty() {
            return Ok(());
        }
        Err(std::io::Error::other(format!(
            "refusing SQLite index rebuild while sidecars exist: {}",
            sidecars
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }

    fn reserve_rebuild_path(db_path: &Path, kind: &str) -> std::io::Result<PathBuf> {
        let parent = db_path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = db_path.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "SQLite index has no file name",
            )
        })?;
        for attempt in 0..100 {
            let candidate = parent.join(format!(
                ".{}.{kind}.{}.{}",
                file_name.to_string_lossy(),
                std::process::id(),
                attempt
            ));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(_) => return Ok(candidate),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not reserve a SQLite recovery path",
        ))
    }

    fn rebuild_corrupt_index(
        db_path: &Path,
        entries: &[StoredEvalHistoryEntry],
    ) -> std::io::Result<()> {
        Self::reject_sqlite_sidecars(db_path)?;

        let replacement = Self::reserve_rebuild_path(db_path, "rebuild")?;
        let rebuild_result = Self::init_sqlite(&replacement)
            .map_err(std::io::Error::other)
            .and_then(|_| Self::rebuild_initialized_index_transactionally(&replacement, entries));
        if let Err(error) = rebuild_result {
            let _ = std::fs::remove_file(&replacement);
            return Err(error);
        }

        // Recheck immediately before replacement: our sidecar lock serializes anno writers, and
        // this guard declines recovery if an external SQLite client created WAL state meanwhile.
        if let Err(error) = Self::reject_sqlite_sidecars(db_path) {
            let _ = std::fs::remove_file(&replacement);
            return Err(error);
        }

        if let Err(error) = std::fs::rename(&replacement, db_path) {
            let _ = std::fs::remove_file(&replacement);
            return Err(error);
        }

        Ok(())
    }

    /// Rebuild SQLite index from JSONL file.
    ///
    /// A sidecar lock serializes anno appenders and rebuilders. Healthy indexes are refreshed in
    /// place in one transaction. A corrupt index is replaced only after a fresh index is built
    /// from fully validated JSONL, and recovery rejects active SQLite sidecars.
    pub fn rebuild_index(&self) -> std::io::Result<()> {
        let _lock = self.lock_history()?;
        if let Some(ref db_path) = self.sqlite_path {
            // Validate the complete source before touching the current index. `load_all` remains
            // permissive for historic inspection, but rebuild must never silently drop a line.
            let entries = self.load_stored_entries_strict()?;

            // Check before opening SQLite: opening a corrupt database can clean up unowned WAL
            // state, so an external sidecar must stop recovery before SQLite sees the database.
            Self::reject_sqlite_sidecars(db_path)?;

            match Self::init_sqlite(db_path) {
                Ok(()) => Self::rebuild_initialized_index_transactionally(db_path, &entries)?,
                Err(error) if Self::is_corrupt_sqlite_error(&error) => {
                    Self::rebuild_corrupt_index(db_path, &entries)?;
                }
                Err(error) => return Err(std::io::Error::other(format!("SQLite error: {error}"))),
            }

            eprintln!(
                "[history] Rebuilt SQLite index with {} entries",
                entries.len()
            );
        }
        Ok(())
    }
}

/// A detected regression in a (backend, dataset) cell.
#[derive(Debug, Clone)]
pub struct RegressionAlert {
    /// Backend name.
    pub backend: String,
    /// Dataset name.
    pub dataset: String,
    /// Mean F1 in the historical (older) half.
    pub old_mean: f64,
    /// Mean F1 in the recent (newer) half.
    pub new_mean: f64,
    /// Size of the drop (old_mean - new_mean, positive = regression).
    pub drop: f64,
    /// Number of observations in the historical half.
    pub n_old: u64,
    /// Number of observations in the recent half.
    pub n_new: u64,
    /// Timestamp where the split occurs.
    pub split_timestamp: String,
}

/// Statistics about evaluation history.
#[derive(Debug, Clone)]
pub struct HistoryStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Count per backend
    pub by_backend: HashMap<String, usize>,
    /// Count per dataset
    pub by_dataset: HashMap<String, usize>,
    /// Average F1 score across all entries
    pub avg_f1: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_entry(seed: u64, n: usize) -> EvalHistoryEntry {
        EvalHistoryEntry {
            timestamp: "2026-08-13T20:00:00Z".to_string(),
            backend: "test-backend".to_string(),
            dataset: "test-dataset".to_string(),
            task: "NER".to_string(),
            seed,
            f1: Some(0.85),
            precision: Some(0.9),
            recall: Some(0.8),
            n,
            duration_ms: Some(10.0),
            error: None,
            metadata: Some("{}".to_string()),
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn sqlite_round_trips_largest_signed_integer_as_unsigned_fields() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let history =
            EvalHistory::new(temp.path().join("history.jsonl")).expect("failed to create history");
        let boundary = i64::MAX as u64;
        let entry = test_entry(boundary, boundary as usize);

        history.append_entry(&entry).expect("append failed");

        let stored = history
            .query_recent("test-backend", 1)
            .expect("query failed")
            .pop()
            .expect("missing stored entry");
        assert_eq!(stored.seed, boundary);
        assert_eq!(stored.n, boundary as usize);
    }

    #[test]
    fn sqlite_rejects_seed_above_signed_integer_range() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let history =
            EvalHistory::new(temp.path().join("history.jsonl")).expect("failed to create history");
        let entry = test_entry(i64::MAX as u64 + 1, 1);

        let error = history
            .append_entry(&entry)
            .expect_err("oversized seed must not be truncated");

        assert!(error.to_string().contains("out of range"));
        assert!(history
            .query_recent("test-backend", 1)
            .expect("query failed")
            .is_empty());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn sqlite_rejects_count_above_signed_integer_range() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let history =
            EvalHistory::new(temp.path().join("history.jsonl")).expect("failed to create history");
        let entry = test_entry(1, i64::MAX as usize + 1);

        let error = history
            .append_entry(&entry)
            .expect_err("oversized count must not be truncated");

        assert!(error.to_string().contains("out of range"));
        assert!(history
            .query_recent("test-backend", 1)
            .expect("query failed")
            .is_empty());
    }

    #[test]
    fn missing_sqlite_table_leaves_jsonl_recoverable_by_rebuild() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let history =
            EvalHistory::new(temp.path().join("history.jsonl")).expect("failed to create history");
        let db_path = history
            .sqlite_path
            .as_ref()
            .expect("SQLite index is enabled");
        let conn = rusqlite::Connection::open(db_path).expect("failed to open SQLite index");
        conn.execute("DROP TABLE eval_results", [])
            .expect("failed to remove SQLite table");

        let error = history
            .append_entry(&test_entry(42, 100))
            .expect_err("missing SQLite table must make index insertion fail");
        assert!(error.to_string().contains("persisted to JSONL"));
        assert_eq!(history.load_all().expect("failed to load JSONL").len(), 1);

        drop(conn);
        history.rebuild_index().expect("rebuild failed");
        assert_eq!(
            history
                .query_recent("test-backend", 10)
                .expect("query failed")
                .len(),
            1
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn failed_rebuild_preserves_existing_index() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let history =
            EvalHistory::new(temp.path().join("history.jsonl")).expect("failed to create history");
        history
            .append_entry(&test_entry(1, 100))
            .expect("failed to append valid entry");

        let oversized = test_entry(2, i64::MAX as usize + 1);
        history
            .append_entry(&oversized)
            .expect_err("oversized entry must fail SQLite insertion");

        let error = history
            .rebuild_index()
            .expect_err("rebuild must reject an unindexable JSONL entry");
        assert!(error.to_string().contains("out of range"));
        let indexed = history
            .query_recent("test-backend", 10)
            .expect("failed to query preserved index");
        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].seed, 1);
    }

    #[test]
    fn malformed_jsonl_prevents_rebuild_without_discarding_index() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");
        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");
        history
            .append_entry(&test_entry(1, 100))
            .expect("failed to append valid entry");

        let valid_jsonl = std::fs::read_to_string(&jsonl_path).expect("failed to read JSONL");
        std::fs::write(&jsonl_path, format!("{valid_jsonl}not JSON\n"))
            .expect("failed to add malformed JSONL line");

        let error = history
            .rebuild_index()
            .expect_err("malformed JSONL must prevent rebuild");
        assert!(error
            .to_string()
            .contains("invalid history JSONL entry on line 2"));
        assert_eq!(
            history
                .query_recent("test-backend", 10)
                .expect("failed to query preserved index")
                .len(),
            1
        );
    }

    #[test]
    fn rebuild_replaces_corrupt_sqlite_index_from_valid_jsonl() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");
        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");
        history
            .append_entry(&test_entry(7, 100))
            .expect("failed to append valid entry");
        let db_path = history
            .sqlite_path
            .as_ref()
            .expect("SQLite index is enabled")
            .clone();
        drop(history);
        std::fs::write(&db_path, b"this is not a SQLite database")
            .expect("failed to corrupt SQLite index fixture");

        let history = EvalHistory::new(&jsonl_path)
            .expect("corrupt SQLite index must not prevent recovery construction");

        history
            .rebuild_index()
            .expect("failed to rebuild corrupt SQLite index");
        let indexed = history
            .query_recent("test-backend", 10)
            .expect("failed to query recovered index");
        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].seed, 7);
    }

    #[test]
    fn corrupt_rebuild_rejects_active_sqlite_sidecars() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let history =
            EvalHistory::new(temp.path().join("history.jsonl")).expect("failed to create history");
        history
            .append_entry(&test_entry(7, 100))
            .expect("failed to append valid entry");
        let db_path = history
            .sqlite_path
            .as_ref()
            .expect("SQLite index is enabled");
        std::fs::write(db_path, b"this is not a SQLite database")
            .expect("failed to corrupt SQLite index fixture");
        let mut wal_path = db_path.as_os_str().to_os_string();
        wal_path.push("-wal");
        let wal_path = PathBuf::from(wal_path);
        std::fs::write(&wal_path, b"external WAL state")
            .expect("failed to create SQLite sidecar fixture");

        let error = history
            .rebuild_index()
            .expect_err("active sidecars must prevent replacement");
        assert!(error.to_string().contains("sidecars exist"));
        assert_eq!(
            std::fs::read(db_path).expect("failed to inspect preserved corrupt index"),
            b"this is not a SQLite database"
        );
        assert_eq!(
            std::fs::read(wal_path).expect("failed to inspect preserved SQLite sidecar"),
            b"external WAL state"
        );
    }

    #[test]
    fn database_owned_lock_excludes_second_handle_until_drop() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let db_path = temp.path().join("eval-history.db");
        let lock = EvalHistory::lock_sqlite(&db_path).expect("failed to acquire first lock");
        let lock_path = EvalHistory::sqlite_lock_path(&db_path);

        let blocked = std::thread::spawn(move || {
            let second = OpenOptions::new()
                .read(true)
                .write(true)
                .open(lock_path)
                .expect("failed to open second lock handle");
            fs2::FileExt::try_lock_exclusive(&second).is_err()
        })
        .join()
        .expect("second lock thread panicked");
        assert!(blocked, "second handle must not acquire the database lock");

        drop(lock);
        let second = EvalHistory::lock_sqlite(&db_path)
            .expect("second handle must acquire lock after first drops");
        drop(second);
    }

    #[test]
    fn rebuild_preserves_recorded_commit_and_leaves_legacy_records_unattributed() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");
        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");
        let recorded = test_entry(1, 100);
        let legacy = test_entry(2, 100);

        std::fs::write(
            &jsonl_path,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&StoredEvalHistoryEntry {
                    entry: recorded,
                    git_commit: Some("historic-commit".to_string()),
                })
                .expect("failed to serialize recorded entry"),
                serde_json::to_string(&legacy).expect("failed to serialize legacy entry"),
            ),
        )
        .expect("failed to write fixture JSONL");

        history.rebuild_index().expect("rebuild failed");
        let conn = rusqlite::Connection::open(
            history
                .sqlite_path
                .as_ref()
                .expect("SQLite index is enabled"),
        )
        .expect("failed to open SQLite index");
        let commits: Vec<Option<String>> = conn
            .prepare("SELECT git_commit FROM eval_results ORDER BY seed")
            .expect("failed to prepare commit query")
            .query_map([], |row| row.get(0))
            .expect("failed to query commits")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("failed to read commits");

        assert_eq!(commits, vec![Some("historic-commit".to_string()), None]);
    }

    #[test]
    fn test_append_and_load() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");

        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");

        // Create a test result
        let result = TaskEvalResult {
            task: crate::eval::task_mapping::Task::NER,
            dataset: crate::eval::loader::DatasetId::WikiGold,
            backend: "test-backend".to_string(),
            backend_display: None,
            seed: 42,
            success: true,
            error: None,
            metrics: {
                let mut m = HashMap::new();
                m.insert("f1".to_string(), 0.85);
                m.insert("precision".to_string(), 0.90);
                m.insert("recall".to_string(), 0.80);
                m
            },
            num_examples: 100,
            duration_ms: Some(5000.0),
            label_shift: None,
            robustness: None,
            stratified: None,
            confidence_intervals: None,
            kb_version: None,
        };

        // Append result
        history.append_result(&result).expect("append failed");

        // Load and verify
        let entries = history.load_all().expect("load failed");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].backend, "test-backend");
        assert_eq!(entries[0].f1, Some(0.85));
    }

    #[test]
    fn test_stats() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");

        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");

        // Add multiple results
        for i in 0..5 {
            let result = TaskEvalResult {
                task: crate::eval::task_mapping::Task::NER,
                dataset: crate::eval::loader::DatasetId::WikiGold,
                backend: format!("backend-{}", i % 2),
                backend_display: None,
                seed: 42,
                success: true,
                error: None,
                metrics: {
                    let mut m = HashMap::new();
                    m.insert("f1".to_string(), 0.8 + (i as f64 * 0.01));
                    m
                },
                num_examples: 100,
                duration_ms: Some(5000.0),
                label_shift: None,
                robustness: None,
                stratified: None,
                confidence_intervals: None,
                kb_version: None,
            };
            history.append_result(&result).expect("append failed");
        }

        let stats = history.stats().expect("stats failed");
        assert_eq!(stats.total_entries, 5);
        assert_eq!(stats.by_backend.len(), 2);
        assert!(stats.avg_f1.is_some());
    }

    #[test]
    fn test_query_recent() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");

        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");

        // Add results with different backends
        for i in 0..10 {
            let result = TaskEvalResult {
                task: crate::eval::task_mapping::Task::NER,
                dataset: crate::eval::loader::DatasetId::WikiGold,
                backend: if i % 2 == 0 {
                    "backend-a".to_string()
                } else {
                    "backend-b".to_string()
                },
                backend_display: None,
                seed: 42,
                success: true,
                error: None,
                metrics: {
                    let mut m = HashMap::new();
                    m.insert("f1".to_string(), 0.8);
                    m
                },
                num_examples: 100,
                duration_ms: Some(1000.0),
                label_shift: None,
                robustness: None,
                stratified: None,
                confidence_intervals: None,
                kb_version: None,
            };
            history.append_result(&result).expect("append failed");
        }

        let recent = history.query_recent("backend-a", 3).expect("query failed");
        assert_eq!(recent.len(), 3);
        assert!(recent.iter().all(|e| e.backend == "backend-a"));
    }

    #[test]
    fn test_query_best() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");

        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");

        // Add results with different F1 scores
        for i in 0..5 {
            let result = TaskEvalResult {
                task: crate::eval::task_mapping::Task::NER,
                dataset: crate::eval::loader::DatasetId::WikiGold,
                backend: "test-backend".to_string(),
                backend_display: None,
                seed: 42,
                success: true,
                error: None,
                metrics: {
                    let mut m = HashMap::new();
                    m.insert("f1".to_string(), 0.5 + (i as f64 * 0.1));
                    m
                },
                num_examples: 100,
                duration_ms: Some(1000.0),
                label_shift: None,
                robustness: None,
                stratified: None,
                confidence_intervals: None,
                kb_version: None,
            };
            history.append_result(&result).expect("append failed");
        }

        let best = history
            .query_best("test-backend", None, 3)
            .expect("query failed");
        assert_eq!(best.len(), 3);
        // Should be sorted by F1 descending
        assert!(best[0].f1.unwrap() > best[1].f1.unwrap());
        assert!(best[1].f1.unwrap() > best[2].f1.unwrap());
    }

    #[test]
    fn test_backends_and_datasets() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");

        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");

        // Add results with different backends and datasets
        let backends = ["backend-a", "backend-b", "backend-c"];

        for backend in backends.iter() {
            let result = TaskEvalResult {
                task: crate::eval::task_mapping::Task::NER,
                dataset: crate::eval::loader::DatasetId::WikiGold,
                backend: backend.to_string(),
                backend_display: None,
                seed: 42,
                success: true,
                error: None,
                metrics: {
                    let mut m = HashMap::new();
                    m.insert("f1".to_string(), 0.8);
                    m
                },
                num_examples: 100,
                duration_ms: Some(1000.0),
                label_shift: None,
                robustness: None,
                stratified: None,
                confidence_intervals: None,
                kb_version: None,
            };
            history.append_result(&result).expect("append failed");
        }

        let backends_list = history.backends().expect("backends failed");
        assert_eq!(backends_list.len(), 3);
        assert!(backends_list.contains(&"backend-a".to_string()));

        let datasets_list = history.datasets().expect("datasets failed");
        assert!(!datasets_list.is_empty());
    }

    #[test]
    fn test_rebuild_index() {
        let temp = TempDir::new().expect("failed to create temp dir");
        let jsonl_path = temp.path().join("history.jsonl");

        let history = EvalHistory::new(&jsonl_path).expect("failed to create history");

        // Add a result
        let result = TaskEvalResult {
            task: crate::eval::task_mapping::Task::NER,
            dataset: crate::eval::loader::DatasetId::WikiGold,
            backend: "test-backend".to_string(),
            backend_display: None,
            seed: 42,
            success: true,
            error: None,
            metrics: {
                let mut m = HashMap::new();
                m.insert("f1".to_string(), 0.85);
                m
            },
            num_examples: 100,
            duration_ms: Some(1000.0),
            label_shift: None,
            robustness: None,
            stratified: None,
            confidence_intervals: None,
            kb_version: None,
        };
        history.append_result(&result).expect("append failed");

        // Rebuild index
        history.rebuild_index().expect("rebuild failed");

        // Verify data is still accessible
        let stats = history.stats().expect("stats failed");
        assert_eq!(stats.total_entries, 1);
    }
}
