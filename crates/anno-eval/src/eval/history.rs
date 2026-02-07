//! Evaluation history tracking with optional SQLite index.
//!
//! This module provides:
//! - JSONL storage (primary, git-friendly, human-readable)
//! - Optional SQLite index for powerful queries (time-series, comparisons, aggregations)
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
//! use anno::eval::history::EvalHistory;
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

        // Initialize SQLite schema if needed
        if let Some(ref db_path) = sqlite_path {
            Self::init_sqlite(db_path)?;
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
        // Always write to JSONL first (source of truth)
        self.append_jsonl(entry)?;

        // Update SQLite index for fast queries
        if let Some(ref db_path) = self.sqlite_path {
            self.insert_sqlite(entry, db_path)?;
        }

        Ok(())
    }

    /// Append entry to JSONL file.
    fn append_jsonl(&self, entry: &EvalHistoryEntry) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.jsonl_path)?;

        let line = serde_json::to_string(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(file, "{}", line)?;
        Ok(())
    }

    /// Load all entries from JSONL file.
    pub fn load_all(&self) -> std::io::Result<Vec<EvalHistoryEntry>> {
        if !self.jsonl_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.jsonl_path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(entry) = serde_json::from_str::<EvalHistoryEntry>(&line) {
                entries.push(entry);
            }
        }

        Ok(entries)
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

    fn init_sqlite(db_path: &Path) -> std::io::Result<()> {
        use rusqlite::Connection;

        let conn = Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

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
        )
        .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        // Create indexes for common queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backend_dataset ON eval_results(backend, dataset)",
            [],
        )
        .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON eval_results(timestamp)",
            [],
        )
        .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_f1 ON eval_results(f1)", [])
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_backend_timestamp ON eval_results(backend, timestamp)",
            [],
        )
        .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        // Schema evolution: add git_commit column if not present.
        // This enables change-point detection tied to specific code versions.
        let _ = conn.execute(
            "ALTER TABLE eval_results ADD COLUMN git_commit TEXT",
            [],
        ); // Silently ignore if column already exists.

        Ok(())
    }

    fn insert_sqlite(&self, entry: &EvalHistoryEntry, db_path: &Path) -> std::io::Result<()> {
        use rusqlite::params;

        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| std::io::Error::other(format!("SQLite error: {}", e)))?;

        // Include git commit hash if available (for change-point detection).
        // Cached to avoid spawning `git` on every insert.
        let git_commit = cached_git_commit();

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
        let conn = rusqlite::Connection::open(db_path)
            .map_err(std::io::Error::other)?;

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
        let conn = rusqlite::Connection::open(db_path)
            .map_err(std::io::Error::other)?;

        // Include evaluation size (n) so we can match comparable observations.
        let mut stmt = conn
            .prepare(
                "SELECT backend, dataset, timestamp, f1, n FROM eval_results \
                 WHERE f1 IS NOT NULL AND error IS NULL \
                 ORDER BY backend, dataset, timestamp",
            )
            .map_err(std::io::Error::other)?;

        let mut cells: HashMap<(String, String), Vec<(String, f64, i64)>> = HashMap::new();
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?, row.get::<_, f64>(3)?,
                    row.get::<_, i64>(4)?))
            })
            .map_err(std::io::Error::other)?;
        for row in rows.flatten() {
            cells.entry((row.0, row.1)).or_default().push((row.2, row.3, row.4));
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
                .filter(|(_, _, n)| {
                    *n >= median_recent_n / 2 && *n <= median_recent_n * 2
                })
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
        let conn = rusqlite::Connection::open(db_path)
            .map_err(std::io::Error::other)?;

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
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?, row.get::<_, f64>(3)?))
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

    /// Rebuild SQLite index from JSONL file.
    ///
    /// Useful if SQLite gets corrupted or out of sync.
    pub fn rebuild_index(&self) -> std::io::Result<()> {
        if let Some(ref db_path) = self.sqlite_path {
            // Delete existing database
            if db_path.exists() {
                std::fs::remove_file(db_path)?;
            }

            // Reinitialize schema
            Self::init_sqlite(db_path)?;

            // Reload all entries and insert
            let entries = self.load_all()?;
            for entry in &entries {
                self.insert_sqlite(entry, db_path)?;
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
