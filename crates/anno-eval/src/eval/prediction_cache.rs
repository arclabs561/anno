//! Prediction caching for incremental evaluation.
//!
//! Caches NER predictions keyed by (text_hash, backend_version, labels_hash).
//! This allows re-running evaluations against new gold data without re-inference.
//!
//! # Use Cases
//!
//! 1. **Same backend, different datasets**: Predictions cached per-text, reused
//! 2. **Same dataset, different backends**: Gold parsed once, predictions per-backend
//! 3. **Re-scoring with updated metrics**: No re-inference needed
//! 4. **Local iteration**: reuse predictions across repeated runs
//!
//! # Important correctness note
//!
//! This cache is inherently **implementation-sensitive**:
//! - changes to tokenization/normalization/decoding can change predictions
//! - some baselines use a coarse `Model::version()` (default is `"1"`)
//!
//! Treat cache reuse in CI with caution; prefer invalidating by commit id / feature set.
//!
//! # Cache Key Design
//!
//! ```text
//! key = sha256(text || backend_name || backend_version || sorted(labels))
//! ```
//!
//! # Storage Format
//!
//! JSONL for append-only, git-friendly storage:
//! ```json
//! {"key":"abc123","text_hash":"...","backend":"nuner","version":"1.0","predictions":[...]}
//! ```

use anno::Entity;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// A cached prediction entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPrediction {
    /// Hash of the input text.
    pub text_hash: String,
    /// Backend name (e.g., "nuner", "gliner2").
    pub backend: String,
    /// Backend version or model hash for invalidation.
    pub version: String,
    /// Sorted label set used for extraction.
    pub labels: Vec<String>,
    /// Predicted entities.
    pub predictions: Vec<CachedEntity>,
    /// Timestamp of prediction.
    pub timestamp: String,
    /// Inference time in milliseconds.
    pub inference_ms: u64,
}

/// Minimal entity representation for caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEntity {
    /// Surface text of the entity.
    pub text: String,
    /// Entity type label (e.g., "PER", "ORG").
    pub entity_type: String,
    /// Start character offset.
    pub start: usize,
    /// End character offset (exclusive).
    pub end: usize,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
}

impl From<&Entity> for CachedEntity {
    fn from(e: &Entity) -> Self {
        Self {
            text: e.text.clone(),
            entity_type: e.entity_type.to_string(),
            start: e.start,
            end: e.end,
            confidence: e.confidence as f32,
        }
    }
}

impl From<CachedEntity> for Entity {
    fn from(c: CachedEntity) -> Self {
        use anno::EntityType;
        Entity::new(
            c.text,
            EntityType::from_label(&c.entity_type),
            c.start,
            c.end,
            c.confidence as f64,
        )
    }
}

/// Prediction cache for incremental evaluation.
/// Thread-safe via interior mutability.
pub struct PredictionCache {
    cache_dir: PathBuf,
    /// In-memory index: key -> cached prediction (thread-safe).
    index: std::sync::RwLock<HashMap<String, CachedPrediction>>,
    /// File write lock for appending to JSONL.
    file_lock: std::sync::Mutex<()>,
}

impl PredictionCache {
    /// Create a new prediction cache in the given directory.
    pub fn new(cache_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        fs::create_dir_all(&cache_dir)?;

        let cache = Self {
            cache_dir,
            index: std::sync::RwLock::new(HashMap::new()),
            file_lock: std::sync::Mutex::new(()),
        };
        cache.load_index()?;
        Ok(cache)
    }

    /// Load or create a prediction cache from a file path.
    /// If the path is a file, uses its parent directory.
    /// Returns empty cache on error (non-fatal).
    pub fn load_or_create(path: &Path) -> Self {
        let cache_dir = if path.is_file() || path.extension().is_some() {
            path.parent().unwrap_or(Path::new(".")).to_path_buf()
        } else {
            path.to_path_buf()
        };

        match Self::new(&cache_dir) {
            Ok(cache) => {
                let count = cache.index.read().map(|idx| idx.len()).unwrap_or(0);
                if count > 0 {
                    eprintln!(
                        "[cache] Loaded {} cached predictions from {}",
                        count,
                        cache_dir.display()
                    );
                }
                cache
            }
            Err(e) => {
                eprintln!("[cache] Warning: Could not load prediction cache: {}", e);
                Self {
                    cache_dir,
                    index: std::sync::RwLock::new(HashMap::new()),
                    file_lock: std::sync::Mutex::new(()),
                }
            }
        }
    }

    /// Check if cache is enabled (has entries or a valid path).
    pub fn is_enabled(&self) -> bool {
        self.cache_dir.exists()
            || self
                .index
                .read()
                .map(|idx| !idx.is_empty())
                .unwrap_or(false)
    }

    /// Get the default path for the prediction cache.
    pub fn default_path() -> PathBuf {
        std::env::var("ANNO_PREDICTION_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("anno")
                    .join("predictions.jsonl")
            })
    }

    /// Generate cache key from components (using raw text).
    pub fn cache_key(text: &str, backend: &str, version: &str, labels: &[&str]) -> String {
        let text_hash = Self::hash_str(text);
        Self::cache_key_from_hash(&text_hash, backend, version, labels)
    }

    /// Generate cache key from pre-computed text hash.
    /// This allows loading from disk where we only have the hash, not the text.
    fn cache_key_from_hash(
        text_hash: &str,
        backend: &str,
        version: &str,
        labels: &[&str],
    ) -> String {
        // Back-compat: older cache files stored a 64-bit `DefaultHasher` text hash.
        // Detect by length and regenerate the legacy key so existing caches still hit.
        if text_hash.len() == 16 {
            return Self::cache_key_from_hash_legacy(text_hash, backend, version, labels);
        }
        Self::cache_key_from_hash_sha256(text_hash, backend, version, labels)
    }

    /// Hash a string to a hex string.
    fn hash_str(s: &str) -> String {
        // Stable, content-addressed hash (matches module doc).
        Self::sha256_hex(s.as_bytes())
    }

    fn cache_key_from_hash_sha256(
        text_hash: &str,
        backend: &str,
        version: &str,
        labels: &[&str],
    ) -> String {
        // Make concatenation unambiguous by inserting a delimiter byte.
        const SEP: u8 = 0x1f;

        // Sort labels for consistent hashing.
        let mut sorted_labels: Vec<&str> = labels.to_vec();
        sorted_labels.sort();

        let mut bytes = Vec::with_capacity(
            text_hash.len()
                + backend.len()
                + version.len()
                + sorted_labels.iter().map(|s| s.len()).sum::<usize>()
                + 8,
        );
        bytes.extend_from_slice(text_hash.as_bytes());
        bytes.push(SEP);
        bytes.extend_from_slice(backend.to_lowercase().as_bytes());
        bytes.push(SEP);
        bytes.extend_from_slice(version.as_bytes());
        bytes.push(SEP);
        for (i, l) in sorted_labels.iter().enumerate() {
            if i > 0 {
                bytes.push(b',');
            }
            bytes.extend_from_slice(l.as_bytes());
        }

        Self::sha256_hex(&bytes)
    }

    fn cache_key_from_hash_legacy(
        text_hash: &str,
        backend: &str,
        version: &str,
        labels: &[&str],
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text_hash.hash(&mut hasher);
        backend.to_lowercase().hash(&mut hasher);
        version.hash(&mut hasher);

        let mut sorted_labels: Vec<&str> = labels.to_vec();
        sorted_labels.sort();
        sorted_labels.hash(&mut hasher);

        format!("{:016x}", hasher.finish())
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        format!("{:x}", hasher.finalize())
    }

    /// Look up cached prediction by key.
    pub fn get(&self, key: &str) -> Option<CachedPrediction> {
        self.index.read().ok()?.get(key).cloned()
    }

    /// Look up cached prediction by components.
    /// Returns entities on hit, None on miss.
    pub fn lookup(
        &self,
        text: &str,
        backend: &str,
        version: &str,
        labels: &[&str],
    ) -> Option<Vec<Entity>> {
        let key = Self::cache_key(text, backend, version, labels);
        if let Some(p) = self.get(&key) {
            return Some(p.predictions.into_iter().map(Entity::from).collect());
        }

        // Back-compat: try legacy key derivation for older caches.
        let text_hash_legacy = Self::hash_str_legacy(text);
        let key_legacy =
            Self::cache_key_from_hash_legacy(&text_hash_legacy, backend, version, labels);
        self.get(&key_legacy)
            .map(|p| p.predictions.into_iter().map(Entity::from).collect())
    }

    /// Store prediction and return cache key.
    /// Thread-safe via interior mutability.
    pub fn store(
        &self,
        text: &str,
        backend: &str,
        version: &str,
        labels: &[&str],
        entities: &[Entity],
        inference_ms: u64,
    ) -> std::io::Result<String> {
        // text_hash is just the hash of the text content
        let text_hash = Self::hash_str(text);
        // Full cache key includes all components
        let key = Self::cache_key_from_hash(&text_hash, backend, version, labels);

        // Skip if already cached (read lock)
        if let Ok(idx) = self.index.read() {
            if idx.contains_key(&key) {
                return Ok(key);
            }
        }

        let prediction = CachedPrediction {
            text_hash, // Just text hash, not full key
            backend: backend.to_string(),
            version: version.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            predictions: entities.iter().map(CachedEntity::from).collect(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            inference_ms,
        };

        self.put(key.clone(), prediction)?;
        Ok(key)
    }

    /// Store prediction in cache (thread-safe).
    fn put(&self, key: String, prediction: CachedPrediction) -> std::io::Result<()> {
        // Acquire file lock for writing
        let _lock = self
            .file_lock
            .lock()
            .map_err(|_| std::io::Error::other("file lock poisoned"))?;

        // Append to JSONL file
        let cache_file = self.cache_dir.join("predictions.jsonl");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&cache_file)?;

        let line = serde_json::to_string(&prediction)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writeln!(file, "{}", line)?;

        // Update in-memory index (write lock)
        if let Ok(mut idx) = self.index.write() {
            idx.insert(key, prediction);
        }
        Ok(())
    }

    /// Load index from cache file.
    fn load_index(&self) -> std::io::Result<()> {
        let cache_file = self.cache_dir.join("predictions.jsonl");
        if !cache_file.exists() {
            return Ok(());
        }

        let file = File::open(&cache_file)?;
        let reader = BufReader::new(file);

        let mut idx = self
            .index
            .write()
            .map_err(|_| std::io::Error::other("index lock poisoned"))?;

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(pred) = serde_json::from_str::<CachedPrediction>(&line) {
                // Regenerate full cache key from stored components
                let labels_refs: Vec<&str> = pred.labels.iter().map(|s| s.as_str()).collect();
                let key = Self::cache_key_from_hash(
                    &pred.text_hash,
                    &pred.backend,
                    &pred.version,
                    &labels_refs,
                );
                idx.insert(key, pred);
            }
        }

        Ok(())
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let idx = match self.index.read() {
            Ok(idx) => idx,
            Err(_) => {
                return CacheStats {
                    total_entries: 0,
                    by_backend: HashMap::new(),
                }
            }
        };

        let by_backend: HashMap<String, usize> = idx.values().fold(HashMap::new(), |mut acc, p| {
            *acc.entry(p.backend.clone()).or_insert(0) += 1;
            acc
        });

        CacheStats {
            total_entries: idx.len(),
            by_backend,
        }
    }

    /// Clear all cached predictions.
    pub fn clear(&self) -> std::io::Result<()> {
        let _lock = self
            .file_lock
            .lock()
            .map_err(|_| std::io::Error::other("file lock poisoned"))?;

        let cache_file = self.cache_dir.join("predictions.jsonl");
        if cache_file.exists() {
            fs::remove_file(&cache_file)?;
        }

        if let Ok(mut idx) = self.index.write() {
            idx.clear();
        }
        Ok(())
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.index.read().map(|idx| idx.len()).unwrap_or(0)
    }

    /// Returns true if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn hash_str_legacy(s: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

/// Cache statistics.
#[derive(Debug)]
pub struct CacheStats {
    /// Total number of cached predictions.
    pub total_entries: usize,
    /// Count of predictions per backend name.
    pub by_backend: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_consistency() {
        let key1 = PredictionCache::cache_key("hello world", "nuner", "1.0", &["PER", "ORG"]);
        let key2 = PredictionCache::cache_key("hello world", "nuner", "1.0", &["ORG", "PER"]);
        // Labels are sorted, so order shouldn't matter
        assert_eq!(key1, key2);

        // Different text should produce different key
        let key3 = PredictionCache::cache_key("goodbye world", "nuner", "1.0", &["PER", "ORG"]);
        assert_ne!(key1, key3);
    }
}
