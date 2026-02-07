//! Incremental Coreference Resolution for Book-Scale Documents
//!
//! This module implements an incremental coreference resolver inspired by
//! Longdoc (Toshniwal et al., 2020, 2021), designed for processing documents
//! that exceed the context window of standard transformer models.
//!
//! # Motivation: The Book-Scale Challenge
//!
//! Standard coreference systems struggle at book scale (200k+ tokens) because:
//!
//! 1. **Memory limitations**: naive “full-context” encoding does not scale.
//! 2. **Long-range dependencies**: coreferent mentions can be very far apart.
//! 3. **Metric divergence**: aggregate metrics can hide long-document failure modes.
//!
//! # The Incremental Approach
//!
//! Instead of processing the entire document at once, incremental coref:
//!
//! 1. Processes text in overlapping **windows** (typically 1500-4000 tokens)
//! 2. Maintains a **memory** of active entity clusters
//! 3. Links new mentions to existing clusters or creates new ones
//! 4. Optionally "forgets" rarely-used clusters to bound memory
//!
//! ```text
//! Document: [====Window 1====][====Window 2====][====Window 3====]...
//!                   |                 |                 |
//!                   v                 v                 v
//!           [Mention Detection] [Link to Memory] [Update Memory]
//!                   |                 |                 |
//!                   +--------+--------+--------+--------+
//!                            |
//!                    [Entity Memory]
//!                     Cluster 1: [John, he, Smith, ...]
//!                     Cluster 2: [Mary, she, ...]
//!                     ...
//! ```
//!
//! # Memory Policies
//!
//! Two policies from Longdoc/Dual-cache:
//!
//! - **Unbounded**: Keep all clusters forever (accurate but memory-intensive)
//! - **LRU (Least Recently Used)**: Evict least-recently-accessed clusters
//! - **LFU (Least Frequently Used)**: Evict least-frequently-accessed clusters
//! - **Dual-cache**: Combines LRU (local) + LFU (global) caches
//!
//! # Memory Paradigms: Heuristic vs Neural
//!
//! This implementation uses **heuristic** entity memory. There are three main
//! paradigms in the literature:
//!
//! | Paradigm | Memory Update | Training | Systems |
//! |----------|---------------|----------|---------|
//! | **Heuristic** (this) | String match + discrete ops | None | Anno EntityMemory |
//! | **Referential Reader** | GRU gates | End-to-end | Liu et al. 2019 |
//! | **SpanEIT** | GRU per coref cluster | Supervised | Hossain et al. 2025 |
//!
//! **Why heuristic?**
//! - No training data required
//! - Fast inference (no neural forward pass)
//! - Interpretable decisions
//! - Useful as a baseline and as plumbing for eval harnesses
//!
//! **When to consider neural:**
//! - Need to handle ambiguous mentions ("the president" → multiple candidates)
//! - Working with languages where string match is unreliable
//! - Building a trained coreference system
//!
//! References:
//! - Liu, Zettlemoyer & Eisenstein (2019): "The Referential Reader" - ACL 2019
//! - Hossain et al. (2025): "SpanEIT" - arXiv:2509.11604
//!
//! # Example
//!
//! ```rust
//! use anno::eval::incremental_coref::{IncrementalCorefResolver, IncrementalConfig, MemoryPolicy};
//! use anno::Entity;
//!
//! let config = IncrementalConfig {
//!     window_size: 1500,
//!     window_overlap: 200,
//!     memory_policy: MemoryPolicy::Unbounded,
//!     similarity_threshold: 0.7,
//!     ..Default::default()
//! };
//!
//! let resolver = IncrementalCorefResolver::new(config);
//!
//! // Process a book incrementally
//! let book_text = "John went to the store. He bought milk..."; // 200k+ tokens
//! let clusters = resolver.resolve_document(book_text);
//! ```
//!
//! # References
//!
//! - Toshniwal et al. (2020): "Learning to Ignore: Long Document Coreference
//!   with Bounded Memory Neural Networks"
//! - Toshniwal et al. (2021): "On Generalization in Coreference Resolution"
//! - Guo et al. (2023): "Dual Cache for Long Document Neural Coreference Resolution"
//! - Martinelli et al. (2025): "BOOKCOREF: Coreference Resolution at Book Scale"

use super::coref::{CorefChain, Mention, MentionType};
use anno::{Entity, EntityType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// =============================================================================
// Configuration
// =============================================================================

/// Memory eviction policy for incremental coreference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryPolicy {
    /// Keep all clusters (unbounded memory, most accurate)
    #[default]
    Unbounded,
    /// Least Recently Used - evict clusters not accessed recently
    LeastRecentlyUsed {
        /// Maximum clusters to keep
        max_clusters: usize,
    },
    /// Least Frequently Used - evict clusters with fewest accesses
    LeastFrequentlyUsed {
        /// Maximum clusters to keep
        max_clusters: usize,
    },
    /// Dual-cache: L-cache (LRU) + G-cache (LFU)
    ///
    /// From Guo et al. (2023), combines local recency with global frequency.
    DualCache {
        /// L-cache (local, LRU) size
        l_cache_size: usize,
        /// G-cache (global, LFU) size
        g_cache_size: usize,
    },
}

/// Configuration for incremental coreference resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Window size in tokens (characters if token_based=false)
    pub window_size: usize,
    /// Overlap between consecutive windows (helps maintain continuity)
    pub window_overlap: usize,
    /// Memory eviction policy
    pub memory_policy: MemoryPolicy,
    /// Similarity threshold for linking mentions to existing clusters
    pub similarity_threshold: f64,
    /// Use token-based windowing (vs character-based)
    pub token_based: bool,
    /// Maximum distance (in windows) for pronoun antecedent search
    pub max_pronoun_search_windows: usize,
    /// Whether to use exact string matching as a strong signal
    pub use_exact_match: bool,
    /// Whether to use substring matching for names
    pub use_substring_match: bool,
    /// Group consecutive windows for second-pass expansion
    /// (from BOOKCOREF pipeline, typically G=10)
    pub grouped_window_size: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            window_size: 1500,
            window_overlap: 200,
            memory_policy: MemoryPolicy::Unbounded,
            similarity_threshold: 0.7,
            token_based: true,
            max_pronoun_search_windows: 3,
            use_exact_match: true,
            use_substring_match: true,
            grouped_window_size: 10,
        }
    }
}

// =============================================================================
// Entity Memory (Cluster Storage)
// =============================================================================

/// Metadata for a cluster in memory.
#[derive(Debug, Clone)]
struct ClusterMetadata {
    /// Unique cluster ID
    id: u64,
    /// Representative mention text (typically the first or longest)
    representative: String,
    /// All mention texts in this cluster
    mentions: Vec<MentionRecord>,
    /// Last window index where this cluster was accessed
    last_accessed_window: usize,
    /// Total access count (for LFU policy)
    access_count: usize,
    /// Entity type (if known)
    entity_type: Option<EntityType>,
    /// Character offset of first mention (for ordering)
    #[allow(dead_code)]
    first_mention_offset: usize,
}

/// Record of a mention within a cluster.
#[derive(Debug, Clone)]
struct MentionRecord {
    text: String,
    start: usize,
    end: usize,
    #[allow(dead_code)]
    window_index: usize,
    mention_type: MentionType,
}

/// Entity memory for incremental coreference.
///
/// Stores active clusters and handles eviction based on memory policy.
#[derive(Debug)]
pub struct EntityMemory {
    /// All clusters by ID
    clusters: HashMap<u64, ClusterMetadata>,
    /// Next cluster ID to assign
    next_cluster_id: u64,
    /// Memory policy
    policy: MemoryPolicy,
    /// Current window index
    current_window: usize,
    /// L-cache for dual-cache policy (cluster IDs in LRU order)
    l_cache: VecDeque<u64>,
    /// G-cache for dual-cache policy (cluster IDs in LFU order)
    g_cache: Vec<u64>,
}

impl EntityMemory {
    /// Create a new entity memory with the given policy.
    pub fn new(policy: MemoryPolicy) -> Self {
        Self {
            clusters: HashMap::new(),
            next_cluster_id: 0,
            policy,
            current_window: 0,
            l_cache: VecDeque::new(),
            g_cache: Vec::new(),
        }
    }

    /// Create a new cluster with an initial mention.
    fn create_cluster(&mut self, mention: &MentionRecord, entity_type: Option<EntityType>) -> u64 {
        let id = self.next_cluster_id;
        self.next_cluster_id += 1;

        let cluster = ClusterMetadata {
            id,
            representative: mention.text.clone(),
            mentions: vec![mention.clone()],
            last_accessed_window: self.current_window,
            access_count: 1,
            entity_type,
            first_mention_offset: mention.start,
        };

        self.clusters.insert(id, cluster);
        self.update_cache_on_access(id);
        self.maybe_evict();

        id
    }

    /// Add a mention to an existing cluster.
    fn add_to_cluster(&mut self, cluster_id: u64, mention: &MentionRecord) {
        if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
            cluster.mentions.push(mention.clone());
            cluster.last_accessed_window = self.current_window;
            cluster.access_count += 1;

            // Update representative if this mention is longer (better name)
            if mention.text.len() > cluster.representative.len()
                && mention.mention_type != MentionType::Pronominal
            {
                cluster.representative = mention.text.clone();
            }
        }
        self.update_cache_on_access(cluster_id);
    }

    /// Find the best matching cluster for a mention.
    pub fn find_best_match(
        &self,
        mention_text: &str,
        mention_type: MentionType,
        similarity_threshold: f64,
        use_exact_match: bool,
        use_substring_match: bool,
    ) -> Option<u64> {
        let mention_key = name_key(mention_text);
        let mut best_match: Option<(u64, f64)> = None;

        for (id, cluster) in &self.clusters {
            // Skip if types are incompatible
            if !self.types_compatible(mention_type, cluster) {
                continue;
            }

            let score = self.compute_match_score(
                &mention_key,
                mention_type,
                cluster,
                use_exact_match,
                use_substring_match,
            );

            if score >= similarity_threshold {
                let should_update = match best_match {
                    None => true,
                    Some((_, best_score)) => score > best_score,
                };
                if should_update {
                    best_match = Some((*id, score));
                }
            }
        }

        best_match.map(|(id, _)| id)
    }

    /// Compute match score between mention and cluster.
    fn compute_match_score(
        &self,
        mention_key: &str,
        mention_type: MentionType,
        cluster: &ClusterMetadata,
        use_exact_match: bool,
        use_substring_match: bool,
    ) -> f64 {
        let rep_key = name_key(&cluster.representative);

        // Exact match is highest confidence
        if use_exact_match && mention_key == rep_key {
            return 1.0;
        }

        // Check against all mentions in cluster
        for m in &cluster.mentions {
            let m_key = name_key(&m.text);
            if mention_key == m_key {
                return 0.95;
            }
        }

        // Substring matching for proper names
        if use_substring_match && mention_type != MentionType::Pronominal {
            // "Smith" matches "John Smith"
            if rep_key.contains(mention_key) || mention_key.contains(&rep_key) {
                return 0.85;
            }

            // Check last name matching
            let mention_parts: Vec<&str> = mention_key.split_whitespace().collect();
            let rep_parts: Vec<&str> = rep_key.split_whitespace().collect();

            if !mention_parts.is_empty() && !rep_parts.is_empty() {
                // Last name match
                if mention_parts.last() == rep_parts.last() {
                    return 0.8;
                }
                // First name match
                if mention_parts.first() == rep_parts.first() {
                    return 0.75;
                }
            }
        }

        // Pronouns match based on gender compatibility (handled elsewhere)
        if mention_type == MentionType::Pronominal {
            return 0.6; // Base score for pronouns - caller should apply distance weighting
        }

        // Trigram similarity for fuzzy matching
        let sim = trigram_similarity(mention_key, &rep_key);
        sim * 0.7 // Scale down trigram similarity
    }

    /// Check if mention type is compatible with cluster.
    fn types_compatible(&self, mention_type: MentionType, cluster: &ClusterMetadata) -> bool {
        match cluster.entity_type {
            Some(EntityType::Person) => {
                // All mention types can refer to persons
                true
            }
            Some(EntityType::Organization) | Some(EntityType::Location) => {
                // Organizations/locations shouldn't have pronoun mentions (usually)
                mention_type != MentionType::Pronominal
            }
            _ => true,
        }
    }

    /// Update cache on cluster access.
    fn update_cache_on_access(&mut self, cluster_id: u64) {
        match self.policy {
            MemoryPolicy::DualCache { l_cache_size, .. } => {
                // Move to front of L-cache (most recently used)
                self.l_cache.retain(|&id| id != cluster_id);
                self.l_cache.push_front(cluster_id);

                // Update position in G-cache based on frequency
                if let Some(cluster) = self.clusters.get(&cluster_id) {
                    let access_count = cluster.access_count;
                    self.g_cache.retain(|&id| id != cluster_id);

                    // Insert in sorted position (by access count, descending)
                    let pos = self.g_cache.iter().position(|&id| {
                        self.clusters
                            .get(&id)
                            .map(|c| c.access_count < access_count)
                            .unwrap_or(true)
                    });
                    match pos {
                        Some(p) => self.g_cache.insert(p, cluster_id),
                        None => self.g_cache.push(cluster_id),
                    }
                }

                // Trim L-cache
                while self.l_cache.len() > l_cache_size {
                    self.l_cache.pop_back();
                }
            }
            MemoryPolicy::LeastRecentlyUsed { .. } => {
                // Just track in clusters via last_accessed_window
            }
            MemoryPolicy::LeastFrequentlyUsed { .. } => {
                // Just track in clusters via access_count
            }
            MemoryPolicy::Unbounded => {}
        }
    }

    /// Evict clusters if over capacity.
    fn maybe_evict(&mut self) {
        match self.policy {
            MemoryPolicy::LeastRecentlyUsed { max_clusters } => {
                while self.clusters.len() > max_clusters {
                    // Find LRU cluster
                    let lru_id = self
                        .clusters
                        .iter()
                        .min_by_key(|(_, c)| c.last_accessed_window)
                        .map(|(&id, _)| id);

                    if let Some(id) = lru_id {
                        self.clusters.remove(&id);
                    } else {
                        break;
                    }
                }
            }
            MemoryPolicy::LeastFrequentlyUsed { max_clusters } => {
                while self.clusters.len() > max_clusters {
                    // Find LFU cluster
                    let lfu_id = self
                        .clusters
                        .iter()
                        .min_by_key(|(_, c)| c.access_count)
                        .map(|(&id, _)| id);

                    if let Some(id) = lfu_id {
                        self.clusters.remove(&id);
                    } else {
                        break;
                    }
                }
            }
            MemoryPolicy::DualCache { g_cache_size, .. } => {
                // Evict from G-cache when over capacity
                while self.g_cache.len() > g_cache_size {
                    if let Some(id) = self.g_cache.pop() {
                        // Only remove from clusters if not in L-cache
                        if !self.l_cache.contains(&id) {
                            self.clusters.remove(&id);
                        }
                    }
                }
            }
            MemoryPolicy::Unbounded => {}
        }
    }

    /// Advance to next window.
    pub fn advance_window(&mut self) {
        self.current_window += 1;
    }

    /// Get all clusters as coreference chains.
    pub fn to_chains(&self) -> Vec<CorefChain> {
        self.clusters
            .values()
            .map(|cluster| {
                let mentions: Vec<Mention> = cluster
                    .mentions
                    .iter()
                    .map(|m| {
                        let mut mention = Mention::new(&m.text, m.start, m.end);
                        mention.mention_type = Some(m.mention_type);
                        mention
                    })
                    .collect();

                let mut chain = CorefChain::new(mentions);
                chain.cluster_id = Some(cluster.id.into());
                chain.entity_type = cluster.entity_type.as_ref().map(|t| format!("{:?}", t));
                chain
            })
            .collect()
    }

    /// Get number of active clusters.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    /// Get total mention count across all clusters.
    pub fn mention_count(&self) -> usize {
        self.clusters.values().map(|c| c.mentions.len()).sum()
    }
}

// =============================================================================
// Incremental Resolver
// =============================================================================

/// Incremental coreference resolver for book-scale documents.
///
/// Processes documents in windows, maintaining entity memory across windows.
#[derive(Debug)]
pub struct IncrementalCorefResolver {
    config: IncrementalConfig,
}

impl Default for IncrementalCorefResolver {
    fn default() -> Self {
        Self::new(IncrementalConfig::default())
    }
}

impl IncrementalCorefResolver {
    /// Create a new incremental resolver with configuration.
    pub fn new(config: IncrementalConfig) -> Self {
        Self { config }
    }

    /// Resolve coreference for an entire document.
    ///
    /// Returns coreference chains linking all coreferent mentions.
    pub fn resolve_document(&self, text: &str) -> Vec<CorefChain> {
        let mut memory = EntityMemory::new(self.config.memory_policy);
        let windows = self.split_into_windows(text);

        for (window_idx, (window_text, window_offset)) in windows.iter().enumerate() {
            // Extract mentions from this window
            let mentions = self.extract_mentions(window_text, *window_offset);

            // Link each mention to existing cluster or create new
            for mention in mentions {
                self.process_mention(&mut memory, &mention);
            }

            memory.advance_window();

            // Optionally: grouped window expansion (BOOKCOREF pipeline)
            if self.config.grouped_window_size > 0
                && (window_idx + 1) % self.config.grouped_window_size == 0
            {
                self.expand_grouped_window(&mut memory, window_idx);
            }
        }

        memory.to_chains()
    }

    /// Process a stream of entities (from external NER).
    ///
    /// Useful when NER is done separately and you just need coref linking.
    pub fn resolve_entities(&self, entities: &[Entity]) -> Vec<Entity> {
        let mut memory = EntityMemory::new(self.config.memory_policy);
        let mut resolved = entities.to_vec();

        // Group entities by approximate window
        let mut current_window_start = 0usize;
        let mut window_idx = 0;

        for (i, entity) in entities.iter().enumerate() {
            // Check if we've moved to a new window
            if entity.start >= current_window_start + self.config.window_size {
                window_idx += 1;
                current_window_start = entity.start.saturating_sub(self.config.window_overlap);
                memory.advance_window();
            }

            let mention = MentionRecord {
                text: entity.text.clone(),
                start: entity.start,
                end: entity.end,
                window_index: window_idx,
                mention_type: self.classify_mention_type(&entity.text),
            };

            let cluster_id = self.process_mention(&mut memory, &mention);
            resolved[i].canonical_id = Some(cluster_id.into());
        }

        resolved
    }

    /// Split text into overlapping windows.
    fn split_into_windows(&self, text: &str) -> Vec<(String, usize)> {
        let mut windows = Vec::new();
        let mut offset = 0;

        if self.config.token_based {
            // Token-based windowing, but offsets are still character offsets.
            // We tokenize in a Unicode-safe way and keep (start_char, end_char) per token.
            #[derive(Debug, Clone, Copy)]
            struct TokenSpan {
                start_char: usize,
                end_char: usize,
            }

            fn tokenize_with_char_offsets(text: &str) -> Vec<TokenSpan> {
                let mut tokens = Vec::new();

                let mut in_word = false;
                let mut word_start_char = 0;
                let mut char_pos = 0;

                for c in text.chars() {
                    if c.is_whitespace() {
                        if in_word {
                            tokens.push(TokenSpan {
                                start_char: word_start_char,
                                end_char: char_pos,
                            });
                            in_word = false;
                        }
                    } else if !in_word {
                        in_word = true;
                        word_start_char = char_pos;
                    }
                    char_pos += 1;
                }

                if in_word {
                    tokens.push(TokenSpan {
                        start_char: word_start_char,
                        end_char: char_pos,
                    });
                }

                tokens
            }

            let tokens = tokenize_with_char_offsets(text);
            let step = self
                .config
                .window_size
                .saturating_sub(self.config.window_overlap);

            while offset < tokens.len() {
                let end = (offset + self.config.window_size).min(tokens.len());
                if end == 0 || offset >= end {
                    break;
                }

                let char_start = tokens[offset].start_char;
                let char_end = tokens[end - 1].end_char;
                let window_text = anno::offset::TextSpan::from_chars(text, char_start, char_end)
                    .extract(text)
                    .to_string();

                windows.push((window_text, char_start));

                if end >= tokens.len() {
                    break;
                }
                offset += step.max(1);
            }
        } else {
            // Character-based windowing
            let text_char_len = text.chars().count();
            let step = self
                .config
                .window_size
                .saturating_sub(self.config.window_overlap);

            while offset < text_char_len {
                let end = (offset + self.config.window_size).min(text_char_len);

                // Adjust to word boundary if possible.
                // NOTE: `offset`/`end` are character offsets; we convert to byte offsets for rfind.
                let adjusted_end = if end < text_char_len {
                    let end_byte = anno::offset::TextSpan::from_chars(text, end, end).byte_start;
                    let mut adjusted_end_byte = end_byte;

                    if let Some(ws_byte_pos) = text[..end_byte].rfind(char::is_whitespace) {
                        // Move past the whitespace character (not just +1 byte).
                        let ws_len = text[ws_byte_pos..]
                            .chars()
                            .next()
                            .map(|c| c.len_utf8())
                            .unwrap_or(1);
                        adjusted_end_byte = ws_byte_pos + ws_len;

                        // Skip consecutive whitespace
                        while adjusted_end_byte < end_byte {
                            match text[adjusted_end_byte..].chars().next() {
                                Some(c) if c.is_whitespace() => {
                                    adjusted_end_byte += c.len_utf8();
                                }
                                _ => break,
                            }
                        }
                    }

                    let (adjusted_end_char, _) =
                        anno::offset::bytes_to_chars(text, adjusted_end_byte, adjusted_end_byte);
                    if adjusted_end_char > offset {
                        adjusted_end_char.min(text_char_len)
                    } else {
                        end
                    }
                } else {
                    end
                };

                let window_text = anno::offset::TextSpan::from_chars(text, offset, adjusted_end)
                    .extract(text)
                    .to_string();
                windows.push((window_text, offset));

                if adjusted_end >= text_char_len {
                    break;
                }
                offset += step.max(1);
            }
        }

        windows
    }

    /// Extract mentions from a window of text.
    ///
    /// This is a simplified mention detector. For production use,
    /// integrate with a proper NER/mention detection model.
    fn extract_mentions(&self, text: &str, offset: usize) -> Vec<MentionRecord> {
        let mut mentions = Vec::new();

        // Unicode-safe tokenization that preserves character offsets.
        let mut in_word = false;
        let mut word_start_byte = 0;
        let mut word_start_char = 0;
        let mut char_pos = 0;

        let mut maybe_push_mention = |word: &str, local_start: usize, local_end: usize| {
            let mention_type = self.classify_mention_type(word);

            // Only track pronouns and capitalized words (potential names)
            let should_track = match mention_type {
                MentionType::Pronominal => true,
                MentionType::Proper => true,
                _ => word
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false),
            };

            if should_track {
                mentions.push(MentionRecord {
                    text: word.to_string(),
                    start: offset + local_start,
                    end: offset + local_end,
                    window_index: 0, // Will be set by caller
                    mention_type,
                });
            }
        };

        for (byte_idx, c) in text.char_indices() {
            if c.is_whitespace() {
                if in_word {
                    let word = &text[word_start_byte..byte_idx];
                    maybe_push_mention(word, word_start_char, char_pos);
                    in_word = false;
                }
            } else if !in_word {
                in_word = true;
                word_start_byte = byte_idx;
                word_start_char = char_pos;
            }
            char_pos += 1;
        }

        if in_word {
            let word = &text[word_start_byte..];
            maybe_push_mention(word, word_start_char, char_pos);
        }

        mentions
    }

    /// Classify mention type from text.
    fn classify_mention_type(&self, text: &str) -> MentionType {
        let lower = text.to_lowercase();

        // Pronouns
        let pronouns = [
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "they",
            "them",
            "their",
            "theirs",
            "themself",
            "themselves",
            "it",
            "its",
            "itself",
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            // Neopronouns
            "xe",
            "xem",
            "xyr",
            "xyrs",
            "ze",
            "zir",
            "zirs",
            "ey",
            "em",
            "eir",
            "eirs",
        ];

        if pronouns.contains(&lower.as_str()) {
            return MentionType::Pronominal;
        }

        // Proper nouns (capitalized, not sentence-initial)
        if text
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
        {
            return MentionType::Proper;
        }

        // Definite descriptions ("the man", "the company")
        if lower.starts_with("the ") {
            return MentionType::Nominal;
        }

        MentionType::Unknown
    }

    /// Process a single mention, linking to existing cluster or creating new.
    fn process_mention(&self, memory: &mut EntityMemory, mention: &MentionRecord) -> u64 {
        // Try to find matching cluster
        if let Some(cluster_id) = memory.find_best_match(
            &mention.text,
            mention.mention_type,
            self.config.similarity_threshold,
            self.config.use_exact_match,
            self.config.use_substring_match,
        ) {
            memory.add_to_cluster(cluster_id, mention);
            cluster_id
        } else {
            // Create new cluster
            let entity_type = if mention.mention_type == MentionType::Proper {
                Some(EntityType::Person) // Default assumption for proper nouns
            } else {
                None
            };
            memory.create_cluster(mention, entity_type)
        }
    }

    /// Expand clusters within a grouped window (BOOKCOREF pipeline step).
    ///
    /// This is where mentions that couldn't be linked in individual windows
    /// get a second chance with broader context.
    fn expand_grouped_window(&self, memory: &mut EntityMemory, _window_idx: usize) {
        // In a full implementation, this would:
        // 1. Run a second-pass coref model on the grouped window
        // 2. Use the broader context to resolve previously unlinked mentions
        // 3. Merge singleton clusters that should be linked

        // For now, we do simple cluster merging based on string similarity
        let cluster_ids: Vec<u64> = memory.clusters.keys().copied().collect();

        for i in 0..cluster_ids.len() {
            for j in (i + 1)..cluster_ids.len() {
                let id_i = cluster_ids[i];
                let id_j = cluster_ids[j];

                // Check if clusters should be merged
                if self.should_merge_clusters(memory, id_i, id_j) {
                    // Merge j into i
                    if let Some(cluster_j) = memory.clusters.remove(&id_j) {
                        if let Some(cluster_i) = memory.clusters.get_mut(&id_i) {
                            cluster_i.mentions.extend(cluster_j.mentions);
                            cluster_i.access_count += cluster_j.access_count;
                        }
                    }
                }
            }
        }
    }

    /// Check if two clusters should be merged.
    fn should_merge_clusters(&self, memory: &EntityMemory, id_a: u64, id_b: u64) -> bool {
        let (cluster_a, cluster_b) = match (memory.clusters.get(&id_a), memory.clusters.get(&id_b))
        {
            (Some(a), Some(b)) => (a, b),
            _ => return false,
        };

        // Don't merge if different entity types
        if cluster_a.entity_type.is_some()
            && cluster_b.entity_type.is_some()
            && cluster_a.entity_type != cluster_b.entity_type
        {
            return false;
        }

        // Check name similarity
        let rep_a = name_key(&cluster_a.representative);
        let rep_b = name_key(&cluster_b.representative);

        // Exact match
        if rep_a == rep_b {
            return true;
        }

        // Substring match (e.g., "Smith" and "John Smith")
        if rep_a.contains(&rep_b) || rep_b.contains(&rep_a) {
            return true;
        }

        // High trigram similarity
        if trigram_similarity(&rep_a, &rep_b) > 0.8 {
            return true;
        }

        false
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Normalize a name / mention surface into a comparison key.
///
/// This is intentionally **comparison-only** normalization (not offset-preserving): it is safe for
/// coreference linking because it does not modify stored spans; it only affects matching decisions.
fn name_key(s: &str) -> String {
    use std::borrow::Cow;

    // Remove BOM only. We intentionally do *not* strip other zero-width characters here because
    // ZWJ/ZWNJ can be orthographically meaningful in multiple scripts.
    fn strip_bom_only(s: &str) -> Cow<'_, str> {
        if !s.chars().any(|c| c == '\u{FEFF}') {
            return Cow::Borrowed(s);
        }
        Cow::Owned(s.chars().filter(|&c| c != '\u{FEFF}').collect())
    }

    let s = strip_bom_only(s);
    let cfg = textprep::ScrubConfig {
        normalize_newlines: true,
        remove_zero_width: false,
        remove_bidi_controls: true,
        collapse_whitespace: true,
        normalization: textprep::ScrubNormalization::Nfc,
        case: textprep::ScrubCase::Lower,
        strip_diacritics: false,
    };
    textprep::scrub_with(s.as_ref(), &cfg)
}

/// Compute trigram (character 3-gram) similarity between two strings.
///
/// Returns Jaccard similarity of trigram sets.
fn trigram_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    // Preserve previous behavior: if either side yields no trigrams, similarity is 0.0.
    if a.chars().count() < 3 || b.chars().count() < 3 {
        return 0.0;
    }
    textprep::similarity::trigram_jaccard(a, b)
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics about incremental resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IncrementalStats {
    /// Number of windows processed
    pub windows_processed: usize,
    /// Total mentions processed
    pub mentions_processed: usize,
    /// Final cluster count
    pub final_clusters: usize,
    /// Clusters evicted due to memory policy
    pub clusters_evicted: usize,
    /// Average mentions per cluster
    pub avg_mentions_per_cluster: f64,
    /// Maximum cluster size
    pub max_cluster_size: usize,
}

impl IncrementalStats {
    /// Compute statistics from entity memory.
    pub fn from_memory(memory: &EntityMemory, windows: usize) -> Self {
        let cluster_sizes: Vec<usize> =
            memory.clusters.values().map(|c| c.mentions.len()).collect();

        Self {
            windows_processed: windows,
            mentions_processed: memory.mention_count(),
            final_clusters: memory.cluster_count(),
            clusters_evicted: 0, // Would need tracking during resolution
            avg_mentions_per_cluster: if memory.cluster_count() > 0 {
                memory.mention_count() as f64 / memory.cluster_count() as f64
            } else {
                0.0
            },
            max_cluster_size: cluster_sizes.into_iter().max().unwrap_or(0),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windowing_and_mentions_use_character_offsets_on_unicode() {
        use anno::offset::TextSpan;

        let config = IncrementalConfig {
            token_based: false,
            window_size: 12,
            window_overlap: 3,
            ..Default::default()
        };
        let resolver = IncrementalCorefResolver::new(config);

        // Mixed-script + emoji prefix: byte offsets and char offsets diverge immediately.
        let text = "🎉 Dr. John went to 東京. He waved.";

        let windows = resolver.split_into_windows(text);
        assert!(!windows.is_empty());

        // Each window's offset is a char offset into the original text, and the window text must
        // equal the corresponding substring in the original text.
        for (window_text, window_offset) in &windows {
            let span = TextSpan::from_chars(
                text,
                *window_offset,
                *window_offset + window_text.chars().count(),
            );
            assert_eq!(span.extract(text), window_text);
        }

        // Mention extraction should produce character offsets that are Unicode-safe.
        let mentions = resolver.extract_mentions(text, 0);
        let john = mentions
            .iter()
            .find(|m| m.text == "John")
            .expect("expected to detect 'John'");
        let extracted = TextSpan::from_chars(text, john.start, john.end).extract(text);
        assert_eq!(extracted, "John");
        assert_eq!(
            john.start, 6,
            "🎉(1) + space(1) + Dr.(2) + .(1) + space(1) = 6"
        );
        assert_eq!(john.end, 10);
    }

    #[test]
    fn test_trigram_similarity() {
        assert!((trigram_similarity("hello", "hello") - 1.0).abs() < 0.001);
        assert!(trigram_similarity("hello", "world") < 0.3);
        assert!(trigram_similarity("john", "johnson") > 0.3);
        assert!(trigram_similarity("", "hello") == 0.0);
    }

    #[test]
    fn test_entity_memory_basic() {
        let mut memory = EntityMemory::new(MemoryPolicy::Unbounded);

        let mention1 = MentionRecord {
            text: "John Smith".to_string(),
            start: 0,
            end: 10,
            window_index: 0,
            mention_type: MentionType::Proper,
        };

        let cluster_id = memory.create_cluster(&mention1, Some(EntityType::Person));
        assert_eq!(memory.cluster_count(), 1);

        let mention2 = MentionRecord {
            text: "Smith".to_string(),
            start: 50,
            end: 55,
            window_index: 0,
            mention_type: MentionType::Proper,
        };

        memory.add_to_cluster(cluster_id, &mention2);
        assert_eq!(memory.mention_count(), 2);
    }

    #[test]
    fn test_entity_memory_lru_eviction() {
        let mut memory = EntityMemory::new(MemoryPolicy::LeastRecentlyUsed { max_clusters: 2 });

        let mention1 = MentionRecord {
            text: "John".to_string(),
            start: 0,
            end: 4,
            window_index: 0,
            mention_type: MentionType::Proper,
        };
        memory.create_cluster(&mention1, None);

        memory.advance_window();

        let mention2 = MentionRecord {
            text: "Mary".to_string(),
            start: 10,
            end: 14,
            window_index: 1,
            mention_type: MentionType::Proper,
        };
        memory.create_cluster(&mention2, None);

        memory.advance_window();

        let mention3 = MentionRecord {
            text: "Bob".to_string(),
            start: 20,
            end: 23,
            window_index: 2,
            mention_type: MentionType::Proper,
        };
        memory.create_cluster(&mention3, None);

        // Should have evicted the oldest (John)
        assert_eq!(memory.cluster_count(), 2);
    }

    #[test]
    fn test_find_best_match() {
        let mut memory = EntityMemory::new(MemoryPolicy::Unbounded);

        let mention1 = MentionRecord {
            text: "John Smith".to_string(),
            start: 0,
            end: 10,
            window_index: 0,
            mention_type: MentionType::Proper,
        };
        memory.create_cluster(&mention1, Some(EntityType::Person));

        // Should match "Smith" to "John Smith"
        let match_result = memory.find_best_match("Smith", MentionType::Proper, 0.7, true, true);
        assert!(match_result.is_some());
    }

    #[test]
    fn test_incremental_resolver_basic() {
        let config = IncrementalConfig {
            window_size: 100,
            window_overlap: 20,
            token_based: false,
            ..Default::default()
        };

        let resolver = IncrementalCorefResolver::new(config);

        let text = "John went to the store. He bought milk. John came home.";
        let chains = resolver.resolve_document(text);

        // Should have at least one chain for John/He
        assert!(!chains.is_empty());
    }

    #[test]
    fn test_resolve_entities() {
        let resolver = IncrementalCorefResolver::default();

        let entities = vec![
            Entity::new("John Smith", EntityType::Person, 0, 10, 0.9),
            Entity::new("Smith", EntityType::Person, 50, 55, 0.85),
            Entity::new("he", EntityType::Person, 100, 102, 0.7),
        ];

        let resolved = resolver.resolve_entities(&entities);

        // First two should share a canonical_id (John Smith ~ Smith)
        assert_eq!(resolved[0].canonical_id, resolved[1].canonical_id);
    }

    #[test]
    fn test_mention_type_classification() {
        let resolver = IncrementalCorefResolver::default();

        assert_eq!(
            resolver.classify_mention_type("he"),
            MentionType::Pronominal
        );
        assert_eq!(
            resolver.classify_mention_type("she"),
            MentionType::Pronominal
        );
        assert_eq!(
            resolver.classify_mention_type("they"),
            MentionType::Pronominal
        );
        assert_eq!(
            resolver.classify_mention_type("xe"),
            MentionType::Pronominal
        );
        assert_eq!(resolver.classify_mention_type("John"), MentionType::Proper);
    }

    #[test]
    fn test_dual_cache_policy() {
        let memory = EntityMemory::new(MemoryPolicy::DualCache {
            l_cache_size: 5,
            g_cache_size: 10,
        });
        assert_eq!(memory.cluster_count(), 0);
    }
}
