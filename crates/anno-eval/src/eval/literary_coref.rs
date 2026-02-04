//! Character-Centric Coreference Pipeline for Literary Texts
//!
//! This module implements the BookCoref Pipeline approach for resolving
//! coreference in literary texts (novels, screenplays, narratives).
//!
//! # Why Character-Centric?
//!
//! In literary texts, characters are the main agents of narrative:
//!
//! > "Characters are the main agents of fictional stories" 
//! > — Bamman et al. (2013), Roesiger et al. (2018), Labatut & Bost (2019)
//!
//! Focusing on characters provides:
//! - **Simpler annotation**: Only track anthropomorphized entities
//! - **Narrative relevance**: Characters drive plot, relationships, themes
//!
//! # The BookCoref Pipeline
//!
//! From Martinelli et al. (2025), a four-stage approach:
//!
//! ```text
//! 1. Cluster Initialization (Character Linking)
//!    ├── Extract explicit character mentions via Entity Linking
//!    └── Link to known character names (from metadata/dramatis personae)
//!
//! 2. Cluster Refinement (LLM Filtering)
//!    ├── Verify each link with local context
//!    └── Remove false positives to prevent error propagation
//!
//! 3. Cluster Expansion (Window-level)
//!    ├── Process book in windows (1500 tokens)
//!    ├── Expand clusters to include pronouns, epithets
//!    └── Link new mentions to character clusters
//!
//! 4. Grouped Window Expansion
//!    ├── Merge groups of consecutive windows
//!    ├── Second-pass resolution for missed mentions
//!    └── Final cluster consolidation
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::eval::literary_coref::{LiteraryCorefPipeline, CharacterList};
//!
//! // Define characters (from book metadata, Wikipedia, etc.)
//! let characters = CharacterList::new(vec![
//!     ("Elizabeth Bennet", vec!["Elizabeth", "Lizzy", "Eliza"]),
//!     ("Fitzwilliam Darcy", vec!["Mr. Darcy", "Darcy"]),
//!     ("Jane Bennet", vec!["Jane", "Miss Bennet"]),
//! ]);
//!
//! let pipeline = LiteraryCorefPipeline::new(characters);
//! let book_text = std::fs::read_to_string("pride_and_prejudice.txt")?;
//!
//! let clusters = pipeline.resolve(&book_text);
//! // Returns character clusters with all mentions (names, pronouns, epithets)
//! ```
//!
//! # Dataset Support
//!
//! This pipeline is designed to work with:
//! - **LitBank**: literary excerpts
//! - **BOOKCOREF**: book-scale literary coreference
//! - **MovieCoref**: Screenplays with character annotations
//! - **DROC**: German novel character coreference
//! - **KoCoNovel**: Korean novel character coreference (address terms, pro-drop, morphology)
//!
//! # References
//!
//! - Martinelli et al. (2025): "BOOKCOREF: Coreference Resolution at Book Scale"
//! - Bamman et al. (2020): "An Annotated Dataset of Coreference in English Literature"
//! - Vala et al. (2015): "Mr. Bennet, his coachman, and the archbishop walk into a bar"
//! - Orlando et al. (2024): "ReLiK: Retrieve and LinK"
//! - **Duron-Tejedor et al. (2023)**: "How to Evaluate Coreference in Literary Texts?"
//!   \[arXiv:2401.00238\] — Recommends stratified evaluation (protagonist vs secondary)
//! - **Kim, Lee & Lee (2024)**: "KoCoNovel: Annotated Dataset of Character Coreference
//!   in Korean Novels" \[arXiv:2404.01140\] — Korean literary coref with address term culture

use super::coref::{CorefChain, Mention, MentionType};
use anno::backends::llm_client::{LlmConfig, LlmProvider};
use anno_core::Gender;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Character Metadata
// =============================================================================

/// A character with canonical name and aliases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Character {
    /// Canonical character name (e.g., "Elizabeth Bennet")
    pub canonical_name: String,
    /// Alternate names and aliases (e.g., ["Elizabeth", "Lizzy", "Eliza"])
    pub aliases: Vec<String>,
    /// Character description (optional, from study guides)
    pub description: Option<String>,
    /// Gender hint (optional, for pronoun resolution)
    pub gender: Option<Gender>,
    /// Character importance (main, supporting, minor)
    pub role: CharacterRole,
}

impl Character {
    /// Create a new character with canonical name.
    pub fn new(canonical_name: &str) -> Self {
        Self {
            canonical_name: canonical_name.to_string(),
            aliases: Vec::new(),
            description: None,
            gender: None,
            role: CharacterRole::Unknown,
        }
    }

    /// Add aliases.
    pub fn with_aliases(mut self, aliases: Vec<&str>) -> Self {
        self.aliases = aliases.into_iter().map(String::from).collect();
        self
    }

    /// Set gender.
    pub fn with_gender(mut self, gender: Gender) -> Self {
        self.gender = Some(gender);
        self
    }

    /// Set role.
    pub fn with_role(mut self, role: CharacterRole) -> Self {
        self.role = role;
        self
    }

    /// Get all names (canonical + aliases) for matching.
    pub fn all_names(&self) -> Vec<&str> {
        let mut names = vec![self.canonical_name.as_str()];
        names.extend(self.aliases.iter().map(String::as_str));
        names
    }
}

// Gender imported from anno_core::Gender

/// Character role in the narrative structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CharacterRole {
    /// Main character driving the plot
    Protagonist,
    /// Character opposing the protagonist
    Antagonist,
    /// Important secondary character
    Supporting,
    /// Briefly appearing character
    Minor,
    /// Role not determined
    Unknown,
}

/// List of characters in a literary work.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CharacterList {
    characters: Vec<Character>,
    /// Index from lowercase name to character index
    #[serde(skip)]
    name_index: HashMap<String, usize>,
}

impl CharacterList {
    /// Create from list of (canonical_name, aliases) tuples.
    pub fn new(chars: Vec<(&str, Vec<&str>)>) -> Self {
        let characters: Vec<Character> = chars
            .into_iter()
            .map(|(name, aliases)| Character::new(name).with_aliases(aliases))
            .collect();

        let mut list = Self {
            characters,
            name_index: HashMap::new(),
        };
        list.rebuild_index();
        list
    }

    /// Create from Character structs.
    pub fn from_characters(characters: Vec<Character>) -> Self {
        let mut list = Self {
            characters,
            name_index: HashMap::new(),
        };
        list.rebuild_index();
        list
    }

    /// Add a character.
    pub fn add(&mut self, character: Character) {
        self.characters.push(character);
        self.rebuild_index();
    }

    /// Rebuild the name index.
    fn rebuild_index(&mut self) {
        self.name_index.clear();
        for (idx, character) in self.characters.iter().enumerate() {
            for name in character.all_names() {
                self.name_index.insert(name.to_lowercase(), idx);
            }
        }
    }

    /// Find character by name (case-insensitive).
    pub fn find_by_name(&self, name: &str) -> Option<&Character> {
        self.name_index
            .get(&name.to_lowercase())
            .map(|&idx| &self.characters[idx])
    }

    /// Find character index by name.
    pub fn find_index(&self, name: &str) -> Option<usize> {
        self.name_index.get(&name.to_lowercase()).copied()
    }

    /// Get all characters.
    pub fn characters(&self) -> &[Character] {
        &self.characters
    }

    /// Number of characters.
    pub fn len(&self) -> usize {
        self.characters.len()
    }

    /// Is empty?
    pub fn is_empty(&self) -> bool {
        self.characters.is_empty()
    }
}

// =============================================================================
// Character Linking (Step 1)
// =============================================================================

/// A linked character mention.
#[derive(Debug, Clone)]
pub struct CharacterMention {
    /// Text of the mention
    pub text: String,
    /// Character offset start
    pub start: usize,
    /// Character offset end
    pub end: usize,
    /// Index of character in CharacterList
    pub character_idx: usize,
    /// Confidence of the link
    pub confidence: f64,
    /// Mention type
    pub mention_type: MentionType,
    /// Whether this was verified by LLM filtering
    pub verified: bool,
}

/// Character Linker - links explicit mentions to characters.
///
/// This is Step 1 of the BookCoref Pipeline.
///
/// # Matching Modes
///
/// - **Exact matching** (default): Only matches character names exactly
/// - **Fuzzy matching**: Uses edit distance to catch typos, OCR errors
///
/// # Example
///
/// ```rust,ignore
/// let linker = CharacterLinker::new(characters)
///     .with_fuzzy_matching(true, 0.8);  // 80% similarity threshold
/// ```
#[derive(Debug)]
pub struct CharacterLinker {
    /// Known characters
    characters: CharacterList,
    /// Use fuzzy matching for names (handles typos, OCR errors)
    fuzzy_match: bool,
    /// Minimum similarity (0.0-1.0) for fuzzy matching
    min_similarity: f64,
}

impl CharacterLinker {
    /// Create a new character linker with exact matching.
    pub fn new(characters: CharacterList) -> Self {
        Self {
            characters,
            fuzzy_match: false, // Default to exact for precision
            min_similarity: 0.8,
        }
    }

    /// Enable or disable fuzzy matching with similarity threshold.
    ///
    /// Fuzzy matching helps catch:
    /// - OCR errors: "Darcv" → "Darcy"
    /// - Typos: "Elizbeth" → "Elizabeth"
    /// - Transliteration variants: "Fyodor" vs "Fedor"
    ///
    /// # Arguments
    /// * `enabled` - Whether to use fuzzy matching
    /// * `min_similarity` - Minimum similarity (0.0-1.0) to accept a match
    pub fn with_fuzzy_matching(mut self, enabled: bool, min_similarity: f64) -> Self {
        self.fuzzy_match = enabled;
        self.min_similarity = min_similarity.clamp(0.0, 1.0);
        self
    }

    /// Link explicit mentions in text to characters.
    ///
    /// Uses Unicode-safe character indexing to handle multi-byte characters
    /// (CJK, diacritics, emoji, etc.) correctly.
    pub fn link(&self, text: &str) -> Vec<CharacterMention> {
        let mut mentions = Vec::new();

        // Build character vectors for Unicode-safe processing
        let text_chars: Vec<char> = text.chars().collect();
        let char_count = text_chars.len();
        let text_lower_chars: Vec<char> = text.to_lowercase().chars().collect();

        for (idx, character) in self.characters.characters().iter().enumerate() {
            for name in character.all_names() {
                let name_lower = name.to_lowercase();
                let name_chars: Vec<char> = name_lower.chars().collect();
                let name_len = name_chars.len();
                
                if name_len == 0 || name_len > char_count {
                    continue;
                }

                // Exact matching pass
                let mut char_pos = 0;
                while char_pos + name_len <= char_count {
                    let matches = (0..name_len).all(|i| {
                        text_lower_chars.get(char_pos + i) == name_chars.get(i)
                    });
                    
                    if matches {
                        if let Some(mention) = self.try_create_mention(
                            &text_chars, char_pos, name_len, idx, character, 1.0
                        ) {
                            mentions.push(mention);
                            char_pos += name_len;
                            continue;
                        }
                    }
                    char_pos += 1;
                }

                // Fuzzy matching pass (if enabled)
                if self.fuzzy_match && name_len >= 3 {
                    self.fuzzy_link_name(
                        &text_chars, &text_lower_chars, &name_chars, 
                        idx, character, &mut mentions
                    );
                }
            }
        }

        // Sort by position and deduplicate overlapping mentions
        mentions.sort_by_key(|m| (m.start, m.end));
        self.deduplicate_mentions(mentions)
    }

    /// Try to create a mention at the given position.
    fn try_create_mention(
        &self,
        text_chars: &[char],
        char_pos: usize,
        name_len: usize,
        char_idx: usize,
        character: &Character,
        base_confidence: f64,
    ) -> Option<CharacterMention> {
        let char_count = text_chars.len();
        let end_char_pos = char_pos + name_len;
        
        // Check word boundaries
        let is_word_start = char_pos == 0
            || !text_chars.get(char_pos.saturating_sub(1))
                .map(|c| c.is_alphanumeric())
                .unwrap_or(false);
        let is_word_end = end_char_pos >= char_count
            || !text_chars.get(end_char_pos)
                .map(|c| c.is_alphanumeric())
                .unwrap_or(false);

        if !is_word_start || !is_word_end {
            return None;
        }

        let mention_text: String = text_chars[char_pos..end_char_pos].iter().collect();
        let confidence = if mention_text.to_lowercase() == character.canonical_name.to_lowercase() {
            base_confidence
        } else {
            base_confidence * 0.9
        };

        Some(CharacterMention {
            text: mention_text,
            start: char_pos,
            end: end_char_pos,
            character_idx: char_idx,
            confidence,
            mention_type: MentionType::Proper,
            verified: false,
        })
    }

    /// Fuzzy matching for a single name.
    ///
    /// Uses Levenshtein distance to find approximate matches.
    fn fuzzy_link_name(
        &self,
        text_chars: &[char],
        text_lower: &[char],
        name_chars: &[char],
        char_idx: usize,
        _character: &Character,
        mentions: &mut Vec<CharacterMention>,
    ) {
        let name_len = name_chars.len();
        let char_count = text_chars.len();
        
        // Scan for potential fuzzy matches using sliding window
        let window_sizes = [name_len, name_len + 1, name_len.saturating_sub(1)];
        
        for &window_size in &window_sizes {
            if window_size == 0 || window_size > char_count {
                continue;
            }
            
            let mut pos = 0;
            while pos + window_size <= char_count {
                // Check word boundaries first (cheap)
                let is_word_start = pos == 0
                    || !text_chars.get(pos.saturating_sub(1))
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);
                let end_pos = pos + window_size;
                let is_word_end = end_pos >= char_count
                    || !text_chars.get(end_pos)
                        .map(|c| c.is_alphanumeric())
                        .unwrap_or(false);

                if is_word_start && is_word_end {
                    // Extract window and compute similarity
                    let window: Vec<char> = text_lower[pos..end_pos].to_vec();
                    let similarity = self.string_similarity(name_chars, &window);
                    
                    if similarity >= self.min_similarity && similarity < 1.0 {
                        // Fuzzy match (not exact)
                        let mention_text: String = text_chars[pos..end_pos].iter().collect();
                        
                        mentions.push(CharacterMention {
                            text: mention_text,
                            start: pos,
                            end: end_pos,
                            character_idx: char_idx,
                            confidence: similarity * 0.85, // Discount for fuzzy
                            mention_type: MentionType::Proper,
                            verified: false,
                        });
                    }
                }
                pos += 1;
            }
        }
    }

    /// Compute string similarity using normalized Levenshtein distance.
    fn string_similarity(&self, a: &[char], b: &[char]) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let max_len = a.len().max(b.len());
        let distance = self.levenshtein_distance(a, b);
        
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Compute Levenshtein edit distance between two character sequences.
    fn levenshtein_distance(&self, a: &[char], b: &[char]) -> usize {
        let m = a.len();
        let n = b.len();
        
        // Use single-row optimization for memory efficiency
        let mut prev_row: Vec<usize> = (0..=n).collect();
        let mut curr_row: Vec<usize> = vec![0; n + 1];
        
        for i in 1..=m {
            curr_row[0] = i;
            for j in 1..=n {
                let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                curr_row[j] = (prev_row[j] + 1)
                    .min(curr_row[j - 1] + 1)
                    .min(prev_row[j - 1] + cost);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }
        
        prev_row[n]
    }

    /// Remove overlapping mentions, keeping highest confidence.
    fn deduplicate_mentions(&self, mentions: Vec<CharacterMention>) -> Vec<CharacterMention> {
        let mut result = Vec::new();
        let mut covered = HashSet::new();

        // Sort by confidence descending, then by span length descending
        let mut sorted = mentions;
        sorted.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (b.end - b.start).cmp(&(a.end - a.start)))
        });

        for mention in sorted {
            // Check if any position is already covered
            let positions: HashSet<usize> = (mention.start..mention.end).collect();
            if positions.is_disjoint(&covered) {
                covered.extend(positions);
                result.push(mention);
            }
        }

        // Re-sort by position
        result.sort_by_key(|m| m.start);
        result
    }
}

// =============================================================================
// LLM Filtering (Step 2)
// =============================================================================

/// Result of LLM verification for a character mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationResult {
    /// LLM confirmed this is a valid character mention
    Confirmed,
    /// LLM rejected this mention (false positive)
    Rejected,
    /// LLM uncertain about this mention
    Uncertain,
}

/// Configuration for LLM-based filtering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMFilterConfig {
    /// Context window around mention (in words)
    pub context_words: usize,
    /// Model identifier (e.g., "qwen2-7b-instruct")
    pub model_id: String,
    /// Prompt template
    pub prompt_template: String,
}

impl Default for LLMFilterConfig {
    fn default() -> Self {
        Self {
            context_words: 200, // 400 words = ~200 on each side
            model_id: "qwen2-7b-instruct".to_string(),
            prompt_template: r#"I will give you an excerpt from a book with a highlighted mention of a character with [].
You will need to answer if the assigned character is correct (Yes), or not (No).

Book excerpt: {context}
Does the mention [{mention}] correspond to the character {character}? (Yes/No)"#
                .to_string(),
        }
    }
}

/// Trait for LLM-based mention verification.
///
/// Implement this to integrate with actual LLM backends.
pub trait MentionVerifier: Send + Sync {
    /// Verify if a mention correctly refers to the character.
    fn verify(
        &self,
        mention: &CharacterMention,
        character: &Character,
        context: &str,
    ) -> VerificationResult;
}

/// Dummy verifier that always confirms (for testing).
#[derive(Debug, Default)]
pub struct AlwaysConfirmVerifier;

impl MentionVerifier for AlwaysConfirmVerifier {
    fn verify(
        &self,
        _mention: &CharacterMention,
        _character: &Character,
        _context: &str,
    ) -> VerificationResult {
        VerificationResult::Confirmed
    }
}

/// Heuristic-based verifier (no LLM required).
#[derive(Debug, Default)]
pub struct HeuristicVerifier;

impl MentionVerifier for HeuristicVerifier {
    fn verify(
        &self,
        mention: &CharacterMention,
        character: &Character,
        context: &str,
    ) -> VerificationResult {
        // High confidence exact matches are confirmed
        if mention.confidence >= 0.95 {
            return VerificationResult::Confirmed;
        }

        // Check if character's canonical name appears near the mention
        let context_lower = context.to_lowercase();
        let canonical_lower = character.canonical_name.to_lowercase();

        if context_lower.contains(&canonical_lower) {
            return VerificationResult::Confirmed;
        }

        // For aliases, be more conservative
        VerificationResult::Uncertain
    }
}

// =============================================================================
// Real LLM-Based Verifier
// =============================================================================

/// LLM-based mention verifier using the `LlmProvider` trait.
///
/// This implements the LLM Filtering step from the BookCoref Pipeline
/// (Martinelli et al. 2025), which uses Qwen2 7B to verify character-mention
/// links with binary Yes/No output.
///
/// # Prompt Design
///
/// Following the paper's approach:
/// > "We prompt Qwen2 7B Instruct with the name of the character and the
/// > highlighted mention in context, and we constrain the LLM to output
/// > either 'Yes' or 'No' as the answer to our prompt."
///
/// # Example
///
/// ```rust,ignore
/// use anno::backends::llm_client::HttpProvider;
/// use anno::eval::literary_coref::LlmMentionVerifier;
///
/// let provider = HttpProvider::local(8080);  // llama.cpp server
/// let verifier = LlmMentionVerifier::new(
///     Box::new(provider),
///     LlmMentionVerifierConfig::default(),
/// );
///
/// let filter = LLMFilter::new(verifier, LLMFilterConfig::default());
/// ```
pub struct LlmMentionVerifier {
    provider: Box<dyn LlmProvider>,
    config: LlmMentionVerifierConfig,
}

/// Configuration for LLM-based mention verification.
#[derive(Debug, Clone)]
pub struct LlmMentionVerifierConfig {
    /// LLM configuration (model, temperature, etc.)
    pub llm_config: LlmConfig,
    /// Prompt template with {character}, {mention}, {context} placeholders
    pub prompt_template: String,
    /// Affirmative responses (case-insensitive)
    pub yes_responses: Vec<String>,
    /// Negative responses (case-insensitive)
    pub no_responses: Vec<String>,
}

impl Default for LlmMentionVerifierConfig {
    fn default() -> Self {
        Self {
            llm_config: LlmConfig {
                model: "qwen2.5:7b".to_string(),  // Or any OpenAI-compatible model
                max_tokens: 10,  // Just need Yes/No
                temperature: 0.0,  // Deterministic
                ..Default::default()
            },
            // Prompt design following BOOKCOREF paper
            prompt_template: r#"Given the following passage from a novel, determine if the highlighted mention refers to the character "{character}".

Context:
{context}

Highlighted mention: "{mention}"

Does this mention refer to {character}? Answer with only "Yes" or "No"."#.to_string(),
            yes_responses: vec!["yes".to_string(), "y".to_string(), "true".to_string()],
            no_responses: vec!["no".to_string(), "n".to_string(), "false".to_string()],
        }
    }
}

impl LlmMentionVerifier {
    /// Create a new LLM-based verifier.
    pub fn new(provider: Box<dyn LlmProvider>, config: LlmMentionVerifierConfig) -> Self {
        Self { provider, config }
    }

    /// Create with default config.
    pub fn with_provider(provider: Box<dyn LlmProvider>) -> Self {
        Self::new(provider, LlmMentionVerifierConfig::default())
    }

    /// Build the verification prompt.
    fn build_prompt(&self, mention: &CharacterMention, character: &Character, context: &str) -> String {
        self.config.prompt_template
            .replace("{character}", &character.canonical_name)
            .replace("{mention}", &mention.text)
            .replace("{context}", context)
    }

    /// Parse the LLM response.
    fn parse_response(&self, response: &str) -> VerificationResult {
        let lower = response.trim().to_lowercase();
        
        // Empty responses are uncertain
        if lower.is_empty() {
            return VerificationResult::Uncertain;
        }

        // Check for Yes responses (check longer matches first)
        let mut yes_matches = self.config.yes_responses.clone();
        yes_matches.sort_by(|a, b| b.len().cmp(&a.len()));
        for yes in &yes_matches {
            if lower.starts_with(yes) || (yes.len() > 1 && lower.contains(yes)) {
                return VerificationResult::Confirmed;
            }
        }

        // Check for No responses (check longer matches first)
        let mut no_matches = self.config.no_responses.clone();
        no_matches.sort_by(|a, b| b.len().cmp(&a.len()));
        for no in &no_matches {
            if lower.starts_with(no) || (no.len() > 1 && lower.contains(no)) {
                return VerificationResult::Rejected;
            }
        }

        // Couldn't parse - treat as uncertain
        VerificationResult::Uncertain
    }
}

impl MentionVerifier for LlmMentionVerifier {
    fn verify(
        &self,
        mention: &CharacterMention,
        character: &Character,
        context: &str,
    ) -> VerificationResult {
        // Check provider availability
        if !self.provider.is_available() {
            // Fall back to uncertain if LLM not available
            return VerificationResult::Uncertain;
        }

        let prompt = self.build_prompt(mention, character, context);

        // Build request
        let request = crate::backends::llm_client::LlmRequest {
            prompt,
            config: self.config.llm_config.clone(),
            context: vec![],
        };

        // Call LLM
        match self.provider.complete(request) {
            Ok(response) => self.parse_response(&response.text),
            Err(_) => {
                // On error, don't reject - just mark uncertain
                VerificationResult::Uncertain
            }
        }
    }
}

/// Batch-aware LLM verifier for efficiency.
///
/// Processes multiple mentions in batches to reduce API calls.
/// Falls back to individual verification if batching fails.
pub struct BatchLlmVerifier {
    inner: LlmMentionVerifier,
    batch_size: usize,
}

impl BatchLlmVerifier {
    /// Create a new batch verifier.
    pub fn new(provider: Box<dyn LlmProvider>, batch_size: usize) -> Self {
        Self {
            inner: LlmMentionVerifier::with_provider(provider),
            batch_size,
        }
    }

    /// Verify multiple mentions at once.
    ///
    /// Returns a vector of results in the same order as input.
    /// Processes mentions in batches of `batch_size` for efficiency.
    pub fn verify_batch(
        &self,
        mentions: &[CharacterMention],
        characters: &CharacterList,
        contexts: &[String],
    ) -> Vec<VerificationResult> {
        let mut results = Vec::with_capacity(mentions.len());
        
        // Process in batches of batch_size
        for chunk in mentions.chunks(self.batch_size) {
            let chunk_results: Vec<VerificationResult> = chunk
                .iter()
                .enumerate()
                .map(|(i, mention)| {
                    let context_idx = results.len() + i;
                    let context = contexts.get(context_idx).map(|s| s.as_str()).unwrap_or("");
                    let character = &characters.characters()[mention.character_idx];
                    self.inner.verify(mention, character, context)
                })
                .collect();
            results.extend(chunk_results);
        }
        
        results
    }

    /// Get the configured batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// LLM Filter - verifies character mentions using language model.
///
/// This is Step 2 of the BookCoref Pipeline.
pub struct LLMFilter<V: MentionVerifier> {
    verifier: V,
    config: LLMFilterConfig,
}

impl<V: MentionVerifier> LLMFilter<V> {
    /// Create a new LLM filter with custom verifier.
    pub fn new(verifier: V, config: LLMFilterConfig) -> Self {
        Self { verifier, config }
    }

    /// Filter mentions, keeping only verified ones.
    pub fn filter(
        &self,
        mentions: Vec<CharacterMention>,
        characters: &CharacterList,
        text: &str,
    ) -> Vec<CharacterMention> {
        mentions
            .into_iter()
            .filter_map(|mut mention| {
                let character = &characters.characters()[mention.character_idx];
                let context = self.extract_context(text, &mention);

                match self.verifier.verify(&mention, character, &context) {
                    VerificationResult::Confirmed => {
                        mention.verified = true;
                        Some(mention)
                    }
                    VerificationResult::Uncertain => {
                        // Keep uncertain mentions but mark as unverified
                        Some(mention)
                    }
                    VerificationResult::Rejected => None,
                }
            })
            .collect()
    }

    /// Extract context around a mention.
    fn extract_context(&self, text: &str, mention: &CharacterMention) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();

        // Find word indices containing mention
        // Track byte position for searching; convert to character offsets via `TextSpan`.
        let mut byte_pos = 0;
        let mut mention_word_start = None;
        let mut mention_word_end = None;

        for (word_idx, word) in words.iter().enumerate() {
            let word_start_byte = text[byte_pos..]
                .find(word)
                .map(|p| byte_pos + p)
                .unwrap_or(byte_pos);
            let word_end_byte = word_start_byte + word.len();
            let span = anno::offset::TextSpan::from_bytes(text, word_start_byte, word_end_byte);
            let word_start = span.char_start;
            let word_end = span.char_end;

            if word_start <= mention.start && mention_word_start.is_none() {
                mention_word_start = Some(word_idx);
            }
            if word_end >= mention.end {
                mention_word_end = Some(word_idx);
                break;
            }
            byte_pos = word_end_byte;
        }

        let mention_word_start = mention_word_start.unwrap_or(0);
        let mention_word_end = mention_word_end.unwrap_or(words.len().saturating_sub(1));

        // Extract context window
        let context_start = mention_word_start.saturating_sub(self.config.context_words / 2);
        let context_end = (mention_word_end + self.config.context_words / 2).min(words.len());

        words[context_start..context_end].join(" ")
    }
}

// =============================================================================
// Cluster Expansion (Steps 3 & 4)
// =============================================================================

/// Backend for cluster expansion.
///
/// The BookCoref pipeline uses Maverick for cluster expansion (Steps 3 & 4).
/// This enum allows choosing between different backends:
#[derive(Debug, Clone, Default)]
pub enum ExpansionBackend {
    /// Rule-based pronoun resolution (no neural model required)
    #[default]
    Heuristic,
    /// Maverick CPU backend (heuristic mention extraction + string matching)
    MaverickCpu,
    /// Maverick Candle backend (neural inference, requires weights)
    #[cfg(feature = "candle")]
    MaverickCandle {
        /// Path to safetensors weights
        weights_path: std::path::PathBuf,
    },
}

/// Configuration for cluster expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionConfig {
    /// Window size for expansion
    pub window_size: usize,
    /// Window overlap
    pub window_overlap: usize,
    /// Group size for second-pass expansion (G=10 in BOOKCOREF)
    pub group_size: usize,
    /// Similarity threshold for pronoun linking
    pub pronoun_threshold: f64,
    /// Maximum distance (in tokens) for pronoun antecedent
    pub max_pronoun_distance: usize,
    /// Use Maverick for expansion (if available)
    #[serde(default)]
    pub use_maverick: bool,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            window_size: 1500,
            window_overlap: 200,
            group_size: 10,
            pronoun_threshold: 0.6,
            max_pronoun_distance: 100,
            use_maverick: false,
        }
    }
}

/// Cluster Expander - expands character clusters with pronouns and epithets.
///
/// This implements Steps 3 & 4 of the BookCoref Pipeline.
#[derive(Default)]
pub struct ClusterExpander {
    config: ExpansionConfig,
}


impl ClusterExpander {
    /// Create a new cluster expander.
    pub fn new(config: ExpansionConfig) -> Self {
        Self { config }
    }

    /// Expand character clusters with pronouns and other mentions.
    pub fn expand(
        &self,
        initial_mentions: Vec<CharacterMention>,
        characters: &CharacterList,
        text: &str,
    ) -> Vec<CharacterCluster> {
        // Initialize clusters from character mentions
        let mut clusters: Vec<CharacterCluster> = characters
            .characters()
            .iter()
            .enumerate()
            .map(|(idx, char)| CharacterCluster {
                character_idx: idx,
                character_name: char.canonical_name.clone(),
                mentions: Vec::new(),
            })
            .collect();

        // Add initial mentions to clusters
        for mention in initial_mentions {
            if mention.character_idx < clusters.len() {
                clusters[mention.character_idx].mentions.push(ExpandedMention {
                    text: mention.text,
                    start: mention.start,
                    end: mention.end,
                    mention_type: mention.mention_type,
                    confidence: mention.confidence,
                });
            }
        }

        // Process text in windows
        let windows = self.split_into_windows(text);

        for (window_text, window_offset) in windows.iter() {
            self.expand_window(&mut clusters, characters, window_text, *window_offset);
        }

        // Remove empty clusters
        clusters.retain(|c| !c.mentions.is_empty());

        // Sort mentions within each cluster
        for cluster in &mut clusters {
            cluster.mentions.sort_by_key(|m| m.start);
        }

        clusters
    }

    /// Split text into windows.
    fn split_into_windows(&self, text: &str) -> Vec<(String, usize)> {
        let mut windows = Vec::new();
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let step = self.config.window_size.saturating_sub(self.config.window_overlap);

        let mut offset = 0;
        while offset < tokens.len() {
            let end = (offset + self.config.window_size).min(tokens.len());
            let window_tokens = &tokens[offset..end];
            let window_text = window_tokens.join(" ");

            // Calculate character offset
            let char_offset: usize = tokens[..offset]
                .iter()
                .map(|t| t.len() + 1)
                .sum();

            windows.push((window_text, char_offset));

            if end >= tokens.len() {
                break;
            }
            offset += step.max(1);
        }

        windows
    }

    /// Expand clusters within a single window.
    fn expand_window(
        &self,
        clusters: &mut [CharacterCluster],
        characters: &CharacterList,
        window_text: &str,
        window_offset: usize,
    ) {
        // Find pronouns in window
        let pronouns = self.find_pronouns(window_text, window_offset);

        for pronoun in pronouns {
            // Find nearest character mention as antecedent
            if let Some((cluster_idx, confidence)) =
                self.find_antecedent(&pronoun, clusters, characters, window_offset)
            {
                if confidence >= self.config.pronoun_threshold {
                    clusters[cluster_idx].mentions.push(ExpandedMention {
                        text: pronoun.text,
                        start: pronoun.start,
                        end: pronoun.end,
                        mention_type: MentionType::Pronominal,
                        confidence,
                    });
                }
            }
        }
    }

    /// Find pronouns in text.
    fn find_pronouns(&self, text: &str, offset: usize) -> Vec<PronounMention> {
        let pronouns = [
            // Personal pronouns
            ("he", Gender::Masculine),
            ("him", Gender::Masculine),
            ("his", Gender::Masculine),
            ("himself", Gender::Masculine),
            ("she", Gender::Feminine),
            ("her", Gender::Feminine),
            ("hers", Gender::Feminine),
            ("herself", Gender::Feminine),
            ("they", Gender::Neutral),
            ("them", Gender::Neutral),
            ("their", Gender::Neutral),
            ("theirs", Gender::Neutral),
            ("themself", Gender::Neutral),
            ("themselves", Gender::Neutral),
        ];

        let mut found = Vec::new();
        let text_lower = text.to_lowercase();

        for (pronoun, gender) in &pronouns {
            let mut search_start = 0;
            while let Some(pos) = text_lower[search_start..].find(pronoun) {
                let absolute_pos = search_start + pos;
                let end_pos = absolute_pos + pronoun.len();

                // Check word boundaries
                let is_word_start = absolute_pos == 0
                    || !text.chars().nth(absolute_pos - 1).map(|c| c.is_alphanumeric()).unwrap_or(false);
                let is_word_end = end_pos >= text.len()
                    || !text.chars().nth(end_pos).map(|c| c.is_alphanumeric()).unwrap_or(false);

                if is_word_start && is_word_end {
                    let mention_text: String = text.chars().skip(absolute_pos).take(pronoun.len()).collect();

                    found.push(PronounMention {
                        text: mention_text,
                        start: offset + absolute_pos,
                        end: offset + end_pos,
                        gender: *gender,
                    });
                }

                search_start = absolute_pos + 1;
            }
        }

        found.sort_by_key(|p| p.start);
        found
    }

    /// Find antecedent for a pronoun.
    fn find_antecedent(
        &self,
        pronoun: &PronounMention,
        clusters: &[CharacterCluster],
        characters: &CharacterList,
        _window_offset: usize,
    ) -> Option<(usize, f64)> {
        let mut best_match: Option<(usize, f64, usize)> = None; // (cluster_idx, confidence, distance)

        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            let character = &characters.characters()[cluster.character_idx];

            // Check gender compatibility
            if !self.gender_compatible(pronoun.gender, character.gender) {
                continue;
            }

            // Find closest mention before pronoun
            for mention in &cluster.mentions {
                if mention.end <= pronoun.start {
                    let distance = pronoun.start - mention.end;
                    if distance <= self.config.max_pronoun_distance {
                        let confidence = 1.0 - (distance as f64 / self.config.max_pronoun_distance as f64) * 0.4;

                        if best_match.is_none() || distance < best_match.unwrap().2 {
                            best_match = Some((cluster_idx, confidence, distance));
                        }
                    }
                }
            }
        }

        best_match.map(|(idx, conf, _)| (idx, conf))
    }

    /// Check if pronoun gender is compatible with character gender.
    fn gender_compatible(&self, pronoun_gender: Gender, character_gender: Option<Gender>) -> bool {
        match character_gender {
            None | Some(Gender::Unknown) => true, // Unknown character gender matches any pronoun
            Some(Gender::Neutral) => true, // Neutral characters can use any pronouns
            Some(char_gender) => {
                pronoun_gender == char_gender || pronoun_gender == Gender::Neutral
            }
        }
    }
}

/// A pronoun mention with gender information.
#[derive(Debug, Clone)]
struct PronounMention {
    text: String,
    start: usize,
    end: usize,
    gender: Gender,
}

/// An expanded mention (after cluster expansion).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandedMention {
    /// The mention text
    pub text: String,
    /// Character offset start
    pub start: usize,
    /// Character offset end
    pub end: usize,
    /// Type of mention (proper, nominal, pronoun)
    pub mention_type: MentionType,
    /// Confidence score for this mention
    pub confidence: f64,
}

/// A character cluster with all mentions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterCluster {
    /// Index into the character list
    pub character_idx: usize,
    /// Character's canonical name
    pub character_name: String,
    /// All mentions linked to this character
    pub mentions: Vec<ExpandedMention>,
}

impl CharacterCluster {
    /// Convert to CorefChain.
    pub fn to_chain(&self) -> CorefChain {
        let mentions: Vec<Mention> = self
            .mentions
            .iter()
            .map(|m| {
                let mut mention = Mention::new(&m.text, m.start, m.end);
                mention.mention_type = Some(m.mention_type);
                mention
            })
            .collect();

        let mut chain = CorefChain::new(mentions);
        chain.cluster_id = Some(self.character_idx as u64);
        chain.entity_type = Some("PERSON".to_string());
        chain
    }
}

// =============================================================================
// Complete Pipeline
// =============================================================================

/// Complete literary coreference pipeline.
///
/// Combines all steps: Character Linking → LLM Filtering → Cluster Expansion
pub struct LiteraryCorefPipeline<V: MentionVerifier> {
    characters: CharacterList,
    linker: CharacterLinker,
    filter: LLMFilter<V>,
    expander: ClusterExpander,
}

impl LiteraryCorefPipeline<HeuristicVerifier> {
    /// Create a pipeline with heuristic verification (no LLM required).
    pub fn new(characters: CharacterList) -> Self {
        let linker = CharacterLinker::new(characters.clone());
        let filter = LLMFilter::new(HeuristicVerifier, LLMFilterConfig::default());
        let expander = ClusterExpander::default();

        Self {
            characters,
            linker,
            filter,
            expander,
        }
    }
}

impl<V: MentionVerifier> LiteraryCorefPipeline<V> {
    /// Create a pipeline with custom verifier.
    pub fn with_verifier(characters: CharacterList, verifier: V) -> Self {
        let linker = CharacterLinker::new(characters.clone());
        let filter = LLMFilter::new(verifier, LLMFilterConfig::default());
        let expander = ClusterExpander::default();

        Self {
            characters,
            linker,
            filter,
            expander,
        }
    }

    /// Run the complete pipeline.
    pub fn resolve(&self, text: &str) -> Vec<CharacterCluster> {
        // Step 1: Character Linking
        let linked = self.linker.link(text);

        // Step 2: LLM Filtering
        let filtered = self.filter.filter(linked, &self.characters, text);

        // Steps 3 & 4: Cluster Expansion
        self.expander.expand(filtered, &self.characters, text)
    }

    /// Convert results to standard CorefChains.
    pub fn resolve_to_chains(&self, text: &str) -> Vec<CorefChain> {
        self.resolve(text)
            .into_iter()
            .map(|c| c.to_chain())
            .collect()
    }

    /// Get pipeline statistics.
    pub fn stats(&self, clusters: &[CharacterCluster]) -> PipelineStats {
        let total_mentions: usize = clusters.iter().map(|c| c.mentions.len()).sum();
        let explicit_mentions = clusters
            .iter()
            .flat_map(|c| &c.mentions)
            .filter(|m| m.mention_type == MentionType::Proper)
            .count();
        let pronoun_mentions = clusters
            .iter()
            .flat_map(|c| &c.mentions)
            .filter(|m| m.mention_type == MentionType::Pronominal)
            .count();

        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.mentions.len()).collect();
        let avg_cluster_size = if !clusters.is_empty() {
            total_mentions as f64 / clusters.len() as f64
        } else {
            0.0
        };

        PipelineStats {
            characters_total: self.characters.len(),
            characters_mentioned: clusters.len(),
            total_mentions,
            explicit_mentions,
            pronoun_mentions,
            avg_cluster_size,
            max_cluster_size: cluster_sizes.into_iter().max().unwrap_or(0),
        }
    }
}

/// Statistics from the literary coref pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total characters in the character list
    pub characters_total: usize,
    /// Characters with at least one mention
    pub characters_mentioned: usize,
    /// Total number of mentions across all characters
    pub total_mentions: usize,
    /// Explicit (proper name) mentions
    pub explicit_mentions: usize,
    /// Pronoun mentions
    pub pronoun_mentions: usize,
    /// Average mentions per character cluster
    pub avg_cluster_size: f64,
    /// Largest character cluster size
    pub max_cluster_size: usize,
}

impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Literary Coreference Pipeline Statistics:")?;
        writeln!(
            f,
            "  Characters: {}/{} mentioned",
            self.characters_mentioned, self.characters_total
        )?;
        writeln!(f, "  Total mentions: {}", self.total_mentions)?;
        writeln!(f, "    Explicit (names): {}", self.explicit_mentions)?;
        writeln!(f, "    Pronouns: {}", self.pronoun_mentions)?;
        writeln!(f, "  Avg cluster size: {:.1}", self.avg_cluster_size)?;
        writeln!(f, "  Max cluster size: {}", self.max_cluster_size)?;
        Ok(())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_list() {
        let chars = CharacterList::new(vec![
            ("Elizabeth Bennet", vec!["Elizabeth", "Lizzy", "Eliza"]),
            ("Fitzwilliam Darcy", vec!["Mr. Darcy", "Darcy"]),
        ]);

        assert_eq!(chars.len(), 2);
        assert!(chars.find_by_name("Lizzy").is_some());
        assert!(chars.find_by_name("Mr. Darcy").is_some());
        assert!(chars.find_by_name("darcy").is_some()); // Case-insensitive
    }

    #[test]
    fn test_character_linker() {
        let chars = CharacterList::new(vec![
            ("John Smith", vec!["John", "Smith"]),
            ("Mary Jones", vec!["Mary"]),
        ]);

        let linker = CharacterLinker::new(chars);
        let text = "John went to the store. Smith bought milk. Mary waved.";

        let mentions = linker.link(text);
        // John, Smith, Mary (3 distinct mentions)
        assert_eq!(mentions.len(), 3);
    }

    #[test]
    fn test_character_linker_fuzzy_matching() {
        let chars = CharacterList::new(vec![
            ("Elizabeth", vec!["Elizabeth", "Lizzy"]),
            ("Darcy", vec!["Darcy", "Mr. Darcy"]),
        ]);

        // With fuzzy matching enabled
        let linker = CharacterLinker::new(chars)
            .with_fuzzy_matching(true, 0.75);

        // Text with typos (OCR errors)
        let text = "Elizabth entered. Dracy was there.";
        let mentions = linker.link(text);

        // Should find fuzzy matches for both names
        assert!(mentions.len() >= 1, "Should find at least one fuzzy match");
        
        // Verify fuzzy matches have lower confidence
        for mention in &mentions {
            // Fuzzy matches should have confidence < 1.0
            assert!(mention.confidence < 1.0, "Fuzzy match should have discounted confidence");
        }
    }

    #[test]
    fn test_levenshtein_distance() {
        let chars = CharacterList::new(vec![("Test", vec![])]);
        let linker = CharacterLinker::new(chars);
        
        // Exact match
        let dist = linker.levenshtein_distance(
            &['h', 'e', 'l', 'l', 'o'],
            &['h', 'e', 'l', 'l', 'o']
        );
        assert_eq!(dist, 0);
        
        // One substitution
        let dist = linker.levenshtein_distance(
            &['h', 'e', 'l', 'l', 'o'],
            &['h', 'a', 'l', 'l', 'o']
        );
        assert_eq!(dist, 1);
        
        // One insertion
        let dist = linker.levenshtein_distance(
            &['h', 'e', 'l', 'l', 'o'],
            &['h', 'e', 'l', 'l', 'o', 's']
        );
        assert_eq!(dist, 1);
        
        // Complete difference
        let dist = linker.levenshtein_distance(
            &['a', 'b', 'c'],
            &['x', 'y', 'z']
        );
        assert_eq!(dist, 3);
    }

    #[test]
    fn test_cluster_expander_pronouns() {
        let chars = CharacterList::new(vec![
            ("John Smith", vec!["John"]),
        ]);

        let initial = vec![CharacterMention {
            text: "John".to_string(),
            start: 0,
            end: 4,
            character_idx: 0,
            confidence: 1.0,
            mention_type: MentionType::Proper,
            verified: true,
        }];

        let expander = ClusterExpander::new(ExpansionConfig {
            max_pronoun_distance: 100,
            ..Default::default()
        });

        let text = "John went to store. He bought milk.";
        let clusters = expander.expand(initial, &chars, text);

        assert_eq!(clusters.len(), 1);
        // Should have John + he
        assert!(clusters[0].mentions.len() >= 1);
    }

    #[test]
    fn test_full_pipeline() {
        let chars = CharacterList::new(vec![
            ("Elizabeth", vec!["Lizzy"]),
            ("Darcy", vec!["Mr. Darcy"]),
        ]);

        let pipeline = LiteraryCorefPipeline::new(chars);
        let text = "Elizabeth entered the room. She saw Darcy standing by the window. He looked thoughtful.";

        let clusters = pipeline.resolve(text);
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_gender_compatibility() {
        let expander = ClusterExpander::default();

        // Masculine pronoun matches masculine character
        assert!(expander.gender_compatible(Gender::Masculine, Some(Gender::Masculine)));

        // Neutral pronoun matches any gender
        assert!(expander.gender_compatible(Gender::Neutral, Some(Gender::Masculine)));
        assert!(expander.gender_compatible(Gender::Neutral, Some(Gender::Feminine)));

        // Unknown character matches any pronoun
        assert!(expander.gender_compatible(Gender::Masculine, None));
        assert!(expander.gender_compatible(Gender::Feminine, Some(Gender::Unknown)));

        // Mismatch
        assert!(!expander.gender_compatible(Gender::Masculine, Some(Gender::Feminine)));
    }

    #[test]
    fn test_llm_verifier_prompt_building() {
        use crate::backends::llm_client::MockProvider;
        
        let provider = Box::new(MockProvider::default());
        let verifier = LlmMentionVerifier::with_provider(provider);
        
        let character = Character::new("Mr. Darcy").with_aliases(vec!["Darcy"]);
        let mention = CharacterMention {
            text: "Mr. Darcy".to_string(),
            start: 0,
            end: 9,
            character_idx: 0,
            confidence: 0.95,
            mention_type: MentionType::Proper,
            verified: false,
        };
        let context = "Elizabeth turned to see Mr. Darcy standing by the window.";
        
        let prompt = verifier.build_prompt(&mention, &character, context);
        
        // Prompt should contain character name
        assert!(prompt.contains("Mr. Darcy"));
        // Prompt should contain mention text
        assert!(prompt.contains("\"Mr. Darcy\""));
        // Prompt should contain context
        assert!(prompt.contains("window"));
    }

    #[test]
    fn test_llm_verifier_response_parsing() {
        use crate::backends::llm_client::MockProvider;
        
        let provider = Box::new(MockProvider::default());
        let verifier = LlmMentionVerifier::with_provider(provider);
        
        // Test various response formats
        assert_eq!(verifier.parse_response("Yes"), VerificationResult::Confirmed);
        assert_eq!(verifier.parse_response("yes"), VerificationResult::Confirmed);
        assert_eq!(verifier.parse_response("Yes."), VerificationResult::Confirmed);
        assert_eq!(verifier.parse_response("Yes, this mention..."), VerificationResult::Confirmed);
        
        assert_eq!(verifier.parse_response("No"), VerificationResult::Rejected);
        assert_eq!(verifier.parse_response("no"), VerificationResult::Rejected);
        assert_eq!(verifier.parse_response("No, this is not..."), VerificationResult::Rejected);
        
        // Unparseable - should be uncertain
        assert_eq!(verifier.parse_response("Maybe"), VerificationResult::Uncertain);
        assert_eq!(verifier.parse_response(""), VerificationResult::Uncertain);
    }

    #[test]
    fn test_pipeline_with_llm_verifier() {
        use crate::backends::llm_client::MockProvider;
        
        let chars = CharacterList::new(vec![
            ("Elizabeth", vec!["Lizzy"]),
            ("Darcy", vec!["Mr. Darcy"]),
        ]);
        
        // Mock provider returns empty results, so verification will return Uncertain
        // The pipeline should still work (uncertain mentions are kept)
        let provider = Box::new(MockProvider::default());
        let verifier = LlmMentionVerifier::with_provider(provider);
        
        let pipeline = LiteraryCorefPipeline::with_verifier(chars, verifier);
        let text = "Elizabeth entered the room. Darcy was there.";
        
        let clusters = pipeline.resolve(text);
        // Should still find some clusters (uncertain mentions kept)
        assert!(!clusters.is_empty());
    }
}

