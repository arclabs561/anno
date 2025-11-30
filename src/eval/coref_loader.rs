//! Coreference dataset loading and parsing.
//!
//! Provides specialized loaders for coreference datasets that return
//! `CorefDocument` structures rather than NER annotations.
//!
//! # Supported Datasets
//!
//! | Dataset | Format | Size | Features |
//! |---------|--------|------|----------|
//! | GAP | TSV | 8,908 pairs | Gender-balanced pronoun resolution |
//! | PreCo | JSON | ~38k docs | Large-scale, includes singletons |
//! | Synthetic | Generated | Configurable | For testing metrics |
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::eval::coref_loader::{CorefLoader, synthetic_coref_dataset};
//!
//! // Load GAP development set (requires eval-advanced feature for download)
//! let loader = CorefLoader::new().unwrap();
//! let docs = loader.load_gap().unwrap();
//!
//! // Or generate synthetic data for testing
//! let synthetic = synthetic_coref_dataset(10);
//! ```

use super::coref::{CorefChain, CorefDocument, Mention, MentionType};
use super::loader::DatasetId;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// =============================================================================
// GAP Dataset Structures
// =============================================================================

/// A single GAP example (pronoun-name pair).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapExample {
    /// Unique identifier
    pub id: String,
    /// Full text context
    pub text: String,
    /// The pronoun to resolve
    pub pronoun: String,
    /// Character offset of pronoun
    pub pronoun_offset: usize,
    /// First candidate name (A)
    pub name_a: String,
    /// Character offset of name A
    pub offset_a: usize,
    /// Whether pronoun refers to A
    pub coref_a: bool,
    /// Second candidate name (B)
    pub name_b: String,
    /// Character offset of name B
    pub offset_b: usize,
    /// Whether pronoun refers to B
    pub coref_b: bool,
    /// Source URL (Wikipedia)
    pub url: Option<String>,
}

impl GapExample {
    /// Convert to coreference document.
    ///
    /// Creates chains based on the coreference labels.
    #[must_use]
    pub fn to_coref_document(&self) -> CorefDocument {
        let mut chains = Vec::new();

        // Create mention for pronoun
        let pronoun_mention = Mention::with_type(
            &self.pronoun,
            self.pronoun_offset,
            self.pronoun_offset + self.pronoun.len(),
            MentionType::Pronoun,
        );

        // Create mentions for names
        let mention_a = Mention::with_type(
            &self.name_a,
            self.offset_a,
            self.offset_a + self.name_a.len(),
            MentionType::Proper,
        );

        let mention_b = Mention::with_type(
            &self.name_b,
            self.offset_b,
            self.offset_b + self.name_b.len(),
            MentionType::Proper,
        );

        // Build chains based on coreference labels
        if self.coref_a {
            // Pronoun refers to A
            chains.push(CorefChain::new(vec![mention_a, pronoun_mention.clone()]));
            chains.push(CorefChain::singleton(mention_b));
        } else if self.coref_b {
            // Pronoun refers to B
            chains.push(CorefChain::singleton(mention_a));
            chains.push(CorefChain::new(vec![mention_b, pronoun_mention.clone()]));
        } else {
            // Neither (pronoun doesn't refer to either candidate)
            chains.push(CorefChain::singleton(mention_a));
            chains.push(CorefChain::singleton(mention_b));
            chains.push(CorefChain::singleton(pronoun_mention));
        }

        CorefDocument::with_id(&self.text, &self.id, chains)
    }
}

// =============================================================================
// PreCo Dataset Structures
// =============================================================================

/// A PreCo document with coreference annotations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreCoDocument {
    /// Document ID
    pub id: String,
    /// Sentences as token arrays
    pub sentences: Vec<Vec<String>>,
    /// Coreference mentions: (sentence_idx, start_token, end_token, cluster_id)
    pub mentions: Vec<(usize, usize, usize, usize)>,
}

impl PreCoDocument {
    /// Convert to coreference document.
    #[must_use]
    pub fn to_coref_document(&self) -> CorefDocument {
        // Reconstruct text from sentences
        let mut text = String::new();
        let mut sentence_offsets: Vec<usize> = Vec::new();
        let mut token_offsets: Vec<Vec<(usize, usize)>> = Vec::new();

        for sentence in &self.sentences {
            sentence_offsets.push(text.len());
            let mut sent_offsets = Vec::new();

            for (i, token) in sentence.iter().enumerate() {
                if i > 0 {
                    text.push(' ');
                }
                let start = text.len();
                text.push_str(token);
                let end = text.len();
                sent_offsets.push((start, end));
            }
            text.push(' ');
            token_offsets.push(sent_offsets);
        }

        // Group mentions by cluster
        let mut clusters: HashMap<usize, Vec<Mention>> = HashMap::new();

        for &(sent_idx, start_tok, end_tok, cluster_id) in &self.mentions {
            if sent_idx >= token_offsets.len() {
                continue;
            }
            let sent_tokens = &token_offsets[sent_idx];
            if start_tok >= sent_tokens.len() || end_tok > sent_tokens.len() {
                continue;
            }

            // Note: sent_tokens stores byte offsets (from text.len())
            let byte_start = sent_tokens[start_tok].0;
            let byte_end = sent_tokens[end_tok.saturating_sub(1).max(start_tok)].1;
            let mention_text = text[byte_start..byte_end].to_string();

            // Convert byte offsets to character offsets for Mention (which expects char offsets)
            let char_start = text[..byte_start].chars().count();
            let char_end = char_start + mention_text.chars().count();

            let mention = Mention::new(mention_text, char_start, char_end);
            clusters.entry(cluster_id).or_default().push(mention);
        }

        // Convert clusters to chains
        let chains: Vec<CorefChain> = clusters
            .into_iter()
            .map(|(id, mentions)| CorefChain::with_id(mentions, id as u64))
            .collect();

        CorefDocument::with_id(text, &self.id, chains)
    }
}

// =============================================================================
// Coreference Loader (delegates to DatasetLoader)
// =============================================================================

/// Loader for coreference datasets.
///
/// This is a thin wrapper around `DatasetLoader` that provides coreference-specific
/// loading methods. For most use cases, you can use `DatasetLoader::load_coref()` directly.
pub struct CorefLoader {
    inner: super::loader::DatasetLoader,
}

impl CorefLoader {
    /// Create a new loader with default cache directory.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: super::loader::DatasetLoader::new()?,
        })
    }

    /// Create loader with custom cache directory.
    pub fn with_cache_dir(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            inner: super::loader::DatasetLoader::with_cache_dir(cache_dir)?,
        })
    }

    /// Load GAP dataset as coreference documents.
    pub fn load_gap(&self) -> Result<Vec<CorefDocument>> {
        self.inner.load_coref(DatasetId::GAP)
    }

    /// Load GAP as raw examples (for detailed analysis).
    pub fn load_gap_examples(&self) -> Result<Vec<GapExample>> {
        let cache_path = self.inner.cache_path(DatasetId::GAP);

        if !cache_path.exists() {
            return Err(Error::InvalidInput(format!(
                "GAP dataset not cached at {:?}",
                cache_path
            )));
        }

        let content = fs::read_to_string(&cache_path)
            .map_err(|e| Error::InvalidInput(format!("Failed to read {:?}: {}", cache_path, e)))?;

        parse_gap_tsv(&content)
    }

    /// Load PreCo dataset as coreference documents.
    pub fn load_preco(&self) -> Result<Vec<CorefDocument>> {
        self.inner.load_coref(DatasetId::PreCo)
    }

    /// Check if a coreference dataset is cached.
    #[must_use]
    pub fn is_cached(&self, id: DatasetId) -> bool {
        self.inner.is_cached(id)
    }

    /// Get the underlying DatasetLoader.
    #[must_use]
    pub fn dataset_loader(&self) -> &super::loader::DatasetLoader {
        &self.inner
    }
}

impl Default for CorefLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create CorefLoader")
    }
}

// =============================================================================
// Parsers
// =============================================================================

/// Parse GAP TSV format.
///
/// Format: `ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL`
///
/// This function is public for use by `DatasetLoader`.
pub fn parse_gap_tsv(content: &str) -> Result<Vec<GapExample>> {
    let mut examples = Vec::new();
    let mut first_line = true;

    for line in content.lines() {
        // Skip header
        if first_line {
            first_line = false;
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 10 {
            continue;
        }

        let id = parts[0].to_string();
        let text = parts[1].to_string();
        let pronoun = parts[2].to_string();
        let pronoun_offset: usize = parts[3].parse().unwrap_or(0);
        let name_a = parts[4].to_string();
        let offset_a: usize = parts[5].parse().unwrap_or(0);
        let coref_a = parts[6].to_lowercase() == "true";
        let name_b = parts[7].to_string();
        let offset_b: usize = parts[8].parse().unwrap_or(0);
        let coref_b = parts[9].to_lowercase() == "true";
        let url = parts.get(10).map(|s| s.to_string());

        examples.push(GapExample {
            id,
            text,
            pronoun,
            pronoun_offset,
            name_a,
            offset_a,
            coref_a,
            name_b,
            offset_b,
            coref_b,
            url,
        });
    }

    Ok(examples)
}

/// Parse PreCo JSON format.
/// Parse PreCo JSON format (public for use by DatasetLoader).
pub fn parse_preco_json(content: &str) -> Result<Vec<PreCoDocument>> {
    let parsed: serde_json::Value = serde_json::from_str(content)
        .map_err(|e| Error::InvalidInput(format!("Invalid PreCo JSON: {}", e)))?;

    let mut docs = Vec::new();

    if let Some(doc_array) = parsed.as_array() {
        for (idx, doc) in doc_array.iter().enumerate() {
            // Extract sentences
            let sentences: Vec<Vec<String>> = doc
                .get("sentences")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|sent| {
                            sent.as_array().map(|tokens| {
                                tokens
                                    .iter()
                                    .filter_map(|t| t.as_str().map(String::from))
                                    .collect()
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            // Extract mentions
            let mentions: Vec<(usize, usize, usize, usize)> =
                doc.get("mention_clusters")
                    .and_then(|m| m.as_array())
                    .map(|clusters| {
                        clusters
                            .iter()
                            .enumerate()
                            .flat_map(|(cluster_id, cluster)| {
                                cluster.as_array().into_iter().flatten().filter_map(
                                    move |mention| {
                                        let arr = mention.as_array()?;
                                        if arr.len() >= 3 {
                                            Some((
                                                arr[0].as_u64()? as usize,
                                                arr[1].as_u64()? as usize,
                                                arr[2].as_u64()? as usize,
                                                cluster_id,
                                            ))
                                        } else {
                                            None
                                        }
                                    },
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();

            let id = doc
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or(&format!("doc_{}", idx))
                .to_string();

            docs.push(PreCoDocument {
                id,
                sentences,
                mentions,
            });
        }
    }

    Ok(docs)
}

// =============================================================================
// Synthetic Coreference Data
// =============================================================================

/// Generate synthetic coreference documents for testing.
///
/// Useful for validating metrics without downloading real datasets.
#[must_use]
pub fn synthetic_coref_dataset(num_docs: usize) -> Vec<CorefDocument> {
    let templates = [
        // Simple pronoun resolution
        (
            "John Smith went to the store. He bought some milk.",
            vec![
                ("John Smith", 0, 10, 0),
                ("He", 35, 37, 0),
            ],
        ),
        // Multiple entities
        (
            "Mary called Bob. She asked him about the meeting.",
            vec![
                ("Mary", 0, 4, 0),
                ("She", 17, 20, 0),
                ("Bob", 12, 15, 1),
                ("him", 27, 30, 1),
            ],
        ),
        // Longer chain
        (
            "The CEO announced the merger. She said the company would benefit. The executive was confident.",
            vec![
                ("The CEO", 0, 7, 0),
                ("She", 30, 33, 0),
                ("The executive", 68, 81, 0),
            ],
        ),
        // Nested mentions (company and its parts)
        (
            "Apple released a new iPhone. The tech giant's device sold well.",
            vec![
                ("Apple", 0, 5, 0),
                ("The tech giant", 29, 43, 0),
                ("iPhone", 21, 27, 1),
                ("device", 46, 52, 1),
            ],
        ),
        // Singletons
        (
            "The weather was nice. Sarah went for a walk in the park.",
            vec![
                ("The weather", 0, 11, 0),
                ("Sarah", 22, 27, 1),
                ("the park", 47, 55, 2),
            ],
        ),
    ];

    let mut docs = Vec::new();

    for i in 0..num_docs {
        let (text, mentions) = &templates[i % templates.len()];

        // Group mentions by cluster
        let mut clusters: HashMap<usize, Vec<Mention>> = HashMap::new();
        for &(mention_text, start, end, cluster_id) in mentions {
            let mention = Mention::new(mention_text, start, end);
            clusters.entry(cluster_id).or_default().push(mention);
        }

        let chains: Vec<CorefChain> = clusters
            .into_iter()
            .map(|(id, mentions)| CorefChain::with_id(mentions, id as u64))
            .collect();

        docs.push(CorefDocument::with_id(
            *text,
            format!("synthetic_{}", i),
            chains,
        ));
    }

    docs
}

/// Generate domain-specific synthetic coreference documents.
#[must_use]
pub fn domain_specific_coref_dataset(domain: &str) -> Vec<CorefDocument> {
    match domain {
        "biomedical" => biomedical_coref_examples(),
        "legal" => legal_coref_examples(),
        "news" => news_coref_examples(),
        _ => synthetic_coref_dataset(5),
    }
}

/// Biomedical coreference examples.
fn biomedical_coref_examples() -> Vec<CorefDocument> {
    vec![
        CorefDocument::with_id(
            "BRCA1 is a tumor suppressor gene. It plays a role in DNA repair. The gene is frequently mutated in breast cancer.",
            "bio_1",
            vec![CorefChain::new(vec![
                Mention::new("BRCA1", 0, 5),
                Mention::new("It", 34, 36),
                Mention::new("The gene", 62, 70),
            ])],
        ),
        CorefDocument::with_id(
            "The patient presented with chest pain. She was diagnosed with myocardial infarction. The woman received immediate treatment.",
            "bio_2",
            vec![
                CorefChain::new(vec![
                    Mention::new("The patient", 0, 11),
                    Mention::new("She", 39, 42),
                    Mention::new("The woman", 85, 94),
                ]),
                CorefChain::singleton(Mention::new("myocardial infarction", 62, 83)),
            ],
        ),
        CorefDocument::with_id(
            "Aspirin inhibits COX-1 and COX-2. The drug reduces inflammation. It is commonly used for pain relief.",
            "bio_3",
            vec![
                CorefChain::new(vec![
                    Mention::new("Aspirin", 0, 7),
                    Mention::new("The drug", 35, 43),
                    Mention::new("It", 65, 67),
                ]),
                CorefChain::singleton(Mention::new("COX-1", 17, 22)),
                CorefChain::singleton(Mention::new("COX-2", 27, 32)),
            ],
        ),
    ]
}

/// Legal coreference examples.
fn legal_coref_examples() -> Vec<CorefDocument> {
    vec![
        CorefDocument::with_id(
            "The defendant entered into a contract with the plaintiff. He failed to deliver the goods. The accused claimed force majeure.",
            "legal_1",
            vec![
                CorefChain::new(vec![
                    Mention::new("The defendant", 0, 13),
                    Mention::new("He", 58, 60),
                    Mention::new("The accused", 89, 100),
                ]),
                CorefChain::singleton(Mention::new("the plaintiff", 43, 56)),
            ],
        ),
        CorefDocument::with_id(
            "Article 5 of the Treaty governs this matter. It states that parties must negotiate in good faith. The provision has been interpreted broadly.",
            "legal_2",
            vec![CorefChain::new(vec![
                Mention::new("Article 5 of the Treaty", 0, 23),
                Mention::new("It", 45, 47),
                Mention::new("The provision", 99, 112),
            ])],
        ),
    ]
}

/// News coreference examples.
fn news_coref_examples() -> Vec<CorefDocument> {
    vec![
        CorefDocument::with_id(
            "President Biden met with Chancellor Scholz. The American leader discussed trade. He emphasized cooperation. Biden later held a press conference.",
            "news_1",
            vec![
                CorefChain::new(vec![
                    Mention::new("President Biden", 0, 14),
                    Mention::new("The American leader", 44, 63),
                    Mention::new("He", 81, 83),
                    Mention::new("Biden", 107, 112),
                ]),
                CorefChain::singleton(Mention::new("Chancellor Scholz", 25, 42)),
            ],
        ),
        CorefDocument::with_id(
            "Nvidia announced record quarterly earnings. The chipmaker exceeded expectations. Its stock rose 5% in after-hours trading.",
            "news_2",
            vec![
                CorefChain::new(vec![
                    Mention::new("Nvidia", 0, 6),
                    Mention::new("The chipmaker", 44, 57),
                    Mention::new("Its", 80, 83),
                ]),
            ],
        ),
        CorefDocument::with_id(
            "The hurricane made landfall in Florida. It caused widespread damage. The storm was Category 4. Authorities ordered evacuations before it arrived.",
            "news_3",
            vec![
                CorefChain::new(vec![
                    Mention::new("The hurricane", 0, 13),
                    Mention::new("It", 40, 42),
                    Mention::new("The storm", 68, 77),
                    Mention::new("it", 133, 135),
                ]),
            ],
        ),
    ]
}

/// Generate adversarial coreference examples.
///
/// These stress-test edge cases in coreference metrics.
#[must_use]
pub fn adversarial_coref_examples() -> Vec<(CorefDocument, CorefDocument, &'static str)> {
    vec![
        // Over-clustering: system merges two distinct entities
        (
            CorefDocument::new(
                "John saw Mary. He waved.",
                vec![
                    CorefChain::new(vec![Mention::new("John", 0, 4), Mention::new("He", 15, 17)]),
                    CorefChain::singleton(Mention::new("Mary", 9, 13)),
                ],
            ),
            CorefDocument::new(
                "John saw Mary. He waved.",
                vec![CorefChain::new(vec![
                    Mention::new("John", 0, 4),
                    Mention::new("Mary", 9, 13),
                    Mention::new("He", 15, 17),
                ])],
            ),
            "over-clustering",
        ),
        // Under-clustering: system splits one entity
        (
            CorefDocument::new(
                "Barack Obama gave a speech. The president was eloquent. Obama smiled.",
                vec![CorefChain::new(vec![
                    Mention::new("Barack Obama", 0, 12),
                    Mention::new("The president", 28, 41),
                    Mention::new("Obama", 56, 61),
                ])],
            ),
            CorefDocument::new(
                "Barack Obama gave a speech. The president was eloquent. Obama smiled.",
                vec![
                    CorefChain::new(vec![
                        Mention::new("Barack Obama", 0, 12),
                        Mention::new("Obama", 56, 61),
                    ]),
                    CorefChain::singleton(Mention::new("The president", 28, 41)),
                ],
            ),
            "under-clustering",
        ),
        // Missed mention: system finds fewer mentions
        (
            CorefDocument::new(
                "The dog ran. It was fast. The animal stopped.",
                vec![CorefChain::new(vec![
                    Mention::new("The dog", 0, 7),
                    Mention::new("It", 13, 15),
                    Mention::new("The animal", 26, 36),
                ])],
            ),
            CorefDocument::new(
                "The dog ran. It was fast. The animal stopped.",
                vec![CorefChain::new(vec![
                    Mention::new("The dog", 0, 7),
                    Mention::new("It", 13, 15),
                ])], // Missing "The animal"
            ),
            "missed-mention",
        ),
        // All singletons vs all in one cluster
        (
            CorefDocument::new(
                "A B C",
                vec![
                    CorefChain::singleton(Mention::new("A", 0, 1)),
                    CorefChain::singleton(Mention::new("B", 2, 3)),
                    CorefChain::singleton(Mention::new("C", 4, 5)),
                ],
            ),
            CorefDocument::new(
                "A B C",
                vec![CorefChain::new(vec![
                    Mention::new("A", 0, 1),
                    Mention::new("B", 2, 3),
                    Mention::new("C", 4, 5),
                ])],
            ),
            "singletons-vs-one-cluster",
        ),
    ]
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_example_to_coref() {
        let example = GapExample {
            id: "test-1".to_string(),
            text: "John saw Mary. He waved.".to_string(),
            pronoun: "He".to_string(),
            pronoun_offset: 15,
            name_a: "John".to_string(),
            offset_a: 0,
            coref_a: true,
            name_b: "Mary".to_string(),
            offset_b: 9,
            coref_b: false,
            url: None,
        };

        let doc = example.to_coref_document();
        assert_eq!(doc.mention_count(), 3);
        assert_eq!(doc.chain_count(), 2); // John+He, Mary
    }

    #[test]
    fn test_synthetic_coref_dataset() {
        let docs = synthetic_coref_dataset(5);
        assert_eq!(docs.len(), 5);

        for doc in &docs {
            assert!(!doc.text.is_empty());
            assert!(!doc.chains.is_empty());
        }
    }

    #[test]
    fn test_adversarial_examples() {
        let examples = adversarial_coref_examples();
        assert!(!examples.is_empty());

        for (gold, pred, name) in &examples {
            assert!(!gold.chains.is_empty(), "Gold chains empty for {}", name);
            assert!(!pred.chains.is_empty(), "Pred chains empty for {}", name);
        }
    }

    #[test]
    fn test_gap_tsv_parsing() {
        let tsv = "ID\tText\tPronoun\tPronoun-offset\tA\tA-offset\tA-coref\tB\tB-offset\tB-coref\tURL\n\
                   1\tJohn saw Mary. He waved.\tHe\t15\tJohn\t0\tTRUE\tMary\t9\tFALSE\thttps://example.com";

        let examples = parse_gap_tsv(tsv).unwrap();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].id, "1");
        assert!(examples[0].coref_a);
        assert!(!examples[0].coref_b);
    }
}
