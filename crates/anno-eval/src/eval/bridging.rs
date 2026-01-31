//! Bridging Anaphora Resolution.
//!
//! # Overview
//!
//! Bridging anaphora involves inferential links where the anaphor is
//! *associated* with but not *identical* to its antecedent.
//!
//! # Examples
//!
//! ```text
//! "The car broke down. The engine had overheated."
//!                       ^^^^^^^^^^ bridge (part-whole)
//! ```
//!
//! The definite NP "The engine" is interpreted via a bridging inference
//! (part-whole relation) to "The car" in the previous sentence.
//!
//! # Bridging Types
//!
//! | Type | Example | Relation |
//! |------|---------|----------|
//! | **Part-Whole** | car → engine | Meronymy |
//! | **Set-Membership** | students → one student | Set containment |
//! | **Role** | company → CEO | Functional role |
//! | **Attribute** | house → price | Property |
//! | **Other** | concert → audience | General association |
//!
//! # Corpora
//!
//! - **ISNotes**: OntoNotes-layer bridging annotations
//! - **BASHI**: bridging subtypes (definite/indefinite/comparative)
//! - **ARRAU RST**: bridging in discourse structure contexts
//!
//! # Annotation Challenges
//!
//! Inter-annotator agreement for bridging can be substantially lower than for
//! identity coreference. This reflects genuine ambiguity in what counts as a
//! “bridge” vs discourse coherence.
//!
//! # References
//!
//! - Hou et al. (2018): "Unrestricted Bridging Resolution"
//! - Rösiger et al. (2018): "Bridging Resolution"

use serde::{Deserialize, Serialize};

/// Type of bridging relation between anaphor and antecedent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BridgingType {
    /// Part-whole (meronymy): "the car" → "the engine"
    PartWhole,
    /// Set-membership: "the students" → "one student"
    SetMembership,
    /// Functional role: "the company" → "the CEO"
    Role,
    /// Attribute/property: "the house" → "the price"
    Attribute,
    /// Producer-product: "the author" → "the book"
    ProducerProduct,
    /// Event participant: "the wedding" → "the bride"
    EventParticipant,
    /// Location: "the city" → "the downtown area"
    Location,
    /// Comparative: "this report" → "the previous one"
    Comparative,
    /// Other associative relation
    Other(String),
}

impl Default for BridgingType {
    fn default() -> Self {
        Self::Other(String::new())
    }
}

impl BridgingType {
    /// Create from string label.
    pub fn from_label(label: &str) -> Self {
        match label.to_lowercase().as_str() {
            "part-whole" | "meronymy" | "part_whole" => Self::PartWhole,
            "set-membership" | "set_membership" | "set" => Self::SetMembership,
            "role" | "functional" => Self::Role,
            "attribute" | "property" => Self::Attribute,
            "producer-product" | "producer_product" => Self::ProducerProduct,
            "event-participant" | "event_participant" | "participant" => Self::EventParticipant,
            "location" | "spatial" => Self::Location,
            "comparative" => Self::Comparative,
            other => Self::Other(other.to_string()),
        }
    }

    /// Get canonical label.
    pub fn as_label(&self) -> &str {
        match self {
            Self::PartWhole => "part-whole",
            Self::SetMembership => "set-membership",
            Self::Role => "role",
            Self::Attribute => "attribute",
            Self::ProducerProduct => "producer-product",
            Self::EventParticipant => "event-participant",
            Self::Location => "location",
            Self::Comparative => "comparative",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// A bridging link between an anaphor and its antecedent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgingLink {
    /// The bridging anaphor (e.g., "the engine")
    pub anaphor: BridgingMention,
    /// The antecedent (e.g., "the car")
    pub antecedent: BridgingMention,
    /// Type of bridging relation
    pub bridging_type: BridgingType,
    /// Confidence in this link
    pub confidence: f64,
    /// Whether this is a lexical (predictable) or referential bridge
    pub is_lexical: bool,
}

impl BridgingLink {
    /// Create a new bridging link.
    pub fn new(
        anaphor: BridgingMention,
        antecedent: BridgingMention,
        bridging_type: BridgingType,
    ) -> Self {
        Self {
            anaphor,
            antecedent,
            bridging_type,
            confidence: 1.0,
            is_lexical: false,
        }
    }

    /// Check if this is a part-whole bridge.
    pub fn is_part_whole(&self) -> bool {
        matches!(self.bridging_type, BridgingType::PartWhole)
    }

    /// Check if this is a set-membership bridge.
    pub fn is_set_membership(&self) -> bool {
        matches!(self.bridging_type, BridgingType::SetMembership)
    }
}

/// A mention in a bridging relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgingMention {
    /// The mention text
    pub text: String,
    /// Start character offset
    pub start: usize,
    /// End character offset (exclusive)
    pub end: usize,
    /// Head word (for head-match evaluation)
    pub head: Option<String>,
    /// Sentence index (for cross-sentence bridging)
    pub sentence_idx: Option<usize>,
}

impl BridgingMention {
    /// Create a new bridging mention.
    pub fn new(text: &str, start: usize, end: usize) -> Self {
        Self {
            text: text.to_string(),
            start,
            end,
            head: None,
            sentence_idx: None,
        }
    }

    /// Set the head word.
    pub fn with_head(mut self, head: &str) -> Self {
        self.head = Some(head.to_string());
        self
    }

    /// Set the sentence index.
    pub fn with_sentence(mut self, idx: usize) -> Self {
        self.sentence_idx = Some(idx);
        self
    }
}

/// Bridging resolution result for a document.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgingDocument {
    /// Document ID
    pub id: String,
    /// Document text
    pub text: String,
    /// Detected bridging links
    pub links: Vec<BridgingLink>,
}

impl BridgingDocument {
    /// Create a new bridging document.
    pub fn new(id: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            text: text.to_string(),
            links: Vec::new(),
        }
    }

    /// Add a bridging link.
    pub fn add_link(&mut self, link: BridgingLink) {
        self.links.push(link);
    }

    /// Number of bridging links.
    pub fn len(&self) -> usize {
        self.links.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Get links by type.
    pub fn links_by_type(&self, bridging_type: &BridgingType) -> Vec<&BridgingLink> {
        self.links
            .iter()
            .filter(|l| &l.bridging_type == bridging_type)
            .collect()
    }
}

/// Evaluation metrics for bridging resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BridgingMetrics {
    /// Precision: correct / predicted
    pub precision: f64,
    /// Recall: correct / gold
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Number of predicted links
    pub predicted: usize,
    /// Number of gold links
    pub gold: usize,
    /// Number of correct links
    pub correct: usize,
    /// Breakdown by bridging type
    pub by_type: std::collections::HashMap<String, (f64, f64, f64)>, // (P, R, F1)
}

impl BridgingMetrics {
    /// Compute metrics from predicted and gold documents.
    pub fn compute(predicted: &[BridgingDocument], gold: &[BridgingDocument]) -> Self {
        let mut total_pred = 0;
        let mut total_gold = 0;
        let mut total_correct = 0;
        let mut by_type: std::collections::HashMap<String, (usize, usize, usize)> =
            std::collections::HashMap::new();

        for (pred_doc, gold_doc) in predicted.iter().zip(gold.iter()) {
            total_pred += pred_doc.links.len();
            total_gold += gold_doc.links.len();

            // Match links (exact span match)
            for pred_link in &pred_doc.links {
                let type_key = pred_link.bridging_type.as_label().to_string();
                by_type.entry(type_key.clone()).or_insert((0, 0, 0)).0 += 1;

                for gold_link in &gold_doc.links {
                    if Self::links_match(pred_link, gold_link) {
                        total_correct += 1;
                        by_type.entry(type_key.clone()).or_insert((0, 0, 0)).2 += 1;
                        break;
                    }
                }
            }

            for gold_link in &gold_doc.links {
                let type_key = gold_link.bridging_type.as_label().to_string();
                by_type.entry(type_key).or_insert((0, 0, 0)).1 += 1;
            }
        }

        let precision = if total_pred > 0 {
            total_correct as f64 / total_pred as f64
        } else {
            0.0
        };
        let recall = if total_gold > 0 {
            total_correct as f64 / total_gold as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        // Compute per-type metrics
        let by_type_metrics: std::collections::HashMap<String, (f64, f64, f64)> = by_type
            .into_iter()
            .map(|(k, (pred, gold, corr))| {
                let p = if pred > 0 {
                    corr as f64 / pred as f64
                } else {
                    0.0
                };
                let r = if gold > 0 {
                    corr as f64 / gold as f64
                } else {
                    0.0
                };
                let f = if p + r > 0.0 {
                    2.0 * p * r / (p + r)
                } else {
                    0.0
                };
                (k, (p, r, f))
            })
            .collect();

        Self {
            precision,
            recall,
            f1,
            predicted: total_pred,
            gold: total_gold,
            correct: total_correct,
            by_type: by_type_metrics,
        }
    }

    /// Check if two links match (exact span match).
    fn links_match(a: &BridgingLink, b: &BridgingLink) -> bool {
        a.anaphor.start == b.anaphor.start
            && a.anaphor.end == b.anaphor.end
            && a.antecedent.start == b.antecedent.start
            && a.antecedent.end == b.antecedent.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridging_type_from_label() {
        assert_eq!(
            BridgingType::from_label("part-whole"),
            BridgingType::PartWhole
        );
        assert_eq!(
            BridgingType::from_label("SET-MEMBERSHIP"),
            BridgingType::SetMembership
        );
        assert_eq!(BridgingType::from_label("role"), BridgingType::Role);
    }

    #[test]
    fn test_bridging_link_creation() {
        let anaphor = BridgingMention::new("the engine", 20, 30);
        let antecedent = BridgingMention::new("The car", 0, 7);
        let link = BridgingLink::new(anaphor, antecedent, BridgingType::PartWhole);

        assert!(link.is_part_whole());
        assert!(!link.is_set_membership());
        assert_eq!(link.bridging_type.as_label(), "part-whole");
    }

    #[test]
    fn test_bridging_document() {
        let mut doc = BridgingDocument::new("doc1", "The car broke down. The engine failed.");

        let anaphor = BridgingMention::new("The engine", 20, 30).with_sentence(1);
        let antecedent = BridgingMention::new("The car", 0, 7).with_sentence(0);
        let link = BridgingLink::new(anaphor, antecedent, BridgingType::PartWhole);

        doc.add_link(link);

        assert_eq!(doc.len(), 1);
        assert!(!doc.is_empty());
        assert_eq!(doc.links_by_type(&BridgingType::PartWhole).len(), 1);
    }

    #[test]
    fn test_bridging_metrics() {
        let mut pred_doc = BridgingDocument::new("doc1", "");
        pred_doc.add_link(BridgingLink::new(
            BridgingMention::new("the engine", 20, 30),
            BridgingMention::new("The car", 0, 7),
            BridgingType::PartWhole,
        ));

        let mut gold_doc = BridgingDocument::new("doc1", "");
        gold_doc.add_link(BridgingLink::new(
            BridgingMention::new("the engine", 20, 30),
            BridgingMention::new("The car", 0, 7),
            BridgingType::PartWhole,
        ));

        let metrics = BridgingMetrics::compute(&[pred_doc], &[gold_doc]);

        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.f1, 1.0);
    }
}
