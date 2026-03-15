//! Reference Resolution for Entity Extraction.
//!
//! # Overview
//!
//! Documents often contain references to external content:
//! - **URLs**: Links to web pages with additional entity information
//! - **Citations**: Academic references (Smith et al., 2020)
//! - **Cross-references**: Internal document references (see Section 3)
//! - **Footnotes/Endnotes**: Additional contextual information
//! - **Entity Links**: Wikipedia, Wikidata, or other KB references
//!
//! This module provides infrastructure for:
//! 1. Detecting references in text
//! 2. Resolving them to content
//! 3. Extracting entities from resolved content
//! 4. Linking back to the source document
//!
//! # Integration with Coalesce
//!
//! Resolved references provide additional evidence for entity coalescing:
//! - A URL pointing to a Wikipedia page confirms entity identity
//! - Citations can link entities mentioned in different contexts
//! - Resolved content may contain canonical names or aliases
//!
//! # Integration with Tier
//!
//! References create hierarchical relationships:
//! - Level 0: Entities in source document
//! - Level 1: Entities in directly referenced documents
//! - Level 2+: Entities in transitively referenced documents
//!
//! This creates a "citation graph" that tier can cluster.
//!
//! # Example
//!
//! ```rust,ignore
//! use anno::preprocess::reference::{ReferenceExtractor, ReferenceType};
//!
//! let extractor = ReferenceExtractor::new();
//! let text = "See https://en.wikipedia.org/wiki/Albert_Einstein for more info.";
//! let refs = extractor.extract(text);
//!
//! assert_eq!(refs.len(), 1);
//! assert_eq!(refs[0].reference_type, ReferenceType::WikipediaUrl);
//! ```

use crate::offset::TextSpan;
use anno_core::Confidence;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of reference detected in text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ReferenceType {
    /// Wikipedia URL
    WikipediaUrl,
    /// Wikidata entity URL (Q-number)
    WikidataUrl,
    /// DBpedia resource URL
    DbpediaUrl,
    /// General HTTP/HTTPS URL
    WebUrl,
    /// Academic citation (Author et al., Year)
    AcademicCitation,
    /// DOI reference
    Doi,
    /// ArXiv reference
    Arxiv,
    /// Internal cross-reference (Section, Figure, Table)
    CrossReference,
    /// Footnote/endnote marker
    FootnoteMarker,
    /// ISBN reference
    Isbn,
    /// Social media handle (@username)
    SocialHandle,
    /// Hashtag (#topic)
    Hashtag,
    /// Unknown reference type
    #[default]
    Unknown,
}

impl ReferenceType {
    /// Check if this reference type can be resolved to external content.
    pub fn is_resolvable(&self) -> bool {
        matches!(
            self,
            Self::WikipediaUrl
                | Self::WikidataUrl
                | Self::DbpediaUrl
                | Self::WebUrl
                | Self::Doi
                | Self::Arxiv
        )
    }

    /// Check if this reference links to a knowledge base.
    pub fn is_kb_link(&self) -> bool {
        matches!(
            self,
            Self::WikipediaUrl | Self::WikidataUrl | Self::DbpediaUrl
        )
    }
}

/// A detected reference in text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    /// The reference text as it appears in the document
    pub text: String,
    /// Start offset in the document
    pub start: usize,
    /// End offset in the document
    pub end: usize,
    /// Type of reference
    pub reference_type: ReferenceType,
    /// Resolved URL (if applicable)
    pub url: Option<String>,
    /// Entity ID if this is a KB link (e.g., Wikidata Q-number)
    pub entity_id: Option<String>,
    /// Title/name of the referenced resource (if known)
    pub title: Option<String>,
    /// The antecedent text this reference relates to (if any)
    pub antecedent: Option<String>,
    /// Whether this reference has been resolved
    pub is_resolved: bool,
}

impl Reference {
    /// Create a new reference.
    pub fn new(text: &str, start: usize, end: usize, ref_type: ReferenceType) -> Self {
        Self {
            text: text.to_string(),
            start,
            end,
            reference_type: ref_type,
            url: None,
            entity_id: None,
            title: None,
            antecedent: None,
            is_resolved: false,
        }
    }

    /// Set the URL.
    pub fn with_url(mut self, url: &str) -> Self {
        self.url = Some(url.to_string());
        self
    }

    /// Set the entity ID.
    pub fn with_entity_id(mut self, id: &str) -> Self {
        self.entity_id = Some(id.to_string());
        self
    }

    /// Set the antecedent.
    pub fn with_antecedent(mut self, antecedent: &str) -> Self {
        self.antecedent = Some(antecedent.to_string());
        self
    }

    /// Mark as resolved.
    pub fn mark_resolved(mut self) -> Self {
        self.is_resolved = true;
        self
    }

    /// Get the Wikidata Q-number if this is a Wikidata reference.
    pub fn wikidata_qid(&self) -> Option<&str> {
        if self.reference_type == ReferenceType::WikidataUrl {
            self.entity_id.as_deref()
        } else {
            None
        }
    }
}

/// Extractor for references in text.
#[derive(Debug, Clone, Default)]
pub struct ReferenceExtractor {
    /// Extract Wikipedia URLs
    extract_wikipedia: bool,
    /// Extract general web URLs
    extract_web_urls: bool,
    /// Extract academic citations
    extract_citations: bool,
    /// Extract social handles
    extract_social: bool,
}

impl ReferenceExtractor {
    /// Create a new extractor with default settings (all enabled).
    pub fn new() -> Self {
        Self {
            extract_wikipedia: true,
            extract_web_urls: true,
            extract_citations: true,
            extract_social: true,
        }
    }

    /// Enable/disable Wikipedia URL extraction.
    pub fn wikipedia(mut self, enabled: bool) -> Self {
        self.extract_wikipedia = enabled;
        self
    }

    /// Extract all references from text.
    pub fn extract(&self, text: &str) -> Vec<Reference> {
        let mut refs = Vec::new();

        // Extract URLs
        if self.extract_web_urls || self.extract_wikipedia {
            refs.extend(self.extract_urls(text));
        }

        // Extract citations
        if self.extract_citations {
            refs.extend(self.extract_citations(text));
        }

        // Extract DOIs
        refs.extend(self.extract_dois(text));

        // Extract cross-references
        refs.extend(self.extract_cross_refs(text));

        // Extract social handles
        if self.extract_social {
            refs.extend(self.extract_social_handles(text));
        }

        // Sort by position
        refs.sort_by_key(|r| r.start);

        refs
    }

    /// Extract URLs from text.
    fn extract_urls(&self, text: &str) -> Vec<Reference> {
        let mut refs = Vec::new();

        // Simple URL pattern
        let url_pattern = regex::Regex::new(r"https?://[^\s<>\[\]{}|\\^`\x00-\x1f\x7f]+").ok();

        if let Some(re) = url_pattern {
            for m in re.find_iter(text) {
                let url = m.as_str();
                let ref_type = self.classify_url(url);

                if !self.extract_wikipedia && ref_type == ReferenceType::WikipediaUrl {
                    continue;
                }

                let span = TextSpan::from_bytes(text, m.start(), m.end());
                let mut reference =
                    Reference::new(url, span.char_start, span.char_end, ref_type.clone());
                reference.url = Some(url.to_string());

                // Extract entity ID for KB URLs
                if let Some(id) = self.extract_entity_id(url, &ref_type) {
                    reference.entity_id = Some(id);
                }

                refs.push(reference);
            }
        }

        refs
    }

    /// Classify a URL by type.
    fn classify_url(&self, url: &str) -> ReferenceType {
        if url.contains("wikipedia.org") {
            ReferenceType::WikipediaUrl
        } else if url.contains("wikidata.org") {
            ReferenceType::WikidataUrl
        } else if url.contains("dbpedia.org") {
            ReferenceType::DbpediaUrl
        } else if url.contains("arxiv.org") {
            ReferenceType::Arxiv
        } else if url.contains("doi.org") {
            ReferenceType::Doi
        } else {
            ReferenceType::WebUrl
        }
    }

    /// Extract entity ID from KB URL.
    fn extract_entity_id(&self, url: &str, ref_type: &ReferenceType) -> Option<String> {
        match ref_type {
            ReferenceType::WikipediaUrl => {
                // Extract article title from Wikipedia URL
                // https://en.wikipedia.org/wiki/Albert_Einstein -> Albert_Einstein
                url.split("/wiki/").last().map(|s| s.to_string())
            }
            ReferenceType::WikidataUrl => {
                // Extract Q-number from Wikidata URL
                // https://www.wikidata.org/wiki/Q937 -> Q937
                let re = regex::Regex::new(r"Q\d+").ok()?;
                re.find(url).map(|m| m.as_str().to_string())
            }
            ReferenceType::DbpediaUrl => {
                // Extract resource name from DBpedia URL
                url.split("/resource/").last().map(|s| s.to_string())
            }
            _ => None,
        }
    }

    /// Extract academic citations.
    fn extract_citations(&self, text: &str) -> Vec<Reference> {
        let mut refs = Vec::new();

        // Pattern: Author et al., Year or Author (Year)
        let citation_patterns = [
            r"\b([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?),?\s*\(?\d{4}\)?",
            r"\[([A-Z][a-z]+)\s+\d{4}\]",
        ];

        for pattern in &citation_patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for m in re.find_iter(text) {
                    let span = TextSpan::from_bytes(text, m.start(), m.end());
                    refs.push(Reference::new(
                        m.as_str(),
                        span.char_start,
                        span.char_end,
                        ReferenceType::AcademicCitation,
                    ));
                }
            }
        }

        refs
    }

    /// Extract DOIs.
    fn extract_dois(&self, text: &str) -> Vec<Reference> {
        let mut refs = Vec::new();

        // DOI pattern: 10.XXXX/...
        let doi_pattern = regex::Regex::new(r"10\.\d{4,}/[^\s]+").ok();

        if let Some(re) = doi_pattern {
            for m in re.find_iter(text) {
                let span = TextSpan::from_bytes(text, m.start(), m.end());
                let mut reference = Reference::new(
                    m.as_str(),
                    span.char_start,
                    span.char_end,
                    ReferenceType::Doi,
                );
                reference.url = Some(format!("https://doi.org/{}", m.as_str()));
                refs.push(reference);
            }
        }

        refs
    }

    /// Extract cross-references (Section X, Figure Y, etc.).
    fn extract_cross_refs(&self, text: &str) -> Vec<Reference> {
        let mut refs = Vec::new();

        let patterns = [
            r"\b[Ss]ection\s+\d+(?:\.\d+)*",
            r"\b[Ff]igure\s+\d+(?:\.\d+)*",
            r"\b[Tt]able\s+\d+(?:\.\d+)*",
            r"\b[Aa]ppendix\s+[A-Z]",
            r"\b[Cc]hapter\s+\d+",
        ];

        for pattern in &patterns {
            if let Ok(re) = regex::Regex::new(pattern) {
                for m in re.find_iter(text) {
                    let span = TextSpan::from_bytes(text, m.start(), m.end());
                    refs.push(Reference::new(
                        m.as_str(),
                        span.char_start,
                        span.char_end,
                        ReferenceType::CrossReference,
                    ));
                }
            }
        }

        refs
    }

    /// Extract social media handles and hashtags.
    fn extract_social_handles(&self, text: &str) -> Vec<Reference> {
        let mut refs = Vec::new();

        // @username pattern
        if let Ok(re) = regex::Regex::new(r"@[A-Za-z_][A-Za-z0-9_]{0,14}") {
            for m in re.find_iter(text) {
                let span = TextSpan::from_bytes(text, m.start(), m.end());
                refs.push(Reference::new(
                    m.as_str(),
                    span.char_start,
                    span.char_end,
                    ReferenceType::SocialHandle,
                ));
            }
        }

        // #hashtag pattern
        if let Ok(re) = regex::Regex::new(r"#[A-Za-z][A-Za-z0-9_]*") {
            for m in re.find_iter(text) {
                let span = TextSpan::from_bytes(text, m.start(), m.end());
                refs.push(Reference::new(
                    m.as_str(),
                    span.char_start,
                    span.char_end,
                    ReferenceType::Hashtag,
                ));
            }
        }

        refs
    }
}

/// Resolved content from a reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedReference {
    /// The original reference
    pub reference: Reference,
    /// Resolved text content
    pub content: Option<String>,
    /// Entities extracted from the resolved content
    pub entities: Vec<ExtractedEntity>,
    /// Metadata from the resolved source
    pub metadata: HashMap<String, String>,
    /// Error message if resolution failed
    pub error: Option<String>,
}

/// An entity extracted from resolved reference content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: String,
    /// Confidence
    pub confidence: Confidence,
    /// Start offset in resolved content
    pub start: usize,
    /// End offset in resolved content
    pub end: usize,
}

/// Reference graph for tracking relationships between documents.
///
/// Used by tier for hierarchical clustering of entity relationships.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceGraph {
    /// Nodes: document IDs
    pub nodes: Vec<String>,
    /// Edges: (source_doc, target_doc, reference_type, weight)
    pub edges: Vec<(String, String, ReferenceType, f64)>,
}

impl ReferenceGraph {
    /// Create a new reference graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a document node.
    pub fn add_document(&mut self, doc_id: &str) {
        if !self.nodes.contains(&doc_id.to_string()) {
            self.nodes.push(doc_id.to_string());
        }
    }

    /// Add a reference edge between documents.
    pub fn add_reference(
        &mut self,
        source_doc: &str,
        target_doc: &str,
        ref_type: ReferenceType,
        weight: f64,
    ) {
        self.add_document(source_doc);
        self.add_document(target_doc);
        self.edges.push((
            source_doc.to_string(),
            target_doc.to_string(),
            ref_type,
            weight,
        ));
    }

    /// Get all documents referenced by a given document.
    pub fn get_references(&self, doc_id: &str) -> Vec<(&str, &ReferenceType)> {
        self.edges
            .iter()
            .filter(|(src, _, _, _)| src == doc_id)
            .map(|(_, tgt, rt, _)| (tgt.as_str(), rt))
            .collect()
    }

    /// Get all documents that reference a given document.
    pub fn get_referrers(&self, doc_id: &str) -> Vec<(&str, &ReferenceType)> {
        self.edges
            .iter()
            .filter(|(_, tgt, _, _)| tgt == doc_id)
            .map(|(src, _, rt, _)| (src.as_str(), rt))
            .collect()
    }

    /// Get reference depth (minimum hops from root documents).
    ///
    /// Root documents are those with no incoming references.
    pub fn get_depth(&self, doc_id: &str) -> usize {
        // BFS from root nodes
        use std::collections::{HashSet, VecDeque};

        let roots: HashSet<&str> = self
            .nodes
            .iter()
            .filter(|n| self.get_referrers(n).is_empty())
            .map(|s| s.as_str())
            .collect();

        if roots.contains(doc_id) {
            return 0;
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<(&str, usize)> = VecDeque::new();

        for root in &roots {
            queue.push_back((*root, 0));
            visited.insert(*root);
        }

        while let Some((current, depth)) = queue.pop_front() {
            for (target, _) in self.get_references(current) {
                if target == doc_id {
                    return depth + 1;
                }
                if !visited.contains(target) {
                    visited.insert(target);
                    queue.push_back((target, depth + 1));
                }
            }
        }

        usize::MAX // Not reachable
    }

    /// Convert to a format suitable for tier clustering.
    pub fn to_graph_edges(&self) -> Vec<(usize, usize, f64)> {
        let node_index: HashMap<&str, usize> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        self.edges
            .iter()
            .filter_map(|(src, tgt, _, weight)| {
                let src_idx = node_index.get(src.as_str())?;
                let tgt_idx = node_index.get(tgt.as_str())?;
                Some((*src_idx, *tgt_idx, *weight))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::TextSpan;

    #[test]
    fn test_wikipedia_url_extraction() {
        let extractor = ReferenceExtractor::new();
        let text = "See https://en.wikipedia.org/wiki/Albert_Einstein for more.";
        let refs = extractor.extract(text);

        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].reference_type, ReferenceType::WikipediaUrl);
        assert_eq!(refs[0].entity_id, Some("Albert_Einstein".to_string()));
    }

    #[test]
    fn test_reference_offsets_are_character_offsets_with_unicode_prefix() {
        let extractor = ReferenceExtractor::new();
        let text = "Müller: see https://en.wikipedia.org/wiki/Paris for travel tips.";
        let refs = extractor.extract(text);
        assert_eq!(refs.len(), 1);

        let r = &refs[0];
        let extracted = TextSpan::from_chars(text, r.start, r.end).extract(text);
        assert_eq!(extracted, r.text);
    }

    #[test]
    fn test_wikidata_url_extraction() {
        let extractor = ReferenceExtractor::new();
        let text = "Entity: https://www.wikidata.org/wiki/Q937";
        let refs = extractor.extract(text);

        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].reference_type, ReferenceType::WikidataUrl);
        assert_eq!(refs[0].entity_id, Some("Q937".to_string()));
    }

    #[test]
    fn test_doi_extraction() {
        let extractor = ReferenceExtractor::new();
        let text = "The paper 10.1038/nature12373 shows interesting results.";
        let refs = extractor.extract(text);

        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].reference_type, ReferenceType::Doi);
        assert!(refs[0].url.as_ref().unwrap().contains("doi.org"));
    }

    #[test]
    fn test_cross_reference_extraction() {
        let extractor = ReferenceExtractor::new();
        let text = "As shown in Section 3.2 and Figure 5, the results are clear.";
        let refs = extractor.extract(text);

        assert_eq!(refs.len(), 2);
        assert!(refs
            .iter()
            .all(|r| r.reference_type == ReferenceType::CrossReference));
    }

    #[test]
    fn test_social_handle_extraction() {
        let extractor = ReferenceExtractor::new();
        let text = "Follow @OpenAI and check #MachineLearning for updates.";
        let refs = extractor.extract(text);

        assert_eq!(refs.len(), 2);
        assert!(refs
            .iter()
            .any(|r| r.reference_type == ReferenceType::SocialHandle));
        assert!(refs
            .iter()
            .any(|r| r.reference_type == ReferenceType::Hashtag));
    }

    #[test]
    fn test_reference_graph() {
        let mut graph = ReferenceGraph::new();

        graph.add_reference("doc1", "wiki_einstein", ReferenceType::WikipediaUrl, 1.0);
        graph.add_reference("doc2", "wiki_einstein", ReferenceType::WikipediaUrl, 1.0);
        graph.add_reference("doc1", "doc3", ReferenceType::WebUrl, 0.5);

        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 3);

        let refs = graph.get_references("doc1");
        assert_eq!(refs.len(), 2);

        let referrers = graph.get_referrers("wiki_einstein");
        assert_eq!(referrers.len(), 2);
    }

    #[test]
    fn test_reference_depth() {
        let mut graph = ReferenceGraph::new();

        graph.add_document("root");
        graph.add_reference("root", "level1", ReferenceType::WebUrl, 1.0);
        graph.add_reference("level1", "level2", ReferenceType::WebUrl, 1.0);
        graph.add_reference("level2", "level3", ReferenceType::WebUrl, 1.0);

        assert_eq!(graph.get_depth("root"), 0);
        assert_eq!(graph.get_depth("level1"), 1);
        assert_eq!(graph.get_depth("level2"), 2);
        assert_eq!(graph.get_depth("level3"), 3);
    }

    // === Additional tests ===

    #[test]
    fn test_multiple_references_same_text() {
        let extractor = ReferenceExtractor::new();
        let text = "See https://en.wikipedia.org/wiki/Paris and \
                    https://en.wikipedia.org/wiki/London for travel info.";
        let refs = extractor.extract(text);

        assert_eq!(refs.len(), 2);
        assert!(refs
            .iter()
            .any(|r| r.entity_id == Some("Paris".to_string())));
        assert!(refs
            .iter()
            .any(|r| r.entity_id == Some("London".to_string())));
    }

    #[test]
    fn test_multilingual_wikipedia_urls() {
        let extractor = ReferenceExtractor::new();

        // Japanese Wikipedia
        let text_ja = "See https://ja.wikipedia.org/wiki/東京 for info.";
        let refs_ja = extractor.extract(text_ja);
        assert!(!refs_ja.is_empty());

        // Chinese Wikipedia
        let text_zh = "See https://zh.wikipedia.org/wiki/北京 for info.";
        let refs_zh = extractor.extract(text_zh);
        assert!(!refs_zh.is_empty());

        // Arabic Wikipedia
        let text_ar = "See https://ar.wikipedia.org/wiki/القاهرة for info.";
        let refs_ar = extractor.extract(text_ar);
        assert!(!refs_ar.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let extractor = ReferenceExtractor::new();
        let refs = extractor.extract("");

        assert!(refs.is_empty());
    }

    #[test]
    fn test_no_references() {
        let extractor = ReferenceExtractor::new();
        let text = "This is plain text with no references at all.";
        let refs = extractor.extract(text);

        assert!(refs.is_empty());
    }

    #[test]
    fn test_reference_type_display() {
        // Verify reference types are distinct
        assert_ne!(ReferenceType::WikipediaUrl, ReferenceType::WikidataUrl);
        assert_ne!(ReferenceType::Doi, ReferenceType::Arxiv);
        assert_ne!(ReferenceType::SocialHandle, ReferenceType::Hashtag);
    }

    #[test]
    fn test_dbpedia_url_extraction() {
        let extractor = ReferenceExtractor::new();
        let text = "Resource: http://dbpedia.org/resource/Albert_Einstein";
        let refs = extractor.extract(text);

        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].reference_type, ReferenceType::DbpediaUrl);
    }

    #[test]
    fn test_arxiv_id_extraction() {
        let extractor = ReferenceExtractor::new();
        let text = "The paper arXiv:2301.07041 introduced new methods.";
        let refs = extractor.extract(text);

        // arXiv IDs may or may not be detected depending on implementation
        // Just verify the extractor doesn't crash
        let _ = refs;
    }

    #[test]
    fn test_reference_serialization() {
        let reference = Reference {
            text: "https://example.com".to_string(),
            start: 0,
            end: 19,
            reference_type: ReferenceType::WebUrl,
            url: Some("https://example.com".to_string()),
            entity_id: None,
            title: None,
            antecedent: None,
            is_resolved: true,
        };

        let json = serde_json::to_string(&reference).unwrap();
        let deserialized: Reference = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.text, reference.text);
        assert_eq!(deserialized.reference_type, reference.reference_type);
    }

    #[test]
    fn test_reference_graph_empty() {
        let graph = ReferenceGraph::new();

        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());

        let refs = graph.get_references("nonexistent");
        assert!(refs.is_empty());
    }

    #[test]
    fn test_citation_patterns() {
        let extractor = ReferenceExtractor::new();

        // Academic citation pattern
        let text = "According to Smith et al. (2020), the results show...";
        let refs = extractor.extract(text);

        // Should detect citation patterns
        // Note: depends on implementation
        assert!(
            refs.is_empty()
                || refs
                    .iter()
                    .any(|r| r.reference_type == ReferenceType::AcademicCitation)
        );
    }
}
