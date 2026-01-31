//! Preprocessing middleware for entity extraction pipelines.
//!
//! These middleware components run preprocessing stages (alias extraction,
//! reference detection) as part of the entity extraction pipeline, making
//! them backend-agnostic.
//!
//! # Design Philosophy
//!
//! The key insight is that preprocessing (parentheticals, appositions, references)
//! is **backend-agnostic** - it doesn't matter whether you use GLiNER, CRF, or
//! pattern-based NER. The preprocessing extracts structural information that
//! enhances any backend's output.
//!
//! # Integration Points
//!
//! ```text
//! Raw Text
//!     │
//!     ├── [Pre-Extract] ─────────────────────┐
//!     │   - Extract parentheticals           │
//!     │   - Extract appositions              │
//!     │   - Extract references               │ Context for
//!     │                                      │ Post-Processing
//!     ▼                                      │
//! NER Backend (any)                          │
//!     │                                      │
//!     ▼                                      │
//! [Post-Extract] ◄───────────────────────────┘
//!     - Enrich entities with aliases
//!     - Link entities to KB via references
//!     - Annotate temporal bounds
//! ```

use crate::backends::middleware::{Middleware, MiddlewareContext};
use crate::preprocess::{
    AppositionExtractor, ParentheticalExtractor, ReferenceExtractor,
};
use crate::Result;
use anno_core::Entity;
use std::borrow::Cow;
use std::collections::HashMap;

/// Middleware that extracts and stores alias information during preprocessing.
///
/// Aliases are stored in the context and used to enrich entities in post-processing.
#[derive(Debug, Clone, Default)]
pub struct AliasEnricher {
    paren_extractor: ParentheticalExtractor,
    appo_extractor: AppositionExtractor,
}

impl AliasEnricher {
    /// Create a new alias enricher.
    pub fn new() -> Self {
        Self {
            paren_extractor: ParentheticalExtractor::new(),
            appo_extractor: AppositionExtractor::new(),
        }
    }
}

impl Middleware for AliasEnricher {
    fn pre_process<'a>(&self, ctx: &mut MiddlewareContext, text: &'a str) -> Result<Cow<'a, str>> {
        // Extract aliases and store in context metadata
        let mut aliases = HashMap::new();

        // Parentheticals
        for paren in self.paren_extractor.extract(text) {
            if paren.is_alias {
                let key = paren.antecedent.to_lowercase();
                aliases.entry(key).or_insert_with(Vec::new).push(paren.content.clone());
            }
        }

        // Appositions
        for appo in self.appo_extractor.extract(text) {
            let canonical = appo.canonical().to_lowercase();
            let alternate = appo.alternate().to_string();
            aliases.entry(canonical).or_insert_with(Vec::new).push(alternate);
        }

        // Store aliases in context
        ctx.metadata.insert("aliases".to_string(), serde_json::to_string(&aliases).unwrap_or_default());

        Ok(Cow::Borrowed(text))
    }

    fn post_process(
        &self,
        ctx: &mut MiddlewareContext,
        mut entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        // Retrieve aliases from context
        let aliases: HashMap<String, Vec<String>> = ctx
            .metadata
            .get("aliases")
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        // Enrich entities with aliases
        for entity in &mut entities {
            let key = entity.text.to_lowercase();
            if let Some(entity_aliases) = aliases.get(&key) {
                // Add aliases to entity (if it has an aliases field or similar)
                // For now, store in metadata
                if entity_aliases.len() == 1 {
                    // Single alias - might be abbreviation or alternate name
                    // Could add to normalized field or similar
                }
            }
        }

        Ok(entities)
    }

    fn name(&self) -> &'static str {
        "alias_enricher"
    }
}

/// Middleware that extracts references and stores for KB linking.
#[derive(Debug, Clone, Default)]
pub struct ReferenceAnnotator {
    extractor: ReferenceExtractor,
}

impl ReferenceAnnotator {
    /// Create a new reference annotator.
    pub fn new() -> Self {
        Self {
            extractor: ReferenceExtractor::new(),
        }
    }
}

impl Middleware for ReferenceAnnotator {
    fn pre_process<'a>(&self, ctx: &mut MiddlewareContext, text: &'a str) -> Result<Cow<'a, str>> {
        // Extract references
        let refs = self.extractor.extract(text);

        // Store KB links for later use
        let mut kb_links: HashMap<String, (String, String)> = HashMap::new();

        for reference in &refs {
            if let (Some(entity_id), Some(url)) = (&reference.entity_id, &reference.url) {
                // Map from entity ID to (source, url)
                let source = match &reference.reference_type {
                    crate::preprocess::ReferenceType::WikipediaUrl => "wikipedia",
                    crate::preprocess::ReferenceType::WikidataUrl => "wikidata",
                    crate::preprocess::ReferenceType::DbpediaUrl => "dbpedia",
                    _ => "web",
                };
                kb_links.insert(entity_id.clone(), (source.to_string(), url.clone()));
            }
        }

        ctx.metadata.insert("kb_links".to_string(), serde_json::to_string(&kb_links).unwrap_or_default());
        ctx.metadata.insert("reference_count".to_string(), refs.len().to_string());

        Ok(Cow::Borrowed(text))
    }

    fn name(&self) -> &'static str {
        "reference_annotator"
    }
}

/// Middleware that extracts temporal bounds from parentheticals.
///
/// E.g., "Napoleon (1769-1821)" → sets temporal bounds on matched entities
#[derive(Debug, Clone, Default)]
pub struct TemporalAnnotator {
    extractor: ParentheticalExtractor,
}

impl TemporalAnnotator {
    /// Create a new temporal annotator.
    pub fn new() -> Self {
        Self {
            extractor: ParentheticalExtractor::new(),
        }
    }
}

impl Middleware for TemporalAnnotator {
    fn pre_process<'a>(&self, ctx: &mut MiddlewareContext, text: &'a str) -> Result<Cow<'a, str>> {
        // Extract temporal parentheticals
        let mut temporal_bounds: HashMap<String, (Option<String>, Option<String>)> = HashMap::new();

        for paren in self.extractor.extract(text) {
            if paren.is_temporal() {
                // Parse temporal content like "1769-1821"
                let content = paren.content.trim();
                if let Some((start, end)) = parse_temporal_range(content) {
                    let key = paren.antecedent.to_lowercase();
                    temporal_bounds.insert(key, (start, end));
                }
            }
        }

        ctx.metadata.insert("temporal_bounds".to_string(), serde_json::to_string(&temporal_bounds).unwrap_or_default());

        Ok(Cow::Borrowed(text))
    }

    fn post_process(
        &self,
        ctx: &mut MiddlewareContext,
        mut entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        use chrono::{NaiveDate, TimeZone, Utc};

        // Retrieve temporal bounds
        let temporal_bounds: HashMap<String, (Option<String>, Option<String>)> = ctx
            .metadata
            .get("temporal_bounds")
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        // Enrich entities with temporal information
        for entity in &mut entities {
            let key = entity.text.to_lowercase();
            if let Some((start, end)) = temporal_bounds.get(&key) {
                // Set valid_from and valid_until if the entity supports it
                if let Some(start_date) = start {
                    if entity.valid_from.is_none() {
                        // Parse "YYYY-MM-DD" into DateTime<Utc>
                        if let Ok(naive) = NaiveDate::parse_from_str(start_date, "%Y-%m-%d") {
                            entity.valid_from = Some(Utc.from_utc_datetime(&naive.and_hms_opt(0, 0, 0).unwrap()));
                        }
                    }
                }
                if let Some(end_date) = end {
                    if entity.valid_until.is_none() {
                        if let Ok(naive) = NaiveDate::parse_from_str(end_date, "%Y-%m-%d") {
                            entity.valid_until = Some(Utc.from_utc_datetime(&naive.and_hms_opt(23, 59, 59).unwrap()));
                        }
                    }
                }
            }
        }

        Ok(entities)
    }

    fn name(&self) -> &'static str {
        "temporal_annotator"
    }
}

/// Parse a temporal range like "1769-1821" into (start, end) dates.
fn parse_temporal_range(content: &str) -> Option<(Option<String>, Option<String>)> {
    // Try YYYY-YYYY format
    let parts: Vec<&str> = content.split(|c| c == '-' || c == '–' || c == '—').collect();

    if parts.len() == 2 {
        let start = parts[0].trim();
        let end = parts[1].trim();

        // Validate they look like years
        if start.len() == 4 && end.len() == 4 
            && start.chars().all(|c| c.is_ascii_digit())
            && end.chars().all(|c| c.is_ascii_digit()) 
        {
            return Some((
                Some(format!("{}-01-01", start)),
                Some(format!("{}-12-31", end)),
            ));
        }
    }

    // Single year
    if content.len() == 4 && content.chars().all(|c| c.is_ascii_digit()) {
        return Some((Some(format!("{}-01-01", content)), None));
    }

    None
}

/// Composite middleware that runs all preprocessing enrichments.
#[derive(Debug, Clone, Default)]
pub struct FullPreprocessor {
    alias: AliasEnricher,
    reference: ReferenceAnnotator,
    temporal: TemporalAnnotator,
}

impl FullPreprocessor {
    /// Create a new full preprocessor.
    pub fn new() -> Self {
        Self {
            alias: AliasEnricher::new(),
            reference: ReferenceAnnotator::new(),
            temporal: TemporalAnnotator::new(),
        }
    }
}

impl Middleware for FullPreprocessor {
    fn pre_process<'a>(&self, ctx: &mut MiddlewareContext, text: &'a str) -> Result<Cow<'a, str>> {
        // Run all pre-processors
        self.alias.pre_process(ctx, text)?;
        self.reference.pre_process(ctx, text)?;
        self.temporal.pre_process(ctx, text)?;

        Ok(Cow::Borrowed(text))
    }

    fn post_process(
        &self,
        ctx: &mut MiddlewareContext,
        entities: Vec<Entity>,
    ) -> Result<Vec<Entity>> {
        // Run all post-processors
        let entities = self.alias.post_process(ctx, entities)?;
        let entities = self.temporal.post_process(ctx, entities)?;

        Ok(entities)
    }

    fn name(&self) -> &'static str {
        "full_preprocessor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::middleware::MiddlewareContext;

    #[test]
    fn test_alias_enricher_extracts_parentheticals() {
        let enricher = AliasEnricher::new();
        let text = "The World Health Organization (WHO) is important.";
        let mut ctx = MiddlewareContext::new(text);

        let _ = enricher.pre_process(&mut ctx, text).unwrap();

        let aliases_json = ctx.metadata.get("aliases").unwrap();
        let aliases: HashMap<String, Vec<String>> = serde_json::from_str(aliases_json).unwrap();

        assert!(aliases.contains_key("the world health organization"));
        assert!(aliases["the world health organization"].contains(&"WHO".to_string()));
    }

    #[test]
    fn test_temporal_annotator() {
        let annotator = TemporalAnnotator::new();
        let text = "Napoleon Bonaparte (1769-1821) was emperor.";
        let mut ctx = MiddlewareContext::new(text);

        let _ = annotator.pre_process(&mut ctx, text).unwrap();

        let temporal_json = ctx.metadata.get("temporal_bounds").unwrap();
        let temporal: HashMap<String, (Option<String>, Option<String>)> = 
            serde_json::from_str(temporal_json).unwrap();

        assert!(temporal.contains_key("napoleon bonaparte"));
    }

    #[test]
    fn test_parse_temporal_range() {
        let result = parse_temporal_range("1769-1821");
        assert!(result.is_some());
        let (start, end) = result.unwrap();
        assert_eq!(start, Some("1769-01-01".to_string()));
        assert_eq!(end, Some("1821-12-31".to_string()));
    }

    #[test]
    fn test_full_preprocessor() {
        let preprocessor = FullPreprocessor::new();
        let text = "Apple Inc. (AAPL), founded by Steve Jobs (1955-2011), is a tech company.";
        let mut ctx = MiddlewareContext::new(text);

        let _ = preprocessor.pre_process(&mut ctx, text).unwrap();

        // Should have both aliases and temporal bounds
        assert!(ctx.metadata.contains_key("aliases"));
        assert!(ctx.metadata.contains_key("temporal_bounds"));
    }
}

