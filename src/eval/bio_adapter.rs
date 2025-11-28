//! BIO tag sequence adapter.
//!
//! This module provides utilities for converting between BIO-tagged sequences
//! and entity spans.
//!
//! # Supported Schemes
//!
//! - IOB1: Inside-Outside-Begin (I appears first, B only when needed)
//! - IOB2: Inside-Outside-Begin (B always starts entity) - **most common**
//! - IOE1: Inside-Outside-End (E appears last, I continues)
//! - IOE2: Inside-Outside-End (E always ends entity)
//! - IOBES/BILOU: Begin-Inside-Last-Outside-Unit
//!
//! # Example
//!
//! ```rust
//! use anno::eval::bio_adapter::{BioScheme, bio_to_entities, entities_to_bio};
//! use anno::{Entity, EntityType};
//!
//! // Convert BIO tags to entities
//! let tokens = ["John", "Smith", "works", "at", "Apple"];
//! let tags = ["B-PER", "I-PER", "O", "O", "B-ORG"];
//! let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
//!
//! assert_eq!(entities.len(), 2);
//! assert_eq!(entities[0].text, "John Smith");
//! assert_eq!(entities[1].text, "Apple");
//! ```

use crate::{Entity, EntityType, Result};
use std::fmt;

/// BIO tagging scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BioScheme {
    /// IOB1: I appears first, B only when two entities of same type are adjacent
    IOB1,
    /// IOB2: B always starts an entity (most common, seqeval default)
    #[default]
    IOB2,
    /// IOE1: E appears last, I continues  
    IOE1,
    /// IOE2: E always ends an entity
    IOE2,
    /// IOBES/BILOU: Begin-Inside-Last-Outside-Unit (single-token entities use U/S)
    IOBES,
}

impl fmt::Display for BioScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BioScheme::IOB1 => write!(f, "IOB1"),
            BioScheme::IOB2 => write!(f, "IOB2"),
            BioScheme::IOE1 => write!(f, "IOE1"),
            BioScheme::IOE2 => write!(f, "IOE2"),
            BioScheme::IOBES => write!(f, "IOBES/BILOU"),
        }
    }
}

/// A parsed BIO tag.
#[derive(Debug, Clone)]
struct ParsedTag {
    prefix: char,
    entity_type: Option<String>,
}

impl ParsedTag {
    fn parse(tag: &str) -> Self {
        if tag == "O" || tag == "o" {
            return Self { prefix: 'O', entity_type: None };
        }
        
        // Handle B-PER, I-LOC, etc.
        if tag.len() >= 2 && (tag.chars().nth(1) == Some('-') || tag.chars().nth(1) == Some('_')) {
            let prefix = tag.chars().next().unwrap_or('O').to_ascii_uppercase();
            let entity_type = tag[2..].to_string();
            return Self { prefix, entity_type: Some(entity_type) };
        }
        
        // Fallback: treat as O
        Self { prefix: 'O', entity_type: None }
    }
    
    fn is_outside(&self) -> bool {
        self.prefix == 'O'
    }
    
    fn is_begin(&self) -> bool {
        self.prefix == 'B'
    }
    
    fn is_inside(&self) -> bool {
        self.prefix == 'I'
    }
    
    fn is_end(&self) -> bool {
        self.prefix == 'E' || self.prefix == 'L'
    }
    
    fn is_single(&self) -> bool {
        self.prefix == 'S' || self.prefix == 'U'
    }
}

/// Convert BIO-tagged tokens to entity spans.
///
/// # Arguments
///
/// * `tokens` - Slice of token strings
/// * `tags` - Slice of BIO tags (same length as tokens)
/// * `scheme` - BIO tagging scheme to use
///
/// # Returns
///
/// Vector of Entity spans with character offsets computed from tokens.
///
/// # Example
///
/// ```rust
/// use anno::eval::bio_adapter::{BioScheme, bio_to_entities};
///
/// let tokens = ["The", "United", "Nations", "met", "today"];
/// let tags = ["O", "B-ORG", "I-ORG", "O", "O"];
///
/// let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
/// assert_eq!(entities.len(), 1);
/// assert_eq!(entities[0].text, "United Nations");
/// ```
pub fn bio_to_entities<S: AsRef<str>>(
    tokens: &[S],
    tags: &[S],
    scheme: BioScheme,
) -> Result<Vec<Entity>> {
    if tokens.len() != tags.len() {
        return Err(crate::Error::invalid_input(format!(
            "Token count ({}) != tag count ({})",
            tokens.len(),
            tags.len()
        )));
    }
    
    // Compute character offsets for each token
    let mut offsets = Vec::with_capacity(tokens.len());
    let mut current_offset = 0;
    for token in tokens {
        let token_str = token.as_ref();
        offsets.push((current_offset, current_offset + token_str.len()));
        current_offset += token_str.len() + 1; // +1 for space
    }
    
    let mut entities = Vec::new();
    let mut current_entity: Option<(usize, String)> = None; // (start_idx, type)
    
    for (i, tag_str) in tags.iter().enumerate() {
        let tag = ParsedTag::parse(tag_str.as_ref());
        
        match scheme {
            BioScheme::IOB2 => {
                if tag.is_begin() || tag.is_single() {
                    // Finish previous entity if any
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                    // Start new entity
                    current_entity = tag.entity_type.clone().map(|t| (i, t));
                    
                    // Single-token entity in IOBES mode
                    if tag.is_single() {
                        if let Some((start_idx, entity_type)) = current_entity.take() {
                            entities.push(build_entity(tokens, &offsets, start_idx, i, &entity_type));
                        }
                    }
                } else if tag.is_inside() {
                    // Continue entity if types match
                    if let Some((_, ref current_type)) = current_entity {
                        if tag.entity_type.as_ref() != Some(current_type) {
                            // Type mismatch - close current and start new
                            if let Some((start_idx, entity_type)) = current_entity.take() {
                                entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                            }
                            current_entity = tag.entity_type.clone().map(|t| (i, t));
                        }
                    } else {
                        // I without B - start new entity (lenient)
                        current_entity = tag.entity_type.clone().map(|t| (i, t));
                    }
                } else if tag.is_end() {
                    // Close entity
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i, &entity_type));
                    }
                } else if tag.is_outside() {
                    // Close any open entity
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                }
            }
            BioScheme::IOB1 => {
                // IOB1: B only appears between adjacent same-type entities
                if tag.is_begin() {
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                    current_entity = tag.entity_type.clone().map(|t| (i, t));
                } else if tag.is_inside() {
                    if current_entity.is_none() || 
                       current_entity.as_ref().map(|(_, t)| t) != tag.entity_type.as_ref() {
                        // New entity starts with I in IOB1
                        if let Some((start_idx, entity_type)) = current_entity.take() {
                            entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                        }
                        current_entity = tag.entity_type.clone().map(|t| (i, t));
                    }
                } else if tag.is_outside() {
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                }
            }
            BioScheme::IOBES => {
                if tag.is_begin() {
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                    current_entity = tag.entity_type.clone().map(|t| (i, t));
                } else if tag.is_inside() {
                    // Continue
                } else if tag.is_end() {
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i, &entity_type));
                    }
                } else if tag.is_single() {
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                    if let Some(t) = tag.entity_type.clone() {
                        entities.push(build_entity(tokens, &offsets, i, i, &t));
                    }
                } else if tag.is_outside() {
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                }
            }
            // IOE1/IOE2 similar logic but ending-focused
            BioScheme::IOE1 | BioScheme::IOE2 => {
                if tag.is_inside() || tag.is_begin() {
                    if current_entity.is_none() {
                        current_entity = tag.entity_type.clone().map(|t| (i, t));
                    }
                } else if tag.is_end() {
                    if current_entity.is_none() {
                        current_entity = tag.entity_type.clone().map(|t| (i, t));
                    }
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i, &entity_type));
                    }
                } else if tag.is_outside() {
                    if let Some((start_idx, entity_type)) = current_entity.take() {
                        entities.push(build_entity(tokens, &offsets, start_idx, i - 1, &entity_type));
                    }
                }
            }
        }
    }
    
    // Close any remaining entity
    if let Some((start_idx, entity_type)) = current_entity {
        entities.push(build_entity(tokens, &offsets, start_idx, tokens.len() - 1, &entity_type));
    }
    
    Ok(entities)
}

/// Build an Entity from token range.
fn build_entity<S: AsRef<str>>(
    tokens: &[S],
    offsets: &[(usize, usize)],
    start_idx: usize,
    end_idx: usize,
    entity_type: &str,
) -> Entity {
    let text: String = tokens[start_idx..=end_idx]
        .iter()
        .map(|t| t.as_ref())
        .collect::<Vec<_>>()
        .join(" ");
    
    let char_start = offsets[start_idx].0;
    let char_end = offsets[end_idx].1;
    
    Entity::new(
        &text,
        string_to_entity_type(entity_type),
        char_start,
        char_end,
        1.0,
    )
}

/// Convert entity type string to EntityType.
fn string_to_entity_type(s: &str) -> EntityType {
    match s.to_uppercase().as_str() {
        "PER" | "PERSON" => EntityType::Person,
        "ORG" | "ORGANIZATION" => EntityType::Organization,
        "LOC" | "LOCATION" | "GPE" => EntityType::Location,
        "MISC" | "MISCELLANEOUS" => EntityType::custom("MISC", crate::EntityCategory::Misc),
        "DATE" => EntityType::Date,
        "TIME" => EntityType::Time,
        "MONEY" | "CURRENCY" => EntityType::Money,
        "PERCENT" | "PERCENTAGE" => EntityType::Percent,
        other => EntityType::custom(other, crate::EntityCategory::Misc),
    }
}

/// Convert entities back to BIO tags.
///
/// # Arguments
///
/// * `text` - The original text
/// * `tokens` - Token boundaries as (start, end) character offsets
/// * `entities` - Entities to convert
/// * `scheme` - BIO scheme to use
///
/// # Returns
///
/// Vector of BIO tags, one per token.
pub fn entities_to_bio(
    tokens: &[(usize, usize)],
    entities: &[Entity],
    scheme: BioScheme,
) -> Vec<String> {
    let mut tags = vec!["O".to_string(); tokens.len()];
    
    for entity in entities {
        let type_label = entity.entity_type.as_label().to_uppercase();
        
        // Find tokens that overlap with this entity
        let mut entity_tokens: Vec<usize> = Vec::new();
        for (i, &(tok_start, tok_end)) in tokens.iter().enumerate() {
            if tok_start < entity.end && tok_end > entity.start {
                entity_tokens.push(i);
            }
        }
        
        if entity_tokens.is_empty() {
            continue;
        }
        
        match scheme {
            BioScheme::IOB2 => {
                for (j, &tok_idx) in entity_tokens.iter().enumerate() {
                    tags[tok_idx] = if j == 0 {
                        format!("B-{}", type_label)
                    } else {
                        format!("I-{}", type_label)
                    };
                }
            }
            BioScheme::IOB1 => {
                // B only if previous token was same type
                for (j, &tok_idx) in entity_tokens.iter().enumerate() {
                    let needs_b = j == 0 && tok_idx > 0 && 
                        tags[tok_idx - 1].ends_with(&format!("-{}", type_label));
                    tags[tok_idx] = if needs_b {
                        format!("B-{}", type_label)
                    } else {
                        format!("I-{}", type_label)
                    };
                }
            }
            BioScheme::IOBES => {
                let len = entity_tokens.len();
                for (j, &tok_idx) in entity_tokens.iter().enumerate() {
                    tags[tok_idx] = if len == 1 {
                        format!("S-{}", type_label)
                    } else if j == 0 {
                        format!("B-{}", type_label)
                    } else if j == len - 1 {
                        format!("E-{}", type_label)
                    } else {
                        format!("I-{}", type_label)
                    };
                }
            }
            BioScheme::IOE2 => {
                let len = entity_tokens.len();
                for (j, &tok_idx) in entity_tokens.iter().enumerate() {
                    tags[tok_idx] = if j == len - 1 {
                        format!("E-{}", type_label)
                    } else {
                        format!("I-{}", type_label)
                    };
                }
            }
            BioScheme::IOE1 => {
                let len = entity_tokens.len();
                for (j, &tok_idx) in entity_tokens.iter().enumerate() {
                    // E only if next token is same type
                    let needs_e = j == len - 1 && tok_idx + 1 < tokens.len() &&
                        tags.get(tok_idx + 1).map(|t| t.ends_with(&format!("-{}", type_label))).unwrap_or(false);
                    tags[tok_idx] = if needs_e {
                        format!("E-{}", type_label)
                    } else {
                        format!("I-{}", type_label)
                    };
                }
            }
        }
    }
    
    tags
}

/// Validate BIO tag sequence for a given scheme.
///
/// Returns errors for invalid transitions (e.g., O -> I in strict IOB2).
/// 
/// Invalid transitions are a common issue in NER model outputs, particularly
/// when not using CRF layers for constraint enforcement during training.
pub fn validate_bio_sequence<S: AsRef<str>>(tags: &[S], scheme: BioScheme) -> Vec<String> {
    let mut errors = Vec::new();
    let mut prev_tag = ParsedTag { prefix: 'O', entity_type: None };
    
    for (i, tag_str) in tags.iter().enumerate() {
        let tag = ParsedTag::parse(tag_str.as_ref());
        
        match scheme {
            BioScheme::IOB2 => {
                // I must follow B or I of same type
                if tag.is_inside() {
                    if prev_tag.is_outside() {
                        errors.push(format!("Position {}: I-{} follows O (should be B-{})", 
                            i, 
                            tag.entity_type.as_deref().unwrap_or("?"),
                            tag.entity_type.as_deref().unwrap_or("?")));
                    } else if tag.entity_type != prev_tag.entity_type {
                        errors.push(format!("Position {}: I-{} follows {}-{} (type mismatch)",
                            i,
                            tag.entity_type.as_deref().unwrap_or("?"),
                            prev_tag.prefix,
                            prev_tag.entity_type.as_deref().unwrap_or("?")));
                    }
                }
            }
            BioScheme::IOBES => {
                // E/L must follow B or I of same type
                if tag.is_end() && !prev_tag.is_begin() && !prev_tag.is_inside() {
                    errors.push(format!("Position {}: E-{} without preceding B or I", 
                        i,
                        tag.entity_type.as_deref().unwrap_or("?")));
                }
                // I must follow B or I
                if tag.is_inside() && !prev_tag.is_begin() && !prev_tag.is_inside() {
                    errors.push(format!("Position {}: I-{} without preceding B or I",
                        i,
                        tag.entity_type.as_deref().unwrap_or("?")));
                }
            }
            _ => {} // IOB1, IOE1, IOE2 are more lenient
        }
        
        prev_tag = tag;
    }
    
    errors
}

/// Repair strategy for invalid BIO sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RepairStrategy {
    /// Convert invalid I tags to B tags (most common approach).
    #[default]
    PromoteToBegin,
    /// Discard invalid transitions by converting to O.
    Discard,
    /// Keep invalid tags as-is (lenient parsing).
    Lenient,
}

/// Repair invalid BIO tag sequences.
/// 
/// Production NER systems often produce invalid sequences (e.g., O followed by I).
/// This function repairs such sequences according to the chosen strategy.
/// 
/// # Strategies
/// 
/// - `PromoteToBegin`: Convert orphan I tags to B tags (recommended)
/// - `Discard`: Convert invalid tags to O
/// - `Lenient`: Keep as-is (caller handles in parsing)
/// 
/// # Example
/// 
/// ```rust
/// use anno::eval::bio_adapter::{repair_bio_sequence, RepairStrategy, BioScheme};
/// 
/// let invalid = vec!["O", "I-PER", "I-PER", "O"];  // Invalid: O->I
/// let repaired = repair_bio_sequence(&invalid, BioScheme::IOB2, RepairStrategy::PromoteToBegin);
/// assert_eq!(repaired, vec!["O", "B-PER", "I-PER", "O"]);  // Fixed
/// ```
pub fn repair_bio_sequence<S: AsRef<str>>(
    tags: &[S],
    scheme: BioScheme,
    strategy: RepairStrategy,
) -> Vec<String> {
    if strategy == RepairStrategy::Lenient {
        return tags.iter().map(|t| t.as_ref().to_string()).collect();
    }
    
    let mut result: Vec<String> = Vec::with_capacity(tags.len());
    let mut prev_tag = ParsedTag { prefix: 'O', entity_type: None };
    
    for tag_str in tags {
        let tag = ParsedTag::parse(tag_str.as_ref());
        let mut repaired = tag_str.as_ref().to_string();
        
        match scheme {
            BioScheme::IOB2 => {
                if tag.is_inside() {
                    let needs_repair = prev_tag.is_outside() || 
                        tag.entity_type != prev_tag.entity_type;
                    
                    if needs_repair {
                        match strategy {
                            RepairStrategy::PromoteToBegin => {
                                if let Some(ref t) = tag.entity_type {
                                    repaired = format!("B-{}", t);
                                }
                            }
                            RepairStrategy::Discard => {
                                repaired = "O".to_string();
                            }
                            RepairStrategy::Lenient => {}
                        }
                    }
                }
            }
            BioScheme::IOBES => {
                if (tag.is_inside() || tag.is_end()) && !prev_tag.is_begin() && !prev_tag.is_inside() {
                    match strategy {
                        RepairStrategy::PromoteToBegin => {
                            if let Some(ref t) = tag.entity_type {
                                // If single invalid I or E, make it S (single)
                                repaired = format!("S-{}", t);
                            }
                        }
                        RepairStrategy::Discard => {
                            repaired = "O".to_string();
                        }
                        RepairStrategy::Lenient => {}
                    }
                }
            }
            _ => {} // Other schemes more lenient
        }
        
        prev_tag = ParsedTag::parse(&repaired);
        result.push(repaired);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iob2_basic() {
        let tokens = ["John", "Smith", "works", "at", "Apple"];
        let tags = ["B-PER", "I-PER", "O", "O", "B-ORG"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "John Smith");
        assert_eq!(entities[0].entity_type, EntityType::Person);
        assert_eq!(entities[1].text, "Apple");
        assert_eq!(entities[1].entity_type, EntityType::Organization);
    }
    
    #[test]
    fn test_iob2_adjacent_same_type() {
        let tokens = ["John", "and", "Mary"];
        let tags = ["B-PER", "O", "B-PER"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "John");
        assert_eq!(entities[1].text, "Mary");
    }
    
    #[test]
    fn test_iob2_multi_token_org() {
        let tokens = ["The", "United", "Nations", "Security", "Council"];
        let tags = ["O", "B-ORG", "I-ORG", "I-ORG", "I-ORG"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "United Nations Security Council");
        assert_eq!(entities[0].entity_type, EntityType::Organization);
    }
    
    #[test]
    fn test_iobes_single_token() {
        let tokens = ["John", "works", "here"];
        let tags = ["S-PER", "O", "O"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOBES).unwrap();
        
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "John");
    }
    
    #[test]
    fn test_iobes_bie_sequence() {
        let tokens = ["New", "York", "City"];
        let tags = ["B-LOC", "I-LOC", "E-LOC"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOBES).unwrap();
        
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "New York City");
    }
    
    #[test]
    fn test_validation_iob2() {
        // Invalid: O -> I
        let tags = ["O", "I-PER", "I-PER"];
        let errors = validate_bio_sequence(&tags, BioScheme::IOB2);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("follows O"));
        
        // Valid: B -> I
        let tags = ["B-PER", "I-PER", "O"];
        let errors = validate_bio_sequence(&tags, BioScheme::IOB2);
        assert!(errors.is_empty());
    }
    
    #[test]
    fn test_validation_type_mismatch() {
        // Invalid: I-LOC after B-PER
        let tags = ["B-PER", "I-LOC"];
        let errors = validate_bio_sequence(&tags, BioScheme::IOB2);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("type mismatch"));
    }
    
    #[test]
    fn test_repair_promote_to_begin() {
        let invalid = vec!["O", "I-PER", "I-PER", "O"];
        let repaired = repair_bio_sequence(&invalid, BioScheme::IOB2, RepairStrategy::PromoteToBegin);
        assert_eq!(repaired, vec!["O", "B-PER", "I-PER", "O"]);
    }
    
    #[test]
    fn test_repair_discard() {
        let invalid = vec!["O", "I-PER", "I-PER", "O"];
        let repaired = repair_bio_sequence(&invalid, BioScheme::IOB2, RepairStrategy::Discard);
        // First I-PER becomes O (orphan), second I-PER also becomes O (no valid predecessor)
        assert_eq!(repaired, vec!["O", "O", "O", "O"]);
    }
    
    #[test]
    fn test_repair_lenient() {
        let invalid = vec!["O", "I-PER", "I-PER", "O"];
        let repaired = repair_bio_sequence(&invalid, BioScheme::IOB2, RepairStrategy::Lenient);
        assert_eq!(repaired, vec!["O", "I-PER", "I-PER", "O"]);
    }
    
    #[test]
    fn test_repair_type_change() {
        // B-PER followed by I-LOC - type mismatch
        let invalid = vec!["B-PER", "I-LOC", "O"];
        let repaired = repair_bio_sequence(&invalid, BioScheme::IOB2, RepairStrategy::PromoteToBegin);
        assert_eq!(repaired, vec!["B-PER", "B-LOC", "O"]);
    }
    
    #[test]
    fn test_roundtrip() {
        let tokens = ["The", "United", "Nations", "met", "in", "New", "York"];
        let tags = ["O", "B-ORG", "I-ORG", "O", "O", "B-LOC", "I-LOC"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        
        // Create token offsets for roundtrip
        let mut offsets = Vec::new();
        let mut pos = 0;
        for t in &tokens {
            offsets.push((pos, pos + t.len()));
            pos += t.len() + 1;
        }
        
        let recovered_tags = entities_to_bio(&offsets, &entities, BioScheme::IOB2);
        
        assert_eq!(recovered_tags, tags);
    }
    
    #[test]
    fn test_empty_input() {
        let tokens: [&str; 0] = [];
        let tags: [&str; 0] = [];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        assert!(entities.is_empty());
    }
    
    #[test]
    fn test_all_outside() {
        let tokens = ["The", "cat", "sat"];
        let tags = ["O", "O", "O"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        assert!(entities.is_empty());
    }
    
    #[test]
    fn test_mismatched_lengths() {
        let tokens = ["John", "Smith"];
        let tags = ["B-PER"];
        
        let result = bio_to_entities(&tokens, &tags, BioScheme::IOB2);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_character_offsets() {
        let tokens = ["John", "Smith"];
        let tags = ["B-PER", "I-PER"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].start, 0);
        // "John" (4) + space (1) + "Smith" (5) = 10, but text is "John Smith"
        // start of "John" = 0, end of "Smith" = 4 + 1 + 5 = 10
        assert_eq!(entities[0].end, 10);
    }
    
    #[test]
    fn test_iob1_scheme() {
        // In IOB1, B is only used when two same-type entities are adjacent
        let tokens = ["John", "Mary", "works"];
        // Both start with I in IOB1 (no adjacency issue)
        let tags = ["I-PER", "I-PER", "O"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB1).unwrap();
        
        // In IOB1 with same types, I-I continues the entity
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "John Mary");
    }
    
    #[test]
    fn test_custom_entity_types() {
        let tokens = ["CRISPR", "is", "a", "technology"];
        let tags = ["B-TECH", "O", "O", "O"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOB2).unwrap();
        
        assert_eq!(entities.len(), 1);
        assert!(matches!(entities[0].entity_type, EntityType::Custom { .. }));
    }
    
    // =============================================================================
    // IOE Scheme Tests
    // =============================================================================
    
    #[test]
    fn test_ioe2_basic() {
        // IOE2: E always ends an entity
        let tokens = ["New", "York", "City"];
        let tags = ["I-LOC", "I-LOC", "E-LOC"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOE2).unwrap();
        
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "New York City");
        assert_eq!(entities[0].entity_type, EntityType::Location);
    }
    
    #[test]
    fn test_ioe2_multiple_entities() {
        let tokens = ["John", "works", "at", "Apple", "Inc"];
        let tags = ["E-PER", "O", "O", "I-ORG", "E-ORG"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOE2).unwrap();
        
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "John");
        assert_eq!(entities[1].text, "Apple Inc");
    }
    
    #[test]
    fn test_ioe1_basic() {
        // IOE1: E only appears when needed (similar to IOB1)
        let tokens = ["New", "York"];
        let tags = ["I-LOC", "I-LOC"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOE1).unwrap();
        
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "New York");
    }
    
    #[test]
    fn test_entities_to_bio_ioe2() {
        let _tokens = ["The", "Big", "Apple"];
        let entities = vec![Entity::new("Big Apple", EntityType::Location, 4, 14, 0.9)];
        
        // Create token offsets: "The" (0-3), "Big" (4-7), "Apple" (8-13)
        let offsets = vec![(0, 3), (4, 7), (8, 13)];
        
        let tags = entities_to_bio(&offsets, &entities, BioScheme::IOE2);
        
        assert_eq!(tags[0], "O");
        // EntityType::Location.as_label() returns "LOC"
        assert_eq!(tags[1], "I-LOC");
        assert_eq!(tags[2], "E-LOC");
    }
    
    // =============================================================================
    // Repair for Different Schemes
    // =============================================================================
    
    #[test]
    fn test_repair_iobes_orphan_inside() {
        let invalid = vec!["O", "I-PER", "O"];
        let repaired = repair_bio_sequence(&invalid, BioScheme::IOBES, RepairStrategy::PromoteToBegin);
        // Orphan I should become S (single) in IOBES
        assert_eq!(repaired, vec!["O", "S-PER", "O"]);
    }
    
    #[test]
    fn test_repair_iobes_orphan_end() {
        let invalid = vec!["O", "E-PER", "O"];
        let repaired = repair_bio_sequence(&invalid, BioScheme::IOBES, RepairStrategy::PromoteToBegin);
        // Orphan E should become S (single) in IOBES
        assert_eq!(repaired, vec!["O", "S-PER", "O"]);
    }
    
    // =============================================================================
    // Roundtrip Tests for All Schemes
    // =============================================================================
    
    #[test]
    fn test_roundtrip_iobes() {
        let tokens = ["The", "United", "Nations"];
        let tags = ["O", "B-ORG", "E-ORG"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOBES).unwrap();
        
        let mut offsets = Vec::new();
        let mut pos = 0;
        for t in &tokens {
            offsets.push((pos, pos + t.len()));
            pos += t.len() + 1;
        }
        
        let recovered = entities_to_bio(&offsets, &entities, BioScheme::IOBES);
        assert_eq!(recovered, tags);
    }
    
    #[test]
    fn test_roundtrip_ioe2() {
        let tokens = ["Visit", "New", "York"];
        let tags = ["O", "I-LOC", "E-LOC"];
        
        let entities = bio_to_entities(&tokens, &tags, BioScheme::IOE2).unwrap();
        
        let mut offsets = Vec::new();
        let mut pos = 0;
        for t in &tokens {
            offsets.push((pos, pos + t.len()));
            pos += t.len() + 1;
        }
        
        let recovered = entities_to_bio(&offsets, &entities, BioScheme::IOE2);
        assert_eq!(recovered, tags);
    }
}

