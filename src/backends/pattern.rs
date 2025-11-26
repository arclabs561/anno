//! Pattern-based NER - Extracts entities via regex patterns only.
//!
//! No hardcoded gazetteers. Only extracts entities that can be reliably
//! identified by their format:
//! - Dates: ISO 8601, MM/DD/YYYY, "January 15, 2024"
//! - Money: $100, $1.5M, "50 dollars"
//! - Percentages: 15%, 3.5%
//!
//! For Person/Organization/Location, use ML models (BERT ONNX, GLiNER).

use crate::{Entity, EntityType, Model, Result};
use once_cell::sync::Lazy;
use regex::Regex;

/// Pattern-based NER - only extracts entities with recognizable formats.
///
/// This is a minimal fallback when ML models are unavailable.
/// It does NOT attempt to identify Person/Org/Location without ML.
pub struct PatternNER;

impl PatternNER {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PatternNER {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for PatternNER {
    fn extract_entities(&self, text: &str, _language: Option<&str>) -> Result<Vec<Entity>> {
        let mut entities = Vec::new();

        // Date patterns
        static DATE_ISO: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").unwrap()
        });
        static DATE_US: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b\d{1,2}/\d{1,2}/\d{4}\b").unwrap()
        });
        static DATE_WRITTEN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b").unwrap()
        });
        static DATE_WRITTEN_EU: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+\d{4})?\b").unwrap()
        });

        for pattern in [&*DATE_ISO, &*DATE_US, &*DATE_WRITTEN, &*DATE_WRITTEN_EU] {
            for m in pattern.find_iter(text) {
                if !overlaps(&entities, m.start(), m.end()) {
                    entities.push(Entity {
                        text: m.as_str().to_string(),
                        entity_type: EntityType::Date,
                        start: m.start(),
                        end: m.end(),
                        confidence: 0.9,
                    });
                }
            }
        }

        // Money patterns
        static MONEY_DOLLAR: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\$[\d,]+\.?\d*(?:\s*(?:billion|million|thousand|B|M|K))?").unwrap()
        });
        static MONEY_WRITTEN: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b\d+(?:\.\d+)?\s*(?:dollars?|USD|EUR|GBP|billion|million)\b").unwrap()
        });

        for pattern in [&*MONEY_DOLLAR, &*MONEY_WRITTEN] {
            for m in pattern.find_iter(text) {
                if !overlaps(&entities, m.start(), m.end()) {
                    entities.push(Entity {
                        text: m.as_str().to_string(),
                        entity_type: EntityType::Money,
                        start: m.start(),
                        end: m.end(),
                        confidence: 0.9,
                    });
                }
            }
        }

        // Percentage patterns
        static PERCENT: Lazy<Regex> = Lazy::new(|| {
            Regex::new(r"\b\d+\.?\d*\s*%").unwrap()
        });

        for m in PERCENT.find_iter(text) {
            if !overlaps(&entities, m.start(), m.end()) {
                entities.push(Entity {
                    text: m.as_str().to_string(),
                    entity_type: EntityType::Percent,
                    start: m.start(),
                    end: m.end(),
                    confidence: 0.9,
                });
            }
        }

        Ok(entities)
    }

    fn supported_types(&self) -> Vec<EntityType> {
        vec![EntityType::Date, EntityType::Money, EntityType::Percent]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "pattern"
    }

    fn description(&self) -> &'static str {
        "Pattern-based NER (dates, money, percentages only)"
    }
}

/// Check if a span overlaps with existing entities.
fn overlaps(entities: &[Entity], start: usize, end: usize) -> bool {
    entities.iter().any(|e| !(end <= e.start || start >= e.end))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_extraction() {
        let ner = PatternNER::new();
        let entities = ner.extract_entities("Meeting on 2024-01-15 and January 20, 2024.", None).unwrap();
        
        let dates: Vec<_> = entities.iter().filter(|e| e.entity_type == EntityType::Date).collect();
        assert_eq!(dates.len(), 2);
        assert!(dates.iter().any(|e| e.text == "2024-01-15"));
        assert!(dates.iter().any(|e| e.text == "January 20, 2024"));
    }

    #[test]
    fn test_money_extraction() {
        let ner = PatternNER::new();
        let entities = ner.extract_entities("Cost is $100.50 or 50 dollars.", None).unwrap();
        
        let money: Vec<_> = entities.iter().filter(|e| e.entity_type == EntityType::Money).collect();
        assert_eq!(money.len(), 2);
    }

    #[test]
    fn test_percent_extraction() {
        let ner = PatternNER::new();
        let entities = ner.extract_entities("Improved by 15% and 3.5%.", None).unwrap();
        
        let percents: Vec<_> = entities.iter().filter(|e| e.entity_type == EntityType::Percent).collect();
        assert_eq!(percents.len(), 2);
    }

    #[test]
    fn test_no_person_org_loc() {
        let ner = PatternNER::new();
        // Pattern NER should NOT extract Person/Org/Location
        let entities = ner.extract_entities("John Smith works at Google in New York.", None).unwrap();
        
        // Only pattern-based types
        assert!(entities.iter().all(|e| matches!(
            e.entity_type, 
            EntityType::Date | EntityType::Money | EntityType::Percent
        )));
    }
}

