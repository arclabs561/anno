//! French-aware heuristic NER backend.
//!
#![allow(missing_docs)]
//! Complements GLiNER2 with deterministic patterns that the model is
//! intrinsically weak on: French organisation suffixes (SAS, SARL, …),
//! French address structures, French date formats (with `date_of_birth`
//! context detection), and international IBANs with mod-97 verification.
//!
//! Implements [`crate::backends::inference::ZeroShotNER`] so it composes
//! with `StackedNER` and can be tested through the same harness as every
//! other backend.

pub mod addresses;
pub mod dates;
pub mod iban_intl;
pub mod orgs;

use crate::backends::inference::ZeroShotNER;
use crate::core::entity::{Entity, EntityType};

/// French heuristic NER: deterministic patterns for entities where
/// GLiNER2 is intrinsically weak (org legal forms, addresses, dates,
/// international IBANs).
#[derive(Debug, Default, Clone)]
pub struct HeuristicFrNer;

impl HeuristicFrNer {
    pub fn new() -> Self {
        Self
    }
}

static DEFAULT_TYPES: &[&str] = &["organization", "address", "date", "date_of_birth", "iban"];

impl ZeroShotNER for HeuristicFrNer {
    fn default_types(&self) -> &[&'static str] {
        DEFAULT_TYPES
    }

    fn extract_with_descriptions(
        &self,
        text: &str,
        descriptions: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<Entity>> {
        // Descriptions are ignored; delegate to label-based extraction.
        self.extract_with_types(text, descriptions, threshold)
    }

    fn extract_with_types(
        &self,
        text: &str,
        types: &[&str],
        threshold: f32,
    ) -> crate::Result<Vec<Entity>> {
        let mut out = Vec::new();

        if types.contains(&"organization") {
            out.extend(
                orgs::extract_orgs(text)
                    .into_iter()
                    .filter(|e| f32::from(e.confidence) >= threshold),
            );
        }

        if types.contains(&"address") {
            out.extend(
                addresses::extract_addresses(text)
                    .into_iter()
                    .filter(|e| f32::from(e.confidence) >= threshold),
            );
        }

        if types.contains(&"date") || types.contains(&"date_of_birth") {
            out.extend(
                dates::extract_dates(text)
                    .into_iter()
                    .filter(|e| f32::from(e.confidence) >= threshold)
                    .filter(|e| match &e.entity_type {
                        EntityType::Date => types.contains(&"date"),
                        EntityType::Custom { name, .. } if name == "date_of_birth" => {
                            types.contains(&"date_of_birth")
                        }
                        _ => false,
                    }),
            );
        }

        if types.contains(&"iban") {
            out.extend(
                iban_intl::extract_iban_intl(text)
                    .into_iter()
                    .filter(|e| f32::from(e.confidence) >= threshold),
            );
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_org_when_asked() {
        let n = HeuristicFrNer::new();
        let r = n
            .extract_with_types("Acme Tech SAS est ici.", &["organization"], 0.5)
            .unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn respects_type_filter() {
        let n = HeuristicFrNer::new();
        // asking only for "address" should not return orgs
        let r = n
            .extract_with_types("Acme Tech SAS est ici.", &["address"], 0.5)
            .unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn empty_types_returns_empty() {
        let n = HeuristicFrNer::new();
        let r = n
            .extract_with_types("Acme Tech SAS est ici.", &[], 0.5)
            .unwrap();
        assert!(r.is_empty());
    }
}
