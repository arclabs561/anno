//! Pair scoring for mention-ranking coreference.
//!
//! Contains `score_pair` and the salience lookup helper used during
//! antecedent ranking.

use super::features::is_type_incompatible;
use super::*;

impl MentionRankingCoref {
    /// Get salience score for an entity (returns 0.0 if not found).
    pub(super) fn get_salience(&self, text: &str) -> f64 {
        self.salience_scores
            .as_ref()
            .and_then(|s| s.get(&text.to_lowercase()).copied())
            .unwrap_or(0.0)
    }

    /// Score a (mention, antecedent) pair.
    ///
    /// # Arguments
    ///
    /// * `mention` - The anaphor being resolved
    /// * `antecedent` - Candidate antecedent
    /// * `distance` - Character distance between mentions
    /// * `text` - Optional source text for context-aware features
    pub(super) fn score_pair(
        &self,
        mention: &RankedMention,
        antecedent: &RankedMention,
        distance: usize,
        text: Option<&str>,
    ) -> f64 {
        let mut score = 0.0;

        // =========================================================================
        // i2b2-inspired context filtering (Chen et al. 2011)
        // Check this first - if context filtering rejects the pair, return low score
        // =========================================================================
        if self.config.enable_context_filtering {
            if let Some(txt) = text {
                if self.should_filter_by_context(txt, mention, antecedent) {
                    return -1.0; // Strong negative signal to reject this pair
                }
            }
        }

        // =========================================================================
        // Semantic type incompatibility filter
        // Prevents merging e.g. "Nobel Prize" with "Marie Curie"
        // =========================================================================
        if is_type_incompatible(&mention.text, &antecedent.text) {
            return -1.0;
        }

        // =========================================================================
        // String match features
        // =========================================================================
        let m_lower = mention.text.to_lowercase();
        let a_lower = antecedent.text.to_lowercase();

        // Exact match
        if m_lower == a_lower {
            score += self.config.string_match_weight * 1.0;
        }
        // Head match
        else if mention.head.to_lowercase() == antecedent.head.to_lowercase() {
            score += self.config.string_match_weight * 0.6;
        }
        // Substring (partial match -- weight low enough that substring alone
        // cannot cross the default link_threshold of 0.45)
        else if m_lower.contains(&a_lower) || a_lower.contains(&m_lower) {
            score += self.config.string_match_weight * 0.15;
        }

        // =========================================================================
        // i2b2-inspired "be phrase" detection (Chen et al. 2011)
        // "Resolution of X is Y" → X and Y are coreferent
        // =========================================================================
        if self.config.enable_be_phrase_detection {
            if let Some(txt) = text {
                if self.is_be_phrase_link(txt, mention, antecedent) {
                    score += self.config.be_phrase_weight;
                }
            }
        }

        // =========================================================================
        // i2b2-inspired acronym matching (Chen et al. 2011)
        // "MRSA" ↔ "Methicillin-resistant Staphylococcus aureus"
        // =========================================================================
        if self.config.enable_acronym_matching && self.is_acronym_match(mention, antecedent) {
            score += self.config.acronym_weight;
        }

        // =========================================================================
        // i2b2-inspired synonym matching (Chen et al. 2011)
        // Uses UMLS concept matching in original; we use a basic synonym table
        // =========================================================================
        if self.config.enable_synonym_matching && self.are_synonyms(mention, antecedent) {
            score += self.config.synonym_weight;
        }

        // =========================================================================
        // Type compatibility
        // =========================================================================
        match (mention.mention_type, antecedent.mention_type) {
            (MentionType::Pronominal, MentionType::Proper) => {
                score += self.config.type_compat_weight * 0.5;
            }
            (MentionType::Pronominal, MentionType::Pronominal)
                if mention.text.to_lowercase() == antecedent.text.to_lowercase() =>
            {
                // Same pronoun
                score += self.config.type_compat_weight * 0.3;
            }
            (MentionType::Proper, MentionType::Proper) => {
                score += self.config.type_compat_weight * 0.4;
            }
            _ => {}
        }

        // =========================================================================
        // Gender agreement
        // =========================================================================
        if let (Some(m_gender), Some(a_gender)) = (mention.gender, antecedent.gender) {
            if m_gender == a_gender {
                score += self.config.type_compat_weight * 0.3;
            } else if m_gender != Gender::Unknown && a_gender != Gender::Unknown {
                score -= self.config.type_compat_weight * 0.5; // Penalty for mismatch
            }
        }

        // =========================================================================
        // Number agreement
        //
        // Uses Number::is_compatible() from `crate::core` which handles:
        // - Unknown is compatible with anything (singular they, "you")
        // - Dual is compatible with Plural (Arabic/Hebrew/Sanskrit dual numbers)
        // - Exact matches are preferred
        // =========================================================================
        if let (Some(m_number), Some(a_number)) = (mention.number, antecedent.number) {
            if m_number == a_number {
                // Exact match: strongest bonus
                score += self.config.type_compat_weight * 0.2;
            } else if m_number.is_compatible(&a_number) {
                // Compatible but not exact (e.g., Unknown with Singular, Dual with Plural)
                // Small bonus - compatible but less certain
                score += self.config.type_compat_weight * 0.05;
            } else {
                // Incompatible numbers (e.g., Singular vs Plural)
                score -= self.config.type_compat_weight * 0.4;
            }
        }

        // =========================================================================
        // Animacy agreement (phi-feature: animate vs inanimate)
        //
        // Hard constraint: animate/inanimate mismatch blocks coreference
        // ("John... it" is ungrammatical). Unknown is wildcard.
        // See Jurafsky & Martin SLP3 Ch. 26, Fig. 26.4.
        // =========================================================================
        match (mention.animacy, antecedent.animacy) {
            (Animacy::Animate, Animacy::Inanimate) | (Animacy::Inanimate, Animacy::Animate) => {
                score -= self.config.type_compat_weight * 0.6;
            }
            (Animacy::Animate, Animacy::Animate) | (Animacy::Inanimate, Animacy::Inanimate) => {
                score += self.config.type_compat_weight * 0.1;
            }
            _ => {} // Unknown is wildcard -- no bonus or penalty
        }

        // =========================================================================
        // Distance penalty
        // =========================================================================
        score -= self.config.distance_weight * (distance as f64).ln().max(0.0);

        // =========================================================================
        // Pronoun-specific: recency boost + entity-type preference
        //
        // For pronoun mentions, favor the most recent compatible antecedent.
        // Recency is the primary signal for pronoun resolution (Hobbs 1978).
        // Additionally, apply entity-type preferences:
        //   - he/him/his/she/her/hers -> prefer Person antecedents
        //   - it/its -> prefer Organization (or non-Person) antecedents
        //   - they/them/their -> neutral (no preference)
        // =========================================================================
        if mention.mention_type == MentionType::Pronominal {
            // Recency boost: closer antecedents get a stronger bonus.
            // Uses inverse distance (in characters) to reward proximity.
            // The boost decays as 1/(1 + distance/50), giving ~0.3 for adjacent
            // mentions and tapering off for distant ones.
            let recency_boost =
                self.config.type_compat_weight * 0.3 / (1.0 + distance as f64 / 50.0);
            score += recency_boost;

            // Entity-type preference: when the antecedent has a known entity type
            // from NER, apply a bonus or penalty based on pronoun gender.
            if let Some(ref ant_type) = antecedent.entity_type {
                let m_lower = mention.text.to_lowercase();
                let is_person_pronoun = matches!(
                    m_lower.as_str(),
                    "he" | "him" | "his" | "she" | "her" | "hers" | "himself" | "herself"
                );
                let is_it_pronoun = matches!(m_lower.as_str(), "it" | "its" | "itself");

                if is_person_pronoun {
                    if matches!(ant_type, crate::EntityType::Person) {
                        score += self.config.type_compat_weight * 0.4;
                    } else {
                        score -= self.config.type_compat_weight * 0.3;
                    }
                } else if is_it_pronoun {
                    if matches!(ant_type, crate::EntityType::Organization) {
                        score += self.config.type_compat_weight * 0.3;
                    } else if matches!(ant_type, crate::EntityType::Person) {
                        score -= self.config.type_compat_weight * 0.3;
                    }
                }
                // they/them/their: no entity-type preference (neutral)
            }
        }

        // =========================================================================
        // External score (e.g., box containment)
        // =========================================================================
        if let Some(ref ext) = self.config.external_scores {
            let key = (mention.start, antecedent.start);
            if let Some(&ext_score) = ext.get(&key) {
                score += self.config.external_score_weight * ext_score;
            }
        }

        // =========================================================================
        // Salience boost
        // =========================================================================
        if self.config.salience_weight > 0.0 {
            let salience = self.get_salience(&antecedent.text);
            score += self.config.salience_weight * salience;
        }

        score
    }
}
