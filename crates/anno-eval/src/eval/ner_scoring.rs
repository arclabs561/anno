//! Dataset scoring views for native NER reports. Authored tokens and gold stay in the loader.

use super::{loader::DatasetId, GoldEntity};
use anno::core::grounded::{Location, Signal, SignalId};
use anno::core::types::TypeLabel;
use anno::EntityType;
use serde::Serialize;

/// Literal source labels, or the documented evaluation policy of the dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NerScoring {
    /// Keep original label spelling, source extents and empty annotations.
    Source,
    /// Use the dataset's documented alignment, if it has one.
    Dataset,
}

/// Reference counts expose every adjustment to the scoring denominator.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ReferenceCounts {
    /// All authored mentions.
    pub source_gold: usize,
    /// References with a location in this evaluation.
    pub scored_gold: usize,
    /// Authored spans with no surface after endpoint adjustment.
    pub excluded_empty_gold: usize,
    /// Nonempty spans whose leading or trailing separator spaces were removed.
    pub trimmed_gold: usize,
}

impl ReferenceCounts {
    /// Explain the author and scored denominators together in a human-readable report.
    pub fn description(self, mode: NerScoring) -> String {
        let mode = match mode {
            NerScoring::Dataset => "dataset",
            NerScoring::Source => "source",
        };
        format!("NER scoring {mode}: {} authored gold, {} scored, {} empty excluded, {} nonempty spans trimmed", self.source_gold, self.scored_gold, self.excluded_empty_gold, self.trimmed_gold)
    }

    /// Pool disjoint observations.
    pub fn accumulate(&mut self, row: Self) {
        self.source_gold += row.source_gold;
        self.scored_gold += row.scored_gold;
        self.excluded_empty_gold += row.excluded_empty_gold;
        self.trimmed_gold += row.trimmed_gold;
    }
}

impl NerScoring {
    /// Create scoring signals while retaining each mention's source ordinal.
    pub fn reference(
        self,
        dataset: Option<DatasetId>,
        gold: &[GoldEntity],
    ) -> (Vec<Signal<Location>>, ReferenceCounts) {
        let histnero = self == Self::Dataset && dataset == Some(DatasetId::HistNERo);
        let mut counts = ReferenceCounts {
            source_gold: gold.len(),
            ..Default::default()
        };
        let mut signals = Vec::with_capacity(gold.len());
        for (i, g) in gold.iter().enumerate() {
            let (mut start, mut end, mut surface) = (g.start, g.end, g.text.as_str());
            let label = if histnero {
                // Source tokens are joined by one ASCII space. In this release only
                // empty token strings introduce ASCII space at the edge of a mention.
                let visible = surface.trim_matches(' ');
                if visible.is_empty() {
                    counts.excluded_empty_gold += 1;
                    continue;
                }
                if visible.len() != g.text.len()
                    && end.checked_sub(start) == Some(g.text.chars().count())
                {
                    // These differences count ASCII separators even for Unicode names.
                    start += g.text.len() - g.text.trim_start_matches(' ').len();
                    end -= g.text.len() - g.text.trim_end_matches(' ').len();
                    surface = visible;
                    counts.trimmed_gold += 1;
                }
                match g.original_label.as_str() {
                    "PERS" => TypeLabel::from(EntityType::Person),
                    "PROD" => TypeLabel::from("PRODUCT"),
                    other => TypeLabel::from(other),
                }
            } else if self == Self::Source {
                TypeLabel::Custom(g.original_label.clone())
            } else {
                // Preserve the existing native per-case report convention for all
                // datasets without their own alignment policy.
                TypeLabel::from(g.original_label.as_str())
            };
            signals.push(Signal::new(
                SignalId::new(i as u64),
                Location::text(start, end),
                surface,
                label,
                1.0,
            ));
        }
        counts.scored_gold = signals.len();
        (signals, counts)
    }

    /// Compare literal strings on both sides when the source mode is requested.
    pub fn predictions(self, signals: &mut [Signal<Location>]) {
        if self == Self::Source {
            for signal in signals {
                signal.label = TypeLabel::Custom(signal.label().to_owned());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::core::grounded::EvalComparison;

    #[test]
    fn histnero_exposes_literal_and_documented_views_of_the_same_reference() {
        let gold = [
            GoldEntity::full("Șerban ", EntityType::from_label("PERS"), "PERS", 0, 7),
            GoldEntity::full("", EntityType::from_label("PROD"), "PROD", 8, 8),
            GoldEntity::full("I  II", EntityType::from_label("PROD"), "PROD", 9, 14),
        ];
        let (aligned, audit) = NerScoring::Dataset.reference(Some(DatasetId::HistNERo), &gold);
        assert_eq!(
            (
                audit.source_gold,
                audit.scored_gold,
                audit.excluded_empty_gold,
                audit.trimmed_gold
            ),
            (3, 2, 1, 1)
        );
        assert_eq!(aligned[0].location, Location::text(0, 6));
        assert_eq!(aligned[0].label(), "PER");
        assert_eq!(aligned[1].id, SignalId::new(2));
        assert_eq!(aligned[1].surface, "I  II");
        assert_eq!(aligned[1].label(), "PRODUCT");
        let (source, audit) = NerScoring::Source.reference(Some(DatasetId::HistNERo), &gold);
        assert_eq!((audit.source_gold, audit.scored_gold), (3, 3));
        assert_eq!(source[0].label(), "PERS");
        assert_eq!(source[0].location, Location::text(0, 7));
        assert_eq!(source[1].label(), "PROD");
        assert_eq!(gold[0].original_label, "PERS");
        let (other, audit) = NerScoring::Dataset.reference(Some(DatasetId::WikiGold), &gold);
        assert_eq!(audit.scored_gold, 3);
        assert_eq!(other[0].label(), "PERS");
        assert_eq!(other[0].location, Location::text(0, 7));
    }

    #[test]
    fn source_mode_compares_literal_spellings_for_other_datasets() {
        let reference = [GoldEntity::full("Zoë", EntityType::Person, "PERSON", 0, 3)];
        let count = |mode: NerScoring| {
            let (gold, _) = mode.reference(None, &reference);
            let mut predicted = vec![Signal::new(
                SignalId::new(0),
                Location::text(0, 3),
                "Zoë",
                EntityType::Person,
                1.0,
            )];
            mode.predictions(&mut predicted);
            EvalComparison::compare("Zoë", gold, predicted).correct_count()
        };
        assert_eq!(count(NerScoring::Source), 0);
        assert_eq!(count(NerScoring::Dataset), 1);
    }
}
