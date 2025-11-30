//! Long-tail entity evaluation for NER.
//!
//! Measures performance on rare entity types and entities that appear infrequently.
//! Critical because aggregate F1 masks poor performance on minority classes.
//!
//! # Example
//!
//! ```rust
//! use anno::eval::long_tail::{LongTailAnalyzer, EntityFrequency};
//!
//! let frequencies = vec![
//!     EntityFrequency::new("PER", 1000),   // Head: very common
//!     EntityFrequency::new("ORG", 800),    // Head
//!     EntityFrequency::new("LOC", 600),    // Mid
//!     EntityFrequency::new("DATE", 200),   // Mid
//!     EntityFrequency::new("DISEASE", 50), // Tail: rare
//!     EntityFrequency::new("GENE", 20),    // Tail
//! ];
//!
//! let analyzer = LongTailAnalyzer::default();
//! let splits = analyzer.split_by_frequency(&frequencies);
//!
//! println!("Head types (top 20%): {:?}", splits.head);
//! println!("Tail types (bottom 20%): {:?}", splits.tail);
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// Data Structures
// =============================================================================

/// Frequency information for an entity type.
#[derive(Debug, Clone)]
pub struct EntityFrequency {
    /// Entity type name
    pub entity_type: String,
    /// Number of occurrences in dataset
    pub count: usize,
}

impl EntityFrequency {
    /// Create new frequency entry.
    pub fn new(entity_type: impl Into<String>, count: usize) -> Self {
        Self {
            entity_type: entity_type.into(),
            count,
        }
    }
}

/// Performance metrics for an entity type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypePerformance {
    /// Entity type name
    pub entity_type: String,
    /// Number of gold entities
    pub count: usize,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Frequency bucket (head/mid/tail)
    pub bucket: FrequencyBucket,
}

/// Frequency bucket classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrequencyBucket {
    /// Top 20% by frequency
    Head,
    /// Middle 60% by frequency
    Mid,
    /// Bottom 20% by frequency
    Tail,
}

impl std::fmt::Display for FrequencyBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrequencyBucket::Head => write!(f, "Head"),
            FrequencyBucket::Mid => write!(f, "Mid"),
            FrequencyBucket::Tail => write!(f, "Tail"),
        }
    }
}

/// Split of entity types by frequency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencySplit {
    /// Head types (most common)
    pub head: Vec<String>,
    /// Mid types
    pub mid: Vec<String>,
    /// Tail types (least common)
    pub tail: Vec<String>,
    /// Percentage of total entities in head bucket
    pub head_coverage: f64,
    /// Percentage of total entities in mid bucket
    pub mid_coverage: f64,
    /// Percentage of total entities in tail bucket
    pub tail_coverage: f64,
}

/// Long-tail analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTailResults {
    /// Performance per entity type
    pub per_type: Vec<TypePerformance>,
    /// Average F1 for head types
    pub head_f1: f64,
    /// Average F1 for mid types
    pub mid_f1: f64,
    /// Average F1 for tail types
    pub tail_f1: f64,
    /// Gap between head and tail (higher = more disparity)
    pub head_tail_gap: f64,
    /// Gini coefficient of performance across types (0=equal, 1=max inequality)
    pub gini_coefficient: f64,
    /// Number of types with F1 < 0.5
    pub struggling_types: usize,
    /// Number of types with F1 = 0 (completely failed)
    pub failed_types: usize,
    /// Insights
    pub insights: Vec<String>,
}

// =============================================================================
// Long-Tail Analyzer
// =============================================================================

/// Analyzer for long-tail entity performance.
#[derive(Debug, Clone)]
pub struct LongTailAnalyzer {
    /// Percentile for head/tail cutoff (e.g., 0.2 = top/bottom 20%)
    pub tail_percentile: f64,
}

impl Default for LongTailAnalyzer {
    fn default() -> Self {
        Self {
            tail_percentile: 0.2,
        }
    }
}

impl LongTailAnalyzer {
    /// Create analyzer with custom tail percentile.
    pub fn new(tail_percentile: f64) -> Self {
        Self {
            tail_percentile: tail_percentile.clamp(0.05, 0.4),
        }
    }

    /// Split entity types by frequency into head/mid/tail buckets.
    pub fn split_by_frequency(&self, frequencies: &[EntityFrequency]) -> FrequencySplit {
        if frequencies.is_empty() {
            return FrequencySplit {
                head: vec![],
                mid: vec![],
                tail: vec![],
                head_coverage: 0.0,
                mid_coverage: 0.0,
                tail_coverage: 0.0,
            };
        }

        // Sort by count descending
        let mut sorted: Vec<_> = frequencies.to_vec();
        sorted.sort_by(|a, b| b.count.cmp(&a.count));

        let total: usize = sorted.iter().map(|f| f.count).sum();
        let n = sorted.len();

        // Determine cutoffs
        let head_cutoff = (n as f64 * self.tail_percentile).ceil() as usize;
        let tail_cutoff = n - head_cutoff;

        let mut head = Vec::new();
        let mut mid = Vec::new();
        let mut tail = Vec::new();

        let mut head_count = 0usize;
        let mut mid_count = 0usize;
        let mut tail_count = 0usize;

        for (i, f) in sorted.iter().enumerate() {
            if i < head_cutoff {
                head.push(f.entity_type.clone());
                head_count += f.count;
            } else if i >= tail_cutoff {
                tail.push(f.entity_type.clone());
                tail_count += f.count;
            } else {
                mid.push(f.entity_type.clone());
                mid_count += f.count;
            }
        }

        let total_f64 = total as f64;
        FrequencySplit {
            head,
            mid,
            tail,
            head_coverage: if total > 0 {
                head_count as f64 / total_f64
            } else {
                0.0
            },
            mid_coverage: if total > 0 {
                mid_count as f64 / total_f64
            } else {
                0.0
            },
            tail_coverage: if total > 0 {
                tail_count as f64 / total_f64
            } else {
                0.0
            },
        }
    }

    /// Classify a single entity type into a bucket.
    pub fn classify_type(
        &self,
        entity_type: &str,
        frequencies: &[EntityFrequency],
    ) -> FrequencyBucket {
        let split = self.split_by_frequency(frequencies);

        if split.head.contains(&entity_type.to_string()) {
            FrequencyBucket::Head
        } else if split.tail.contains(&entity_type.to_string()) {
            FrequencyBucket::Tail
        } else {
            FrequencyBucket::Mid
        }
    }

    /// Analyze long-tail performance from per-type metrics.
    pub fn analyze(&self, type_metrics: &[(String, usize, f64, f64, f64)]) -> LongTailResults {
        // type_metrics: (type_name, count, precision, recall, f1)

        if type_metrics.is_empty() {
            return LongTailResults {
                per_type: vec![],
                head_f1: 0.0,
                mid_f1: 0.0,
                tail_f1: 0.0,
                head_tail_gap: 0.0,
                gini_coefficient: 0.0,
                struggling_types: 0,
                failed_types: 0,
                insights: vec!["No entity types to analyze".into()],
            };
        }

        // Build frequency list
        let frequencies: Vec<_> = type_metrics
            .iter()
            .map(|(name, count, _, _, _)| EntityFrequency::new(name.clone(), *count))
            .collect();

        let split = self.split_by_frequency(&frequencies);

        // Build per-type performance with bucket assignment
        let per_type: Vec<_> = type_metrics
            .iter()
            .map(|(name, count, prec, rec, f1)| {
                let bucket = self.classify_type(name, &frequencies);
                TypePerformance {
                    entity_type: name.clone(),
                    count: *count,
                    precision: *prec,
                    recall: *rec,
                    f1: *f1,
                    bucket,
                }
            })
            .collect();

        // Compute bucket averages
        let head_types: Vec<_> = per_type
            .iter()
            .filter(|t| t.bucket == FrequencyBucket::Head)
            .collect();
        let mid_types: Vec<_> = per_type
            .iter()
            .filter(|t| t.bucket == FrequencyBucket::Mid)
            .collect();
        let tail_types: Vec<_> = per_type
            .iter()
            .filter(|t| t.bucket == FrequencyBucket::Tail)
            .collect();

        let head_f1 = if head_types.is_empty() {
            0.0
        } else {
            head_types.iter().map(|t| t.f1).sum::<f64>() / head_types.len() as f64
        };

        let mid_f1 = if mid_types.is_empty() {
            0.0
        } else {
            mid_types.iter().map(|t| t.f1).sum::<f64>() / mid_types.len() as f64
        };

        let tail_f1 = if tail_types.is_empty() {
            0.0
        } else {
            tail_types.iter().map(|t| t.f1).sum::<f64>() / tail_types.len() as f64
        };

        let head_tail_gap = head_f1 - tail_f1;

        // Compute Gini coefficient of F1 scores
        let gini_coefficient = compute_gini(&per_type.iter().map(|t| t.f1).collect::<Vec<_>>());

        // Count struggling and failed types
        let struggling_types = per_type.iter().filter(|t| t.f1 < 0.5).count();
        let failed_types = per_type.iter().filter(|t| t.f1 < 0.01).count();

        // Generate insights
        let mut insights = Vec::new();

        if head_tail_gap > 0.3 {
            insights.push(format!(
                "Large head-tail gap ({:.0}%): tail types severely underperforming",
                head_tail_gap * 100.0
            ));
        } else if head_tail_gap < 0.1 {
            insights.push("Low head-tail gap: relatively uniform performance across types".into());
        }

        if gini_coefficient > 0.4 {
            insights.push(format!(
                "High inequality (Gini={:.2}): performance very uneven across types",
                gini_coefficient
            ));
        }

        if failed_types > 0 {
            insights.push(format!(
                "{} entity types completely failed (F1=0%)",
                failed_types
            ));
        }

        if !tail_types.is_empty() && tail_f1 < 0.3 {
            let tail_names: Vec<_> = tail_types.iter().map(|t| t.entity_type.as_str()).collect();
            insights.push(format!(
                "Tail types struggling: {:?}",
                &tail_names[..tail_names.len().min(3)]
            ));
        }

        // Coverage insight
        if split.tail_coverage > 0.0 && split.tail_coverage < 0.1 {
            insights.push(format!(
                "Tail types represent only {:.1}% of data - may need upsampling",
                split.tail_coverage * 100.0
            ));
        }

        LongTailResults {
            per_type,
            head_f1,
            mid_f1,
            tail_f1,
            head_tail_gap,
            gini_coefficient,
            struggling_types,
            failed_types,
            insights,
        }
    }
}

/// Compute Gini coefficient for a list of values.
fn compute_gini(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    if mean < 1e-10 {
        return 0.0; // All zeros
    }

    let mut sum_abs_diff = 0.0;
    for v1 in values {
        for v2 in values {
            sum_abs_diff += (v1 - v2).abs();
        }
    }

    sum_abs_diff / (2.0 * n * n * mean)
}

/// Format long-tail results for display.
pub fn format_long_tail_results(results: &LongTailResults) -> String {
    let mut output = String::new();

    output.push_str("Long-Tail Analysis:\n");
    output.push_str(&format!("  Head F1: {:.1}%\n", results.head_f1 * 100.0));
    output.push_str(&format!("  Mid F1:  {:.1}%\n", results.mid_f1 * 100.0));
    output.push_str(&format!("  Tail F1: {:.1}%\n", results.tail_f1 * 100.0));
    output.push_str(&format!(
        "  Head-Tail Gap: {:.1}%\n",
        results.head_tail_gap * 100.0
    ));
    output.push_str(&format!(
        "  Gini Coefficient: {:.3}\n",
        results.gini_coefficient
    ));
    output.push_str(&format!(
        "  Struggling types (F1<50%): {}\n",
        results.struggling_types
    ));
    output.push_str(&format!(
        "  Failed types (F1=0%): {}\n",
        results.failed_types
    ));

    if !results.insights.is_empty() {
        output.push_str("\nInsights:\n");
        for insight in &results.insights {
            output.push_str(&format!("  - {}\n", insight));
        }
    }

    output
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_split() {
        let frequencies = vec![
            EntityFrequency::new("A", 100),
            EntityFrequency::new("B", 80),
            EntityFrequency::new("C", 60),
            EntityFrequency::new("D", 40),
            EntityFrequency::new("E", 20),
        ];

        let analyzer = LongTailAnalyzer::new(0.2);
        let split = analyzer.split_by_frequency(&frequencies);

        // With 5 types and 20% cutoff, head=1, tail=1, mid=3
        assert_eq!(split.head.len(), 1);
        assert!(split.head.contains(&"A".to_string()));
        assert_eq!(split.tail.len(), 1);
        assert!(split.tail.contains(&"E".to_string()));
    }

    #[test]
    fn test_gini_coefficient() {
        // Equal values = 0
        let equal = vec![0.5, 0.5, 0.5, 0.5];
        assert!(compute_gini(&equal) < 0.01);

        // Unequal values = higher gini
        let unequal = vec![1.0, 0.0, 0.0, 0.0];
        assert!(compute_gini(&unequal) > 0.5);
    }

    #[test]
    fn test_analyze_long_tail() {
        let type_metrics = vec![
            ("PER".to_string(), 100, 0.9, 0.85, 0.87),
            ("ORG".to_string(), 80, 0.8, 0.75, 0.77),
            ("LOC".to_string(), 60, 0.7, 0.65, 0.67),
            ("DATE".to_string(), 40, 0.6, 0.55, 0.57),
            ("DISEASE".to_string(), 20, 0.3, 0.25, 0.27),
        ];

        let analyzer = LongTailAnalyzer::new(0.2);
        let results = analyzer.analyze(&type_metrics);

        // Head (PER) should have highest F1
        assert!(results.head_f1 > results.tail_f1);
        // Tail (DISEASE) struggling
        assert!(results.tail_f1 < 0.5);
        // Gap should be significant
        assert!(results.head_tail_gap > 0.3);
    }

    #[test]
    fn test_empty_input() {
        let analyzer = LongTailAnalyzer::default();
        let results = analyzer.analyze(&[]);

        assert_eq!(results.per_type.len(), 0);
        assert!(!results.insights.is_empty());
    }

    #[test]
    fn test_bucket_assignment() {
        let frequencies = vec![
            EntityFrequency::new("A", 100),
            EntityFrequency::new("B", 50),
            EntityFrequency::new("C", 10),
        ];

        let analyzer = LongTailAnalyzer::new(0.33);

        assert_eq!(
            analyzer.classify_type("A", &frequencies),
            FrequencyBucket::Head
        );
        assert_eq!(
            analyzer.classify_type("C", &frequencies),
            FrequencyBucket::Tail
        );
    }
}
