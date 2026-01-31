//! Temporal types for historical date handling.
//!
//! This module provides types for representing dates that may:
//! - Be BCE (negative years)
//! - Have uncertain precision (year only, year-month, full date)
//! - Represent validity ranges for temporal entity disambiguation
//!
//! # Design Notes
//!
//! Standard date libraries (chrono, time) don't handle BCE dates well.
//! This module provides `HistoricalDate` specifically for historical NLP
//! where entities like "Julius Caesar" need BCE birth/death dates.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Precision of a historical date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DatePrecision {
    /// Only year is known
    #[default]
    Year,
    /// Year and month are known
    Month,
    /// Full date (year, month, day) is known
    Day,
}

/// A historical date that can represent BCE dates and partial precision.
///
/// Unlike chrono's types, this can represent:
/// - BCE dates (negative years like -44 for 44 BCE)
/// - Partial dates (year only, year-month)
/// - Far-past dates beyond chrono's range
///
/// # ISO 8601 Representation
///
/// Follows ISO 8601 extended format:
/// - CE dates: "2024-03-15", "2024-03", "2024"
/// - BCE dates: "-0044-03-15" (44 BCE March 15)
///
/// # Examples
///
/// ```rust,ignore
/// use anno_core::types::HistoricalDate;
///
/// // Julius Caesar's death
/// let caesar_death = HistoricalDate::bce(44, 3, 15);
/// assert_eq!(caesar_death.to_iso8601(), "-0044-03-15");
///
/// // Modern date
/// let modern = HistoricalDate::ce(2024, 6, 15);
/// assert_eq!(modern.to_iso8601(), "2024-06-15");
///
/// // Year only
/// let year_only = HistoricalDate::year_only(1990);
/// assert_eq!(year_only.to_iso8601(), "1990");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HistoricalDate {
    /// Year (negative for BCE, positive for CE)
    /// Year 0 does not exist in historical dating; -1 = 1 BCE, 1 = 1 CE
    pub year: i32,
    /// Month (1-12), None if unknown
    pub month: Option<u8>,
    /// Day (1-31), None if unknown
    pub day: Option<u8>,
    /// Precision of this date
    pub precision: DatePrecision,
}

impl HistoricalDate {
    /// Create a CE (Common Era) date.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use anno_core::types::HistoricalDate;
    ///
    /// let date = HistoricalDate::ce(2024, 6, 15);
    /// assert_eq!(date.year, 2024);
    /// ```
    #[must_use]
    pub fn ce(year: i32, month: u8, day: u8) -> Self {
        Self {
            year,
            month: Some(month),
            day: Some(day),
            precision: DatePrecision::Day,
        }
    }

    /// Create a BCE (Before Common Era) date.
    ///
    /// Internally stored as negative year.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use anno_core::types::HistoricalDate;
    ///
    /// let date = HistoricalDate::bce(44, 3, 15);  // 44 BCE March 15
    /// assert_eq!(date.year, -44);
    /// ```
    #[must_use]
    pub fn bce(year: u32, month: u8, day: u8) -> Self {
        Self {
            year: -(year as i32),
            month: Some(month),
            day: Some(day),
            precision: DatePrecision::Day,
        }
    }

    /// Create a year-only date.
    #[must_use]
    pub fn year_only(year: i32) -> Self {
        Self {
            year,
            month: None,
            day: None,
            precision: DatePrecision::Year,
        }
    }

    /// Create a year-month date.
    #[must_use]
    pub fn year_month(year: i32, month: u8) -> Self {
        Self {
            year,
            month: Some(month),
            day: None,
            precision: DatePrecision::Month,
        }
    }

    /// Check if this is a BCE date.
    #[must_use]
    pub fn is_bce(&self) -> bool {
        self.year < 0
    }

    /// Get the year as a positive number with era suffix.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use anno_core::types::HistoricalDate;
    ///
    /// let bce = HistoricalDate::bce(44, 3, 15);
    /// assert_eq!(bce.format_year(), "44 BCE");
    ///
    /// let ce = HistoricalDate::ce(2024, 6, 15);
    /// assert_eq!(ce.format_year(), "2024 CE");
    /// ```
    #[must_use]
    pub fn format_year(&self) -> String {
        if self.year < 0 {
            format!("{} BCE", -self.year)
        } else {
            format!("{} CE", self.year)
        }
    }

    /// Convert to ISO 8601 string.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use anno_core::types::HistoricalDate;
    ///
    /// let date = HistoricalDate::bce(44, 3, 15);
    /// assert_eq!(date.to_iso8601(), "-0044-03-15");
    /// ```
    #[must_use]
    pub fn to_iso8601(&self) -> String {
        match self.precision {
            DatePrecision::Year => {
                if self.year < 0 {
                    format!("-{:04}", -self.year)
                } else {
                    format!("{}", self.year)
                }
            }
            DatePrecision::Month => {
                let month = self.month.unwrap_or(1);
                if self.year < 0 {
                    format!("-{:04}-{:02}", -self.year, month)
                } else {
                    format!("{}-{:02}", self.year, month)
                }
            }
            DatePrecision::Day => {
                let month = self.month.unwrap_or(1);
                let day = self.day.unwrap_or(1);
                if self.year < 0 {
                    format!("-{:04}-{:02}-{:02}", -self.year, month, day)
                } else {
                    format!("{}-{:02}-{:02}", self.year, month, day)
                }
            }
        }
    }

    /// Parse from ISO 8601 string.
    ///
    /// Handles:
    /// - "2024-06-15" (full CE date)
    /// - "-0044-03-15" (full BCE date)
    /// - "2024-06" (year-month)
    /// - "2024" or "-0044" (year only)
    ///
    /// # Errors
    ///
    /// Returns error if the string cannot be parsed.
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();

        // Handle BCE dates (leading minus)
        let (is_negative, rest) = if let Some(stripped) = s.strip_prefix('-') {
            (true, stripped)
        } else {
            (false, s)
        };

        let parts: Vec<&str> = rest.split('-').collect();

        let year: i32 = parts
            .first()
            .ok_or_else(|| "Missing year".to_string())?
            .parse()
            .map_err(|e| format!("Invalid year: {}", e))?;

        let year = if is_negative { -year } else { year };

        match parts.len() {
            1 => Ok(Self::year_only(year)),
            2 => {
                let month: u8 = parts[1]
                    .parse()
                    .map_err(|e| format!("Invalid month: {}", e))?;
                Ok(Self::year_month(year, month))
            }
            3 => {
                let month: u8 = parts[1]
                    .parse()
                    .map_err(|e| format!("Invalid month: {}", e))?;
                let day: u8 = parts[2]
                    .parse()
                    .map_err(|e| format!("Invalid day: {}", e))?;
                Ok(Self {
                    year,
                    month: Some(month),
                    day: Some(day),
                    precision: DatePrecision::Day,
                })
            }
            _ => Err(format!("Invalid date format: {}", s)),
        }
    }

    /// Extract just the year from an ISO 8601 string.
    ///
    /// Useful for quick temporal comparisons without full parsing.
    #[must_use]
    pub fn parse_year(s: &str) -> Option<i32> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        let (is_negative, rest) = if let Some(stripped) = s.strip_prefix('-') {
            (true, stripped)
        } else {
            (false, s)
        };

        let year_str = rest.split('-').next()?;
        let year: i32 = year_str.parse().ok()?;

        Some(if is_negative { -year } else { year })
    }
}

impl PartialOrd for HistoricalDate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HistoricalDate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare years first
        match self.year.cmp(&other.year) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Compare months (None < Some)
        match (self.month, other.month) {
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(a), Some(b)) => match a.cmp(&b) {
                Ordering::Equal => {}
                ord => return ord,
            },
            (None, None) => {}
        }

        // Compare days (None < Some)
        match (self.day, other.day) {
            (None, Some(_)) => Ordering::Less,
            (Some(_), None) => Ordering::Greater,
            (Some(a), Some(b)) => a.cmp(&b),
            (None, None) => Ordering::Equal,
        }
    }
}

impl std::fmt::Display for HistoricalDate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_iso8601())
    }
}

impl std::str::FromStr for HistoricalDate {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Temporal validity range for an entity.
///
/// Represents the time period during which an entity reference is valid.
/// Essential for historical document disambiguation.
///
/// # Use Cases
///
/// - **People**: birth to death dates
/// - **Organizations**: founding to dissolution
/// - **Positions**: term start to term end
/// - **Events**: start to end time
///
/// # Examples
///
/// ```rust,ignore
/// use anno_core::types::{HistoricalDate, TemporalValidity};
///
/// // Julius Caesar: 100 BCE - 44 BCE
/// let caesar = TemporalValidity {
///     from: Some(HistoricalDate::bce(100, 7, 12)),
///     until: Some(HistoricalDate::bce(44, 3, 15)),
/// };
///
/// // Check if entity was valid at a given date
/// let query = HistoricalDate::bce(50, 1, 1);
/// assert!(caesar.is_valid_at(&query));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TemporalValidity {
    /// Start of validity (None = unknown/always)
    pub from: Option<HistoricalDate>,
    /// End of validity (None = still valid/unknown)
    pub until: Option<HistoricalDate>,
}

impl TemporalValidity {
    /// Create a validity range.
    #[must_use]
    pub fn new(from: Option<HistoricalDate>, until: Option<HistoricalDate>) -> Self {
        Self { from, until }
    }

    /// Create an always-valid range (no temporal constraints).
    #[must_use]
    pub fn always() -> Self {
        Self {
            from: None,
            until: None,
        }
    }

    /// Create a range that started at a date and is still valid.
    #[must_use]
    pub fn from_date(date: HistoricalDate) -> Self {
        Self {
            from: Some(date),
            until: None,
        }
    }

    /// Create a range that ended at a date.
    #[must_use]
    pub fn until_date(date: HistoricalDate) -> Self {
        Self {
            from: None,
            until: Some(date),
        }
    }

    /// Check if the entity is valid at a given date.
    ///
    /// Returns true if the date falls within the validity range.
    /// Unknown bounds (None) are treated as unbounded.
    #[must_use]
    #[allow(clippy::unnecessary_map_or)]
    pub fn is_valid_at(&self, date: &HistoricalDate) -> bool {
        let after_start = self.from.as_ref().map_or(true, |from| date >= from);
        let before_end = self.until.as_ref().map_or(true, |until| date <= until);
        after_start && before_end
    }

    /// Check if this range overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &TemporalValidity) -> bool {
        // Two ranges overlap unless one ends before the other starts
        let self_ends_before = match (&self.until, &other.from) {
            (Some(end), Some(start)) => end < start,
            _ => false,
        };

        let other_ends_before = match (&other.until, &self.from) {
            (Some(end), Some(start)) => end < start,
            _ => false,
        };

        !self_ends_before && !other_ends_before
    }

    /// Check if this entity is currently valid (no end date).
    #[must_use]
    pub fn is_current(&self) -> bool {
        self.until.is_none()
    }

    /// Compute temporal compatibility score for entity linking.
    ///
    /// Returns a score in [0.0, 1.0] based on how well the document date
    /// matches the entity's validity period.
    ///
    /// - 1.0: Document date is within validity period
    /// - 0.5-0.9: Document date is close to validity period
    /// - 0.1-0.5: Document date is far from validity period
    #[must_use]
    pub fn compatibility_score(&self, document_date: &HistoricalDate) -> f64 {
        let doc_year = document_date.year;

        match (&self.from, &self.until) {
            (None, None) => 1.0, // Unknown = compatible

            (Some(from), None) => {
                if doc_year >= from.year {
                    1.0
                } else {
                    // Before entity "born"
                    let years_before = from.year - doc_year;
                    (1.0 - years_before as f64 / 50.0).max(0.1)
                }
            }

            (None, Some(until)) => {
                if doc_year <= until.year {
                    1.0
                } else {
                    // After entity "died"
                    let years_after = doc_year - until.year;
                    (1.0 - years_after as f64 / 50.0).max(0.1)
                }
            }

            (Some(from), Some(until)) => {
                if doc_year >= from.year && doc_year <= until.year {
                    1.0
                } else if doc_year < from.year {
                    let years_before = from.year - doc_year;
                    (1.0 - years_before as f64 / 50.0).max(0.1)
                } else {
                    let years_after = doc_year - until.year;
                    (1.0 - years_after as f64 / 50.0).max(0.1)
                }
            }
        }
    }
}

impl std::fmt::Display for TemporalValidity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (&self.from, &self.until) {
            (None, None) => write!(f, "always"),
            (Some(from), None) => write!(f, "{} - present", from),
            (None, Some(until)) => write!(f, "? - {}", until),
            (Some(from), Some(until)) => write!(f, "{} - {}", from, until),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_date_ce() {
        let date = HistoricalDate::ce(2024, 6, 15);
        assert_eq!(date.year, 2024);
        assert_eq!(date.month, Some(6));
        assert_eq!(date.day, Some(15));
        assert!(!date.is_bce());
        assert_eq!(date.to_iso8601(), "2024-06-15");
    }

    #[test]
    fn test_historical_date_bce() {
        let date = HistoricalDate::bce(44, 3, 15);
        assert_eq!(date.year, -44);
        assert!(date.is_bce());
        assert_eq!(date.to_iso8601(), "-0044-03-15");
        assert_eq!(date.format_year(), "44 BCE");
    }

    #[test]
    fn test_historical_date_year_only() {
        let date = HistoricalDate::year_only(1990);
        assert_eq!(date.to_iso8601(), "1990");

        let bce = HistoricalDate::year_only(-500);
        assert_eq!(bce.to_iso8601(), "-0500");
    }

    #[test]
    fn test_historical_date_parse() {
        assert_eq!(
            HistoricalDate::parse("2024-06-15").expect("parse ISO date"),
            HistoricalDate::ce(2024, 6, 15)
        );
        assert_eq!(
            HistoricalDate::parse("-0044-03-15").expect("parse BCE ISO date"),
            HistoricalDate::bce(44, 3, 15)
        );
        assert_eq!(
            HistoricalDate::parse("2024").expect("parse year-only date"),
            HistoricalDate::year_only(2024)
        );
    }

    #[test]
    fn test_historical_date_ordering() {
        let bce_100 = HistoricalDate::year_only(-100);
        let bce_50 = HistoricalDate::year_only(-50);
        let ce_50 = HistoricalDate::year_only(50);
        let ce_100 = HistoricalDate::year_only(100);

        assert!(bce_100 < bce_50);
        assert!(bce_50 < ce_50);
        assert!(ce_50 < ce_100);
    }

    #[test]
    fn test_temporal_validity_is_valid_at() {
        let caesar = TemporalValidity::new(
            Some(HistoricalDate::bce(100, 7, 12)),
            Some(HistoricalDate::bce(44, 3, 15)),
        );

        assert!(caesar.is_valid_at(&HistoricalDate::bce(50, 1, 1)));
        assert!(!caesar.is_valid_at(&HistoricalDate::bce(200, 1, 1)));
        assert!(!caesar.is_valid_at(&HistoricalDate::ce(2024, 1, 1)));
    }

    #[test]
    fn test_temporal_validity_compatibility() {
        let bush_sr = TemporalValidity::new(
            Some(HistoricalDate::ce(1924, 6, 12)),
            Some(HistoricalDate::ce(2018, 11, 30)),
        );

        // Document from 1990 - Bush was alive
        let score_1990 = bush_sr.compatibility_score(&HistoricalDate::year_only(1990));
        assert!((score_1990 - 1.0).abs() < 0.01);

        // Document from 2024 - Bush died 6 years ago
        let score_2024 = bush_sr.compatibility_score(&HistoricalDate::year_only(2024));
        assert!(score_2024 < 1.0);
        assert!(score_2024 > 0.5);
    }

    #[test]
    fn test_temporal_validity_overlaps() {
        let range1 = TemporalValidity::new(
            Some(HistoricalDate::year_only(1990)),
            Some(HistoricalDate::year_only(2000)),
        );
        let range2 = TemporalValidity::new(
            Some(HistoricalDate::year_only(1995)),
            Some(HistoricalDate::year_only(2005)),
        );
        let range3 = TemporalValidity::new(
            Some(HistoricalDate::year_only(2010)),
            Some(HistoricalDate::year_only(2020)),
        );

        assert!(range1.overlaps(&range2));
        assert!(!range1.overlaps(&range3));
    }

    #[test]
    fn test_serde_roundtrip() {
        let date = HistoricalDate::bce(44, 3, 15);
        let json = serde_json::to_string(&date).expect("serialize HistoricalDate");
        let recovered: HistoricalDate =
            serde_json::from_str(&json).expect("deserialize HistoricalDate");
        assert_eq!(date, recovered);

        let validity = TemporalValidity::new(Some(date), None);
        let json = serde_json::to_string(&validity).expect("serialize TemporalValidity");
        let recovered: TemporalValidity =
            serde_json::from_str(&json).expect("deserialize TemporalValidity");
        assert_eq!(validity, recovered);
    }
}
