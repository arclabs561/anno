//! Historical text citations and reviewed navigation between editions.
//!
//! The source store supplies the stream, named edition and grounded document. Only
//! pass editions and editorial records the reader may access. A digest identifies
//! the bytes of the grounded text. Distinct named editions can have identical text.
//! Catalog attributions and their review revisions are independent of this locator.

use crate::core::grounded::html_source::source_sha256;
use crate::offset::{chars_to_bytes, CharRange, TextSpan};
use crate::GroundedDocument;
use serde::{Deserialize, Serialize};

/// Identity of one published text edition, scoped like a publication stream.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EditionLocator {
    /// Issuer or publication stream owning the document ID.
    pub stream: String,
    /// Document identifier inside the stream.
    pub document: String,
    /// Source's opaque edition identifier.
    pub source_revision: String,
    /// SHA-256 of the exact UTF-8 bytes of the grounded text.
    pub source_sha256: String,
}

/// Reader-accessible text from the source store with its verified content digest.
#[derive(Debug)]
pub struct TextEdition<'a> {
    locator: EditionLocator,
    text: &'a str,
}

/// Failure to bind source evidence or apply an editorial record.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CitationError {
    /// The stream, document and source revision must all identify the supplied edition.
    #[error("expected a named stream, document and source revision")]
    InvalidEdition,
    /// The original edition or exact cited range could not be opened.
    #[error("historical citation evidence unavailable")]
    EvidenceUnavailable,
    /// The target must be a different named edition of the same scoped document.
    #[error("expected another edition of the same scoped document")]
    WrongTarget,
    /// The half-open range must select at least one scalar in the target text.
    #[error("invalid citation range")]
    InvalidRange,
    /// Editorial revisions are comparable only for a named authority.
    #[error("expected an editorial authority")]
    MissingAuthority,
    /// Incompatible decisions share the last visible revision of one authority.
    #[error("conflicting editorial decisions at the selected revision")]
    ConflictingDecisions,
}

impl<'a> TextEdition<'a> {
    /// Bind the named edition to the actual bytes of a retrieved grounded document.
    pub fn new(
        stream: &str,
        source_revision: &str,
        doc: &'a GroundedDocument,
    ) -> Result<Self, CitationError> {
        if [stream, source_revision, doc.id()]
            .iter()
            .any(|v| v.trim().is_empty())
        {
            return Err(CitationError::InvalidEdition);
        }
        Ok(Self {
            locator: EditionLocator {
                stream: stream.into(),
                document: doc.id().into(),
                source_revision: source_revision.into(),
                source_sha256: source_sha256(doc.text()),
            },
            text: doc.text(),
        })
    }

    /// Serializable identity that can accompany a stored citation or review.
    pub fn locator(&self) -> &EditionLocator {
        &self.locator
    }

    /// Cite a nonempty half-open range in Unicode scalar offsets.
    pub fn cite(&self, range: CharRange) -> Result<TextCitation, CitationError> {
        self.span(range).ok_or(CitationError::InvalidRange)?;
        Ok(TextCitation {
            edition: self.locator.clone(),
            range,
        })
    }

    fn span(&self, range: CharRange) -> Option<TextSpan> {
        let span = TextSpan::from_chars(self.text, range.start.get(), range.end.get());
        (!range.is_empty() && span.char_end == range.end.get()).then_some(span)
    }
}

/// A historical citation contains no copy of the source quote.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextCitation {
    /// Exactly which text edition was cited.
    pub edition: EditionLocator,
    /// Half-open Unicode scalar offsets.
    pub range: CharRange,
}

/// An editor's claim about the original field in one particular target edition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Correspondence {
    /// The field survives at this scalar range. Its wording may have changed.
    Confirmed(CharRange),
    /// The authority explicitly recorded the field's withdrawal in this edition.
    Withdrawn,
}

/// Immutable editorial record. The caller selects records from its trusted source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuccessorDecision {
    /// Historical citation to which the review applies.
    pub original: TextCitation,
    /// Exact target edition covered by the review.
    pub target: EditionLocator,
    /// Authority responsible for the review.
    pub authority: String,
    /// Monotone recording revision within that authority.
    pub revision: u64,
    /// Recorded correspondence or withdrawal.
    pub correspondence: Correspondence,
}

/// Reader's view of one authority's editorial history.
pub struct EditorialReview<'a> {
    /// Authority whose decision the reader requests.
    pub authority: &'a str,
    /// Include decisions recorded up to and including this revision.
    pub through_revision: u64,
    /// Records that the application has authenticated and authorized for this reader.
    pub decisions: &'a [SuccessorDecision],
}

/// Evidence for navigating from a historical citation to a particular target edition.
#[derive(Debug, PartialEq, Eq)]
pub enum Successor<'a> {
    /// All literal quote matches with the requested exact prefix and suffix, in source order.
    /// Zero matches leave the successor unknown. One match still requires review.
    Candidates {
        /// Edition actually searched, including when the result is empty.
        target: EditionLocator,
        /// Maximum number of original scalars matched exactly on each side of the quote.
        context_chars: usize,
        /// Half-open scalar ranges in the target edition.
        ranges: Vec<CharRange>,
    },
    /// The visible editorial record confirms this citation in the target edition.
    Confirmed {
        /// Target range checked against the actual target text.
        citation: TextCitation,
        /// Authority, recording revision and original assertion.
        decision: &'a SuccessorDecision,
    },
    /// The authority recorded an explicit withdrawal.
    Withdrawn(&'a SuccessorDecision),
}

impl TextCitation {
    /// Open the exact historical citation. An inaccessible, missing or mismatched
    /// edition returns the same unavailable result. Never searches another edition.
    pub fn open<'a>(&self, edition: Option<&TextEdition<'a>>) -> Result<&'a str, CitationError> {
        let edition = edition.ok_or(CitationError::EvidenceUnavailable)?;
        let span = self.verified_span(edition)?;
        Ok(&edition.text[span.byte_start..span.byte_end])
    }

    fn verified_span(&self, edition: &TextEdition<'_>) -> Result<TextSpan, CitationError> {
        if self.edition != edition.locator {
            return Err(CitationError::EvidenceUnavailable);
        }
        edition
            .span(self.range)
            .ok_or(CitationError::EvidenceUnavailable)
    }

    /// Look up a correspondence in a reader-accessible edition of the same document.
    /// The caller selects the intended target. Edition names have no implicit ordering.
    /// The latest visible decision from the selected authority wins. Without one,
    /// search literally, using up to `context_chars` scalars on each side from the
    /// original edition. A candidate search includes overlapping occurrences.
    pub fn successor<'a>(
        &self,
        original: &TextEdition<'_>,
        target: &TextEdition<'_>,
        context_chars: usize,
        review: EditorialReview<'a>,
    ) -> Result<Successor<'a>, CitationError> {
        let span = self.verified_span(original)?;
        let next = target.locator();
        if self.edition.stream != next.stream
            || self.edition.document != next.document
            || self.edition.source_revision == next.source_revision
        {
            return Err(CitationError::WrongTarget);
        }
        if review.authority.trim().is_empty() {
            return Err(CitationError::MissingAuthority);
        }
        let applicable = review.decisions.iter().filter(|d| {
            d.original == *self
                && d.target == *next
                && d.authority == review.authority
                && d.revision <= review.through_revision
        });
        if let Some(decision) = applicable.clone().max_by_key(|d| d.revision) {
            if applicable
                .filter(|d| d.revision == decision.revision)
                .any(|d| d != decision)
            {
                return Err(CitationError::ConflictingDecisions);
            }
            return match decision.correspondence {
                Correspondence::Confirmed(range) => Ok(Successor::Confirmed {
                    citation: target.cite(range)?,
                    decision,
                }),
                Correspondence::Withdrawn => Ok(Successor::Withdrawn(decision)),
            };
        }

        let (left, right) = chars_to_bytes(
            original.text,
            span.char_start.saturating_sub(context_chars),
            span.char_end.saturating_add(context_chars),
        );
        let prefix = &original.text[left..span.byte_start];
        let quote = &original.text[span.byte_start..span.byte_end];
        let suffix = &original.text[span.byte_end..right];
        let mut ranges = Vec::new();
        // Inspect scalar starts so adjacent and overlapping matches retain separate ranges.
        for (scalar, (byte, _)) in target.text.char_indices().enumerate() {
            if let Some(after) = target.text[byte..].strip_prefix(quote) {
                if target.text[..byte].ends_with(prefix) && after.starts_with(suffix) {
                    ranges.push(CharRange::from_raw(scalar, scalar + self.range.len()));
                }
            }
        }
        Ok(Successor::Candidates {
            target: next.clone(),
            context_chars,
            ranges,
        })
    }
}
