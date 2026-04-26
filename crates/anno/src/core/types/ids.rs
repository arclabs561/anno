//! Type-safe identifier types for the annotation hierarchy.
//!
//! These newtypes provide compile-time safety to prevent accidentally
//! mixing up different kinds of IDs (e.g., passing a `TrackId` where
//! a `SignalId` is expected).
//!
//! # Design
//!
//! - `#[repr(transparent)]` ensures zero-cost abstraction
//! - Serde serializes as plain u64 for backwards compatibility
//! - All common traits derived for ergonomic use
//!
//! # Example
//!
//! ```rust
//! use anno::{SignalId, TrackId};
//!
//! fn process_signal(id: SignalId) { /* ... */ }
//! fn process_track(id: TrackId) { /* ... */ }
//!
//! let signal_id = SignalId::new(1);
//! let track_id = TrackId::new(1);
//!
//! process_signal(signal_id);  // OK
//! process_track(track_id);    // OK
//! // process_signal(track_id); // Compile error! Type safety.
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// Macro to define a type-safe ID newtype.
macro_rules! define_id {
    (
        $(#[$meta:meta])*
        $name:ident
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Zero ID constant.
            pub const ZERO: Self = Self(0);

            /// Create a new ID.
            #[inline]
            #[must_use]
            pub const fn new(id: u64) -> Self {
                Self(id)
            }

            /// Get the raw u64 value.
            #[inline]
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }

            /// Increment the ID and return the new value.
            #[inline]
            #[must_use]
            pub const fn next(self) -> Self {
                Self(self.0 + 1)
            }

            /// Increment this ID in place and return the old value.
            /// Useful for generating sequential IDs.
            #[inline]
            pub fn next_mut(&mut self) -> Self {
                let old = *self;
                self.0 += 1;
                old
            }
        }

        impl std::ops::AddAssign<u64> for $name {
            #[inline]
            fn add_assign(&mut self, rhs: u64) {
                self.0 += rhs;
            }
        }

        impl std::ops::Add<u64> for $name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: u64) -> Self {
                Self(self.0 + rhs)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<u64> for $name {
            #[inline]
            fn from(id: u64) -> Self {
                Self(id)
            }
        }

        impl From<usize> for $name {
            #[inline]
            fn from(id: usize) -> Self {
                Self(id as u64)
            }
        }

        impl From<i32> for $name {
            #[inline]
            fn from(id: i32) -> Self {
                Self(id as u64)
            }
        }

        impl From<$name> for u64 {
            #[inline]
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl From<$name> for usize {
            #[inline]
            fn from(id: $name) -> Self {
                id.0 as usize
            }
        }

        // Serde: serialize as plain u64 for backwards compatibility
        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                u64::deserialize(deserializer).map(Self)
            }
        }
    };
}

define_id! {
    /// Unique identifier for a signal within a document.
    ///
    /// Signals are Level 1 in the hierarchy: raw entity detections.
    /// Each signal has a unique ID within its containing document.
    SignalId
}

define_id! {
    /// Unique identifier for a track within a document.
    ///
    /// Tracks are Level 2: within-document coreference chains.
    /// A track groups multiple signals that refer to the same entity.
    TrackId
}

define_id! {
    /// Unique identifier for an identity within a corpus.
    ///
    /// Identities are Level 3: cross-document entities with optional KB links.
    /// An identity may span multiple documents and link to knowledge bases.
    IdentityId
}

define_id! {
    /// Unique identifier for a coreference cluster.
    ///
    /// Used on `Entity.canonical_id` to group coreferent mentions.
    /// Within a document, entities with the same `CanonicalId` refer to
    /// the same real-world entity.
    CanonicalId
}

// =============================================================================
// Offset Types: Type-Safe Text Positions
// =============================================================================

/// A character (Unicode scalar value) offset in text.
///
/// Character offsets count `char` values, which is what `text.chars().count()`
/// returns and what `String::char_indices()` uses.
///
/// # Why This Exists
///
/// In Unicode text, byte offsets differ from character offsets:
/// - "日本語" has 9 bytes but 3 characters
/// - Indexing by bytes (`text[0..3]`) risks splitting multi-byte characters
///
/// Using `CharOffset` instead of `usize` makes the offset unit explicit and
/// prevents accidental mixing of byte and character offsets at compile time.
///
/// # Example
///
/// ```rust
/// use anno::core::types::CharOffset;
///
/// let text = "日本語";
/// let start = CharOffset::new(0);
/// let end = CharOffset::new(3);
///
/// // Convert to range and extract
/// let chars: String = text.chars()
///     .skip(start.get())
///     .take(end.get() - start.get())
///     .collect();
/// assert_eq!(chars, "日本語");
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Debug)]
#[repr(transparent)]
pub struct CharOffset(usize);

impl CharOffset {
    /// Zero offset constant.
    pub const ZERO: Self = Self(0);

    /// Create a new character offset.
    #[inline]
    #[must_use]
    pub const fn new(offset: usize) -> Self {
        Self(offset)
    }

    /// Get the raw usize value.
    #[inline]
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for CharOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for CharOffset {
    #[inline]
    fn from(offset: usize) -> Self {
        Self(offset)
    }
}

impl From<CharOffset> for usize {
    #[inline]
    fn from(offset: CharOffset) -> Self {
        offset.0
    }
}

impl Serialize for CharOffset {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CharOffset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        usize::deserialize(deserializer).map(Self)
    }
}

/// A byte offset in text.
///
/// Byte offsets are what `str` methods like `get(start..end)` use, but they
/// must align with UTF-8 code unit boundaries.
///
/// # When to Use
///
/// - When interfacing with regex crate (returns byte offsets)
/// - When calling `str::get(start..end)` or `str[start..end]`
/// - When parsing byte-oriented formats
///
/// Use `anno::offset::bytes_to_chars` to convert to `CharOffset` for entity storage.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Debug)]
#[repr(transparent)]
pub struct ByteOffset(usize);

impl ByteOffset {
    /// Zero offset constant.
    pub const ZERO: Self = Self(0);

    /// Create a new byte offset.
    #[inline]
    #[must_use]
    pub const fn new(offset: usize) -> Self {
        Self(offset)
    }

    /// Get the raw usize value.
    #[inline]
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for ByteOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for ByteOffset {
    #[inline]
    fn from(offset: usize) -> Self {
        Self(offset)
    }
}

impl From<ByteOffset> for usize {
    #[inline]
    fn from(offset: ByteOffset) -> Self {
        offset.0
    }
}

impl Serialize for ByteOffset {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ByteOffset {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        usize::deserialize(deserializer).map(Self)
    }
}

/// A character span: (start, end) in character offsets.
///
/// This is a typed wrapper around `(CharOffset, CharOffset)` for convenience.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct CharSpan {
    /// Start character offset (inclusive).
    pub start: CharOffset,
    /// End character offset (exclusive).
    pub end: CharOffset,
}

impl CharSpan {
    /// Create a new character span.
    ///
    /// If `start > end`, the values are swapped to maintain the invariant `start <= end`.
    #[inline]
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        if start <= end {
            Self {
                start: CharOffset::new(start),
                end: CharOffset::new(end),
            }
        } else {
            Self {
                start: CharOffset::new(end),
                end: CharOffset::new(start),
            }
        }
    }

    /// Create from raw offsets.
    #[inline]
    #[must_use]
    pub const fn from_offsets(start: CharOffset, end: CharOffset) -> Self {
        if start.0 <= end.0 {
            Self { start, end }
        } else {
            Self {
                start: end,
                end: start,
            }
        }
    }

    /// Length in characters.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.0.saturating_sub(self.start.0)
    }

    /// Check if the span is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start.0 >= self.end.0
    }

    /// Convert to a `Range<usize>`.
    #[inline]
    #[must_use]
    pub const fn to_range(&self) -> std::ops::Range<usize> {
        self.start.0..self.end.0
    }
}

impl From<(usize, usize)> for CharSpan {
    #[inline]
    fn from((start, end): (usize, usize)) -> Self {
        Self::new(start, end)
    }
}

impl From<std::ops::Range<usize>> for CharSpan {
    #[inline]
    fn from(range: std::ops::Range<usize>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl Serialize for CharSpan {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.start.0, self.end.0).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CharSpan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (start, end) = <(usize, usize)>::deserialize(deserializer)?;
        Ok(Self::new(start, end))
    }
}

/// A byte span: (start, end) in byte offsets.
///
/// Use when working with regex or byte-oriented parsers.
/// Convert to `CharSpan` before storing in entities.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct ByteSpan {
    /// Start byte offset (inclusive).
    pub start: ByteOffset,
    /// End byte offset (exclusive).
    pub end: ByteOffset,
}

impl ByteSpan {
    /// Create a new byte span.
    ///
    /// If `start > end`, the values are swapped to maintain the invariant `start <= end`.
    #[inline]
    #[must_use]
    pub const fn new(start: usize, end: usize) -> Self {
        if start <= end {
            Self {
                start: ByteOffset::new(start),
                end: ByteOffset::new(end),
            }
        } else {
            Self {
                start: ByteOffset::new(end),
                end: ByteOffset::new(start),
            }
        }
    }

    /// Length in bytes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end.0.saturating_sub(self.start.0)
    }

    /// Check if the span is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.start.0 >= self.end.0
    }

    /// Convert to a `Range<usize>`.
    #[inline]
    #[must_use]
    pub const fn to_range(&self) -> std::ops::Range<usize> {
        self.start.0..self.end.0
    }
}

impl From<(usize, usize)> for ByteSpan {
    #[inline]
    fn from((start, end): (usize, usize)) -> Self {
        Self::new(start, end)
    }
}

impl From<std::ops::Range<usize>> for ByteSpan {
    #[inline]
    fn from(range: std::ops::Range<usize>) -> Self {
        Self::new(range.start, range.end)
    }
}

impl Serialize for ByteSpan {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.start.0, self.end.0).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ByteSpan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (start, end) = <(usize, usize)>::deserialize(deserializer)?;
        Ok(Self::new(start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_safety() {
        let signal_id = SignalId::new(1);
        let track_id = TrackId::new(1);

        // Same underlying value but different types
        assert_eq!(signal_id.get(), track_id.get());
        assert_eq!(signal_id.get(), 1);
        assert_eq!(track_id.get(), 1);

        // Cannot compare directly (different types)
        // This would fail to compile:
        // assert_eq!(signal_id, track_id);
    }

    #[test]
    fn test_from_u64() {
        let id: SignalId = 42u64.into();
        assert_eq!(id.get(), 42);

        let back: u64 = id.into();
        assert_eq!(back, 42);
    }

    #[test]
    fn test_serde_roundtrip() {
        let id = SignalId::new(123);
        let json = serde_json::to_string(&id).expect("serialize SignalId");
        assert_eq!(json, "123"); // Serializes as plain number

        let recovered: SignalId = serde_json::from_str(&json).expect("deserialize SignalId");
        assert_eq!(id, recovered);
    }

    #[test]
    fn test_display() {
        let id = TrackId::new(42);
        assert_eq!(format!("{}", id), "42");
        assert_eq!(format!("{:?}", id), "TrackId(42)");
    }

    #[test]
    fn test_next() {
        let id = SignalId::new(5);
        let next = id.next();
        assert_eq!(next.get(), 6);
    }

    #[test]
    fn test_ordering() {
        let a = SignalId::new(1);
        let b = SignalId::new(2);
        let c = SignalId::new(1);

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, c);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(SignalId::new(1));
        set.insert(SignalId::new(2));
        set.insert(SignalId::new(1)); // Duplicate

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_default() {
        let id = SignalId::default();
        assert_eq!(id.get(), 0);
    }

    #[test]
    fn test_next_mut_sequential() {
        let mut gen = SignalId::new(10);
        let first = gen.next_mut();
        let second = gen.next_mut();
        assert_eq!(first.get(), 10);
        assert_eq!(second.get(), 11);
        assert_eq!(gen.get(), 12);
    }

    #[test]
    fn test_add_assign() {
        let mut id = TrackId::new(5);
        id += 3;
        assert_eq!(id.get(), 8);
    }

    #[test]
    fn test_from_i32() {
        let id = IdentityId::from(42i32);
        assert_eq!(id.get(), 42);
    }

    #[test]
    fn test_char_span_basics() {
        let span = CharSpan::new(5, 10);
        assert_eq!(span.len(), 5);
        assert!(!span.is_empty());
        assert_eq!(span.to_range(), 5..10);

        let empty = CharSpan::new(3, 3);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_char_span_from_tuple_and_range() {
        let from_tuple: CharSpan = (2, 7).into();
        let from_range: CharSpan = (2..7).into();
        assert_eq!(from_tuple, from_range);
    }

    #[test]
    fn test_char_span_serde_roundtrip() {
        let span = CharSpan::new(10, 20);
        let json = serde_json::to_string(&span).expect("serialize CharSpan");
        assert_eq!(json, "[10,20]");
        let recovered: CharSpan = serde_json::from_str(&json).expect("deserialize CharSpan");
        assert_eq!(span, recovered);
    }

    #[test]
    fn test_byte_span_basics() {
        let span = ByteSpan::new(0, 9);
        assert_eq!(span.len(), 9);
        assert!(!span.is_empty());
        assert_eq!(span.to_range(), 0..9);
    }

    #[test]
    fn test_byte_offset_display_and_conversions() {
        let offset = ByteOffset::new(42);
        assert_eq!(format!("{}", offset), "42");
        let raw: usize = offset.into();
        assert_eq!(raw, 42);
        let back: ByteOffset = raw.into();
        assert_eq!(back, offset);
    }

    #[test]
    fn test_canonical_id_zero_constant() {
        assert_eq!(CanonicalId::ZERO.get(), 0);
        assert_eq!(CanonicalId::ZERO, CanonicalId::default());
    }
}
