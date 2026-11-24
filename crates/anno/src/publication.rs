// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare complete, ordered publications of a downstream index.
//!
//! A stream identifies one projection, including its source, audience, row schema and
//! filtering policy. Its producer assigns increasing sequence numbers. An occurrence
//! key can use document, source revision and signal ID as separate string components.
//! Each value is exactly the JSON data the caller intends to index.
//!
//! A checkpoint can be saved alongside the rows. Its fingerprint is opaque. This
//! implementation sorts JSON object fields recursively and hashes the serialized rows.
//! The fingerprint commits to projected JSON, including array order and number
//! representation. Reconstructing a snapshot of private Rust state requires choosing
//! its JSON projection first.
//!
//! ```
//! use anno::publication::IndexSnapshot;
//! use serde_json::json;
//! use std::collections::BTreeMap;
//!
//! let key = vec!["transcript-7".into(), "edition-1".into(), "signal-3".into()];
//! let old = IndexSnapshot::new("public-transcripts", 4,
//!     BTreeMap::from([(key.clone(), json!({"subject": ["music", "17"]}))]))?;
//! let new = IndexSnapshot::new("public-transcripts", 8,
//!     BTreeMap::from([(key, json!({"subject": ["ships", "17"]}))]))?;
//! let delta = new.changes_since(&old)?;
//! assert!(delta.needs_apply(old.checkpoint())?);
//! assert_eq!(delta.removed()[0].1["subject"][0], "music");
//! assert_eq!(delta.upserted()[0].1["subject"][0], "ships");
//! assert!(!delta.needs_apply(new.checkpoint())?);
//! # Ok::<(), anno::publication::PublicationError>(())
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{cmp::Ordering, collections::BTreeMap};
use thiserror::Error;

/// The identity of one complete index publication. Persist this with the indexed rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexCheckpoint {
    stream: String,
    sequence: u64,
    fingerprint: [u8; 32],
}

impl IndexCheckpoint {
    /// The producer's scope for this projection.
    pub fn stream(&self) -> &str {
        &self.stream
    }

    /// The producer-assigned ordering within this stream.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Check a received delta's base and target against this stored checkpoint.
    /// Identical redelivery returns false. Persist the row updates and the target
    /// checkpoint in one transaction to keep the guard effective after a crash.
    /// This validates revision ordering and predecessor identity. To check the
    /// received rows themselves, reconstruct the complete resulting snapshot and
    /// compare its checkpoint with the target before committing.
    pub fn needs_delta(&self, base: &Self, target: &Self) -> Result<bool, PublicationError> {
        target.advances(base)?;
        if !target.advances(self)? {
            return Ok(false);
        }
        if base != self {
            return Err(PublicationError::WrongPredecessor);
        }
        Ok(true)
    }

    fn advances(&self, previous: &Self) -> Result<bool, PublicationError> {
        if self.stream.trim().is_empty() {
            return Err(PublicationError::EmptyStream);
        }
        if self.stream != previous.stream {
            return Err(PublicationError::DifferentStream);
        }
        match self.sequence.cmp(&previous.sequence) {
            Ordering::Greater => Ok(true),
            Ordering::Less => Err(PublicationError::Stale),
            Ordering::Equal if self == previous => Ok(false),
            Ordering::Equal => Err(PublicationError::ConflictingRevision),
        }
    }
}

/// Failures a publication caller can distinguish during capture and replay.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PublicationError {
    /// The caller omitted a scope for its downstream projection.
    #[error("the publication stream must have a nonempty name")]
    EmptyStream,
    /// The two checkpoints describe independently scoped projections.
    #[error("publication streams differ")]
    DifferentStream,
    /// Publishing this revision would move the consumer backwards.
    #[error("publication is older than the current checkpoint")]
    Stale,
    /// The producer reused a sequence number for different row contents.
    #[error("publication sequence has conflicting contents")]
    ConflictingRevision,
    /// This delta was computed from a different state than the consumer has.
    #[error("delta predecessor differs from the stored checkpoint")]
    WrongPredecessor,
    /// A projected row could not be encoded.
    #[error("could not encode publication row: {0}")]
    Encoding(#[from] serde_json::Error),
}

#[derive(Debug)]
struct Row {
    value: Value,
    fingerprint: [u8; 32],
}

/// All rows visible in one publication, including an authoritative empty result.
#[derive(Debug)]
pub struct IndexSnapshot {
    checkpoint: IndexCheckpoint,
    rows: BTreeMap<Vec<String>, Row>,
}

impl IndexSnapshot {
    /// Capture complete rows. Keys are ordered lists of components, so punctuation
    /// in a document or catalog identifier cannot collide with a field separator.
    /// Callers assembling rows from a list must decide how duplicate keys are handled.
    pub fn new(
        stream: impl Into<String>,
        sequence: u64,
        rows: BTreeMap<Vec<String>, Value>,
    ) -> Result<Self, PublicationError> {
        let stream = stream.into();
        if stream.trim().is_empty() {
            return Err(PublicationError::EmptyStream);
        }
        let mut snapshot_hash = Sha256::new();
        snapshot_hash.update(b"anno.index.json.v1\0");
        let mut indexed = BTreeMap::new();
        for (key, mut value) in rows {
            value.sort_all_objects();
            let wire = serde_json::to_vec(&(&key, &value))?;
            let fingerprint = <[u8; 32]>::from(Sha256::digest(wire));
            snapshot_hash.update(fingerprint);
            indexed.insert(key, Row { value, fingerprint });
        }
        Ok(Self {
            rows: indexed,
            checkpoint: IndexCheckpoint {
                stream,
                sequence,
                fingerprint: snapshot_hash.finalize().into(),
            },
        })
    }

    /// Persist this with the row update in the same transaction.
    pub fn checkpoint(&self) -> &IndexCheckpoint {
        &self.checkpoint
    }

    /// All projected rows in component-key order.
    pub fn rows(&self) -> impl ExactSizeIterator<Item = (&[String], &Value)> {
        self.rows
            .iter()
            .map(|(key, row)| (key.as_slice(), &row.value))
    }

    /// Whether this full snapshot can replace the current projection. Identical
    /// redelivery returns false. A later full snapshot can recover a missing delta.
    pub fn needs_replace(&self, current: &IndexCheckpoint) -> Result<bool, PublicationError> {
        self.checkpoint.advances(current)
    }

    /// Compute the net changes from a retained, complete predecessor to this snapshot.
    /// Intermediate publications may be skipped when both endpoints are available.
    /// A row whose key survives a correction appears in both lists with its old and
    /// new values. This lets consumers retract facets derived from the earlier value.
    pub fn changes_since<'a>(
        &'a self,
        previous: &'a Self,
    ) -> Result<IndexDelta<'a>, PublicationError> {
        self.checkpoint.advances(&previous.checkpoint)?;
        Ok(IndexDelta {
            base: &previous.checkpoint,
            target: &self.checkpoint,
            removed: previous.changed_rows(self),
            upserted: self.changed_rows(previous),
        })
    }

    fn changed_rows(&self, other: &Self) -> Vec<(&[String], &Value)> {
        if self.checkpoint.fingerprint == other.checkpoint.fingerprint {
            return Vec::new();
        }
        self.rows
            .iter()
            .filter(|(key, row)| {
                other
                    .rows
                    .get(*key)
                    .is_none_or(|other_row| row.fingerprint != other_row.fingerprint)
            })
            .map(|(key, row)| (key.as_slice(), &row.value))
            .collect()
    }
}

/// Net changes between two known complete snapshots, in deterministic order.
#[derive(Debug)]
pub struct IndexDelta<'a> {
    base: &'a IndexCheckpoint,
    target: &'a IndexCheckpoint,
    removed: Vec<(&'a [String], &'a Value)>,
    upserted: Vec<(&'a [String], &'a Value)>,
}

impl<'a> IndexDelta<'a> {
    /// Checkpoint from which these changes were computed.
    pub fn base(&self) -> &IndexCheckpoint {
        self.base
    }

    /// Checkpoint to commit together with all the changes.
    pub fn target(&self) -> &IndexCheckpoint {
        self.target
    }

    /// Rows to retract before inserting replacements.
    pub fn removed(&self) -> &[(&'a [String], &'a Value)] {
        &self.removed
    }

    /// New or corrected rows to insert after retractions.
    pub fn upserted(&self) -> &[(&'a [String], &'a Value)] {
        &self.upserted
    }

    /// Check the consumer's stored checkpoint inside its write transaction.
    /// Returns false for a redelivery of an already published result. On a missing
    /// or divergent predecessor, obtain a fresh full snapshot or the missing changes.
    pub fn needs_apply(&self, current: &IndexCheckpoint) -> Result<bool, PublicationError> {
        current.needs_delta(self.base, self.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn snapshot(stream: &str, version: u64, rows: &[(&[&str], Value)]) -> IndexSnapshot {
        let map = rows
            .iter()
            .map(|(key, value)| {
                (
                    key.iter().map(|part| (*part).to_owned()).collect(),
                    value.clone(),
                )
            })
            .collect();
        IndexSnapshot::new(stream, version, map).unwrap()
    }

    #[test]
    fn changed_binding_retracts_old_facets_and_updates_untouched_occurrence() {
        let key = &["report/13", "edition1", "0"][..];
        let old = snapshot(
            "public-search",
            5,
            &[
                (key, json!({"hit": "She", "catalog": ["A", "17"]})),
                (&["retired"], json!(3)),
            ],
        );
        let new = snapshot(
            "public-search",
            9,
            &[
                (key, json!({"hit": "She", "catalog": ["B", "17"]})),
                (&["new"], json!("unresolved")),
            ],
        );
        let delta = new.changes_since(&old).unwrap();
        assert!(delta.needs_apply(old.checkpoint()).unwrap());
        assert!(!delta.needs_apply(new.checkpoint()).unwrap());
        assert_eq!(delta.removed().len(), 2);
        assert_eq!(delta.upserted().len(), 2);
        // Apply the two lists to an independent materialized view.
        let mut local: BTreeMap<_, _> = old.rows().collect();
        for (key, old_value) in delta.removed() {
            assert_eq!(local.remove(key), Some(*old_value));
        }
        for (key, new_value) in delta.upserted() {
            assert!(local.insert(key, new_value).is_none());
        }
        assert_eq!(local, new.rows().collect());
        assert_eq!(delta.base(), old.checkpoint());
        assert_eq!(delta.target(), new.checkpoint());
        let wire = serde_json::to_vec(&(delta.base(), delta.target())).unwrap();
        let (base, target): (IndexCheckpoint, IndexCheckpoint) =
            serde_json::from_slice(&wire).unwrap();
        assert!(old.checkpoint().needs_delta(&base, &target).unwrap());
        assert!(!new.checkpoint().needs_delta(&base, &target).unwrap());
        assert!(matches!(
            old.checkpoint().needs_delta(&target, &base),
            Err(PublicationError::Stale)
        ));
    }

    #[test]
    fn consumer_needs_the_actual_predecessor_or_an_authoritative_snapshot() {
        let earlier = snapshot("catalog", 1, &[]);
        let before = snapshot("catalog", 2, &[(&["subject"], json!("old name"))]);
        let fork = snapshot("catalog", 2, &[(&["subject"], json!("other name"))]);
        let empty = snapshot("catalog", 3, &[]);
        let foreign = snapshot(
            "restricted-catalog",
            2,
            &[(&["subject"], json!("old name"))],
        );
        let delta = empty.changes_since(&before).unwrap();
        for invalid in [&earlier, &fork] {
            assert!(matches!(
                delta.needs_apply(invalid.checkpoint()),
                Err(PublicationError::WrongPredecessor)
            ));
            assert!(empty.needs_replace(invalid.checkpoint()).unwrap());
        }
        assert!(matches!(
            fork.changes_since(&before),
            Err(PublicationError::ConflictingRevision)
        ));
        assert!(matches!(
            before.changes_since(&empty),
            Err(PublicationError::Stale)
        ));
        assert!(matches!(
            delta.needs_apply(foreign.checkpoint()),
            Err(PublicationError::DifferentStream)
        ));
        assert!(matches!(
            empty.needs_replace(foreign.checkpoint()),
            Err(PublicationError::DifferentStream)
        ));
        assert_eq!(delta.removed().len(), 1);
        assert!(delta.upserted().is_empty());
        assert!(!empty.needs_replace(empty.checkpoint()).unwrap());
        assert!(matches!(
            before.needs_replace(empty.checkpoint()),
            Err(PublicationError::Stale)
        ));
    }

    #[test]
    fn content_fingerprint_scopes_keys_and_orders_json_object_fields() {
        let old = snapshot(
            "names",
            4,
            &[
                (&["a", "b/c"], json!({"aliases": [{"x": 1, "y": 2}]})),
                (&["a/b", "c"], json!("two")),
            ],
        );
        let again = snapshot(
            "names",
            4,
            &[
                (&["a/b", "c"], json!("two")),
                (&["a", "b/c"], json!({"aliases": [{"y": 2, "x": 1}]})),
            ],
        );
        assert_eq!(old.checkpoint(), again.checkpoint());
        let duplicate = again.changes_since(&old).unwrap();
        assert!(duplicate.removed().is_empty() && duplicate.upserted().is_empty());
        let saved = serde_json::to_vec(old.checkpoint()).unwrap();
        let restored: IndexCheckpoint = serde_json::from_slice(&saved).unwrap();
        assert!(!duplicate.needs_apply(&restored).unwrap());

        let sorted = snapshot("numbers", 1, &[(&["x"], json!([-0.0, 1]))]);
        let changed = snapshot("numbers", 2, &[(&["x"], json!([0.0, 1]))]);
        let delta = changed.changes_since(&sorted).unwrap();
        assert_eq!(delta.removed().len(), 1);
        assert_eq!(delta.upserted().len(), 1);
    }

    #[test]
    fn later_identical_contents_still_advance_the_checkpoint() {
        let first = snapshot("current-roster", 12, &[]);
        let later = snapshot("current-roster", 21, &[]);
        let delta = later.changes_since(&first).unwrap();
        assert!(delta.removed().is_empty() && delta.upserted().is_empty());
        assert!(delta.needs_apply(first.checkpoint()).unwrap());
        assert_eq!(delta.target().sequence(), 21);
        assert!(!delta.needs_apply(later.checkpoint()).unwrap());
        assert!(matches!(
            IndexSnapshot::new(" ", 0, BTreeMap::new()),
            Err(PublicationError::EmptyStream)
        ));
    }

    proptest! {
        #[test]
        fn sparse_delta_reconstructs_exact_target(
            original in prop::collection::btree_map(0u8..8, prop::option::of(0u8..4), 0..8),
            revised in prop::collection::btree_map(0u8..8, prop::option::of(0u8..4), 0..8),
        ) {
            // Explicit JSON null, absent keys, and unchanged neighboring rows coexist.
            let project = |input: BTreeMap<u8, Option<u8>>| -> BTreeMap<_, _> {
                input.into_iter().map(|(id, value)| (vec![id.to_string()], json!(value))).collect()
            };
            let mut replica = project(original);
            let expected = project(revised);
            let old = IndexSnapshot::new("property", 2, replica.clone()).unwrap();
            let new = IndexSnapshot::new("property", 7, expected.clone()).unwrap();
            let delta = new.changes_since(&old).unwrap();
            prop_assert!(delta.needs_apply(old.checkpoint()).unwrap());
            for &(key, old_value) in delta.removed() {
                prop_assert_ne!(expected.get(key), Some(old_value));
                let removed = replica.remove(key);
                prop_assert_eq!(removed.as_ref(), Some(old_value));
            }
            for &(key, new_value) in delta.upserted() {
                prop_assert!(replica.insert(key.to_vec(), new_value.clone()).is_none());
            }
            prop_assert_eq!(replica, expected);
            prop_assert!(!delta.needs_apply(new.checkpoint()).unwrap());
        }
    }
}
