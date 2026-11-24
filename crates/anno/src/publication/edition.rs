// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{IndexCheckpoint, PublicationError};
use serde::{Deserialize, Serialize};

/// Publisher-defined release domain. Edition numbers are comparable only inside this scope.
/// The audience identifies a projection. Applications enforce access before serving it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexEditionScope {
    /// Name of the collection whose membership is being declared.
    pub collection: String,
    /// Authority publishing this collection.
    pub source: String,
    /// Audience for which the publisher prepared these projections.
    pub audience: String,
}

impl IndexEditionScope {
    /// Require each component while preserving its exact case and spelling.
    pub fn validate(&self) -> Result<(), PublicationError> {
        if [&self.collection, &self.source, &self.audience]
            .iter()
            .any(|s| s.trim().is_empty())
        {
            return Err(PublicationError::EmptyEditionScope);
        }
        Ok(())
    }
}

/// A complete release as declared by its publisher. Every checkpoint, including a
/// checkpoint for a zero-row selection, is required. An empty member list describes
/// a complete empty collection. Edition numbers may skip values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "EditionWire")]
pub struct IndexEdition {
    scope: IndexEditionScope,
    edition: u64,
    members: Vec<IndexCheckpoint>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EditionWire {
    scope: IndexEditionScope,
    edition: u64,
    members: Vec<IndexCheckpoint>,
}

impl TryFrom<EditionWire> for IndexEdition {
    type Error = PublicationError;
    fn try_from(value: EditionWire) -> Result<Self, Self::Error> {
        Self::new(value.scope, value.edition, value.members)
    }
}

impl IndexEdition {
    /// Pin the exact native checkpoint of each required projection, in stream order.
    pub fn new(
        scope: IndexEditionScope,
        edition: u64,
        mut members: Vec<IndexCheckpoint>,
    ) -> Result<Self, PublicationError> {
        scope.validate()?;
        members.sort_by(|a, b| a.stream().cmp(b.stream()));
        for (i, member) in members.iter().enumerate() {
            if member.stream().trim().is_empty() {
                return Err(PublicationError::EmptyStream);
            }
            if i > 0 && members[i - 1].stream() == member.stream() {
                return Err(PublicationError::DuplicateMember(member.stream().into()));
            }
        }
        Ok(Self {
            scope,
            edition,
            members,
        })
    }

    /// Collection, publishing source and audience.
    pub fn scope(&self) -> &IndexEditionScope {
        &self.scope
    }

    /// Publisher-assigned edition within this scope.
    pub fn edition(&self) -> u64 {
        self.edition
    }

    /// Complete, ordered set of exact required checkpoints.
    pub fn members(&self) -> &[IndexCheckpoint] {
        &self.members
    }
}
