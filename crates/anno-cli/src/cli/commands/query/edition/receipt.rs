// SPDX-License-Identifier: MIT OR Apache-2.0
//! Read the ordinary query-publication wire format, including its hidden source row.
use super::super::publication::{occurrence_key, source_marker};
use super::Result;
use anno::publication::{IndexCheckpoint, IndexSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum Delivery {
    Snapshot(Snapshot),
    Delta(Delta),
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Snapshot {
    pub checkpoint: IndexCheckpoint,
    pub source_sha256: String,
    pub source_revision: Option<String>,
    pub rows: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Delta {
    pub base: IndexCheckpoint,
    pub target: IndexCheckpoint,
    base_source_sha256: String,
    base_source_revision: Option<String>,
    target_source_sha256: String,
    target_source_revision: Option<String>,
    removed: Vec<Value>,
    upserted: Vec<Value>,
}

fn key(row: &Value, hash: &str, revision: Option<&str>) -> Result<Vec<String>> {
    if row["source_sha256"].as_str() != Some(hash)
        || row.get("source_revision") != Some(&serde_json::json!(revision))
    {
        return Err("posting source does not match the receipt".into());
    }
    Ok(occurrence_key(
        row["document"].as_str().ok_or("missing posting document")?,
        hash,
        revision,
        row["signal_id"].as_u64().ok_or("invalid signal ID")?,
    ))
}

impl Snapshot {
    fn postings(&self) -> Result<BTreeMap<Vec<String>, Value>> {
        let mut map = BTreeMap::new();
        for row in &self.rows {
            let key = key(row, &self.source_sha256, self.source_revision.as_deref())?;
            if map.insert(key, row.clone()).is_some() {
                return Err("duplicate posting key".into());
            }
        }
        Ok(map)
    }

    pub fn verify(&mut self) -> Result<()> {
        let mut rows = self.postings()?;
        rows.insert(
            vec![],
            source_marker(&self.source_sha256, self.source_revision.as_deref()),
        );
        let native =
            IndexSnapshot::new(self.checkpoint.stream(), self.checkpoint.sequence(), rows)?;
        if native.checkpoint() != &self.checkpoint {
            return Err("projected rows do not match the native checkpoint".into());
        }
        // Receipt arrays may arrive reordered. The public reader always returns
        // native key order, even when an equivalent snapshot is redelivered.
        self.rows = native
            .rows()
            .filter(|(key, _)| !key.is_empty())
            .map(|(_, row)| row.clone())
            .collect();
        Ok(())
    }
}

impl Delta {
    pub fn apply(self, before: Snapshot) -> Result<Snapshot> {
        before.checkpoint.needs_delta(&self.base, &self.target)?;
        if self.base_source_sha256 != before.source_sha256
            || self.base_source_revision != before.source_revision
        {
            return Err("delta base source does not match staged snapshot".into());
        }
        let mut rows = before.postings()?;
        for row in self.removed {
            let key = key(
                &row,
                &self.base_source_sha256,
                self.base_source_revision.as_deref(),
            )?;
            if rows.remove(&key).as_ref() != Some(&row) {
                return Err("delta retraction differs from the stored posting".into());
            }
        }
        for row in self.upserted {
            let key = key(
                &row,
                &self.target_source_sha256,
                self.target_source_revision.as_deref(),
            )?;
            if rows.insert(key, row).is_some() {
                return Err("duplicate or unretracted delta posting".into());
            }
        }
        Ok(Snapshot {
            checkpoint: self.target,
            source_sha256: self.target_source_sha256,
            source_revision: self.target_source_revision,
            rows: rows.into_values().collect(),
        })
    }
}
