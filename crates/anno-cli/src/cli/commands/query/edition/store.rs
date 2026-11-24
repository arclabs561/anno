// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    receipt::{Delivery, Snapshot},
    Result,
};
use anno::publication::{IndexCheckpoint, IndexEdition, IndexEditionScope};
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use serde_json::{json, Value};
use std::path::Path;

// Sequence and edition numbers cover all u64 values. Fixed-width decimal orders
// correctly in SQLite, including numbers beyond its signed INTEGER range.
fn ordinal(number: u64) -> String {
    format!("{number:020}")
}

pub(super) struct EditionStore {
    db: Connection,
    scope: String,
}

impl EditionStore {
    pub fn open(path: &Path, scope: &IndexEditionScope) -> Result<Self> {
        scope.validate()?;
        let db = Connection::open(path)?;
        db.execute_batch(
            "PRAGMA synchronous=FULL;
             CREATE TABLE IF NOT EXISTS edition_versions (
                 scope TEXT NOT NULL, stream TEXT NOT NULL, seq TEXT NOT NULL,
                 checkpoint BLOB NOT NULL, snapshot BLOB,
                 PRIMARY KEY(scope, stream, seq)
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS published_editions (
                 scope TEXT NOT NULL, edition TEXT NOT NULL, manifest BLOB NOT NULL,
                 PRIMARY KEY(scope, edition)
             ) WITHOUT ROWID;",
        )?;
        Ok(Self {
            db,
            scope: serde_json::to_string(scope)?,
        })
    }

    pub fn open_reader(path: &Path, scope: &IndexEditionScope) -> Result<Self> {
        scope.validate()?;
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        Ok(Self {
            db,
            scope: serde_json::to_string(scope)?,
        })
    }

    pub fn stage(&mut self, delivery: Delivery) -> Result<IndexCheckpoint> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut snapshot = match delivery {
            Delivery::Snapshot(snapshot) => snapshot,
            Delivery::Delta(delta) => {
                let base = member(&tx, &self.scope, &delta.base)?
                    .ok_or("delta base unavailable; stage a complete snapshot")?;
                delta.apply(base)?
            }
        };
        snapshot.verify()?;
        let cp = &snapshot.checkpoint;
        let wire = serde_json::to_vec(cp)?;
        let changed = tx.execute(
            "INSERT INTO edition_versions(scope, stream, seq, checkpoint, snapshot)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(scope,stream,seq) DO UPDATE SET snapshot=excluded.snapshot
             WHERE checkpoint=excluded.checkpoint",
            params![
                self.scope,
                cp.stream(),
                ordinal(cp.sequence()),
                wire,
                serde_json::to_vec(&snapshot)?
            ],
        )?;
        if changed != 1 {
            return Err("stream sequence already bound to a different checkpoint".into());
        }
        tx.commit()?;
        Ok(snapshot.checkpoint)
    }

    pub fn publish(&mut self, manifest: &IndexEdition) -> Result<Value> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let result = publish_in(&tx, &self.scope, manifest)?;
        tx.commit()?;
        Ok(result)
    }

    pub fn read(&mut self, edition: Option<u64>) -> Result<Value> {
        // Bind both the manifest and the member bodies to one SQLite read transaction.
        let tx = self.db.transaction()?;
        let release =
            published(&tx, &self.scope, edition)?.ok_or("requested edition unavailable")?;
        let mut members = Vec::with_capacity(release.members().len());
        for cp in release.members() {
            members.push(member(&tx, &self.scope, cp)?.ok_or_else(|| {
                format!(
                    "edition {} needs evicted member {} at {}; restore its complete snapshot",
                    release.edition(),
                    cp.stream(),
                    cp.sequence()
                )
            })?);
        }
        tx.commit()?;
        Ok(json!({"manifest":release,"members":members}))
    }

    pub fn evict(&mut self, cp: &IndexCheckpoint) -> Result<bool> {
        let tx = self
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if published(&tx, &self.scope, None)?.is_some_and(|release| release.members().contains(cp))
        {
            return Err("cannot evict a member of the current edition".into());
        }
        let count = tx.execute(
            "UPDATE edition_versions SET snapshot=NULL WHERE scope=?1 AND stream=?2 AND seq=?3
             AND checkpoint=?4 AND snapshot IS NOT NULL",
            params![
                self.scope,
                cp.stream(),
                ordinal(cp.sequence()),
                serde_json::to_vec(cp)?
            ],
        )?;
        tx.commit()?;
        Ok(count == 1)
    }
}

fn member(db: &Connection, scope: &str, cp: &IndexCheckpoint) -> Result<Option<Snapshot>> {
    let row: Option<(Vec<u8>, Option<Vec<u8>>)> = db.query_row(
        "SELECT checkpoint,snapshot FROM edition_versions WHERE scope=?1 AND stream=?2 AND seq=?3",
        params![scope, cp.stream(), ordinal(cp.sequence())],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional()?;
    let Some((key, bytes)) = row else {
        return Ok(None);
    };
    if serde_json::from_slice::<IndexCheckpoint>(&key)? != *cp {
        return Err("stored checkpoint conflicts with required member".into());
    }
    let Some(bytes) = bytes else { return Ok(None) };
    let mut snapshot: Snapshot = serde_json::from_slice(&bytes)?;
    if snapshot.checkpoint != *cp {
        return Err("stored receipt identifies a different checkpoint".into());
    }
    snapshot.verify()?;
    Ok(Some(snapshot))
}

fn published(db: &Connection, scope: &str, edition: Option<u64>) -> Result<Option<IndexEdition>> {
    let row: Option<(String, Vec<u8>)> = db.query_row(
        "SELECT edition,manifest FROM published_editions WHERE scope=?1 AND (?2 IS NULL OR edition=?2)
         ORDER BY edition DESC LIMIT 1",
        params![scope, edition.map(ordinal)], |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional()?;
    let Some((number, bytes)) = row else {
        return Ok(None);
    };
    let manifest: IndexEdition = serde_json::from_slice(&bytes)?;
    if ordinal(manifest.edition()) != number || serde_json::to_string(manifest.scope())? != scope {
        return Err("stored manifest does not match its edition and scope".into());
    }
    Ok(Some(manifest))
}

fn publish_in(
    db: &rusqlite::Transaction<'_>,
    scope: &str,
    manifest: &IndexEdition,
) -> Result<Value> {
    if serde_json::to_string(manifest.scope())? != scope {
        return Err("manifest scope differs from requested scope".into());
    }
    let current = published(db, scope, None)?;
    if let Some(head) = &current {
        if manifest.edition() <= head.edition() {
            if published(db, scope, Some(manifest.edition()))?.as_ref() == Some(manifest) {
                return Ok(json!({"status":"already_published", "current_edition":head.edition()}));
            }
            return Err("edition is stale or conflicts with a published manifest".into());
        }
    }
    let mut missing = Vec::new();
    for cp in manifest.members() {
        if member(db, scope, cp)?.is_none() {
            missing.push(cp);
        }
    }
    if !missing.is_empty() {
        return Ok(json!({"status":"waiting", "missing":missing,
            "current_edition":current.as_ref().map(IndexEdition::edition)}));
    }
    db.execute(
        "INSERT INTO published_editions(scope, edition, manifest) VALUES (?1,?2,?3)",
        params![
            scope,
            ordinal(manifest.edition()),
            serde_json::to_vec(manifest)?
        ],
    )?;
    Ok(json!({"status":"published","current_edition":manifest.edition()}))
}
