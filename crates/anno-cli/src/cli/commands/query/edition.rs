// SPDX-License-Identifier: MIT OR Apache-2.0
//! SQLite subscriber for exact native query-publication receipts.
mod receipt;
mod store;

use anno::publication::{IndexCheckpoint, IndexEdition, IndexEditionScope};
use clap::{Args, Subcommand};
use receipt::Delivery;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};
use std::{
    error::Error,
    fs::File,
    io::{self, BufReader, Write},
    path::{Path, PathBuf},
};
use store::EditionStore;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Args, Debug)]
pub struct EditionArgs {
    /// Publisher-defined collection identifier.
    #[arg(long)]
    collection: String,
    /// Authority which selected these members and declared this release.
    #[arg(long)]
    source: String,
    /// Projection audience. The caller controls access to the database.
    #[arg(long)]
    audience: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Write a publisher-authored manifest from complete publication receipts.
    Manifest {
        edition: u64,
        snapshots: Vec<PathBuf>,
    },
    /// Stage an exact full snapshot or apply a delta to its retained base.
    Stage { database: PathBuf, receipt: PathBuf },
    /// Atomically expose this manifest if every exact member is available.
    Publish {
        database: PathBuf,
        manifest: PathBuf,
    },
    /// Read one complete edition. Omit the number for the latest published release.
    Read {
        database: PathBuf,
        /// Read exactly this published edition.
        edition: Option<u64>,
        /// Require the latest published edition in this scope to be at least this number.
        #[arg(long, conflicts_with = "edition")]
        min_edition: Option<u64>,
    },
    /// Drop retained rows for one checkpoint, supplied as JSON. The current edition is protected.
    Evict {
        database: PathBuf,
        checkpoint: PathBuf,
    },
}

fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    if path == Path::new("-") {
        Ok(serde_json::from_reader(io::stdin().lock())?)
    } else {
        Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
    }
}

fn execute(args: EditionArgs) -> Result<Value> {
    let scope = IndexEditionScope {
        collection: args.collection,
        source: args.source,
        audience: args.audience,
    };
    scope.validate()?;
    match args.command {
        Command::Manifest { edition, snapshots } => {
            let mut members = Vec::new();
            for path in snapshots {
                let mut snapshot: receipt::Snapshot = read(&path)?;
                snapshot.verify()?;
                members.push(snapshot.checkpoint);
            }
            Ok(serde_json::to_value(IndexEdition::new(
                scope, edition, members,
            )?)?)
        }
        Command::Stage { database, receipt } => {
            let delivery: Delivery = read(&receipt)?;
            Ok(json!({"checkpoint": EditionStore::open(&database, &scope)?.stage(delivery)?}))
        }
        Command::Publish { database, manifest } => {
            let manifest: IndexEdition = read(&manifest)?;
            if manifest.scope() != &scope {
                return Err("manifest scope differs from requested scope".into());
            }
            Ok(EditionStore::open(&database, &scope)?.publish(&manifest)?)
        }
        Command::Read {
            database,
            edition,
            min_edition,
        } => EditionStore::open_reader(&database, &scope)?.read(edition, min_edition),
        Command::Evict {
            database,
            checkpoint,
        } => {
            let cp: IndexCheckpoint = read(&checkpoint)?;
            Ok(json!({"evicted": EditionStore::open(&database, &scope)?.evict(&cp)?}))
        }
    }
}

pub(super) fn run(args: EditionArgs) -> std::result::Result<(), String> {
    let value = execute(args).map_err(|e| e.to_string())?;
    let mut out = io::stdout().lock();
    serde_json::to_writer(&mut out, &value).map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())
}
