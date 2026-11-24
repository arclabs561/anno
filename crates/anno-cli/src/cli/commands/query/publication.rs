//! Source-anchored catalog postings for one versioned grounded text document.

use super::super::super::utils::source_sha256;
use anno::publication::IndexSnapshot;
use anno::{GroundedDocument, Identity};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct PublicationArgs {
    /// JSON envelope with stream, sequence, document and optional source_revision. Use - for stdin.
    ///
    /// sequence orders complete index views. source_revision identifies the cited edition of
    /// the text independently of later annotation changes; omit it for a digest-only anchor.
    current: PathBuf,

    /// Compare with an earlier complete revision from the same stream and document.
    #[arg(long, value_name = "PREVIOUS")]
    since: Option<PathBuf>,

    /// Limit returned occurrences to this catalog and accession; each selection gets its own checkpoint.
    #[arg(long, num_args = 2, value_names = ["CATALOG", "ACCESSION"])]
    catalog_record: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct CatalogRecord {
    catalog: String,
    accession: String,
}

impl CatalogRecord {
    pub(super) fn from_args(args: &[String]) -> Result<Option<Self>, String> {
        match args {
            [] => Ok(None),
            [catalog, accession]
                if !catalog.trim().is_empty()
                    && !catalog.trim().eq_ignore_ascii_case("unknown")
                    && !accession.trim().is_empty() =>
            {
                Ok(Some(Self {
                    catalog: catalog.clone(),
                    accession: accession.clone(),
                }))
            }
            _ => Err("catalog-record requires a named catalog and a nonempty accession".into()),
        }
    }

    pub(super) fn as_ref(&self) -> (&str, &str) {
        (&self.catalog, &self.accession)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Revision {
    stream: String,
    sequence: u64,
    source_revision: Option<String>,
    document: GroundedDocument,
}

impl Revision {
    fn read(path: &Path) -> Result<Self, String> {
        let parsed = if path == Path::new("-") {
            serde_json::from_reader(io::stdin().lock())
        } else {
            let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            serde_json::from_reader(BufReader::new(file))
        };
        parsed.map_err(|e| format!("{}: invalid publication input: {e}", path.display()))
    }

    fn snapshot(&self, filter: Option<&CatalogRecord>) -> Result<(IndexSnapshot, String), String> {
        let doc = &self.document;
        if self.stream.trim().is_empty() || doc.id().trim().is_empty() {
            return Err("publication requires a nonempty stream and document ID".into());
        }

        if self
            .source_revision
            .as_ref()
            .is_some_and(|r| r.trim().is_empty())
        {
            return Err("a supplied source_revision must be nonempty".into());
        }

        // A tuple retains exact, case-sensitive domain, document and query boundaries,
        // even when any of their identifiers contain separators.
        let stream =
            json!(["anno.query.publication.v1", self.stream, doc.id(), filter]).to_string();
        let hash = source_sha256(doc.text());
        let mut rows = BTreeMap::new();
        // Bind empty documents to their source edition too. An empty key can never
        // collide with an occurrence key and is omitted from observed postings.
        rows.insert(
            Vec::new(),
            json!({"source_sha256": hash, "source_revision": self.source_revision}),
        );
        for signal in doc.signals() {
            let catalog_record = doc
                .identity_for_signal(signal.id)
                .and_then(Identity::qualified_kb_ref);
            if filter.is_some_and(|selection| Some(selection.as_ref()) != catalog_record) {
                continue;
            }
            let mut key = vec![doc.id().to_owned(), hash.clone()];
            key.extend(self.source_revision.iter().cloned());
            key.push(signal.id.to_string());
            rows.insert(key, json!({
                "document": doc.id(),
                "source_sha256": hash,
                "source_revision": self.source_revision,
                "signal_id": signal.id,
                "surface": signal.surface(),
                "location": signal.location,
                "label": signal.label(),
                "catalog_record": catalog_record.map(|(catalog, accession)| json!({ "catalog": catalog, "accession": accession })),
            }));
        }
        let snapshot =
            IndexSnapshot::new(stream, self.sequence, rows).map_err(|e| e.to_string())?;
        Ok((snapshot, hash))
    }
}

fn postings<'a>(entries: impl Iterator<Item = (&'a [String], &'a Value)>) -> Vec<&'a Value> {
    entries
        .filter(|(key, _)| !key.is_empty())
        .map(|(_, row)| row)
        .collect()
}

pub(super) fn run(args: PublicationArgs) -> Result<(), String> {
    if args.current == Path::new("-") && args.since.as_deref() == Some(Path::new("-")) {
        return Err("only one publication input may read stdin".into());
    }
    let filter = CatalogRecord::from_args(&args.catalog_record)?;
    let current_revision = Revision::read(&args.current)?;
    let (current, source_hash) = current_revision.snapshot(filter.as_ref())?;
    let result = if let Some(path) = args.since {
        let base_revision = Revision::read(&path)?;
        let (previous, base_hash) = base_revision.snapshot(filter.as_ref())?;
        let changes = current
            .changes_since(&previous)
            .map_err(|e| e.to_string())?;
        json!({
            "base": changes.base(),
            "target": changes.target(),
            "base_source_sha256": base_hash,
            "base_source_revision": base_revision.source_revision,
            "target_source_revision": current_revision.source_revision,
            "target_source_sha256": source_hash,
            "removed": postings(changes.removed().iter().copied()),
            "upserted": postings(changes.upserted().iter().copied()),
        })
    } else {
        json!({
            "checkpoint": current.checkpoint(),
            "source_sha256": source_hash,
            "source_revision": current_revision.source_revision,
            "rows": postings(current.rows()),
        })
    };
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &result).map_err(|e| e.to_string())?;
    writeln!(stdout).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use anno::{Entity, EntityType};

    fn revision(source_revision: &str) -> Revision {
        let doc = GroundedDocument::from_entity_signals(
            "doc",
            "Nia returned.",
            &[Entity::new("Nia", EntityType::Person, 0, 3, 1.0)],
        );
        Revision {
            stream: "roster".into(),
            sequence: 7,
            source_revision: Some(source_revision.into()),
            document: doc,
        }
    }

    #[test]
    fn changed_source_edition_replaces_the_old_citation_even_for_identical_words() {
        let first = revision("alpha").snapshot(None).unwrap().0;
        let mut second = revision("beta");
        assert!(second
            .snapshot(None)
            .unwrap()
            .0
            .changes_since(&first)
            .is_err());
        second.sequence += 1;
        let next = second.snapshot(None).unwrap().0;
        let changes = next.changes_since(&first).unwrap();
        let removed = postings(changes.removed().iter().copied());
        let added = postings(changes.upserted().iter().copied());
        assert_eq!(removed.len(), 1);
        assert_eq!(added.len(), 1);
        assert_eq!(removed[0]["source_revision"], "alpha");
        assert_eq!(added[0]["source_revision"], "beta");
        assert_eq!(removed[0]["surface"], added[0]["surface"]);
        assert_eq!(removed[0]["source_sha256"], added[0]["source_sha256"]);
    }

    #[test]
    fn empty_selections_have_independent_checkpoints() {
        let view = revision("alpha");
        let a = CatalogRecord::from_args(&["first".into(), "17".into()]).unwrap();
        let b = CatalogRecord::from_args(&["second".into(), "17".into()]).unwrap();
        let first = view.snapshot(a.as_ref()).unwrap().0;
        let second = view.snapshot(b.as_ref()).unwrap().0;
        assert!(postings(first.rows()).is_empty());
        assert!(postings(second.rows()).is_empty());
        assert_ne!(first.checkpoint(), second.checkpoint());
        assert!(first.changes_since(&second).is_err());
    }
}
