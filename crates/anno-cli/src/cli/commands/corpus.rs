// SPDX-License-Identifier: MIT OR Apache-2.0
//! A durable local corpus. The ordered SQLite journal is the authoritative state.
//! Recording an accepted operation commits its effects and its receipt together.
use anno::core::grounded::{
    Corpus, DeliveryError, DeliveryOutcome, DeliveryReview, DocumentDelivery, KbAssignmentPreview,
    TrackRef,
};
use clap::{Args, Subcommand};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io, path::PathBuf, time::Duration};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Args)]
pub struct CorpusArgs {
    /// Local SQLite journal. Receipts and document versions are retained for its lifetime.
    #[arg(long)]
    store: PathBuf,
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Describe an operation and show any scope that requires approval.
    Preview(OperationFiles),
    /// Apply a new delivery, or commit a previously inspected review.
    Apply {
        #[command(flatten)]
        files: OperationFiles,
        /// Exact approval token returned by preview for this operation and corpus head.
        #[arg(long)]
        approval: Option<String>,
    },
    /// Print a current grounded document using the document_id in a delivery receipt.
    Read { document_id: String },
}

#[derive(Args)]
struct OperationFiles {
    /// JSON delivery or assignment operation.
    operation: PathBuf,
    /// Exact grounded document bytes. Required for a delivery.
    #[arg(long)]
    document: Option<PathBuf>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Operation {
    Delivery {
        request: DocumentDelivery,
    },
    Assignment {
        trigger: TrackRef,
        kb_name: String,
        kb_id: String,
        canonical_name: String,
    },
}

impl OperationFiles {
    fn read(&self) -> Result<(Operation, Vec<u8>)> {
        let operation = serde_json::from_slice(&fs::read(&self.operation)?)?;
        let payload = match (&operation, &self.document) {
            (Operation::Assignment { .. }, None) => Vec::new(),
            (Operation::Delivery { .. }, Some(path)) => fs::read(path)?,
            _ => return Err("pass --document only and always with a delivery".into()),
        };
        Ok((operation, payload))
    }
}

/// Preparing a fresh delivery changes only the temporary, replayed corpus.
enum Prepared {
    Delivery(DeliveryOutcome),
    Replacement(Box<DeliveryReview>),
    Assignment(Box<KbAssignmentPreview>),
}

impl Operation {
    fn prepare(&self, corpus: &mut Corpus, payload: &[u8]) -> Result<Prepared> {
        match self {
            Self::Delivery { request } => match corpus.deliver_document(request, payload) {
                Ok(outcome) => Ok(Prepared::Delivery(outcome)),
                Err(DeliveryError::ReviewRequired(review)) => Ok(Prepared::Replacement(review)),
                Err(error) => Err(error.into()),
            },
            Self::Assignment {
                trigger,
                kb_name,
                kb_id,
                canonical_name,
            } => Ok(Prepared::Assignment(Box::new(
                corpus.preview_kb_assignment(trigger, kb_name, kb_id, canonical_name)?,
            ))),
        }
    }
}

impl Prepared {
    fn proposal(&self) -> Value {
        match self {
            Self::Delivery(DeliveryOutcome::Applied(receipt)) => {
                json!({"status":"would_apply","receipt":receipt})
            }
            Self::Delivery(DeliveryOutcome::PreviouslyApplied(receipt)) => {
                json!({"status":"previously_applied","receipt":receipt})
            }
            Self::Replacement(review) => {
                json!({"status":"review_required","scope":review.summary()})
            }
            Self::Assignment(review) => {
                json!({"status":"review_required","scope":review.assignment()})
            }
        }
    }

    fn commit(self, corpus: &mut Corpus, approved: bool) -> Result<Value> {
        let receipt = match self {
            Self::Delivery(DeliveryOutcome::Applied(receipt)) => serde_json::to_value(receipt)?,
            Self::Delivery(DeliveryOutcome::PreviouslyApplied(_)) => {
                return Err("duplicate delivery in journal".into())
            }
            Self::Replacement(review) if approved => {
                serde_json::to_value(corpus.accept_document_replacement(*review)?.receipt())?
            }
            Self::Assignment(review) if approved => {
                serde_json::to_value(corpus.apply_kb_assignment(*review)?)?
            }
            _ => return Err("review required; preview the operation and pass its approval".into()),
        };
        Ok(json!({"status":"applied","receipt":receipt}))
    }
}

fn operation_key(
    head: &str,
    operation: &Operation,
    payload: &[u8],
    proposal: &Value,
) -> Result<String> {
    let material = serde_json::to_string(&(
        "anno.corpus.journal.v1",
        head,
        operation,
        Sha256::digest(payload).to_vec(),
        proposal,
    ))?;
    Ok(crate::cli::utils::source_sha256(&material))
}

fn replay(tx: &rusqlite::Transaction<'_>) -> Result<(Corpus, String)> {
    let (mut head, committed_head): (String, String) = tx.query_row(
        "SELECT origin,coalesce(head,origin) FROM corpus_origin WHERE id=1",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let mut corpus = Corpus::new();
    let mut statement = tx.prepare(
        "SELECT operation, payload, receipt, token, proposal FROM corpus_events ORDER BY position",
    )?;
    let mut rows = statement.query([])?;
    while let Some(row) = rows.next()? {
        let operation: Operation = serde_json::from_str(&row.get::<_, String>(0)?)?;
        let payload = row.get::<_, Vec<u8>>(1)?;
        let prior: Value = serde_json::from_str(&row.get::<_, String>(2)?)?;
        let prepared = operation.prepare(&mut corpus, &payload)?;
        let proposal = prepared.proposal();
        let recorded: Value = serde_json::from_str(&row.get::<_, String>(4)?)?;
        if proposal != recorded {
            return Err("stored review scope differs from the current policy".into());
        }
        let key = operation_key(&head, &operation, &payload, &proposal)?;
        if key != row.get::<_, String>(3)? {
            return Err("stored corpus operation does not match its predecessor or payload".into());
        }
        if prepared.commit(&mut corpus, true)? != prior {
            return Err("stored receipt differs from the replayed change".into());
        }
        head = key;
    }
    if head != committed_head {
        return Err("corpus history does not reach its committed head".into());
    }
    Ok((corpus, head))
}

fn execute(args: CorpusArgs) -> Result<Value> {
    // Read the request before taking the database lock.
    let request = match &args.action {
        Action::Preview(files) | Action::Apply { files, .. } => Some(files.read()?),
        Action::Read { .. } => None,
    };
    let reading = matches!(args.action, Action::Read { .. });
    let mut db = if reading {
        Connection::open_with_flags(args.store, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?
    } else {
        Connection::open(args.store)?
    };
    db.busy_timeout(Duration::from_secs(15))?;
    if !reading && !db.table_exists(None, "corpus_origin")? {
        db.execute_batch("BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS corpus_origin(id INTEGER PRIMARY KEY CHECK(id=1), origin TEXT NOT NULL, head TEXT);
        INSERT OR IGNORE INTO corpus_origin(id,origin) VALUES (1,lower(hex(randomblob(32))));
        CREATE TABLE IF NOT EXISTS corpus_events(position INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE,
            operation TEXT NOT NULL, payload BLOB NOT NULL, receipt TEXT NOT NULL, proposal TEXT NOT NULL);
        COMMIT;")?;
    }
    let apply = matches!(args.action, Action::Apply { .. });
    let behavior = if apply {
        TransactionBehavior::Immediate
    } else {
        TransactionBehavior::Deferred
    };
    let tx = db.transaction_with_behavior(behavior)?;
    let (mut corpus, head) = replay(&tx)?;
    if let Action::Read { document_id } = &args.action {
        let document = corpus
            .get_document(document_id)
            .ok_or("unknown corpus document")?;
        return Ok(serde_json::to_value(document)?);
    }
    let (operation, payload) = request.ok_or("missing corpus operation")?;
    let index = serde_json::to_string(&operation)?;
    let approval = match &args.action {
        Action::Apply { approval, .. } => approval.as_ref(),
        _ => None,
    };
    if let Some(approval) = approval {
        // An acknowledged review may itself be retried after its target was replaced.
        let previous: Option<String> = tx
            .query_row(
                "SELECT receipt FROM corpus_events WHERE token=? AND operation=? AND payload=?",
                params![approval, index, payload],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(text) = previous {
            let mut value: Value = serde_json::from_str(&text)?;
            value["status"] = json!("previously_applied");
            return Ok(value);
        }
    }
    let prepared = operation.prepare(&mut corpus, &payload)?;
    let proposal = prepared.proposal();
    if matches!(
        prepared,
        Prepared::Delivery(DeliveryOutcome::PreviouslyApplied(_))
    ) {
        if apply {
            return Ok(proposal);
        }
        let original: String = tx.query_row(
            "SELECT token FROM corpus_events WHERE operation=? AND payload=?",
            params![index, payload],
            |r| r.get(0),
        )?;
        return Ok(json!({"head":head,"approval":original,"proposal":proposal}));
    }
    let token = operation_key(&head, &operation, &payload, &proposal)?;
    if !apply {
        return Ok(json!({"head":head, "approval":token, "proposal":proposal}));
    }
    if approval.is_some_and(|a| a != &token) {
        return Err("stale, foreign or mismatched corpus approval; preview again".into());
    }
    let result = prepared.commit(&mut corpus, approval.is_some())?;
    tx.execute(
        "INSERT INTO corpus_events(token,operation,payload,receipt,proposal) VALUES(?,?,?,?,?)",
        params![
            token,
            index,
            payload,
            serde_json::to_string(&result)?,
            serde_json::to_string(&proposal)?
        ],
    )?;
    tx.execute("UPDATE corpus_origin SET head=? WHERE id=1", [&token])?;
    #[cfg(test)]
    if std::env::var_os("ANNO_TEST_EXIT_WITH_OPEN_CORPUS_TRANSACTION").is_some() {
        std::process::exit(75);
    }
    tx.commit()?;
    Ok(result)
}

pub fn run(args: CorpusArgs) -> std::result::Result<(), String> {
    let value = execute(args).map_err(|e| e.to_string())?;
    serde_json::to_writer(io::stdout().lock(), &value).map_err(|e| e.to_string())?;
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use tempfile::tempdir;

    fn deliver_in(path: &std::path::Path) -> Result<Value> {
        execute(CorpusArgs {
            store: path.join("corpus.sqlite"),
            action: Action::Apply {
                files: OperationFiles {
                    operation: path.join("request.json"),
                    document: Some(path.join("doc.json")),
                },
                approval: None,
            },
        })
    }

    // Invoked in a new test process so abrupt death discards the real writer connection.
    #[test]
    fn crash_child() {
        if let Some(directory) = std::env::var_os("ANNO_TEST_EXIT_WITH_OPEN_CORPUS_TRANSACTION") {
            deliver_in(std::path::Path::new(&directory)).unwrap();
            panic!("expected process death before commit");
        }
    }

    #[test]
    fn interrupted_writer_can_be_retried() -> Result<()> {
        let temp = tempdir()?;
        let p = temp.path();
        fs::write(p.join("request.json"), br#"{"kind":"delivery","request":{"issuer":"publisher","delivery_id":"msg-1","source_revision":1,"ownership":{"mode":"independent"}}}"#)?;
        fs::write(
            p.join("doc.json"),
            br#"{"id":"doc","text":"","signals":[],"tracks":{},"identities":{}}"#,
        )?;
        let status = Command::new(std::env::current_exe()?)
            .args(["--exact", "cli::commands::corpus::tests::crash_child"])
            .env("ANNO_TEST_EXIT_WITH_OPEN_CORPUS_TRANSACTION", p)
            .status()?;
        assert_eq!(status.code(), Some(75));
        let db = Connection::open(p.join("corpus.sqlite"))?;
        assert_eq!(
            db.query_row("SELECT count(*) FROM corpus_events", [], |r| r
                .get::<_, i64>(0))?,
            0
        );
        assert_eq!(deliver_in(p)?["status"], "applied");
        assert_eq!(deliver_in(p)?["status"], "previously_applied");
        Ok(())
    }
}
