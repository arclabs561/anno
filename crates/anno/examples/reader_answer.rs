// SPDX-License-Identifier: MIT OR Apache-2.0
//! Read explicitly scoped source, credit and display assertions from the reader study.
//! Usage: `reader_answer path/to/cases.json`. Only the fixture holds source data;
//! the separately authored answer key is used by the verification runner.
use anno::citation::{TextCitation, TextEdition};
use anno::offset::CharRange;
use anno::GroundedDocument;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, Write};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Fixture {
    version: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    work: String,
    sources: Vec<Source>,
    steps: Vec<Step>,
    policies: Vec<Policy>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Source {
    id: String,
    edition: String,
    text: String,
}

#[derive(Deserialize)]
struct Step {
    revision: u64,
    #[serde(default)]
    put: Vec<Assertion>,
    #[serde(default)]
    withdraw: Vec<String>,
}

#[derive(Deserialize)]
struct Assertion {
    id: String,
    issuer: String,
    source: String,
    target: String,
    #[serde(flatten)]
    claim: Claim,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Claim {
    Printed { start: usize, end: usize },
    Reference,
    Credit { role: String },
    Redirect { from: String },
}

#[derive(Deserialize)]
struct Policy {
    revision: u64,
    hidden: BTreeSet<String>,
}

#[derive(Deserialize)]
struct Query {
    id: String,
    at: u64,
    policy: u64,
    audience: Audience,
    source: Option<String>,
    issuer: Option<String>,
    #[serde(flatten)]
    question: Question,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Audience {
    Public,
    Staff,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Question {
    Printed,
    Reference,
    Credit { role: String },
    Evidence { role: String, target: String },
}

type Documents<'a> = BTreeMap<&'a str, (&'a str, GroundedDocument)>;
type Snapshot<'a> = BTreeMap<&'a str, Published<'a>>;

#[derive(Clone)]
struct Published<'a> {
    issuer: &'a str,
    source: &'a str,
    target: &'a str,
    claim: PublishedClaim<'a>,
}

#[derive(Clone)]
enum PublishedClaim<'a> {
    Printed(TextCitation),
    Reference,
    Credit { role: &'a str },
    Redirect { from: &'a str },
}

fn edition<'a>(scope: &str, documents: &'a Documents<'_>, source: &str) -> Result<TextEdition<'a>> {
    let (revision, document) = documents
        .get(source)
        .ok_or_else(|| format!("unknown source edition {source}"))?;
    Ok(TextEdition::new(scope, revision, document)?)
}

fn snapshots<'a>(case: &'a Case, documents: &Documents<'_>) -> Result<BTreeMap<u64, Snapshot<'a>>> {
    let mut ordered = BTreeMap::new();
    for step in &case.steps {
        if ordered.insert(step.revision, step).is_some() {
            return Err(format!("duplicate content revision {}", step.revision).into());
        }
    }
    let mut history = BTreeMap::new();
    let mut active = Snapshot::new();
    for (expected, (revision, step)) in (1..).zip(ordered) {
        if revision != expected {
            return Err(
                format!("missing content patch {expected} before revision {revision}").into(),
            );
        }
        for id in &step.withdraw {
            active.remove(id.as_str());
        }
        for item in &step.put {
            if item.issuer.trim().is_empty() || !documents.contains_key(item.source.as_str()) {
                return Err(format!("missing issuer or source for {}", item.id).into());
            }
            // Freeze the edition and exact UTF-8 content digest at assertion time.
            let claim = match &item.claim {
                Claim::Printed { start, end } => PublishedClaim::Printed(
                    edition(&case.id, documents, &item.source)?
                        .cite(CharRange::from_raw(*start, *end))?,
                ),
                Claim::Reference => PublishedClaim::Reference,
                Claim::Credit { role } => PublishedClaim::Credit { role },
                Claim::Redirect { from } => PublishedClaim::Redirect { from },
            };
            active.insert(
                &item.id,
                Published {
                    issuer: &item.issuer,
                    source: &item.source,
                    target: &item.target,
                    claim,
                },
            );
        }
        history.insert(revision, active.clone());
    }
    Ok(history)
}

fn answer(
    case: &Case,
    documents: &Documents<'_>,
    active: &Snapshot<'_>,
    policy: &Policy,
    query: &Query,
) -> Result<BTreeSet<String>> {
    let visible = active.iter().filter(|(id, _)| {
        matches!(query.audience, Audience::Staff) || !policy.hidden.contains(**id)
    });
    // A different source or issuer can document the redirect. Select visible
    // navigation evidence before filtering the originating assertions.
    let mut redirects = BTreeMap::new();
    for (_, item) in visible.clone() {
        if let PublishedClaim::Redirect { from } = item.claim {
            if redirects.insert(from, item.target).is_some() {
                return Err(format!("multiple visible redirects from {from}").into());
            }
        }
    }
    let mut values = BTreeSet::new();
    for (id, item) in visible {
        if query
            .source
            .as_deref()
            .is_some_and(|source| source != item.source)
            || query
                .issuer
                .as_deref()
                .is_some_and(|issuer| issuer != item.issuer)
        {
            continue;
        }
        match (&query.question, &item.claim) {
            (Question::Printed, PublishedClaim::Printed(cited)) => {
                let source = edition(&case.id, documents, item.source)?;
                values.insert(cited.open(Some(&source))?.to_owned());
            }
            (Question::Reference, PublishedClaim::Reference) => {
                values.insert(
                    redirects
                        .get(item.target)
                        .copied()
                        .unwrap_or(item.target)
                        .to_owned(),
                );
            }
            (Question::Credit { role }, PublishedClaim::Credit { role: claimed })
                if role == claimed =>
            {
                values.insert(item.target.to_owned());
            }
            (Question::Evidence { role, target }, PublishedClaim::Credit { role: claimed })
                if role == claimed && target == item.target =>
            {
                values.insert((*id).to_owned());
            }
            _ => {}
        }
    }
    Ok(values)
}

fn evaluate(case: &Case) -> Result<BTreeMap<&str, BTreeSet<String>>> {
    if case.work.trim().is_empty() {
        return Err("expected one named work per case".into());
    }
    let mut documents = Documents::new();
    for source in &case.sources {
        let doc = GroundedDocument::new(&source.id, &source.text);
        if documents
            .insert(&source.id, (&source.edition, doc))
            .is_some()
        {
            return Err(format!("source id {} must name one immutable edition", source.id).into());
        }
    }
    let history = snapshots(case, &documents)?;
    let mut policies = BTreeMap::new();
    for policy in &case.policies {
        if policies.insert(policy.revision, policy).is_some() {
            return Err(format!("duplicate policy revision {}", policy.revision).into());
        }
    }

    let mut answers = BTreeMap::new();
    for query in &case.queries {
        let state = history
            .get(&query.at)
            .ok_or("content revision unavailable")?;
        let policy = policies
            .get(&query.policy)
            .ok_or("policy revision unavailable")?;
        let value = answer(case, &documents, state, policy, query)?;
        if answers.insert(query.id.as_str(), value).is_some() {
            return Err(format!("duplicate query id {}", query.id).into());
        }
    }
    Ok(answers)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("usage: reader_answer <cases.json>")?;
    if args.next().is_some() {
        return Err("usage: reader_answer <cases.json>".into());
    }
    let data: Fixture = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    if data.version != 1 {
        return Err("unsupported reader fixture version".into());
    }
    let mut answers = BTreeMap::new();
    for case in &data.cases {
        if answers.insert(&case.id, evaluate(case)?).is_some() {
            return Err(format!("duplicate case id {}", case.id).into());
        }
    }
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(
        &mut stdout,
        &serde_json::json!({"version": data.version, "answers": answers}),
    )?;
    writeln!(stdout)?;
    Ok(())
}
