// SPDX-License-Identifier: MIT OR Apache-2.0
//! Read explicitly scoped source, credit and display assertions from the reader study.
//! Usage: `reader_answer path/to/cases.json`. Only the fixture holds source data;
//! the separately authored answer key is used by the verification runner.
//! With `--ask CASE_ID`, read one trusted JSON question from stdin and print only its answer.
use anno::citation::{TextCitation, TextEdition};
use anno::offset::CharRange;
use anno::GroundedDocument;
use serde::Deserialize;
use serde_json::{json, Value};
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
    withdraw: Vec<Selector>,
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
    Printed {
        start: usize,
        end: usize,
    },
    Reference,
    Credit {
        role: String,
        excerpt: Option<CharRange>,
    },
    Redirect {
        from: String,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Selector {
    Local(String),
    Qualified { issuer: String, id: String },
}

type Key<'a> = (&'a str, &'a str);
type Owners<'a> = BTreeMap<&'a str, BTreeSet<&'a str>>;

impl Selector {
    fn resolve<'a>(&'a self, owners: &Owners<'a>) -> Result<Option<Key<'a>>> {
        match self {
            Self::Qualified { issuer, id }
                if !issuer.trim().is_empty() && !id.trim().is_empty() =>
            {
                Ok(Some((issuer, id)))
            }
            Self::Local(id) if !id.trim().is_empty() => match owners.get(id.as_str()) {
                Some(set) if set.len() > 1 => {
                    Err(format!("ambiguous assertion {id}; specify issuer and id").into())
                }
                set => Ok(set
                    .and_then(|values| values.first())
                    .map(|&issuer| (issuer, id.as_str()))),
            },
            _ => Err("empty assertion selector".into()),
        }
    }
}

#[derive(Deserialize)]
struct Policy {
    revision: u64,
    hidden: Vec<Selector>,
    #[serde(default)]
    hidden_excerpts: Vec<Selector>,
}

struct Visibility<'a> {
    hidden: BTreeSet<Key<'a>>,
    hidden_excerpts: BTreeSet<Key<'a>>,
}

fn resolve_set<'a>(items: &'a [Selector], owners: &Owners<'a>) -> Result<BTreeSet<Key<'a>>> {
    items
        .iter()
        .filter_map(|item| item.resolve(owners).transpose())
        .collect()
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
    Credit {
        role: String,
    },
    Evidence {
        role: String,
        target: String,
    },
    #[serde(rename = "cited_credit")]
    CitedCredit {
        role: String,
        target: String,
    },
}

type Documents<'a> = BTreeMap<&'a str, (&'a str, GroundedDocument)>;
type Snapshot<'a> = BTreeMap<Key<'a>, Published<'a>>;

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
    Credit {
        role: &'a str,
        citation: Option<TextCitation>,
    },
    Redirect {
        from: &'a str,
    },
}

fn edition<'a>(scope: &str, documents: &'a Documents<'_>, source: &str) -> Result<TextEdition<'a>> {
    let (revision, document) = documents
        .get(source)
        .ok_or_else(|| format!("unknown source edition {source}"))?;
    Ok(TextEdition::new(scope, revision, document)?)
}

fn snapshots<'a>(
    case: &'a Case,
    documents: &Documents<'_>,
    owners: &mut Owners<'a>,
) -> Result<BTreeMap<u64, Snapshot<'a>>> {
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
        for item in &step.put {
            owners.entry(&item.id).or_default().insert(&item.issuer);
        }
        for selector in &step.withdraw {
            if let Some(key) = selector.resolve(owners)? {
                active.remove(&key);
            }
        }
        for item in &step.put {
            if [&item.id, &item.issuer, &item.target]
                .iter()
                .any(|s| s.trim().is_empty())
                || !documents.contains_key(item.source.as_str())
            {
                return Err(
                    format!("invalid assertion identifiers or source for {}", item.id).into(),
                );
            }
            // Freeze the edition and exact UTF-8 content digest at assertion time.
            let claim = match &item.claim {
                Claim::Printed { start, end } => PublishedClaim::Printed(
                    edition(&case.id, documents, &item.source)?
                        .cite(CharRange::from_raw(*start, *end))?,
                ),
                Claim::Reference => PublishedClaim::Reference,
                Claim::Credit { role, .. } if role.trim().is_empty() => {
                    return Err("credit requires a nonempty role".into());
                }
                Claim::Credit { role, excerpt } => PublishedClaim::Credit {
                    role,
                    citation: excerpt
                        .map(|range| -> Result<TextCitation> {
                            edition(&case.id, documents, &item.source)?
                                .cite(range)
                                .map_err(Into::into)
                        })
                        .transpose()?,
                },
                Claim::Redirect { from } if from.trim().is_empty() => {
                    return Err("redirect requires a nonempty origin".into());
                }
                Claim::Redirect { from } => PublishedClaim::Redirect { from },
            };
            active.insert(
                (item.issuer.as_str(), item.id.as_str()),
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
    policy: &Visibility<'_>,
    query: &Query,
) -> Result<Value> {
    let visible = active.iter().filter(|(id, _)| {
        matches!(query.audience, Audience::Staff) || !policy.hidden.contains(*id)
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
    let mut receipts = Vec::new();
    for (&(issuer, id), item) in visible {
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
            (Question::Credit { role }, PublishedClaim::Credit { role: claimed, .. })
                if role == claimed =>
            {
                values.insert(item.target.to_owned());
            }
            (Question::Evidence { role, target }, PublishedClaim::Credit { role: claimed, .. })
                if role == claimed && target == item.target =>
            {
                let unique = values.insert(id.to_owned());
                if !unique {
                    return Err(format!(
                        "ambiguous evidence id {id}; select an issuer or request cited_credit"
                    )
                    .into());
                }
            }
            (
                Question::CitedCredit { role, target },
                PublishedClaim::Credit {
                    role: claimed,
                    citation,
                },
            ) if role == claimed && target == item.target => {
                let may_quote = matches!(query.audience, Audience::Staff)
                    || !policy.hidden_excerpts.contains(&(issuer, id));
                let mut excerpt = Value::Null;
                if let Some(cited) = citation.as_ref().filter(|_| may_quote) {
                    let source = edition(&case.id, documents, item.source)?;
                    let quote = cited.open(Some(&source))?;
                    excerpt = match query.audience {
                        Audience::Public => json!({"quote":quote}),
                        Audience::Staff => json!({"quote":quote,"citation":cited}),
                    };
                }
                receipts.push(json!({"work":case.work,"issuer":issuer,"id":id,
                    "role":claimed,"target":item.target,"excerpt":excerpt}));
            }
            _ => {}
        }
    }
    Ok(if matches!(query.question, Question::CitedCredit { .. }) {
        json!(receipts)
    } else {
        json!(values)
    })
}

fn evaluate(case: &Case) -> Result<BTreeMap<&str, Value>> {
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
    let mut owners = Owners::new();
    let history = snapshots(case, &documents, &mut owners)?;
    let mut policies = BTreeMap::new();
    for policy in &case.policies {
        i64::try_from(policy.revision).map_err(|_| "policy revision outside reader range")?;
        let view = Visibility {
            hidden: resolve_set(&policy.hidden, &owners)?,
            hidden_excerpts: resolve_set(&policy.hidden_excerpts, &owners)?,
        };
        if policies.insert(policy.revision, view).is_some() {
            return Err(format!("duplicate policy revision {}", policy.revision).into());
        }
    }

    let mut answers = BTreeMap::new();
    for query in &case.queries {
        i64::try_from(query.at).map_err(|_| "content revision outside reader range")?;
        i64::try_from(query.policy).map_err(|_| "policy revision outside reader range")?;
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
    let ask = match (args.next(), args.next(), args.next()) {
        (None, None, None) => None,
        (Some(flag), Some(id), None) if flag == "--ask" => {
            Some(id.into_string().map_err(|_| "invalid case ID")?)
        }
        _ => return Err("usage: reader_answer <cases.json> [--ask CASE_ID]".into()),
    };
    let data: Fixture = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    if data.version != 1 {
        return Err("unsupported reader fixture version".into());
    }
    if let Some(id) = ask {
        let mut matched = data.cases.into_iter().filter(|case| case.id == id);
        let mut case = matched.next().ok_or("unknown case ID")?;
        if matched.next().is_some() {
            return Err("duplicate case ID".into());
        }
        let query: Query = serde_json::from_reader(io::stdin().lock())?;
        case.queries = vec![query];
        let values = evaluate(&case)?;
        let mut stdout = io::stdout().lock();
        serde_json::to_writer(&mut stdout, &values[case.queries[0].id.as_str()])?;
        writeln!(stdout)?;
        return Ok(());
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
