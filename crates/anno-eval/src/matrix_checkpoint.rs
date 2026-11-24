//! Build the trusted matrix artifact from the scheduling fields the next run reads.
use super::*;
use crate::eval::history::{EvalHistory, EvalHistoryEntry};
use crate::eval::task_evaluator::CorefPredictedMentions;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter};
use std::path::Path;

#[derive(Serialize, Deserialize)]
struct JournalRow {
    #[serde(flatten)]
    entry: EvalHistoryEntry,
    git_commit: Option<String>,
}

struct Catalog {
    backends: BTreeSet<&'static str>,
    tasks: BTreeSet<&'static str>,
    languages: BTreeSet<String>,
    domains: BTreeSet<String>,
}

impl Catalog {
    fn new() -> Self {
        let mut catalog = Self {
            backends: Task::all()
                .iter()
                .flat_map(|&t| get_task_backends(t))
                .collect(),
            tasks: Task::all().iter().map(Task::code).collect(),
            languages: ["mixed".into(), "unknown".into()].into(),
            domains: ["mixed".into(), "unknown".into()].into(),
        };
        for dataset in DatasetId::all() {
            let slice = mh::muxer_slice_tag("ner", &[*dataset], true).unwrap();
            let (_, facets) = slice.as_str().split_once(".lang=").unwrap();
            let (lang, domain) = facets.split_once(".dom=").unwrap();
            catalog.languages.insert(lang.into());
            catalog.domains.insert(domain.into());
        }
        catalog
    }

    fn slice(&self, value: &str) -> bool {
        if let Some((task, facets)) = value.split_once(".lang=") {
            return self.tasks.contains(task)
                && facets
                    .split_once(".dom=")
                    .is_some_and(|(l, d)| self.languages.contains(l) && self.domains.contains(d));
        }
        self.tasks.contains(value)
    }

    fn arm(&self, value: &str) -> bool {
        let (backend, dataset) = value
            .split_once("@@")
            .map_or((value, None), |(b, d)| (b, Some(d)));
        self.backends.contains(backend)
            && dataset.is_none_or(|name| {
                name.parse::<DatasetId>()
                    .is_ok_and(|dataset| format!("{dataset:?}") == name)
            })
    }

    fn identity(&self, e: &EvalHistoryEntry, t: Task, d: DatasetId) -> bool {
        let (Some(run), Some(id)) = (&e.muxer_run_id, &e.observation_id) else {
            return false;
        };
        let prefix = format!("seed={} slice=", e.seed);
        let Some((slice, strategy)) = run
            .strip_prefix(&prefix)
            .and_then(|s| s.split_once(" strategy="))
        else {
            return false;
        };
        let valid_strategy = matches!(strategy, "Random" | "WorstFirst" | "MlOnly" | "Estimate")
            || strategy
                .strip_prefix("fixed fixed=")
                .is_some_and(|b| b.split(',').all(|b| self.backends.contains(b)));
        let prefix = format!("{run}|task={t:?}|dataset={d:?}|backend={}|pid=", e.backend);
        let Some((pid, rest)) = id
            .strip_prefix(&prefix)
            .and_then(|s| s.split_once("|started_ns="))
        else {
            return false;
        };
        self.slice(slice)
            && valid_strategy
            && pid.parse::<u32>().is_ok()
            && rest.split_once("|seq=").is_some_and(|(time, seq)| {
                time.parse::<u128>().is_ok() && seq.parse::<u64>().is_ok()
            })
    }
}

fn require(ok: bool) -> Result<(), &'static str> {
    if ok {
        Ok(())
    } else {
        Err("unrecognized matrix checkpoint dimensions")
    }
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, &'static str> {
    serde_json::from_reader(File::open(path).map_err(|_| "unable to open matrix state")?)
        .map_err(|_| "invalid matrix state")
}

fn write(path: &Path, value: &impl Serialize) -> Result<(), &'static str> {
    serde_json::to_writer(
        File::create_new(path).map_err(|_| "unable to create matrix state")?,
        value,
    )
    .map_err(|_| "unable to serialize matrix state")
}

/// Only canonical producer-generated cohort keys may reach the artifact. The consumer compares
/// the original string byte for byte when deciding which cells have already been measured.
fn valid_cohort(value: &str, task: Task, dataset: DatasetId) -> bool {
    #[derive(Deserialize)]
    struct Inputs {
        max_examples: Option<usize>,
        cached_only: bool,
    }
    let Ok(c) = serde_json::from_str::<Inputs>(value) else {
        return false;
    };
    [
        (false, CorefPredictedMentions::BackendDefault),
        (true, CorefPredictedMentions::BackendDefault),
        (false, CorefPredictedMentions::NerWithEnglishPronouns),
    ]
    .into_iter()
    .any(|(gold, mentions)| {
        policy_cohort_key(
            task,
            dataset,
            0,
            c.max_examples,
            c.cached_only,
            gold,
            mentions,
        )
        .as_deref()
            == Some(value)
    })
}

fn project_journal(src: &Path, dest: &Path, catalog: &Catalog) -> Result<(), &'static str> {
    let src = File::open(src).map_err(|_| "missing evaluation journal")?;
    let mut out = BufWriter::new(File::create_new(dest).map_err(|_| "unable to create journal")?);
    // Diagnostic equality is useful to retry scheduling. No diagnostic bytes or content hashes
    // leave this process. Partition prior groups and new private messages separately.
    let mut diagnoses = BTreeMap::new();
    for line in BufReader::new(src).lines() {
        let mut row: JournalRow = serde_json::from_str(&line.map_err(|_| "unreadable journal")?)
            .map_err(|_| "invalid evaluation journal")?;
        let e = &mut row.entry;
        let dataset: DatasetId = e.dataset.parse().map_err(|_| "unknown journal dataset")?;
        let task = Task::all()
            .iter()
            .copied()
            .find(|t| format!("{t:?}") == e.task)
            .ok_or("unknown journal task")?;
        require(
            e.dataset == dataset.name()
                && catalog.backends.contains(e.backend.as_str())
                && catalog.identity(e, task, dataset)
                && e.policy_cohort_key
                    .as_deref()
                    .is_none_or(|k| valid_cohort(k, task, dataset)),
        )?;
        require(
            e.timestamp.parse::<jiff::Timestamp>().is_ok()
                && row.git_commit.as_ref().is_none_or(|s| {
                    (7..=40).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_hexdigit())
                }),
        )?;
        // Canonicalize the timestamp so only the measured instant survives.
        e.timestamp = e.timestamp.parse::<jiff::Timestamp>().unwrap().to_string();
        e.error = e.error.as_ref().map(|_| "evaluation_failed".into());
        if let Some(raw) = e.metadata.take() {
            let object: Value =
                serde_json::from_str(&raw).map_err(|_| "invalid evaluation receipt")?;
            let prior = object.get("checkpoint_retry_version") == Some(&json!(1));
            let r: TaskEvalResult =
                serde_json::from_value(object).map_err(|_| "invalid evaluation receipt")?;
            require(
                r.task == task
                    && r.dataset == dataset
                    && r.backend == e.backend
                    && r.seed == e.seed,
            )?;
            if backend_was_attempted(&r) {
                let diagnosis = r
                    .error
                    .as_deref()
                    .and_then(retry::infrastructure_diagnosis)
                    .map(|message| {
                        let next = diagnoses.len();
                        let group = diagnoses.entry((prior, message.to_owned())).or_insert(next);
                        if mh::classify_failure_kind(message) == "dataset" {
                            format!("failed to load dataset: checkpoint group {group:b}")
                        } else {
                            format!("failed to fetch checkpoint group {group:b}")
                        }
                    });
                // TaskEvalResult supports absent provenance in historical rows. This projection
                // preserves the fields matrix_retry reads and keeps skips out of its input.
                let metrics = r
                    .primary_f1()
                    .map_or_else(|| json!({}), |f1| json!({"f1": f1}));
                e.metadata = Some(
                    json!({"checkpoint_retry_version": 1, "task": task, "dataset": dataset,
                    "backend": e.backend, "success": r.success, "seed": e.seed,
                    "num_examples": r.num_examples, "metrics": metrics, "error": diagnosis})
                    .to_string(),
                );
            }
        }
        serde_json::to_writer(&mut out, &row).map_err(|_| "unable to serialize journal")?;
        writeln!(out).map_err(|_| "unable to write journal")?;
    }
    out.flush().map_err(|_| "unable to flush journal")
}

fn project(root: &Path, target: &Path) -> Result<(), &'static str> {
    let catalog = Catalog::new();
    let journal = target.join("eval-results.jsonl");
    project_journal(&root.join("eval-results.jsonl"), &journal, &catalog)?;
    // Rebuild in a clean directory. A SQLite backup could include arbitrary tables, old pages
    // or a second copy of the private error and metadata columns.
    EvalHistory::new(&journal)
        .and_then(|h| h.rebuild_index())
        .map_err(|_| "unable to rebuild checkpoint index")?;
    fs::remove_file(target.join("eval-history.db.lock"))
        .map_err(|_| "unable to remove index lock")?;
    let mut windows = 0;
    for path in fs::read_dir(root).map_err(|_| "unable to list state")? {
        let path = path.map_err(|_| "unable to access state")?;
        let name = path.file_name();
        let name = name.to_str().ok_or("invalid state filename")?;
        if name == "linucb_global_state.json" {
            let state: muxer::LinUcbState = read(&path.path())?;
            require(state.arms.keys().all(|k| catalog.arm(k)))?;
            write(&target.join(name), &state)?;
        } else if let Some(slice) = name
            .strip_prefix("muxer_history.")
            .and_then(|n| n.strip_suffix(".json"))
        {
            require(catalog.slice(slice))?;
            let state: BackendHistory = read(&path.path())?;
            require(
                state
                    .windows
                    .keys()
                    .chain(state.fail_kinds.keys())
                    .all(|k| catalog.arm(k))
                    && state.fail_kinds.values().flatten().flatten().all(|k| {
                        matches!(
                            k.as_str(),
                            "rate_limit"
                                | "timeout"
                                | "dataset"
                                | "io"
                                | "metric"
                                | "backend"
                                | "unknown"
                                | "low_signal"
                        )
                    })
                    && state
                        .exp3ix_state
                        .as_ref()
                        .is_none_or(|s| s.arms.iter().all(|k| catalog.arm(k)))
                    && state
                        .linucb_state
                        .as_ref()
                        .is_none_or(|s| s.arms.keys().all(|k| catalog.arm(k))),
            )?;
            write(&target.join(name), &state)?;
            windows += 1;
        }
    }
    require(windows != 0)
}

#[test]
#[ignore = "called by scripts/qa/matrix_checkpoint.py on a completed trusted run"]
fn export() {
    let source = std::env::var_os("ANNO_CHECKPOINT_SOURCE").expect("checkpoint source");
    let target = std::env::var_os("ANNO_CHECKPOINT_TARGET").expect("checkpoint target");
    project(Path::new(&source), Path::new(&target)).expect("project trusted matrix checkpoint");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(seq: u32, error: Option<&str>, skipped: bool) -> JournalRow {
        let mut r: TaskEvalResult = serde_json::from_value(json!({
            "task":Task::NER,"dataset":DatasetId::WikiGold,"backend":"heuristic","seed":42,
            "success":error.is_none(),"error":error,"metrics":{"strict_f1":1.0},"num_examples":1,
            "backend_display":"PRIVATE_SENTINEL"
        }))
        .unwrap();
        if skipped {
            r.provenance.schema_version = 2;
            r.provenance.runtime.scheduling = EvaluationScheduling::NotRun {
                reason: "capability_unavailable".into(),
            };
        }
        let mut entry = EvalHistoryEntry::from(&r);
        let run = "seed=42 slice=ner strategy=Random";
        entry.muxer_run_id = Some(run.into());
        entry.observation_id = Some(format!(
            "{run}|task=NER|dataset=WikiGold|backend=heuristic|pid=12|started_ns=23|seq={seq}"
        ));
        entry.policy_cohort_key = policy_cohort_key(
            r.task,
            r.dataset,
            r.seed,
            Some(max_examples_per_dataset()),
            true,
            matrix_uses_gold_mentions(),
            matrix_predicted_mentions(),
        );
        JournalRow {
            entry,
            git_commit: Some("abcdef1".into()),
        }
    }

    fn input(root: &Path, rows: &[JournalRow]) {
        let mut journal = File::create(root.join("eval-results.jsonl")).unwrap();
        for row in rows {
            serde_json::to_writer(&mut journal, &row).unwrap();
            writeln!(journal).unwrap();
        }
        fs::write(root.join("muxer_history.ner.json"), r#"{"version":3,"window_cap":32,"windows":{"heuristic@@WikiGold":{"cap":32,"buf":[]}},"unknown":"PRIVATE_SENTINEL"}"#).unwrap();
    }

    #[test]
    fn round_trip_preserves_native_retry_decisions_and_excludes_private_records() {
        let _lock = env_lock();
        let directory = tempfile::tempdir().unwrap();
        let loader = DatasetLoader::with_cache_dir(directory.path()).unwrap();
        fs::write(
            directory.path().join(DatasetId::WikiGold.cache_filename()),
            include_bytes!("../../../scripts/qa/fixtures/matrix-smoke.conll"),
        )
        .unwrap();
        let old = std::env::var_os("ANNO_EVAL_HISTORY");
        struct Restore(Option<std::ffi::OsString>);
        impl Drop for Restore {
            fn drop(&mut self) {
                match self.0.take() {
                    Some(previous) => std::env::set_var("ANNO_EVAL_HISTORY", previous),
                    None => std::env::remove_var("ANNO_EVAL_HISTORY"),
                }
            }
        }
        let _restore = Restore(old);
        let decisions = |root: &Path| {
            std::env::set_var("ANNO_EVAL_HISTORY", root.join("eval-results.jsonl"));
            (0..24).map(|seed| {
                let plan = retry::next_cell(&loader,true,seed,32,Some("WikiGold"),None,None).unwrap();
                json!({"first":plan.first,"deferred":plan.defer(&[Task::NER],Some(&[DatasetId::WikiGold]),"heuristic")})
            }).collect::<Vec<_>>()
        };
        let failed = Some("Failed to download tokenizer PRIVATE_SENTINEL");
        for (name, rows) in [
            (
                "failure",
                vec![receipt(0, failed, false), receipt(1, failed, true)],
            ),
            (
                "recovered",
                vec![receipt(0, failed, false), receipt(1, None, false)],
            ),
            (
                "confirmed",
                vec![receipt(0, failed, false), receipt(1, failed, false)],
            ),
            (
                "distinct",
                vec![
                    receipt(0, failed, false),
                    receipt(1, Some("Failed to download other PRIVATE_SENTINEL"), false),
                ],
            ),
            ("unavailable", vec![receipt(0, failed, true)]),
        ] {
            let raw = directory.path().join(name);
            fs::create_dir(&raw).unwrap();
            input(&raw, &rows);
            let db = rusqlite::Connection::open(raw.join("eval-history.db")).unwrap();
            db.execute("CREATE TABLE private(value TEXT)", []).unwrap();
            db.execute("INSERT INTO private VALUES ('PRIVATE_SENTINEL')", [])
                .unwrap();
            drop(db);
            let original = decisions(&raw);
            match name {
                "failure" | "distinct" => assert!(original
                    .iter()
                    .all(|d| d["first"]["reason"] == "outstanding_failure")),
                "confirmed" => {
                    assert!(original
                        .iter()
                        .any(|d| d["first"]["reason"] == "repeated_infrastructure"));
                    assert!(original.iter().any(|d| d["deferred"] == true));
                }
                _ => assert!(original.iter().all(|d| d["first"].is_null())),
            }
            let mut src = raw;
            for version in 1..=2 {
                let exported = directory.path().join(format!("{name}-{version}"));
                fs::create_dir(&exported).unwrap();
                project(&src, &exported).unwrap();
                for file in fs::read_dir(&exported).unwrap() {
                    let data = fs::read(file.unwrap().path()).unwrap();
                    assert!(!data
                        .windows(b"PRIVATE_SENTINEL".len())
                        .any(|w| w == b"PRIVATE_SENTINEL"));
                }
                let db = rusqlite::Connection::open(exported.join("eval-history.db")).unwrap();
                assert_eq!(db.query_row("SELECT COUNT(*) FROM eval_results WHERE git_commit='abcdef1' AND observation_id IS NOT NULL",[],|r|r.get::<_,usize>(0)).unwrap(),rows.len());
                assert_eq!(original, decisions(&exported), "{name} export {version}");
                src = exported;
            }
            if name == "confirmed" {
                let mut journal = fs::OpenOptions::new()
                    .append(true)
                    .open(src.join("eval-results.jsonl"))
                    .unwrap();
                serde_json::to_writer(&mut journal, &receipt(2, failed, false)).unwrap();
                writeln!(journal).unwrap();
                assert!(decisions(&src)
                    .iter()
                    .all(|d| d["first"]["reason"] == "outstanding_failure"));
            }
        }
    }

    #[test]
    fn private_strings_in_policy_keys_or_arm_names_fail_closed() {
        let catalog = Catalog::new();
        let mut row = receipt(0, None, false);
        let key = row.entry.policy_cohort_key.take().unwrap();
        let mut key: Value = serde_json::from_str(&key).unwrap();
        key["private"] = json!("PRIVATE_SENTINEL");
        row.entry.policy_cohort_key = Some(key.to_string());
        let raw = tempfile::tempdir().unwrap();
        let target = tempfile::tempdir().unwrap();
        input(raw.path(), &[row]);
        assert_eq!(
            project(raw.path(), target.path()),
            Err("unrecognized matrix checkpoint dimensions")
        );
        assert!(!catalog.arm("heuristic@@PRIVATE_SENTINEL") && !catalog.arm("PRIVATE_SENTINEL"));
        assert!(!catalog.slice("ner.lang=en.dom=PRIVATE_SENTINEL"));
    }
}
