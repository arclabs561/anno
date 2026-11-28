//! CPU inference with the optional `maverick-coref` Python runtime. One worker
//! retains its model across documents. Supply prepared MES safetensors or an official Hub ID prepared in the local cache.
use super::{CorefBackend, CorefCluster};
use crate::backends::python_worker::Worker;
use crate::{Error, Result};
use parking_lot::Mutex;
use serde::Deserialize;
use std::ffi::{OsStr, OsString};
use std::process::Command;
use std::time::{Duration, Instant};

/// A persistent Maverick worker. Install `maverick-coref==1.0.7` in the selected
/// Python environment. `singletons` is useful with LitBank and PreCo weights.
pub struct MaverickCoref {
    python: OsString,
    checkpoint: OsString,
    startup_timeout: Duration,
    request_timeout: Duration,
    worker: Mutex<Option<Worker>>,
    singletons: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Reply {
    Clusters(Vec<Vec<(usize, usize)>>),
    Error(String),
}

fn start_worker(python: &OsStr, checkpoint: &OsStr, timeout: Duration) -> Result<Worker> {
    let mut command = Command::new(python);
    command
        .args(["-u", "-c", include_str!("maverick_worker.py")])
        .arg(checkpoint)
        .arg(include_str!("maverick_models.json"));
    Worker::start(command, "Maverick", timeout)
}

impl MaverickCoref {
    /// Start Python and wait until the model and tokenizer are ready.
    pub fn new(
        python: impl AsRef<OsStr>,
        checkpoint: impl AsRef<OsStr>,
        singletons: bool,
    ) -> Result<Self> {
        Self::with_timeouts(
            python,
            checkpoint,
            singletons,
            Duration::from_secs(900),
            Duration::from_secs(300),
        )
    }

    /// Limit each model startup and request separately. Request time includes
    /// queueing behind concurrent callers and writing the input. A failed worker
    /// is reaped and replaced on the next call; its document is never replayed.
    pub fn with_timeouts(
        python: impl AsRef<OsStr>,
        checkpoint: impl AsRef<OsStr>,
        singletons: bool,
        startup_timeout: Duration,
        request_timeout: Duration,
    ) -> Result<Self> {
        if startup_timeout.is_zero() || request_timeout.is_zero() {
            return Err(Error::invalid_input("Maverick timeouts must be positive"));
        }
        let python = python.as_ref().to_owned();
        let checkpoint = checkpoint.as_ref().to_owned();
        let worker = start_worker(&python, &checkpoint, startup_timeout)?;
        Ok(Self {
            python,
            checkpoint,
            startup_timeout,
            request_timeout,
            worker: Mutex::new(Some(worker)),
            singletons,
        })
    }

    /// Cluster caller-supplied mention spans in half-open character offsets.
    /// Each span must align with the worker's English tokenizer.
    pub fn resolve_mentions(
        &self,
        text: &str,
        mentions: &[(usize, usize)],
    ) -> Result<Vec<CorefCluster>> {
        self.predict(text, Some(mentions))
    }

    fn predict(
        &self,
        text: &str,
        mentions: Option<&[(usize, usize)]>,
    ) -> Result<Vec<CorefCluster>> {
        let request =
            serde_json::json!({"text": text, "mentions": mentions, "singletons": self.singletons})
                .to_string();
        let started = Instant::now();
        let mut slot = self
            .worker
            .try_lock_for(self.request_timeout)
            .ok_or_else(|| Error::inference("timed out waiting for the Maverick worker"))?;
        let remaining = self.request_timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(Error::inference(
                "timed out waiting for the Maverick worker",
            ));
        }
        if !slot.as_mut().is_some_and(|w| w.running()) {
            *slot = None;
            *slot = Some(start_worker(
                &self.python,
                &self.checkpoint,
                self.startup_timeout,
            )?);
        }
        match slot
            .as_ref()
            .expect("initialized Maverick worker")
            .request(request, remaining)
        {
            Err(error) => {
                // A late or partial reply must never be read as the next document's result.
                *slot = None;
                Err(error)
            }
            Ok(Reply::Error(error)) => Err(Error::inference(error)),
            Ok(Reply::Clusters(groups)) => {
                drop(slot);
                align_clusters(text, groups)
            }
        }
    }
}

fn align_clusters(text: &str, groups: Vec<Vec<(usize, usize)>>) -> Result<Vec<CorefCluster>> {
    let bytes = crate::offset::build_char_to_byte_map(text);
    groups
        .into_iter()
        .enumerate()
        .map(|(index, spans)| {
            let mut mentions = Vec::with_capacity(spans.len());
            for &(start, end) in &spans {
                if start >= end || end >= bytes.len() {
                    return Err(Error::inference(format!(
                        "invalid Maverick character range [{start}, {end})"
                    )));
                }
                mentions.push(text[bytes[start]..bytes[end]].to_owned());
            }
            let canonical = mentions
                .iter()
                .max_by_key(|s| s.chars().count())
                .ok_or_else(|| Error::inference("empty Maverick cluster"))?
                .clone();
            let id =
                u32::try_from(index).map_err(|_| Error::inference("too many Maverick clusters"))?;
            Ok(CorefCluster {
                id,
                mentions,
                spans,
                canonical,
            })
        })
        .collect()
}

impl CorefBackend for MaverickCoref {
    fn resolve(&self, text: &str) -> Result<Vec<CorefCluster>> {
        self.predict(text, None)
    }
    fn resolve_mentions(
        &self,
        text: &str,
        mentions: &[(usize, usize)],
    ) -> Result<Vec<CorefCluster>> {
        MaverickCoref::resolve_mentions(self, text, mentions)
    }

    fn name(&self) -> &'static str {
        "maverick"
    }
    fn is_available(&self) -> bool {
        self.worker
            .try_lock()
            .is_none_or(|mut slot| slot.as_mut().is_some_and(|worker| worker.running()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    mod lifecycle {
        use super::*;
        use std::fs;
        use std::path::Path;

        fn launch(path: &Path) -> Result<MaverickCoref> {
            MaverickCoref::with_timeouts(
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/tests/fixtures/maverick-worker.sh"
                ),
                path,
                false,
                Duration::from_secs(3),
                Duration::from_secs(1),
            )
        }

        fn pids(path: &Path) -> Vec<String> {
            fs::read_to_string(path.join("pids"))
                .unwrap()
                .lines()
                .map(str::to_owned)
                .collect()
        }

        fn assert_reaped(pid: &str) {
            let pid = rustix::process::Pid::from_raw(pid.parse().unwrap()).unwrap();
            assert_eq!(
                rustix::process::test_kill_process(pid),
                Err(rustix::io::Errno::SRCH),
                "worker {pid} is still present"
            );
        }

        #[test]
        fn hung_startup_is_bounded_and_reaped() {
            let dir = tempfile::tempdir().unwrap();
            fs::write(dir.path().join("hang-start"), "").unwrap();
            let start = Instant::now();
            let error = launch(dir.path()).err().expect("hung worker started");
            assert!(error.to_string().contains("startup timed out"), "{error}");
            assert!(start.elapsed() < Duration::from_secs(10));
            assert_reaped(&pids(dir.path())[0]);
            #[cfg(target_os = "linux")]
            {
                let pid = fs::read_to_string(dir.path().join("helper-pid")).unwrap();
                let status = format!("/proc/{}/status", pid.trim());
                let deadline = Instant::now() + Duration::from_secs(2);
                while let Ok(info) = fs::read_to_string(&status) {
                    if info
                        .lines()
                        .any(|line| line.starts_with("State:") && line.contains("zombie"))
                    {
                        break;
                    }
                    assert!(
                        Instant::now() < deadline,
                        "model helper outlived the timed-out worker"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }

        #[test]
        fn stalled_requests_broken_frames_and_exits_restart_cleanly() {
            let dir = tempfile::tempdir().unwrap();
            let model = launch(dir.path()).unwrap();
            for (document, expected_error) in [
                ("hang", "request timed out"),
                ("broken", "invalid Maverick output"),
                ("crash", "worker exited"),
            ] {
                let before = pids(dir.path());
                let start = Instant::now();
                let error = model.resolve(document).unwrap_err();
                assert!(error.to_string().contains(expected_error), "{error}");
                assert!(start.elapsed() < Duration::from_secs(10));
                assert!(!model.is_available());
                assert_reaped(before.last().unwrap());
                assert_eq!(model.resolve("É").unwrap()[0].mentions, ["É"]);
                assert_eq!(pids(dir.path()).len(), before.len() + 1);
            }
            let before = pids(dir.path());
            assert!(model.resolve("reject").is_err());
            assert_eq!(model.resolve("A").unwrap()[0].mentions, ["A"]);
            assert_eq!(before, pids(dir.path()));
        }

        #[test]
        fn a_queue_timeout_keeps_the_active_worker() {
            let dir = tempfile::tempdir().unwrap();
            let model = launch(dir.path()).unwrap();
            let before = pids(dir.path());
            let active_request = model.worker.lock();
            let error = model.resolve("queued").unwrap_err();
            assert!(error
                .to_string()
                .contains("waiting for the Maverick worker"));
            drop(active_request);
            assert_eq!(model.resolve("Q").unwrap()[0].mentions, ["Q"]);
            assert_eq!(pids(dir.path()), before);
        }

        #[test]
        fn timeout_also_covers_blocked_writes() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("block-input");
            fs::write(&marker, "").unwrap();
            let model = launch(dir.path()).unwrap();
            let start = Instant::now();
            let error = model.resolve(&"a".repeat(2 * 1024 * 1024)).unwrap_err();
            assert!(error.to_string().contains("request timed out"), "{error}");
            assert!(start.elapsed() < Duration::from_secs(10));
            assert_reaped(&pids(dir.path())[0]);
            fs::remove_file(marker).unwrap();
            assert_eq!(model.resolve("Z").unwrap()[0].mentions, ["Z"]);
        }
    }

    #[test]
    #[ignore = "requires MAVERICK_MODEL_PATH and a Python runtime with maverick-coref"]
    fn checkpoint_links_predefined_unicode_spans_across_whitespace() {
        let checkpoint = std::env::var_os("MAVERICK_MODEL_PATH").unwrap();
        let python = std::env::var_os("MAVERICK_PYTHON").unwrap_or_else(|| "python3".into());
        let model = MaverickCoref::new(python, checkpoint, false).unwrap();
        let text = "Zoë introduced herself.\n\nShe spoke.";
        let supplied = [(0, 3), (15, 22), (25, 28)];
        assert!(model.resolve_mentions(text, &[(1, 3)]).is_err());
        let predicted = model.resolve_mentions(text, &supplied).unwrap();
        assert!(predicted
            .iter()
            .any(|c| c.spans.contains(&(0, 3)) && c.spans.contains(&(25, 28))));
        for cluster in predicted {
            assert!(cluster.spans.iter().all(|s| supplied.contains(s)));
        }
    }

    #[test]
    fn source_offsets_preserve_unicode_and_repeated_mentions() {
        let groups = vec![vec![(0, 3), (6, 9)]];
        let clusters = align_clusters("Éva.  Éva", groups).unwrap();
        assert_eq!(clusters[0].mentions, ["Éva", "Éva"]);
        assert_eq!(clusters[0].spans, [(0, 3), (6, 9)]);
        assert!(align_clusters("Éva", vec![vec![(0, 4)]]).is_err());
    }
}
