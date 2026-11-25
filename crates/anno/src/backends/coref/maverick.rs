//! CPU inference with the optional `maverick-coref` Python runtime. One worker
//! retains its model across documents. Supply an upstream checkpoint or hub ID.
use super::{CorefBackend, CorefCluster};
use crate::{Error, Result};
use serde::Deserialize;
use std::ffi::OsStr;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

/// A persistent Maverick worker. Install `maverick-coref==1.0.7` in the selected
/// Python environment. `singletons` is useful with LitBank and PreCo weights.
pub struct MaverickCoref {
    worker: Mutex<Worker>,
    singletons: bool,
}

struct Worker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl MaverickCoref {
    /// Start Python and wait until the model and tokenizer are ready.
    pub fn new(
        python: impl AsRef<OsStr>,
        checkpoint: impl AsRef<OsStr>,
        singletons: bool,
    ) -> Result<Self> {
        let mut child = Command::new(python)
            .args(["-u", "-c", include_str!("maverick_worker.py")])
            .arg(checkpoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| Error::model_init(format!("cannot start Maverick Python: {e}")))?;
        let input = child.stdin.take().expect("piped stdin");
        let output = BufReader::new(child.stdout.take().expect("piped stdout"));
        let mut worker = Worker {
            child,
            input,
            output,
        };
        let mut line = String::new();
        worker.output.read_line(&mut line)?;
        if line.trim() != "\"ready\"" {
            return Err(Error::model_init(
                "Maverick did not load; see worker stderr",
            ));
        }
        Ok(Self {
            worker: Mutex::new(worker),
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
        #[derive(Deserialize)]
        #[serde(rename_all = "lowercase")]
        enum Reply {
            Clusters(Vec<Vec<(usize, usize)>>),
            Error(String),
        }
        let request =
            serde_json::json!({"text": text, "mentions": mentions, "singletons": self.singletons});
        let mut worker = self
            .worker
            .lock()
            .map_err(|_| Error::inference("Maverick worker poisoned"))?;
        writeln!(worker.input, "{request}")?;
        worker.input.flush()?;
        let mut line = String::new();
        if worker.output.read_line(&mut line)? == 0 {
            return Err(Error::inference("Maverick worker exited"));
        }
        drop(worker);
        let reply = serde_json::from_str(&line)
            .map_err(|e| Error::inference(format!("invalid Maverick output: {e}")))?;
        match reply {
            Reply::Error(error) => Err(Error::inference(error)),
            Reply::Clusters(groups) => align_clusters(text, groups),
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
            .lock()
            .ok()
            .is_some_and(|mut w| matches!(w.child.try_wait(), Ok(None)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
