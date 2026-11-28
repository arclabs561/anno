//! Persistent line-oriented Python inference. The reader thread lets the caller
//! enforce deadlines even if a child stops consuming requests.
use crate::{Error, Result};
use parking_lot::Mutex;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::ffi::{OsStr, OsString};
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

pub(crate) struct Worker {
    name: &'static str,
    child: Child,
    requests: Sender<String>,
    replies: Receiver<io::Result<String>>,
}

impl Worker {
    pub(crate) fn start(
        mut command: Command,
        name: &'static str,
        timeout: Duration,
    ) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| Error::model_init(format!("cannot start {name} Python: {e}")))?;
        let mut input = child.stdin.take().expect("piped stdin");
        let mut output = BufReader::new(child.stdout.take().expect("piped stdout"));
        let (requests, pending) = mpsc::channel::<String>();
        let (completed, replies) = mpsc::channel();
        let worker = Self {
            name,
            child,
            requests,
            replies,
        };

        // The caller must be able to kill the child while either pipe is blocked,
        // including a large request that fills stdin before the model reads it.
        std::thread::Builder::new()
            .name(format!("{name}-io"))
            .spawn(move || loop {
                let mut line = String::new();
                let result = match output.read_line(&mut line) {
                    Ok(0) => Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "worker exited",
                    )),
                    Ok(_) => Ok(line),
                    Err(error) => Err(error),
                };
                let failed = result.is_err();
                if completed.send(result).is_err() || failed {
                    break;
                }
                let Ok(request) = pending.recv() else {
                    break;
                };
                if let Err(error) = writeln!(input, "{request}").and_then(|()| input.flush()) {
                    let _ = completed.send(Err(error));
                    break;
                }
            })
            .map_err(|e| Error::model_init(format!("cannot start {name} I/O thread: {e}")))?;

        let line = worker
            .receive(timeout, "startup")
            .map_err(Error::model_init)?;
        if line.trim() != "\"ready\"" {
            let reason = serde_json::from_str::<serde_json::Value>(&line)
                .ok()
                .and_then(|v| v.get("error")?.as_str().map(str::to_owned))
                .unwrap_or_else(|| "missing ready frame; see worker stderr".into());
            return Err(Error::model_init(format!(
                "{name} startup failed: {reason}"
            )));
        }
        Ok(worker)
    }

    fn receive(&self, timeout: Duration, stage: &str) -> std::result::Result<String, String> {
        let name = self.name;
        match self.replies.recv_timeout(timeout) {
            Ok(Ok(line)) => Ok(line),
            Ok(Err(error)) => Err(format!("{name} {stage} failed: {error}")),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(format!("{name} {stage} timed out")),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(format!("{name} {stage} worker disconnected"))
            }
        }
    }

    pub(crate) fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub(crate) fn request<T: DeserializeOwned>(
        &self,
        request: String,
        timeout: Duration,
    ) -> Result<T> {
        let name = self.name;
        self.requests
            .send(request)
            .map_err(|_| Error::inference(format!("{name} worker exited")))?;
        let line = self.receive(timeout, "request").map_err(Error::inference)?;
        serde_json::from_str(&line)
            .map_err(|e| Error::inference(format!("invalid {name} output: {e}")))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            // A wrapper or model helper may hold the pipes open after the
            // interpreter dies. Signal only the group this adapter created.
            #[cfg(unix)]
            {
                use rustix::process::{kill_process_group, Pid, Signal};
                let _ = kill_process_group(Pid::from_child(&self.child), Signal::KILL);
            }
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Workers emit a JSON string "ready" at startup, then {"ok": value} or
/// {"error": message} for each JSON request. Process failures invalidate the
/// pipe. The next caller can start a fresh worker without replaying a request.
pub(crate) struct PythonWorker {
    python: OsString,
    source: &'static str,
    args: Vec<OsString>,
    name: &'static str,
    worker: Mutex<Option<Worker>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Response<T> {
    Ok(T),
    Error(String),
}

impl PythonWorker {
    pub(crate) fn start(
        python: &OsStr,
        source: &'static str,
        args: Vec<OsString>,
        name: &'static str,
    ) -> Result<Self> {
        let runtime = Self {
            python: python.to_owned(),
            source,
            args,
            name,
            worker: Mutex::new(None),
        };
        *runtime.worker.lock() = Some(runtime.spawn()?);
        Ok(runtime)
    }

    fn spawn(&self) -> Result<Worker> {
        let mut command = Command::new(&self.python);
        command.args(["-u", "-c", self.source]).args(&self.args);
        Worker::start(command, self.name, Duration::from_secs(900))
    }

    pub(crate) fn request<T: DeserializeOwned>(&self, request: &impl Serialize) -> Result<T> {
        let timeout = Duration::from_secs(300);
        let started = Instant::now();
        let frame =
            serde_json::to_string(request).map_err(|e| Error::invalid_input(e.to_string()))?;
        let mut slot = self
            .worker
            .try_lock_for(timeout)
            .ok_or_else(|| Error::inference(format!("timed out waiting for {}", self.name)))?;
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(Error::inference(format!(
                "timed out waiting for {}",
                self.name
            )));
        }
        if !slot.as_mut().is_some_and(Worker::running) {
            *slot = None;
            *slot = Some(self.spawn()?);
        }
        match slot
            .as_ref()
            .expect("worker initialized")
            .request(frame, remaining)
        {
            Ok(Response::Ok(output)) => Ok(output),
            Ok(Response::Error(message)) => {
                Err(Error::inference(format!("{}: {message}", self.name)))
            }
            Err(error) => {
                *slot = None;
                Err(error)
            }
        }
    }

    pub(crate) fn is_available(&self) -> bool {
        self.worker
            .try_lock()
            .is_none_or(|mut slot| slot.as_mut().is_some_and(Worker::running))
    }
}
