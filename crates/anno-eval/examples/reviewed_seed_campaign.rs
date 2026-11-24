//! Allocate one mutation to a muxer-selected seed and one to a rotating control.
#[allow(dead_code)] // This binary uses the receipt-reporting entry point.
#[path = "../../anno/tests/support/reviewed_kb_fuzz.rs"]
mod harness;

use fs2::FileExt;
use muxer::{Extract, MabConfig, Objective, Summary};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::io::Write;
use std::panic::{catch_unwind, resume_unwind};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

macro_rules! seed {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!("../../../fuzz/seeds/reviewed_kb/", $name)),
        )
    };
}
// Only reviewed, checked-in inputs can become policy arms or enter the public state.
const SEEDS: &[(&str, &[u8])] = &[
    seed!("changed-and-restored"),
    seed!("foreign-owner"),
    seed!("independent-same-key"),
    seed!("local-rebind"),
    seed!("local-siblings"),
    seed!("shared-namespace-change"),
    seed!("unlinked-track"),
];
const SCHEMA: u8 = 1;
const MAX_ROUNDS: u32 = 65535;
const FAILURE: u32 = 1 << 18;
// Bits per track: stale, single-track receipt, two-track receipt. Bit 18 is a
// reproduced visible-binding invariant failure. Higher bits are invalid.
// Shared tracks (0 and 1) can return two; the other four can return one.
const VALID_BITS: u32 = FAILURE | 0b011_011_011_011_101_101;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    calls: u32,
    wins: u32,
    seen: u32,
}
type Lane = BTreeMap<String, Arm>;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    schema: u8,
    corpus: String,
    next_round: u32,
    adaptive: Lane,
    control: Lane,
}

fn corpus_version() -> String {
    let mut hash = Sha256::new();
    hash.update(b"reviewed-kb-campaign-v1");
    hash.update(include_bytes!(
        "../../anno/tests/support/reviewed_kb_fuzz.rs"
    ));
    for (name, bytes) in SEEDS {
        hash.update(name.as_bytes());
        hash.update([0, bytes.len() as u8]);
        hash.update(bytes);
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl State {
    fn new() -> Self {
        let lane: Lane = SEEDS
            .iter()
            .map(|&(id, _)| (id.into(), Arm::default()))
            .collect();
        Self {
            schema: SCHEMA,
            corpus: corpus_version(),
            next_round: 0,
            adaptive: lane.clone(),
            control: lane,
        }
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.schema != SCHEMA || self.corpus != corpus_version() || self.next_round > MAX_ROUNDS
        {
            return Err("incompatible campaign version or round");
        }
        for lane in [&self.adaptive, &self.control] {
            if lane.len() != SEEDS.len()
                || SEEDS.iter().any(|(id, _)| !lane.contains_key(*id))
                || lane.values().any(|a| {
                    a.wins > a.calls
                        || a.wins > a.seen.count_ones()
                        || a.calls > self.next_round
                        || a.seen & !VALID_BITS != 0
                        || (a.calls == 0 && a.seen != 0)
                })
                || lane.values().map(|a| a.wins).sum::<u32>()
                    > lane.values().fold(0, |seen, a| seen | a.seen).count_ones()
                || lane.values().map(|a| u64::from(a.calls)).sum::<u64>()
                    != u64::from(self.next_round)
            {
                return Err("invalid seed identities, exposure counts or outcome classes");
            }
        }
        Ok(())
    }
}

fn observe(bytes: &[u8]) -> u32 {
    let mut signals = 0;
    harness::replay_observed(bytes, |track, receipt| {
        let class = match receipt {
            None => 0,
            Some(1) => 1,
            Some(2) => 2,
            _ => panic!("unexpected receipt scope"),
        };
        signals |= 1 << (3 * track + class);
    });
    signals
}

// Only a named, repeatable oracle violation is a failure signal. Propagate
// every other panic and leave the previous checkpoint intact.
fn reproduced(run: impl Fn() -> u32 + std::panic::RefUnwindSafe) -> u32 {
    let attempt = || match catch_unwind(&run) {
        Ok(bits) => Some(bits),
        Err(error) if error.is::<harness::VisibleBindingMismatch>() => None,
        Err(error) => resume_unwind(error),
    };
    if let Some(bits) = attempt() {
        return bits;
    }
    assert!(
        attempt().is_none(),
        "visible-binding failure did not reproduce"
    );
    FAILURE
}

fn mutate(bytes: &[u8], round: u32) -> Vec<u8> {
    let salt = Sha256::digest(round.to_le_bytes());
    let commands = b"012345pABNLmc!r";
    let mut out = bytes.to_vec();
    for &[position, command] in salt[..8].as_chunks::<2>().0 {
        out.insert(
            usize::from(position) % (out.len() + 1),
            commands[usize::from(command) % commands.len()],
        );
    }
    out.truncate(harness::MAX_INPUT_BYTES);
    out
}

fn select(lane: &Lane) -> String {
    let summaries = lane
        .iter()
        .map(|(id, a)| {
            (
                id.clone(),
                Summary {
                    calls: u64::from(a.calls),
                    ok: u64::from(a.wins),
                    ..Default::default()
                },
            )
        })
        .collect();
    let cfg = MabConfig {
        objectives: vec![Objective::maximize(Extract::OkRateUcb, 1.0)],
        ..Default::default()
    };
    muxer::select_mab(&lane.keys().cloned().collect::<Vec<_>>(), &summaries, cfg).chosen
}

fn credit(lane: &mut Lane, id: &str, signals: u32, baseline: u32) -> u32 {
    let known = lane.values().fold(baseline, |bits, a| bits | a.seen);
    let novel = signals & !known;
    let arm = lane.get_mut(id).unwrap();
    arm.seen |= signals;
    arm.calls += 1;
    arm.wins += u32::from(novel != 0);
    novel
}

fn run(path: &Path, requested: u32) -> Result<(), Box<dyn Error>> {
    let mut lock: OsString = path.as_os_str().into();
    lock.push(".lock");
    let lock = File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock)?;
    lock.lock_exclusive()?;
    let mut state = match File::open(path) {
        Ok(file) => {
            if file.metadata()?.len() > 16384 {
                return Err("checkpoint exceeds 16 KiB".into());
            }
            serde_json::from_reader(file)?
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => State::new(),
        Err(e) => return Err(e.into()),
    };
    state.validate()?;
    if state
        .adaptive
        .values()
        .chain(state.control.values())
        .any(|arm| arm.seen & FAILURE != 0)
    {
        return Err(
            "checkpoint contains a reproduced invariant failure; inspect its last round".into(),
        );
    }
    if requested < state.next_round {
        println!("round {requested} already recorded");
        return Ok(());
    }
    if requested != state.next_round || requested >= MAX_ROUNDS {
        return Err("expected next campaign round".into());
    }

    // Always replay every original; existing smoke behavior is ineligible for reward.
    let mut baseline = 0;
    for &(_, bytes) in SEEDS {
        baseline |= observe(bytes);
    }
    let adaptive = select(&state.adaptive);
    let control = SEEDS[(requested as usize + 3) % SEEDS.len()].0;
    let mut failed = false;
    for (name, lane, id) in [
        ("adaptive", &mut state.adaptive, adaptive.as_str()),
        ("control", &mut state.control, control),
    ] {
        let seed = SEEDS.iter().find(|s| s.0 == id).unwrap().1;
        let probe = mutate(seed, requested);
        let signals = reproduced(|| observe(&probe));
        let novel = credit(lane, id, signals, baseline);
        failed |= signals & FAILURE != 0;
        println!("round={requested} lane={name} seed={id} new_semantic_classes={} reproduced_failure={} new_failure={}",
                 (novel & !FAILURE).count_ones(), signals & FAILURE != 0, novel & FAILURE != 0);
    }
    state.next_round += 1;
    state.validate()?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temp, &state)?;
    writeln!(temp)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    if failed {
        return Err("reviewed binding invariant failed twice; see lane and round".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = PathBuf::from(
        args.next()
            .ok_or("expected STATE.json and zero-based ROUND")?,
    );
    let round = args
        .next()
        .ok_or("expected ROUND")?
        .to_str()
        .ok_or("invalid round")?
        .parse()?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    run(&path, round)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn native_seed_replay_and_idempotent_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("reviewed.json");
        run(&path, 0).unwrap();
        let first = fs::read(&path).unwrap();
        run(&path, 0).unwrap();
        assert_eq!(fs::read(&path).unwrap(), first);
        assert!(run(&path, 2).is_err());
        run(&path, 1).unwrap();
        let state: State = serde_json::from_reader(File::open(path).unwrap()).unwrap();
        assert_eq!(state.next_round, 2);
        for lane in [state.adaptive, state.control] {
            assert_eq!(lane.values().map(|a| a.calls).sum::<u32>(), 2);
        }
    }

    #[test]
    fn injected_failure_requires_reproduction_and_earns_one_reward() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let attempts = AtomicU32::new(0);
        let outcome = reproduced(|| {
            attempts.fetch_add(1, Ordering::Relaxed);
            std::panic::panic_any(harness::VisibleBindingMismatch)
        });
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        assert_eq!(outcome, FAILURE);
        let mut state = State::new();
        let (seed, _) = SEEDS[0];
        let baseline = 1;
        assert_eq!(credit(&mut state.adaptive, seed, baseline, baseline), 0);
        assert_eq!(
            credit(&mut state.adaptive, seed, outcome, baseline),
            FAILURE
        );
        assert_eq!(credit(&mut state.adaptive, seed, outcome, baseline), 0);
        assert_eq!(
            credit(&mut state.adaptive, SEEDS[1].0, outcome, baseline),
            0
        );
        assert_eq!(state.adaptive[seed].wins, 1);
        assert_eq!(state.adaptive[seed].calls, 3);
        assert!(catch_unwind(|| reproduced(|| panic!("infrastructure failure"))).is_err());
        assert_eq!(state.control[seed].calls, 0);
    }

    #[test]
    fn incompatible_or_external_state_is_rejected() {
        let original = serde_json::to_value(State::new()).unwrap();
        for change in ["schema", "corpus", "identity", "class", "exposure", "raw"] {
            let mut v = original.clone();
            match change {
                "schema" => v["schema"] = 99.into(),
                "corpus" => v["corpus"] = "other-corpus".into(),
                "identity" => {
                    let arms = v["adaptive"].as_object_mut().unwrap();
                    let existing = arms.remove(SEEDS[0].0).unwrap();
                    arms.insert("unreviewed-held-out".into(), existing);
                }
                "class" => v["control"][SEEDS[0].0]["seen"] = (1 << 21).into(),
                "exposure" => v["control"][SEEDS[0].0]["wins"] = 1.into(),
                "raw" => v["raw_text"] = "untrusted".into(),
                _ => unreachable!(),
            }
            assert!(
                !serde_json::from_value::<State>(v).is_ok_and(|s| s.validate().is_ok()),
                "{change}"
            );
        }
    }
}
