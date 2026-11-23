//! FP32 CPU inference and document-budget measurements for the Linux cgroup replay.
use anno::{FCoref, FCorefConfig};
use regex::Regex;
use serde_json::{json, Value};
use std::{env, fs, path::Path, time::Instant};
use tokenizers::Tokenizer;

fn fixture(dir: &Path, count: usize) -> (String, Vec<(usize, usize)>) {
    let text = "x ".repeat(count - 19)
        + "Alice reviewed the proposal. She approved it. Bob objected. She overruled him.";
    let words: Vec<_> = Regex::new(r"\w+|[^\s\w]")
        .unwrap()
        .find_iter(&text)
        .map(|m| (m.start(), m.end()))
        .collect();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let surfaces: Vec<_> = words.iter().map(|&(s, e)| &text[s..e]).collect();
    let encoding = tokenizer.encode(surfaces, true).unwrap();
    assert_eq!(encoding.get_ids().len(), count);
    assert_eq!(encoding.get_special_tokens_mask().iter().sum::<u32>(), 2);
    (text, words)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let [model_dir, threads, tokens, budget] = args.as_slice() else {
        panic!("usage: fcoref_resource_profile MODEL_DIR THREADS ENCODED_TOKENS BUDGET");
    };
    let threads: usize = threads.parse().expect("invalid encoder thread count");
    let count: usize = tokens.parse().expect("invalid encoded token count");
    let budget: usize = budget.parse().expect("invalid document budget");
    assert!(
        (1..=2).contains(&threads),
        "choose one or two encoder threads"
    );
    assert!(
        (511..=522).contains(&count),
        "fixture supports 511–522 encoded tokens"
    );
    let dir = Path::new(model_dir);
    let (text, words) = fixture(dir, count);
    let ref_text = include_str!("../tests/fixtures/fcoref_word_budget.json");
    let reference: Vec<(usize, usize, f32)> = serde_json::from_str(ref_text).unwrap();
    let load = Instant::now();
    let model = FCoref::from_path_with_document_budget(
        dir.to_str().unwrap(),
        FCorefConfig {
            num_threads: threads,
            ..Default::default()
        },
        Some(budget),
    )
    .unwrap();
    let load_ms = load.elapsed().as_secs_f64() * 1e3;
    let mut times = Vec::new();
    let mut first: Option<Value> = None;
    for _ in 0..6 {
        let timer = Instant::now();
        let result = model.analyze_words(&text, &words);
        times.push(timer.elapsed().as_secs_f64() * 1e3);
        let predictions = if count > budget {
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains(&format!("{count} tokens")) && err.contains(&budget.to_string()),
                "{err}"
            );
            json!({"error":err})
        } else {
            let r = result.unwrap();
            let mut max_delta = 0f32;
            let mentions: Vec<_> = r
                .mentions
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    assert_eq!(m.text, text[m.character_start..m.character_end]);
                    if count == 521 {
                        let (s, e, score) = reference[i];
                        assert_eq!((s, e), (m.character_start, m.character_end));
                        let delta = (score - m.raw_mention_score).abs();
                        assert!(delta < 0.01);
                        max_delta = max_delta.max(delta);
                    }
                    json!([
                        m.character_start,
                        m.character_end,
                        m.text,
                        m.raw_mention_score
                    ])
                })
                .collect();
            let clusters: Vec<_> = r.clusters.into_iter().map(|c| c.spans).collect();
            if count == 521 {
                assert_eq!(mentions.len(), 126);
                assert_eq!(
                    clusters,
                    [
                        vec![(1004, 1009), (1033, 1036), (1064, 1067)],
                        vec![(1019, 1031), (1046, 1048)],
                        vec![(1050, 1053), (1078, 1081)]
                    ]
                );
            }
            json!({"mentions":mentions,"clusters":clusters,"max_reference_score_delta": if count == 521 {Some(max_delta)} else {None}})
        };
        if let Some(previous) = first.as_ref() {
            assert_eq!(*previous, predictions);
        } else {
            first = Some(predictions);
        }
    }
    let cg = [
        "cpu.max",
        "cpuset.cpus.effective",
        "memory.max",
        "memory.swap.max",
        "memory.peak",
        "memory.events",
        "cpu.stat",
    ]
    .into_iter()
    .map(|key| {
        (
            key,
            fs::read_to_string(Path::new("/sys/fs/cgroup").join(key)).ok(),
        )
    })
    .collect::<std::collections::BTreeMap<_, _>>();
    let proc_limits = fs::read_to_string("/proc/self/limits").unwrap();
    let proc_cgroup = fs::read_to_string("/proc/self/cgroup").unwrap();
    let proc_status = fs::read_to_string("/proc/self/status").unwrap();
    let thread_masks: Vec<_> = fs::read_dir("/proc/self/task")
        .unwrap()
        .filter_map(|entry| {
            let status = fs::read_to_string(entry.ok()?.path().join("status")).ok()?;
            status.lines().find_map(|s| {
                s.strip_prefix("Cpus_allowed_list:")
                    .map(|v| v.trim().to_string())
            })
        })
        .collect();
    println!(
        "{}",
        json!({"threads_configured":threads,"encoded_tokens":count,"special_tokens":2,"budget":budget,"text":text,"word_count":words.len(),"load_ms":load_ms,"first_call_ms":times[0],"warm_ms":times[1..],"predictions":first,"proc_status":proc_status,"thread_affinities":thread_masks,"cgroup":cg,"proc_limits":proc_limits,"proc_cgroup":proc_cgroup})
    );
}
