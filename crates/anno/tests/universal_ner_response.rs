//! Exercise the public extraction API with a local Ollama-compatible HTTP fixture.
#![cfg(feature = "llm")]

use anno::backends::universal_ner::{PromptStrategy, UniversalNER};
use anno::Model;
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::Mutex,
    thread,
};

fn completion(content: &str) -> Value {
    json!({"choices": [{"message": {"content": content}}]})
}

fn with_replies(replies: Vec<Value>, check: impl FnOnce(UniversalNER)) {
    // All tests in this executable share the process-level Ollama route.
    static ENV: Mutex<()> = Mutex::new(());
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    for key in [
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANNO_LLM_ENDPOINT",
    ] {
        std::env::remove_var(key);
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::env::set_var("OLLAMA_HOST", format!("http://{addr}"));
    std::env::set_var("NO_PROXY", "127.0.0.1");
    let server = thread::spawn(move || {
        for reply in replies {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&stream);
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            assert_eq!(request.trim_end(), "POST /v1/chat/completions HTTP/1.1");
            let mut length = None;
            for line in reader.by_ref().lines() {
                let line = line.unwrap();
                if line.is_empty() {
                    break;
                }
                let (name, value) = line.split_once(':').unwrap();
                if name.eq_ignore_ascii_case("content-length") {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
            }
            reader.read_exact(&mut vec![0; length.unwrap()]).unwrap();
            let body = reply.to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}", body.len(), body).unwrap();
        }
    });
    check(UniversalNER::new().unwrap());
    // A missing request must fail the server's request assertion instead of blocking join.
    drop(TcpStream::connect(addr));
    server.join().unwrap();
}

#[test]
fn incomplete_extraction_is_an_error_and_can_be_retried() {
    for invalid in [
        json!({}),
        json!({"choices":[{"message":{"content":null}}]}),
        completion(" "),
        json!({"error":{"message":"provider failed"},"choices":[{"message":{"content":"[]"}}]}),
        json!({"choices":[{"error":{"message":"provider failed"},"message":{"content":"[]"}}]}),
        json!({"choices":[{"finish_reason":"error","message":{"content":"[]"}}]}),
        json!({"choices":[{"finish_reason":"tool_calls","message":{"content":"[]"}}]}),
        json!({"choices":[{"message":{"content":"[]","refusal":"declined"}}]}),
        json!({"choices":[{"finish_reason":"content_filter","message":{"content":"[]"}}]}),
        json!({"choices":[{"finish_reason":"length","message":{"content":"[]"}}]}),
    ] {
        with_replies(vec![invalid, completion("[]")], |model| {
            assert!(model.extract_entities("Ada", None).is_err());
            assert!(model.extract_entities("Ada", None).unwrap().is_empty());
        });
    }
}

#[test]
fn only_complete_verification_can_remove_entities() {
    let names = r#"[{"text":"Ada","type":"person","start":0,"end":3},{"text":"Oslo","type":"location","start":7,"end":11}]"#;
    let both = ["Ada", "Oslo"].as_slice();
    for (verification, expected) in [
        (completion("[false,true]"), ["Oslo"].as_slice()),
        (completion("[false]"), both),
        (json!({}), both),
        (completion("[]"), both),
        (completion("[false,true,false]"), both),
        (completion("[false,null]"), both),
        (
            json!({"choices":[{"finish_reason":"length","message":{"content":"[false,false]"}}]}),
            both,
        ),
    ] {
        with_replies(vec![completion(names), verification], |model| {
            let result = model
                .self_verify(true)
                .extract_entities("Ada in Oslo", None)
                .unwrap();
            let texts: Vec<_> = result.iter().map(|e| e.text.as_str()).collect();
            assert_eq!(texts, expected);
        });
    }
}

#[test]
fn compact_requires_an_array() {
    for invalid in ["no array", "bad ] order [", r#"{"result":[]}"#, r#""[]""#] {
        with_replies(vec![completion(invalid), completion("[]")], |model| {
            let model = model.prompt_strategy(PromptStrategy::Compact);
            assert!(model.extract_entities("Ada", None).is_err());
            assert!(model.extract_entities("Ada", None).unwrap().is_empty());
        });
    }
}

#[test]
fn failed_verification_is_retried_and_complete_verdicts_are_cached() {
    let name = r#"[{"text":"Ada","type":"person","start":0,"end":3}]"#;
    with_replies(
        vec![
            completion(name),
            completion("[]"),
            completion(name),
            completion("[false]"),
        ],
        |model| {
            let model = model.self_verify(true);
            assert_eq!(model.extract_entities("Ada", None).unwrap().len(), 1);
            assert!(model.extract_entities("Ada", None).unwrap().is_empty());
            assert!(model.extract_entities("Ada", None).unwrap().is_empty());
        },
    );
}
