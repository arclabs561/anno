//! The routing checks run in their own process to isolate provider environment variables.
#![cfg(feature = "llm")]

use anno::backends::{llm_client::LlmConfig, universal_ner::UniversalNER};
use anno::Model;
use std::{
    env,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    thread,
};

#[test]
fn explicit_destination_and_credentials_apply_to_extraction_and_verification() {
    for key in [
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OLLAMA_HOST",
    ] {
        env::remove_var(key);
    }
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let host = format!("http://{addr}");
    let server = thread::spawn(move || {
        let mut received = Vec::new();
        for answer in [
            r#"[{"text":"Ada","type":"person","start":0,"end":3}]"#,
            "[false]",
            "[]",
            "[]",
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(&stream);
            let mut request = String::new();
            if reader.read_line(&mut request).unwrap() == 0 {
                break;
            }
            let mut auth = None;
            let mut body_size = None;
            for line in reader.by_ref().lines() {
                let line = line.unwrap();
                if line.is_empty() {
                    break;
                }
                let (key, val) = line.split_once(':').unwrap();
                if key.eq_ignore_ascii_case("authorization") {
                    auth = Some(val.trim().to_owned());
                }
                if key.eq_ignore_ascii_case("content-length") {
                    body_size = Some(val.trim().parse::<usize>().unwrap());
                }
            }
            let mut body = vec![0; body_size.unwrap()];
            reader.read_exact(&mut body).unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            received.push((
                request.trim_end().to_owned(),
                auth,
                body["model"].as_str().unwrap().to_owned(),
            ));
            let response =
                serde_json::json!({"choices":[{"message":{"content":answer}}]}).to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}", response.len(), response).unwrap();
        }
        received
    });
    let config = LlmConfig {
        endpoint: Some(format!("{host}/explicit")),
        model: "fixture".into(),
        api_key: Some("dummy".into()),
        ..Default::default()
    };
    let explicit = UniversalNER::with_config(config.clone())
        .unwrap()
        .self_verify(true);
    assert!(explicit.is_available());
    env::set_var("OLLAMA_HOST", format!("{host}/ambient"));
    assert!(explicit.extract_entities("Ada", None).unwrap().is_empty());
    let anonymous = UniversalNER::with_config(LlmConfig {
        api_key: None,
        ..config
    })
    .unwrap();
    anonymous.extract_entities("Ora", None).unwrap();
    UniversalNER::new()
        .unwrap()
        .extract_entities("Kai", None)
        .unwrap();
    // Wake a server still waiting after a missing request, so the assertion can fail.
    drop(TcpStream::connect(addr));
    let requests = server.join().unwrap();
    let received: Vec<_> = requests
        .iter()
        .map(|(line, auth, model)| (line.as_str(), auth.as_deref(), model.as_str()))
        .collect();
    assert_eq!(
        received,
        [
            ("POST /explicit HTTP/1.1", Some("Bearer dummy"), "fixture"),
            ("POST /explicit HTTP/1.1", Some("Bearer dummy"), "fixture"),
            ("POST /explicit HTTP/1.1", None, "fixture"),
            (
                "POST /ambient/v1/chat/completions HTTP/1.1",
                Some("Bearer ollama"),
                "llama3.2:3b"
            ),
        ]
    );
    env::set_var("OPENROUTER_API_KEY", "unrelated");
    assert!(!UniversalNER::with_config(LlmConfig::groq("test"))
        .unwrap()
        .is_available());
    env::set_var("GROQ_API_KEY", "matching");
    assert!(UniversalNER::with_config(LlmConfig::groq("test"))
        .unwrap()
        .is_available());
}
