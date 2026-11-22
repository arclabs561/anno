//! Human-annotated excerpts from released documents. See fixtures/public_sources/README.md.
use serde_json::Value;
use std::path::Path;
use std::process::Command;

fn verify(flag: &str, input: &str, gold: &[(&str, &str, usize, usize)]) {
    let output = Command::new(env!("CARGO_BIN_EXE_anno"))
        .args([
            "extract", flag, input, "--model", "pattern", "--format", "json",
        ])
        .env("ANNO_NO_DOWNLOADS", "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{input}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut actual: Vec<_> = result["entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e["text"].as_str().unwrap(),
                e["type"].as_str().unwrap(),
                usize::try_from(e["start"].as_u64().unwrap()).unwrap(),
                usize::try_from(e["end"].as_u64().unwrap()).unwrap(),
            )
        })
        .collect();
    actual.sort_by_key(|entity| (entity.2, entity.3));
    assert_eq!(actual, gold, "{input}");
}

const BOE: &[(&str, &str, usize, usize)] = &[("21 de julio de 1876", "DATE", 99, 118)];

#[test]
fn published_excerpts_have_expected_pattern_spans() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/public_sources");
    for (name, gold) in [
        ("boe.txt", BOE),
        (
            "nasa-eclipse.txt",
            &[("2001 Jan 09", "DATE", 7, 18), ("20:21:40", "TIME", 20, 28)][..],
        ),
        ("aozora.txt", &[][..]),
        (
            "tzdb.txt",
            &[("2025-03-22", "DATE", 16, 26), ("13:40:46", "TIME", 27, 35)][..],
        ),
        (
            "tzdb-midnight.txt",
            &[("1978-11-10", "DATE", 35, 45), ("24:00", "TIME", 49, 54)][..],
        ),
    ] {
        verify("--file", root.join(name).to_str().unwrap(), gold);
    }
}

#[cfg(feature = "eval")]
fn verify_http(
    body: &'static [u8],
    content_type: &'static str,
    path: &str,
    gold: &[(&str, &str, usize, usize)],
) {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    let server = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let address = server.local_addr().unwrap();
    let responder = std::thread::spawn(move || {
        let (mut stream, _) = server.accept().unwrap();
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        while reader.read_line(&mut line).unwrap() != 0 && line != "\r\n" {
            line.clear();
        }
        write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
        stream.write_all(body).unwrap();
    });
    verify("--url", &format!("http://{address}/{path}"), gold);
    responder.join().unwrap();
}

#[cfg(feature = "eval")]
#[test]
fn fetch_published_span_over_http() {
    verify_http(
        include_bytes!("fixtures/public_sources/boe.txt"),
        "text/plain; charset=utf-8",
        "boe.txt",
        BOE,
    );
}

#[cfg(feature = "eval")]
#[test]
fn aozora_colophon_url_ends_at_its_fullwidth_delimiter() {
    verify_http(
        include_bytes!("fixtures/public_sources/aozora-footer.html.utf8"),
        "text/html; charset=utf-8",
        "footer.html",
        &[("http://www.aozora.gr.jp/", "URL", 25, 49)],
    );
}
