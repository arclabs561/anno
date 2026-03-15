use anno::backends::{HeuristicNER, RegexNER, StackedNER};
use anno::Model;

fn usage() -> &'static str {
    r#"anno - minimal CLI (workspace facade)

USAGE:
  anno extract [--model <MODEL>] [--format <FORMAT>] --text <TEXT>
  anno extract [--model <MODEL>] [--format <FORMAT>] <TEXT>
  anno --help
  anno --version

MODELS:
  pattern | regex | heuristic | stacked

FORMATS:
  json | text

NOTES:
  - This binary intentionally keeps dependencies small so you can install it with:
      - cargo install --path . --bin anno
      - cargo install --git https://github.com/arclabs561/anno --package anno --bin anno
    without enabling heavyweight ML features by default.
  - For richer workflows, use the workspace CLI crate under `crates/anno-cli/`.
"#
}

#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Json,
    Text,
}

#[derive(Debug, Clone)]
struct ExtractArgs {
    text: String,
    model: String,
    format: OutputFormat,
}

fn parse_extract_args(args: Vec<String>) -> Result<ExtractArgs, String> {
    let mut it = args.into_iter().peekable();

    let mut text: Option<String> = None;
    let mut model: String = "pattern".to_string();
    let mut format = OutputFormat::Json;

    while let Some(tok) = it.next() {
        match tok.as_str() {
            "--text" => {
                let Some(v) = it.next() else {
                    return Err("`--text` requires a value".to_string());
                };
                text = Some(v);
            }
            "--model" => {
                let Some(v) = it.next() else {
                    return Err("`--model` requires a value".to_string());
                };
                model = v;
            }
            "--format" => {
                let Some(v) = it.next() else {
                    return Err("`--format` requires a value".to_string());
                };
                format = match v.as_str() {
                    "json" => OutputFormat::Json,
                    "text" => OutputFormat::Text,
                    _ => return Err(format!("unknown format: {v} (expected: json|text)")),
                };
            }
            "--help" | "-h" => {
                return Err("__HELP__".to_string());
            }
            _ if tok.starts_with("--") => {
                return Err(format!("unknown flag: {tok}"));
            }
            _ => {
                // Positional TEXT: accept one token, or join remainder as a single text.
                let mut parts = vec![tok];
                parts.extend(it);
                text = Some(parts.join(" "));
                break;
            }
        }
    }

    let Some(text) = text else {
        return Err("missing text (use `--text <TEXT>` or pass <TEXT> positionally)".to_string());
    };

    Ok(ExtractArgs {
        text,
        model,
        format,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(first) = args.next() else {
        eprintln!("{}", usage());
        std::process::exit(2);
    };

    match first.as_str() {
        "--help" | "-h" | "help" => {
            print!("{}", usage());
            Ok(())
        }
        "--version" | "-V" | "version" => {
            println!("anno {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        "extract" => {
            let rest: Vec<String> = args.collect();
            let parsed = match parse_extract_args(rest) {
                Ok(v) => v,
                Err(e) if e == "__HELP__" => {
                    print!("{}", usage());
                    return Ok(());
                }
                Err(e) => {
                    eprintln!("{e}\n\n{}", usage());
                    std::process::exit(2);
                }
            };

            let ents = match parsed.model.as_str() {
                "pattern" | "regex" => RegexNER::new().extract_entities(&parsed.text, None)?,
                "heuristic" => HeuristicNER::default().extract_entities(&parsed.text, None)?,
                "stacked" => StackedNER::default().extract_entities(&parsed.text, None)?,
                other => {
                    eprintln!("unknown model: {other}\n\n{}", usage());
                    std::process::exit(2);
                }
            };

            match parsed.format {
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&ents)?);
                }
                OutputFormat::Text => {
                    for e in ents {
                        println!(
                            "{}:{}-{} \"{}\"",
                            e.entity_type.as_label(),
                            e.start(),
                            e.end(),
                            e.text.replace('\n', "\\n")
                        );
                    }
                }
            }
            Ok(())
        }
        other => {
            eprintln!("unknown command: {other}\n\n{}", usage());
            std::process::exit(2);
        }
    }
}
