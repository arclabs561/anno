//! Constituency parse import command.

use std::fs::File;
use std::io::{self, Read};

use anno::core::syntax::ParseDocument;
use clap::{Parser, ValueEnum};

const MAX_INPUT_BYTES: u64 = 16 * 1024 * 1024;

/// Import constituency parses from a parser JSON document.
#[derive(Parser, Debug)]
#[command(about = "Import constituency parses from a parser JSON document")]
pub struct ParseArgs {
    /// Parser JSON document to import, or "-" for standard input
    #[arg(short, long, default_value = "-", value_name = "FILE")]
    pub input: String,

    /// Output format
    #[arg(long, default_value = "brackets", value_enum)]
    pub format: ParseFormat,
}

/// Output format for imported constituency parses.
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum ParseFormat {
    /// Re-emit the normalized parser JSON document
    Json,
    /// Print one Penn Treebank-style tree per sentence
    #[default]
    Brackets,
}

/// Import a parser JSON document and render its constituency trees.
pub fn run(args: ParseArgs) -> Result<(), String> {
    let input = read_input(&args.input)?;
    let document = ParseDocument::from_json(&input)
        .map_err(|error| format!("Failed to parse constituency JSON: {error}"))?;

    match args.format {
        ParseFormat::Json => {
            let json = serde_json::to_string_pretty(&document)
                .map_err(|error| format!("Failed to serialize constituency JSON: {error}"))?;
            println!("{json}");
        }
        ParseFormat::Brackets => {
            for sentence in document.sentences() {
                println!("{}", sentence.to_bracketed());
            }
        }
    }

    Ok(())
}

fn read_input(path: &str) -> Result<String, String> {
    if path == "-" {
        read_limited(io::stdin().lock(), "stdin")
    } else {
        let file = File::open(path).map_err(|error| format!("Failed to read {path}: {error}"))?;
        let length = file
            .metadata()
            .map_err(|error| format!("Failed to read metadata for {path}: {error}"))?
            .len();
        if length > MAX_INPUT_BYTES {
            return Err(input_too_large(path));
        }
        read_limited(file, path)
    }
}

fn read_limited(reader: impl Read, source: &str) -> Result<String, String> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Failed to read {source}: {error}"))?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err(input_too_large(source));
    }
    String::from_utf8(bytes)
        .map_err(|error| format!("Input from {source} was not valid UTF-8: {error}"))
}

fn input_too_large(source: &str) -> String {
    format!("Input from {source} exceeds the 16 MiB limit for constituency parse imports")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use clap::Parser;

    use super::{read_limited, ParseArgs, ParseDocument, ParseFormat, MAX_INPUT_BYTES};

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: ParseArgs,
    }

    const VALID_DOCUMENT: &str = r#"{
        "text": "Anno.",
        "sentences": [{
            "tokens": [
                {"text": "Anno", "start": 0, "end": 4, "tag": "NNP"},
                {"text": ".", "start": 4, "end": 5, "tag": "."}
            ],
            "constituents": [
                {"start": 0, "end": 2, "labels": ["S"]},
                {"start": 0, "end": 1, "labels": ["NP"]},
                {"start": 1, "end": 2, "labels": []}
            ]
        }]
    }"#;

    #[test]
    fn defaults_to_stdin_and_bracket_output() {
        let cli =
            TestCli::try_parse_from(["test"]).expect("the parse command defaults should be valid");
        assert_eq!(cli.args.input, "-");
        assert!(matches!(cli.args.format, ParseFormat::Brackets));
    }

    #[test]
    fn accepts_json_output_and_a_file_input() {
        let cli = TestCli::try_parse_from(["test", "--input", "tree.json", "--format", "json"])
            .expect("explicit parse arguments should be valid");
        assert_eq!(cli.args.input, "tree.json");
        assert!(matches!(cli.args.format, ParseFormat::Json));
    }

    #[test]
    fn imports_a_valid_parser_document() {
        ParseDocument::from_json(VALID_DOCUMENT).expect("fixture should import");
    }

    #[test]
    fn rejects_malformed_parser_document() {
        assert!(ParseDocument::from_json("{not json").is_err());
    }

    #[test]
    fn rejects_input_larger_than_the_limit() {
        let bytes = vec![b'x'; MAX_INPUT_BYTES as usize + 1];
        let error = read_limited(Cursor::new(bytes), "fixture").expect_err("input is too large");
        assert!(error.contains("16 MiB limit"));
    }
}
