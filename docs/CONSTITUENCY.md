# Constituency parsing

Anno imports constituency trees from external parsers. The optional
`scripts/parse_benepar.py` adapter runs the Berkeley Neural Parser locally in
Python and emits JSON; Rust validates the trees and renders or serializes them.
The Rust importer needs no model runtime or Python installation.

## Run benepar and import the result

From this repository, install the model and parse text:

```sh
uv run --locked scripts/parse_benepar.py --download \
  --text 'Mira builds small robots. She smiles.' > parse.json
cargo run -p anno-cli -- parse --input parse.json
```

Subsequent adapter runs can omit `--download`. The script then requires a cached
or local model and disables Hugging Face downloads. `ANNO_NO_DOWNLOADS=1`
rejects `--download`. The `uv` dependency installation is separate from model
downloads; provision that environment first for an offline deployment. The
adjacent script lockfile pins the Python dependency resolution.

The adapter uses spaCy's blank English tokenizer and rule-based sentencizer by
default. `--language` selects a different spaCy tokenizer, and `--model` selects
a matching benepar model or local model directory. Additional tokenizer
dependencies may be needed for other languages. Only the English path has been
tested here. Sentence segmentation and tokenization affect model predictions.
Whitespace tokens are omitted from model input while their original gaps and
all source offsets remain intact.

`anno parse` reads JSON from stdin when `--input` is omitted or `-`.
`--format json` re-emits the validated document; the default `brackets` prints
one Penn-style tree per sentence. Imports are limited to 16 MiB.

## Rust API

```rust
use anno::core::syntax::ParseDocument;

let document = ParseDocument::from_json(&std::fs::read_to_string("parse.json")?)?;
for sentence in document.sentences() {
    println!("{}", sentence.to_bracketed());
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Interchange contract

The document contains `text` and `sentences`. Each sentence has `tokens` and
`constituents`:

- Tokens contain exact source `text`, global character offsets `start` and
  `end` (exclusive), and an optional POS `tag`.
- Constituents contain sentence-local **token indices** `start` and `end`
  (exclusive), and `labels` ordered from outermost to innermost unary node.
- Constituents appear in preorder, starting with the sentence root. Every
  token has a singleton constituent, possibly with `labels: []`. POS tags
  belong to tokens, not to the constituent label chain.
- A root covers all sentence tokens. Children partition their parent;
  crossing spans, duplicate intervals, and omitted leaves are errors.
- Tokens must match the original text exactly. Only whitespace may occur
  between tokens or outside the sentences. Whitespace-only input has no
  sentences. Invalid ranges are rejected, never silently reordered.

Nested constituents are intentionally preserved. Syntax results remain separate
from entity labels and coreference clusters. This release does not change NER
or coreference behavior or infer syntactic heads.

Bracket rendering uses source token text with bracket and control characters escaped. It
preserves phrasal unary chains and available POS tags; missing tags remain
absent. Model-normalized leaf spellings can differ from the original text, so
the rendered string is not promised to match upstream's normalized tree string.
JSON preserves source text and offsets for lossless interchange.

## Implementation boundary

Benepar's published export is a PyTorch model bundle. Native ONNX inference is
not provided here: it would need verified word/subword alignment, the complete
neural scorer (including T5 decoder behavior for the English model), and CKY
decoding parity. The JSON boundary allows another parser to supply the same
validated result without inheriting a Python runtime dependency in Rust.

Primary references:

- [Benepar usage and model export](https://github.com/nikitakit/self-attentive-parser)
- [Compressed trees and unary chains](https://github.com/nikitakit/self-attentive-parser/blob/master/src/benepar/parse_base.py)
- [Word/subword alignment](https://github.com/nikitakit/self-attentive-parser/blob/master/src/benepar/retokenization.py)
- [Chart decoder](https://github.com/nikitakit/self-attentive-parser/blob/master/src/benepar/decode_chart.py)
