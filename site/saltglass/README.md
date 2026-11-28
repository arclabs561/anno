# Saltglass Reach

The Anno project's playable GitHub Pages port. Reader lab input stays in the browser. The port gazetteer uses Anno's LexiconNER with a fictional local registry. The pattern and heuristic readers run Anno's RegexNER and HeuristicNER. Each instrument has a two-position bench test. At the mission desk, the Outfitter also previews the selected tide notice. Local storage keeps equipment and dispatch progress. Edited passages stay in page memory. The lab compares the five readers, groups detected mentions with SimpleCorefResolver, runs heuristic relation extraction and previews pattern-based PII redaction. Redaction exports contain the preview without the original passage.

Run `git lfs pull` and serve this directory with a static HTTP server. GitHub Pages serves a materialized copy from the gh-pages branch. The generated WebAssembly and JavaScript bindings are checked in so loading the port requires no build service.

From this directory, to rebuild the reader with Rust and wasm-bindgen-cli 0.2.130:

```sh
rustup target add wasm32-unknown-unknown
cargo build --manifest-path engine/Cargo.toml --locked --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir engine/pkg engine/target/wasm32-unknown-unknown/release/saltglass_reader.wasm
```

The upstream Anno source revision is pinned in engine/Cargo.toml. The port landscape and module illustrations were generated for this demo.

Fresh pilots start with Harbor pilot, which keeps registered names first, then non-overlapping pattern matches, then remaining predictions from Anno’s CrfNER. The port uses the hand-tuned CRF configuration with bundled trained weights disabled. The raw CRF remains available separately. The port log’s watch ledger compares the fitted reader with the clerk’s four references to the captain. The reader lab also runs Anno’s brat standoff and CoNLL BIO exporters.

The optional Survey receiver runs GLiNER-small-v2 through GLiNER.js and ONNX Runtime Web, alongside the fitted Anno WASM reader. Open its Outfitter bench, enter a passage and entity labels, and start the scan. The first scan retrieves 183,403,734 bytes of quantized weights from the pinned onnx-community checkpoint, plus tokenizer and runtime files. Model weights can be cached by the browser. The JSON log includes both readings, the checkpoint revision, requested labels, threshold, model scores and Unicode character offsets. The adapted GLiNER.js browser module comes from npm package gliner 0.0.19 with its distributed license in vendor/gliner-LICENSE.
