# Saltglass Reach

The Anno project's playable GitHub Pages port. Reader lab input stays in the browser. The port gazetteer uses Anno's LexiconNER with a fictional local registry. The pattern and heuristic readers run Anno's RegexNER and HeuristicNER. The Outfitter previews equipment against the same passage before fitting it. Local storage keeps equipment and dispatch progress. Edited passages stay in page memory. The lab compares the three readers, groups detected mentions with SimpleCorefResolver, runs heuristic relation extraction and previews pattern-based PII redaction. Redaction exports contain the preview without the original passage.

Serve this branch with a static HTTP server. GitHub Pages serves its root. The generated WebAssembly and JavaScript bindings are checked in so loading the port requires no build service.

To rebuild the reader with Rust and wasm-bindgen-cli 0.2.130:

```sh
rustup target add wasm32-unknown-unknown
cargo build --manifest-path engine/Cargo.toml --locked --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir engine/pkg engine/target/wasm32-unknown-unknown/release/saltglass_reader.wasm
```

The upstream Anno source revision is pinned in engine/Cargo.toml. The port landscape and module illustrations were generated for this demo.
