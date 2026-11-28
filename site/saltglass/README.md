# Saltglass Reach

The Anno project's playable GitHub Pages port. Reader lab input stays in the browser. The port gazetteer uses Anno's LexiconNER with a fictional local registry. The pattern and heuristic readers run Anno's RegexNER and HeuristicNER. The Outfitter previews equipment against the same passage before fitting it. Local storage keeps the loadout, lab draft and dispatch progress.

Run `git lfs pull` and serve this directory with a static HTTP server. GitHub Pages serves a materialized copy from the gh-pages branch. The generated WebAssembly and JavaScript bindings are checked in so loading the port requires no build service.

From this directory, to rebuild the reader with Rust and wasm-bindgen-cli 0.2.130:

```sh
rustup target add wasm32-unknown-unknown
cargo build --manifest-path engine/Cargo.toml --locked --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir engine/pkg engine/target/wasm32-unknown-unknown/release/saltglass_reader.wasm
```

The upstream Anno source revision is pinned in engine/Cargo.toml. The port landscape and module illustrations were generated for this demo.
