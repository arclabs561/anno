# Jump from a pronoun to its caption

A caption editor can show the passages that FCoref groups together and play the cue containing each one. The supplied fixture has two cues. The second starts at 14.9 seconds and contains “She said it would reopen on Monday.” The pinned model links “She” to “Émilie Durand” and “it” to “the east gate”. [Original CC0 fixture](gate-captions.json)

From the repository root, set `ORT_DYLIB_PATH` to the absolute path of the ONNX Runtime CPU 1.24.4 shared library for your platform. Set `FCOREF_MODEL_PATH` to a local FP32 export of `biu-nlp/f-coref` at revision `e91dfff12879495d882ba9460d0c5d5dd44ade59`. The model archive and runtime are installed separately. The workflow includes the export commands and artifact checksums. [Pinned CPU setup](../../../../.github/workflows/fcoref-cli.yml)

```sh
cargo build --locked --release -p anno --no-default-features \
  --features onnx,ort/load-dynamic --example fcoref_word_spans

example=crates/anno/examples/caption_links
binary=target/release/examples/fcoref_word_spans
export ORT_DYLIB_PATH FCOREF_MODEL_PATH
python3 "$example/caption_links.py" "$binary" "$FCOREF_MODEL_PATH" \
  "$example/gate-captions.json" > caption-links.json
```

In `suggested_clusters`, the original “She” member contains:

```json
{"quote":"She","chars":[39,42],"cues":[{"id":"cue-18","start_ms":14900,"end_ms":17600,"chars":[0,3]}]}
```

Use `start_ms` and `end_ms` to play the whole cue. Member offsets index the transcript formed by joining the input texts in order with one newline. Each cue's offsets index its own text. Both ranges count Unicode code points from zero with an exclusive end. A mention spanning two cues lists both intersections. The adapter checks the FCoref receipt against the SHA-256 of the exact UTF-8 transcript before mapping any offsets.

Replay the original and a revision with `🎙 ` inserted at the start of cue 17:

```sh
python3 "$example/test_caption_links.py" "$binary" "$FCOREF_MODEL_PATH"
```

The replay checks the actual model clusters, quotes and cue ranges. In the revision “She” moves to `[41,44)` in the transcript and stays at `[0,3)` in cue 18, with the same playback interval. The adapter accepts other files with the same JSON schema. On new recordings these clusters are model suggestions for an editor to review.
