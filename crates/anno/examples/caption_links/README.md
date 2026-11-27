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

The saved `start_ms` and `end_ms` describe the cue at extraction time. Member offsets index the transcript formed by joining the input texts in order with one newline. Each cue's offsets index its own text. Both ranges count Unicode code points from zero with an exclusive end. A mention spanning two cues lists both intersections. The adapter checks the FCoref receipt against the SHA-256 of the exact UTF-8 transcript before mapping any offsets.

Replay the original and a revision with `🎙 ` inserted at the start of cue 17:

```sh
python3 "$example/test_caption_links.py" "$binary" "$FCOREF_MODEL_PATH"
```

The replay checks the actual model clusters, quotes and cue ranges. In the revision “She” moves to `[41,44)` in the transcript and stays at `[0,3)` in cue 18, with the same playback interval. The adapter accepts other files with the same JSON schema. On new recordings these clusters are model suggestions for an editor to review.

## Reopen a saved link in a caption editor

Load [playback.js](playback.js) on the editor’s HTTPS or localhost page. Pass the current cues in transcript order, the saved adapter output, and the selected member of `suggested_clusters`. Use the media asset associated with that caption project. The receipt hashes the transcript text; it carries no recording identifier.

```js
const targets = await resolvePlaybackTargets(currentCues, receipt, member);
for (const target of targets) {
    // Each node contains exactly the source text of this cue.
    const node = cueTextNodes.get(target.id);
    const range = document.createRange();
    range.setStart(node, target.selectionStart);
    range.setEnd(node, target.selectionEnd);
    showPassage(range, target.startTime, target.endTime);
}
```

The caller supplies `showPassage`. Its seek action can assign `media.currentTime = target.startTime`; both times are seconds. A cross-cue quote offers one passage per cue. The reader uses current cue times after a timing edit and rejects changed transcript text or changed cue intersections. DOM selections use UTF-16 offsets, including the extra code unit occupied by `🎙`.

The [browser replay](test_playback.py) reads a saved receipt in fresh Chromium processes, checks the actual DOM ranges, and waits for completed seeks on a silent PCM recording. Run it without model weights:

```sh
python3 -m pip install playwright==1.62.0
python3 "$example/test_playback.py" "$(command -v google-chrome)"
```

The FCoref CI job also passes `--browser "$(command -v google-chrome)"` to the existing FP32 replay. That check selects the native model’s “Émilie Durand” suggestion after a microphone emoji and seeks to a revised cue time.
