# Can a resolver follow a change in who signed?

“Ada is the pilot. Béa is the medic. Today she, the pilot, signed.” Changing the last *pilot* to *medic* reverses the intended signer while keeping the two names, the pronoun and their character offsets identical. We froze 20 such semantic pairs across three constructions before running the models. The native entity interface abstained on both halves of every pair. FCoref read the full text and supplied seven uniquely correct named answers to individual questions. It recovered the intended named signer on both halves of zero pairs. [Original panel and scorer](../../scripts/qa/signer_roles.py) · [Replay receipt](signer-roles.json)

| Raw FCoref, exact named answer | Questions | Correct | Wrong | Exact query missing | Exact query returned, candidate missing and no named link |
|---|---:|---:|---:|---:|---:|
| Appositive, “she, the pilot” | 16 | 0 | 0 | 16 | 0 |
| Exclusive role, “Only the pilot signed … she signed it” | 16 | 5 | 4 | 0 | 7 |
| Order cue, “Only the former signed … she signed it” | 8 | 2 | 0 | 0 | 6 |

On the exclusive-role questions all nine unique FCoref commitments named the later person. The two order-cue commitments named the earlier person. In every one of the 13 missing-candidate cases, FCoref linked the marked pronoun to the correct cue phrase, such as “Only the medic” or “Only the latter”. Its returned cluster supplied neither marked name as the answer. In eight appositive passages the model linked a longer mention containing *she* to the later name. Six of these links matched the signer and two selected the other person. Counting all enclosing spans as recovered names would silently introduce errors.

Across all 52 passages FCoref never returned all three exact marked endpoints in its public clusters. Its raw output cannot tell us how its scorer would decide given two retained names. Requesting supplied mentions on its stock public interface returned a feature-unavailable error on the first passage. We record that operation separately from a model returning clusters without an answer. [Backend methods](../../crates/anno/src/backends/coref/resolve.rs) · [Exact-boundary observations](signer-roles.json)

## Controls and scoring

The 40 cue-bearing passages use two pairs of original names in both orders, including a multibyte character in *Béa*. The two role families cross name order with role assignment. A final one-word substitution changes the required signer. The third family changes *former* to *latter*. A pair earns credit only when both outputs choose the correct single candidate. The four repeated-name pairs instead change the spelling of the marked query to match one of two antecedents. Four further passages state that both people signed, then mention that “she went home”. These lack evidence for a unique antecedent.

| Interface | Correct cue-bearing pairs / 20 | Correct repeated-name queries / 8 | Unique commitments on 4 underspecified queries |
|---|---:|---:|---:|
| Native supplied entities | 0 | 8 | 0 |
| Native raw text, defaults | 0 | 4 | 0 |
| Native raw text, copula feature enabled | 0 | 4 | 0 |
| FCoref raw text, FP32 | 0 | 4 | 1 |

The native entity adapter received all three supplied mentions and their positions. On raw input the native resolver omitted the exact query from its returned clusters for every cue-bearing question. Removing the copula bonus for explicit bare denials left every emitted component in this panel unchanged. On FCoref's four repeated-name misses the query linked to the expected antecedent plus the following period, such as “Béa.”. This punctuation boundary error differs from the missing name-to-role connection in the exclusive and order-cue questions.

The scorer records a unique link even when the other offered name is absent. It reports both names linked, missing query, missing candidate, and all three endpoints returned without a supported name separately. Here “missing” always refers to a final returned exact character span. At this interface an absent span could have been proposed and dropped later. Exact offsets count Unicode scalar values and are half-open. Rows share templates and names, so the counts describe behavior on these controlled substitutions.

## Reproduce

Build the repository's stock example with ONNX enabled. For FCoref, follow the pinned CPU export in CI, then set `FCOREF_MODEL_PATH` to the export directory and `ORT_DYLIB_PATH` to your ONNX Runtime 1.24.2 shared library when using dynamic loading. The runner verifies all four model-file digests. We used the repository's public native entity adapter and raw backend methods. The copula selector enables the existing optional feature for this diagnostic. [Build and verified export](../../.github/workflows/fcoref-cli.yml) · [Rust worker](../../crates/anno/examples/coref_gap_panel.rs)

```sh
cargo build --locked -p anno --no-default-features --features onnx,ort/load-dynamic --example coref_gap_panel
python -m unittest discover -s scripts/qa -p test_signer_roles.py
python scripts/qa/signer_roles.py target/debug/examples/coref_gap_panel signer-roles.json \
  --arm native-entities --arm native-raw --arm native-copula-raw \
  --arm fcoref-raw --arm fcoref-supplied --details signer-spans.json
```

The optional details contain the emitted character spans for every document and backend. The passages and tests were authored for Anno. The external FCoref checkpoint by Otmazgin, Cattan and Goldberg is identified as MIT at the pinned upstream revision. The repository stores fixture generation code, metrics and hashes. [Pinned model card](https://huggingface.co/biu-nlp/f-coref/blob/e91dfff12879495d882ba9460d0c5d5dd44ade59/README.md)
