# Can a resolver follow a change in who signed?

“Ada is the pilot. Béa is the medic. Today she, the pilot, signed.” Changing the last *pilot* to *medic* reverses the intended signer while keeping the two names, the pronoun and their character offsets identical. We froze 20 such semantic pairs across three constructions before running the models. Maverick recovered the intended named signer on both halves of 15 pairs when it received the source text and three exact spans, and on eight pairs when it detected its own mentions. Neither raw FCoref nor the native entity interface solved both halves of any pair. [Original panel and scorer](../../scripts/qa/signer_roles.py) · [Replay receipt](signer-roles.json)

## Where the answer was lost

Maverick supplied 20 correct unique named answers to the 40 cue-bearing questions on raw input. On all 16 appositives it linked the correct name to the full “she, the [role]” phrase, but omitted the exact marked pronoun. On the four *former* questions it linked the exact pronoun to the cue phrase while neither offered name appeared in that cluster. The four *latter* questions returned the intended later name. Supplying the three endpoints preserved all 20 correct answers, recovered 15 appositives and merged both offered names in the other five cases. Each such merge includes an incorrect name. Maverick solved both halves of seven of eight supplied appositive pairs and all eight exclusive-role pairs. It solved zero of the four former/latter pairs.

Raw FCoref supplied seven correct unique answers and four wrong ones. Its output explains why raw resolution needs independent tests of mention boundaries and relationships between roles and named people.

| Raw FCoref, exact named answer | Questions | Correct | Wrong | Exact query missing | Exact query returned, candidate missing and no named link |
|---|---:|---:|---:|---:|---:|
| Appositive, “she, the pilot” | 16 | 0 | 0 | 16 | 0 |
| Exclusive role, “Only the pilot signed … she signed it” | 16 | 5 | 4 | 0 | 7 |
| Order cue, “Only the former signed … she signed it” | 8 | 2 | 0 | 0 | 6 |

On the exclusive-role questions all nine unique FCoref commitments named the later person. The two order-cue commitments named the earlier person. In every one of the 13 missing-candidate cases, FCoref linked the marked pronoun to the correct cue phrase, such as “Only the medic” or “Only the latter”. Its returned cluster supplied neither marked name as the answer. In eight appositive passages the model linked a longer mention containing *she* to the later name. Six of these links matched the signer and two selected the other person. Counting all enclosing spans as recovered names would silently introduce errors.

Across all 52 passages FCoref never returned all three exact marked endpoints in its public clusters. Its raw output cannot tell us how its scorer would decide given two retained names. Requesting supplied mentions on its stock public interface returned a feature-unavailable error on the first passage. We record that operation separately from a model returning clusters without an answer. [Backend methods](../../crates/anno/src/backends/coref/resolve.rs) · [Exact-boundary observations](signer-roles.json)

## Controls and scoring

The 40 cue-bearing passages use two pairs of original names in both orders, including a multibyte character in *Béa*. The two role families cross name order with role assignment. A final one-word substitution changes the required signer. The third family changes *former* to *latter*. A pair earns credit only when both outputs choose the correct single candidate. The four repeated-name pairs instead change the spelling of the marked query to match one of two antecedents. Four further passages state that both people signed, then mention that “she went home”. These lack evidence for a unique antecedent.

| Interface | Correct cue-bearing pairs / 20 | Correct repeated-name queries / 8 | Unique / both-name commitments on 4 underspecified queries |
|---|---:|---:|---:|
| Native supplied entities | 0 | 8 | 0 / 0 |
| Native raw text, defaults | 0 | 4 | 0 / 0 |
| Native raw text, copula feature enabled | 0 | 4 | 0 / 0 |
| FCoref raw text, FP32 | 0 | 4 | 1 / 0 |
| Maverick raw text | 8 | 8 | 1 / 0 |
| Maverick text and supplied spans | 15 | 8 | 0 / 4 |

The native entity adapter received all three supplied mentions and their positions. On raw input the native resolver omitted the exact query from its returned clusters for every cue-bearing question. Removing the copula bonus for explicit bare denials left every emitted component in this panel unchanged. On FCoref's four repeated-name misses the query linked to the expected antecedent plus the following period, as in the span `Béa.`. This punctuation boundary error differs from the missing name-to-role connection in the exclusive and order-cue questions.

Maverick receives text and predefined complete-token mentions in its supplied mode. The worker disables singleton emission, so a supplied competitor can be absent from the returned groups. On each underspecified passage the supplied mode grouped both distinct people with the pronoun. An API for preserving rival interpretations needs to carry those alternatives separately from this asserted group. [Maverick worker](../../crates/anno/src/backends/coref/maverick_worker.py)

The scorer records a unique link even when the other offered name is absent. A component containing both names records a merge. The scorer counts it separately from a unique choice, a missing query, a missing candidate and all three endpoints returned without a linked name. Here “missing” always refers to a final returned exact character span. At this interface an absent span could have been proposed and dropped later. Exact offsets count Unicode scalar values and are half-open. Gold identifies an occurrence within a single passage. The four names, one pronoun form and shared English templates limit what these counts say about other constructions.

## Reading the result downstream

A downstream caller can retain the emitted component for inspection and defer the named-person decision. When there is independent evidence for alternative antecedents, Anno's `UncertainReference` can carry those candidates. A component containing both people supplies no relative weights for them. [Uncertain references and candidates](../../crates/anno/src/discourse/uncertain_reference.rs)

The opt-in native copula feature also has a separate regression at the public resolver boundary. It links “Zoë is Iva” and withholds that cue for “Zoë is not Iva”. More complex predicates such as “Zoë is coaching Iva” can still earn the bonus. Across 4,454 previously studied GAP passages the denial fix changed three complete partitions with the feature enabled. All 8,908 labeled name–pronoun decisions stayed the same. [Public-path regression](../../crates/anno/src/backends/coref/mention_ranking/algorithm/mod.rs) · [Pinned GAP inputs](native-gap-endpoint-stages.md)

## Reproduce

Build the repository's stock example with ONNX enabled. For FCoref, follow the pinned CPU export in CI, then set `FCOREF_MODEL_PATH` to the export directory and `ORT_DYLIB_PATH` to your ONNX Runtime 1.24.2 shared library when using dynamic loading. The runner verifies all four FCoref file digests. For Maverick, prepare the OntoNotes tensors and encoder cache using the repository's pinned workflow, set `MAVERICK_PYTHON` and `MAVERICK_MODEL_PATH`, and point the Hugging Face cache at that prepared snapshot. The runner checks the prepared OntoNotes digest and invokes both backends offline. [Maverick preparation and runtime](../../.github/workflows/maverick-coref.yml) · [Model and encoder hashes](../../crates/anno/src/backends/coref/maverick_models.json)

We used the repository's public native entity adapter and raw backend methods. The copula selector enables the existing optional feature for this diagnostic. [Build and verified export](../../.github/workflows/fcoref-cli.yml) · [Rust worker](../../crates/anno/examples/coref_gap_panel.rs)

```sh
cargo build --locked -p anno --no-default-features --features onnx,ort/load-dynamic --example coref_gap_panel
python -m unittest discover -s scripts/qa -p test_signer_roles.py
python scripts/qa/signer_roles.py target/debug/examples/coref_gap_panel signer-roles.json \
  --arm native-entities --arm native-raw --arm native-copula-raw \
  --arm fcoref-raw --arm fcoref-supplied --arm maverick-raw --arm maverick-supplied \
  --details signer-spans.json
cmp signer-roles.json docs/eval/signer-roles.json
```

The optional details contain the emitted character spans for every document and backend. The passages and tests were authored for Anno. The external FCoref checkpoint by Otmazgin, Cattan and Goldberg is identified as MIT at the pinned upstream revision. SapienzaNLP identifies its Maverick MES OntoNotes weights as CC BY-NC-SA 4.0. Its DeBERTa-v3-large encoder is MIT-licensed. Obtain the model artifacts separately under the publishers’ terms. The repository stores fixture generation code, metrics and hashes. [Maverick model card](https://huggingface.co/sapienzanlp/maverick-mes-ontonotes/blob/087a90f0380ec1f20ed19732b657f53fd3542c81/README.md) · [DeBERTa model card](https://huggingface.co/microsoft/deberta-v3-large/tree/64a8c8eab3e352a784c658aef62be1662607476f) · [FCoref model card](https://huggingface.co/biu-nlp/f-coref/blob/e91dfff12879495d882ba9460d0c5d5dd44ade59/README.md)
