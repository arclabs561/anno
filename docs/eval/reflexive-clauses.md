# Does changing the clause change the referent of a reflexive?

“Yesterday Ada, who saw Béa, had praised herself” assigns the praise to Ada. “Yesterday Ada said how Béa once praised herself” assigns it to Béa. The two names and the reflexive occupy identical character ranges in both inputs. A reader that always picks the nearer name gets one sentence in each pair right and solves zero pairs.

We froze 12 such pairs using two name sets, both name orders, and three verbs. The unit of comparison is whether the resolver uniquely links the exact reflexive to the correct exact name on **both** sides. The native entity-only resolver receives identical surfaces, types and offsets on both sides. Its input cannot express this distinction. Raw-text backends can use the surrounding words and punctuation. [Panel generator and scorer](../../scripts/qa/signer_roles.py) · [First panel](reflexive.json)

On all 12 relative-clause passages, FCoref clustered the reflexive with the complete phrase headed by the earlier name, such as “Ada, who saw Béa”. On the other 12 it returned the later name alone. This is evidence that its output responds to the change in clause structure. A caller needs another source of evidence to turn the first result into an exact name selection. Both names fall inside the returned phrase, so selecting an arbitrary enclosed name can choose the wrong person.

After inspecting the first panel, we froze a second one with new names and verbs. “Today Iris, beside Léna, washed herself” refers to Iris. “Today Iris advised Léna to wash herself” refers to Léna. Before this replay we specified the exact span from the start of the first name through the end of the second as a separate diagnostic in the parenthetical construction. A hit on this diagnostic earns no exact-name credit. We also kept these spans fixed across each pair. The two panels contain 48 authored sentences built from two grammatical contrasts. They do not provide 48 independent estimates of English coreference accuracy. [Second panel](reflexive-infinitive.json)

| Backend and input | Relative vs. reported clause, correct sentences / 24 | Both sides correct / 12 | Parenthetical vs. infinitive, correct sentences / 24 | Both sides correct / 12 |
|---|---:|---:|---:|---:|
| Native, supplied entities | 0 | 0 | 0 | 0 |
| Native, raw text | 0 | 0 | 12 | 0 |
| FCoref, raw text | 12 | 0 | 20 | 8 |

On the second panel native raw text selected the later name in all 24 sentences. FCoref returned the correct exact name in all 12 infinitives and eight parentheticals. On one other parenthetical it returned the full phrase beginning with the correct name. The three remaining outputs omitted the exact reflexive. Those three shared a name order across the three verbs. Counts of missing endpoints describe the public cluster output, which also omits unlinked mentions.

## Replay

Build the public example from the repository root. The native arms work with no default features. The FCoref arm additionally needs the pinned FP32 export and CPU runtime from the real-weights workflow. The Maverick worker uses the pinned OntoNotes model and encoder. The original signer study lists model versions, sources and terms. [Runtime and checkpoint details](signer-roles.md#reproduce) · [FCoref workflow](../../.github/workflows/fcoref-cli.yml)

```sh
cargo build --locked -p anno --no-default-features --features onnx,ort/load-dynamic --example coref_gap_panel
for panel in reflexive reflexive-infinitive; do
  python3 scripts/qa/signer_roles.py target/debug/examples/coref_gap_panel "/tmp/$panel.json" \
    --panel "$panel" --arm native-entities --arm native-raw --arm fcoref-raw \
    --check "docs/eval/$panel.json"
done
```

The check compares the panel hash, the outcome counts and a digest of each arm’s per-passage outputs. A nearer-name-only control exercises the difference between getting one sentence right and solving a pair. A separate test feeds the scorer a cluster containing the reflexive and a span covering both names. It verifies that the two exact-name links remain absent. [Scoring controls](../../scripts/qa/test_signer_roles.py)
