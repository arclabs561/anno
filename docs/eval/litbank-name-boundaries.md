# Native proper-name boundaries in literary prose

This transfer trial asks whether joining adjacent capitalized tokens recovers exact person-name boundaries in English fiction, and what other boundaries the join displaces. The candidate covered 90 of the 196 multiword names. Stock covered none. All 90 survived overlap removal, which also displaced two previously retained one-word person names in two different works. [Frozen candidate and GAP trial](native-gap-multiword.md) · [Frozen selection](litbank-name-boundaries-selection.json) · [Matched score](litbank-name-boundaries-score.json)

| Exact proper-person boundaries | Stock proposed | Candidate proposed | Stock retained | Candidate retained |
|---|---:|---:|---:|---:|
| Multiword (196 annotated) | 0 | 90 | 0 | 90 |
| One word (241 annotated) | 199 | 199 | 199 | 197 |

The candidate gained multiword names in 15 of 27 works and tied in the others. Of its 227 additional proposals, 90 were aligned proper-person names and 50 matched other annotated boundaries. Two enclosed an annotated person name without matching any annotated boundary, 15 had another overlap with a proper name and 70 were unassessed. After overlap removal, the two enclosures displaced exact one-word names. The candidate also displaced 64 other exact annotated boundaries.

A descriptive follow-up found that one harmful expansion extended the person's span to the left and the other to the right. Of 159 two-word proper names, 90 were newly proposed. None of the 37 longer names were recovered. This candidate improves raw coverage, but feeding the new spans into the existing overlap policy trades away some previously correct retained boundaries.

## Corpus and scoring

Use the complete annotated excerpt of each of the 27 works pinned by Anno's earlier LitBank competing-antecedent evaluation. These books have been used for a different task. We fixed the work list, selection digest, and score definitions before running the two systems. Each work is a separate input. Every test worker sees only its work ID and unmarked text. The BRAT reader verifies the file hashes, checks that entity and coreference text layers agree, and validates source spans. [Selection and scoring code](../../scripts/qa/litbank_name_boundaries.py)

A proper person name must carry `PROP_PER` in the coreference annotation and `PER` at identical offsets in the entity annotation. We count distinct half-open Unicode scalar spans without requiring an unambiguous chain ID. Two or more whitespace-separated tokens make a name multiword. This gives 196 multiword and 241 one-word names across 297,234 scalars. Upstream names can include titles, punctuation, or other structures beyond the rule's two simple tokens. [Source release and annotation conventions](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8)

We report exact coverage before and after native overlap removal. We also partition newly added and removed spans at each stage. An exact aligned proper-person match comes first, then any other exact entity or typed coreference annotation. For the remaining spans, we distinguish strict enclosure of an annotated proper-person span, another overlap with such a span, and unassessed proposals. This priority preserves legitimate nested annotations. An enclosing proposal with no matching annotation has a demonstrable mismatch against the contained name's boundary. The corpus supplies no universal negative label for unrelated proposals. These counts measure coverage and boundary changes. Link correctness, downstream retrieval, and proposal precision require their own denominators.

## Reproduction and attribution

Point `litbank` at the publisher's pinned LitBank checkout, and `stock` and `candidate` at the two Rust library test executables from the matched rule trial. Run from the Anno checkout. The scoring command checks the frozen selection and records worker SHA-256 values. Rebuilt executables may have different hashes. Re-run the same controls and compare stage counts. Only hashes and aggregates are stored in the repository.

```sh
python3 -m unittest discover -s scripts/qa -p test_litbank_name_boundaries.py
python3 scripts/qa/litbank_name_boundaries.py "$litbank" > /tmp/litbank-name-selection.json
cmp /tmp/litbank-name-selection.json docs/eval/litbank-name-boundaries-selection.json
python3 scripts/qa/litbank_name_boundaries.py "$litbank" --stock "$stock" --candidate "$candidate" > /tmp/litbank-name-results.json
```

David Bamman, Sejal Popat and Sheng Shen (2019) published the literary entity annotations. David Bamman, Olivia Lewke and Anya Mansoor (2020) published the coreference annotations. The publisher identifies the source editions as public-domain Project Gutenberg texts and licenses LitBank under CC BY 4.0. This evaluation derives span counts from the two annotation layers. [Publisher, papers and original release](https://github.com/dbamman/litbank/tree/3e50db0ffc033d7ccbb94f4d88f6b99210328ed8) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
