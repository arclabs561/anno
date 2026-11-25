# Character retrieval over complete novels

An annotator can only select a returning character from a shortlist if the
candidate survives eviction and is ranked highly enough to appear. We
replayed the three complete, manually annotated BookCoref gold novels.
Character IDs are revealed only after all mentions ending at the same token
have been scored. At query time the baselines see only candidate state,
with no mention text or character names.

| Return gap at capacity 24 | Queries | Resident | Recency top four | Frequency top four |
|---|---:|---:|---:|---:|
| All returns | 22,987 | 22,948 | 20,873 | 14,508 |
| At least 128 tokens | 2,323 | 2,284 | 570 | 893 |
| At least 1,024 tokens | 654 | 615 | 9 | 121 |
| At least 8,192 tokens | 90 | 54 | 0 | 4 |

For gaps of at least 1,024 tokens a 24-character memory retained 94.0% of
returning characters. Popularity retrieved 18.5% and recency retrieved
1.4% in the first four. Increasing memory to 64 retained all 654, while
popularity still retrieved only 118. At capacity eight, eviction reduced
the ceiling to 113/654. A text-conditioned scorer earns a place if it
improves retrieval conditional on residence on the identical trajectory.
Keep these gap strata separate from aggregate accuracy. Recency in the top
four recovered 90.8% of *all* returns at capacity 24.

Each memory uses LRU eviction. Frequency counts only mentions since
admission, with recency breaking ties. The budget counts entity slots and
is not a byte-matched comparison against learned vectors. The gap measures
inclusive mention-end token index minus the immediately preceding
mention-end index for the same gold character. Each threshold is cumulative.
The full history used to label gaps and first mentions is scorer-inaccessible.
The 67 novel-local first mentions are excluded from return denominators.
The data contain no equal-coordinate duplicate mentions. The oracle history
and gold character spans isolate retrieval; predicted histories may fragment
or conflate characters. No encoder, learned box or predicted mention detector
ran. This three-book diagnostic cannot establish a general novel effect.
Pride and Prejudice supplies 484 of the 654 long returns and was reserved
from the previous LitBank run.

## Reproduce

Use Python 3.10 or later and the full test annotation JSONL. The experiment
does not require rebuilding the Gutenberg text. Raw data are excluded.

```sh
curl -fL https://huggingface.co/datasets/sapienzanlp/bookcoref/resolve/2c2587aca956a4225712abdcf1af72b93f18a28f/bookcoref_annotations/full/test.jsonl -o bookcoref-test.jsonl
echo 'c4d426a722d3901489bf7bfcb3554d734db29a689876fcff269719ec7288dc9a  bookcoref-test.jsonl' | sha256sum -c -
python run.py bookcoref-test.jsonl
```

[BookCoref paper](https://aclanthology.org/2025.acl-long.1197/) · [Pinned data release](https://huggingface.co/datasets/sapienzanlp/bookcoref/tree/2c2587aca956a4225712abdcf1af72b93f18a28f)

## Anno reference experiment

This repository owns the replay and aggregate receipt used for evaluating
candidate retrieval. Anno's existing
[BookCoref downloader](../../../../../scripts/download_bookcoref.py) can also
populate the annotation cache. The pinned command above reproduces the
specific data revision used here. Run these checks in this directory:

```sh
python -m unittest discover -p 'test_*.py'
python run.py bookcoref-test.jsonl | cmp - results.json
```

The offline case distinguishes a resident character falling below rank four
from eviction and verifies that nested spans ending together share one
prefix. The replay, aggregate receipt and protocol originated in
[Box Coref PR 17](https://github.com/arclabs561/box-coref/pull/17/files).
