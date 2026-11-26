# Multilingual antecedent retrieval on CorefUD 1.4

Three deterministic rules select an earlier annotated source mention. Each
query must have a strictly earlier, non-overlapping mention of the same entity.
We count one hit if the selected earlier span has the right entity ID, and give
the ranker no access to those IDs. This measures source-pair retrieval on
returning mentions with gold boundaries. It does not measure detection,
abstention, cluster quality, or the error rate of automatic merges.

Across 1,751 development documents from 29 corpora in 19 languages, the
three rules produced the following counts of correct suggestions. The
surface rule prefers the most recent exact case-folded source text, backing
off to the most recent eligible mention. The head rule additionally tries
matching the annotated head's surface text before falling back to recency.

| Query group | Queries | Most recent | Exact surface | Surface then head |
|---|---:|---:|---:|---:|
| All returning mentions | 91,178 | 24,232 | 43,358 | 50,336 |
| Current form previously used by the target | 32,943 | 7,577 | 30,445 | 30,445 |
| Current form never previously used by the target | 58,235 | 16,655 | 12,913 | 19,891 |
| At least 128 intervening tokens since latest gold antecedent | 8,796 | 0 | 3,481 | 5,027 |

**Consequence for Anno.** Among the 58,235 new-form mentions, exact-surface
priority costs 3,742 correct pairs relative to recency. This occurs when
another entity used the same form previously. The diagnosis uses gold target
history and is unavailable at prediction time. An annotated-head fallback
recovers 6,978 correct pairs overall compared with surface plus recency.
That advantage is conditional on the supplied head boundaries. A predicted
dependency-head baseline is a necessary comparator before crediting learned
geometry with better alias retrieval. The English counts are 7,317 and 8,169
correct of 11,456 for surface and head fallback. Czech gives 12,204 and
14,815 of 31,998, Korean 1,420 and 1,532 of 3,365, and Latin 80 and 79 of
146. The pooled counts weight queries equally. Corpora within each language
differ in genre and annotation; scores do not isolate language effects.

**Scope.** The input shard contains 175,492 mentions. We excluded 1,545
invalid or non-alphanumeric spans, 4,452 mentions with duplicate observable
coordinates and 78,317 without an eligible gold antecedent. Most recent
means greatest end offset, with ties broken by later start. Groups for new
and repeated forms partition returning mentions. The long-gap group cuts
across both. Surface matches use Unicode case folding and normalized
whitespace. This development set includes previously studied English LitBank
and GUM material. Cross-corpus and cross-language performance on the
non-public test data remains unknown.

**Reproduce.** The script verifies the pinned public development shard's
SHA-256 before reading it. Python 3.12 and PyArrow 20.0.0 were used. Raw
corpus text stays in the supplied local cache. See the original corpus for
individual resource licenses and attribution. [CorefUD 1.4](https://hdl.handle.net/11234/1-6108) · [Pinned reformatted data](https://huggingface.co/datasets/lattice-nlp/corefud-1-4/tree/3ded5a1638e43d89b81ee24bf5b98f66943e1c60)

```bash
python -m pip install pyarrow==20.0.0
python corefud_retrieval.py /tmp/corefud-dev.parquet results.json
```

### Predicted heads and abstention

We sampled up to three development documents per corpus by SHA-256 of filename.
Eighty documents were scored across 18 languages. Biblical Hebrew accounts for
three additional selected documents with no matching parser. UDPipe tags and
parses the supplied sentence and token boundaries. All methods receive the same
gold mention boundaries. A head is usable only for an aligned span with one
token governed from outside the span (or the sentence root). We retain first
mentions to measure the cost of offering a suggestion where none is valid.

| Predecessor policy | Correct | Wrong on returning mentions | Wrong without eligible antecedent | Abstained |
| --- | ---: | ---: | ---: | ---: |
| Exact normalized surface | 7,724 | 2,774 | 639 | 10,362 |
| Word and punctuation sequence | 7,725 | 2,774 | 639 | 10,361 |
| Surface, then matching predicted head | 8,444 | 3,103 | 1,463 | 8,489 |

Keep head matching experimental. Of 1,873 additional suggestions, 720 were
correct and 1,153 were wrong. Effects varied across the sample. In Spanish,
head fallback added 15 correct suggestions and three errors on mentions with
no eligible antecedent. In Dutch those counts were 139 and 356. Neither
language estimate is a release threshold. The punctuation comparison measures
one normalization rule on supplied mention spans. End-to-end mention discovery,
learned segmentation and out-of-domain accuracy need separate evaluation.
Next compare a gated fallback against surface matching on held-out documents,
reporting both added correct links and added wrong links per language and domain.

Reproduce from this directory with the development shard linked above.
The manifest pins the Git blob of every model. The results include document IDs
and counts for each language. Download models subject to their source licenses.
[UDPipe model repository](https://github.com/jwijffels/udpipe.models.ud.2.5/tree/17d6bbfb7f138f9ba2cf04692f15d892becf0246)

```bash
python -m pip install ufal.udpipe==1.4.0.1 pyarrow==26.0.1
mkdir -p /tmp/udpipe-models
jq -r '.[].path' abstaining-head-models.json | while read -r path; do
  curl -fL "https://raw.githubusercontent.com/jwijffels/udpipe.models.ud.2.5/17d6bbfb7f138f9ba2cf04692f15d892becf0246/$path" -o "/tmp/udpipe-models/${path##*/}" || exit 1
done
python abstaining_heads.py /tmp/corefud-dev.parquet /tmp/udpipe-models replay.json
cmp abstaining-head-results.json replay.json
```

**Implementation references.** Three interface decisions affect how to interpret
these results and reproduce them in downstream tools.

- brat represents discontinuous text-bound annotations as multiple half-open
  character ranges in an unchanged UTF-8 source. Preserve each range and the
  identity of the source revision when exchanging annotations. [brat standoff specification](https://brat.nlplab.org/standoff.html)
- spaCy's original LLM NER task reconstructs offsets from returned strings.
  Its configuration exposes repeated matching, case sensitivity and alignment
  modes. Repeated identical strings can therefore receive different occurrence
  assignments across adapters. Score the individual occurrences and record the
  alignment policy when evaluating a string-returning adapter. [spaCy LLM task reference](https://spacy.io/api/large-language-models)
- UDPipe distinguishes loading pretokenized words from segmenting raw text.
  Our replay constructs a sentence from the supplied words before tagging and
  parsing. A raw-text comparison must also measure tokenizer boundaries and
  count alignment failures in its denominator. [UDPipe 1 API reference](https://ufal.mff.cuni.cz/udpipe/1/api-reference)

## Predicted dependency heads

The paired replay selects up to four complete development documents per component
by the lowest salted SHA-256 of the source text, deduplicated within that component.
Selection uses no labels. UDPipe 1 consumes the supplied sentence boundaries and
token forms. It predicts its own tags and dependency arcs before the scorer sees
the supplied mention spans. An exact token-aligned span gets a head when exactly
one of its tokens has an external governor. An ambiguous or unaligned span falls
back to surface matching. The study uses 18 language-matched UD 2.5 parsers.
Ancient Hebrew has no selected model and its sampled documents are counted separately.

The retrieval measure asks which earlier mention belongs to the returning entity.
The two clustering arms instead link exact surface matches, optionally using a
predicted-head match as a fallback, and create a new cluster when neither matches.
The reported pair counts include transitive links between all usable supplied
mentions. The 95% interval resamples documents within components, 2,000 times.
Model load time is excluded from the per-document parse and span-alignment
latency. Parser failures retain all affected documents in the surface-only
fallback. This panel assesses a frozen rule on a previously studied development
release. The parser training treebanks can share source material with CorefUD;
we have not established training-document disjointness for pretrained parsers.
Its head and pair scores assume supplied mention boundaries.
Release-system scores over their own detected mentions on Dutch and French are
available in the [full-document replay](../../../../../../docs/eval/coref-corpus-qa.md).

On 104 scored documents, the predicted-head rule won 670 and lost 89 retrieval
decisions relative to surface matching. That moved correctness from 10,391 to
10,972 of 17,310 returning mentions (60.03% to 63.39%). The paired change was
+3.36 percentage points, with a document-bootstrap 95% interval of +2.74 to
+4.03 points. Seventeen languages gained and Latin lost one retrieval. Four
ancient-Hebrew documents had no available model. The parser gave an admissible
head to 23,917 of 24,989 usable supplied mentions.

With supplied mentions, head matching increased true cluster pairs from
176,220 to 182,618 but also increased unmatched pairs from 152,001 to 161,474.
Pair recall rose from 12.01% to 12.45% and precision fell from 53.69% to
53.07%. Large entities contribute many pairs. The [aggregate receipt](predicted-head-results.json)
includes language denominators and run time on this CPU host.

The Anno native mention-ranking resolver also ran on the 104 full texts.
It received one document ID and the raw text per call. The 2,169 returned
mentions contained 1,267 exact reference mentions, from 24,989 usable annotated
mentions. The returned clusters had 1,184 correct pairs among 2,372. Rebuilding
clusters over these same returned spans produced 7,214 matches and 8,207
unmatched pairs with surface matching alone. Adding predicted heads left the
7,214 matches unchanged and increased unmatched pairs by seven. These two
ablations use only already-returned spans. Missing native mentions cannot
participate. On this cohort the supplied-span retrieval gain gives no basis to
enable a multilingual parser by default. The final run spent 85.6 seconds in
parsing and head alignment, with a per-document p95 of 2.78 seconds.

Use an authorized local copy of the pinned development shard and local copies
of the files listed in [the model manifest](udpipe-models.json). The script
verifies the shard SHA-256 and each parser's content against its pinned Git blob.
Model and treebank terms apply independently. The parsers by Straka and Straková,
trained on Universal Dependencies 2.5, are distributed under CC BY-NC-SA 4.0.
This local research comparison stores aggregate counts. Models and upstream
text stay outside Git. [UDPipe model release and attribution](https://github.com/jwijffels/udpipe.models.ud.2.5/tree/17d6bbfb7f138f9ba2cf04692f15d892becf0246/inst/udpipe-ud-2.5-191206)
· [Model license](https://github.com/jwijffels/udpipe.models.ud.2.5/blob/17d6bbfb7f138f9ba2cf04692f15d892becf0246/inst/udpipe-ud-2.5-191206/LICENSE)

```bash
# Build the native text-only driver from the Anno checkout if using --native-runner.
cargo build --locked -p anno --features onnx --example coref_gap_panel
python -m pip install pyarrow==18.1.0 ufal.udpipe==1.4.0.1
python predicted_heads.py /tmp/corefud-dev.parquet /tmp/udpipe-2.5 /tmp/predicted-heads.json \
  --native-runner /path/to/anno/target/debug/examples/coref_gap_panel
python -m unittest -v test_predicted_heads
```
